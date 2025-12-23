from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
import math

from typing import Any, Optional, Union, Dict
from baselines.common.mpi_adam import MpiAdam

from enum import Enum

MPI=None
# from mpi4py import MPI
from collections import deque
import os
import shutil
from scipy import spatial
import gymnasium as gym
import matplotlib.pyplot as plt
import random

from tensorflow.keras.layers import Input, Reshape, Dense, concatenate, Lambda
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

class OptionStepCounter:
    def __init__(self, num_options):
        # Timestep count per option
        self.option_timestep_counts = {option: 0 for option in range(num_options)}

    def incCount(self, option):
        # Increments the counter of the specified option
        if option in self.option_timestep_counts:
            self.option_timestep_counts[option] += 1
        else:
            raise ValueError(f"Option {option} is not recognized")

    def getCount(self, option):
        if option in self.option_timestep_counts:
            return self.option_timestep_counts[option]
        else:
            raise ValueError(f"Option {option} is not recognized")

    def resetCount(self):
        for option in self.option_timestep_counts:
            self.option_timestep_counts[option] = 0

def apply_her(transitions, env, k, hermvobj, objcoeff, its2herdis, kdecay):
    augmented_transitions = []
    totalTimesteps = 0
    has_obj = env.unwrapped.has_object
    is_slide = env.spec.id.startswith("FetchSlide")
    if is_slide: #For the slide environment, the distance considered for moving the object must be greater, due to its inherent sliding
        dist_objMoved = 5e-3
    else:
        dist_objMoved = 1e-4

    try:
        initial_position = transitions[0]["state"][3:6]
        final_position = transitions[-1]["state"][3:6]

        aux_t = transitions[-1] #Select the last element as the goal, just for reference to get the split_sizes
        split_sizes = [aux_t["next_state"].size, aux_t["desired_goal"].size, aux_t["achieved_goal"].size]
        cumsum_splits = np.cumsum(split_sizes)
        s1, s2, s3 = cumsum_splits[0], cumsum_splits[1], cumsum_splits[2]

        if (kdecay>0): #Computes k_her decay
            k = math.ceil(k-(iters_so_far/kdecay))

        if (hermvobj==0 or (not np.allclose(initial_position, final_position, atol=dist_objMoved))) and (k>0):

            for j in range(k):
                for t_idx, t in enumerate(transitions): #Uses strategy "future"
                    next_state_reconstructed = t["state"][:s1]
                    future_timestep = np.random.randint(t_idx, len(transitions))

                    if iters_so_far <= its2herdis and has_obj: #Applies 2HER only if the environment has object interaction
                        hindsight_object = transitions[future_timestep]["state"][0:3]
                        object_reward = env.compute_reward(t["state"][0:3], hindsight_object, info={})

                        future_timestep = np.random.randint(future_timestep, len(transitions))
                        hindsight_goal = transitions[future_timestep]["achieved_goal"]
                        goal_reward = env.compute_reward(t["achieved_goal"], hindsight_goal, info={})
                    
                        new_reward = objcoeff*object_reward + (1-objcoeff)*goal_reward
                        arrBlockGripperPos = [round(hindsight_object[0]-t['state'][0],8), round(hindsight_object[1]-t['state'][1],8), round(hindsight_object[2]-t['state'][2],8)]
                        next_state_reconstructed[3:6] = hindsight_object
                        next_state_reconstructed[6:9] = arrBlockGripperPos
                    else:
                        future_timestep = np.random.randint(future_timestep, len(transitions))
                        hindsight_goal = transitions[future_timestep]["achieved_goal"]
                        goal_reward = env.compute_reward(t["achieved_goal"], hindsight_goal, info={})

                        new_reward = goal_reward

                    if goal_reward == 0: # If the object reaches the goal, the reward is reset to avoid incoherent learning
                        new_reward = 0
                    
                    new_transition = {
                        "state": np.concatenate(( #next_observation
                            next_state_reconstructed,
                            hindsight_goal, #desired_goal
                            t["achieved_goal"], #achieved_goal,
                        )),
                        "new": t["new"],
                        "option": t["option"],
                        "last_option": t["last_option"],
                        "prevac": t["prevac"],
                        "action": t["action"],
                        "reward": new_reward,
                    }
                    augmented_transitions.append(new_transition)

    except Exception as e:
        print(f"Error applying HER: {e}")

    return augmented_transitions

import numpy as np

def append_transitions(array, transitions, key, stack_type="vstack", dtype=None, flatten=False):
    new_data = np.array([t[key] for t in transitions])
    if flatten:
        new_data = new_data.flatten()
    if dtype:
        new_data = new_data.astype(dtype)
    
    if array.size == 0:
        return new_data
    else:
        if stack_type == "vstack":
            return np.vstack([array, new_data])
        elif stack_type == "hstack":
            return np.hstack([array, new_data])
        else:
            raise ValueError(f"Invalid stack type: {stack_type}")


def traj_segment_generator(pi,env,horizon,stochastic,num_options,saves,rewbuffer,epoch,seed,w_intfc,switch,gamma,eta,option_counter,render,is_goal_env,kher,hermvobj,objcoeff,its2herdis,kdecay):
    
    #Initializes auxiliar HER buffers
    her_state = np.empty((0,))
    her_new = np.empty((0,))
    her_option = np.empty((0,))
    her_last_option = np.empty((0,))
    her_prevac = np.empty((0,))
    her_action = np.empty((0,))
    her_reward = np.empty((0,))

    cont_episodes = 0
    cont_solved = 0

    t = 0
    ac = env.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob, obs_orig = env.reset(seed=seed)

    #render=0
    iters_so_far=0

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of completed episodes in this segment

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    realrews = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    opts = np.zeros(horizon, 'int32')
    activated_options = np.zeros((horizon, num_options), 'float32')
    last_options=np.zeros(horizon, 'int32')

    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    print("ob: ", ob)
    option,active_options_t = pi.get_option(ob)
    last_option=option


    ep_states=[[] for _ in range(num_options)] 
    ep_states[option].append(ob)
    ep_states_term=[[] for _ in range(num_options)] 
    ep_num =0
    episode_transitions = []

    opt_duration = [[] for _ in range(num_options)]
    curr_opt_duration = 0.

    base_env = env
    while hasattr(base_env, "env"):
        base_env = base_env.env

    is_slide = base_env.spec.id.startswith("FetchSlide")
    if is_slide:
        dist_solved = 0.2 #For the Slide task, an episode is solved if the last position <= 0.2
    else:
        dist_solved = 0.07 #Last position <= 0.07 for another tasks
    
    while True:
        prevac = ac
        ac = pi.act(stochastic, ob, option)
        
        option_counter.incCount(option)
        
        if render:
            env.render()
            time.sleep(0.05)
            print(option)
        
        if t > 0 and t % horizon == 0:

            # ===== HER application and calculations =====
            augmented_transitions = apply_her(episode_transitions, env, kher, hermvobj, objcoeff, its2herdis, kdecay)

            # Increments all HER buffers for further processing
            if (len(augmented_transitions)>0):
                her_state = append_transitions(her_state, augmented_transitions, "state", stack_type="vstack")
                her_new = append_transitions(her_new, augmented_transitions, "new", stack_type="hstack", dtype=int, flatten=True)
                her_option = append_transitions(her_option, augmented_transitions, "option", stack_type="hstack", dtype=int, flatten=True)
                her_last_option = append_transitions(her_last_option, augmented_transitions, "last_option", stack_type="hstack", dtype=int, flatten=True)
                her_prevac = append_transitions(her_prevac, augmented_transitions, "prevac", stack_type="vstack")
                her_action = append_transitions(her_action, augmented_transitions, "action", stack_type="vstack")
                her_reward = append_transitions(her_reward, augmented_transitions, "reward", stack_type="hstack", flatten=True)

            episode_transitions = []

            # ===== Concatenation of her_elements elements to the original buffer =====
            # Stores the original values ​​to consider her buffer only at this stage
            backup_obs = obs
            backup_news = news
            backup_opts = opts
            backup_last_options = last_options
            backup_prevacs = prevacs
            backup_acs = acs
            backup_rews = rews
            backup_realrews = realrews

            try:
                obs = np.concatenate((her_state, obs), axis=0)
                news = np.concatenate((her_new, news), axis=0)
                opts = np.concatenate((her_option, opts), axis=0)
                last_options = np.concatenate((her_last_option, last_options), axis=0)
                prevacs = np.concatenate((her_prevac, prevacs), axis=0)
                acs = np.concatenate((her_action, acs), axis=0)
                rews = np.concatenate((her_reward, rews), axis=0)
                realrews = np.concatenate((her_reward, realrews), axis=0)
            except Exception as e:
                print(f"Error applying HER: {e}")

            vpreds, op_vpreds, vpred, op_vpred, op_probs, intfc, pi_I = pi.get_allvpreds(obs, ob)
            term_ps, term_p, all_term_ps = pi.get_alltpreds(obs, ob)
            last_betas=term_ps[range(len(last_options)),last_options]

            all_opts = np.append(opts,option)
            term_ratios=np.zeros((len(all_opts),num_options))
            for o in range(num_options):
                one_hot = np.zeros(len(all_opts))
                one_hot[np.where(all_opts==o)] = 1.
                term_ratios[:,o]=(all_term_ps[:,o] * pi_I[range(len(all_opts)),all_opts] + (1-all_term_ps[:,o]) * one_hot)
            term_ratios = np.log(term_ratios[1:]) - np.log(term_ratios[range(1,len(all_opts)),all_opts[:-1]][...,None])
            

            logps = np.zeros( (len(obs),num_options))
            for o in range(num_options):
                logps[:,o] = pi._logps(True,obs,[o],acs)[0]
            action_ratios = logps - logps[range(len(obs)), opts][...,None]
            
            prev_action_ratios = np.vstack((action_ratios[0],action_ratios[:-1])) # a little bias here

            last_options_onehot = np.zeros((len(last_options),num_options))
            last_options_onehot[range(len(last_options)),last_options] = 1.
            prob_curr_opt = last_betas[...,None] * pi_I[:-1] + (1-last_betas[...,None]) * last_options_onehot
            prob_prev_opt = np.vstack((prob_curr_opt[0],prob_curr_opt[:-1])) # a little bias here

            sampled_eta = float(np.random.rand()<eta)
            options_onehot = np.zeros((len(opts),num_options))
            options_onehot[range(len(opts)),opts] = 1.
            prob_curr_opt = sampled_eta * prob_curr_opt + (1-sampled_eta) * options_onehot
            prob_prev_opt= sampled_eta * prob_prev_opt + (1-sampled_eta) * last_options_onehot

            # Data ​​used in the learning stages
            yield {"ob" : obs, "rew" : rews, "realrew": realrews, "vpred" : vpreds, "op_vpred": op_vpreds, "new" : news,
                    "ac" : acs, "opts" : opts, "opt": option, "prevac" : prevacs, "nextvpred": vpred * (1 - new), "nextop_vpred": op_vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, 'term_p': term_ps, 'next_term_p':term_p,
                     "op_probs":op_probs, "last_betas":last_betas, "intfc":intfc, 
                      "action_ratios": action_ratios, "term_ratios":term_ratios, "prev_action_ratios": prev_action_ratios,
                      "last_options": last_options, "last_option":last_option, "prob_curr_opt": prob_curr_opt, "prob_prev_opt":prob_prev_opt, "cont_episodes":cont_episodes, "cont_solved":cont_solved, "opt_dur": opt_duration}

            ep_rets = []
            ep_lens = []
            #opt_duration = [[] for _ in range(num_options)]
            #curr_opt_duration = 0.
            iters_so_far+=1
            cont_episodes = 0
            cont_solved = 0

            ###### Switching Goal ##########
            '''
            if iters_so_far==switch_iter and switch:
                # import pdb;pdb.set_trace()
                if hasattr(env,'NAME') and env.NAME=='AntWalls': # Switch the goal for AntWalls
                    from antwalls import AntWallsEnv
                    env=AntWallsEnv(num_walls=2)
                    env.seed(seed) 
                elif env.spec.id == 'HalfCheetahDir-v1':
                    env.env.env.reset_task({'direction':-1})
                elif env.spec.id == 'Walker2dStand2-v1':
                    env.env.reset_task('run')
            '''
            ################################

            # Resetting the variables related to the HER buffer
            her_state = np.empty((0,))
            her_new = np.empty((0,))
            her_option = np.empty((0,))
            her_last_option = np.empty((0,))
            her_prevac = np.empty((0,))
            her_action = np.empty((0,))
            her_reward = np.empty((0,))

            # For efficiency calculations and other steps, return the original trajectories, disregarding values ​​generated by the her buffer
            obs = backup_obs
            news = backup_news
            opts = backup_opts
            last_options = backup_last_options
            prevacs = backup_prevacs
            acs = backup_acs
            rews = backup_rews
            realrews = backup_realrews

        i = t % horizon
        obs[i] = ob
        last_options[i]=last_option

        news[i] = new
        opts[i] = option
        acs[i] = ac
        prevacs[i] = prevac
        activated_options[i] = active_options_t

        ## RL loop ##
        if not isinstance(env.unwrapped, gym.envs.mujoco.MujocoEnv): # For environments that are not mujoco, the action space must be converted
            ac = ac[0]

        if (is_goal_env):
            state = ob #Get state before action
            ob, rew, done, truncated, _, obs_orig = env.step(ac,iters_so_far)
            
            transition = {
                "state": state,
                "new": new,
                "option": option,
                "last_option": last_option,
                "prevac": prevac, #previous action
                "action": ac,
                "reward": rew,
                "her": 0,
                "next_state": obs_orig['observation'], #next_obs
                "achieved_goal": obs_orig["achieved_goal"], #next_obs
                "desired_goal": obs_orig["desired_goal"], #next_obs
            }
        else:
            state = ob #Get state before action
            ob, rew, done, truncated, _ = env.step(ac,iters_so_far)
            transition = {
                "state": state,
                "action": ac,
                "reward": rew,
                "next_state": ob,
            }

        if np.any(np.isnan(ob)) or np.any(np.isinf(ob)):
            print("Restarting episode due to instability")
            ob, obs_orig = env.reset(iters_so_far)
        
        new = done or truncated
        episode_transitions.append(transition)

        rews[i] = rew
        realrews[i] = rew

        candidate_option,active_options_t = pi.get_option(ob)

        term = pi.get_term([ob],[option])
        last_option=option
        
        if term:
            '''
            opt_duration[option].append(curr_opt_duration)
            #print(f"opt_duration[{option}]", opt_duration[option])
            curr_opt_duration = 0.
            '''

            ep_states_term[option].append(ob)
            option = candidate_option


        ep_states[option].append(ob)
        cur_ep_ret += rew
        cur_ep_len += 1
        
        if new:

            # Checks if the environment has been resolved based on the last recorded position
            dist_euclidiana = np.linalg.norm(obs_orig['desired_goal'] - obs_orig['achieved_goal'])
            if (dist_euclidiana <= dist_solved):
                cont_solved += 1
            cont_episodes += 1
            
            augmented_transitions = apply_her(episode_transitions, env, kher, hermvobj, objcoeff, its2herdis, kdecay)

            # Increments all HER buffers for further processing
            if (len(augmented_transitions)>0):
                her_state = append_transitions(her_state, augmented_transitions, "state", stack_type="vstack")
                her_new = append_transitions(her_new, augmented_transitions, "new", stack_type="hstack", dtype=int, flatten=True)
                her_option = append_transitions(her_option, augmented_transitions, "option", stack_type="hstack", dtype=int, flatten=True)
                her_last_option = append_transitions(her_last_option, augmented_transitions, "last_option", stack_type="hstack", dtype=int, flatten=True)
                her_prevac = append_transitions(her_prevac, augmented_transitions, "prevac", stack_type="vstack")
                her_action = append_transitions(her_action, augmented_transitions, "action", stack_type="vstack")
                her_reward = append_transitions(her_reward, augmented_transitions, "reward", stack_type="hstack", flatten=True)

            episode_transitions = []

            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0

            ep_num +=1
            ob, obs_orig = env.reset(iters_so_far)
            option,active_options_t = pi.get_option(ob)
            last_option=option
            ep_states[option].append(ob)
        t += 1


def add_vtarg_and_adv(seg, gamma, lam, num_options):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    T = len(seg["rew"])
    arrival_options = np.append(seg["last_options"],seg["last_option"])
    opts = np.append(seg["opts"],seg["opt"])
    rew = seg["rew"]

    op_vpred = np.append(seg["op_vpred"], seg["nextop_vpred"])
    term_p = np.vstack((np.array(seg["term_p"]),np.array(seg["next_term_p"])))
    q_sw = np.vstack((seg["vpred"],seg["nextvpred"]))
    all_u_sw = (1-term_p) * q_sw + term_p * np.tile(op_vpred[:,None],num_options)
    u_sw = all_u_sw[range(len(all_u_sw)),arrival_options]
    
    
    seg["op_adv"] = gaelam = np.empty(T, 'float32')
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * u_sw[t+1] * nonterminal - u_sw[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam


    seg["adv"] = gaelam = np.empty(T, 'float32')
    vpred= q_sw[range(len(opts)),opts]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam

    seg["tdlamret"] = seg["adv"] + vpred[:-1]

    seg["term_adv"] = seg["vpred"] - np.tile(seg["op_vpred"][:,None],num_options)




def learn(env, policy_func, *,
        timesteps_per_batch, # timesteps per actor per update
        clip_param, entcoeff, # clipping parameter epsilon, entropy coeff
        optim_epochs, optim_stepsize, optim_batchsize,# optimization hypers
        gamma, lam, # advantage estimation
        max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
        callback=None, # you can do anything in the callback, since it takes locals(), globals()
        adam_epsilon=1e-5,
        schedule='constant', # annealing for stepsize parameters (epsilon and adam)
        num_options=1,
        app='',
        saves=False,
        wsaves=False,
        epoch=0,
        seed=1,
        w_intfc=True,switch=False,intlr=1e-4,piolr=1e-4,multi=False,eta=0.1,
        render=False,
        is_goal_env=False,
        kher=1,
        hermvobj=1,
        objcoeff=1,
        its2herdis=150,
        kdecay=50
        ):


    optim_batchsize_ideal = optim_batchsize 
    np.random.seed(seed)
    tf.set_random_seed(seed)
    


    ### Book-keeping
    if hasattr(env,'NAME'):
        gamename = env.NAME.lower() #change this for plots
    else:
        gamename = env.spec.id[:-3].lower()
    gamename += 'seed' + str(seed)

    #Saving the wigths
    dirname = 'savedmodels/{}_{}opts_saves/'.format(gamename,num_options)

    if wsaves:
        first=True
        if not os.path.exists(dirname):
            os.makedirs(dirname)
            first = False
    ###


    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space) # Construct network for new policy
    oldpi = policy_func("oldpi", ob_space, ac_space) # Network for old policy
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return
    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon



    prob_cur_opt = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of current option
    is_ratio = tf.placeholder(dtype=tf.float32, shape=[None]) # IS ratio for correcting off-policyness


    ob = U.get_placeholder_cached(name="ob")
    option = U.get_placeholder_cached(name="option")
    term_adv = U.get_placeholder(name='term_adv', dtype=tf.float32, shape=[None])
    op_adv = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    betas = tf.placeholder(dtype=tf.float32, shape=[None]) # Probability of termination (Used to weight meta-updates)
    oldvpred = tf.placeholder(tf.float32, [None])
    ac = pi.pdtype.sample_placeholder([None])

    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = U.mean(kloldnew)
    meanent = U.mean(ent)
    pol_entpen = (-entcoeff) * meanent

    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac) + is_ratio)
    surr1 = ratio * atarg # surrogate from conservative policy iteration
    surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg 
    pol_surr = - U.mean(tf.minimum(surr1, surr2)  * prob_cur_opt )  # PPO's pessimistic surrogate (L^CLIP)


    vf_loss = U.mean(tf.square(pi.vpred - ret) * tf.exp(is_ratio) * prob_cur_opt)

    total_loss = pol_surr + pol_entpen + vf_loss
    losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
    loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

    # Loss for termination function
    option_hot = tf.one_hot(option,depth=num_options)
    term_loss= U.mean(( tf.reduce_sum(pi.tpred * option_hot, axis=1) * term_adv) )

    # Loss for interest function
    pi_w = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (pi.intfc ) * pi_w / tf.expand_dims(tf.reduce_sum((pi.intfc ) * pi_w,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    int_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)

    # Loss for policy over options
    intfc = tf.placeholder(dtype=tf.float32, shape=[None,num_options])
    pi_I = (intfc ) * pi.op_pi / tf.expand_dims(tf.reduce_sum( (intfc ) * pi.op_pi,axis=1),1)
    pi_I = tf.clip_by_value(pi_I,1e-6,1-1e-6)
    op_loss = - tf.reduce_sum(betas *tf.reduce_sum(pi_I * option_hot,axis=1)    * op_adv)
    log_pi = tf.log(tf.clip_by_value(pi.op_pi, 1e-20, 1.0))
    op_entropy = -tf.reduce_mean(pi.op_pi * log_pi, reduction_indices=1)
    op_loss -= 0.01*tf.reduce_sum(op_entropy)



    var_list = pi.get_trainable_variables()
    lossandgrad = U.function([ob, ac, atarg, ret, lrmult, option,  prob_cur_opt, is_ratio], losses + [U.flatgrad(total_loss, var_list)])
    termgrad = U.function([ob, option, term_adv], [U.flatgrad(term_loss, var_list)]) # Since we might use a different step size.
    opgrad = U.function([ob, option, betas, op_adv, intfc], [U.flatgrad(op_loss, var_list)]) # Since we might use a different step size.
    intgrad = U.function([ob, option, betas, op_adv, pi_w], [U.flatgrad(int_loss, var_list)]) # Since we might use a different step size.
    adam = MpiAdam(var_list, epsilon=adam_epsilon)

    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_losses = U.function([ob, ac, atarg, ret, lrmult, option], losses)


    U.initialize()
    adam.sync()


    saver = tf.train.Saver(max_to_keep=10000)
    #saver = tf.train.Saver(max_to_keep=0)

    ### More book-kepping
    # results=[]
    # if saves:
    #     directory_res = "res/opt{}/".format(num_options) #if not fewshot else "res_fewshot/opt{}/".format(num_options) 
    #     print(f'\ndirectory_res: {directory_res}')
    #     if not os.path.exists(directory_res):
    #         os.makedirs(directory_res)       
    #     if w_intfc: 
    #         results = open(directory_res + gamename +'intfc{}_lr_{}_intlr{}_piolr{}_eta{}_seed{}'.format(int(w_intfc),optim_stepsize,intlr,piolr,eta,seed) + '.csv','w')
    #     else:
    #         results = open(directory_res + gamename +'intfc{}_lr_{}_intlr{}_piolr{}_eta{}_seed{}'.format(int(w_intfc),optim_stepsize,intlr,piolr,eta,seed) + '.csv','w')
    #     out = 'epoch,avg_reward,num_opts_used'

    #     out+='\n'
    #     results.write(out)
    #     results.flush()

    if epoch:

        #dirname = 'savedmodels/moc/{}_{}opts_saves/'.format(gamename,num_options)
        dirname = 'savedmodels/{}_{}opts_saves/'.format(gamename,num_options)
        print("Loading weights from iteration: " + str(epoch) + " from " + dirname)

        filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,epoch)
        saver.restore(U.get_session(),filename)
    ###  


    episodes_so_far = 0
    timesteps_so_far = 0
    global iters_so_far
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=10) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=10) # rolling buffer for episode rewards

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"








    ######################################################### Prepare for rollouts #########################################################
    # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    
    option_counter = OptionStepCounter(num_options)
    seg_gen = traj_segment_generator(pi,env,timesteps_per_batch,stochastic=True,num_options=num_options,saves=saves,rewbuffer=rewbuffer,epoch=epoch,seed=seed,w_intfc=w_intfc,switch=switch,gamma=gamma,eta=eta,option_counter=option_counter,render=render,is_goal_env=is_goal_env,kher=kher,hermvobj=hermvobj,objcoeff=objcoeff,its2herdis=its2herdis,kdecay=kdecay)

    datas = [0 for _ in range(num_options)]

    while True:
        #print('Iniciando learning...')
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Iteration %i ************"%iters_so_far)
        seg = seg_gen.__next__()

        add_vtarg_and_adv(seg, gamma, lam,num_options)

        ob, ac, opts, atarg, tdlamret, op_atarg  = seg["ob"], seg["ac"], seg["opts"], seg["adv"], seg["tdlamret"], seg["op_adv"] 
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        assign_old_eq_new() # set old parameter values to new parameter values

        #Savind weigths
        if iters_so_far % 10 == 0 and wsaves:
            print("weights are saved...")
            filename = dirname + '{}_epoch_{}.ckpt'.format(gamename,iters_so_far)
            save_path = saver.save(U.get_session(),filename)
        

        min_batch=160 # Arbitrary
        for opt in range(num_options):
                       

            if multi: ### multi-updates here ###
                inds = np.arange(len(ob))
                is_ratios =seg["action_ratios"] + seg["term_ratios"]
                is_ratios=is_ratios[:,opt]
                prob_curr_opt= seg["prob_curr_opt"][:,opt]
                d = Dataset(dict(ob=ob[inds], ac=ac[inds], atarg=atarg[inds], vtarg=tdlamret[inds],  prob_curr_opt=prob_curr_opt[inds], is_ratios=is_ratios[inds], oldvpred=seg["vpred"][inds,opt]), shuffle=not pi.recurrent)

                #logger.log("Optimizing...")
                # Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    losses = [] # list of tuples, each of which gives the loss for a minibatch
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], batch["prob_curr_opt"], batch["is_ratios"])
                        adam.update(grads, optim_stepsize * cur_lrmult) 
                        losses.append(newlosses)

            else:
                indices = np.where(opts==opt)[0]
                print("batch size:",indices.size)
                if not indices.size:
                    continue

                if datas[opt] != 0:

                    if (indices.size < min_batch and datas[opt].n > min_batch):
                        datas[opt] = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)
                        continue

                    elif indices.size + datas[opt].n < min_batch:
                        oldmap = datas[opt].data_map

                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)
                        continue

                    elif (indices.size + datas[opt].n > min_batch and datas[opt].n < min_batch) or (indices.size > min_batch and datas[opt].n < min_batch):

                        oldmap = datas[opt].data_map
                        cat_ob = np.concatenate((oldmap['ob'],ob[indices]))
                        cat_ac = np.concatenate((oldmap['ac'],ac[indices]))
                        cat_atarg = np.concatenate((oldmap['atarg'],atarg[indices]))
                        cat_vtarg = np.concatenate((oldmap['vtarg'],tdlamret[indices]))
                        datas[opt] = d = Dataset(dict(ob=cat_ob, ac=cat_ac, atarg=cat_atarg, vtarg=cat_vtarg), shuffle=not pi.recurrent)

                    if (indices.size > min_batch and datas[opt].n > min_batch):
                        datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)

                elif datas[opt] == 0:
                    datas[opt] = d = Dataset(dict(ob=ob[indices], ac=ac[indices], atarg=atarg[indices], vtarg=tdlamret[indices]), shuffle=not pi.recurrent)



                optim_batchsize = optim_batchsize or ob.shape[0]

                #Here we do a bunch of optimization epochs over the data
                for _ in range(optim_epochs):
                    for batch in d.iterate_once(optim_batchsize):
                        *newlosses, grads = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult, [opt], np.ones_like(batch["vtarg"]), np.zeros_like(batch["vtarg"]))
                        adam.update(grads, optim_stepsize * cur_lrmult)


        termg = termgrad(seg["ob"], seg['last_options'], seg["term_adv"][range(len(seg["last_options"])),seg["last_options"]] )[0]
        adam.update(termg, piolr)

        if w_intfc:
            intgrads = intgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["op_probs"])[0]
            adam.update(intgrads, intlr)

        opgrads = opgrad(seg['ob'],seg['opts'], seg["last_betas"], op_atarg, seg["intfc"])[0]
        adam.update(opgrads, intlr)     
        
        lrlocal = (seg["ep_lens"], seg["ep_rets"])
        listoflrpairs=[lrlocal]
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        
        try:
            percent_solved = seg["cont_solved"]/seg["cont_episodes"]
        except:
            percent_solved = 0

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpSolved", percent_solved)
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)

        for option in range(num_options): # Show timesteps counter for each option
            count = option_counter.getCount(option) # Get the count for each option
            total_steps = sum(option_counter.option_timestep_counts.values())
            percent = (count / total_steps) * 100 if total_steps > 0 else 0
            option_str = f"{count} stp, {percent:.2f}%"
            logger.record_tabular("Option" + str(option), option_str)

        logger.dump_tabular()
        option_counter.resetCount()

        ### Book keeping
        # if saves:
        #     out = "{},{},{}"
        #     out+="\n"
        #     #avg_num_options=np.mean(np.sum(seg["activated_options"],axis=1))
        #     info = [iters_so_far, np.mean(rewbuffer), '']#,avg_num_options]
        #     results.write(out.format(*info))
        #     results.flush()
        ###




def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
