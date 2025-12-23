# !/usr/bin/env python
from baselines.common import set_global_seeds, tf_util as U
import gymnasium as gym
import gymnasium_robotics
from gymnasium.wrappers import FlattenObservation
from gym import spaces
from gymnasium.wrappers import TimeLimit
from gymnasium.core import Wrapper
import logging
from baselines import logger
from half_cheetah import *
from walker2d import *
import panda_gym
import matplotlib as mpl
import matplotlib.pyplot as plt

class CustomFlattenObservation(gym.Wrapper):
    def __init__(self, env, objcoeff, its2herdis):
        super().__init__(env)
        self.old_position = None
        self.new = True
        self.auxEsperaInteracao = False
        self.cont = 0
        self.initial_position = np.array([0, 0, 0])
        self.objcoeff = objcoeff
        self.its2herdis = its2herdis
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(
                   env.observation_space['observation'].shape[0] +
                   env.observation_space['desired_goal'].shape[0] +
                   env.observation_space['achieved_goal'].shape[0],),
            dtype=np.float32
        )

    def reset(self, iters_so_far=0, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.flatten_obs_reset(obs)[0], obs

    def step(self, action, iters_so_far):
        obs, reward, done, truncated, info = self.env.step(action)
        goal_reward = self.env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info={})
        object_reward = self.env.compute_reward(obs["observation"][0:3], obs['observation'][3:6], info={})

        if iters_so_far <= self.its2herdis:
            reward = self.objcoeff*object_reward + (1-self.objcoeff)*goal_reward
        else:
            reward = goal_reward
        
        if goal_reward == 0: # If the object reaches the goal, the reward is reset to avoid incoherent learning
            reward = 0
        
        return self.flatten_obs(obs), reward, done, truncated, info, obs #It also returns the original obs, split, to make it easier to implement HER

    def flatten_obs(self, obs):
        # Concatenate observation, desired_goal and achieved_goal
        return np.concatenate((
            obs['observation'],
            obs['desired_goal'],
            obs['achieved_goal'],
        ))
    
    def flatten_obs_reset(self, obs):
        # Concatenates observation, desired_goal and achieved_goal
        return np.concatenate((
            obs['observation'],
            obs['desired_goal'],
            obs['achieved_goal'],
        )), []


def is_goal_env(env):
    """
    Verify if is a goal-based env.
    """
    obs_space = env.observation_space
    return isinstance(obs_space, gym.spaces.Dict) and \
           all(key in obs_space.spaces for key in ["observation", "desired_goal", "achieved_goal"])

def custom_reset_sim(self): # Custom function that will replace _reset_sim, keeping the object's initial position fixed
    try:
        if not hasattr(self, "_custom_env_name"):
            env_name = getattr(self, 'spec', None)
            if env_name is not None:
                env_name = self.spec.id
            elif hasattr(self, 'model') and hasattr(self.model, 'name'):
                env_name = self.model.name.decode('utf-8') if isinstance(self.model.name, bytes) else self.model.name
            else:
                env_name = type(self).__name__
            
            self._custom_env_name = env_name
            print(f"[custom_reset_sim] Ambiente detectado: {env_name}")

        self.data.time = self.initial_time
        self.data.qpos[:] = np.copy(self.initial_qpos)
        self.data.qvel[:] = np.copy(self.initial_qvel)
        if self.model.na != 0:
            self.data.act[:] = None
        
        if getattr(self, "has_object", False): # Checks if the environment has an object
            # Defines a fixed starting position (must be one of the positions used by the environment)
            if "Slide" in self._custom_env_name:
                object_xpos = np.array([1.08673492, 0.67469755]) # For slide. Can be any position set by the environment
            else:
                object_xpos = np.array([1.21251978, 0.69473044]) # For others. Can be any position set by the environment

            object_qpos = self._utils.get_joint_qpos(
                self.model, self.data, "object0:joint"
            )
            assert object_qpos.shape == (7,), "object0:joint tem formato inesperado."

            object_qpos[:2] = object_xpos
            self._utils.set_joint_qpos(
                self.model, self.data, "object0:joint", object_qpos
            )

        self._mujoco.mj_forward(self.model, self.data)
        return True

    except Exception as e:
        print(f"[custom_reset_sim] Erro detectado: {e}")
        return False


def train(env_id,num_timesteps,seed,num_options,app,saves,wsaves,epoch,w_intfc,switch,mainlr,intlr,piolr,multi,eta,render,optimsize,entcoeff,kher,hermvobj,objcoeff,its2herdis,kdecay):
    import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(seed)
    goal_env = False
    
    if env_id=="AntWalls":
        from antwalls import AntWallsEnv
        env=AntWallsEnv()
    else:

        if render:
            env = gym.make(env_id, render_mode='human', max_episode_steps=50)
        else:
            env = gym.make(env_id, max_episode_steps=50)

        if hasattr(env.unwrapped, "_reset_sim"): #Replaces _reset_sim from the environment with a custom one, with a fixed initial position of the object
            env.unwrapped._reset_sim = custom_reset_sim.__get__(env.unwrapped, type(env.unwrapped))

        if is_goal_env(env):
            goal_env = True
            print(f"{env_id} is a Goal Environment. Applying FlattenObservation.")
            env = CustomFlattenObservation(env, objcoeff, its2herdis)

        obs, obs_org = env.reset()


    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2, num_options=num_options, w_intfc=w_intfc)

    gym.logger.setLevel(logging.WARN)

    print(f'num_options: {num_options}')
    print(f'seed: {seed}')

    '''
    if not multi:
        if num_options ==1:
            optimsize=64
        elif num_options ==2:
            optimsize=32
        else:
            optimsize=int(64/num_options)
    else:
        optimsize=64
    '''

    num_timesteps = num_timesteps
    tperbatch = 2001
    pposgd_simple.learn(env, policy_fn, 
            max_timesteps=num_timesteps,
            timesteps_per_batch=tperbatch,
            clip_param=0.2, entcoeff=entcoeff,
            optim_epochs=10, optim_stepsize=mainlr, optim_batchsize=optimsize,
            gamma=0.99, lam=0.95, schedule='constant', num_options=num_options,
            app=app, saves=saves, wsaves=wsaves, epoch=epoch, seed=seed,
            w_intfc=w_intfc,switch=switch,intlr=intlr,piolr=piolr,multi=multi,
            eta=eta,render=render,is_goal_env=goal_env, kher=kher, hermvobj=hermvobj, 
            objcoeff=objcoeff, its2herdis=its2herdis, kdecay=kdecay
        )
    env.close()

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--env', help='environment ID', default='AntWalls')
    parser.add_argument('--timesteps', help='number of timesteps', type=int, default=3e7)
    parser.add_argument('--seed', help='RNG seed', type=int, default=1)
    parser.add_argument('--opt', help='number of options', type=int, default=2) 
    parser.add_argument('--app', help='Append to folder name', type=str, default='')        
    parser.add_argument('--saves', help='Save the returns at each iteration', dest='saves', action='store_true', default=False)
    parser.add_argument('--wsaves', help='Save the weights',dest='wsaves', action='store_true', default=False)    
    parser.add_argument('--switch', help='Switch task after 150 iterations', dest='switch', action='store_true', default=False)    
    parser.add_argument('--nointfc', help='Disables interet functions', dest='w_intfc', action='store_false', default=True)    
    parser.add_argument('--epoch', help='Load weights from a certain epoch', type=int, default=0) 
    parser.add_argument('--mainlr', type=float, default=1e-4)
    parser.add_argument('--intlr', type=float, default=1e-4)
    parser.add_argument('--piolr', type=float, default=1e-4)
    parser.add_argument('--optimsize', type=int, default=64)
    parser.add_argument('--entcoeff', type=float, default=0.00)
    parser.add_argument('--kher', type=int, default=4) #Value of k for HER
    parser.add_argument('--kdecay', type=int, help='decay rate of k', default=50) #Decay rate of k
    parser.add_argument('--hermvobj', type=int, help='HER only on trajectories with object movement', default=1) #Activates HER only on trajectories with object movement
    parser.add_argument('--objcoeff', type=float, help='coefficient of object_reward (0-1)', default=1) #Object interaction reward utilization coefficient in 2HER
    parser.add_argument('--its2herdis', type=int, help='iters to disable object_reward and 2HER', default=150) #Iteration to disable 2HER, keeping only standard HER active
    parser.add_argument('--multi', help='Multi updates', dest='multi', action='store_true', default=False)  
    parser.add_argument('--eta', type=float, default=0.1, help='trade off updates')
    parser.add_argument('--render', action='store_true', default=False)

    args = parser.parse_args()

    print('\n**************************************************\n', 
        args, '\n**************************************************\n')

    train(args.env, num_timesteps=args.timesteps, seed=args.seed, num_options=args.opt, app=args.app,
     saves=args.saves, wsaves=args.wsaves, epoch=args.epoch,w_intfc=args.w_intfc,
     switch=args.switch,mainlr=args.mainlr,intlr=args.intlr,piolr=args.piolr,multi=args.multi, eta=args.eta, 
     render=args.render, optimsize=args.optimsize, entcoeff=args.entcoeff, kher=args.kher, hermvobj=args.hermvobj, 
     objcoeff=args.objcoeff, its2herdis=args.its2herdis, kdecay=args.kdecay)


if __name__ == '__main__':
    main()
