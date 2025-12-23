# Multi-Updates Option Critic with Dual Objectives Hindsight Experience Replay

Hierarchical Reinforcement Learning (HRL) frameworks like Option-Critic (OC) and Multi-updates Option Critic (MOC) have introduced significant advancements in learning reusable options. However, these methods underperform in multi-goal environments with sparse rewards, where actions must be linked to temporally distant outcomes. To address this limitation, we first propose MOC-HER, which integrates the Hindsight Experience Replay (HER) mechanism into the MOC framework. By relabeling goals from achieved outcomes, MOC-HER can solve sparse reward environments that are intractable for the original MOC. However, this approach is insufficient for object manipulation tasks, where the reward depends on the object reaching the goal rather than on the agent’s direct interaction. This makes it extremely difficult for HRL agents to discover how to interact with these objects. To overcome this issue, we introduce Dual Objectives Hindsight Experience Replay (2HER), a novel extension that creates two sets of virtual goals. In addition to relabeling goals based on the object's final state (standard HER), 2HER also generates goals from the agent's effector positions, rewarding the agent for both interacting with the object and completing the task. Experimental results in robotic manipulation environments show that MOC-2HER achieves success rates of up to 90%, compared to less than 11% for both MOC and MOC-HER. These results highlight the effectiveness of our dual objective relabeling strategy in sparse reward, multi-goal tasks.

#### Installation
```
virtualenv moc_cc --python=python3
source moc_cc/bin/activate
pip install tensorflow==1.15.0 
cd continuous_control
pip install -e . 
pip install gym==0.9.3
pip install mujoco-py==0.5.1
```

#### Launch
```
cd baselines/ppoc_int
python run_mujoco.py --env FetchPush-v2 --seed 35 --opt 2 --entcoeff 0.005 --optimsize 64 --kher 8 --hermvobj 1 --its2herdis 150 --kdecay 37 --objcoeff=0.8
```
