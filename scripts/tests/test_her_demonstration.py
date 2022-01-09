"""
@brief   This script is used to test a training with DDPG+HER of the pick & 
         place task using demonstrations recorded on the simulation enviroment. 
@author  Claudia D'Ettorre
@date    28 Aug 2020
"""
import argparse                                                                                                                                                    
import gym                                                                                                                                                         
import numpy as np                                                                                                                                                 
import os                                                                                                                                                          
import warnings                                                                                                                                                    
from typing import Dict 

# My imports
import dVRL_simulator
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, obj, target, kidney, rail
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
from stable_baselines.gail import ExpertDataset 
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper


def main():
    model_class = DDPG  # works also with SAC, DDPG and TD3

    # Initialization of the enviroment 
    env = gym.make("dVRLPick-v0")  

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future' # some of GoalSelectionStrategy.FUTURE

    # Wrap the model
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, 
            goal_selection_strategy=goal_selection_strategy, verbose=1)

    # Uploading the demonstrations 
    dataset = ExpertDataset(expert_path='/home/claudia/catkin_ws/src/dVRL/dVRL_simulator/record_demonstration_dVRL/dvrl_obj_demo.npz',
                                    traj_limitation=-1, batch_size=128)

    # Pretrain the model
    #import pudb; pudb.set_trace()
    model.pretrain(dataset, n_epochs=1000)


    # Train the model
    model.learn(1000)

    model.save("./her_bit_env")
    return

if __name__ == '__main__':                                                                                                                                         
    main()   
