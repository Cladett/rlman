"""
@brief    Script to test the pretrain with demonstrations. It uses DDPG+HER 
          with .npz file of the demonstrations. Using STABLE-BASELINES
          The pretrain run for 1000epochs and the model is train for 2*10e5 
          timesteps
@author   Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date     1 Sep 2020
"""

import argparse
import gym
import numpy as np
import os
import stable_baselines 

# My imports
import dVRL_simulator
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, targetK, rail
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines.gail import ExpertDataset 
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec


def add_parameters(parser): 
    parser.add_argument("--npz-path", help="Path to where npz file.")


def main():
    # Read command line parameters 
    parser = argparse.ArgumentParser() 
    add_parameters(parser)
    args = parser.parse_args()
   
    # Tensorboard configuration
    #fmt_str = 'stdout,log,csv,tensorboard' 
    #folder = '/home/claudia/catkin_ws/src/dVRL/dVRL_simulator/record_demonstration_dVRL/logs'
    #stable_baselines.logger.configure(folder, fmt_str)
    
    # Initialise gym environment
    env_dvrk = gym.make("dVRLPickPlace-v0")
    wrapped_env = HERGoalEnvWrapper(env_dvrk)

    # Loading the demonstrations
    dataset = ExpertDataset(expert_path=args.npz_path, traj_limitation=-1, 
            batch_size=128, verbose=1)
    
    # Training with HER  
    #model_class = DDPG  # works also with SAC, DDPG and TD3
    #goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE
    #print('Ready to test HER')
    #model = HER('MlpPolicy', wrapped_env, model_class, n_sampled_goal=4, goal_selection_strategy=goal_selection_strategy,
    #            tensorboard_log="./her_dvrl_tensorboard/" ,verbose=1 )
    model = DDPG('MlpPolicy', wrapped_env, verbose=1)

    # Pretrain the model using the demonstration
    model.pretrain(dataset, n_epochs=1000)
    model.save("./models/pretrain_7.12.20", cloudpickle = True)
    # Train the model
    #print('Starting the training')
    #model.learn(2*10e5, tb_log_name="first_run")
    #model.save("./models/herddpg1_dvrl_env")
    print('Done')

    return


if __name__ == '__main__':
    main() 
