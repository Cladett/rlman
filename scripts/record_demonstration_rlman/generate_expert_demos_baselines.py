"""
@brief   This is a script is used to record expert demonstrations used for 
         baselines algorithm. 
         The exectuted task is the Pick and Place of a PAF over a target. The 
         demonstrations are recorded giving pre-defined actions to the robot 
         in the simulation enviroment and use those action to guide the 
         "demonstrated" motions. 
@input   The file has as input parameters the following
         --save-path   the path where the demonstration are saved as npz
         --episodes    number of trajectories (demo) we want to record
@author  Claudia D'Ettorre (c.dettorre@ucl.ac.uk). 
@date    27 Aug 2020.
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
from dVRL_simulator.vrep.simObjects import table, targetK, rail
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions 
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper

class Demo():

    def __init__(self, n_episodes, dic_path, task):
        # Store params
        self.dic_path = dic_path
        self.n_episodes = n_episodes
        self.task = task

        # Initialise data structures
        #self.episode_returns = []
        self.actions = []
        self.obs = []
        self.info = []
        #self.rewards = []
        #self.episode_starts = []

        # Initializing the different evns
        if self.task == 'pickrail':
            self.env_dvrk_unwrapped = gym.make("dVRLPickRail-v0")
            # Testing unwrapped env.
            self.env_dvrk = self.env_dvrk_unwrapped
            self.env_dvrk = HERGoalEnvWrapper(self.env_dvrk_unwrapped)
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 10 
        elif self.task == 'pickplaceobj':
            self.env_dvrk_unwrapped = gym.make("dVRLPick-v0")
            self.env_dvrk = HERGoalEnvWrapper(self.env_dvrk_unwrapped)
            # Defining number of timesteps for each part of the task 
            # self.APPROACH_STEPS = 25  # ee start from pos [0, 0, -0.11] 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        elif self.task == 'pickplacekidney':
            self.env_dvrk_unwrapped = gym.make("dVRLPickPlace-v0")
            self.env_dvrk = HERGoalEnvWrapper(self.env_dvrk_unwrapped)
            # Defining number of timesteps for each part of the task 
            # self.APPROACH_STEPS = 25  # ee start from pos [0, 0, -0.11] 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        elif self.task == 'pickplacetarget':
            self.env_dvrk_unwrapped = gym.make("dVRLPickPlaceTarget-v0")
            self.env_dvrk = HERGoalEnvWrapper(self.env_dvrk_unwrapped)
            # Defining number of timesteps for each part of the task 
            # self.APPROACH_STEPS = 25  # ee start from pos [0, 0, -0.11] 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        

    def run_episode(self):
        """
        @brief  Each episode runs a reset of the environment and whole 
                demonstration 
        """
        # Set up the enviroment 
        prev_obs = self.env_dvrk.reset()
        self.env_dvrk.render()
        
        # Run steps: the pickrail task does not have any reach phase
        if self.task == 'pickrail':
            # Run steps
            prev_obs = self.approach(prev_obs)
            prev_obs = self.grasp(prev_obs)
        else:
            # Run steps
            prev_obs = self.approach(prev_obs)
            prev_obs = self.grasp(prev_obs)
            prev_obs = self.reach(prev_obs)
        
        # Compute final return for each episode
        #episode_total_reward = sum(self.rewards[-self.env_dvrk_unwrapped._max_episode_steps:])
        #self.episode_returns.append(episode_total_reward)
        
    def approach(self, prev_obs):
        """
        @brief This function moves the end effector from the initial position 
               towards the object. 
        """
        for i in range(self.APPROACH_STEPS):
            # If object is not randomised and start from below the zero position
            # of the ee 
            #action = [0, 0, -1, 1]  
            # If the initial position of the object is randomised
            pos_ee   = prev_obs[0:3]
            pos_obj  = prev_obs[4:7]
            desired_goal  = prev_obs[-3:]
            if self.task == 'pickrail':
                raw_action = np.array(desired_goal - pos_ee)
            else:
                raw_action = np.array(pos_obj - pos_ee)
            action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                10 * raw_action[2], 1], -1, 1)

            # Execute action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  
            
            # Store action results in episode lists
            self.actions.append(action)
            self.obs.append(prev_obs) 
            self.info.append(info)
            #self.rewards.append(reward) 
            #self.episode_starts.append(i == 0) 
            prev_obs = obs
        return prev_obs

    def grasp(self, prev_obs):
        """
        @brief This function executes the static grasping of the object. 
        """
        for i in range(self.GRASP_STEPS):
            # Execute the action in the enviroment 
            action = [0, 0, 0, -0.6]
            obs, reward, done, info = self.env_dvrk.step(action)  
            
            # Store action results in episode lists
            self.actions.append(action) 
            self.obs.append(prev_obs) 
            self.info.append(info)
            #self.rewards.append(reward) 
            #self.episode_starts.append(False) 
            prev_obs = obs
        return prev_obs

    def reach(self, prev_obs):
        """
        @brief This method executes the reaching towards the target.
               Using a wrapped environment the observation are not present as 
               a dic of 'desired_goal', 'observation' anymore. They are now a 
               single numpy array so in order to access each component we need
               to acces different part of the array with the following structure
               
               'observation': 
                array([ 5.00679016e-05, -6.19888306e-05,  8.96453857e-05,  
                        1.15444704e-03, -7.15255737e-05, -2.86102295e-05, 
                        -8.39406013e-01]),
                'achieved_goal': array([-7.15255737e-05, -2.86102295e-05, 
                                        -8.39406013e-01]),
                'desired_goal': array([-0.85572243,  0.70448399, -0.37535763])}

               
        """
        steps = self.env_dvrk_unwrapped._max_episode_steps \
                - self.APPROACH_STEPS \
                - self.GRASP_STEPS
        for i in range(steps):
            goal     = prev_obs['desired_goal']
            pos_ee   = prev_obs['observation'][-3:]
            pos_obj  = prev_obs['observation'][-4:]
            raw_action = np.array(goal - pos_ee)
            action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                10 * raw_action[2], -0.5], -1, 1)

            # Executing the action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  
            
            # Store action results in episode lists
            self.actions.append(action) 
            self.obs.append(prev_obs) 
            self.info.append(info)
            #self.rewards.append(reward) 
            #self.episode_starts.append(False) 
            prev_obs = obs

        # Adding control if the goal is not reached
        return prev_obs


    def save(self):
        assert(len(self.actions) == len(self.obs))
        #assert(len(self.obs) == len(self.rewards))
        episode_update_dict = {
            'info': self.info,
            'acs': self.actions,
            'obs': self.obs,
            #'rewards': self.rewards,
            #'episode_returns': self.episode_returns,
            #'episode_starts': self.episode_starts,
        }
        np.savez(self.dic_path, **episode_update_dict)


def add_parameters(parser):
    parser.add_argument("--save-path", help="Path to save demonstration npz.")
    parser.add_argument("--episodes", help="Number of episodes to record.")
    parser.add_argument("--task", help="Type of task demonstrated (pickrail,\
                        pickplacetarget, pickplacekidney).")


def convert_param_to_correct_type(args):
    args.episodes = int(args.episodes)
    return args


def main(): 
    # Read command line parameters
    parser = argparse.ArgumentParser()
    add_parameters(parser)
    args = parser.parse_args()
    args = convert_param_to_correct_type(args)
    
    expert_demo = Demo(args.episodes, args.save_path, args.task)
    for i in range(args.episodes):
        print('Running demonstration number', (i+1))
        expert_demo.run_episode()
    expert_demo.save()
    
    return 

if __name__ == '__main__':
    main()
