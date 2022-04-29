"""
@brief   This is a script is used to record expert demonstrations. 
         The target task is given as input. The task that are actually
         implemented for now are:
         1. pickrail: the ee simply picks the rail
         2. pickplaceobj: the ee has to pick the little cilinder and drive it towards
            the target 
         3. pickplacekidney: the ee has to pick the rail and drive it towards
            the kidney with the right orientation and to the right target 
            position.
         The demonstrations are recorded giving pre-defined actions to the robot 
         in the simulation enviroment and use those action to guide the 
         "demonstrated" motions. 
@input   The file has as input parameters the following
         --save-path   the path where the demonstration are saved as npz
         --episodes    number of trajectories (demo) we want to record
         --task        type of task that we want to record
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
        self.episode_returns = []
        self.actions = []
        self.obs = []
        self.rewards = []
        self.episode_starts = []
        
        # Initialise gym env and wrapping it for having obs as single array 
        # of floats instead of a dictionary. Selecting the env based on the task
        if self.task == 'pickrail':
            self.env_dvrk_unwrapped = gym.make("dVRLPickRail-v0")
            self.env_dvrk = HERGoalEnvWrapper(self.env_dvrk_unwrapped)
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
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

    def run_episode(self, count_success):
        """
        @brief  Each episode runs a reset of the environment and whole 
                demonstration 
        """
        # Set up the enviroment 
        prev_obs = self.env_dvrk.reset()
        self.env_dvrk.render()
        
        # Run steps: the pickrail task does not have any reach phase
        if self.task == 'pickrail':
            prev_obs = self.approach(prev_obs)
            prev_obs, info_grasp = self.grasp(prev_obs)
        else: 
            prev_obs = self.approach(prev_obs)
            prev_obs = self.grasp(prev_obs)
            prev_obs, info = self.reach(prev_obs)
        
        # Compute final return for each episode
        episode_total_reward = sum(self.rewards[-self.env_dvrk_unwrapped._max_episode_steps:])
        self.episode_returns.append(episode_total_reward)

        # Counting how many demonstrations are sucessful
        if self.task == 'pickrail':
            if info_grasp['is_success']:
                count_success +=1
        else:
            if info['is_success']:
                count_success +=1
        return count_success

        
    def approach(self, prev_obs):
        """
        @brief This function moves the end effector from the initial position 
               towards the object. 
        """
        for i in range(self.APPROACH_STEPS):
            pos_ee_start = 0 
            pos_ee_end = 3
            pos_ee   = prev_obs[pos_ee_start: pos_ee_end] 
            pos_obj_start = 4
            pos_obj_end = 7
            pos_obj  = prev_obs[pos_obj_start:pos_obj_end] 
            raw_action = np.array(pos_obj - pos_ee)
            action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                10 * raw_action[2], 1], -1, 1)
            # Execute action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  

            # Store action results in episode lists
            self.actions.append(action)
            #self.obs.append(prev_obs) 
            self.obs.append(np.append(prev_obs[0:7],prev_obs[10:13]))
            self.rewards.append(reward) 
            self.episode_starts.append(i == 0) 
            prev_obs = obs
        return prev_obs

    def grasp(self, prev_obs):
        """
        @brief This function executes the static grasping of the object. 
        """
        for i in range(self.GRASP_STEPS):
            # Execute the action in the enviroment 
            action = [0, 0, 0, -0.5]
            obs, reward, done, info = self.env_dvrk.step(action)  

            # Store action results in episode lists
            self.actions.append(action) 
            #self.obs.append(prev_obs) 
            self.obs.append(np.append(prev_obs[0:7],prev_obs[10:13]))
            self.rewards.append(reward) 
            self.episode_starts.append(False) 
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

                In the new release SB3 the definition of obeservation has been                                                                                     
                updated using only 'observation' and 'desired_goal'.

               
        """
        steps = self.env_dvrk_unwrapped._max_episode_steps \
                - self.APPROACH_STEPS \
                - self.GRASP_STEPS
        for i in range(steps):
            desired_goal_start = 10
            desired_goal_end = 13
            goal     = prev_obs[desired_goal_start: desired_goal_end]
            pos_ee_start = 0 
            pos_ee_end = 3
            pos_ee   = prev_obs[pos_ee_start: pos_ee_end] 
            pos_obj_start = 4
            pos_obj_end = 7
            pos_obj  = prev_obs[pos_obj_start:pos_obj_end] 
            raw_action = np.array(goal - pos_obj)
            action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                10 * raw_action[2], -0.5], -1, 1)

            # Executing the action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  
            
            # Store action results in episode lists
            self.actions.append(action) 
            #self.obs.append(prev_obs) 
            # Only storing observation and desired goal needed for the pretrain SB3
            self.obs.append(np.append(prev_obs[0:7],prev_obs[10:13])) 
            self.rewards.append(reward) 
            self.episode_starts.append(False) 
            prev_obs = obs

        # Adding control if the goal is not reached
        return prev_obs, info


    def save(self):
        assert(len(self.actions) == len(self.obs))
        assert(len(self.obs) == len(self.rewards))
        episode_update_dict = {
            'actions': self.actions,
            'obs': self.obs,
            'rewards': self.rewards,
            'episode_returns': self.episode_returns,
            'episode_starts': self.episode_starts,
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
    
    # Perform simulation
    #generate_expert_demonstration(save_path=args.save_path, 
    #                              n_episodes=args.episodes)

    # Initializing counter for episode success
    count_success = 0
    expert_demo = Demo(args.episodes, args.save_path, args.task)
    for i in range(args.episodes):
        print('Running demonstration number', (i+1))
        count_success = expert_demo.run_episode(count_success)

    print('Total demonstrations', args.episodes)
    print('Successful demos', count_success)
    expert_demo.save()
    
    return 

if __name__ == '__main__':
    main()
