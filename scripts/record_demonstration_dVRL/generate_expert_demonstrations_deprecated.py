"""
******DEPRECATED *******

Please refer to generate_expert_demos

************************
"""

"""
@bried   Script to record demonstration of pick and place task. 
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


def generate_expert_demonstrations(save_path=None, n_episodes=100):
    
    #import pudb; pudb.set_trace()
    """
    Method to generate expert demonstration using Pick environment.
    
    :param save_path: (str) Path without the extension where the expert 
                       dataset will be saved.
                       (ex: 'expert_cartpole' -> creates 'expert_cartpole.npz').
                       If not specified, it will not save, and just return 
                       the generated expert trajectories.
    :param n_episodes: (int) Number of trajectories (episodes) to record.

    :return: (dict) the generated expert trajectories. 
    """
    
    env_dvrk = gym.make("dVRLPick-v0") 

    actions = []
    observations = []
    rewards = []
    episode_returns = np.zeros((n_episodes,))
    episode_starts = []

    ep_idx = 0
    reward_sum = 0.0

    while ep_idx < n_episodes:

        episode_starts.append(True)
        obs = env_dvrk.reset()
        env_dvrk.render()
        observations.append(obs)

        step  = 0 
        # Approaching phase
        for i in range(0,13):
            a = [0,0, -1, 1]
            state,r, done ,info = env_dvrk.step(a)
            step +=1

            actions.append(a)
            observations.append(state)
            rewards.append(r)
            reward_sum += r
            episode_starts.append(done)

        # Grasping phase 
        for i in range(0,2):
            a = [0,0, 0, -0.5]
            state,r, done ,info =  env_dvrk.step(a)
            step +=1

            actions.append(a)
            observations.append(state)
            rewards.append(r)
            reward_sum += r
            episode_starts.append(done)

        # Driving towards the target
        while step < env_dvrk._max_episode_steps:
            goal     = state['desired_goal']
            pos_ee   = state['observation'][-3:]
            pos_obj  = state['observation'][-4:]
            action = np.array(goal - pos_ee)

            a = np.clip([10*action[0], 10*action[1], 10*action[2], -0.5], -1, 1)
            state,r, done,info =  env_dvrk.step(a)
            step +=1

            actions.append(a)
            observations.append(state)
            rewards.append(r)
            reward_sum += r
            episode_starts.append(False)

                
        episode_returns[ep_idx] = reward_sum
        reward_sum = 0.0
        ep_idx +=1

    print('Final Reward at {} is {}'.format(ep_idx,r))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    #if len(observations) != len(actions):
    #    print("The number of actions is not the same of obeservations")

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
        } # type: Dic[str, np.array]

    # Save .npz file 
    if save_path is not None:
        np.savez(save_path, **numpy_dict) 

    env_dvrk.close()

    return numpy_dict

def add_parameters(parser):
    parser.add_argument("--save-path", help="Path to save demonstration npz.")
    parser.add_argument("--episodes", help="Number of episodes to record.")

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
    generate_expert_demonstrations(save_path=args.save_path,
            n_episodes=args.episodes)
    return


if __name__ == '__main__':  
    main()
