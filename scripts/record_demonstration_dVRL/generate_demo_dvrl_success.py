"""
@brief  This script is provided by dVRL authors to record demonstration  
        according their implementation of the pick and place task. This script
        records simulation in the simulated environment and save it as .npz
@dat    27 Aug 2020
"""

import gym
import numpy as np

# My imports 
import dVRL_simulator
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, targetK, rail
#from dVRL_simulator.vrep.simObjects_reachkid_neri import table, rail, targetK, target 
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions

# Initialization of the environment
#env_dvrk = gym.make("dVRLPickPlace-v0") 
#$env_dvrk = gym.make("dVRLPickRail-v0") 
#env_dvrk = gym.make("dVRLReachKidney-v0") # please che the script that needs to be changed 
#env_dvrk = gym.make("dVRLPick-v0") 
env_dvrk = gym.make("dVRLPickPlaceTarget-v0") 
#env_dvrk = gym.make("dVRLPickPlace-v0") 
# Number of demonstration we want to record.
#numEp_forData = 10
#numEp_forData = 25
#numEp_forData = 50
numEp_forData = 100
#numEp_forData = 4
#numEp_forData = 15 

actions = []
observations = []
infos = []
it = 0
#for it in range(0,numEp_forData):
while it < numEp_forData:

    episodeActs = []
    episodeObs  = []
    episodeInfo = []

    state = env_dvrk.reset() # to fix bug with the rnd reset
    state = env_dvrk.reset()
    episodeObs.append(state)
    env_dvrk.render()

    step  = 0

# This is the approaching phase
    for i in range(0,60):
    #for i in range(0,120): # Modiyin the script only for the rail pick
        #a = [0,0, -1, 1]

        # Modifyin the script for recording PickRail
        goal     = state['desired_goal']
        pos_obj  = state['observation'][-3:]
        pos_ee   = state['observation'][0:3]
        #action = np.array(goal - pos_ee)  # for the pick rail
        action = np.array(pos_obj - pos_ee)  # for the pick and place
        

        a = np.clip([10*action[0], 10*action[1], 10*action[2], 1], -1, 1)
        state,r, _,info = env_dvrk.step(a)
        step += 1

        episodeActs.append(a)
        episodeObs.append(state)
        episodeInfo.append(info)

# Static grasping 
    for i in range(0, 2):
        a = [0,0, 0, -0.5]
        state,r, _,info =  env_dvrk.step(a)
        step += 1

        episodeActs.append(a)
        episodeObs.append(state)
        episodeInfo.append(info)

# Driving the object towards the target. 
    while step < env_dvrk._max_episode_steps:
        goal     = state['desired_goal']
        #pos_ee   = state['observation'][-3:]
        pos_ee   = state['observation'][0:3]
        pos_obj  = state['observation'][-3:]
        #action = np.array(goal - pos_ee) #for reach kidney taks
        action = np.array(goal - pos_obj) # for the pick and place task over kidney

        a = np.clip([10*action[0], 10*action[1], 10*action[2], -0.5], -1, 1)
        #a = a[0:3] # for reach kidney we don't need the grip
        state,r, _,info =  env_dvrk.step(a)
        step += 1

        episodeActs.append(a)
        episodeObs.append(state)
        episodeInfo.append(info)

    #import pudb;pudb.set_trace()
    if r == 0:
        actions.append(episodeActs)
        observations.append(episodeObs)
        infos.append(episodeInfo)
        it+=1
        
    print('Demo num: ', it)
    print('Reward: ', r)
#
print('Final Reward at {} is {}'.format(it,r))

# Saving the file as npz
fileName = "pickplacetarget_10.11.21_100demo_success"
np.savez_compressed(fileName, acs=actions, obs=observations, info=infos)
