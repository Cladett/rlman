"""
@brief   Script to run some random execution of the pick and place task. 
         Mainly to record the video.
@author  Claudia D'Ettorre
@date    23 Oct 2020

"""

import gym
import numpy as np
from scipy.spatial.transform import Rotation as R
from numpy.linalg import inv
import dVRL_simulator
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, targetK, rail
import transforms3d.euler as euler
import time
import transforms3d.quaternions as quaternions

env_pick = gym.make("dVRLPickPlace-v0") 

for _ in range(5):
    s = env_pick.reset() ; env_pick.render()

    # If I wanna used the loops
    for _ in range(60):
        #a = np.array([0,0,-1,1]) ; s,r,_,info = env_pick.step(a) 
        a = np.append(np.clip(10*(s['observation'][-3:] - s['observation'][0:3]), -1, 1), [1]) ; s, r, _, info = env_pick.step(a) # for randomize initial position
                  
    for _ in range(2):
        a = np.array([0,0,0,-0.6]) ; s,r,_,info = env_pick.step(a)        
                    
    for _ in range(100):
        a = np.append(np.clip(10*(s['desired_goal'] - s['observation'][-3:]), -1, 1), [-0.6]) ; s,r,_,info = env_pick.step(a)

