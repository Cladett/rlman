'''
@breif Testing running vrep in a docker container
@author Claudia D'Ettorre 
@date 26 Feb 2021
'''

import time
import numpy as np
import gym
import dVRL_simulator

env = gym.make("dVRLPickRail-v0") 
s = env.reset() ; env.render()
print('all good')
