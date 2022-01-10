#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import dVRL_simulator
import numpy as np


# In[3]:
env_reach = gym.make("dVRLReach-v0")


# In[4]:

# During this step I am doing the following (defined in PsmEnv.py):
# - calling _reset_sim(): the entire enviroment is reset in the starting position
# - _sample_goal() given as output in the observation.
# - s=_get_obs(): i am getting an observation of my enviroment (in the starting position)
s = env_reach.reset()

# This renders the enviroment always in mode 'human'. It's used to actually disply the simulation.
# There two other modalities that can be found inside PsmEnv.py
# - matplotlib
# - rgb
# Those two are both related to the use of the camera in the simulation.
env_reach.render()

# I am arbitrary defining the 100 steps.
for _ in range(100):
    a = np.clip(10*(s['desired_goal'] - s['observation']), -1, 1)
    s, r, _, info = env_reach.step(a)
    #print('running')

print(info)


# In[4]:


#env_reach.close()

# In[5]:


env_pick = gym.make("dVRLPick-v0")


# In[6]:


s = env_pick.reset()
env_pick.render()

for _ in range(20):
    a = np.array([0,0,-1,1])
    s, r, _, info = env_pick.step(a)
    #print('running-A')
    print(info,'running-A')

for _ in range(2):
    a = np.array([0,0,0,-0.6])
    s, r, _, info = env_pick.step(a)
    #print('running-B') 
    print(info,'running-B')

for _ in range(50):
    a = np.append(np.clip(10*(s['desired_goal'] - s['observation'][-3:]), -1, 1), [-0.6])
    s, r, _, info = env_pick.step(a)
    #print('running-C')
    print(info,'running-C')
    
print(info)


# In[7]:


#env_pick.close()


# In[ ]:



