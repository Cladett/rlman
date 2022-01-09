#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gym
import dVRL_simulator
import numpy as np
import time


# In[2]:


env = gym.make("dVRLPickPlace-v0")

# In[3]:

for _ in range(2):
	s = env.reset() ; env.render() 

#print('ee_initial', s['observation'][0:3])
#print('rail_pose', s['observation'][-3:])
#print('target', s['desired_goal'])
#print('d_ee-obj',np.linalg.norm(s['observation'][0:3]-s['observation'][-3:]))
#print('d_ee-tar',np.linalg.norm(s['observation'][0:3]-s['desired_goal']))
#print('d_tar-obj',np.linalg.norm(s['desired_goal']-s['observation'][-3:]))

time.sleep(2)

for _ in range(60):
	a = np.append(np.clip(10*(s['observation'][-3:] - s['observation'][0:3]), -1, 1), [1]); s, r, _, info = env.step(a)
	time.sleep(0.1)
for _ in range(2):
	a = np.array([0,0,0,-0.6]) ; s,r,_,info = env.step(a)   
	time.sleep(0.1)
for _ in range(88):
	a = np.append(np.clip(10*(s['desired_goal'] - s['achieved_goal']), -1, 1), [-0.6]) ; s,r,_,info = env.step(a)
	#a = np.append(np.clip(10*(s['desired_goal_t'] - s['achieved_goal_t']), -1, 1), [-0.6]) ; s,r,_,info = env.step(a)
	time.sleep(0.1)
print(info)

# In[4]:


#env_reach.close()
