import gym
import numpy as np
import dVRL_simulator
import numpy as np
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, obj, target
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions

env_pick = gym.make("dVRLPick-v0") 

s = env_pick.reset() ; env_pick.render()

# Checking the initial distances
#print('ee_initial',s['observation'][0:3])
#print('obj_pose',s['observation'][-3:])
#print('target',s['desired_goal'])
#print('d_ee-obj',np.linalg.norm(s['observation'][0:3]-s['observation'][-3:]))
#print('d_ee-tar',np.linalg.norm(s['observation'][0:3]-s['desired_goal']))
#print('d_tar-obj',np.linalg.norm(s['desired_goal']-s['observation'][-3:]))


print('running-Approach') 
for _ in range(14):                                                             
    a = np.array([0,0,-1,1])                                                        
    s, r, _, info = env_pick.step(a)                                                
    #print('d_ee-obj',np.linalg.norm(s['observation'][0:3]-s['observation'][-3:]))
    #print('d_ee-tar',np.linalg.norm(s['observation'][0:3]-s['desired_goal']))
    #print('d_tar-obj',np.linalg.norm(s['desired_goal']-s['observation'][-3:]))

print('running-pick') 
for _ in range(2):                                                             
    a = np.array([0,0,0,-0.6])                                                        
    s, r, _, info = env_pick.step(a)                                                
    #print('d_ee-obj',np.linalg.norm(s['observation'][0:3]-s['observation'][-3:]))
    #print('d_ee-tar',np.linalg.norm(s['observation'][0:3]-s['desired_goal']))
    #print('d_tar-obj',np.linalg.norm(s['desired_goal']-s['observation'][-3:]))

print('running-reach')
for _ in range(30):
    a = np.append(np.clip(10*(s['desired_goal'] - s['observation'][-3:]), -1, 1), [-0.6])
    s, r, _, info = env_pick.step(a)
    #print('d_ee-obj',np.linalg.norm(s['observation'][0:3]-s['observation'][-3:]))
    print('d_ee-tar',np.linalg.norm(s['observation'][0:3]-s['desired_goal']))
    print('d_tar-obj',np.linalg.norm(s['desired_goal']-s['observation'][-3:]))
