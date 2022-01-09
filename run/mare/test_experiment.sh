#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to if envs can train properly 

# TO TEST: no demo

#python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=10000 
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=10000 
#python3 -m baselines.run --alg=her --env=dVRLReach-v0 --num_timesteps=10000 
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
#
## TO TEST: with demo 
#python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=10000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail/pickrail_06.10.21_50demo_rnd.npz 
python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=10000 --num_demo 50 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail/pickrail_02.07.21_50demo.npz
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)

## TO TEST: with demo 
#python3 -m baselines.run --alg=her --env=dVRLPickPlace-v0 --num_timesteps=10000 --num_demo 50 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplace/pickplace_13.07.21_50demo_rnd.npz  
#python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=10000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_rnd.npz 
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)

