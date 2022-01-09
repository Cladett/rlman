#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Nov 2021  
#  @brief: script to run training of pickplacetarget using her+ddpg with demo.
#          Testing effects of having different numnber of demos. 

## SEED 1
##mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_10.11.21_10demo_success.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_22.11.21_10demo_s --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_22.11.21_10demo_s --num_demo=10

## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
## SEE 2
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_10.11.21_25demo_success.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_22.11.21_25demo_s --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_22.11.21_25demo_s --num_demo=25
#
# Removing all the previous containers
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
#
## SEED 3
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_10.11.21_50demo_success.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_22.11.21_50demo_s --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_22.11.21_50demo_s --num_demo=50 
#
# Removing all the previous containers
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
## SEED 4
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_10.11.21_100demo_success.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_22.11.21_100demo_s --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_22.11.21_100demo_s  --num_demo=100
#
# Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
