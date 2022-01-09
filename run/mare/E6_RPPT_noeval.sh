#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of pickplacetarget using her+ddpg with demo.
#          N.B. all the variable have been randomized in this training

# SEED 1
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_14.10.21_1 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_14.10.21_1
#
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
## SEE 2
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_14.10.21_2 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_14.10.21_2
#
## Removing all the previous containers
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)

## SEED 3
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_14.10.21_3 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_14.10.21_3
#
## Removing all the previous containers
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
## SEED 4
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_rnd.npz --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/models/pickplacetarget_07.10.21_4 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetarget/logs/pickplacetarget_07.10.21_4
#
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
