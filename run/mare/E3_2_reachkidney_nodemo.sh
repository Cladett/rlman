#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of env using her+ddpg with demo 

# SEED 1
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReachKidney-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/models/reachkidney_06.10.21_nodemo_1 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/logs/reachkidney_06.10.21_nodemo_1
#
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
## SEE 2
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReachKidney-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/models/reachkidney_06.10.21_nodemo_2 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/logs/reachkidney_06.10.21_nodemo_2
#
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
## SEED 3
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReachKidney-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/models/reachkidney_06.10.21_nodemo_3 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/logs/reachkidney_06.10.21_nodemo_3
#
## Removing all the previous containers
#docker stop $(docker ps -a -q)
#docker rm $(docker ps -a -q)
# SEED 4
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReachKidney-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/models/reachkidney_06.10.21_nodemo_4 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/logs/reachkidney_06.10.21_nodemo_4

# Removing all the previous containers
docker stop $(docker ps -a -q)
docker rm $(docker ps -a -q)
