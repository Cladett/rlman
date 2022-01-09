#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of env using her+ddpg with demo 

#CUDA_VISIBLE_DEVICES=3 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --eval_env=dVRLPickPlaceTargetEval-v0 --num_timesteps=200000 --save_video_interval=100 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_video --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_video --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_09.07.21_50demo_fixed.npz
# TO TEST on my computer
#python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz

# SEED 1
# Running now
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReach-v0 --num_timesteps=400000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/reach_02.10.2021 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/reach_02.10.2021

# SEED 2
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReach-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/reach_02.10.2021_2 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/reach_02.10.2021_2
# SEED 3
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReach-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/reach_02.10.2021_3 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/reach_02.10.2021_3

# SEED 4
mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLReach-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/reach_02.10.2021_4 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/reach_02.10.2021_4
