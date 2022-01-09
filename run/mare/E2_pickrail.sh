#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of env using her+ddpg with demo 

# Testing the running of the experiments with baselines
# Testing with no demo
#python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=10000 # FAILS
#python3 -m baselines.run --alg=her --env=dVRLReachRail-v0 --num_timesteps=10000 
# Testing with demo
#python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=10000 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz


# SEED 1
# Running now
python3 -m baselines.run --alg=her --env=dVRLReachRail-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/reachchrail_04.10.2021_1 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/reachrail_04.10.2021
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/pickrail_03.10.2021_1 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/pickrail_03.10.2021 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz

# SEED 2
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/pickrail_03.10.2021_2 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/pickrail_03.10.2021_2 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz
# SEED 3
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/pickrail_03.10.2021_3 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/pickrail_03.10.2021_3 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz

# SEED 4
#mpirun -n 4 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/reach/models/pickrail_03.10.2021_4 --log_path=/home/claudia/data/EXPERIMENTS_RL/reach/logs/pickrail_03.10.2021_4 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickrail_02.07.21_50demo.npz
