#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 21 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 


# E1 - seed1
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlace-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplace_21.10.21_50demo_E1_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplace_21.10.21_50demo_E1_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplace/pickplace_21.10.21_50demos.npz
# Killing and removing only the containers of the environment i am running
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplaceenv --format="{{.ID}}"))
# E1 - seed2
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlace-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplace_21.10.21_50demo_E1_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplace_21.10.21_50demo_E1_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplace/pickplace_21.10.21_50demos.npz
# Killing and removing only the containers of the environment i am running
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplaceenv --format="{{.ID}}"))
