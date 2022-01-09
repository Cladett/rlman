#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 

# E5 - seed1

####################################
# E5 - seed1
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE5-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E5_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E5_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_3.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete5env --format="{{.ID}}"))
# E5 - seed2
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE5-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E5_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E5_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_3.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete5env --format="{{.ID}}"))
####################################

