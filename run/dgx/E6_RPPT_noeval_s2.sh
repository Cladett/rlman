#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 


# E6 - seed1
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE6-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E6_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E6_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_4.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete6env --format="{{.ID}}"))
# E6 - seed1
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE6-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E6_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E6_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_4.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete6env --format="{{.ID}}"))

