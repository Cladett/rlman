#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 


# seed1
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_12.10.21_50demo_rnd2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_12.10.21_50demo_rnd2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_20.07.21_50demo_rnd2.npz
# Killing and removing only the containers of the environment i am running
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
# seed2
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_12.10.21_50demo_rnd2_2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_12.10.21_50demo_rnd2_2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_20.07.21_50demo_rnd2.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))

# seed3
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_12.10.21_50demo_rnd2_3 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_12.10.21_50demo_rnd2_3 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_20.07.21_50demo_rnd2.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))

# seed4
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_12.10.21_50demo_rnd2_4 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_12.10.21_50demo_rnd2_4 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_20.07.21_50demo_rnd2.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))


