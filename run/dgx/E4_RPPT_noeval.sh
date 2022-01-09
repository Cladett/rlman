#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 


## E1 - seed1
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE1-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E1_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E1_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_5.npz
## Killing and removing only the containers of the environment i am running
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete1env --format="{{.ID}}"))
## E1 - seed2
##CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE1-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E1_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E1_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_5.npz
## Killing and removing only the containers of the environment i am running
##docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete1env --format="{{.ID}}"))
######################################
## E2 - seed1
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE2-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E2_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E2_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete2env --format="{{.ID}}"))
## E2 - seed2
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE2-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E2_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E2_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz
##docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete2env --format="{{.ID}}"))
######################################
## E3 - seed1
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E3_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E3_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_2.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
## E3 - seed2
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E3_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E3_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_2.npz
##docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
####################################
# E4 - seed1
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE4-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E4_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E4_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete4env --format="{{.ID}}"))
# E4 - seed2
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE4-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E4_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E4_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete4env --format="{{.ID}}"))

####################################
# E5 - seed1
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE5-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E5_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E5_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_3.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete5env --format="{{.ID}}"))
# E5 - seed2
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE5-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E5_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E5_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_3.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete5env --format="{{.ID}}"))
####################################
# E6 - seed1
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE6-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E6_s1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E6_s1 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_4.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete6env --format="{{.ID}}"))
# E6 - seed1
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTargetE6-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_14.10.21_50demo_E6_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_14.10.21_50demo_E6_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_4.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete6env --format="{{.ID}}"))

