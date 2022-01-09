#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 11 Oct 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and different seeds 


# seed1
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_11.10.21_50demofixed_rnd --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_11.10.21_50demofixed_rnd --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_fixed.npz
# Killing and removing only the containers of the environment i am running
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetenv --format="{{.ID}}"))
# seed2
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_11.10.21_50demofixed_rnd_2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_11.10.21_50demofixed_rnd_2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_fixed.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetenv --format="{{.ID}}"))

# seed3
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_11.10.21_50demofixed_rnd_3 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_11.10.21_50demofixed_rnd_3 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_fixed.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetenv --format="{{.ID}}"))

# seed4
CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_11.10.21_50demofixed_rnd_4 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_11.10.21_50demofixed_rnd_4 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_09.07.21_50demo_fixed.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetenv --format="{{.ID}}"))


