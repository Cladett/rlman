#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of pickplacetarget env with no eval env and differen seeds 


# seed2
CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=400000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_30.09.21_50demo_rnd1_s2 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_30.09.21_50demo_rnd1_s2 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_20.07.21_50demo_rnd1.npz

# seed3
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=400000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_30.09.21_50demo_rnd1_s3 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_30.09.21_50demo_rnd1_s3 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_20.07.21_50demo_rnd1.npz

# seed4
#CUDA_VISIBLE_DEVICES=0 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=400000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_30.09.21_50demo_rnd1_s4 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_30.09.21_50demo_rnd1_s4 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_20.07.21_50demo_rnd1.npz


