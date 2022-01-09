#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of env using her+ddpg with demo 

CUDA_VISIBLE_DEVICES=3 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_09.07.21_50demo_fixed.npz
#CUDA_VISIBLE_DEVICES=3 python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --eval_env=dVRLPickPlaceTargetEval-v0 --num_timesteps=200000 --save_video_interval=100 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickplacetarget_video --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickplacetarget_video --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickplacetarget_09.07.21_50demo_fixed.npz
#CUDA_VISIBLE_DEVICES=1 python3 -m baselines.run --alg=her --env=dVRLPickRail-v0 --num_timesteps=200000 --save_path=/home/cladet/workspace/dVRL/baselines/models/pickrail_nodemo_30.09.21_1 --log_path=/home/cladet/workspace/dVRL/baselines/logs/pickrail__nodemo_30.09.21 --demo_file=/home/cladet/workspace/dVRL/scripts/record_demonstration_dVRL/pickrail_30.09.21_50demo.npz
#python3 -m baselines.run --alg=her --env=dVRLPickPlaceTarget-v0 --num_timesteps=200000  
