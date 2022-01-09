#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run training of pickplacetarget using her+ddpg with demo.
#          N.B. all the variable have been randomized in this training

# E2 - seed1
mpirun -n 4 python -m baselines.run --alg=her --env=dVRLPickPlaceTargetE2-v0 --eval_env=dVRLPickPlaceTargetEvalE2-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/models/pickplacetarget_19.10.21_50demo_E2_s1 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/logs/pickplacetarget_19.10.21_50demo_E2_s1 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete2env --format="{{.ID}}"))
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale2env --format="{{.ID}}"))
## E2 - seed2
#mpirun -n 4 python -m baselines.run --alg=her --env=dVRLPickPlaceTargetE2-v0 --eval_env=dVRLPickPlaceTargetEvalE2-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/models/pickplacetarget_19.10.21_50demo_E2_s2 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/logs/pickplacetarget_19.10.21_50demo_E2_s2 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_13.10.21_50demos.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete2env --format="{{.ID}}"))
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale2env --format="{{.ID}}"))
## E3 - seed1
#mpirun -n 4 python -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --eval_env=dVRLPickPlaceTargetEvalE3-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/models/pickplacetarget_19.10.21_50demo_E3_s1 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/logs/pickplacetarget_19.10.21_50demo_E3_s1 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_2.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale3env --format="{{.ID}}"))
### E3 - seed2
#mpirun -n 4 python -m baselines.run --alg=her --env=dVRLPickPlaceTargetE3-v0 --eval_env=dVRLPickPlaceTargetEvalE3-v0 --num_timesteps=200000 --save_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/models/pickplacetarget_19.10.21_50demo_E3_s2 --log_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/pickplacetargeteval/logs/pickplacetarget_19.10.21_50demo_E3_s2 --demo_file=/home/claudia/catkin_ws/src/dVRL/scripts/record_demonstration_dVRL/pickplacetarget/pickplacetarget_14.10.21_50demos_2.npz
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
#docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale3env --format="{{.ID}}"))

