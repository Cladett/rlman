#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to test a trained agent 


## TO TEST: with demo 
python3 -m baselines.run --alg=her --env=dVRLReachKidney-v0 --num_timesteps=0 --load_path=/home/claudia/data/EXPERIMENTS_RL/dVRL/reachkidney/models/reachkidney_05.10.21_4 --play

