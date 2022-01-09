#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run testing of the pickplacetarget different envs 


# Env1
python test_environment.py --env-id dVRLPickPlaceTargetEvalE1-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale1env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetEvalE2-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale2env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetEvalE3-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale3env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetEvalE4-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale4env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetEvalE5-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale5env --format="{{.ID}}"))
# Env6
python test_environment.py --env-id dVRLPickPlaceTargetEvalE6-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargetevale6env --format="{{.ID}}"))
