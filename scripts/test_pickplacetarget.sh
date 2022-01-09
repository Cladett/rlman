#!/bin/bash
#
#
#  @author: Claudia D'Ettorre
#  @date: 20 Sep 2021  
#  @brief: script to run testing of the pickplacetarget different envs 


# Env1
python test_environment.py --env-id dVRLPickPlaceTargetE1-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete1env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetE2-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete2env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetE3-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete3env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetE4-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete4env --format="{{.ID}}"))
# Env1
python test_environment.py --env-id dVRLPickPlaceTargetE5-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete5env --format="{{.ID}}"))
# Env6
python test_environment.py --env-id dVRLPickPlaceTargetE6-v0
docker rm $(docker stop $(docker ps -a -q --filter ancestor=psmpickplacetargete6env --format="{{.ID}}"))
