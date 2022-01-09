#!/bin/bash

cd dVRL_simulator/environments/
cd reach_ee_dockerfile/ && docker build -t vrep_ee_reach .
cd ../pick_ee_dockerfile/ && docker build -t vrep_ee_pick .
cd ../pickplace_ee_dockerfile/ && docker build -t vrep_ee_pickplace .
cd ../pickrail_dockerfile/ && docker build -t vrep_ee_pickrail .
