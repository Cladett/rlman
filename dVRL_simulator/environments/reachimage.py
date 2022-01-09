"""
@brief  Configuration script for the reaching task using image as state 
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020
"""

from gym import utils
from dVRL_simulator.PsmEnv_Position_reachimage import PSMEnv_Position_reachimage
import numpy as np

class PSMReachImageEnv(PSMEnv_Position_reachimage):#, utils.EzPickle):
    def __init__(self, psm_num = 1, reward_type='dense', action_type='discrete'):
        #initial_pos = np.array([0,  0, -0.09])
        initial_pos = np.array([ 0.10323,  0.012187, -0.1435]) 

        super(PSMReachImageEnv, self).__init__(psm_num = psm_num, n_substeps = 1,
                        block_gripper = True, has_object = False, 
                        target_in_the_air = True, height_offset = 0.0,             
                        target_offset = [0,0,0], obj_range = 0.05,
                        target_range = 0.015, distance_threshold = 0.003,
                        initial_pos = initial_pos, reward_type = reward_type,
                        dynamics_enabled = False, two_dimension_only = False, 
                        randomize_initial_pos_obj = False,
                        randomize_initial_pos_ee = False,            
                        action_type=action_type,
                        docker_container = self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)



