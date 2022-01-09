"""
@brief  Configuration script for the place task of the rail over 
        the kidney.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020
"""

from gym import utils
from dVRL_simulator.PsmEnv_Position_reachkid import PSMEnv_Position_reachkid
import numpy as np


class PSMReachKidneyEnv(PSMEnv_Position_reachkid):#, utils.EzPickle):
    def __init__(self, psm_num=1, reward_type='sparse', 
                 randomize_obj=True, randomize_ee=True,
                 #randomize_grasp=False, 
                 randomize_kid = False, randomize_kid_or=False,
                 randomize_target_point=True, # test
                 action_type='continuous'):
        initial_pos_ee=np.array([0, 0, -0.07])
        initial_pos_k=np.array([0.05, 0.07, 0])
        #initial_pos_k=np.array([0.05, -0.05, 0])

        super(PSMReachKidneyEnv, self).__init__(
                        psm_num=psm_num, n_substeps=1, 
                        block_gripper=True,
                        has_object=True, 
                        target_in_the_air=True, 
                        height_offset=0.0001,
                        target_offset=[0,0,0.038], 
                        obj_range=0.025, 
                        target_range=0.075,
                        distance_threshold=0.003, 
                        initial_pos=initial_pos_ee, 
                        initial_pos_k=initial_pos_k, 
                        reward_type=reward_type,
                        dynamics_enabled=False, 
                        two_dimension_only=False, 
                        randomize_initial_pos_ee=randomize_ee,
                        randomize_initial_pos_obj=randomize_obj, 
			randomize_initial_pos_kidney=randomize_kid, 
                        randomize_initial_or_kidney=randomize_kid_or,
                        # To debug
                        randomize_target_point=randomize_target_point, # test
                        #randomize_grasping_site=randomize_grasp, # test
			#randomize_initial_or_obj=False, # test 
                        action_type=action_type,
                        docker_container = self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)
