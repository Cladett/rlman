"""
@brief  Configuration script for the pick and place task of the rail over 
        the kidney.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020
"""

from gym import utils
from dVRL_simulator.PsmEnv_Position_pickplace_k import PSMEnv_Position_pickplace_k
import numpy as np

class PSMPickPlaceEnv(PSMEnv_Position_pickplace_k):#, utils.EzPickle):
    def __init__(self, psm_num=1, reward_type='sparse', 
                 randomize_obj=False, randomize_obj_or =False , randomize_ee=False,
                 randomize_grasp=True, randomize_target_point=True,
                 randomize_kid_pos=False, randomize_kid_or=False,
                 action_type='continuous'):
        initial_pos_ee=np.array([0, 0, -0.11])
        initial_pos_k=np.array([0.05, 0.07, 0])

        super(PSMPickPlaceEnv, self).__init__(
                        psm_num=psm_num, n_substeps=1, 
                        block_gripper=False,
                        has_object=True, 
                        target_in_the_air=True, 
                        height_offset=0.0001,
                        target_offset=[0,0,0.038], 
                        obj_range=0.025, 
                        target_range=0.075,
                        x_range = [-30, 0], y_range = [-45, 45], 
                        z_range = [-45, 45],
                        distance_threshold=0.003, 
                        initial_pos=initial_pos_ee, 
                        initial_pos_k=initial_pos_k, 
                        reward_type=reward_type,
                        dynamics_enabled=False, 
                        two_dimension_only=False, 
                        randomize_initial_pos_ee=randomize_ee,
                        randomize_initial_pos_obj=randomize_obj, 
                        randomize_initial_or_obj=randomize_obj_or, 
                        randomize_initial_pos_kidney=randomize_kid_pos, 
                        randomize_initial_or_kidney=randomize_kid_or,
                        randomize_target_point=randomize_target_point,
                        randomize_grasping_site=randomize_grasp,
                        action_type=action_type,
                        docker_container = self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)
