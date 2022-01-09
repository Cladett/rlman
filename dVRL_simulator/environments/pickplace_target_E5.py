"""
@brief  Configuration script for the pick and place task of the rail over a 
        circular target
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020
"""

from gym import utils
from dVRL_simulator.PsmEnv_Position_rail_target import PSMEnv_Position_pickplace_t
import numpy as np

class PSMPickPlaceTargetE5Env(PSMEnv_Position_pickplace_t):
    def __init__(self, psm_num = 1, reward_type='sparse',
                 randomize_obj=True, randomize_obj_orientation=True,
                 randomize_ee=True, randomize_grasp=True,
                 randomize_target=False, action_type='continuous'):
        # This is the coordinate of the initial position of the ee on the /base
        initial_pos=np.array([0, 0, -0.07])
        # Definition of all the parameter used in the simulation 
        super(PSMPickPlaceTargetE5Env, self).__init__(psm_num=psm_num, n_substeps=1,
                        block_gripper=False, has_object=True, 
                        target_in_the_air=True,
                        # height_offset=0.0001,
                        # Testing the respondable rail 
                        height_offset=0.008,
                        target_offset=[0,0,0.025],
                        obj_range=0.025, target_range=0.025, 
                        # x_range [-45, 0], y_range [-180,+180], z_range [-45,+45]
                        x_range = [-30, 0], y_range = [-45, 45], 
                        z_range = [-45, 45],
                        distance_threshold=0.003, initial_pos=initial_pos, 
                        reward_type=reward_type, dynamics_enabled=False, 
                        two_dimension_only=False, 
                        randomize_initial_pos_obj=randomize_obj, 
                        randomize_initial_or_obj=randomize_obj_orientation, 
                        randomize_initial_pos_ee=randomize_ee, 
                        randomize_grasping_site=randomize_grasp,
                        randomize_target=randomize_target,
                        action_type=action_type,
                        docker_container=self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)
