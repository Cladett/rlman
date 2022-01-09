"""
@bief trying to debug the problem of running reachkidney env
"""

from gym import utils
from dVRL_simulator.PsmEnv_Position_reachkid import PSMEnv_Position_reachkid
import numpy as np

class PSMReachKidneyEnv(PSMEnv_Position_reachkid):#, utils.EzPickle):
    def __init__(self, psm_num = 1, reward_type='sparse', randomize_obj = False, randomize_ee = True, action_type='continuous'):
        initial_pos = np.array([ 0,  0, -0.07])
        super(PSMReachKidneyEnv, self).__init__(psm_num = psm_num, n_substeps = 1,
                        block_gripper = True, has_object = False, 
                        target_in_the_air = True, height_offset = 0.0001,
                        target_offset = [0,0,0.038],
                        obj_range = 0.025, target_range = 0.075,
                        distance_threshold = 0.003, initial_pos = initial_pos, 
                        reward_type = reward_type, dynamics_enabled = False, 
                        two_dimension_only = False, 
                        randomize_initial_pos_obj = randomize_obj, 
                        randomize_initial_pos_ee = randomize_ee, 
			randomize_initial_or_kidney = True,
			randomize_initial_pos_kidney = True,
                        action_type=action_type,
                        docker_container = self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)
