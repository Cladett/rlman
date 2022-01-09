"""
@brief  Configuration script for the reach task. In this task the ee tooltip
        has to reach a sferical target to complete the task.
        Few input parameters will change compared to the original task to adapt
        the experiment to the one that uses images as state info.

        target_range=0.05 --> 0.0015 This is the value that is sum +- to the ee
                                     position.
        docker_container  --> name of the docker image, which get the same name
                              of the class but all lower case.
        initial_pos = np.array([0, 0, -0.11]) --> [ 0.10323,  0.012187, -0.1435]
                                                  New starting position at reset 
                                                  of robot tooltip.

@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   16 Feb 2021
"""
from gym import utils
from dVRL_simulator.PsmEnv_Position_pick_reach import PSMEnv_Position_pick_reach
import numpy as np


class PSMReachEnv(PSMEnv_Position_pick_reach):  # , utils.EzPickle):
    def __init__(self, psm_num=1, reward_type='sparse',
                 action_type='continuous'):
        # initial_pos = np.array([0.10323,  0.012187, -0.1435])
        initial_pos = np.array([0, 0, -0.14])

        super(PSMReachEnv, self).__init__(psm_num = psm_num, n_substeps = 1,
                            block_gripper=True,
                            has_object=False,
                            target_in_the_air=True,
                            height_offset=0.01,
                            target_offset=[0, 0, 0.005],
                            obj_range=0.025,
                            target_range=0.015,
                            # distance_threshold=0.003,
                            distance_threshold=0.005,
                            initial_pos=initial_pos,
                            reward_type=reward_type,
                            dynamics_enabled=False,
                            two_dimension_only=False,
                            randomize_initial_pos_obj=False,
                            randomize_initial_pos_ee=False,
                            action_type=action_type,
                            docker_container =self.__class__.__name__.lower())

        utils.EzPickle.__init__(self)
