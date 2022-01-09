"""
@brief  Script used to control the main steps of the pick of the PAF rail 
        and place it over the red circular target.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020

"""

import time
import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation as R

# My imports 
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects_pickplacetarget import table, target, rail
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PSMEnv_Position_pickrail(PSMEnv):

    def __init__(self, psm_num, n_substeps, block_gripper, has_object,
            target_in_the_air, height_offset, target_offset, obj_range,
            target_range, x_range, y_range, z_range, distance_threshold,
            initial_pos, reward_type, dynamics_enabled, two_dimension_only,
            randomize_initial_pos_obj, randomize_initial_or_obj,
            randomize_initial_pos_ee, randomize_grasping_site,
            randomize_target, docker_container, action_type):

        """Initializes a new signle PSM Position Controlled Environment
        Args:
            psm_num (int): which psm you are using (1 or 2)
            n_substeps (int): number of substeps the simulation runs on every 
                              call to step
            gripper_extra_height (float): additional height above the table 
                                          when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked 
                                     (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be 
                                         in the air above the table or on the 
                                         table surface
            height_offset (float): offset from the table for everything
            target_offset ( array with 3 elements): offset of the target, 
                                                    usually z is set to the 
                                                    height of the object
            obj_range (float): range of a uniform distribution for sampling 
                               initial object positions
            target_range (float): range of a uniform distribution for sampling 
                                  a target Note: target_range must be 
                                  set > obj_range
            distance_threshold (float): the threshold after which a goal is 
                                        considered achieved
            initial_pos  (3x1 float array): The initial position for the PSM 
                                            when reseting the environment. 
            reward_type ('sparse' or 'dense'): the reward type, i.e. 
                                               sparse or dense
            dynamics_enabled (boolean): To enable dynamics or not
            two_dimension_only (boolean): To only do table top or not. 
                                          target_in_the_air must be set off too.
            randomize_initial_pos_obj (boolean)
            docker_container (string): name of the docker container that loads 
                                       the v-rep
            randomize_initial_or_obj (boolean)
            randomize_initial_pos_kidney (boolean)
            randomize_initial_or_kidney (boolean)
            randomize_grasping_site (boolean)
            randomize_target (boolean)
            action_type ('continuous', 'discrete'): the action space type, i.e. continuous or discrete

        """

        #self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.height_offset = height_offset
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.initial_pos = initial_pos
        self.reward_type = reward_type
        self.dynamics_enabled = dynamics_enabled
        self.two_dimension_only = two_dimension_only
        self.randomize_initial_pos_obj = randomize_initial_pos_obj
        self.randomize_initial_or_obj = randomize_initial_or_obj
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.randomize_initial_pos_ee = randomize_initial_pos_ee
        self.randomize_grasping_site = randomize_grasping_site
        self.randomize_target = randomize_target
        self.action_type = action_type

        if self.block_gripper:
            self.n_actions = 3
            self.n_states = 3 + self.has_object * 3
        else:
            self.n_actions = 4
            self.n_states = 4 + self.has_object * 3

        super(
            PSMEnv_Position_pickrail,
            self).__init__(
            psm_num=psm_num,
            n_substeps=n_substeps,
            n_states=self.n_states,
            n_goals=3,
            n_actions=self.n_actions,
            camera_enabled=False,
            docker_container=docker_container,
            action_type=action_type)

        # Claudia: modifing the code to grasp the rail
        self.target = target(self.clientID, psm_num)
        if self.has_object:
            self.rail = rail(self.clientID)
        self.table = table(self.clientID)

        self.prev_ee_pos = np.zeros((3,))
        self.prev_ee_rot = np.zeros((3,))
        self.prev_rail_pos = np.zeros((3,))  # rail
        self.prev_rail_rot = np.zeros((3,))  # rail
        self.prev_jaw_pos = 0

        if(psm_num == 1):
            self.psm = self.psm1
        else:
            self.psm = self.psm2

        # Start the streaming from VREP for specific data:

        # PSM Arms:
        self.psm.getPoseAtEE(ignoreError=True, initialize=True)
        self.psm.getJawAngle(ignoreError=True, initialize=True)

        # Used for _sample_goal
        self.target.getPosition(
            self.psm.base_handle,
            ignoreError=True,
            initialize=True)

        # Used for _reset_sim
        self.table.getPose(
            self.psm.base_handle,
            ignoreError=True,
            initialize=True)
        # Claudia: used to initialize the streaming from all the dummies.
        if self.has_object:
            self.rail.getPose(
                self.rail.dummy1_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy2_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy3_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy4_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy5_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy6_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy7_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            self.rail.getPose(
                self.rail.dummy8_rail_handle,
                self.psm.base_handle,
                ignoreError=True,
                initialize=True)  # Also used in _get_obs
            # Used for _get_obs
            grasp = self.rail.isGrasped(ignoreError=True, initialize=True)
            self.rail.readProximity(
                ignoreError=True)  # for the proximity sensor

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):

        #d = goal_distance(achieved_goal, goal) * \
        #    self.target_range  # Need to scale it back!

        #if self.reward_type == 'sparse':
        #    return -(d > self.distance_threshold).astype(np.float32)
        #else:
        #    return -100 * d

        ################
        ## TODO:testing new reward function 
        #import pudb;pudb.set_trace()
        d = goal_distance(achieved_goal, goal) * \
            self.target_range  # Need to scale it back!
        grasp_reward = self.rail.isGrasped(ignoreError=True,
                                            initialize=True)

        #print('Is grasped?', grasp_reward)
        #print('d', d)
        #print((d < self.distance_threshold))
        #print(grasp_reward)
        #print(type(((d < self.distance_threshold) and grasp_reward)))

        if self.reward_type == 'sparse':
            #return -(d > self.distance_threshold).astype(np.float32)
            #return -(not grasp_reward or (d > self.distance_threshold)).astype(np.float32)
            #return -((d > self.distance_threshold) or (not grasp_reward)).astype(np.float32)
            return np.bool_(np.logical_and((d < self.distance_threshold), grasp_reward)).astype(np.float32) - 1
        else:
            return -100 * d
        ##############

        #if self.reward_type == 'sparse':
        #    grasp_reward = self.rail.isGrasped(ignoreError=True,
        #                                        initialize=True)
        #    if grasp_reward:
        #        reward = 0.0
        #    else:
        #        reward = -1.0
        #    return reward
        #else:
        #    return -100*d

    # PsmEnv methods
    # ----------------------------

    def _set_action(self, action):
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action 
                                # outside of this scope

        if self.block_gripper:
            pos_ctrl = action
        else:
            pos_ctrl, gripper_ctrl = action[:3], action[3]
            gripper_ctrl = (gripper_ctrl + 1.0) / 2.0

        pos_ee, quat_ee = self.psm.getPoseAtEE()
        pos_ee = pos_ee + pos_ctrl * 0.001  # the maximum change in position 
                                            # is 0.1cm

        # Get table information to constrain orientation and position
        pos_table, q_table = self.table.getPose(self.psm.base_handle)

        # Make sure tool tip is not in the table by checking tt and which side
        # of the table it is on

        # DH parameters to find tt position
        ct = np.cos(0)
        st = np.sin(0)

        ca = np.cos(-np.pi / 2.0)
        sa = np.sin(-np.pi / 2.0)

        T_x = np.array([[1, 0, 0, 0],
                        [0, ca, -sa, 0],
                        [0, sa, ca, 0],
                        [0, 0, 0, 1]])
        T_z = np.array([[ct, -st, 0, 0],
                        [st, ct, 0, 0],
                        [0, 0, 1, 0.0102],
                        [0, 0, 0, 1]])

        ee_T_tt = np.dot(T_x, T_z)

        pos_tt, quat_tt = self.psm.matrix2posquat(
            np.dot(self.psm.posquat2Matrix(pos_ee, quat_ee), ee_T_tt))

        pos_tt_on_table, distanceFromTable = self._project_point_on_table(
            pos_tt)

        # if the distance from the table is negative, then we need to project pos_tt onto the table top.
        # Or if two dim only are enabled
        if distanceFromTable < 0 or self.two_dimension_only:
            pos_ee, _ = self.psm.matrix2posquat(
                np.dot(
                    self.psm.posquat2Matrix(
                        pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))

        # Claudia: changing constrain orientation to the rail.
        _, q_dummy = self.rail.getPose(
            self.dummy_rail_handle, self.psm.base_handle)
        # the position is computed related to the dummy. I had Rx(-90)
        temp_q = quaternions.qmult([q_dummy[3], q_dummy[0], q_dummy[1], 
                                    q_dummy[2]], [0.7, -0.7, 0, 0])  
        rot_ctrl = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

        if self.block_gripper:
            gripper_ctrl = 0

        # Make sure the new pos doesn't go out of bounds!!!
        upper_bound = self.initial_pos + self.target_range + 0.01
        lower_bound = self.initial_pos - self.target_range - 0.01

        pos_ee = np.clip(pos_ee, lower_bound, upper_bound)

        self.psm.setPoseAtEE(pos_ee, rot_ctrl, gripper_ctrl)
        self._simulator_step() # Fixing the step bug
        
        return

    def _get_obs(self):
        #Normalize ee_position:
        ee_pos,  _ = self.psm.getPoseAtEE()
        ee_pos = (ee_pos - self.initial_pos)/self.target_range
        jaw_pos = self.psm.getJawAngle()

        grasped = self.rail.isGrasped()

        if not grasped:
            # TODO: should i add the jaw as well ?
            rail_pos, _ = self.rail.getPose(self.dummy_rail_handle, 
                                            self.psm.base_handle)
            rail_pos = (rail_pos - self.initial_pos) / self.target_range
            achieved_goal = np.squeeze(ee_pos)

        else:
            rail_pos = np.zeros((3,))
            achieved_goal = np.squeeze(ee_pos)

        obs = np.concatenate((ee_pos, np.array([jaw_pos]), rail_pos)) 

        self.prev_ee_pos = ee_pos
        self.prev_ee_rot = np.zeros((3,))
        self.prev_rail_pos = rail_pos
        self.prev_rail_rot = np.zeros((3,))
        self.prev_jaw_pos = jaw_pos


        # Adding check control on the grasp.
        grasp_success = self.rail.isGrasped(ignoreError=True, initialize=True)

        return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),  # ee_pos 
                'desired_goal' : self.goal.copy()
        }

    def _reset_sim(self):
        """
        @brief      This is the first method that is called in the simulation. 
                    It is used to reset the entire scene everytime the reset() 
                    is called.
        @details    The object of the scene are reset as follows
                    1. The tooltip position is set to a random initial 
                       position and with orientation perpendicular to the table.
                       Randomized in target_range.
                    2. The rail is set to a random intial position. Ranzomised in
                       obj_range.
        """
        pos_ee = self._define_tt_pose()
        
        # If the object (usually the rail) is grasped from a previous simulation
        # we release it
        if self.has_object:
            self.rail.removeGrasped(ignoreError=True)
        self._simulator_step()
        
        # Disable dynamics because they make the simulation slower
        if self.dynamics_enabled:
            self.psm.setDynamicsMode(1, ignoreError=True)

        # Place the rail in a random position within the scene
        if self.has_object:
            self._define_rail_pose(pos_ee)
        else:
            self.prev_rail_pos = self.prev_rail_rot = np.zeros((3,)) 
        self._simulator_step()

        return True

    def _define_tt_pose(self, initial_tt_offset=0.035):
        """
        This methos is called to define the random initial position
        of the tooltip(tt) of the end effector.
        The tt is randomised inside the volume of the target_space.
        Initially the target volume is centered in the /table
        It is then moved so that it get centered around the tt initial
        position defined in the configuration file.
        @param[in]  initial_tt_offset  This is an offset express in m. It is
                                       set because we always want the tt to
                                       have an itialia offset from table surface

        """

        # Get the constrained orientation of the tt initilly perpendicular
        # to the table surface.
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)
        temp_q = quaternions.qmult([q_table[3], q_table[0], q_table[1],
                                    q_table[2]], [0.5, -0.5, -0.5, 0.5])
        ee_quat_constrained = np.array(
            [temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

        # Put the EE in the correct orientation
        self.psm.setDynamicsMode(0, ignoreError=True)
        self._simulator_step()

        # Definition of [x,y,z] coordinate in the /table
        if self.randomize_initial_pos_ee:
            pos_ee = self._random_testing_volume(initial_tt_offset, b_T_table)
        else:
            pos_ee = self.initial_pos

        self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError=True)

        return pos_ee

    def _random_testing_volume(self, initial_tt_offset, b_T_table):

        # To randomly select one of the subvolume
        volume = np.random.randint(1,4)
        
        # Volume 1
        if volume == 1:
            x_min = -self.target_range 
            x_max = self.target_range
            y_min = -self.target_range
            y_max = self.target_range
            z_min = self.target_range/2 
            z_max = self.target_range
        # Volume 2
        elif volume == 2:
            x_min = 0 
            x_max = self.target_range
            y_min = -self.target_range
            y_max = self.target_range
            z_min = 0 
            z_max = self.target_range/2
        # Volume 3
        elif volume == 3:
            x_min = -self.target_range 
            x_max = 0 
            y_min = 0 
            y_max = self.target_range
            z_min = 0 
            z_max = self.target_range/2

        if self.target_in_the_air:
            z = self.np_random.uniform(
                z_min, z_max) + initial_tt_offset
        else:
            z = initial_tt_offset

        # Add target_offset for goal. And rotating from /table to /base
        deltaPos_b_homogeneous_one = np.append(
            self.np_random.uniform(x_min, x_max), 
            self.np_random.uniform(y_min, y_max))
        deltaPos_b_homogeneous_one = np.append(
            deltaPos_b_homogeneous_one, [z, 0])
        deltaPos_b_homogeneous = np.dot(
            b_T_table, deltaPos_b_homogeneous_one)  # applying rotation

        # Project EE on to the table and add the deltaEEPos to that -
        # adding z-translation to the final coordinate position
        pos_ee_projectedOnTable, _ = self._project_point_on_table(
            self.initial_pos)
        pos = pos_ee_projectedOnTable + deltaPos_b_homogeneous[0:3]

        return pos

    def _define_rail_pose(
            self,
            pos_ee,
            initial_rail_offset=0.015,
            dist_from_ee=0,
            minimum_d=0.015,
            grasp_site=None):
        """
        This method is called to define the rail random initial position inside
        the obj_range.
        1. Position: First [x,y,z] coordinate are seletected in the /table. 
           Then they are rotate from /table to /base and then the offset 
           from the tooltip is computed projecting the tt position on the table.
           So that at the end the randomization of the object is done in 
           the /base around the initial position of the tt.
        2. Orientation: it is randomizes based on range of rotation around the
           x,y,z axes of the rail. The admissible range are defined inside the
           configuration file based on reasonable oprational values
        3. Grasping site: random selection of one of the possible 8 grasping
           position. The position is defined by a dummy which is randomly
           selected for every reset and the value is print on screeen with the
           respective dummy color.
        4. Correct the rail position if it is below the table surface. 
           So that the dummy is always reachable above the table
           surface.

        @param[in]   pos_ee                Position of the end-effector.
                     initial_rail_offset   This is an initial offset counted 
                                           from the table surface.
                     dist_from_ee          This variable is initialised to 0 and
                                           it is used to check that the rail
                                           initial position does not crush
                                           with the tt.
                     minimum_d             This value representes the minimum
                                           acccettable distance in m between the
                                           tt and the rail so that they do not
                                           crush.
                     grasp_site            Variable used to defined the selected
                                           grasping site.
        
        @details    Randomization: in the configuration file pick.py the randomization
                    is initialised. It can be chosen if the position, orientation 
                    and grasping site need to randomized or be fixed to a 
                    standard initial position.
                    If not randomised: orientation like the table, grasping 
                    site number=4 (middle one) and starting position right
                    below the tool tip initial_position.

        """
        # 1.Computing position of the dummy of the rail
        dummy_pos = self._set_dummy_pos(pos_ee, initial_rail_offset,
                dist_from_ee, minimum_d)

        # 2.Computing Orientation of the dummy 
        if self.randomize_initial_or_obj:
            q_dummy = self._randomize_dummy_orientation()
        else:
            q_dummy = ([0, 0, 0, 1])  # same orientation table surface if it 
                                      # is not randomized.

        # 3.Setting which dummy is gonna be grasped
        if self.randomize_grasping_site:
            grasp_site = self.np_random.randint(1, 9)
        else:
            grasp_site = 4 

        # FIXME: forcing the rail initil position below the table to see if
        # the control function works
        #dummy_pos = [0, 0, -0.14]
        #dummy_pos[2] = -0.14
        #q_dummy = [-0.13052619, 0, 0, 0.99144486] # quat with only -15 rotx 

        # I am giving as input the position of the selected dummy and i get 
        # as output the pos_rail_set the position of the rail in the /base 
        self.dummy_rail_handle, pos_rail_set = self.rail.setPose(
            dummy_pos, q_dummy, grasp_site, self.psm.base_handle, 
            ignoreError=True)  

        self._simulator_step()

        # FIXME: printing some value to debug
        #print('initil position of the dummy chosen', dummy_pos)
        #print('position rail before correct /base', pos_rail_set)
        #print('position rail before correct /table', self._transf_base_to_table(pos_rail_set))

        # 4. Correct rail position if below table surface
        self._correct_rail_pos(q_dummy)
        self._simulator_step()

        return

    def _set_dummy_pos(self, pos_ee, initial_rail_offset, dist_from_ee, minimum_d):
        """
        @details   In this method we work with the dummy position. 
                   The position of the rail is set based on the dummy randomly
                   selected. So when we chose the cartesian coordinate is 
                   referred to the dummy. The position of the reference
                   frame of the rail is then computed accordingly when 
                   setPose is called.

        @param[in]    pos_ee                 Cartesian position of the endeffector
                                             (ee) in the /base.
                      initial_rail_offset    Initial offset on the z-axe of the 
                                             /table. 
                      dist_from_ee           Variable used to compute the distance
                                             between the tt and the rail. It is 
                                             initialised to 0
                      minimum_d              It is the minimum distance that we 
                                             need between the rail and the tt to 
                                             be sure they are not overlapping 
                                             with each other. 

        @param[out]   dummy_pos   Position of the dummy in the /base

        """
        # Checking that the rail does not overlap with the tt position
        z = initial_rail_offset
        while dist_from_ee < minimum_d:

            if self.randomize_initial_pos_obj:
                x = self.np_random.uniform(-self.obj_range, self.obj_range)
                y = self.np_random.uniform(-self.obj_range, self.obj_range)
            else:
                x = 0
                y = 0

            # Rotating from /table to /base
            pos_table, q_table = self.table.getPose(self.psm.base_handle)
            b_T_table = self.psm.posquat2Matrix(pos_table, q_table)
            deltaObject_b_homogeneous = np.dot(b_T_table, np.array(
                [x, y, z, 0])) 

            # Adding z-translation and setting the final position
            pos_ee_projectedOnTable, _ = self._project_point_on_table(
                self.initial_pos)
            # dummy position in the /base 
            dummy_pos = pos_ee_projectedOnTable + deltaObject_b_homogeneous[0:3]

            # Checking there is no tt-rail crush
            if self.randomize_initial_pos_obj:
                dist_from_ee = np.linalg.norm(dummy_pos - pos_ee)
            else:
                dist_from_ee = 1

        return dummy_pos
    

    def _randomize_dummy_orientation(self):

        """
        This method is used to randomised the orientation of the dummy.
        The orientation is defined with the rotation around the x, y, z axes
        of the dummy. For each axe a reasonable range of rotation angle has been
        defined inside the configuration file for the task (pick.py)
        
        @details     The rotation is randomly generated chosing three random 
                     angles. The range of rotation are expressed in respect of 
                     the /rail. When setting the q_dummy the rotiation is instead
                     in the /base. Which is why the order of rotation does not 
                     correspond. 

        @param[out]  q_dummy   The orientation of the dummy in quaternions. 

        """

        # The rotational range are defined around the /dummy 
        x_range_l = self.x_range[0] 
        x_range_h = self.x_range[1] 
        x = self.np_random.randint(x_range_l, x_range_h)
        y_range_l = self.y_range[0] 
        y_range_h = self.y_range[1]
        y = self.np_random.randint(y_range_l, y_range_h)
        z_range_l = self.z_range[0]
        z_range_h = self.z_range[1]
        z = self.np_random.randint(z_range_l, z_range_h)

        # The orientation of the rail is defined in the /base which is why we 
        # follow order [x, z, y]
        rot_dummy = R.from_euler('yxz', [y, x, z], degrees=True)
        q_dummy = rot_dummy.as_quat()        
        return q_dummy


    def _print_dummy(self, grasp_site):
        """
        This method is called to print at screen the randomly selected dummy
        with the respective color. The colore have been chosen in coppelia scene.

        @param[in]   grasp_site   Varible of the selected grasping site
        """
        if grasp_site == 1:
            print('the dummy color YELLOW is number', grasp_site)
        elif grasp_site == 2:
            print('the dummy color RED is number', grasp_site)
        elif grasp_site == 3:
            print('the dummy color PINK is number', grasp_site)
        elif grasp_site == 4:
            print('the dummy color FUCSIA is number', grasp_site)
        elif grasp_site == 5:
            print('the dummy color GREEN is number', grasp_site)
        elif grasp_site == 6:
            print('the dummy color BLUE is number', grasp_site)
        elif grasp_site == 7:
            print('the dummy color BROWN is number', grasp_site)
        elif grasp_site == 8:
            print('the dummy color WHITE is number', grasp_site)

        return

    def _correct_rail_pos(self, q_dummy, grasp_site=8, safe_offset=None):
        """
        This method is used to check if the rail is always on a position that
        allows to reach the dummies. When the random orientation is set it could
        happen that the rail is in a position below the table surface and so the
        selected dummy is actually not reachable from the tt.
        In order to verify the rail is always above the table surface a check on
        the first and last dummy is done. If the distance between the dummy and
        the table surface is negative, means the dummy is below the surface so
        the dummy is projected on the table surface and that is used as new
        position and the entire rail is translated accordingly.
        The safe_offset is a safety margin that is add on the position to ensure
        the rail is always above the table surface.

        @details     We always work with respect of dummy position. So we getPose
                     and setPose are called they alway get as input/output the 
                     position of the respective dummy.

        @param[in]   q_dummy       Orientation quaternion of the rail.
                     grasp_site    We are considering only eight dummy
                     safe_offset   List with [x, y, z] safety offset in m to ensure
                                   the rail is above the table.
        @returns     Nothing.
        """
        
        safe_offset = [0, 0, 0.008]
        
        # Get the position of the dummy8 in the /base
        pos_dummy_eight, _ = self.rail.getPose(
            self.rail.dummy8_rail_handle, self.psm.base_handle)

        # Project dummy8 position on the table and get distance between dummy8 
        # and table surface
        pos_dummy_eight_on_table, distanceFromTable_eight = \
                self._project_point_on_table(pos_dummy_eight)

        # Check if the dummy8 is below the table
        if distanceFromTable_eight < 0:
            # Move dummy8 above the table
            pos_dummy_eight_above_table = pos_dummy_eight_on_table + safe_offset

            # Move the rail above the table
            # FIXME: i think i need to remove the handle here 
            _ , pos_rail_set  = self.rail.setPose(
                pos_dummy_eight_above_table, q_dummy, grasp_site, 
                self.psm.base_handle, ignoreError=True)
            #print('Position of the rail has been corrected')

            # FIXME: printing value to check 
            #print('dummy8 /base', pos_dummy_eight)
            #print('dummy8 /table', self._transf_base_to_table(pos_dummy_eight))
            #print('dummy8 on the table /base', pos_dummy_eight_on_table)
            #print('dummy8 on the table /table', self._transf_base_to_table(pos_dummy_eight_on_table))
            #print('Distance from the table', distanceFromTable_eight)
            #print('dummy8 above the table  /base', pos_dummy_eight_above_table)
            #print('dummy8 above the table /table', self._transf_base_to_table(pos_dummy_eight_above_table))
            #print('position of the rail corrected /base', pos_rail_set)
            #print('position of the rail corrected /table', self._transf_base_to_table(pos_rail_set))

        return  

    # Must be called immediately after _reset_sim since the goal is sampled
    # around the position of the EE

    def _sample_goal(self):
        self._simulator_step()

        # Setting as goal the grasping site of the rail. 
        rail_pos, _ = self.rail.getPose(self.dummy_rail_handle,
                                        self.psm.base_handle)
        # Overlapping the target to the dummy pose and adding a height-offset. 
        h_offset = 0.004  # liting the target of 3 mm compared to the dummy pos.
        new_pos = rail_pos
        new_pos[2] = new_pos[2] + h_offset 
        self.target.setPosition(new_pos, self.psm.base_handle, ignoreError=True)
        # Normalizing position
        rail_pos = (rail_pos - self.initial_pos) / self.target_range
        goal = rail_pos 

        return goal.copy()

    def _transf_base_to_table(self, point_base):

        """
        This methods compute the transformation from the /base into the /table. 
        
        @details    For the code the convention used for naming the transformation 
                    is the inverted one. So if the transformation is called
                    b_T_table  is the one that transform from /table to /base.

        @param[in]   point_base   Coordinate [x, y, z] of the 
                                  selected point in the /base 

        @param[out]  point_table  Coordinate of the input point in 
                                  the /table 
        """

        # Computing the transformation  
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)
        table_T_b = inv(b_T_table)
        
        # Transforming the point 
        point_base_h = np.append(point_base, 1)
        point_table = np.dot(table_T_b, point_base_h)
        point_table = point_table[:3]

        return point_table


    def _transf_table_to_base(self, point_table):

        """
        This methods compute the transformation from the /table into the /base. 
        
        @details    For the code the convention used for naming the transformation 
                    is the inverted one. So if the transformation is called
                    b_T_table  is the one that transform from /table to /base.

        @param[in]   point_table   Coordinate [x, y, z] of the 
                                   selected point in the /table 

        @param[out]  point_base    Coordinate of the input point in 
                                   the /base
        """

        #import pudb; pudb.set_trace()
        # Computing the transformation  
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)
        
        # Transforming the point 
        point_table_h = np.append(point_table, 1)
        point_base = np.dot(b_T_table, point_table_h)
        point_base = point_base[:3]


        return point_base

    def _is_success(self, achieved_goal, desired_goal):

        #d = goal_distance(achieved_goal, desired_goal) * \
        #    self.target_range  # Need to scale it back!

        #return (d < self.distance_threshold).astype(np.float32)
        d = goal_distance(achieved_goal, desired_goal)*self.target_range 

        grasp_success = self.rail.isGrasped(ignoreError=True, initialize=True)
        
        #print('grasp_success', grasp_success)
        #print('distance ', d)
        
        # Condition for success of the task: the robot has to be close to 
        # the target and have a successful grasping.
        if grasp_success and d < self.distance_threshold:
            success = 1.0 
        else:
            success = 0.0
        return success

    # Already accounts for height_offset!!!!
    def _project_point_on_table(self, point):
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

        normalVector_TableTop = b_T_table[0:3, 2]
        distanceFromTable = np.dot(normalVector_TableTop.transpose(
        ), (point - ((self.height_offset) * normalVector_TableTop + pos_table)))
        point_projected_on_table = point - distanceFromTable * normalVector_TableTop

        #import pudb; pudb.set_trace()
        return point_projected_on_table, distanceFromTable
