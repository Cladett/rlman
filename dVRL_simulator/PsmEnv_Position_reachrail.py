"""
@brief  Script used to control the main steps of the pick of the PAF rail
        The aim of the task is just to pick the rail.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   03 Sep 2020

"""
import numpy as np
from scipy.spatial.transform import Rotation as R
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
import time

# My imports
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, rail, target 
from dVRL_simulator.vrep.vrepObject import vrepObject


def goal_distance(goal_a, goal_b):
	assert goal_a.shape == goal_b.shape
	return np.linalg.norm(goal_a - goal_b, axis=-1)


class PSMEnv_Position_reachrail(PSMEnv):

    def __init__(self, psm_num, n_substeps, block_gripper,
            has_object, target_in_the_air, height_offset, target_offset, 
            obj_range, target_range, x_range, y_range, z_range,
            distance_threshold, initial_pos, 
            reward_type, dynamics_enabled, two_dimension_only,
            randomize_initial_pos_ee, 
            randomize_initial_pos_obj, randomize_initial_or_obj, 
            randomize_grasping_site, docker_container, action_type):

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
            randomize_grasping_site (boolean)
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
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.randomize_initial_pos_ee = randomize_initial_pos_ee
        self.randomize_initial_or_obj = randomize_initial_or_obj
        self.randomize_grasping_site = randomize_grasping_site
        self.action_type = action_type

        if self.block_gripper:
            self.n_actions = 3 
            self.n_states  = 3 + self.has_object*3 
        else:
            self.n_actions = 4
            self.n_states  = 4 + self.has_object*3

        super(PSMEnv_Position_reachrail, self).__init__(
            psm_num = psm_num, n_substeps=n_substeps, 
            n_states = self.n_states, n_goals = 3, 
            n_actions=self.n_actions, camera_enabled = False,
            docker_container =docker_container, action_type=action_type)

        self.vrepObject=vrepObject(self.clientID)

        # Checking if there is obj in the scene
        self.target = target(self.clientID, psm_num)
        if self.has_object:
            self.rail = rail(self.clientID)
        self.table = table(self.clientID)

        # Initialization
        self.prev_ee_pos  = np.zeros((3,))
        self.prev_ee_rot  = np.zeros((3,))
        self.prev_obj_pos = np.zeros((3,))
        self.prev_obj_rot = np.zeros((3,))
        self.prev_jaw_pos = 0
        
        # Selecting how many PSM are in the scene
        if(psm_num == 1):
            self.psm = self.psm1
        else:
            self.psm = self.psm2


        # Start the streaming from VREP for specific data:
        # PSM Arms:
        self.psm.getPoseAtEE(ignoreError = True, initialize = True)
        self.psm.getJawAngle(ignoreError = True, initialize = True)
        
        # TODO:
        # Used for _sample_goal
        # self.targetK.getPosition(
        #    self.psm.base_handle,
        #    ignoreError=True,
        #    initialize=True)

        # Used for _reset_sim
        self.table.getPose(
            self.psm.base_handle,
            ignoreError=True,
            initialize=True)

        # Initilization of the streaming of the dummies
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

        d = goal_distance(achieved_goal, goal)*self.target_range 
        #Need to scale it back!

        if self.reward_type == 'sparse':
            grasp_reward = self.rail.isGrasped(ignoreError=True,
                                                initialize=True)
            if grasp_reward:
                reward = 0.0
            else:
                reward = -1.0
            return reward
        else:
            return -100*d


    # PsmEnv methods
    # ----------------------------


    def _set_action(self, action):
        '''
        @details: method to set action constrains.
                  Constraining the action applied to the actual robot so that 
                  the variation in positions don't pass a threshold (0.001m).
                  Checking that the tooltip is not entering insde the table
                  surface.
                  Constraining the ee so that approaches the rail perpendicularly.
        '''
        assert action.shape == (self.n_actions,)
        action = action.copy()  # ensure that we don't change the action 
                                # outside of this scope

        if self.block_gripper:
            pos_ctrl = action[0:3]
            gripper_ctrl = 0
        else:
            pos_ctrl, gripper_ctrl = action[0:3], action[3]
            gripper_ctrl = (gripper_ctrl+1.0)/2.0 #gripper_ctrl bound to 0 and 1

        # Cheking if the rail object has any parents
        grasped = self.rail.isGrasped()
        
        # Get EE's pose:
        pos_ee, quat_ee = self.psm.getPoseAtEE()
        # Add position control:
        pos_ee = pos_ee + pos_ctrl*0.001 
        # eta = 1mm used to avoid overshoot on real robot

        # Get table information to constrain orientation and position:
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        # Make sure tool tip is not in the table by checking tt and which 
        # side of the table it is on.
        # DH parameters to find tt position:
        ct = np.cos(0)
        st = np.sin(0)

        ca = np.cos(-np.pi/2.0)
        sa = np.sin(-np.pi/2.0)

        T_x = np.array([[1,  0,  0, 0],
                       [0, ca, -sa, 0],
                       [0, sa,  ca, 0],
                       [0, 0, 0,    1]])
        T_z = np.array([[ct, -st, 0, 0],
                        [st,  ct, 0, 0],
                        [0,    0, 1, 0.0102],
                        [0,    0, 0, 1]])

        ee_T_tt = np.dot(T_x, T_z)

        pos_tt, quat_tt = self.psm.matrix2posquat(
                np.dot(self.psm.posquat2Matrix(pos_ee, quat_ee), ee_T_tt))

        pos_tt_on_table, distanceFromTable = self._project_point_on_table(
                pos_tt)

        # If the distance from the table is negative, then we need to 
        # project pos_tt onto the table top. Or if two dim only are enabled.
        if distanceFromTable < 0 or self.two_dimension_only:
            pos_ee, _ = self.psm.matrix2posquat(
                np.dot(
                    self.psm.posquat2Matrix(
                        pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))


        # Make sure the new pos doesn't go out of bounds!!!
        # Note: these are the bounds for the reachable space of the EE.
        upper_bound = self.initial_pos + self.target_range + 0.01
        lower_bound = self.initial_pos - self.target_range - 0.01

        pos_ee = np.clip(pos_ee, lower_bound, upper_bound)

        # For the approaching phase
        # q_target = self.targetK.getOrientationGoals(self.psm.base_handle)

        if not grasped:
            # Constrain ee so that approches rail perpendicularly.
            _, q_dummy = self.rail.getPose(
                self.dummy_rail_handle, self.psm.base_handle)
            # the position is computed related to the dummy. I had Rx(-90)
            temp_q = quaternions.qmult([q_dummy[3], q_dummy[0], q_dummy[1], 
                                        q_dummy[2]], [0.7, -0.7, 0, 0])  
            rot_ctrl = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])
            self.psm.setPoseAtEE(pos_ee, rot_ctrl, gripper_ctrl)
            self._simulator_step() # Fixing step bug

        return  


    # Mothod to allign the ee accordingly to the target
    def _align_to_target_orientation(self, k = 0.15, threshold = 8):
        
        # Get pose of target:
        q_target = self.targetK.getOrientationGoals(self.psm.base_handle)
        # Convert target quaternion to euler angles (radians):
        eul_target = self.vrepObject.quat2Euler(q_target)
        # Convert target euler angles to degrees:
        eul_target_deg = eul_target * (180/np.pi)

        # Get pose of EE:
        _, q_ee = self.psm.getPoseAtEE()
        # Convert EE quaternion to euler angles (radians)
        eul_ee = self.vrepObject.quat2Euler(q_ee)
        # Convert EE euler angles to degrees:
        eul_ee_deg = eul_ee * (180/np.pi) 

        # Get pose of the dummy of the rail 
        _, q_dummy, _, _, _, _ = self.rail.getPoseAchievedGoals(
                self.psm.base_handle)
        #Convert EE quaternion to euler angles (radians)
        eul_dummy = self.vrepObject.quat2Euler(q_dummy)
        #Convert EE euler angles to degrees:
        eul_dummy_deg = eul_dummy * (180/np.pi) 

        # Proportional control
        delta_rot_x = eul_target_deg[0] - eul_dummy_deg[0] -k*(
                      eul_target_deg[0] - eul_dummy_deg[0])
        delta_rot_y = eul_target_deg[1] - eul_dummy_deg[1]
        delta_rot_z = eul_target_deg[2] - eul_dummy_deg[2] -k*(
                      eul_target_deg[2]-eul_dummy_deg[2])

        # We want to slowly reach the target's orientation.
        # At each time-step, the EE is rotated by 10% the delta_rot at 
        # that time-step.
        rot_ctrl_x = delta_rot_x * 0.1
        rot_ctrl_y = delta_rot_y * 0.1
        rot_ctrl_z = delta_rot_z * 0.1

        # The new orientation for the EE is its previous + the change in 
        # orientation along each axis:
        new_eul_dummy_deg = np.array([eul_dummy_deg[0]+rot_ctrl_x, 
                                      eul_dummy_deg[1]+rot_ctrl_y, 
                                      eul_dummy_deg[2]+rot_ctrl_z])		
        # Converting to radians:
        new_eul_dummy = new_eul_dummy_deg*(np.pi/180)
        # Converting to quat:
        new_q_dummy = self.vrepObject.euler2Quat(new_eul_dummy)

        # Converting back to ee orientation: rotating of Rx-90
        temp_q = quaternions.qmult([new_q_dummy[3], new_q_dummy[0], 
                                    new_q_dummy[1], new_q_dummy[2]], 
                                    [0.7, -0.7, 0, 0])  
        new_q_ee = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

        done = False
        # If the orientation is almost the one of the target, stop adding the 
        # difference:
        norm_delta_rot = np.linalg.norm(np.array([delta_rot_x, delta_rot_y,
                                        delta_rot_z])) 
        #"almost" is quantified by the norm of the error vector
        if norm_delta_rot < threshold:
            done = True
        else:
            done = False

        return new_q_ee, done 


    def _get_obs(self):
        '''
        @details: This method generates the dict with the observation, 
                  achieved goal and desired goal. Desired goal is always the 
                  central dummy on the kidney. This dummy is one of the 5 
                  possible dummies and randomised. 

                  Achieved goal and observation change according to whether 
                  the rail is grasped or not.
                  If the rail is not grasped: the goal is to reach the grasping 
                  site of the rail and pick the rail.
                  If the rail is grasped: the goal is to reach the kidney and 
                  place the rail. 
        '''
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
                    2. The rail is set to a random intial position. Ranzomised 
                       in obj_range.
        """
        pos_ee = self._define_tt_pose()
        
        # If the object (usually the rail) is grasped from a previous 
        # simulation we release it
        if self.has_object:
            self.rail.removeGrasped(ignoreError=True)
        self._simulator_step()
        self.rail.removeGrasped(ignoreError=True) # to remove bug 
        
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

            if self.target_in_the_air:
                z = self.np_random.uniform(
                    0, self.target_range) + initial_tt_offset
            else:
                z = initial_tt_offset

            # Add target_offset for goal. And rotating from /table to /base
            deltaEEPos_b_homogeneous_one = np.append(
                self.np_random.uniform(-self.target_range, self.target_range,
                    size=2), [z, 0])
            deltaEEPos_b_homogeneous = np.dot(
                b_T_table, deltaEEPos_b_homogeneous_one)  # applying rotation

            # Project EE on to the table and add the deltaEEPos to that -
            # adding z-translation to the final coordinate position
            pos_ee_projectedOnTable, _ = self._project_point_on_table(
                self.initial_pos)
            pos_ee = pos_ee_projectedOnTable + deltaEEPos_b_homogeneous[0:3]

        else:
            pos_ee = self.initial_pos

        self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError=True)

        return pos_ee

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
            # Setting orientation if not randomize. 
            x_rot = 0   
            y_rot = 0   
            z_rot = 60 
            rot = R.from_euler('yxz', [y_rot, x_rot, z_rot], degrees=True)
            q_dummy = rot.as_quat()
            #q_dummy = ([0, 0, 0, 1])  # same orientation table surface if it 
                                      # is not randomized.

        # 3.Setting which dummy is gonna be grasped
        if self.randomize_grasping_site:
            grasp_site = self.np_random.randint(1, 9)
        else:
            grasp_site = 4 

        # I am giving as input the position of the selected dummy and i get 
        # as output the pos_rail_set the position of the rail in the /base 
        self.dummy_rail_handle, pos_rail_set = self.rail.setPose(
            dummy_pos, q_dummy, grasp_site, self.psm.base_handle, 
            ignoreError=True)  

        self._simulator_step()

        # 4. Correct rail position if below table surface
        self._correct_rail_pos(q_dummy)
        self._simulator_step()

        return

    def _set_dummy_pos(self, pos_ee, initial_rail_offset, dist_from_ee, 
                       minimum_d):
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
            _ , pos_rail_set  = self.rail.setPose(
                pos_dummy_eight_above_table, q_dummy, grasp_site, 
                self.psm.base_handle, ignoreError=True)

        return  

    # Must be called immediately after _reset_sim since the goal is sampled 
    # around the position of the EE
    # Defines kidney position in the workspace
    def _sample_goal(self):
        '''
        @datils: the target is placed over the grasping site of the rail that 
                 is selected as a goal.

        @Returns: goal.copy, the position of the central target on the cuboid.
        
        '''
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


    def _is_success(self, achieved_goal, desired_goal):

        # achieved_goal is the ee position 
        # desired_goal is the grasping site which correspong also target pos.
        d = goal_distance(achieved_goal, desired_goal)*self.target_range 

        # Considering this semplified
        #grasp_success = self.rail.isGrasped(ignoreError=True, initialize=True)

        # Condition for success of the task: the robot has to be close to 
        # the target and have a successful grasping.
        #if grasp_success and d < self.distance_threshold:
        #if d < self.distance_threshold:
        #    success = 1.0 
        #else:
        #    success = 0.0
        #return success
        return (d < self.distance_threshold).astype(np.float32)

    # Already accounts for height_offset!!!!
    def _project_point_on_table(self, point):
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

        normalVector_TableTop = b_T_table[0:3, 2]
        distanceFromTable = np.dot(normalVector_TableTop.transpose(), (point - ((self.height_offset)*normalVector_TableTop + pos_table)))
        point_projected_on_table = point - distanceFromTable*normalVector_TableTop

        return point_projected_on_table, distanceFromTable

    # Method to print to terminal the position of the objects
    def printpose(self, obj): 
        k = 0.15
        q_target = self.targetK.getOrientationGoals(self.psm.base_handle)
        eul_target = self.vrepObject.quat2Euler(q_target)
        eul_target_deg = eul_target * (180/np.pi)

        if obj == 'ee':
            pos_ee, quat_ee = self.psm.getPoseAtEE()
            #print('pos_ee respect RCM', pos_ee)
            print('q_ee respec RCM', quat_ee)
            eul_ee = self.vrepObject.quat2Euler(quat_ee)
            eul_ee_deg = eul_ee * (180/np.pi) # (-90,-20,0) (rxyz)
            print('orientation in degrees', eul_ee_deg)
            # Computing the distance for each angle with the kidney
            delta_rot_x = eul_target_deg[0] - eul_ee_deg[0] -k*(eul_target_deg[0] - eul_ee_deg[0])
            delta_rot_y = eul_target_deg[1] - eul_ee_deg[1]
            delta_rot_z = eul_target_deg[2] - eul_ee_deg[2] -k*(eul_target_deg[2]-eul_ee_deg[2])
            print('Angle distance for x-axe between kidney-ee', delta_rot_x)
            print('Angle distance for y-axe between kidney-ee', delta_rot_y)
            print('Angle distance for z-axe between kidney-ee', delta_rot_z)
        if obj == 'kidney':
            print('q_kidney respect RCM', q_target)
            print('orientation in degrees respect RCM', eul_target_deg)
        if obj == 'rail':
            _, q_dummy = self.rail.getPose(
                self.dummy_rail_handle, self.psm.base_handle)
            eul_dummy = self.vrepObject.quat2Euler(q_dummy)
            eul_dummy_deg = eul_dummy * (180/np.pi) 
            print('q_dummy_rail grasp respect RCM', q_dummy)
            print('orientation in degrees', eul_dummy_deg)
            delta_rot_x = eul_target_deg[0] - eul_dummy_deg[0] -k*(eul_target_deg[0] - eul_dummy_deg[0])
            delta_rot_y = eul_target_deg[1] - eul_dummy_deg[1]
            delta_rot_z = eul_target_deg[2] - eul_dummy_deg[2] -k*(eul_target_deg[2]-eul_dummy_deg[2])
            print('Angle distance for x-axe between kidney-dummy', 
                    delta_rot_x)
            print('Anle distance for y-axe between kidney-dummy', 
                    delta_rot_y)
            print('Angle distance for z-axe between kidney-dummy', 
                    delta_rot_z)
        if obj == 'rail_bottom':
            _, q_dummy_c, _, _, _, _ = self.rail.getPoseAchievedGoals(self.psm.base_handle)
            eul_dummy = self.vrepObject.quat2Euler(q_dummy_c)
            eul_dummy_deg = eul_dummy * (180/np.pi) 
            print('q_dummy_rail BOTTOM respec RCM', q_dummy_c)
            print('orientation in degrees BOTTOM respec RCM', eul_dummy_deg)
            delta_rot_x = eul_target_deg[0] - eul_dummy_deg[0] -k*(eul_target_deg[0] - eul_dummy_deg[0])
            delta_rot_y = eul_target_deg[1] - eul_dummy_deg[1]
            delta_rot_z = eul_target_deg[2] - eul_dummy_deg[2] -k*(eul_target_deg[2]-eul_dummy_deg[2])
            print('Angle distance for x-axe BOTTOM-kidneyBOTTOM-kidney', 
                    delta_rot_x)
            print('Angle distance for y-axe BOTTOM-kidneyBOTTOM-kidney', 
                    delta_rot_y)
            print('Angle distance for z-axe BOTTOM-kidneyBOTTOM-kidney', 
                    delta_rot_z)
        if obj == 'table':
            pos_table, q_table = self.table.getPose(self.psm.base_handle)
            eul_table = self.vrepObject.quat2Euler(q_table)
            eul_table_deg = eul_table * (180/np.pi) 
            print('q_table', q_table)
            print('orientation table in degrees', eul_table_deg)
        
        return 


