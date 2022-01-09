import numpy as np
from scipy.spatial.transform import Rotation as R

from dVRL_simulator.PsmEnv_reachkid import PSMEnv
from dVRL_simulator.vrep.simObjects_reachkid_neri import table, rail, targetK, target 
from dVRL_simulator.vrep.vrepObject import vrepObject

import transforms3d.euler as euler
import transforms3d.quaternions as quaternions


import time


def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class PSMEnv_Position_reachkid(PSMEnv):

    def __init__(self, psm_num, n_substeps, block_gripper, has_object, 
            target_in_the_air, height_offset, target_offset,
            obj_range, target_range, distance_threshold,
            initial_pos, reward_type,
            dynamics_enabled, two_dimension_only,
            randomize_initial_pos_obj,
            randomize_initial_pos_ee, docker_container,
            randomize_initial_pos_kidney, 
            randomize_initial_or_kidney, action_type):
        '''Initializes a new signle PSM Position Controlled Environment
            Args:
            psm_num (int): which psm you are using (1 or 2)
            n_substeps (int): number of substeps the simulation runs on every call to step
            gripper_extra_height (float): additional height above the table when positioning the gripper
            block_gripper (boolean): whether or not the gripper is blocked (i.e. not movable) or not
            has_object (boolean): whether or not the environment has an object
            target_in_the_air (boolean): whether or not the target should be in the air above the table or on the table surface
            height_offset (float): offset from the table for everything
            target_offset ( array with 3 elements): offset of the target, usually z is set to the height of the object
            obj_range (float): range of a uniform distribution for sampling initial object positions
            target_range (float): range of a uniform distribution for sampling a target Note: target_range must be set > obj_range
            distance_threshold (float): the threshold after which a goal is considered achieved
            initial_pos  (3x1 float array): The initial position for the PSM when reseting the environment.
            reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
            dynamics_enabled (boolean): To enable dynamics or not
            two_dimension_only (boolean): To only do table top or not. target_in_the_air must be set off too.
            docker_container (string): name of the docker container that loads the v-rep

            randomize_initial_pos_obj (boolean): whether you want to randomise the rail's initial pos or not
            randomize_initial_pos_kidney (boolean): whether you want to randomise the kidney's initial pos or not
            randomize_initial_or_kidney (boolean): whether you want to randomise the kidney's initial orientation or not
        '''

        # self.gripper_extra_height = gripper_extra_height
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
        self.randomize_initial_pos_ee = randomize_initial_pos_ee

        self.randomize_initial_pos_obj = randomize_initial_pos_obj      
        self.randomize_initial_or_kidney = randomize_initial_or_kidney
        self.randomize_initial_pos_kidney = randomize_initial_pos_kidney
        self.action_type = action_type

        if self.block_gripper:
            self.n_actions = 3
            self.n_states = 3 + self.has_object*3
        else:
            self.n_actions = 4
            self.n_states = 4 + self.has_object*3

        super(PSMEnv_Position_reachkid, self).__init__(
            psm_num = psm_num, n_substeps=n_substeps,
            n_states = self.n_states, n_goals = 3,
            n_actions=self.n_actions, camera_enabled = False,
            docker_container=docker_container, action_type=action_type)

        self.vrepObject=vrepObject(self.clientID)
        self.target = target(self.clientID, psm_num)
        self.targetK = targetK(self.clientID)

        #if self.has_object:
        self.rail = rail(self.clientID)
        self.table = table(self.clientID)

        if(psm_num == 1):
            self.psm = self.psm1
        else:
            self.psm = self.psm2

        # Start the streaming from VREP for specific data:

        # PSM Arms:
        self.psm.getPoseAtEE(ignoreError=True, initialize=True)
        self.psm.getJawAngle(ignoreError=True, initialize=True)

        # Used for _sample_goal
        self.target.getPosition(self.psm.base_handle,
                                ignoreError=True, initialize=True)
        self.targetK.getPosition(self.psm.base_handle, ignoreError = True, initialize = True)

        # Used for _reset_sim
        self.table.getPose(self.psm.base_handle, ignoreError=True, initialize=True)
        #if self.has_object:
        self.rail.getPose(self.rail.dummy_rail_handle, self.psm.base_handle, ignoreError=True, initialize=True)

        self.rail.getPositionAchievedGoals(self.psm.base_handle, ignoreError=True, initialize=True)

        # Used for _get_obs
        grasp = self.rail.isGrasped(ignoreError=True, initialize=True)

    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):

        d = goal_distance(achieved_goal, goal) * \
                          self.target_range  # Need to scale it back!

        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -100*d

    # PsmEnv methods
    # ----------------------------

    def _set_action(self, action#, new
            ):
        '''
        @Info: 
        Set the EE to a new position and orientation, closer to the target's ones, at each time-step.
        
        - 1st step: copy action from script/policy
        
        - 2nd step: if the LND is closed, then gripper_ctrl is 0. The action is simply position_ctrl.
        Else, the gripper_ctrl is the fourth element of the action vector.

        - 3rd step: pos_ctrl is actually multiplied by 1 mm, which is the eta factor discussed in dVRL paper.

        - 4th step: check if the toop-tip of the EE is below the table. If it is, raise it above the table.

        - 5th step: get a new quaternion for the EE, closer to the target's orientation. Set it to the new
        quaternion if the orientations are not yet close enough (threshold dictates this). Else, set the orientation
        equal to the target's. This is done because the error doesn't converge to 0, due to the instability
        of setting an orientation in V-Rep. 
        '''
        assert action.shape == (self.n_actions,)
        action = action.copy()  #ensure that we don't change the action outside of this scope

        #Gripper is considered blocked (and closed, grasping the rail) in reach training.
        if self.block_gripper:
            pos_ctrl = action[0:3]
            gripper_ctrl = 0
        else:
            pos_ctrl, gripper_ctrl = action[0:3], action[3]
            gripper_ctrl = (gripper_ctrl+1.0)/2.0 #gripper_ctrl bound to 0 and 1

        #Get EE's pose:
        pos_ee, quat_ee = self.psm.getPoseAtEE()
        #print("Old quaternion:", quat_ee)
        #Apply position control:
        pos_ee = pos_ee + pos_ctrl*0.001 #as the paper states, eta = 1mm used to avoid overshoot on real robot

        #Get table information to constrain orientation and position:
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        #Make sure tool tip is not in the table by checking tt and which side of the table it is on.
        #DH parameters to find tt position:
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

        pos_tt, quat_tt = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_ee, quat_ee), ee_T_tt))

        pos_tt_on_table, distanceFromTable = self._project_point_on_table(pos_tt)

        # If the distance from the table is negative, then we need to project pos_tt onto the table top.
        # Or if two dim only are enabled.
        if distanceFromTable < 0 or self.two_dimension_only:
            pos_ee, _ = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))

        #Make sure the new pos doesn't go out of bounds!!!
        #Note: these are the bounds for the reachable space of the EE.
        upper_bound = self.initial_pos + self.target_range + 0.01
        lower_bound = self.initial_pos - self.target_range - 0.01

        pos_ee = np.clip(pos_ee, lower_bound, upper_bound)

        #print("New before align", new)
        #Neri: compute orientation control
        #new_ee_quat, done, new = self._align_to_target(new)
        #print("New after align, should equal new degrees EE:", new)

        new_ee_quat, done = self._align_to_target()

        q_target = self.targetK.getOrientationGoals(self.psm.base_handle)

        if done: #if the EE (and rail) is oriented like the target, stop changing orientation
            #self.psm.setPoseAtEE(pos_ee, quat_ee, gripper_ctrl)
            self.psm.setPoseAtEE(pos_ee, q_target, gripper_ctrl)
            self._simulator_step() # fixing the step bug
        else:
            self.psm.setPoseAtEE(pos_ee, new_ee_quat, gripper_ctrl)                 
            self._simulator_step() # fixing the step bug

        #return done

    def _align_to_target(self, k = 0.15, threshold = 8 #, new
                ):
        '''
        @Info:
        This method is used by _set_action() to compute the error in orientation
        between the EE and the target. 
        The error is computed by subtracting components values one by one.

        - 1st step: we get the quaternions of the frames of target and EE. 
        @Note: the target's quaternion is not the cuboid's, but the quaternion of a dummy
        centered in the cuboid called "Kidney_orientation_ctrl". Why? This has the y axis downward,
        so I can compute the orientation error in a more uncomplicated way, just subtracting corresponding
        components.

        - 2nd step: from these quaternions we obtain angles in degrees so we can
        compute the error between the orientations.
        @Note: these two phases should be done by premultiplying the transposed rotation matrix
        of EE by the rotation matrix of target. This is the orientation error matrix between the two frames.
        From this, we obtain the angles in degrees (remember constraints on angles!).

        - 3rd step: we compute the errors and move the EE orientation by 10% of the error. 
        X and Z angles however also have a proportional factor "k" to increase stability. 
        This factor was chosen empirically.

        - 4th step: get the new quaternion of the EE and return it to _set_action() to set pose at EE.

        - 5th step: if the norm of the error vector is less than value "threshold" (empirically set), then
        we consider the orientation reached, done = True.

        @Returns: new_quaternion for EE, done signal.
        '''
        #Get pose of target:
        q_target = self.targetK.getOrientationGoals(self.psm.base_handle)
        #Convert target quaternion to euler angles (radians):
        eul_target = self.vrepObject.quat2Euler(q_target)
        #Convert target euler angles to degrees:
        eul_target_deg = eul_target*(180/np.pi)

        #Get pose of EE:
        _, q_ee = self.psm.getPoseAtEE()
        #Convert EE quaternion to euler angles (radians)
        eul_ee = self.vrepObject.quat2Euler(q_ee)
        #Convert EE euler angles to degrees:
        eul_ee_deg = eul_ee*(180/np.pi) # (-90,-20,0) (rxyz)

        #Sort of proportional control. Due to the instability of setting the orientation of the EE,
        #the quaternion we want the EE to go to is different from the quaternion set by V-Rep!
        #Therefore, errors sum up causing a non perfect alignment of the frames. 
        #To limit this, this parameter k was used to do some sort of proportional control. 
        #Not applied to error y because it caused more issues.
        delta_rot_x = eul_target_deg[0] - eul_ee_deg[0] -k*(eul_target_deg[0] - eul_ee_deg[0])
        delta_rot_y = eul_target_deg[1] - eul_ee_deg[1]
        delta_rot_z = eul_target_deg[2] - eul_ee_deg[2] -k*(eul_target_deg[2]-eul_ee_deg[2])

        #We want to slowly reach the target's orientation.
        #At each time-step, the EE is rotated by 10% the delta_rot at that time-step.
        rot_ctrl_x = delta_rot_x * 0.1
        rot_ctrl_y = delta_rot_y * 0.1
        rot_ctrl_z = delta_rot_z * 0.1

        #The new orientation for the EE is its previous + the change in orientation along each axis:
        new_eul_ee_deg = np.array([eul_ee_deg[0]+rot_ctrl_x, eul_ee_deg[1]+rot_ctrl_y, eul_ee_deg[2]+rot_ctrl_z])       
        #Back to radians:
        new_eul_ee = new_eul_ee_deg*(np.pi/180)
        #Back to quat:
        new_q_ee = self.vrepObject.euler2Quat(new_eul_ee)

        #NOTE: 
        #Print the new quaternion, which we'll use to set the pose of the EE, then check the "old" quaternion
        #at the next time-step. You'll see they are different, because the setPose method has numerical errors.

        done = False
        #If the orientation is almost the one of the target, stop adding the difference:
        norm_delta_rot = np.linalg.norm(np.array([delta_rot_x,delta_rot_y,delta_rot_z])) #"almost" is quantified by the norm of the error vector
        if norm_delta_rot < threshold:
            done = True
        else:
            done = False

        return new_q_ee, done #, new_eul_ee_deg

        #NOTE: the best would be not to read the EE quaternion to calculate the current error
        #at a given time-step, but employ the new_eul_ee_deg to compute the next error (use the desired, not actual orientation).
        #However, I couldn't understand how to make this in this script, because
        #at any time-step, these methods are executed top to bottom. So, how can I save its value 
        #and read it at the next time-step to compute the error with it? I tried one implementation but didn't work.
        #Variable "new" is part of this attempt.


    def _get_obs(self):
        '''
        @Info: 
        Define observation, achieved_goal, desired_goal.

        - Observation: achieved_goal_central is the central dummy below the rail.
          Achieved goal has to reach the target, which is a random dummy in the center of the cuboid's face. 
          Achieved goal isn't the rail's center of mass, otherwise the rail would enter the kidney to reach the target.
          Instead, it is below, so the rail simply is laid on the kidney without entering it.

        - Achieved goal: is achieved_goal_central
        - Desired goal: is random central dummy on the kidney

        @Returns: dict containing observation, achieved and desired goals.
        '''
        #Normalize EE position
        ee_pos, ee_quat = self.psm.getPoseAtEE()
        ee_pos = (ee_pos - self.initial_pos)/self.target_range

        jaw_pos = self.psm.getJawAngle()
        
        #-----------------------------------------------------------------------------------
        #OLD ACHIEVED GOAL AS RAIL'S COM
        #Normalize rail position
        #rail_pos,  _ = self.rail.getPose(self.rail.dummy_rail_handle, self.psm.base_handle)
        #rail_pos = (rail_pos - self.initial_pos)/self.target_range
        #achieved_goal = np.squeeze(rail_pos) 
        #-----------------------------------------------------------------------------------

        achieved_goal_central, _, _ = self.rail.getPositionAchievedGoals(self.psm.base_handle)
        achieved_goal_central = (achieved_goal_central - self.initial_pos)/self.target_range
        achieved_goal = np.squeeze(achieved_goal_central)

        obs = achieved_goal_central

        return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(), #central dummy below the rail
                'desired_goal': self.goal.copy() #central dummy sampled as desired_goal
        }

    def _reset_sim(self, initial_tt_offset=0.035):
        '''
        @Info:
        Reset the simulation with a random or initial pose for EE.
        Set the Rail in a grasped configuration, with its grasping site corresponding
        to the EE dummy. 
        '''
        # Get the constrained orientation of the ee
        _, q_table = self.table.getPose(self.psm.base_handle)

        temp_q = quaternions.qmult([q_table[3], q_table[0], q_table[1], q_table[2]], \
                                   [0.5, -0.5, -0.5,  0.5])
        ee_quat_constrained = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])

        # Put the EE in the correct orientation
        self.psm.setDynamicsMode(0, ignoreError=True)
        self._simulator_step()

        if self.randomize_initial_pos_ee:
            if self.target_in_the_air:
                z = self.np_random.uniform(0, self.obj_range) + initial_tt_offset
            else:
                z = initial_tt_offset

            # Add target_offset for goal.
            deltaEEPos = np.append(self.np_random.uniform(-self.obj_range, self.obj_range, size=2),[z])

            # Project EE on to the table and add the deltaEEPos to that
            pos_ee_projectedOnTable, _ = self._project_point_on_table(self.initial_pos)
            pos_ee = pos_ee_projectedOnTable + deltaEEPos

        else:
            pos_ee = self.initial_pos

        self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError=True)

        # if self.has_object:
        #   self.rail.isGrasped(ignoreError=True)
        # self._simulator_step
        if self.dynamics_enabled:
            self.psm.setDynamicsMode(1, ignoreError=True)

        self._simulator_step()

        #if self.has_object:
        self.rail.setPose(pos_ee, q_table, self.psm.base_handle, ignoreError=True)

        self._simulator_step()

        return True

    def _sample_goal(self, initial_pos_k = [0.05, -0.05, 0]):
        '''
        @Info: 
        If desired, computes a random position and orientation for the kidney's cuboid.
        Else, sets the cuboid in initial_pos_k and horizontal. 
        The position is randomised like this: 
        - x coordinate between (-2*obj_range, 2*obj_range)
        - y coordinate as x 
        - z coordinate is 0
        Then, add to this (x,y,z) target_offset. The value allows the kidney
        not to penetrate the table, whatever the orientations allowed. 
        Lastly, add pos_ee_projectedOnTable to lower the height.

        Orientation is sampled if desired through the called method randomize_k_orientation.
        
        Lastly, once the pose is set, a random set of targets is sampled.
        There are 5 triplets of dummies on the cuboid's surface. They are one central, one bottom, one top.
        The position of the central target is set as goal. The rail will have to be laid on the kidney
        so that a central dummy below it, called achieved_goal_central, reaches the central dummy on the cuboid.

        @Returns: goal.copy, the position of the central target on the cuboid.
        
        '''
        self._simulator_step()

        if self.target_in_the_air:
            z = 0
        else:
            z = 0

        randomize = self.randomize_initial_pos_kidney

        if randomize:

            random_kidney_pos = np.append(self.np_random.uniform(-2*self.obj_range, 2*self.obj_range, size=2), [z]) + self.target_offset

            #Project EE on to the table and add the deltaGoal to that
            pos_ee_projectedOnTable, _ = self._project_point_on_table(self.initial_pos)

            kidney_pos = pos_ee_projectedOnTable + random_kidney_pos

            if self.randomize_initial_or_kidney:
                rand_cuboid_quat = self.randomize_k_orientation()
            else:
                rand_cuboid_quat = [0, 0, 1, 0] 
            self.targetK.setPose(kidney_pos, rand_cuboid_quat, self.psm.base_handle, ignoreError=True)
            self._simulator_step()
        else: #if not randomize the kidney, set it to fixed position (initial_pos_k) and orientation (0,0,-10)
            #Project EE on to the table and add the deltaGoal to that
            pos_ee_projectedOnTable, _ = self._project_point_on_table(self.initial_pos)
            kidney_pos = np.array(initial_pos_k) + np.array(self.target_offset) + np.array(pos_ee_projectedOnTable)
            # These rot angles are ok to avoid collisions.
            x_rot = 0
            y_rot = 0
            z_rot = -10
            rot = R.from_euler('yxz', [y_rot, x_rot, z_rot], degrees=True)
            fixed_quat = rot.as_quat()

            self.targetK.setPose(kidney_pos, fixed_quat, self.psm.base_handle, ignoreError=True)
            self._simulator_step()

        #Once the pose is set, sample a target off the 5 sets. The goal is only the central dummy.
        #The top and bottom targets are not useful here. Later, in is_success, they will be used
        #to check if they correspond to the targets below the rail (orange dummies), achieved_goal_top and achieved_goal_bottom.
        goal = self.targetK.getPositionGoal(self.psm.base_handle)
        goal = (goal - self.initial_pos)/self.target_range
 
        return goal.copy()


    def randomize_k_orientation(self):
        '''
        @Info:
        Randomise the kidney's orientation.
        In kidney cuboid's frame: rotation ranges are defined by:
        - Pitch rotation, about the x axis between (-20,20)
        - Roll rotation, about the y axis between (-30,30)
        - Yaw rotation, about the z axis between (-89, 0) 

        @Note:
        The z angle is not between (-89,90) because of this:
        the rail has to reach the orientation of the kidney_orientation_ctrl dummy.
        Why: this dummy is oriented with y axis downward and x and z axes so that
        the difference between orientation components x, y, z can be computed in _align_to_target()
        easily, just subtracting corresponding components of angles of EE and kidney (target).
        However, due to the orientation of x and z, the rail is laid flat with the suction channel
        towards the opposite side of the kidney's adrenal gland. 
        Therefore, if you allow the kidney to have a yaw of 90° for example, the rail will have to 
        do a big rotation to lay itself on the kidney so that the suction channel is on the opposite
        side of the kidney's adrenal gland (btw, I don't care if it is towards or against).
        This big difference in rotation causes the gripper to lose the rail while trying to rotate that much.
        SO: I didn't have time to implement something like: if the difference in rotation is big
        lay the rail with the suction channel towards the adrenal gland. And decided to keep this angle between (-89,0).

        @Returns: a random quaternion.
        '''
        #The rotation ranges are defined around the /cuboid
        x = self.np_random.randint(-20, 20) #Pitch
        y = self.np_random.randint(-30, 30) #Roll
        z = self.np_random.randint(-89, 0) #Yaw

        #Random orientation in radians:
        rand_eul = np.array([x,y,z])*(np.pi/180)
        #Random orientation as quaternion:
        rand_quat = self.vrepObject.euler2Quat(rand_eul)

        return rand_quat
    

    def _is_success(self, achieved_goal, desired_goal):
        #Achieved goal is a central dummy below the rail.
        #Desired goal is a central dummy on the kidney's surface.
        #Compute the distance between central dummy below the rail and central dummy on the surface of the kidney:
        d = goal_distance(achieved_goal, desired_goal)*self.target_range #Need to scale it back! 

        #Get the positions of the dummies below the rail, top and bottom:
        _, achieved_goal_t, achieved_goal_b = self.rail.getPositionAchievedGoals(self.psm.base_handle, ignoreError=True, initialize=True)
        #Get the positions of the dummies on the kidney's surface, top and bottom
        desired_goal_t, desired_goal_b = self.targetK.getPositionGoalTopBottom(self.psm.base_handle, ignoreError=True, initialize=True)

        #Compute the distance between top dummy below the rail and top dummy on the surface of the kidney:
        d_top = goal_distance(achieved_goal_t, desired_goal_t)*self.target_range #Need to scale it back!
        #Compute the distance between bottom dummy below the rail and bottom dummy on the surface of the kidney:
        d_bottom = goal_distance(achieved_goal_b, desired_goal_b)*self.target_range #Need to scale it back!

        #Return 1 only if all the distances are below the threshold.
        return (d < self.distance_threshold).astype(np.float32)*(d_top < self.distance_threshold).astype(np.float32)* \
                                    (d_bottom <self.distance_threshold).astype(np.float32)
    
    
    def _project_point_on_table(self, point):
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table) 
        normalVector_TableTop = b_T_table[0:3, 2]
        distanceFromTable = np.dot(normalVector_TableTop.transpose(), (point - ((self.height_offset)*normalVector_TableTop + pos_table)))
        point_projected_on_table = point - distanceFromTable*normalVector_TableTop
    
        return point_projected_on_table, distanceFromTable      
