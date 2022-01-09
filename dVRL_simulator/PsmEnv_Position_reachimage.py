"""
@brief   Script used for the reach environment using as state info the images 
@author  Claudia D'Ettorre
@date    03 Aug 2021
"""
import numpy as np
import time

# My import
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, obj, target
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions
from scipy.spatial.transform import Rotation as R
from dVRL_simulator.vrep.vrepObject import vrepObject


def goal_distance(goal_a, goal_b):
        assert goal_a.shape == goal_b.shape
        return np.linalg.norm(goal_a - goal_b, axis=-1)
    

class PSMEnv_Position_reachimage(PSMEnv):

    def __init__(self, psm_num, n_substeps, block_gripper,
                has_object, target_in_the_air, height_offset, target_offset,
                obj_range, target_range, distance_threshold, initial_pos,
                reward_type, dynamics_enabled, two_dimension_only,
                randomize_initial_pos_obj, randomize_initial_pos_ee,
                docker_container, action_type):

        """Initializes a new single PSM Position Controlled Environment
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
            reward_type ('sparse1', 'sparse2', 'dense'): the reward type, i.e. sparse1, sparse2 (fully sparse) or dense
            dynamics_enabled (boolean): To enable dynamics or not
            two_dimension_only (boolean): To only do table top or not. target_in_the_air must be set off too.
            randomize_initial_pos_obj (boolean): If set true, it will randomize the initial position uniformly between 
                                                                        [-target_range + initial_pos, target_range + initial_pos] for x and y
                                                                        and [0+ initial_pos, initial_pos+ initial_pos] for z if target_in_air
            docker_container (string): name of the docke container that loads the v-rep
            action_type ('continuous', 'discrete'): the action space type, i.e. continuous or discrete

        """
        
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
        self.randomize_initial_pos_ee = randomize_initial_pos_ee
        self.corr_dir = False
        self.stay_still = False
        self.action_type = action_type
        
        if self.block_gripper:
            self.n_actions = 3 
            self.n_states  = 3 + self.has_object*3 
        else:
            self.n_actions = 4
            self.n_states  = 4 + self.has_object*3


        super(PSMEnv_Position_reachimage, self).__init__(
            psm_num = psm_num, n_substeps=n_substeps, n_states = self.n_states, 
            n_goals = 3, n_actions=self.n_actions, camera_enabled = True, 
            docker_container=docker_container, action_type=action_type)


        self.vrepObject=vrepObject(self.clientID)
        self.target = target(self.clientID, psm_num) 
        self.table = table(self.clientID)


        if(psm_num == 1):
            self.psm = self.psm1
        else:
            self.psm = self.psm2


        #--Start the streaming from VREP for specific data:

        #--PSM Arms:
        self.psm.getPoseAtEE(ignoreError = True, initialize = True)
        self.psm.getJawAngle(ignoreError = True, initialize = True)
        
        #--Used for _sample_goal
        self.target.getPosition(self.psm.base_handle, ignoreError = True, initialize = True)
        self.target.getQuaternion(self.psm.base_handle, ignoreError = True, initialize = True)

        #--Used for _reset_sim
        self.table.getPose(self.psm.base_handle, ignoreError = True, initialize = True)




    # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):

        d = goal_distance(achieved_goal, goal)*self.target_range         #Need to scale it back!

        if self.reward_type == 'sparse2':
            return -(d > self.distance_threshold).astype(np.float32)

        if self.reward_type == 'sparse1':
            if (d < self.distance_threshold).astype(np.float32) == 1.0:
                if self.stay_still == True: 
                    reward = 1.0
                else:
                    reward = 0.0
            elif self.corr_dir == True:
                reward = -0.5
                self.corr_dir = False			
            else:
                reward = -1.0
            return reward

        else:
            return -100*d


    # PsmEnv methods
    # ----------------------------

    def _set_action(self, action):		
            
        if self.action_type == 'discrete':		
            pos_ee, _ = self.psm.getPoseAtEE()			
            pos_target = self.target.getPosition(self.psm.base_handle)
            self.stay_still = False

            if action == 0:                         # if the action is "GO LEFT"               
                pos_ee[0] += 0.003              #0.3cm
                if pos_target[0]>pos_ee[0]:
                    self.corr_dir = True                           

            if action == 1:                         # if the action is "GO RIGHT"            
                pos_ee[0] -= 0.003
                if pos_target[0]<pos_ee[0]:
                    self.corr_dir = True                            

            if action == 2:                         # if the action is "GO UP" 
                pos_ee[1] -= 0.003
                if pos_target[1]<pos_ee[1]:
                    self.corr_dir = True                         

            if action == 3:                         # if the action is "GO DOWN"             
                pos_ee[1] += 0.003
                if pos_target[1]>pos_ee[1]:
                    self.corr_dir = True  

            if action == 4:                         # if the action is "STAY STILL"         
                self.stay_still = True  	

            if action == 5:                         #if the action is "GO ABOVE image plane"
                pos_ee[2] += 0.003
                if pos_target[2]>pos_ee[2]:
                    self.corr_dir = True

            if action == 6:                         #if the action is "GO BELOW image plane"            
                pos_ee[2] -= 0.003
                if pos_target[2]<pos_ee[2]:
                    self.corr_dir = True
        else:
            assert action.shape == (self.n_actions,)
            action = action.copy()  # ensure that we don't change the action outside of this scope

            if self.block_gripper:
                pos_ctrl = action
            else:
                pos_ctrl, gripper_ctrl = action[:3], action[3]
                gripper_ctrl = (gripper_ctrl+1.0)/2.0

            pos_ee, quat_ee = self.psm.getPoseAtEE()		
            pos_ee = pos_ee + pos_ctrl*0.003  # the maximum change in position is 0.3cm			


            #Get table information to constrain orientation and position
            pos_table, q_table = self.table.getPose(self.psm.base_handle)			


            #Make sure tool tip is not in the table by checking tt and which side of the table it is on

            #DH parameters to find tt position
            ct = np.cos(0)
            st = np.sin(0)

            ca = np.cos(-np.pi/2.0)
            sa = np.sin(-np.pi/2.0)

            T_x = np.array([[1,  0,  0, 0],          
                            [0, ca, -sa, 0 ],
                            [0, sa,  ca, 0 ],
                            [0, 0, 0,    1 ]])
            T_z = np.array([[ct, -st, 0, 0],         
                            [st,  ct, 0, 0],
                            [0,    0, 1, 0.0102],  
                            [0,    0, 0, 1]])

            ee_T_tt = np.dot(T_x, T_z) 

            pos_tt, quat_tt = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_ee,quat_ee), ee_T_tt))

            pos_tt_on_table, distanceFromTable = self._project_point_on_table(pos_tt)

            #if the distance from the table is negative, then we need to project pos_tt onto the table top.
            #Or if two dim only are enabled
            if distanceFromTable < 0 or self.two_dimension_only:
                pos_ee, _ = self.psm.matrix2posquat(np.dot(self.psm.posquat2Matrix(pos_tt_on_table, quat_tt), np.linalg.inv(ee_T_tt)))


        if self.block_gripper:
            gripper_ctrl = 0

        ##--Make sure the new pos doesn't go out of bounds!!!
        upper_bound = self.initial_pos+self.target_range +0.01
        z_upper_bound = self.initial_pos[2]+self.target_range
        upper_bound = np.array([upper_bound[0], upper_bound[1], z_upper_bound])

        lower_bound = self.initial_pos-self.target_range -0.01
        z_lower_bound = self.initial_pos[2]-self.target_range
        lower_bound = np.array([lower_bound[0], lower_bound[1], z_lower_bound])


        pos_ee = np.clip(pos_ee, lower_bound, upper_bound) #Given an interval, values outside the interval are clipped to the interval edges

        new_ee_quat, done = self._align_to_target()
        pos_target = self.target.getPosition(self.psm.base_handle)
        q_target = self.target.getQuaternion(self.psm.base_handle)
        
        if done:
            self.psm.setPoseAtEE(pos_ee, q_target, gripper_ctrl)
        else:
            self.psm.setPoseAtEE(pos_ee, new_ee_quat, gripper_ctrl)
            

    def _align_to_target(self, threshold = 15):
        '''
        @Info:
        Function originally implemented by N.N. Dei in https://github.com/nndei/dVRL_Neri

        This method is used by _set_action() to compute the error in orientation
        between the EE and the target. 
        The error is computed by subtracting components values one by one.
        - 1st step: we get the quaternions of the frames of target and EE. 
        - 2nd step: from these quaternions we obtain angles in degrees so we can
        compute the error between the orientations.
        @Note: these two phases should be done by premultiplying the transposed rotation matrix
        of EE by the rotation matrix of target. This is the orientation error matrix between the two frames.
        From this, we obtain the angles in degrees (remember constraints on angles!).
        - 3rd step: we compute the errors and move the EE orientation by 40% of the error (value chosen empirically considering the small volume where target is reset and arm moves). 
        - 4th step: get the new quaternion of the EE and return it to _set_action() to set pose at EE.
        - 5th step: if the norm of the error vector is less than value "threshold" (empirically set), then
        we consider the orientation reached, done = True.
        @Returns: new_quaternion for EE, done signal.
        '''
        #Get pose of target:
        q_target = self.target.getQuaternion(self.psm.base_handle)
        #Convert target quaternion to euler angles (radians):
        eul_target = self.vrepObject.quat2Euler(q_target)
        #Convert target euler angles to degrees:
        eul_target_deg = eul_target*(180/np.pi)
        

        #Get pose of EE:
        _, q_ee = self.psm.getPoseAtEE()
        #Convert EE quaternion to euler angles (radians)
        eul_ee = self.vrepObject.quat2Euler(q_ee)
        #Convert EE euler angles to degrees:
        eul_ee_deg = eul_ee*(180/np.pi)


        
        delta_rot_x = eul_target_deg[0] - eul_ee_deg[0]
        delta_rot_y = eul_target_deg[1] - eul_ee_deg[1]
        delta_rot_z = eul_target_deg[2] - eul_ee_deg[2]


        #We want to slowly reach the target's orientation.
        #At each time-step, the EE is rotated by 40% the delta_rot at that time-step.
        rot_ctrl_x = delta_rot_x * 0.40 
        rot_ctrl_y = delta_rot_y * 0.40 
        rot_ctrl_z = delta_rot_z * 0.40
        


        #The new orientation for the EE is its previous + the change in orientation along each axis:
        new_eul_ee_deg = np.array([eul_ee_deg[0]+rot_ctrl_x, eul_ee_deg[1]+rot_ctrl_y, eul_ee_deg[2]+rot_ctrl_z])		
        #Back to radians:
        new_eul_ee = new_eul_ee_deg*(np.pi/180)
        #Back to quat:
        new_q_ee = self.vrepObject.euler2Quat(new_eul_ee)


        done = False
        #If the orientation is almost the one of the target, stop adding the difference:
        norm_delta_rot = np.linalg.norm(np.array([delta_rot_x,delta_rot_y,delta_rot_z])) #"almost" is quantified by the norm of the error vector

        if norm_delta_rot < threshold:
            done = True
        else:
            done = False

        return new_q_ee, done 


    def _get_obs(self):
        #Normalize ee_position:
        ee_pos,  ee_quat = self.psm.getPoseAtEE()
        ee_pos = (ee_pos - self.initial_pos)/self.target_range

        jaw_pos = self.psm.getJawAngle()

        obj_pos = np.zeros((3,))
        achieved_goal = np.squeeze(ee_pos) 

        if self.block_gripper:
            obs = ee_pos
        else:
            obs = np.concatenate((ee_pos, np.array([jaw_pos]))) 		


        return {
                'observation': obs.copy(),
                'achieved_goal': achieved_goal.copy(),
                'desired_goal' : (self.goal.copy()-self.initial_pos)/self.target_range
        }


    def _reset_sim(self):
        #Get the constrained orientation of the ee
        #import pudb; pudb.set_trace()

        pos_table, q_table = self.table.getPose(self.psm.base_handle)

        temp_q =  quaternions.qmult([q_table[3], q_table[0], q_table[1], q_table[2]], [ 0.5, -0.5, -0.5,  0.5])
        ee_quat_constrained = np.array([temp_q[1], temp_q[2], temp_q[3], temp_q[0]])


        #Put the EE in the correct orientation
        self.psm.setDynamicsMode(0, ignoreError = True)
        self._simulator_step()

        if self.randomize_initial_pos_ee: 
            if self.target_in_the_air:
                z = self.np_random.uniform(-self.target_range, self.target_range)
            else:
                z = -0.1435

            y = self.np_random.uniform(-self.target_range, self.target_range)
            x = self.np_random.uniform(-self.target_range, self.target_range)
            deltaEEPos = np.array([x, y, z])	

            pos_ee_projectedOnTable, distanceFromTable = self._project_point_on_table(self.initial_pos)
            if distanceFromTable < 0 or self.two_dimension_only:		
                pos_ee = pos_ee_projectedOnTable + deltaEEPos

            pos_ee = self.initial_pos + deltaEEPos 
            
        else:
            pos_ee = self.initial_pos

        self.psm.setPoseAtEE(pos_ee, ee_quat_constrained, 0, ignoreError = True)
        self._simulator_step()


        if self.dynamics_enabled:
            self.psm.setDynamicsMode(1, ignoreError = True)

        
        self._simulator_step()

        return True


    #--Must be called immediately after _reset_sim since the goal is sampled around the position of the EE
    def _sample_goal(self):
        self._simulator_step()

        #--Get a random x,y,z vector to offset from the EE from the goals initial position.
        #--	x,y plane is parallel to table top and is from -target_range to target_range
        #--	z is perpindicular to the table top and 
        #--		if target_in_the_air from from -target_range to target_range 
        #--		else -0.1435
        pos_table, q_table = self.table.getPose(self.psm.base_handle)

        if self.target_in_the_air:
            z = self.np_random.uniform(-self.target_range, self.target_range) 
        else:
            z = -0.1435		

        y = self.np_random.uniform(-self.target_range, self.target_range)
        x = self.np_random.uniform(-self.target_range, self.target_range)
        deltaGoal = np.array([x, y, z])


        pos_ee_projectedOnTable, distanceFromTable = self._project_point_on_table(self.initial_pos)
        if distanceFromTable < 0 or self.two_dimension_only:		
            goal = pos_ee_projectedOnTable + deltaGoal
                                                
        goal = self.initial_pos + deltaGoal 
        
        self.target.setPosition(goal, self.psm.base_handle, ignoreError = True)
        self._simulator_step()

        goal = self.target.getPosition(self.psm.base_handle) 
        return goal.copy()


    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)*self.target_range #Need to scale it back!
        return (d<self.distance_threshold).astype(np.float32)


    #Already accounts for height_offset!!!!
    def _project_point_on_table(self, point):
        pos_table, q_table = self.table.getPose(self.psm.base_handle)
        b_T_table = self.psm.posquat2Matrix(pos_table, q_table)

        normalVector_TableTop = b_T_table[0:3, 2]
        distanceFromTable = np.dot(normalVector_TableTop.transpose(), (point - ((self.height_offset)*normalVector_TableTop + pos_table)))
        point_projected_on_table = point - distanceFromTable*normalVector_TableTop

        return point_projected_on_table, distanceFromTable
