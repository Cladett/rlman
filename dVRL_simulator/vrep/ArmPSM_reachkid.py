# This can be used if the vrep connection is already made and clientID is passed
# V-REP must be running in synchronuous mode to use this.
# So the trigger to continue simulation must be done outside of this class
# To ensure data is sync'ed properly. Use the following two commands outside of this class:
#		vrep.simxSynchronousTrigger(clientID)
#		vrep.simxGetPingTime(cliendID)


# Make sure you initialize the "get" commands before using them. Otherwise inconsistent things will happen
# Ex: self.getJointAngles(ignoreError = True, initialize = True)


try:
	import dVRL_simulator.vrep.vrep as vrep
except:
	print ('--------------------------------------------------------------')
	print ('"vrep.py" could not be imported. This means very probably that')
	print ('either "vrep.py" or the remoteApi library could not be found.')
	print ('Make sure both are in the same folder as this file,')
	print ('or appropriately adjust the file "vrep.py"')
	print ('--------------------------------------------------------------')
	print ('')

import numpy as np
import transforms3d.quaternions as quaternions
import time

from dVRL_simulator.vrep.vrepObject import vrepObject

class ArmPSM(vrepObject):
	def __init__(self, clientID, armNumber = 1):

		super(ArmPSM, self).__init__(clientID)

		self.psm = armNumber
		self.ik_mode = 1;

		#Get all the handles we need
		self.base_handle = self.getHandle('RCM_PSM{}'.format(self.psm))
		self.j1_handle   = self.getHandle('J1_PSM{}'.format(self.psm))
		self.j2_handle   = self.getHandle('J2_PSM{}'.format(self.psm))
		self.j3_handle   = self.getHandle('J3_PSM{}'.format(self.psm))
		self.j4_handle   = self.getHandle('J1_TOOL{}'.format(self.psm))
		self.j5_handle   = self.getHandle('J2_TOOL{}'.format(self.psm))
		self.j6d_handle  = self.getHandle('J3_dx_TOOL{}'.format(self.psm))
		self.j6s_handle  = self.getHandle('J3_sx_TOOL{}'.format(self.psm))

		self.j5_dummy_handle   = self.getHandle('J2_virtual_TOOL{}'.format(self.psm))

		self.j6d_tip_dummy_handle   = self.getHandle('J3_dx_tip_TOOL{}'.format(self.psm))
		self.j6s_tip_dummy_handle   = self.getHandle('J3_sx_tip_TOOL{}'.format(self.psm))
		self.ik_tip_centre_handle   = self.getHandle('IK_tip_centre')

		self.ik_target_dx_dummy_handle = self.getHandle('IK_target_dx_PSM{}'.format(self.psm))
		self.ik_target_sx_dummy_handle = self.getHandle('IK_target_sx_PSM{}'.format(self.psm))
		self.ik_target_centre_dummy_handle = self.getHandle('IK_target_centre_PSM{}'.format(self.psm))

		self.EE_virtual_handle = self.getHandle('EE_virtual_TOOL{}'.format(self.psm))

		#Set IK mode off to save on computation for VREP:
		self.setIkMode(0, ignoreError = True)
		#Set dynamics mode off to save on compuation time for VREP:
		self.setDynamicsMode(0, ignoreError = True)

	#dyn_mode = 1 turns on dynamics
	#dyn_mode = 0 turns off dynamics
	def setDynamicsMode(self, dyn_mode, ignoreError = False):
		self.dyn_mode = dyn_mode
		self.setIntegerSignal("run_dyn_PSM{}".format(self.psm), self.dyn_mode, ignoreError)

	#ik_mode = 1 turns on ik_mode
	#ik_mode = 0 turns off ik_mode
	def setIkMode(self, ik_mode, ignoreError = False):
		self.ik_mode = ik_mode
		# res = vrep.simxSetIntegerSignal(self.clientID, "run_IK_PSM{}".format(self.psm), self.ik_mode, vrep.simx_opmode_oneshot)
		
		# if res!=vrep.simx_return_ok and not ignoreError:
		# 	print('Faled to set ik_mode')
		# 	print(res)
		self.setIntegerSignal("run_IK_PSM{}".format(self.psm), self.ik_mode, ignoreError)

	def getJawAngle(self, ignoreError = False, initialize = False):
		pos6s = self.getJointPosition(self.j6s_handle, ignoreError, initialize)
		pos6d = self.getJointPosition(self.j6d_handle, ignoreError, initialize)

		jawAngle = 0.5*(pos6d + pos6s)/0.4106

		return jawAngle

	def getJointAngles(self, ignoreError = False, initialize = False):
		pos1  = self.getJointPosition(self.j1_handle,  ignoreError, initialize)
		pos2  = self.getJointPosition(self.j2_handle,  ignoreError, initialize)
		pos3  = self.getJointPosition(self.j3_handle,  ignoreError, initialize)
		pos4  = self.getJointPosition(self.j4_handle,  ignoreError, initialize)
		pos5  = self.getJointPosition(self.j5_handle,  ignoreError, initialize)
		pos6s = self.getJointPosition(self.j6s_handle, ignoreError, initialize)
		pos6d = self.getJointPosition(self.j6d_handle, ignoreError, initialize)

		pos6     = 0.5*(pos6d - pos6s)
		jawAngle = 0.5*(pos6d + pos6s)/0.4106

		jointAngles = np.array([pos1, pos2, pos3, pos4, pos5, pos6])

		return jointAngles, jawAngle

	def getJointVelocities(self, ignoreError = False, initialize = False):
		vel1  = self.getJointVelocity(self.j1_handle,  ignoreError, initialize)
		vel2  = self.getJointVelocity(self.j2_handle,  ignoreError, initialize)
		vel3  = self.getJointVelocity(self.j3_handle,  ignoreError, initialize)
		vel4  = self.getJointVelocity(self.j4_handle,  ignoreError, initialize)
		vel5  = self.getJointVelocity(self.j5_handle,  ignoreError, initialize)
		vel6s = self.getJointVelocity(self.j6s_handle, ignoreError, initialize)
		vel6d = self.getJointVelocity(self.j6d_handle, ignoreError, initialize)

		vel6   = 0.5*(vel6s - vel6d)
		jawVel = 0.5*(vel6s + vel6d)/0.4106

		jointVelocities = np.array([vel1, vel2, vel3, vel4, vel5, vel6])

		return jointVelocities, jawVel

	def setJointAngles(self, jointAngles, jawAngle, ignoreError = False):
		if self.ik_mode == 1:
			self.setIkMode(0, ignoreError)

		self.setJointPosition(self.j1_handle, jointAngles[0], ignoreError)
		self.setJointPosition(self.j2_handle, jointAngles[1], ignoreError)
		self.setJointPosition(self.j3_handle, jointAngles[2], ignoreError)
		self.setJointPosition(self.j4_handle, jointAngles[3], ignoreError)
		self.setJointPosition(self.j5_handle, jointAngles[4], ignoreError)

		pos6s = 0.4106*jawAngle - jointAngles[5]
		pos6d = 0.4106*jawAngle + jointAngles[5]

		self.setJointPosition(self.j6s_handle, pos6s, ignoreError)
		self.setJointPosition(self.j6d_handle, pos6d, ignoreError)



	#Zero will get you pose of base frame in world frame
	#Rest of poses are from base frame
	def getPoseAtJoint(self, j, ignoreError = False, initialize = False):
		if j == 0:
			pos, quat = self.getPoseAtHandle(self.base_handle, -1, ignoreError, initialize )
		elif j == 1:
			pos, quat = self.getPoseAtHandle(self.j2_handle, self.base_handle, ignoreError, initialize)
			T = self.posquat2Matrix(pos,quat)
			rot90x = [[1, 0,  0, 0], 
					  [0, 0, -1, 0], 
					  [0, 1,  0, 0],
					  [0, 0,  0, 1]]
			pos, quat = self.matrix2posquat(np.dot(T, rot90x))
		elif j == 2:
			pos, quat = self.getPoseAtHandle(self.j3_handle, self.base_handle, ignoreError, initialize)
			T = self.posquat2Matrix(pos,quat)
			rot    = [[0,  0,  1, 0], 
					  [-1, 0,  0, 0], 
					  [0, -1,  0, 0],
					  [0,  0,  0, 1]]
			pos, quat = self.matrix2posquat(np.dot(T, rot))
		elif j == 3:
			pos, quat = self.getPoseAtHandle(self.j4_handle, self.base_handle, ignoreError, initialize)
			T = self.posquat2Matrix(pos,quat)
			rot    = [[-1, 0,  0, 0,], 
					  [0, -1,  0, 0], 
					  [0,  0,  1, 0],
					  [0,  0,  0, 1]]
			pos, quat = self.matrix2posquat(np.dot(T, rot))
		elif j == 4:
			pos, quat = self.getPoseAtHandle(self.j5_handle, self.base_handle, ignoreError, initialize)
			T = self.posquat2Matrix(pos,quat)
			rot    = [[0, 0, -1, 0], 
					  [1, 0,  0, 0], 
					  [0,-1,  0, 0],
					  [0, 0,  0, 1]]
			pos, quat = self.matrix2posquat(np.dot(T, rot))
		elif j == 5:
			pos, quat = self.getPoseAtHandle(self.j5_dummy_handle, self.base_handle, ignoreError, initialize)
		else:
			#Manually compute foreward kin from j5 to j6 or EE
			pos, quat = self.getPoseAtHandle(self.EE_virtual_handle, self.base_handle, ignoreError, initialize)			

			if j != 6:
				T = self.posquat2Matrix(pos,quat)

				#DH parameters for tool tip
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
				T = np.dot(np.dot(T,T_x), T_z)

				pos, quat = self.matrix2posquat(T)

		return np.array(pos), np.array(quat)


	def getPoseAtEE(self, ignoreError = False, initialize = False):
		return self.getPoseAtJoint(6, ignoreError, initialize)

	def getVelocityAtEE(self, ignoreError = False, initialize = False):
		return self.getVelocityAtHandle(self.EE_virtual_handle, ignoreError, initialize)


	#Set the EE pose using V-REP IK. The input pose in the base frame of the PSM arm.
	def setPoseAtEE(self, pos, quat, jawAngle, ignoreError = False):
		#Set pose at central IK target, which corresponds to EE dummy.		
		self.setPoseAtHandle(self.ik_target_centre_dummy_handle, self.base_handle, pos, quat, ignoreError)

		if self.ik_mode == 0:
			self.setIkMode(1, ignoreError)
