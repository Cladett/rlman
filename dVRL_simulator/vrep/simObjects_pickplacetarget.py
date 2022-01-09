"""
@brief  This script defines all the class for the objects decleared inside the 
        vrep simulation environment.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date   05 Oct 2020
"""

from dVRL_simulator.vrep.vrepObject import vrepObject
import numpy as np
from numpy.linalg import inv

# Camera: connected to a vision sensor. Used to get frames from the environment
class camera(vrepObject):
        def __init__(self, clientID, rgb=True):
                super(camera, self).__init__(clientID)
                self.camera_handle = self.getHandle('Vision_Sensor')
                self.rgb = rgb

                self.getVisionSensorImage(self.camera_handle, self.rgb, 
                                          ignoreError=True, initialize=True)

        def getImage(self, ignoreError=False):
                data, resolution = self.getVisionSensorImage(self.camera_handle, 
                                    self.rgb, ignoreError=ignoreError,
                                    initialize=False)

                if self.rgb:
                        #return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])
                        return np.flipud(np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3]))
                else:
                        #return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]])
                        return np.flipud(np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]]))

# Table: surface where everything is sitting.
class table(vrepObject):
        def __init__(self, clientID):
                super(table, self).__init__(clientID)
                self.table_top_handle = self.getHandle('customizableTable_tableTop')

        def getPose(self, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(self.table_top_handle, relative_handle, ignoreError, initialize)

# Object: it's a small cilinder. It has a dummy along the z-axe at +0.001m.
# The object position is dected thank to the dummy
class obj(vrepObject):
        def __init__(self, clientID):
                super(obj, self).__init__(clientID)

                self.obj_handle = self.getHandle('Object')
                self.dummy_handle = self.getHandle('Object_Dummy')
                
                #Claudia: initializing value for reading proximity sensor
                self.proximity_handle= self.getHandle('TOOL1_proxSensor')

                self.readProximitySensor(self.proximity_handle, ignoreError=True, initialize=True)

        def setPose(self, pos, quat, relative_handle, ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                d_T_o = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.001], [0, 0, 0, 1]])
                pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_o))

                self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

        def getPose(self, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.obj_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.obj_handle, ignoreError, initialize))
        
        #Claudia: add the possibility to read from proximity sensor 
        def readProximity(self, ignoreError=False):
                success_detection, det_point, det_handle, 
                distance_norm = self.readProximitySensor(self.proximity_handle, 
                                ignoreError=ignoreError, initialize=True)
                return  success_detection, det_point, det_handle, distance_norm 

# Adding the obstacles
class obs1(vrepObject):
        def __init__(self, clientID):
                super(obs1, self).__init__(clientID)

                self.obs1_handle = self.getHandle('obstacle1')
                #self.dummy_handle = self.getHandle('obstacle1_dunny')
                
        def setPose(self, pos, quat, relative_handle, ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                d_T_o = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.001], [0, 0, 0, 1]])
                pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_o))

                self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

        def getPose(self, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.obj_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.obj_handle, ignoreError, initialize))
        
class obs2(vrepObject):
        def __init__(self, clientID):
                super(obs2, self).__init__(clientID)

                self.obs2_handle = self.getHandle('obstacle2')
                #self.dummy_handle = self.getHandle('obstacle1_dunny')
                
        def setPose(self, pos, quat, relative_handle, ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                d_T_o = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.001], [0, 0, 0, 1]])
                pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_o))

                self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

        def getPose(self, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.obj_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.obj_handle, ignoreError, initialize))
            
class obs3(vrepObject):
        def __init__(self, clientID):
                super(obs3, self).__init__(clientID)

                self.obs3_handle = self.getHandle('obstacle3')
                #self.dummy_handle = self.getHandle('obstacle1_dunny')
                
        def setPose(self, pos, quat, relative_handle, ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                d_T_o = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.001], [0, 0, 0, 1]])
                pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_o))

                self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

        def getPose(self, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.obj_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.obj_handle, ignoreError, initialize))

# Target: represent by a red ball in the simulation. Used as target for the 
# reach task and pick task.
class target(vrepObject):
        def __init__(self, clientID, psm_number):
                super(target, self).__init__(clientID)

                self.target_handle = self.getHandle('Target_PSM{}'.format(psm_number))

                self.getPosition(-1, ignoreError=True, initialize=True)

        def setPosition(self, pos, relative_handle, ignoreError=False):
                self.setPoseAtHandle(self.target_handle, relative_handle, 
                                     pos, [1, 0, 0, 1], ignoreError)

        def getPosition(self, relative_handle, ignoreError=False, initialize=False):
                pos, _ = self.getPoseAtHandle(self.target_handle, 
                              relative_handle, ignoreError, initialize)
                return pos

        def getQuaternion(self, relative_handle, ignoreError = False, initialize = False):
                _, quat = self.getPoseAtHandle(self.target_handle, relative_handle, ignoreError, initialize)
                return quat

# Rail: Pneumatic Attachable Flexible (PAF) rail. It is characterised by 8 dummy.
# The Dummy is located for now with only a traslation on z-axe 
# (longer axe of the rail) of +0.010 m and a rotation of 90 along 
# x-axe parent frame.
# The rail is used to execute pick and pick and place task. 
# We are using the opposite convention of rotation (see notes)
class rail(vrepObject):
        def __init__(self, clientID):
                super(rail, self).__init__(clientID)

                # Respondable rail 
                # self.rail_handle = self.getHandle('rail_respondable')
                # Realistic rail
                # self.rail_real_handle = self.getHandle('rail')
                # To use when using only mesh of the rail
                self.rail_handle = self.getHandle('rail')
                #if using single dummy
                #self.dummy_rail_handle = self.getHandle('rail_Dummy') 

                # Dummies for the grasping site definition
                self.dummy1_rail_handle = self.getHandle('rail_Dummy1')
                self.dummy2_rail_handle = self.getHandle('rail_Dummy2') 
                self.dummy3_rail_handle = self.getHandle('rail_Dummy3') 
                self.dummy4_rail_handle = self.getHandle('rail_Dummy4') 
                self.dummy5_rail_handle = self.getHandle('rail_Dummy5') 
                self.dummy6_rail_handle = self.getHandle('rail_Dummy6') 
                self.dummy7_rail_handle = self.getHandle('rail_Dummy7') 
                self.dummy8_rail_handle = self.getHandle('rail_Dummy8') 

                # Definition of three dummies on the bottom of the rail.
                # One is central, one is bottom and one is top 
                # (towards adrenal gland of kidney)
                #self.rail_achieved_top = self.getHandle('rail_Achieved_t')
                #self.rail_achieved_bottom = self.getHandle('rail_Achieved_b')
                #self.rail_achieved_central = self.getHandle('rail_Achieved_goal')

                # Initializing value to read proximity sensor
                self.proximity_handle= self.getHandle('TOOL1_proxSensor')
                self.readProximitySensor(self.proximity_handle, ignoreError=True, 
                                         initialize=True)

        def setPose(self, pos, quat, grasp_site, relative_handle, 
                    ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                pos_forced=pos
                
                # Adding randomize choice of the grasping site (MULTIPLE dummies) 
                if grasp_site == 1:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0325], 
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy1
                    #d_T_r = np.array([[0, 0, 1,0.0325 ], [-1, 0, 0, 0], 
                    #                  [0, -1, 0, -0.004], [0, 0,0, 1]]) 
                    self.dummy_rail_handle = self.dummy1_rail_handle
                elif grasp_site == 2:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0225],
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy2
                    #d_T_r = np.array([[0, 0, 1, 0.0225 ], [-1, 0, 0, 0], 
                    #                  [0, -1, 0, -0.004], [0, 0, 0, 1]]) 
                    self.dummy_rail_handle = self.dummy2_rail_handle
                elif grasp_site == 3:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0125],
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy3
                    #d_T_r = np.array([[0, 0, 1, 0.0125 ], [-1, 0, 0, 0], 
                    #                  [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy3_rail_handle
                elif grasp_site == 4:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, 0.0025], 
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy4
                    #d_T_r = np.array([[0, 0, 1, 0.0025 ], [-1, 0, 0, 0], 
                    #                [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy4_rail_handle
                elif grasp_site == 5:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, -0.0075],
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy5
                    #d_T_r = np.array([[0, 0, 1, 0.0075 ], [-1, 0, 0, 0],
                    #                [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy5_rail_handle
                elif grasp_site == 6:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, -0.0175], 
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy6
                    #d_T_r = np.array([[0, 0, 1, -0.0175 ], [-1, 0, 0, 0],
                    #                  [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy6_rail_handle
                elif grasp_site == 7:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, -0.0275], 
                                      [0, -1, 0, -0.004], [0, 0,0, 1]]) #Coordinate Dummy7
                    #d_T_r = np.array([[0, 0, 1, -0.0275 ], [-1, 0, 0, 0],
                    #                  [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy7_rail_handle
                elif grasp_site == 8:
                    d_T_r = np.array([[1, 0, 0, 0], [0, 0, 1, -0.0375], 
                                     [0, -1, 0, -0.004], [0, 0, 0, 1]]) #Coordinate Dummy8
                    #d_T_r= inv(d_T_r)
                    #d_T_r = np.array([[0, 0, 1, -0.0375 ], [-1, 0, 0, 0], 
                    #                   [0, -1, 0, -0.004], [0, 0, 0, 1]])
                    self.dummy_rail_handle = self.dummy8_rail_handle

                
                pos, quat = self.matrix2posquat(np.dot(b_T_d, d_T_r))
                
                #b_T_table = np.array([
                #   [ 3.41779961e-01, 9.39780005e-01, -8.02261448e-06,  1.89346313e-01],
                #   [-9.39780005e-01, 3.41779961e-01,  5.97869184e-07,  2.79672146e-02],
                #   [ 3.30383437e-06, 7.33515297e-06,  1.00000000e+00, -1.23435020e-01],
                #   [ 0,  0,  0, 1]
                #])

                self.setPoseAtHandle(self.rail_handle, relative_handle, 
                        pos, quat, ignoreError) # relative handle: psm.base_handle
                return self.dummy_rail_handle , pos

        #def getPoseAchievedGoals(self, relative_handle, ignoreError=False, 
                #                     initialize=False):
		#Get position of dummies below the rail: central, top and bottom.
                #pos_achieved_central, q_central = self.getPoseAtHandle(
                #        self.rail_achieved_central, relative_handle, 
                #        ignoreError, initialize)
                #pos_achieved_top, q_top = self.getPoseAtHandle(
                #        self.rail_achieved_top, relative_handle, 
                #        ignoreError, initialize)
                #pos_achieved_bottom, q_bottom = self.getPoseAtHandle(
                #        self.rail_achieved_bottom, relative_handle, 
                #        ignoreError, initialize)

        #        return pos_achieved_central, q_central, pos_achieved_top, q_top, pos_achieved_bottom, q_bottom 

        # Gives the pose of the dummy chosen 
        def getPose(self, dummy_rail_handle, relative_handle, ignoreError=False, 
                    initialize=False):
                return self.getPoseAtHandle(dummy_rail_handle, relative_handle, 
                                            ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_rail_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.rail_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.rail_handle, ignoreError, initialize))
       
        # Adding function to read the proximity sensor
        def readProximity(self, ignoreError=False):
                success_detection, det_point, det_handle, distance_norm = self.readProximitySensor(self.proximity_handle, 
                                     ignoreError=ignoreError, initialize=True)
                return  success_detection, det_point, det_handle, distance_norm 

        # TargetK: definition of the target kidney
class targetK(vrepObject):
        def __init__(self, clientID):
                super(targetK, self).__init__(clientID)
                
                # Step 1: get the handles of the objects in the scene.
                # Parent
                self.k_res_handle = self.getHandle('Kidney_respondable')

                # Realistic Kidney
                self.k_handle = self.getHandle('Kidney')

                # Shape used for collision check
                self.convex = self.getHandle('Convex')

                # Surface dummies handles:
                # dh = dummy handle,
                # t = top (towards positive axis and adrenal gland), b = bottom
                self.k_dh_0t = self.getHandle('Kidney_Dummy_0t')
                self.k_dh_0b = self.getHandle('Kidney_Dummy_0b')
                self.k_dh_1t = self.getHandle('Kidney_Dummy_1t')
                self.k_dh_1b = self.getHandle('Kidney_Dummy_1b')
                self.k_dh_2t = self.getHandle('Kidney_Dummy_2t')
                self.k_dh_2b = self.getHandle('Kidney_Dummy_2b')
                self.k_dh_3t = self.getHandle('Kidney_Dummy_3t')
                self.k_dh_3b = self.getHandle('Kidney_Dummy_3b')
                self.k_dh_4t = self.getHandle('Kidney_Dummy_4t')
                self.k_dh_4b = self.getHandle('Kidney_Dummy_4b')
                #Dummies reached by the rail's central dummy below it
                self.k_dh_c0 = self.getHandle('Kidney_Dummy_c0')
                self.k_dh_c1 = self.getHandle('Kidney_Dummy_c1')
                self.k_dh_c2 = self.getHandle('Kidney_Dummy_c2')
                self.k_dh_c3 = self.getHandle('Kidney_Dummy_c3')
                self.k_dh_c4 = self.getHandle('Kidney_Dummy_c4')
        
                self.k_orientation_ctrl = self.getHandle('Kidney_orientation_ctrl')

                # Step 2: get the position of these objects,
                # otherwise the script can't tell where they are and prints 
                # "failed to get position and orientation".
                # -1 means relative to base frame.
                self.getPosition(-1, ignoreError = True, initialize = True)

        # This method sets the pose (position and quaternion) of the
        # cuboid shape, which then "sets" its children along with it.
        def setPose(self, pos, quat, relative_handle, ignoreError=False):
                self.setPoseAtHandle(self.k_res_handle, relative_handle, 
                                     pos, quat, ignoreError)

        def getPosition(self, relative_handle, ignoreError=False, 
                        initialize = False):
                pos_res, _ = self.getPoseAtHandle(self.k_res_handle, 
                             relative_handle, ignoreError, initialize)
                pos_k, _ = self.getPoseAtHandle(self.k_handle, relative_handle, 
                           ignoreError, initialize)
                pos_convex, _ = self.getPoseAtHandle(self.convex, relative_handle, 
                                ignoreError, initialize)
                pos_0t, _ = self.getPoseAtHandle(self.k_dh_0t, relative_handle, 
                                ignoreError, initialize)
                pos_0b, _ = self.getPoseAtHandle(self.k_dh_0b, relative_handle, 
                                ignoreError, initialize)
                pos_1t, _ = self.getPoseAtHandle(self.k_dh_1t, relative_handle, 
                                ignoreError, initialize)
                pos_1b, _ = self.getPoseAtHandle(self.k_dh_1b, relative_handle, 
                                ignoreError, initialize)
                pos_2t, _ = self.getPoseAtHandle(self.k_dh_2t, relative_handle, 
                                ignoreError, initialize)
                pos_2b, _ = self.getPoseAtHandle(self.k_dh_2b, relative_handle, 
                                ignoreError, initialize)
                pos_3t, _ = self.getPoseAtHandle(self.k_dh_3t, relative_handle, 
                                ignoreError, initialize)
                pos_3b, _ = self.getPoseAtHandle(self.k_dh_3b, relative_handle, 
                                ignoreError, initialize)
                pos_4t, _ = self.getPoseAtHandle(self.k_dh_4t, relative_handle, 
                                ignoreError, initialize)
                pos_4b, _ = self.getPoseAtHandle(self.k_dh_4b, relative_handle, 
                                ignoreError, initialize)
                pos_c0, _ = self.getPoseAtHandle(self.k_dh_c0, relative_handle, 
                                ignoreError, initialize)
                pos_c1, _ = self.getPoseAtHandle(self.k_dh_c1, relative_handle, 
                                ignoreError, initialize)
                pos_c2, _ = self.getPoseAtHandle(self.k_dh_c2, relative_handle, 
                                ignoreError, initialize)
                pos_c3, _ = self.getPoseAtHandle(self.k_dh_c3, relative_handle, 
                                ignoreError, initialize)
                pos_c4, _ = self.getPoseAtHandle(self.k_dh_c4, relative_handle, 
                                ignoreError, initialize)
                pos_orientation_ctrl, _ = self.getPoseAtHandle(self.k_orientation_ctrl, 
                                          relative_handle, ignoreError, 
                                          initialize)

        # This method defines the target used at goal.
        # This target is sampled off the 5 available ones if ranomize
        # is true otherwise the number 2 is selected. 
        def getPositionGoal(self, relative_handle, randomize, 
                            ignoreError=False, initialize=False):
            
                # Checking the target randomization
                if randomize:
                    self.dummy_number = np.random.randint(0, 5)
                else:
                    self.dummy_number = 2

                if self.dummy_number == 0:
                    pos_c, _ = self.getPoseAtHandle(self.k_dh_c0, 
                            relative_handle, ignoreError, initialize)
                #print("Dummy pair PINK is goal.")
                elif self.dummy_number == 1:
                    pos_c, _ = self.getPoseAtHandle(self.k_dh_c1, 
                            relative_handle, ignoreError, initialize)
                #print("Dummy pair GREEN is goal.")
                elif self.dummy_number == 2:
                    pos_c, _ = self.getPoseAtHandle(self.k_dh_c2, 
                            relative_handle, ignoreError, initialize)
                #print("Dummy pair BLUE is goal.")
                elif self.dummy_number == 3:
                    pos_c, _ = self.getPoseAtHandle(self.k_dh_c3, 
                            relative_handle, ignoreError, initialize)
                #print("Dummy pair YELLOW is goal.")
                else:
                    pos_c, _ = self.getPoseAtHandle(self.k_dh_c4, 
                            relative_handle, ignoreError, initialize)
                #print("Dummy pair LILAC is goal.")

                return pos_c

        # This method returns the top and bottom targets sampled by getPositionGoal.
        # They are front-facing dummies on opposite sides.
        # pos_t is the position of the top-side dummy,
        # pos_b the position of the bottom-side dummy.
        def getPositionGoalTopBottom(self, relative_handle, ignoreError=False, 
                                     initialize=False):
                if self.dummy_number == 0:
                    pos_t, _ = self.getPoseAtHandle(self.k_dh_0t, 
                            relative_handle, ignoreError, initialize)
                    pos_b, _ = self.getPoseAtHandle(self.k_dh_0b, 
                            relative_handle, ignoreError, initialize)
                elif self.dummy_number == 1:
                    pos_t, _ = self.getPoseAtHandle(self.k_dh_1t, 
                            relative_handle, ignoreError, initialize)
                    pos_b, _ = self.getPoseAtHandle(self.k_dh_1b, 
                            relative_handle, ignoreError, initialize)
                elif self.dummy_number == 2:
                    pos_t, _ = self.getPoseAtHandle(self.k_dh_2t, 
                            relative_handle, ignoreError, initialize)
                    pos_b, _ = self.getPoseAtHandle(self.k_dh_2b, 
                            relative_handle, ignoreError, initialize)
                elif self.dummy_number == 3:
                    pos_t, _ = self.getPoseAtHandle(self.k_dh_3t, 
                            relative_handle, ignoreError, initialize)
                    pos_b, _ = self.getPoseAtHandle(self.k_dh_3b, 
                            relative_handle, ignoreError, initialize)
                else:
                    pos_t, _ = self.getPoseAtHandle(self.k_dh_4t, 
                            relative_handle, ignoreError, initialize)
                    pos_b, _ = self.getPoseAtHandle(self.k_dh_4b, 
                            relative_handle, ignoreError, initialize)

                return pos_t, pos_b

        # This method returns the orientation of Kidney_orientation_ctrl. 
        def getOrientationGoals(self, relative_handle, ignoreError=False, 
                                initialize=False):
                pos, quat = self.getPoseAtHandle(self.k_orientation_ctrl, 
                            relative_handle, ignoreError, initialize)
                return quat

            
# Checking the collision between robot arm and table
class collisionCheck(vrepObject):
        def __init__(self, clientID, psm_number):
                super(collisionCheck, self).__init__(clientID)

                self.collision_TTs_TableTop = self.getCollisionHandle('PSM{}_TTs_Table'.format(psm_number))
                self.collision_TTd_TableTop = self.getCollisionHandle('PSM{}_TTd_Table'.format(psm_number))

                #Collision objects of Rail. Check against the kidney, cuboid and convex shell.
                self.collision_Kidney_Rail = self.getCollisionHandle('Collision_Kidney_Rail')
                self.collision_Cuboid_Rail = self.getCollisionHandle('Collision_Cuboid_Rail')
                self.collision_Convex_Rail = self.getCollisionHandle('Collision_Convex_Rail')

                #Collision objects of Robot. Check against convex all the possible colliding parts.
		#These are, the black cylinder, the TT's body and tips.
                self.collision_Convex_Cylinder = self.getCollisionHandle('Collision_Convex_Cylinder')
                self.collision_Convex_TT_body = self.getCollisionHandle('Collision_Convex_TT_body')
                self.collision_Convex_TT_sx = self.getCollisionHandle('Collision_Convex_TT_sx')
                self.collision_Convex_TT_dx = self.getCollisionHandle('Collision_Convex_TT_dx') 

                # Init
                super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Kidney_Rail, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Cuboid_Rail, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Convex_Rail, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Convex_Cylinder,
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Convex_TT_body, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Convex_TT_sx, 
                                            ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_Convex_TT_dx, 
                                            ignoreError=True, initialize=True)

    # Returns True if in collision and False if not in collision
        def checkCollision(self, ignoreError=False):
                c1 = super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError)
                c2 = super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError)
                return c1 or c2
    
    # Any checkCollision call returns True if collision and False if not collision
        def KidneyCollision(self, ignoreError=False):
                #c_r = collision result
                c_r1 = super(collisionCheck,self).checkCollision(self.collision_Kidney_Rail, 
                                                                 ignoreError)
                c_r2 = super(collisionCheck,self).checkCollision(self.collision_Cuboid_Rail, 
                                                                 ignoreError)
                c_r3 = super(collisionCheck,self).checkCollision(self.collision_Convex_Rail, 
                                                                 ignoreError)
                c_r4 = super(collisionCheck,self).checkCollision(self.collision_Convex_Cylinder,
                                                                 ignoreError)
                c_r5 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_body, 
                                                                 ignoreError)
                c_r6 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_sx, 
                                                                 ignoreError)
                c_r7 = super(collisionCheck,self).checkCollision(self.collision_Convex_TT_dx, 
                                                                 ignoreError)
                return c_r1, c_r2, c_r3, c_r4, c_r5, c_r6, c_r7 
