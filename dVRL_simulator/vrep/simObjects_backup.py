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
                        return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0], 3])
                else:
                        return np.array(data, dtype=np.uint8).reshape([resolution[1], resolution[0]])

# Checking the collision between robot arm and table
class collisionCheck(vrepObject):
        def __init__(self, clientID, psm_number):
                super(collisionCheck, self).__init__(clientID)

                self.collision_TTs_TableTop = self.getCollisionHandle('PSM{}_TTs_Table'.format(psm_number))
                self.collision_TTd_TableTop = self.getCollisionHandle('PSM{}_TTd_Table'.format(psm_number))

                super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError=True, initialize=True)
                super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError=True, initialize=True)

    # Returns True if in collision and False if not in collision
        def checkCollision(self, ignoreError=False):
                c1 = super(collisionCheck, self).checkCollision(self.collision_TTs_TableTop, ignoreError)
                c2 = super(collisionCheck, self).checkCollision(self.collision_TTd_TableTop, ignoreError)

                return c1 or c2


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

# Target: represent by a red ball in the simulation. Used as target for the 
# reach task and pick task.
class target(vrepObject):
        def __init__(self, clientID, psm_number):
                super(target, self).__init__(clientID)

                self.target_handle = self.getHandle('Target_PSM{}'.format(psm_number))

                self.getPosition(-1, ignoreError=True, initialize=True)

        def setPosition(self, pos, relative_handle, ignoreError=False):
                self.setPoseAtHandle(self.target_handle, relative_handle, pos, [1, 0, 0, 1], ignoreError)

        def getPosition(self, relative_handle, ignoreError=False, initialize=False):
                pos, _ = self.getPoseAtHandle(self.target_handle, relative_handle, ignoreError, initialize)
                return pos


# Claudia: adding the new objects. kidney and rail
class kidney(vrepObject):
        def __init__(self,clientID):
                super(kidney, self).__init__(clientID)

                self.kidney_handle = self.getHandle('kidney')
                self.dummy1_kidney_handle = self.getHandle('kidney_Dummy1')
                self.dummy2_kidney_handle = self.getHandle('kidney_Dummy2')
                self.dummy3_kidney_handle = self.getHandle('kidney_Dummy3')
                self.getPosition(-1, ignoreError = True, initialize = True)

        def setPose(self, pos, quat, relative_handle, ignoreError = False): 
                b_T_d = self.posquat2Matrix(pos, quat)
                d_T_o = np.array([[1, 0, 0, 0], [0,1,0,0], [0,0,1,0.001], [0,0,0,1]])
                pos, quat = self.matrix2posquat(np.dot(b_T_d,d_T_o))
                self.setPoseAtHandle(self.obj_handle, relative_handle, pos, quat, ignoreError)

        def getPose(self, relative_handle, ignoreError = False, initialize = False):                                                     
                return self.getPoseAtHandle(self.dummy_handle, relative_handle, ignoreError, initialize)                                 

        def getVel(self, ignoreError = False, initialize = False):                                                                       
                return self.getVelocityAtHandle(self.dummy_handle, ignoreError, initialize)                                              


# Claudia: declearing object reail to be picked.
# The Dummy is located for now with only a traslation on z-axe (longer axe of the rail) of +0.010 m and a rotation of 90 along x-axe parent frame.
# We are using the opposite convention of rotation (see notes)
class rail(vrepObject):
        def __init__(self, clientID):
                super(rail, self).__init__(clientID)

                self.rail_handle = self.getHandle('rail')
                #self.dummy_rail_handle = self.getHandle('rail_Dummy') #if using single dummy
                self.dummy1_rail_handle = self.getHandle('rail_Dummy1') #if using single dummy
                self.dummy2_rail_handle = self.getHandle('rail_Dummy2') #if using single dummy
                self.dummy3_rail_handle = self.getHandle('rail_Dummy3') #if using single dummy
                self.dummy4_rail_handle = self.getHandle('rail_Dummy4') #if using single dummy
                self.dummy5_rail_handle = self.getHandle('rail_Dummy5') #if using single dummy
                self.dummy6_rail_handle = self.getHandle('rail_Dummy6') #if using single dummy
                self.dummy7_rail_handle = self.getHandle('rail_Dummy7') #if using single dummy
                self.dummy8_rail_handle = self.getHandle('rail_Dummy8') #if using single dummy

                
                # Claudia: initializing value to read proximity sensor
                self.proximity_handle= self.getHandle('TOOL1_proxSensor')
                self.readProximitySensor(self.proximity_handle, ignoreError=True, initialize=True)

        def setPose(self, pos, quat, grasp_site, relative_handle, 
                    ignoreError=False):
                b_T_d = self.posquat2Matrix(pos, quat)
                pos_forced=pos
                #import pudb; pudb.set_trace()
                
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
                #print('rail pos computed from the dummy in the /base \n', pos)
                
                #b_T_table = np.array([
                #   [ 3.41779961e-01, 9.39780005e-01, -8.02261448e-06,  1.89346313e-01],
                #   [-9.39780005e-01, 3.41779961e-01,  5.97869184e-07,  2.79672146e-02],
                #   [ 3.30383437e-06, 7.33515297e-06,  1.00000000e+00, -1.23435020e-01],
                #   [ 0,  0,  0, 1]
                #])

                self.setPoseAtHandle(self.rail_handle, relative_handle, 
                        pos, quat, ignoreError) # relative handle: psm.base_handle
                #import pudb; pudb.set_trace()
                return self.dummy_rail_handle , pos
 
        def getPose(self, dummy_rail_handle, relative_handle, ignoreError=False, initialize=False):
                return self.getPoseAtHandle(dummy_rail_handle, relative_handle, ignoreError, initialize)

        def getVel(self, ignoreError=False, initialize=False):
                return self.getVelocityAtHandle(self.dummy_rail_handle, ignoreError, initialize)

        def removeGrasped(self, ignoreError=False):
                self.setParent(self.rail_handle, -1, True, ignoreError)

        def isGrasped(self, ignoreError=False, initialize=False):
                return not (-1 == self.getParent(self.rail_handle, ignoreError, initialize))
       
        #Claudia: adding function to read the proximity sensor
        def readProximity(self, ignoreError=False):
                success_detection, det_point, det_handle, distance_norm = self.readProximitySensor(self.proximity_handle, ignoreError=ignoreError, initialize=True)
                return  success_detection, det_point, det_handle, distance_norm 
