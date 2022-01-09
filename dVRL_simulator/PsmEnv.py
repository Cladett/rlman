"""
@brief   Script that connects with the simulaton environment running inside 
         docker container and executes the main reset, step, render.
@author  Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date    06 Oct 2020
"""

#try:
import dVRL_simulator.vrep.vrep as vrep
#except:
#  print ('--------------------------------------------------------------')
#  print ('"vrep.py" could not be imported. This means very probably that')
#  print ('either "vrep.py" or the remoteApi library could not be found.')
#  print ('Make sure both are in the same folder as this file,')
#  print ('or appropriately adjust the file "vrep.py"')
#  print ('--------------------------------------------------------------')
#  print ('')


#import copy
import numpy as np
import matplotlib.pyplot as plt
#import os.path
import socket
import fcntl
import struct
import pathlib
import subprocess
import re
import docker
import os
import time
import gym
from gym import error, spaces
from gym.utils import seeding

# My imports
from dVRL_simulator.vrep.ArmPSM import ArmPSM
from dVRL_simulator.vrep.simObjects import camera



class PSMEnv(gym.GoalEnv):

  @staticmethod
  def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915,  struct.pack('256s', ifname[:15].encode('utf-8')))[20:24])

  @staticmethod
  def prep_docker_env():
    # Detect if there is a local X server
    display = os.environ['DISPLAY']
    local_display_pattern = '^[:]\d+$'
    m = re.match(local_display_pattern, display)    
    
    # If there is a local X server
    env = {'QT_X11_NO_MITSHM': 1}
    vol = {}
    if m:  
      # Set the DISPLAY environment variable 
      env['DISPLAY'] = display
        
      # Mapping folder of unix sockets
      vol['/tmp/.X11-unix'] = {
        'bind': "/tmp/.X11-unix",
        'mode':"rw"
      }

      # Give the root user running vrep inside the container permission to
      # write to the unix socket so that we can display
      cmd = ['xhost','+SI:localuser:root']
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
      out, err = proc.communicate()

    # If we are in a remote machine via SSH X11 forwarding
    else:
      # Set the DISPLAY environment variable 
      display = os.environ['DISPLAY']
      ip = PSMEnv.get_ip_address('docker0')
      port_offset = display.split(':')[1]
      display = ip + ':' + port_offset
      env['DISPLAY'] = display

      # Create an empty Xauthority file
      env['XAUTHORITY'] = '/tmp/.docker.xauth'
      if not os.path.isfile(env['XAUTHORITY']):
        pathlib.Path(env['XAUTHORITY']).touch()

      # Get X11 cookie
      cmd = ['xauth', 'nlist', os.environ['DISPLAY']]
      proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, 
                              stderr=subprocess.STDOUT)
      out, err = proc.communicate() 
      out = out.decode('ascii')
      out = 'ffff' + out[4:]

      # Save X11 cookie in a temporary file
      cookie_path = '/tmp/.myx11cookie'
      text_file = open(cookie_path, 'w')
      text_file.write(out)
      text_file.close()

      # Merge cookie in Xauthority file
      cmd = ['xauth', '-f', env['XAUTHORITY'], 'nmerge', cookie_path]
      proc = subprocess.Popen(cmd)
      proc.wait()

      # Mount Xauthority file inside the container
      vol[env['XAUTHORITY']] = {
        'bind': env['XAUTHORITY'],
        'mode':"rw"
      }

    return env, vol


  """Initializes a new double PSM Environment
  Args:
    psm_num (int): which psm to enable, if not 1 or 2 both are enabled
    n_actions (int): the number of actions possible in the environment
    n_states (int): the state dimension
    n_goals (int): the goal dimension
    n_substeps (int): number of substeps between each "step" of the 
                      environment.
    camera_enabled (bool): if the cameras should be enabled. This slows 
                           down the environment a lot...
    docker_container (string): name of the docker container that loads the 
                               v-rep
    action_type (string): type of action space that can be chosen
  """
  def __init__(self, psm_num, n_actions, n_states, n_goals, n_substeps,
      camera_enabled, action_type, docker_container):

    self.viewer  = None

    # Create docker of the environment
    client = docker.from_env()
    env, vol = PSMEnv.prep_docker_env()
    kwargs = {
                'environment' : env,
                'volumes' : vol,
                'runtime' : 'nvidia',
    } 
        
    # Run the container
    self.container = client.containers.run(docker_container, detach = True, **kwargs)

    # Get the IP address of the container
    cmd = ['docker','inspect','-f',"'{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}'", self.container.id] 
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out, err = proc.communicate()
    ip_pattern = '\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}'  # Regular expression of IPv4
    self.container_ip = re.findall(ip_pattern, str(out))[0]
    print('Container ip', self.container_ip)
    #input('wait')

    # Try to connnect to v-rep via the port
    res = -1
    attempts = 0
    max_attempts = 10
    vrep_port = 19999
    while res != vrep.simx_return_ok and attempts < max_attempts:
      attempts += 1
      time.sleep(1)
      #vrep.simxFinish(-1)
      self.clientID = vrep.simxStart(self.container_ip, vrep_port, True, True, 5000, 5) # Connect to V-REP
      res = vrep.simxSynchronous(self.clientID , True)

    if res != vrep.simx_return_ok:
      raise IOError('V-Rep failed to load!')

    vrep.simxStartSimulation(self.clientID , vrep.simx_opmode_oneshot)

    #Only initializes the psm asked for, otherwise initializes both
    self.psm_num = psm_num
    if self.psm_num == 1:
      self.psm1 = ArmPSM(self.clientID, 1)
      self.psm2 = None
    elif self.psm_num == 2:
      self.psm1 = None
      self.psm2 = ArmPSM(self.clientID, 2)
    else:
      self.psm1 = ArmPSM(self.clientID, 1)
      self.psm2 = ArmPSM(self.clientID, 2)

    self.n_substeps = n_substeps

    # Time step is set to in V-REP
    self.sim_timestep   = 0.1

    self.viewer = None
    self.camera_enabled = camera_enabled
    if self.camera_enabled:
      self.metadata = {'render.modes': ['matplotlib', 'rgb', 'human']}
      self.camera = camera(self.clientID, rgb = True)
    else:
      self.metadata = {'render.modes': ['human']}


    self.seed()
    self._env_setup()
    
    self.action_type = action_type
    if self.action_type == 'discrete':
        self.discrete_actions = [0, 1, 2, 3, 4, 5, 6] # seven actions: stay still or move forward/backward in the 3 dimensions 
        self.action_space = spaces.Discrete(7)
        self.action = np.random.choice(self.discrete_actions)
    else:
        self.action_space = spaces.Box(-1., 1., shape=(n_actions,), 
                                       dtype='float32')
    self.observation_space = spaces.Dict(dict(
      desired_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
      achieved_goal=spaces.Box(-np.inf, np.inf, shape=(n_goals,), dtype='float32'),
      observation=spaces.Box(-np.inf, np.inf, shape=(n_states,), dtype='float32'),
      ))

  def __del__(self):
    self.close()

  @property
  def dt(self):
    return self.sim_timestep * self.n_substeps

  # Env methods
  # ----------------------------

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    if self.action_type == 'continuous':
        action = np.clip(action, self.action_space.low, self.action_space.high)
    obs_prev = self._get_obs()
    self._set_action(action)

    self._simulator_step()
    self._step_callback()

    obs = self._get_obs()

    done = False
    info = {
      #'is_success': self._is_success(obs[-3:], self.goal),
      'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
      }
    #reward = self.compute_reward(obs[-3:], self.goal, info)
    reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'], info)
    return obs, reward, done, info

  def reset(self):
    self.psm1.setBooleanParameter(vrep.sim_boolparam_display_enabled, False, ignoreError = True)
    # Attempt to reset the simulator. Since we randomize initial conditions, it
    # is possible to get into a state with numerical issues (e.g. due to penetration or
    # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
    # In this case, we just keep randomizing until we eventually achieve a valid initial
    # configuration.
    did_reset_sim = False
    while not did_reset_sim:
        did_reset_sim = self._reset_sim()
    self.goal = self._sample_goal().copy()
    obs = self._get_obs()
    
    #########
    # TODO: fixing bug always successful 
    #info = {
    #  #'is_success': self._is_success(obs[-3:], self.goal),
    #  'is_success': self._is_success(obs['achieved_goal'], obs['desired_goal']),
    #  }
    #print('distance ee-goal', (np.linalg.norm(obs['desired_goal'] - obs['achieved_goal'], axis=-1))*self.target_range)
    #print('successful', info)
    #print('obs at reset', obs)
    ############
    return obs

  def close(self):
    if self.viewer is not None:
      plt.close(self.viewer.number)
      self.viewer = None
    vrep.simxFinish(self.clientID)
    self.container.kill()



  def render(self, mode = 'human'):
    if mode == 'human':
      self.psm1.setBooleanParameter(vrep.sim_boolparam_display_enabled, True, ignoreError = True)
    elif mode == 'matplotlib' and self.camera_enabled:
      if self.viewer is None:
        self.viewer = plt.figure()
      plt.figure(self.viewer.number)
      img = self.camera.getImage()
      plt.imshow(img, origin='lower')
    elif mode == 'rgb' and self.camera_enabled:
      return self.camera.getImage()

  def _get_viewer(self):
    """ no viewer has been made yet! 
    """    
    raise NotImplementedError()


  # Extension methods
  # ----------------------------

  def _simulator_step(self):
    for i in range(0, self.n_substeps):
      vrep.simxSynchronousTrigger(self.clientID)
    vrep.simxGetPingTime(self.clientID)

  def _reset_sim(self):
    """Resets a simulation and indicates whether or not it was successful.
    If a reset was unsuccessful (e.g. if a randomized state caused an error in the
    simulation), this method should indicate such a failure by returning False.
    In such a case, this method will be called again to attempt a the reset again.
    """

    return True

  def _get_obs(self):
    """Returns the observation.
    """
    raise NotImplementedError()

  def _set_action(self, action):
    """Applies the given action to the simulation.
    """
    raise NotImplementedError()

  def _is_success(self, achieved_goal, desired_goal):
    """Indicates whether or not the achieved goal successfully achieved the desired goal.
    """
    raise NotImplementedError()

  def _sample_goal(self):
    """Samples a new goal and returns it.
    """
    raise NotImplementedError()

  def _env_setup(self):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    pass

  def _viewer_setup(self):
    """Initial configuration of the viewer. Can be used to set the camera position,
    for example.
    """
    pass

  def _render_callback(self):
    """A custom callback that is called before rendering. Can be used
    to implement custom visualizations.
    """
    pass

  def _step_callback(self):
    """A custom callback that is called after stepping the simulation. Can be used
    to enforce additional constraints on the simulation state.
    """
    pass


