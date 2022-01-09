"""
@brief    Script use to test if the environment are run correctly. 
          The script runs the Coppeliasim container with the specified
          environment id and executes twice a sample task for each of the selected
          environment. 
@authord  Claudia D'Ettorre (c.dettorre@ucl.ac.uk)
@date     21 Apr 2021
"""
import argparse
import numpy as np
import docker
import gym
import dVRL_simulator

class Demo():

    def __init__(self, env_id):
        # Store params
        self.env_id = env_id

        #import pudb; pudb.set_trace()
        # Initialise data structures
        # self.episode_returns = []
        # self.actions = []
        # self.obs = []
        # self.rewards = []
        # self.episode_starts = []
        
        # Initialise gym env and wrapping it for having obs as single array 
        # of floats instead of a dictionary. Selecting the env based on the task
        if self.env_id == 'dVRLReach-v0':
            self.env_dvrk = gym.make("dVRLReach-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
        if self.env_id == 'dVRLReachImage-v0':
            self.env_dvrk = gym.make("dVRLReachImage-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
        if self.env_id == 'dVRLReachRail-v0':
            self.env_dvrk = gym.make("dVRLReachRail-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLReachKidney-v0':
            self.env_dvrk = gym.make("dVRLReachKidney-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 0  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 0 
        if self.env_id == 'dVRLPickRail-v0':
            self.env_dvrk = gym.make("dVRLPickRail-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 80  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTarget-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTarget-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
            ###############################################################
        if self.env_id == 'dVRLPickPlaceTargetE1-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE1-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetE2-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE2-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetE3-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE3-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetE4-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE4-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetE5-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE5-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetE6-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetE6-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE1-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE1-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE2-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE2-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE3-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE3-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE4-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE4-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE5-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE5-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEvalE6-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEvalE6-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
            ###############################################################
        if self.env_id == 'dVRLPickPlaceTargetObs-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetObs-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        if self.env_id == 'dVRLPickPlaceTargetEval-v0':
            self.env_dvrk = gym.make("dVRLPickPlaceTargetEval-v0")
            # Defining number of timesteps for each part of the task 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        elif self.env_id == 'dVRLPick-v0':
            self.env_dvrk = gym.make("dVRLPick-v0")
            # Defining number of timesteps for each part of the task 
            # self.APPROACH_STEPS = 25  # ee start from pos [0, 0, -0.11] 
            self.APPROACH_STEPS = 40  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 
        elif self.env_id == 'dVRLPickPlace-v0':
            self.env_dvrk = gym.make("dVRLPickPlace-v0")
            # Defining number of timesteps for each part of the task 
            # self.APPROACH_STEPS = 25  # ee start from pos [0, 0, -0.11] 
            self.APPROACH_STEPS = 90  # rail ee start from pos [0, 0, -0.07]
            self.GRASP_STEPS = 5 

    def run_episode(self):
        """
        @brief  Each episode runs a reset of the environment and whole 
                demonstration 
        """
        # Set up the enviroment 
        prev_obs = self.env_dvrk.reset()
        prev_obs = self.env_dvrk.reset() #fixing bug with rnd orientation
        self.env_dvrk.render()
        

        # Run steps: the pickrail task does not have any reach phase
        if self.env_id == 'dVRLReach-v0' or self.env_id == 'dVRLReachRail-v0' or self.env_id == 'dVRLReachImage-v0':
            prev_obs = self.approach(prev_obs)
        elif self.env_id == 'dVRLPickRail-v0':
            # For the pickrail env need to reset twice otherways 
            # does not display second execution of the task
            #prev_obs = self.env_dvrk.reset(); self.env_dvrk.render()
            prev_obs = self.approach(prev_obs)
            #prev_obs, info_grasp = self.grasp(prev_obs)
            prev_obs = self.grasp(prev_obs)
        elif self.env_id == 'dVRLReachKidney-v0':
            prev_obs, info = self.reach(prev_obs)
        else: 
            prev_obs = self.approach(prev_obs)
            prev_obs = self.grasp(prev_obs)
            prev_obs, info = self.reach(prev_obs)
        
        
    def approach(self, prev_obs):
        """
        @brief This function moves the end effector from the initial position 
               towards the object. 
        """
        for i in range(self.APPROACH_STEPS):
            if self.env_id == 'dVRLReach-v0':
                pos_goal = prev_obs['desired_goal']
                pos_ee = prev_obs['observation']
                raw_action = np.array(pos_goal - pos_ee)
                action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                                  10 * raw_action[2]], -1, 1)
            elif self.env_id == 'dVRLReachImage-v0': 
                action = np.random.uniform(0,6)
            else:
                pos_ee_start = 0 
                pos_ee_end = 3
                pos_ee   = prev_obs['observation'][pos_ee_start: pos_ee_end] 
                pos_obj_start = 4
                pos_obj_end = 7
                pos_obj  = prev_obs['observation'][pos_obj_start:pos_obj_end] 
                raw_action = np.array(pos_obj - pos_ee)
                action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                    10 * raw_action[2], 1], -1, 1)
            # Execute action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  
            prev_obs = obs

        if self.env_id == 'dVRLReach-v0' or self.env_id == 'dVRLReachRail-v0':
            print('Is the task successeful', info['is_success'])
        return prev_obs

    def grasp(self, prev_obs):
        """
        @brief This function executes the static grasping of the object. 
        """
        for i in range(self.GRASP_STEPS):
            # Execute the action in the enviroment 
            action = [0, 0, 0, -0.5]
            obs, reward, done, info = self.env_dvrk.step(action)  

            # Store action results in episode lists
            prev_obs = obs

        if self.env_id == 'dVRLPickRail-v0':
            print('Is the task successeful', info['is_success'])
        return prev_obs

    def reach(self, prev_obs):
        """
        @brief This method executes the reaching towards the target.
               Using a wrapped environment the observation are not present as 
               a dic of 'desired_goal', 'observation' anymore. They are now a 
               single numpy array so in order to access each component we need
               to acces different part of the array with the following structure
               
               'observation': 
                array([ 5.00679016e-05, -6.19888306e-05,  8.96453857e-05,  
                        1.15444704e-03, -7.15255737e-05, -2.86102295e-05, 
                        -8.39406013e-01]),
                'achieved_goal': array([-7.15255737e-05, -2.86102295e-05, 
                                        -8.39406013e-01]),
                'desired_goal': array([-0.85572243,  0.70448399, -0.37535763])}

                In the new release SB3 the definition of obeservation has been                                                                                     
                updated using only 'observation' and 'desired_goal'.

               
        """
        steps = self.env_dvrk._max_episode_steps \
                - self.APPROACH_STEPS \
                - self.GRASP_STEPS

        pos_ee_start = 0 
        pos_ee_end = 3
        pos_obj_start = 4
        pos_obj_end = 7
        
        # For the reach kidney env the observation is array of 3 elements
        if self.env_id == 'dVRLReachKidney-v0':
            pos_obj_start = 0
            pos_obj_end = 3

        for i in range(steps):
            goal     = prev_obs['desired_goal']
            pos_ee   = prev_obs['observation'][pos_ee_start: pos_ee_end] 
            pos_obj  = prev_obs['observation'][pos_obj_start:pos_obj_end] 
            raw_action = np.array(goal - pos_obj)
            action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                10 * raw_action[2], -0.5], -1, 1)

            if self.env_id == 'dVRLReachKidney-v0':
                action = np.clip([10 * raw_action[0], 10 * raw_action[1], 
                    10 * raw_action[2]], -1, 1)
            # Executing the action in the enviroment 
            obs, reward, done, info = self.env_dvrk.step(action)  
            
            # Store action results in episode lists
            prev_obs = obs

        print('Is the task successeful', info['is_success'])
        # Adding control if the goal is not reached
        return prev_obs, info


def parse_cmdline_args():
    parser = argparse.ArgumentParser(description='Running sample of environment.')
    parser.add_argument('--env-id', required=True, 
                        help='Environment id used to register the environment.')
    args = parser.parse_args()
    return args

def main():
    # Read command line parameters
    args = parse_cmdline_args()

    # Number of demo task 
    number_demo = 2 

    print('id', args.env_id)
    # Perform sample of the task  
    task_demo = Demo(args.env_id)
    for i in range(number_demo):
        print('Running test environment number', (i+1))
        task_demo.run_episode()

    return 


if __name__ == '__main__':
    main()
