import gym
import numpy as np
import dVRL_simulator
from dVRL_simulator.PsmEnv import PSMEnv
from dVRL_simulator.vrep.simObjects import table, obj, target, kidney, rail
import transforms3d.euler as euler
import transforms3d.quaternions as quaternions

from stable_baselines import HER, DQN, SAC, DDPG, TD3
from stable_baselines.her import GoalSelectionStrategy, HERGoalEnvWrapper
from stable_baselines import results_plotter
from stable_baselines.common import set_global_seeds, make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

# Running multiples enviroments in parallel
def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG 
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

# Main 

if __name__ == '__main__':
    env_id = "dVRLPick-v0"
    num_cpu = 8 # Number of processes to use
    # Create the vectorized environment
    env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

    model_class = DDPG # works also with SAC, DDPG and TD3

    #env = gym.make("dVRLPick-v0") 

    # Available strategies (cf paper): future, final, episode, random
    goal_selection_strategy = 'future' # equivalent to GoalSelectionStrategy.FUTURE

    # Wrap the model
    model = HER('MlpPolicy', env, model_class, n_sampled_goal=4, 
                goal_selection_strategy=goal_selection_strategy, verbose=1)
    # Train the model
    model.learn(1000)

    model.save("./results_training_her/her_bit_env")

    # WARNING: you must pass an env
    # or wrap your environment with HERGoalEnvWrapper to use the predict method
    model = HER.load('./results_training_her/her_bit_env', env=env)

    episode_reward = 0
    obs = env.reset()
    for _ in range(100):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            print("Reward:", episode_reward)
            episode_reward = 0.0
            obs = env.reset()



