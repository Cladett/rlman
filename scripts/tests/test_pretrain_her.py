"""
@brief  Example of HER usage with stable baselines and model pretraining.
@author Claudia D'Ettorre (c.dettorre@ucl.ac.uk).
@date   28 AUg 2020.
"""
import stable_baselines


def main():
    # Create DL model to generate the expert demo
    #expert_model = stable_baselines.DQN('MlpPolicy', 'CartPole-v1', verbose=1)

    # Train a DQN agent for 1e5 timesteps and generate trajectories
    # data will be saved in a numpy archive named expert_cartpole.npz
    #stable_baselines.gail.generate_expert_traj(expert_model,
    #                                           'expert_cartpole',
    #                                           n_timesteps=0,
    #                                           n_episodes=3)
    dataset = stable_baselines.gail.ExpertDataset(
        expert_path='/home/claudia/catkin_ws/src/dVRL/stable-baselines/stable_baselines/gail/dataset/expert_cartpole.npz', traj_limitation=-1, batch_size=128)

    #import pudb; pudb.set_trace()

    # Create DL model that learns from expert demo
    model = stable_baselines.HER('MlpPolicy', 'CartPole-v1', 
                model_class=stable_baselines.DQN, n_sample_goal=4,
                goal_selection_strategy='future', verbose=1)

    # Pretrain the PPO2 model
    #model.pretrain(dataset, n_epochs=10)

    # As an option, you can train the RL agent
    #model.learn(int(1e5))

if __name__ == '__main__':
    main()
