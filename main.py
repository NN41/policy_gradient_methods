
import gymnasium as gym

from src.agent import Agent
from src.trainer import Trainer
from src.config import Config

if __name__ == '__main__':
    print('Running training loop...')
    config = Config(weight_kind='gae', avg_kind='a', num_epochs_policy_network=5, num_episodes=20)
    env = gym.make(config.env_name)
    agent = Agent(env, config)
    trainer = Trainer(env, agent, config)
    trainer.train()
