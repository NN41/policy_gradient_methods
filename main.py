
import gymnasium as gym

from src.agent import Agent
from src.trainer import Trainer
from src.config import Config

if __name__ == '__main__':
    print('Running training loop...')

    config = Config(
        render_every_n_epochs=1,
        log_params_every_n_epochs=5,
        num_epochs_policy_network=200,
        policy_hidden_size=8,
        policy_learning_rate=0.01,
        num_episodes=20,
        weight_kind='dfr',
    )

    env = gym.make(config.env_name)
    agent = Agent(env, config)
    trainer = Trainer(env, agent, config)

    trainer.train()
