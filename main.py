
import gymnasium as gym

from src.agent import Agent
from src.trainer import Trainer
from src.config import Config

if __name__ == '__main__':

    print('\nRunning training loop...')

    # set up non-default hyperparameters
    config = Config(
        render_every_n_epochs=1,
        log_params_every_n_epochs=1,
        num_epochs_policy_network=20,
        weight_kind='dfr', # discounted future returns
    )

    # set up environment, agent and trainer
    env = gym.make(config.env_name)
    agent = Agent(env, config)
    trainer = Trainer(env, agent, config)

    # run a single training loop
    trainer.train()
