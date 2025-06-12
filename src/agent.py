# %%

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import torch.nn as nn
from src.utils import Config
from src.networks import PolicyMLP, ValueMLP

class Agent():
    """
    An agent that learns to solve an environment using a policy gradient algorithm.
    Action space must be discrete.

    Contains the networks to approximate the policy and the value function, and the logic
    for selecting an action and updating the network parameters.
    """
    def __init__(self, num_features: int, num_actions: int, config: Config) -> None:

        self.num_features = num_features
        self.num_actions = num_actions
        self.config = config

        self.policy_network = PolicyMLP(num_features, self.config.policy_hidden_size, num_actions).to(self.config.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.policy_learning_rate)

        self.value_network = ValueMLP(num_features, self.config.value_hidden_size, 1).to(self.config.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate)

    def reset_value_optimizer(self):
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate)

    def select_action(self, observation: np.ndarray):
        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.config.device)
        self.policy_network.train()
        logits = self.policy_network(observation_tensor)
        action_probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based and compute log-probability
        action = np.random.choice(self.num_actions, p=action_probs.detach().cpu().numpy())
        log_prob = torch.log(action_probs[action])
        
        return action, log_prob

    def update_policy_network(self, batch_loss_policy_network):
        self.policy_network.train()
        self.policy_optimizer.zero_grad()
        batch_loss_policy_network.backward()
        self.policy_optimizer.step()

    def update_value_network(self, batch_loss_value_network):
        self.reset_value_optimizer()
        self.value_network.train()
        self.value_optimizer.zero_grad()
        batch_loss_value_network.backward()
        self.value_optimizer.step()

        

# config = Config()
# print(f"Using {config.device} device")
# agent = Agent(4, 2, config)

# x = agent.select_action(np.array([1.0, 0.0, 0.0, 0.0]))
# # %%
# import numpy as np
# import gymnasium as gym
# env = gym.make("CartPole-v1")
# obs, info = env.reset()
# isinstance(obs, np.ndarray)
