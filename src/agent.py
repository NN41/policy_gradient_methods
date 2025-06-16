# %%

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym

from src.config import Config
from src.networks import PolicyMLP, ValueMLP

class Agent():
    """
    An agent that learns to solve an environment using a policy gradient algorithm.
    Action space must be discrete.

    Contains the networks to approximate the policy and the value function, and the logic
    for selecting an action and updating the network parameters.
    """
    def __init__(self, env: gym.Env, config: Config) -> None:

        self.num_actions = env.action_space.n
        self.num_features = env.observation_space.shape[0]
        self.config = config

        self.policy_network = PolicyMLP(self.num_features, self.config.policy_hidden_size, self.num_actions).to(self.config.device)
        self.policy_optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.policy_learning_rate)

        self.value_network = ValueMLP(self.num_features, self.config.value_hidden_size, 1).to(self.config.device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate)

    def select_action(self, observation: np.ndarray, inference_mode: bool = False):

        observation_tensor = torch.tensor(observation, dtype=torch.float32).to(self.config.device)
        if not inference_mode:
            self.policy_network.train()
            logits = self.policy_network(observation_tensor)
        else:
            self.policy_network.eval()
            with torch.no_grad():
                logits = self.policy_network(observation_tensor)

        action_probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based and compute log-probability
        action = np.random.choice(self.num_actions, p=action_probs.detach().cpu().numpy())
        log_prob = torch.log(action_probs[action])
        
        return action, log_prob

    def update_policy_network(self, batch_loss_policy_network):
        self.policy_optimizer.zero_grad()
        batch_loss_policy_network.backward()
        self.policy_optimizer.step()

    def reset_value_optimizer(self):
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.config.value_learning_rate)

    # def update_value_network(self, batch_loss_value_network):
    #     self.value_optimizer.zero_grad()
    #     batch_loss_value_network.backward()
    #     self.value_optimizer.step()

        

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
