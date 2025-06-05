# %% 

import numpy as np
import os
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.networks import PolicyMLP, ValueMLP
from src.training import create_dataloaders_for_value_network, train_value_network
from src.utils import run_episode, render_epsiode

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions: {num_actions}")

# %%

class Config:
    def __init__(self, **kwargs):

        self.num_episodes = 50
        self.num_epochs_policy_network = 100
        self.render_every_n_epochs = 500
        self.log_params_every_n_epochs = 3

        self.policy_hidden_size = 4
        self.policy_learning_rate = 0.01

        self.value_hidden_size = 8
        self.value_learning_rate = 0.001
        self.num_epochs_value_network = 1

        self.weight_kind = 'rtgv' # or 'rtg' or 'rtgv'
        self.avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 

        self.log_dir = None

        self.experiment_name = ""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
            self.experiment_name += "|" + key + ":" + str(value)

config = Config(num_episodes=100)
config.experiment_name


# %%


def train_agent(config: Config):

    num_episodes = config.num_episodes
    num_epochs_policy_network = config.num_epochs_policy_network
    render_every_n_epochs = config.render_every_n_epochs
    log_params_every_n_epochs = config.log_params_every_n_epochs
    policy_hidden_size = config.policy_hidden_size
    policy_learning_rate = config.policy_learning_rate
    value_hidden_size = config.value_hidden_size
    value_learning_rate = config.value_learning_rate
    num_epochs_value_network = config.num_epochs_value_network
    weight_kind = config.weight_kind
    avg_kind = config.avg_kind
    log_dir = config.log_dir

    writer = SummaryWriter(log_dir)

    policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)

    for epoch in range(num_epochs_policy_network):

        print(f"(Policy Net) Epoch {epoch+1} / {num_epochs_policy_network}:")

        batch_returns = []
        batch_lengths = []
        batch_lprobs = []
        batch_obs = []
        # batch_rewards = []
        # batch_rewards_to_go = []

        batch_full_returns = []
        batch_future_returns = []
        batch_weights = []

        policy_network.eval()
        print(f"\tSimulating {num_episodes} episodes...")
        for episode in range(num_episodes):
            
            ep_rewards, ep_obs, ep_lprobs = run_episode(env, policy_network)

            ep_return = sum(ep_rewards)
            ep_length = len(ep_rewards)
            ep_rewards_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()

            batch_returns.append(ep_return)
            batch_lengths.append(ep_length)
            batch_obs += ep_obs
            batch_lprobs += ep_lprobs

            batch_future_returns += ep_rewards_to_go
            batch_full_returns += [ep_return] * ep_length

        print(f"\tUpdating network parameters...")
        if weight_kind in ['r']:
            batch_weights = batch_full_returns
        elif weight_kind in ['rtg','rtgv']:
            batch_weights = batch_future_returns

        if weight_kind in ['r','rtg']:
            batch_baselines = [0] * len(batch_obs)
        elif weight_kind in ['rtgv']:
            print(f"\tTraining value function network...")
            value_network = ValueMLP(num_features, value_hidden_size, 1).to(device)
            value_optimizer = torch.optim.Adam(value_network.parameters(), lr=value_learning_rate)
            train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_future_returns)
            test_loss_info, train_loss_info = train_value_network(
                value_network, nn.MSELoss(), value_optimizer, 
                train_dataloader, test_dataloader, num_epochs_value_network, n_updates=-1
            )
            value_network.eval()
            with torch.no_grad():
                batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
                b_unbounded = value_network(batch_obs_tensor)
                batch_baselines = torch.maximum(b_unbounded, torch.zeros_like(b_unbounded)).squeeze().tolist()

            # inter_epoch_train_loss = train_loss_info[0]
            # inter_epoch_train_steps = train_loss_info[1]
            # for loss, sub_epoch in zip(inter_epoch_train_loss, inter_epoch_train_steps):
            #     n = inter_epoch_train_steps[-1]
            #     writer.add_scalar("Inter_Epoch/Train_Loss", loss, epoch + sub_epoch / (n+1))
            # inter_epoch_test_loss = test_loss_info[0]
            # inter_epoch_test_steps = test_loss_info[1]
            # for loss, sub_epoch in zip(inter_epoch_test_loss, inter_epoch_test_steps):
            #     n = inter_epoch_test_steps[-1]
            #     writer.add_scalar("Inter_Epoch/Test_Loss", loss, epoch + sub_epoch / (n+1))

        batch_weighted_lprobs = [lp * (r - b) for lp, r, b in zip(batch_lprobs, batch_weights, batch_baselines)]
        if avg_kind == 'a':
            batch_loss = -sum(batch_weighted_lprobs) / len(batch_weighted_lprobs)
        elif avg_kind == 't':
            batch_loss = -sum(batch_weighted_lprobs) / num_episodes

        # compute a variance proxy for the policy gradient estimates
        weights = [r - b for r, b in zip(batch_weights, batch_baselines)]
        var_weights = float(np.var(weights))

        batch_avg_return = np.mean(batch_returns)
        writer.add_scalar("Metrics/Weight_Variance", var_weights, epoch)
        writer.add_scalar("Metrics/Avg_Episode_Return", batch_avg_return, epoch)
        writer.add_histogram('Episode_Returns_Distribution', np.array(batch_returns), epoch)
        
        policy_network.train()
        policy_optimizer.zero_grad()
        batch_loss.backward()
        policy_optimizer.step()

        if (epoch+1) % log_params_every_n_epochs == 0:
            print(f"\tLogging network params info...")
            for name, param in policy_network.named_parameters():
                writer.add_histogram(f'Policy_Param_Values/{name}', param.data, epoch)
                writer.add_scalar(f'Policy_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Policy_Param_Grads/{name}', param.grad, epoch)
                    writer.add_scalar(f'Policy_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)

        if (epoch+1) % render_every_n_epochs == 0:
            print(f"\tVisualizing episode...")
            render_epsiode(env_name, policy_network)

        if epoch % 1 == 0:
            print(f"\tAvg return = {batch_avg_return:.2f}")

    writer.flush()
    writer.close()
    env.close()

# %%
train_agent(config)


# %%
