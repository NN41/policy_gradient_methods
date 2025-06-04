# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from src.networks import PolicyMLP, ValueMLP
from src.training import create_dataloaders_for_value_network, test, train, train_value_function_network

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

# %%

env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions: {num_actions}")

# %%

def run_episode(env, policy_network, render=False):

    observation, info = env.reset()
    ep_rewards = []
    ep_obs = []
    ep_lprobs = []
    episode_done = False

    while not episode_done:

        if render:
            env.render()
            time.sleep(0.02)

        # get probabilities from policy
        logits = policy_network(torch.tensor(observation, dtype=torch.float32).to(device))
        probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based and compute log-probability
        idx = np.random.choice(range(num_actions), p=probs.detach().cpu().numpy())
        action = idx
        log_prob = torch.log(probs[action])

        # act in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # log data and determine if episode is over
        ep_rewards.append(reward)
        ep_obs.append(observation.tolist())
        ep_lprobs.append(log_prob)
        # batch_obs.append(observation.tolist())
        # batch_lprobs.append(log_prob)
        episode_done = terminated or truncated

    return ep_rewards, ep_obs, ep_lprobs

def render_epsiode(env_name, policy_network):
    render_env = gym.make(env_name, render_mode='human')
    policy_network.eval()
    with torch.no_grad():
        run_episode(render_env, policy_network, render=True)
    render_env.close()

# %%

num_episodes = 100
num_epochs_policy_network = 50
render_every_n_epochs = 500
log_params_every_n_epochs = 3

policy_hidden_size = 4
policy_learning_rate = 0.01

value_hidden_size = 4
value_learning_rate = 0.001
num_epochs_value_network = 5

weight_kind = 'rtg' # or 'rtg' or 'rtgv'
avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 

policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)

# logs = []
# initial_logs = {
#     "epoch": 0,
#     "var_weights": 0
# }
# logs.append(initial_logs)
logs = {
    "epoch": [0],
    "var_weights": [None]
}

writer = SummaryWriter()
# log_dir = "runs/test"
# writer = SummaryWriter(log_dir)

for epoch in range(num_epochs_policy_network):

    print(f"Epoch {epoch+1} / {num_epochs_policy_network} (Policy Network)")

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
    print(f"\t\tSimulating {num_episodes} episodes...")
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

    print(f"\t\tUpdating network parameters...")
    if weight_kind in ['r']:
        batch_weights = batch_full_returns
    elif weight_kind in ['rtg','rtgv']:
        batch_weights = batch_future_returns

    if weight_kind in ['r','rtg']:
        batch_baselines = [0] * len(batch_obs)
    elif weight_kind in ['rtgv']:
        value_network = ValueMLP(num_features, value_hidden_size, 1).to(device)
        value_optimizer = torch.optim.Adam(value_network.parameters(), lr=value_learning_rate)
        train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_future_returns)
        train_value_function_network(value_network, nn.MSELoss(), value_optimizer, train_dataloader, test_dataloader, num_epochs_value_network)
        value_network.eval()
        with torch.no_grad():
            batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
            batch_baselines = value_network(batch_obs_tensor).squeeze().tolist()

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
        print(f"\t\tLogging network params info...")
        for name, param in policy_network.named_parameters():
            writer.add_histogram(f'Policy_Param_Values/{name}', param.data, epoch)
            writer.add_scalar(f'Policy_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
            if param.grad is not None:
                writer.add_histogram(f'Policy_Param_Grads/{name}', param.grad, epoch)
                writer.add_scalar(f'Policy_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)

    if (epoch+1) % render_every_n_epochs == 0:
        print(f"\t\tVisualizing episode...")
        render_epsiode(env_name, policy_network)

    if epoch % 1 == 0:
        print(f"\tAvg return = {batch_avg_return:.2f}")

    logs["epoch"].append(epoch+1)
    logs["var_weights"].append(var_weights)

writer.flush()

env.close()
writer.close()

# %%


for n, p in policy_network.named_parameters():
    print(n)

p.data, p.data.norm().item()

# %%
print("\n--- Saving the trained policy network ---")

model_name = "policy_network"
models_dir = "results\models"
os.makedirs(models_dir, exist_ok=True) 
current_date = datetime.now().strftime("%m-%d")
model_filename = f"{model_name}_{env_name}_{current_date}.pth" 
model_filepath = os.path.join(models_dir, model_filename)

# Save the model's state_dict (learned parameters)
torch.save(policy_network.state_dict(), model_filepath)
print(f"Model saved to: {model_filepath}")

# Get and print the size of the saved file
file_size_bytes = os.path.getsize(model_filepath)
file_size_kb = file_size_bytes / 1024
print(f"File size: {file_size_kb:.2f} KB")

# %%
# # Assuming PolicyMLP is defined and num_features, policy_hidden_size, num_actions are known
# loaded_policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
# loaded_policy_network.load_state_dict(torch.load(model_filepath))
# loaded_policy_network.eval() # Set to evaluation mode after loading
