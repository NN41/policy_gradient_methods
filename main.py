# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
import time
import os
from datetime import datetime

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



# %%

num_episodes = 50
num_epochs_policy_network = 200
render_every_n_epochs = 5

policy_hidden_size = 4
policy_learning_rate = 0.01

value_hidden_size = 4
value_learning_rate = 0.001
num_epochs_value_network = 5

weight_kind = 'rtg' # or 'rtg' or 'rtgv'
avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 

policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)

for epoch in range(num_epochs_policy_network):

    print(f"Epoch {epoch+1} / {num_epochs_policy_network} (Policy Network)")

    # batch_full_returns = []
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

        batch_full_returns.append(ep_return)
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

    # # batch_weighted_lprobs_test = [(reward - base) * log_prob for reward, base, log_prob in zip(batch_rewards, batch_baselines_test, batch_lprobs)]
    # print(f"\tvar of weighted_lprobs      = {np.var([x.item() for x in batch_weighted_lprobs]):.2f}")
    # # print(f"\tvar of weighted_lprobs test = {np.var([x.item() for x in batch_weighted_lprobs_test]):.2f}")
    # # print(f"\tavg l1 loss of value network on test  = {test(value_network, nn.L1Loss(), test_dataloader)[0]:.2f}")
    # # print(f"\tavg l1 loss of value network on train = {test(value_network, nn.L1Loss(), train_dataloader)[0]:.2f}")
    
    policy_network.train()
    policy_optimizer.zero_grad()
    batch_loss.backward()
    policy_optimizer.step()

    if epoch % render_every_n_epochs == 0:
        print(f"\t\tVisualizing episode...")
        render_env = gym.make(env_name, render_mode='human')
        policy_network.eval()
        with torch.no_grad():
            run_episode(render_env, policy_network, render=True)
        render_env.close()

    if epoch % 1 == 0:
        print(f"\tAvg return = {np.mean(batch_full_returns):.2f}")

env.close()
# %%

render_env = gym.make(env_name, render_mode='human')
policy_network.eval()
with torch.no_grad():
    run_episode(render_env, policy_network, render=True)
render_env.close()


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
