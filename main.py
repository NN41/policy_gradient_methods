# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

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

def run_episode(env, policy_network):

    observation, info = env.reset()
    ep_rewards = []
    ep_obs = []
    ep_lprobs = []
    episode_done = False

    while not episode_done:

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

num_episodes = 500

policy_hidden_size = 4
policy_learning_rate = 0.01
num_epochs_policy_network = 100

value_hidden_size = 4
value_learning_rate = 0.001
num_epochs_value_network = 5

# weight_kind = 'rtg' # 'r' (reward) or 'rtg' (reward-to-go) or 'rtgv' (reward-to-go with value function baseline)
# avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 
# assert weight_kind in ['r','rtg','rtgv'], "weight is of the wrong kind"
# assert avg_kind in ['a','t']

return_kind_v2 = 'rtg' # 'r' or 'rtg'
baseline_kind_v2 = 'v' # 'v' for 'on-policy value function', or None
weight_kind_v3 = 'r' # or 'rtg' or 'rtgv'
avg_kind_v2 = 'a' # 'a' for 'all' and 't' for 'trajectories' 

policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)

for epoch in range(num_epochs_policy_network):

    batch_returns = []
    batch_lengths = []
    batch_lprobs = []
    batch_obs = []
    batch_rewards = []
    batch_rewards_to_go = []

    batch_returns_v2 = []
    batch_returns_to_go_v2 = []
    batch_weights_v2 = []

    policy_network.eval()
    for episode in range(num_episodes):
        
        ep_rewards, ep_obs, ep_lprobs = run_episode(env, policy_network)

        ep_return = sum(ep_rewards)
        ep_length = len(ep_rewards)
        ep_rewards_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()

        batch_returns.append(ep_return)
        batch_lengths.append(ep_length)
        batch_obs += ep_obs
        batch_lprobs += ep_lprobs

        batch_returns_to_go_v2 += ep_rewards_to_go
        batch_returns_v2 += [ep_return] * ep_length

        # if weight_kind in ['r']:
        #     batch_rewards += [ep_return] * ep_length
        # elif weight_kind in ['rtg','rtgv']:
        #     batch_rewards_to_go += ep_rewards_to_go
        #     batch_rewards += ep_rewards_to_go

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Avg return = {np.mean(batch_returns):.2f} over {num_episodes} episodes")

    if weight_kind_v3 in ['r']:
        batch_weights_v2 = batch_returns_v2
    elif weight_kind_v3 in ['rtg','rtgv']:
        batch_weights_v2 = batch_returns_to_go_v2

    if weight_kind_v3 in ['r','rtg']:
        batch_baselines_v2 = [0] * len(batch_obs)
    elif weight_kind_v3 in ['rtgv']:
        value_network = ValueMLP(num_features, value_hidden_size, 1).to(device)
        value_optimizer = torch.optim.Adam(value_network.parameters(), lr=value_learning_rate)
        train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_returns_to_go_v2)
        train_value_function_network(value_network, nn.MSELoss(), value_optimizer, train_dataloader, test_dataloader, num_epochs_value_network)
        value_network.eval()
        with torch.no_grad():
            batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
            batch_baselines_v2 = value_network(batch_obs_tensor).squeeze().tolist()

    batch_weighted_lprobs_v2 = [lp * (r - b) for lp, r, b in zip(batch_lprobs, batch_weights_v2, batch_baselines_v2)]
    if avg_kind_v2 == 'a':
        batch_loss_v2 = -sum(batch_weighted_lprobs_v2) / len(batch_weighted_lprobs_v2)
    elif avg_kind_v2 == 't':
        batch_loss_v2 = -sum(batch_weighted_lprobs_v2) / num_episodes

    batch_loss = batch_loss_v2

    # batch_baselines = [0] * len(batch_obs)
    # if weight_kind in ['rtgv']:

    #     value_network = ValueMLP(num_features, 32, 1).to(device)
    #     value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.0001)

    #     # train network on state observations and corresponding rewards-to-go of current epoch,
    #     # starting with previous epoch's parameters. Network approximates on-policy value function
    #     train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_rewards_to_go)
    #     train_value_function_network(value_network, nn.MSELoss(), value_optimizer, train_dataloader, test_dataloader, num_epochs_value_network)
        
    #     # compute baseline values corresponding to state observations of current epoch
    #     value_network.eval()
    #     with torch.no_grad():
    #         batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
    #         batch_baselines = value_network(batch_obs_tensor).squeeze().tolist()
        
    #     # # to remove
    #     # print(f"\tvalue func: min {np.min(batch_baselines):.1f}, max {np.max(batch_baselines):.3f}, mean {np.mean(batch_baselines):.3f}")
    #     # batch_baselines_test = batch_baselines

    # # compute the batch loss, using reward-to-go, on-policy value functiona as baseline, and log-probs
    # batch_weighted_lprobs = [(reward - base) * log_prob for reward, base, log_prob in zip(batch_rewards, batch_baselines, batch_lprobs)]
    # batch_loss = -sum(batch_weighted_lprobs) / len(batch_weighted_lprobs)

    # # batch_weighted_lprobs_test = [(reward - base) * log_prob for reward, base, log_prob in zip(batch_rewards, batch_baselines_test, batch_lprobs)]
    # print(f"\tvar of weighted_lprobs      = {np.var([x.item() for x in batch_weighted_lprobs]):.2f}")
    # # print(f"\tvar of weighted_lprobs test = {np.var([x.item() for x in batch_weighted_lprobs_test]):.2f}")
    # # print(f"\tavg l1 loss of value network on test  = {test(value_network, nn.L1Loss(), test_dataloader)[0]:.2f}")
    # # print(f"\tavg l1 loss of value network on train = {test(value_network, nn.L1Loss(), train_dataloader)[0]:.2f}")
    
    policy_network.train()
    policy_optimizer.zero_grad()
    batch_loss.backward()
    policy_optimizer.step()

# %%

test(value_network, nn.MSELoss(), test_dataloader)
train(value_network, nn.MSELoss(), value_optimizer, train_dataloader, n_updates=10)