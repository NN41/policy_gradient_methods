# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from src.policy import PolicyMLP

# %%

env_name = "CartPole-v1"
env = gym.make(env_name)

action_space = env.action_space
num_actions = action_space.n # for discrete spaces

observation_space = env.observation_space
num_features = observation_space.shape[0]
assert len(observation_space.shape) == 1 and isinstance(observation_space.shape, tuple), "Observation space is of unexpected type/size"

env.close()

print(f"Number of elements in state: {num_features} | Number of actions {num_actions}")

# %%

def run_episodes(env, policy_network, num_episodes):

    state, info = env.reset()

    episode_length_list = []
    total_reward_list = []
    total_log_prob_list = []

    action_list = []

    episode_length = 0
    total_reward = 0
    total_log_prob = 0

    episode = 0
    while episode < num_episodes:

        episode_length += 1

        # get probabilities from policy
        logits = policy_network(torch.tensor(state))
        probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based on policy probabilities
        idx = np.random.choice(range(num_actions), p=probs.detach().numpy())
        action = idx
        # action = env.action_space.sample()

        action_list.append(action)

        # track log probs
        log_prob = probs[idx].log()
        total_log_prob += log_prob

        # apply action and collect rewards
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:

            # print(f"ep {episode+1} | tot reward {total_reward}")

            state, info = env.reset()
            episode += 1

            total_reward_list.append(total_reward)
            total_log_prob_list.append(total_log_prob)
            episode_length_list.append(episode_length)

            total_reward = 0
            total_log_prob = 0
            episode_length = 0

    print(f"Avg reward: {np.mean(total_reward_list)}")
    return total_log_prob_list, total_reward_list

# %%

policy_network = PolicyMLP(num_features, 8, num_actions)
optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.01)

env = gym.make(env_name, 
            #    render_mode='human'
)

policy_network.train()

num_episodes = 1000
num_updates = 100

for _ in range(num_updates):

    print(f"update {_} / {num_updates}")
    total_log_prob_list, total_reward_list = run_episodes(env, policy_network, num_episodes)

    criterion = (-1) * sum(x * y for x, y in zip(total_log_prob_list, total_reward_list)) / num_episodes

    optimizer.zero_grad()
    criterion.backward(retain_graph=True)
    optimizer.step()

env.close()

# %%

policy_network.train()

num_updates = 1000
num_episodes = 100

env = gym.make(env_name)
state, info = env.reset()
update = 0
while update < num_updates:

    episode_length_list = []
    total_reward_list = []
    total_log_prob_list = []

    episode = 0
    episode_length = 0
    total_reward = 0
    total_log_prob = 0

    while episode < num_episodes:

        episode_length += 1

        # get probabilities from policy
        logits = policy_network(torch.tensor(state))
        probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based on policy probabilities
        idx = np.random.choice(range(num_actions), p=probs.detach().numpy())
        action = idx
        # action = env.action_space.sample()

        # track log probs
        log_prob = probs[idx].log()
        total_log_prob += log_prob

        # apply action and collect rewards
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            state, info = env.reset()
            episode += 1

            total_reward_list.append(total_reward)
            total_log_prob_list.append(total_log_prob)
            episode_length_list.append(episode_length)

            total_reward = 0
            total_log_prob = 0
            episode_length = 0
    
    policy_gradient = sum(x * y for x, y in zip(total_reward_list, total_log_prob_list)) / num_episodes
    criterion = -policy_gradient

    optimizer.zero_grad()
    criterion.backward()
    optimizer.step()

    update += 1

    print(f"update step {update} | avg episode reward {np.mean(total_reward_list)}")

env.close()

# %%

total_reward_list