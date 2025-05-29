# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from src.policy import PolicyMLP

# %%
env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions {num_actions}")

policy_network = PolicyMLP(num_features, 8, num_actions)
optimizer = torch.optim.SGD(policy_network.parameters(), lr=0.01)

policy_network.train()

num_episodes = 100
num_epochs = 50

for epoch in range(num_epochs):

    batch_returns = []
    batch_lengths = []
    batch_weights = []
    batch_lprobs = []

    for episode in range(num_episodes):

        # if episode % 100 == 0:
        #     print(f"episode {episode+1}")
        
        observation, info = env.reset()
        ep_rewards = []
        episode_done = False

        while not episode_done:

            # get probabilities from policy
            logits = policy_network(torch.tensor(observation))
            probs = nn.Softmax(dim=-1)(logits)
            
            # randomly choose an action based on policy probabilities
            idx = np.random.choice(range(num_actions), p=probs.detach().numpy())
            action = idx
            # action = env.action_space.sample()

            log_prob = torch.log(probs[action]) 

            observation, reward, terminated, truncated, info = env.step(action)

            ep_rewards.append(reward)
            batch_lprobs.append(log_prob)
            episode_done = terminated or truncated

        ep_return = sum(ep_rewards)
        ep_length = len(ep_rewards)
        batch_returns.append(ep_return)
        batch_lengths.append(ep_length)
        batch_weights += [ep_return] * ep_length

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Avg return = {np.mean(batch_returns):.2f} over {len(batch_returns)} episodes")

    batch_weighted_lprobs = [weight * lprob for weight, lprob in zip(batch_weights, batch_lprobs)]
    batch_criterion = -sum(batch_weighted_lprobs) / len(batch_weighted_lprobs)

    optimizer.zero_grad()
    batch_criterion.backward()
    optimizer.step()
