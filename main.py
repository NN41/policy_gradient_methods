# %% 

import gymnasium as gym

# %%

env_name = "CartPole-v1"
env = gym.make(env_name, render_mode="human")

action_space = env.action_space
num_actions = action_space.n # for discrete spaces

observation_space = env.observation_space
assert len(observation_space.shape) == 1 and isinstance(observation_space.shape, tuple)
num_features = observation_space.shape[0]


# %%

# Let's build the policy network

import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using device {device}")

class PolicyMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = F.relu(self.fc1(x))
        logits = self.fc2(hidden)
        return logits

policy = PolicyMLP(num_features, 16, num_actions)
X_batch = torch.rand(20, num_features)
X_sample = torch.rand(num_features)
policy.eval()
with torch.no_grad():
    logits_batch = policy(X_batch)
    logits_sample = policy(X_sample)
print(logits_batch, logits_sample)



# %%

state, info = env.reset(seed=42)

for t in range(1000):
    # my policy network should choose an action here using softmax
    action = env.action_space.sample()

    state, reward, terminated, truncated, info = env.step(action)
    # print(observation, reward, terminated, truncated, info)

    if terminated or truncated:
        state, info = env.reset(seed=42)

env.close()

