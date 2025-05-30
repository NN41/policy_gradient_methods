# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from src.policy import PolicyMLP
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

# %%
env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions {num_actions}")

policy_network = PolicyMLP(num_features, 4, num_actions)
optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

policy_network.train()

num_episodes = 100
num_epochs = 1

for epoch in range(num_epochs):

    batch_returns = []
    batch_lengths = []
    batch_weights = []
    batch_lprobs = []
    batch_obs = []
    batch_rewards_to_go = []

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
            batch_obs.append(observation.tolist())
            batch_lprobs.append(log_prob)
            episode_done = terminated or truncated

        ep_return = sum(ep_rewards)
        ep_rewards_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()
        ep_length = len(ep_rewards)
        batch_returns.append(ep_return)
        batch_lengths.append(ep_length)
        batch_weights += ep_rewards_to_go
        # batch_weights += [ep_return] * ep_length
        batch_rewards_to_go += ep_rewards_to_go

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Avg return = {np.mean(batch_returns):.2f} over {len(batch_returns)} episodes")

    batch_weighted_lprobs = [weight * lprob for weight, lprob in zip(batch_weights, batch_lprobs)]
    batch_criterion = -sum(batch_weighted_lprobs) / len(batch_weighted_lprobs)

    optimizer.zero_grad()
    batch_criterion.backward()
    optimizer.step()

#%%

X = torch.tensor(batch_obs, dtype=torch.float32)
y = torch.tensor(batch_rewards_to_go, dtype=torch.float32)



full_dataset = TensorDataset(X, y)

train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)




# %%


def test(model, criterion, test_dataloader):
    loss_total = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in enumerate(test_dataloader):
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            n_samples += len(y)
            loss_total += criterion(y_pred, y).item() * len(y)
    avg_loss = loss_total / n_samples
    return avg_loss, 0


def train(model, criterion, optimizer, n_updates=0):

    n_batches = len(train_dataloader)
    # n_updates = 3
    update_batches = np.linspace(0, n_batches-1, n_updates).astype(int)
    total_loss_between_updates = 0
    total_loss = 0
    n_samples_between_updates = 0
    n_samples = 0

    # if n_updates > 0:
    #     print(f"[batch] / {n_batches} | [avg train loss between updates]")
    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X).squeeze()
        loss = criterion(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        n_samples_between_updates += len(y)
        n_samples += len(y)
        total_loss += loss.item() * len(y)
        total_loss_between_updates += loss.item() * len(y)
        if batch in update_batches:
            avg_loss_between_updates = total_loss_between_updates / n_samples_between_updates
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_between_updates:.5f}")
            n_samples_between_updates = 0
            total_loss_between_updates = 0
    
    avg_train_loss = total_loss / n_samples
    return avg_train_loss


# %% 
value_network = PolicyMLP(num_features, 16, 1).to(device)
criterion = nn.MSELoss()
optimizer_val_func = torch.optim.Adam(value_network.parameters(), lr=0.01)


model = value_network
optimizer = optimizer_val_func

EPOCHS = 100

test_loss, test_acc = test(model, criterion, test_dataloader)
print(f"avg test loss = {test_loss:.5f} | accuracy = {test_acc * 100:.2f}%")

# logs = []
# logs_flattened = []
# initial_log = {
#     "epoch": 0,
#     "test_loss": test_loss,
#     "test_acc": test_acc,
#     "train_loss": None,
#     "": get_all_param_metrics(model),
# }
# logs.append(initial_log)
# logs_flattened.append(flatten_nested_dict(initial_log, separator='/'))

for epoch in range(EPOCHS):
    print(f"\nepoch {epoch+1} / {EPOCHS}")

    train_loss = train(model, criterion, optimizer, n_updates=0)

    test_loss, test_acc = test(model, criterion, test_dataloader)
    print(f"avg test loss = {test_loss:.5f} | accuracy = {test_acc * 100:.2f}%")

    # log = {
    #     "epoch": epoch+1,
    #     "test_loss": test_loss,
    #     "test_acc": test_acc,
    #     "train_loss": train_loss,
    #     "": get_all_param_metrics(model),
    # }
    # logs.append(log)
    # logs_flattened.append(flatten_nested_dict(log, separator='/'))