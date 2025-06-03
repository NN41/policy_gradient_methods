# %% 

import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np

from src.networks import PolicyMLP, ValueMLP
from torch.utils.data import DataLoader, TensorDataset, random_split

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

# %%

def create_dataloaders_for_value_network(batch_observations, batch_rewards_to_go):
    X = torch.tensor(batch_observations, dtype=torch.float32)
    y = torch.tensor(batch_rewards_to_go, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader

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

def train(model, criterion, optimizer, train_dataloader, n_updates):

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

def train_value_function_network(model, criterion, optimizer, train_dataloader, test_dataloader, EPOCHS, verbose=False):


    test_loss, test_acc = test(model, criterion, test_dataloader)
    print(f"\tepoch {0} / {EPOCHS} | avg test loss = {test_loss:.5f}")

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

        train_loss = train(model, criterion, optimizer, train_dataloader, n_updates=0)

        if epoch % 1 == 0:
            test_loss, test_acc = test(model, criterion, test_dataloader)
            print(f"\tepoch {epoch+1} / {EPOCHS} | avg test loss = {test_loss:.5f}")
            if verbose:
                print(f"\t\ttrain loss {train_loss}")

        # log = {
        #     "epoch": epoch+1,
        #     "test_loss": test_loss,
        #     "test_acc": test_acc,
        #     "train_loss": train_loss,
        #     "": get_all_param_metrics(model),
        # }
        # logs.append(log)
        # logs_flattened.append(flatten_nested_dict(log, separator='/'))

# %%

env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions: {num_actions}")


# %%

policy_network = PolicyMLP(num_features, 16, num_actions).to(device)
policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=0.01)

weight_kind = 'rtgv' # 'r' (reward) or 'rtg' (reward-to-go) or 'rtgv' (reward-to-go with value function baseline)
assert weight_kind in ['r','rtg','rtgv'], "weight is of the wrong kind"
num_episodes = 1000
num_epochs_policy_network = 100
num_epochs_value_network = 1

zero_weight = 0

for epoch in range(num_epochs_policy_network):

    batch_returns = []
    batch_lengths = []
    batch_lprobs = []
    batch_obs = []
    batch_rewards = []
    batch_rewards_to_go = []

    for episode in range(num_episodes):

        # if episode % 100 == 0:
        #     print(f"episode {episode+1}")
        
        observation, info = env.reset()
        ep_rewards = []
        episode_done = False

        policy_network.eval()
        while not episode_done:

            # get probabilities from policy
            logits = policy_network(torch.tensor(observation, dtype=torch.float32).to(device))
            probs = nn.Softmax(dim=-1)(logits)
            
            # randomly choose an action based on policy probabilities
            idx = np.random.choice(range(num_actions), p=probs.detach().cpu().numpy())
            action = idx
            # action = env.action_space.sample()

            log_prob = torch.log(probs[action])

            observation, reward, terminated, truncated, info = env.step(action)

            ep_rewards.append(reward)
            batch_obs.append(observation.tolist())
            batch_lprobs.append(log_prob)
            episode_done = terminated or truncated

        ep_return = sum(ep_rewards)
        ep_length = len(ep_rewards)
        batch_returns.append(ep_return)
        batch_lengths.append(ep_length)

        if weight_kind in ['r']:
            batch_rewards += [ep_return] * ep_length
        elif weight_kind in ['rtg','rtgv']:
            ep_rewards_to_go = np.cumsum(ep_rewards[::-1])[::-1].tolist()
            batch_rewards_to_go += ep_rewards_to_go
            batch_rewards += ep_rewards_to_go

    if epoch % 1 == 0:
        print(f"Epoch {epoch+1} | Avg return = {np.mean(batch_returns):.2f} over {num_episodes} episodes")

    batch_baselines = [0] * len(batch_obs)
    if weight_kind in ['rtgv']:

        value_network = ValueMLP(num_features, 32, 1).to(device)
        value_optimizer = torch.optim.Adam(value_network.parameters(), lr=0.0001)

        # train network on state observations and corresponding rewards-to-go of current epoch,
        # starting with previous epoch's parameters. Network approximates on-policy value function
        train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_rewards_to_go)
        train_value_function_network(value_network, nn.MSELoss(), value_optimizer, train_dataloader, test_dataloader, num_epochs_value_network)
        
        # compute baseline values corresponding to state observations of current epoch
        value_network.eval()
        with torch.no_grad():
            batch_obs_tensor = torch.tensor(batch_obs, dtype=torch.float32).to(device)
            batch_baselines = value_network(batch_obs_tensor).squeeze().tolist() # some values are negative, which doesn't make sense
        
        # to remove
        print(f"\tvalue func: min {np.min(batch_baselines):.1f}, max {np.max(batch_baselines):.3f}, mean {np.mean(batch_baselines):.3f}")
        batch_baselines_test = batch_baselines
        batch_baselines = (1-zero_weight) * np.array(batch_baselines)

    # compute the batch loss, using reward-to-go, on-policy value functiona as baseline, and log-probs
    batch_weighted_lprobs = [(reward - base) * log_prob for reward, base, log_prob in zip(batch_rewards, batch_baselines, batch_lprobs)]
    batch_loss = -sum(batch_weighted_lprobs) / len(batch_weighted_lprobs)

    batch_weighted_lprobs_test = [(reward - base) * log_prob for reward, base, log_prob in zip(batch_rewards, batch_baselines_test, batch_lprobs)]
    print(f"\tvar of weighted_lprobs      = {np.var([x.item() for x in batch_weighted_lprobs]):.2f}")
    print(f"\tvar of weighted_lprobs test = {np.var([x.item() for x in batch_weighted_lprobs_test]):.2f}")
    print(f"\tavg l1 loss of value network on test  = {test(value_network, nn.L1Loss(), test_dataloader)[0]:.2f}")
    print(f"\tavg l1 loss of value network on train = {test(value_network, nn.L1Loss(), train_dataloader)[0]:.2f}")
    policy_network.train()
    policy_optimizer.zero_grad()
    batch_loss.backward()
    policy_optimizer.step()
