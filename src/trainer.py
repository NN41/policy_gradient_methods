#%%

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def create_dataloaders_for_value_network(batch_observations, batch_future_returns):
    X = torch.tensor(batch_observations, dtype=torch.float32)
    y = torch.tensor(batch_future_returns, dtype=torch.float32)
    full_dataset = TensorDataset(X, y)
    train_dataset, test_dataset = random_split(full_dataset, [0.8, 0.2])
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_dataloader, test_dataloader

def test(model, loss, test_dataloader):
    loss_total = 0
    n_samples = 0
    model.eval()
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X).squeeze()
            n_samples += len(y)
            loss_total += loss(y_pred, y).item() * len(y)
    avg_loss = loss_total / n_samples
    return avg_loss

def train(model, loss, optimizer, train_dataloader, n_updates=0):

    n_batches = len(train_dataloader)
    total_loss_update = 0
    total_loss = 0
    n_samples_update = 0
    n_samples = 0

    if n_updates == -1:
        n_updates = n_batches
    update_batches = np.linspace(-1, n_batches-1, min(n_batches,n_updates)+1, dtype=int)[1:]
    avg_loss_updates = []

    model.train()
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)

        # forward pass
        y_pred = model(X).squeeze()
        batch_loss = loss(y_pred, y)

        # backward pass
        optimizer.zero_grad()
        batch_loss.backward()
        # nn.utils.clip_grad_norm_(model.parameters(), 3)
        optimizer.step()

        n_samples += len(y)
        total_loss += batch_loss.item() * len(y)

        n_samples_update += len(y)
        total_loss_update += batch_loss.item() * len(y)
        if batch in update_batches:
            avg_loss_update = total_loss_update / n_samples_update
            avg_loss_updates.append(avg_loss_update)
            print(f"\t{batch+1} / {n_batches} | avg train loss {avg_loss_update:.5f}")
            n_samples_update = 0
            total_loss_update = 0
    
    avg_loss = total_loss / n_samples
    return avg_loss, (avg_loss_updates, update_batches.tolist())

def train_value_network(model, loss, optimizer, train_dataloader, test_dataloader, n_epochs, n_updates=0):

    test_losses = []
    train_losses = []

    if n_updates != 0:
        test_loss = test(model, loss, test_dataloader)
        print(f"\t\t(Value Net) Epoch {0} / {n_epochs} | train loss = NaN    | test loss = {test_loss:.5f}")

    if n_updates == -1:
        n_updates = n_epochs
    update_epochs = np.linspace(0, n_epochs-1, min(n_epochs,n_updates), dtype=int)

    for epoch in range(n_epochs):
        train_loss, _ = train(model, loss, optimizer, train_dataloader, n_updates=0)
        train_losses.append(train_loss)
        if epoch in update_epochs:
            test_loss = test(model, loss, test_dataloader)
            test_losses.append(test_loss)
            print(f"\t\t(Value Net) Epoch {epoch+1} / {n_epochs} | train loss = {train_loss:.5f} | test loss = {test_loss:.5f}")

    test_loss_info = (test_losses, update_epochs.tolist())
    train_loss_info = (train_losses, range(n_epochs))
    return test_loss_info, train_loss_info

# %%

from src.utils import Config, compute_discounted_future_returns, compute_gaes
from src.agent import Agent
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np


class Trainer():
    def __init__(self, env: gym.Env, agent: Agent, config: Config):
        self.agent = agent
        self.config = config
        self.env = env
        self.writer = SummaryWriter(config.log_dir)

    def _run_episode(self) -> dict:

        episode_data = {
            'rewards': [],
            'observations': [],
            'log_probs': []
        }

        episode_done = False
        observation, info = self.env.reset()

        while not episode_done:

            # if render:
            #     self.env.render()

            # get action from agent, act in environment and determine if epsisode is over
            action, log_prob = self.agent.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            episode_done = terminated or truncated

            # log data
            episode_data['rewards'].append(reward)
            episode_data['observations'].append(observation.tolist())
            episode_data['log_probs'].append(log_prob)

        return episode_data
    
    def _render_episode(self):
        """Visualize a single episode"""
        render_env = gym.make(self.config.env_name, render_mode='human')
        obs, info = render_env.reset()
        done = False
        while not done:
            act, lprob = self.agent.select_action(obs, inference_mode=True)
            obs, rew, term, trunc, info = render_env.step(act)
            done = term or trunc
        render_env.close()
        
    def _collect_batch(self) -> tuple[dict, dict]:

        gamma_gae = self.config.gamma_gae
        lambda_gae = self.config.lambda_gae
        value_network = self.agent.value_network

        # Reset batch data
        batch_returns = []
        batch_lengths = []
        batch_full_returns = []
        batch_fut_returns = []
        batch_disc_fut_returns = []
        batch_gaes = []
        batch_td_errors = []
        batch_dfr_baseline = []
        batch_obs = []
        batch_lprobs = []

        print(f"\tSimulating {self.config.num_episodes} episodes...")
        for ep in range(self.config.num_episodes):

            episode_data = self._run_episode()
            ep_rewards = episode_data['rewards']
            ep_obs = episode_data['observations']
            ep_lprobs = episode_data['log_probs']

            # collect episode statistics
            ep_return = sum(ep_rewards)
            ep_length = len(ep_rewards)
            ep_full_returns = [ep_return] * ep_length
            ep_fut_returns = compute_discounted_future_returns(ep_rewards, 1)
            ep_disc_fut_returns = compute_discounted_future_returns(ep_rewards, gamma_gae)
            ep_gaes = compute_gaes(ep_obs, ep_rewards, gamma_gae, lambda_gae, value_network, set_to_zero=False)
            ep_td_errors = compute_gaes(ep_obs, ep_rewards, gamma_gae, 0, value_network, set_to_zero=False)
            ep_dfr_baseline = compute_gaes(ep_obs, ep_rewards, gamma_gae, 1, value_network, set_to_zero=False)
            # value_network.eval()
            # with torch.no_grad():
            #     ep_vals = value_network(torch.tensor(ep_obs, dtype=torch.float32).to(device)).squeeze().cpu().numpy().tolist()
            #     ep_vals[-1] = 0 # Value of being in the last (terminated) state is 0. This is ensures that GAEs with lambda = 1 indeed match disc future returns minus value baseline

            batch_returns.append(ep_return)
            batch_lengths.append(ep_length)
            batch_full_returns += ep_full_returns
            batch_fut_returns += ep_fut_returns
            batch_disc_fut_returns += ep_disc_fut_returns
            batch_gaes += ep_gaes
            batch_td_errors += ep_td_errors
            batch_dfr_baseline += ep_dfr_baseline
            batch_obs += ep_obs
            batch_lprobs += ep_lprobs

        weight_kind = self.config.weight_kind
        if weight_kind in ['r']:
            batch_weights = batch_full_returns
        elif weight_kind in ['fr']:
            batch_weights = batch_fut_returns
        elif weight_kind in ['dfr']:
            batch_weights = batch_disc_fut_returns
        elif weight_kind in ['gae']:
            batch_weights = batch_gaes
        elif weight_kind in ['td']:
            batch_weights = batch_td_errors
        elif weight_kind in ['dfrb']:
            batch_weights = batch_dfr_baseline

        metrics_to_log = {
            'Distributions/Episode_Returns': np.array(batch_returns),
            'Metrics/Avg_Episode_Return': np.mean(batch_returns),
            'Metrics/Avg_Episode_Length': np.mean(batch_lengths),
            'Weight_Variances/Full_Returns': float(np.var(batch_full_returns)),
            'Weight_Variances/Future_Returns': float(np.var(batch_fut_returns)),
            'Weight_Variances/Disc_Future_Returns': float(np.var(batch_disc_fut_returns)),
            'Weight_Variances/GAEs': float(np.var(batch_gaes)),
            'Weight_Variances/TD_Errors': float(np.var(batch_td_errors)),
            'Weight_Variances/DFR_Baseline': float(np.var(batch_dfr_baseline)),
        }

        batch_data = {
            'obs': batch_obs,
            'lprobs': batch_lprobs,
            'disc_fut_returns': batch_disc_fut_returns,
            'weights': batch_weights
        }

        return batch_data, metrics_to_log
    
    def _train_one_epoch(self):
        
        # simulate multiple episodes to collect batch training data
        batch, metrics_to_log = self._collect_batch()
        
        # compute loss and update network policy
        print(f"\tUpdating policy...")
        batch_loss = self._compute_policy_loss(batch)
        self.agent.update_policy_network(batch_loss)

        if self.config.weight_kind in ['gae', 'td', 'dfrb']:
            print(f"\tTraining value function...")
            train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch['obs'], batch['disc_fut_returns'])
            self.agent.reset_value_optimizer()
            test_loss_info, train_loss_info = train_value_network(self.agent.value_network, nn.MSELoss(), self.agent.value_optimizer, train_dataloader, test_dataloader, self.config.num_epochs_value_network, n_updates=-1)

        return metrics_to_log

    def _compute_policy_loss(self, batch):
        """Compute batch loss for updating policy network"""
        policy_gradient_terms = [lp * w for lp, w in zip(batch['lprobs'], batch['weights'])]
        if self.config.avg_kind == 'a': # take sample mean over all state-action pairs
            loss = -sum(policy_gradient_terms) / len(policy_gradient_terms)
        elif self.config.avg_kind == 't': # take sample mean only wrt the number of trajectories
            loss = -sum(policy_gradient_terms) / self.config.num_episodes
        return loss

    def _log_metrics(self, metrics: dict, epoch: int):
        """Log all metrics from the training process to TensorBoard."""

        # Log metrics from the batch data collection
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                self.writer.add_histogram(key, value, epoch)
            else:
                self.writer.add_scalar(key, value, epoch)

        # Log network parameters periodically
        if (epoch+1) % self.config.log_params_every_n_epochs == 0:
            print(f"\tLogging network params info...")
            for name, param in self.agent.policy_network.named_parameters():
                self.writer.add_histogram(f'Policy_Param_Values/{name}', param.data, epoch)
                self.writer.add_scalar(f'Policy_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Policy_Param_Grads/{name}', param.grad, epoch)
                    self.writer.add_scalar(f'Policy_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)
            for name, param in self.agent.value_network.named_parameters():
                self.writer.add_histogram(f'Value_Param_Values/{name}', param.data, epoch)
                self.writer.add_scalar(f'Value_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    self.writer.add_histogram(f'Value_Param_Grads/{name}', param.grad, epoch)
                    self.writer.add_scalar(f'Value_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)

    def train(self):

        for epoch in range(self.config.num_epochs_policy_network):
            print(f"(Policy Net) Epoch {epoch+1} / {self.config.num_epochs_policy_network}:")

            # Train one epoch and log metrics related to training and network parameters
            metrics_to_log = self._train_one_epoch()
            self._log_metrics(metrics_to_log, epoch)

            # Visualize an episode
            if (epoch+1) % self.config.render_every_n_epochs == 0:
                print(f"\tVisualizing episode...")
                self._render_episode()

            avg_batch_return = metrics_to_log['Metrics/Avg_Episode_Return']
            avg_batch_length = metrics_to_log['Metrics/Avg_Episode_Length']
            print(f"\tAvg return: {avg_batch_return:.2f} | Avg length: {avg_batch_length:.2f}")

        self.writer.flush()
        self.writer.close()
        self.env.close()

# config = Config(weight_kind='fr', avg_kind='a')
# agent = Agent(4,2,config)
# trainer = Trainer(agent, config)
# trainer.train()

#%%
if __name__ == '__main__':
    
    from src.networks import ValueMLP

    print(f"Using {device} device")

    batch_size = 500
    model = ValueMLP(4,2,1).to(device)
    batch_features = np.random.rand(batch_size,4)
    batch_targets = np.random.rand(batch_size,)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_features, batch_targets)

    assert model(torch.rand(100,4).to(device)).shape == (100,1), "Failed output shape test"
    assert isinstance(test(model, loss, test_dataloader), float), 'Failed "test" function test'
    train(model, loss, optimizer, train_dataloader, n_updates=-1)
    train_value_network(model, loss, optimizer, train_dataloader, test_dataloader, n_epochs=7, n_updates=-1);

    