# %% 

import numpy as np
import os
from datetime import datetime

import gymnasium as gym
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# for reloading modules
from importlib import reload
import src.networks, src.utils, src.training
reload(src.utils)
reload(src.training)
reload(src.networks)

from src.networks import PolicyMLP, ValueMLP
from src.training import create_dataloaders_for_value_network, train_value_network
from src.utils import run_episode, render_epsiode, compute_discounted_future_returns, compute_gaes, compute_td_errors

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")

env_name = "CartPole-v1"
env = gym.make(env_name)

num_actions = env.action_space.n # for discrete spaces
num_features = env.observation_space.shape[0]
print(f"Number of elements in observation: {num_features} | Number of actions: {num_actions}")

# %%



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
    gamma_gae = config.gamma_gae
    lambda_gae = config.lambda_gae
    log_dir = config.log_dir
    # log_dir = 'runs\\test'

    writer = SummaryWriter(log_dir)

    policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=policy_learning_rate)

    value_network = ValueMLP(num_features, value_hidden_size, 1).to(device)

    for epoch in range(num_epochs_policy_network):
        print(f"(Policy Net) Epoch {epoch+1} / {num_epochs_policy_network}:")

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

        var_full_returns = 0
        var_fut_returns = 0
        var_disc_fut_returns = 0
        var_gaes = 0
        var_td_errors = 0
        var_dfr_baseline = 0

        print(f"\tSimulating {num_episodes} episodes...")
        for episode in range(num_episodes):

            ep_rewards, ep_obs, ep_lprobs = run_episode(env, policy_network)

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

        print(f"\tUpdating policy...")
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

        # compute variances of different reward measures
        var_full_returns = float(np.var(batch_full_returns))
        var_fut_returns = float(np.var(batch_fut_returns))
        var_disc_fut_returns = float(np.var(batch_disc_fut_returns))
        var_gaes = float(np.var(batch_gaes))
        var_td_errors = float(np.var(batch_td_errors))
        var_dfr_baseline = float(np.var(batch_dfr_baseline))

        # print(f"\tVariance of weights:")
        # print(f"\t\tFull returns: {var_full_returns:.2f}")
        # print(f"\t\tFuture returns: {var_fut_returns:.2f}")
        # print(f"\t\tDiscounted future returns: {var_disc_fut_returns:.2f}")
        # print(f"\t\tGAEs: {var_gaes:.2f}")
        # print(f"\t\tTD errors: {var_td_errors:.2f}")
        # print(f"\t\tDiscounted future returns - baseline: {var_dfr_baseline:.2f}")

        writer.add_scalar("Weight_Variances/Full_Returns", var_full_returns, epoch)
        writer.add_scalar("Weight_Variances/Future_Returns", var_fut_returns, epoch) 
        writer.add_scalar("Weight_Variances/Disc_Future_Returns", var_disc_fut_returns, epoch)
        writer.add_scalar("Weight_Variances/GAEs", var_gaes, epoch)
        writer.add_scalar("Weight_Variances/TD_Errors", var_td_errors, epoch)
        writer.add_scalar("Weight_Variances/DFR_Baseline", var_dfr_baseline, epoch)

        batch_policy_grad_terms = [lprob * weight for lprob, weight in zip(batch_lprobs, batch_weights)]
        if avg_kind == 'a':
            batch_loss = -sum(batch_policy_grad_terms) / len(batch_weights)
        elif avg_kind == 't':
            batch_loss = -sum(batch_policy_grad_terms) / num_episodes

        # update policy network
        policy_network.train()
        policy_optimizer.zero_grad() 
        batch_loss.backward()
        policy_optimizer.step()
        
        # update value network when using GAE
        if weight_kind in ['gae', 'td', 'dfrb']:
            print(f"\tTraining value function...")
            value_optimizer = torch.optim.Adam(value_network.parameters(), lr=value_learning_rate)
            train_dataloader, test_dataloader = create_dataloaders_for_value_network(batch_obs, batch_disc_fut_returns)
            test_loss_info, train_loss_info = train_value_network(value_network, nn.MSELoss(), value_optimizer, train_dataloader, test_dataloader, num_epochs_value_network, n_updates=-1)
        
        # log network parameters
        if (epoch+1) % log_params_every_n_epochs == 0:
            print(f"\tLogging network params info...")
            for name, param in policy_network.named_parameters():
                writer.add_histogram(f'Policy_Param_Values/{name}', param.data, epoch)
                writer.add_scalar(f'Policy_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Policy_Param_Grads/{name}', param.grad, epoch)
                    writer.add_scalar(f'Policy_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)
            
            for name, param in value_network.named_parameters():
                writer.add_histogram(f'Value_Param_Values/{name}', param.data, epoch)
                writer.add_scalar(f'Value_Param_Values_Norm/{name}', param.data.norm().item(), epoch)
                if param.grad is not None:
                    writer.add_histogram(f'Value_Param_Grads/{name}', param.grad, epoch)
                    writer.add_scalar(f'Value_Param_Grads_Norm/{name}', param.grad.norm().item(), epoch)

        # render episode if needed
        if (epoch+1) % render_every_n_epochs == 0:
            print(f"\tVisualizing episode...")
            render_epsiode(env_name, policy_network)
        
        print(f"\tAvg return: {np.mean(batch_returns):.2f} | Avg length: {np.mean(batch_lengths):.2f}")
        writer.add_scalar("Metrics/Avg_Episode_Return", np.mean(batch_returns), epoch)
        writer.add_histogram('Episode_Returns_Distribution', np.array(batch_returns), epoch)

    writer.flush()
    writer.close()
    env.close()

# %%

def run_experiment(param_name: str, values_to_check: list, experiment_name: str, num_runs: int = 1):
    """
    Run experiment varying a single parameter.
    Args:
        param_name: Name of the config parameter to vary
        values_to_check: List of values to try for the parameter
        experiment_name: Name of the experiment (used for folder organization)
        num_runs: Number of times to run each value (default: 1)
    """
    
    for run in range(num_runs):
        for val in values_to_check:
            print(f"\nTesting {param_name}={val} (Run {run+1}/{num_runs})")
            
            # Create config and set parameter
            config = Config(**{param_name: val})
            
            # Set up experiment organization
            config.experiment_group_name = experiment_name
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_tag = f"{param_name}_{val}_run{run+1}_{timestamp}"
            config.log_dir = os.path.join(config.base_log_dir, experiment_name, run_tag)
            
            # Run training
            train_agent(config)


def run_multi_param_experiment(param_dict: dict, experiment_name: str, num_runs: int = 1):
    """
    Run experiment varying multiple parameters.
    Args:
        param_dict: Dictionary where keys are parameter names and values are lists of values to try
        experiment_name: Name of the experiment (used for folder organization)
        num_runs: Number of times to run each combination (default: 1)
    """
    from itertools import product
    
    # Get all combinations of parameter values
    param_names = list(param_dict.keys())
    param_values = list(param_dict.values())
    param_combinations = list(product(*param_values))
    
    for run in range(num_runs):
        for combination in param_combinations:
            # Create parameter dictionary for this combination
            params = dict(zip(param_names, combination))
            
            param_str = '_'.join([f"{name}={val}" for name, val in params.items()])
            print(f"\nTesting {param_str} (Run {run+1}/{num_runs})")
            
            # Create config and set parameters
            config = Config(**params)
            
            # Set up experiment organization
            config.experiment_group_name = experiment_name
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_tag = f"{param_str}_run{run+1}_{timestamp}"
            config.log_dir = os.path.join(config.base_log_dir, experiment_name, run_tag)
            
            # Run training
            train_agent(config)


class Config:
    def __init__(self, **kwargs):

        self.num_episodes = 50
        self.num_epochs_policy_network = 100
        self.render_every_n_epochs = 500
        self.log_params_every_n_epochs = 3

        self.policy_hidden_size = 4
        self.policy_learning_rate = 0.01

        self.value_hidden_size = 16
        self.value_learning_rate = 0.1
        self.num_epochs_value_network = 1

        self.weight_kind = 'gae' # 'r' for 'returns', 'fr' for 'future returns', 'dfr' for 'discounted future returns',
        # 'gae' for 'generalized advantage estimates', 'td' for 'temporal difference errors', 'dfrb' for 'discounted future returns - baseline'
        self.avg_kind = 'a' # 'a' for 'all' and 't' for 'trajectories' 
        self.gamma_gae = 0.99
        self.lambda_gae = 0.96

        self.log_dir = "runs\\test"
        self.base_log_dir = "runs"
        self.experiment_group_name = "exp_vf_lr"

        self.run_tag = ""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown config parameter: {key}")
            self.run_tag += key + "_" + str(value) + "_"


# Example of multi-parameter experiment
param_dict = {
    "value_learning_rate": np.exp(np.linspace(np.log(1e-9), np.log(1), 5)),
    "num_epochs_value_network": [1, 2, 3, 10]
}
run_multi_param_experiment(param_dict, "exp_gae_vf_training", num_runs=3)

#%%
# Example usage:
if __name__ == "__main__":
    # Single test run
    train_agent(Config())
    
    # Run experiment varying value_learning_rate with 3 runs per value
    values_to_check = np.exp(np.linspace(np.log(0.001), np.log(0.1), 5))
    run_experiment("value_learning_rate", values_to_check, "experiment_value_lr", num_runs=3)
    
    # Run experiment varying weight_kind with 5 runs per value
    weight_kinds = ['r', 'fr', 'dfr', 'gae', 'td']
    run_experiment("weight_kind", weight_kinds, "experiment_weight_kind", num_runs=5)
    
    # Example of multi-parameter experiment
    param_dict = {
        "value_learning_rate": np.exp(np.linspace(np.log(0), np.log(1), 5)),
        "num_epochs_value_network": [1, 2, 3, 10]
    }
    run_multi_param_experiment(param_dict, "experiment_vf_training", num_runs=3)


