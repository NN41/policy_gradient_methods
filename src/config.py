import torch
from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass
class Config:
    """
    Configuration object for the training and experimentation pipeline.

    This class contains hyperparameters and settings for the agent, as well as the training process.
    An instance of this class acts as the single source of truth for a given run and can be
    serialized for logging and reproducibility purposes of experiments.
    """
    
    # --- System and environment ---
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    env_name: str = "CartPole-v1"

    # --- Training loop hyperparameters ---
    num_episodes: int = 20
    num_epochs_policy_network: int = 50 
    num_epochs_value_network: int = 1

    # --- Policy network hyperparameters ---
    policy_hidden_size: int = 4
    policy_learning_rate: float = 0.01

    # --- Value function network hyperparameters ---
    value_hidden_size: int = 16
    value_learning_rate: float = 0.1

    # --- Algorithm hyperparameters ---
    # GAE-specific parameters
    gamma_gae: float = 0.99 # Equivalent to standard discount factor
    lambda_gae: float = 0.96 # GAE-specific parameter for bias-variance tradeoff

    # Defines the weight for the log-probabilities in the policy gradient estimate.
    # 'r': full returns, 'dfr': discounted future returns, 'fr': future returns (rewards-to-go),
    # 'gae': generalized advantage estimates, 'td': temporal difference errors, 'dfrb': discounted future returns - value function baseline
    weight_kind: Literal['r', 'fr', 'dfr', 'gae', 'td', 'dfrb'] = 'fr'

    # Defines the averaging method for the policy gradient estimate
    # 'a': average over all episodes, 't': average over trajectories (i.e., per episode)
    avg_kind: Literal['a', 't'] = 'a'

    # --- Logging and experimentation metadata ---
    render_every_n_epochs: int = 1
    log_params_every_n_epochs: int = 5 # When to log the gradient (norms) and parameter (norms) of the policy and value networks
    log_dir: Optional[str] = None  # Directory for logging results, to be used by TensorBoard and populated at runtime
    experiment_name: Optional[str] = None # Tag for the run, to be used for logging and experiment tracking, populated at runtime
    seed: Optional[int] = None

if __name__ == '__main__':

    # Create a default config
    default_config = Config()
    print(f"\n--- Default Config ---")
    print(default_config)

    # Create a custom config for an experiment
    experiment_config = Config(
        weight_kind='gae',
        num_episodes=10,
        policy_learning_rate=0.001,
        experiment_name='exp_test'
    )
    print(f"\n--- Experiment Config ---")
    print(experiment_config)
    print(f"\n")