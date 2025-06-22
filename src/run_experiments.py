import torch
import numpy as np
import gymnasium as gym
import os
import json
import itertools
from datetime import datetime
from typing import Optional
from dataclasses import asdict, replace

from src.config import Config
from src.agent import Agent
from src.trainer import Trainer

def set_seed(seed: int):
    """Sets the random seed for reproducility of experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ExperimentRunner:
    """
    Manages and executes a sequence of training runs, handling parameter sweeps,
    logging, and reproducibility based on a shared base configuration.
    """

    def __init__(self, base_config: Optional[Config] = None, base_log_dir: str = "runs"):
        """
        Initializes the ExperimentRunner with a base configuration.

        Args:
            base_config (Optional[Config]): A Config object with baseline parameters for all experiments.
                                            The base_config is not part of the experiment itself and will be ignored in generating run tags.
                                            If None, the default Config values will be used.
            base_log_dir (str): The root directory where all experiment results will be saved.
        """
        self.base_config = base_config if base_config is not None else Config()
        self.base_log_dir = base_log_dir
        if not os.path.exists(self.base_log_dir):
            os.makedirs(self.base_log_dir)
        print(f"\n--- Experiment Runner Initialized ---")
        print(f"Base Config: {self.base_config}")

    def _format_param_value(self, value) -> str:
        """Helper function to format parameter values for run tags. Returns a string."""
        if isinstance(value, str):
            return value
        if isinstance(value, float): # Use scientific notation for small floats, otherwise use regular formatting
            return f"{value:.2g}" 
        return str(value)

    def run(self, experiment_name: str, param_grid: dict[str, list], num_runs: int = 1, base_seed: int = 42):
        """
        Runs a full experiment by sweeping over a grid of parameter settings. For each parameter setting,
        a full training run will be repeated for a total number of num_runs times. The Nth run for each
        parameter setting will use the same random seed for reproducibility purposes.
        
        Args:
            experiment_name (str): A short name describing the experiment, used to group runs.
            param_grid (dict[str, list]): A dictionary of parameters to sweep over (using a Cartesian product).
            num_runs (int): The number of times to repeat each parameter setting with a different seed.
            base_seed (int): The starting random seed for reproducibility purposes.
        """
        print(f"\n--- Starting Experiment: {experiment_name} ---")

        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values)) # Create a Cartesian product for the grid search

        total_trials = len(param_combinations) * num_runs
        trial_counter = 0
        
        for run_idx in range(num_runs):
            for combo in param_combinations:
                trial_counter += 1
                experiment_params = dict(zip(param_names, combo))
                
                print(f"\n[INFO] Running trial {trial_counter}/{total_trials} (Run {run_idx+1}/{num_runs})")

                # Create the config for this run
                config = replace(self.base_config, **experiment_params)

                # Set and use the seed for this run
                current_seed = base_seed + run_idx
                config.seed = current_seed
                set_seed(current_seed)

                # Create a unique run tag
                param_tag = "_".join([f"{name}={self._format_param_value(val)}" for name, val in experiment_params.items()])
                timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
                run_tag = f"{param_tag}_run={run_idx+1}_{timestamp}"
                
                # Define where to store results
                run_log_dir = os.path.join(self.base_log_dir, experiment_name, run_tag)
                os.makedirs(run_log_dir, exist_ok=True)
                config.log_dir = run_log_dir
                
                print(f"[CONFIG] Parameters: {experiment_params}")
                print(f"[LOG] Logging to: {run_log_dir}")
                
                # Save the config
                config_path = os.path.join(run_log_dir, "config.json")
                with open(config_path, 'w') as f:
                    json.dump(asdict(config), f, indent=4)

                # Instantiate and run training
                print("[TRAINING] Starting training for this configuration...")
                env = gym.make(config.env_name)
                agent = Agent(env, config)
                trainer = Trainer(env, agent, config)
                trainer.train()
                print("[SUCCESS] Training trial completed.")
                
        print(f"\n--- Experiment '{experiment_name}' Finished ---")

if __name__ == '__main__':

    # Tip: Use regex expression (.+)_run=\d+ in Tensorboard for unified coloring per experiment setting,
    # or use regex expression param1_(.+)_ to color based on a certain key parameter.

    # # Instantiate the experiment runner using a base config suitable for
    # # demonstrating the experimentation framework.
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1, # To verify in Tensorboard at the random seed is working
    #     num_episodes=50,
    #     num_epochs_policy_network=50,
    #     weight_kind='fr'
    # )
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )

    # # Example 1: Sweeping a single parameter
    # single_param_grid = {
    #     'policy_learning_rate': [0.01, 1, 0.0001]
    # }
    # runner.run(
    #     experiment_name="example_single_param_sweep",
    #     param_grid=single_param_grid,
    #     num_runs=3 # Run each value 3 times (using 3 different seeds)
    # )

    # # Example 2: Grid search for multiple parameters (using cartesian product)
    # multi_param_grid = {
    #     'num_epochs_value_network': [1, 5],
    #     'value_learning_rate': [0.1, 0.01],
    #     'weight_kind': ['dfr','dfrb','gae']
    # }
    # runner.run(
    #     experiment_name="example_multi_param_sweep",
    #     param_grid=multi_param_grid,
    #     num_runs=1 # Run each of the 2*2*2=8 parameter combinations one time (all using the same seed)
    # )

    # # Experiment 1: Grid search for policy network hyperparameters for high performance with low cost
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1,
    #     num_epochs_policy_network=50,
    #     weight_kind='fr' # Use rewards-to-go
    # )
    # param_grid = {
    #     'num_episodes': [10, 20, 50],
    #     'policy_learning_rate': [0.1, 0.01, 0.001],
    #     'policy_hidden_size': [2, 4, 8]
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp_policy_hyperparam_sweep',
    #     param_grid=param_grid,
    #     num_runs=3
    # )

    # # Experiment 2: Hyperparameter sweep for value function using GAE.
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1,
    #     num_epochs_policy_network=70,
    #     policy_hidden_size=8, # From Experiment 1
    #     policy_learning_rate=0.01, # From Experiment 1
    #     num_episodes=20, # From Experiment 1
    #     lambda_gae=0.96, # From GAE paper
    #     gamma_gae=0.98, # From GAE paper
    #     value_hidden_size=20, # From GAE paper
    #     weight_kind='gae'
    # )
    # param_grid = {
    #     'value_learning_rate': [0.01, 0.001, 0.0001],
    #     'num_epochs_value_network' : [1, 2, 5]
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp_gae_hyperparam_sweep',
    #     param_grid=param_grid,
    #     num_runs=3
    # )

    # # Experiment 3: Comparing dfr, dfrb, gae. 
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1,
    #     num_epochs_policy_network=70,
    #     policy_hidden_size=8, # From Experiment 1
    #     policy_learning_rate=0.01, # From Experiment 1
    #     num_episodes=20, # From Experiment 1
    #     lambda_gae=0.96, # From GAE paper
    #     gamma_gae=0.98, # From GAE paper
    #     value_hidden_size=20, # From GAE paper
    #     value_learning_rate=0.0001, # From Experiment 2
    #     num_epochs_value_network=2 # From Experiment 2
    # )
    # param_grid = {
    #     'weight_kind': ['dfr','dfrb','gae']
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp3_weight_types',
    #     param_grid=param_grid,
    #     num_runs=5
    # )

    # # Experiment 4: Showing that gae and dfrb don't work well because either they barely learn or they collapse
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1,
    #     num_epochs_policy_network=70,
    #     policy_hidden_size=8, # From Experiment 1
    #     policy_learning_rate=0.01, # From Experiment 1
    #     num_episodes=20, # From Experiment 1
    #     gamma_gae=0.98, # From GAE paper
    #     lambda_gae=0.96, # From GAE paper
    #     value_hidden_size=20, # From GAE paper
    # )
    # param_grid = {
    #     'weight_kind': ['dfr']
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp4_vf_collapse_or_underfit',
    #     param_grid=param_grid,
    #     num_runs=3
    # )

    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=1,
    #     num_epochs_policy_network=70,
    #     policy_hidden_size=8, # From Experiment 1
    #     policy_learning_rate=0.01, # From Experiment 1
    #     num_episodes=20, # From Experiment 1
    #     gamma_gae=0.98, # From GAE paper
    #     lambda_gae=0.96, # From GAE paper
    #     value_hidden_size=20, # From GAE paper
    #     weight_kind='gae',
    # )
    # param_grid = {
    #     'lambda_gae': [0, 0.36, 0.96, 1], # 0 is TD error, 1 is disc fut ret with baseline
    #     'value_learning_rate': [0.01, 0.001, 0.0001, 0],
    #     'num_epochs_value_network' : [1,3,10],
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp4_vf_collapse_or_underfit',
    #     param_grid=param_grid,
    #     num_runs=3
    # )

    # # Experiment 5: Diving deeper into the performance collapse
    # base_config = Config(
    #     render_every_n_epochs=999, # Don't render
    #     log_params_every_n_epochs=999,
    #     num_epochs_policy_network=200,
    #     policy_hidden_size=8, # From Experiment 1
    #     policy_learning_rate=0.01, # From Experiment 1
    #     num_episodes=20, # From Experiment 1
    #     gamma_gae=0.98, # From GAE paper
    #     value_hidden_size=20, # From GAE paper
    #     weight_kind='dfrb',
    #     num_epochs_value_network=3,
    # )
    # param_grid = {
    #     'value_learning_rate': [float(round(x,8)) for x in np.linspace(0.0001, 0.0005, 20)]
    # }
    # runner = ExperimentRunner(
    #     base_config=base_config
    # )
    # runner.run(
    #     experiment_name='exp6_vf_collapse_crit_point',
    #     param_grid=param_grid,
    #     num_runs=2
    # )



    # test
    base_config = Config(
        render_every_n_epochs=1, # Don't render
        log_params_every_n_epochs=999,
        num_epochs_policy_network=20,
        policy_hidden_size=8, # From Experiment 1
        policy_learning_rate=0.01, # From Experiment 1
        num_episodes=200, # From Experiment 1
        gamma_gae=0.98, # From GAE paper
        weight_kind='dfr',
    )
    param_grid = {
        'value_learning_rate': [float(round(x,8)) for x in np.linspace(0.0001, 0.0005, 20)]
    }
    runner = ExperimentRunner(
        base_config=base_config
    )
    runner.run(
        experiment_name='exp6_vf_collapse_crit_point',
        param_grid=param_grid,
        num_runs=2
    )