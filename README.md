# Implementing REINFORCE agent from scratch for CartPole-v1
This project implements the REINFORCE algorithm from scratch to solve the OpenAI Gym CartPole-v1 environment using Pytorch.

## Project Goals
- Implement REINFORCE from scratch with an MLP policy.
- Investigate the impact of a value function baseline.
- Experiment with network architecture.

## Setup
1.  **Clone the repository (if you haven't already):**
    ```bash
    git clone https://github.com/YourUsername/cartpole-reinforce.git
    cd cartpole-reinforce
    ```

2.  **Create and activate the Conda environment:**
    This project uses a Conda environment named `rl_project`. If you have the `environment.yml` file (you can create one with `conda env export > environment.yml`):
    ```bash
    conda env create -f environment.yml
    conda activate rl_project
    ```
    Alternatively, if you only have `requirements.txt`:
    ```bash
    conda create -n rl_project python=3.8  # Or your preferred Python version
    conda activate rl_project
    pip install -r requirements.txt
    ```
    Ensure PyTorch and Gym are installed.

## How to Run
To train the REINFORCE agent:
```bash
python main.py