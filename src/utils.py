import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_episode(env, policy_network, render=False):

    num_actions = env.action_space.n # for discrete spaces
    policy_network.eval()

    observation, info = env.reset()
    ep_rewards = []
    ep_obs = []
    ep_lprobs = []
    episode_done = False

    while not episode_done:

        if render:
            env.render()

        # get probabilities from policy
        logits = policy_network(torch.tensor(observation, dtype=torch.float32).to(device))
        probs = nn.Softmax(dim=-1)(logits)
        
        # randomly choose an action based and compute log-probability
        idx = np.random.choice(range(num_actions), p=probs.detach().cpu().numpy())
        action = idx
        log_prob = torch.log(probs[action])

        # act in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # log data and determine if episode is over
        ep_rewards.append(reward)
        ep_obs.append(observation.tolist())
        ep_lprobs.append(log_prob)
        # batch_obs.append(observation.tolist())
        # batch_lprobs.append(log_prob)
        episode_done = terminated or truncated

    return ep_rewards, ep_obs, ep_lprobs

def render_epsiode(env_name, policy_network):
    render_env = gym.make(env_name, render_mode='human')
    policy_network.eval()
    with torch.no_grad():
        run_episode(render_env, policy_network, render=True)
    render_env.close()

def compute_discounted_reverse_cumsums(vals, disc_factor):
    disc_sums = []
    disc_sum = 0
    for val in vals[::-1]:
        disc_sum = val + disc_factor * disc_sum
        disc_sums.append(disc_sum)
    return disc_sums[::-1]

def compute_discounted_future_returns(ep_rewards, disc_factor):
    return compute_discounted_reverse_cumsums(ep_rewards, disc_factor)

def compute_td_errors(ep_observations, ep_rewards, disc_factor, value_network):
    ep_rews = np.array(ep_rewards) 
    ep_obs = torch.tensor(ep_observations, device=device)
    value_network.eval()
    with torch.no_grad():
        ep_vals = value_network(ep_obs).squeeze().cpu().numpy()
    ep_vals[-1] = 0 # value of being in the last (terminated) state is 0. This is necessary to keep
    ep_vals_next = np.append(ep_vals[1:], 0) # similar reasoning as for ep_vals
    ep_td_errors = ep_rews + disc_factor * ep_vals_next - ep_vals # = Rt + gamma * V(St+1) - V(St)
    # ep_td_errors[-2] = 0 # I might prefer this, to make sure the TD errors give stable values at all timesteps
    return ep_td_errors.tolist()

def compute_gaes(ep_observations, ep_rewards, gamma_gae, lambda_gae, value_network, set_to_zero=False):
    """
    When lambda_gae = 0, evaluates to TD errors.
    When lambda_gae = 1, evaluates to discounted future returns minus the value function baseline.
    Both of this is confirmed (of course, only when set_to_zero = False).
    """
    ep_td_errors = compute_td_errors(ep_observations, ep_rewards, gamma_gae, value_network)
    if set_to_zero:
        ep_td_errors[-2] = 0 # We might need this for stability
    gae_discount_factor = gamma_gae * lambda_gae 
    ep_gaes = compute_discounted_reverse_cumsums(ep_td_errors, gae_discount_factor)
    return ep_gaes

class Config:
    def __init__(self, **kwargs):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_name = "CartPole-v1"


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


# # %%
# print("\n--- Saving the trained policy network ---")

# model_name = "policy_network"
# models_dir = "results\models"
# os.makedirs(models_dir, exist_ok=True) 
# current_date = datetime.now().strftime("%m-%d")
# model_filename = f"{model_name}_{env_name}_{current_date}.pth" 
# model_filepath = os.path.join(models_dir, model_filename)

# # Save the model's state_dict (learned parameters)
# torch.save(policy_network.state_dict(), model_filepath)
# print(f"Model saved to: {model_filepath}")

# # Get and print the size of the saved file
# file_size_bytes = os.path.getsize(model_filepath)
# file_size_kb = file_size_bytes / 1024
# print(f"File size: {file_size_kb:.2f} KB")

# %%
# # Assuming PolicyMLP is defined and num_features, policy_hidden_size, num_actions are known
# loaded_policy_network = PolicyMLP(num_features, policy_hidden_size, num_actions).to(device)
# loaded_policy_network.load_state_dict(torch.load(model_filepath))
# loaded_policy_network.eval() # Set to evaluation mode after loading
