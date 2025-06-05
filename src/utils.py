import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def run_episode(env, policy_network, render=False):

    num_actions = env.action_space.n # for discrete spaces

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
