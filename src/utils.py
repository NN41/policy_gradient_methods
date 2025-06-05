import torch
import torch.nn as nn
import numpy as np
import gymnasium as gym

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