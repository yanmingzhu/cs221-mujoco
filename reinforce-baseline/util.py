
import gymnasium as gym
from agent import Agent
import torch
import numpy as np
from matplotlib import pyplot as plt

def train_agent(seed = 89, max_steps = 1000000):
    print(f"-- training with seed {seed} ----")
    env = gym.make('Hopper-v5')#, render_mode  = "human")
    env._max_episode_steps = 1000
    episode_batch = 100
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, episode_batch)  # Records episode-reward

    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    torch.manual_seed(seed)

    agent = Agent(obs_space_dims, action_space_dims)
    print(f"discount factor = {agent.gamma}")
    rewards_over_episodes = []
    rewards_to_plot = []

    trainingSteps = 0
    episode = 0
    while trainingSteps < max_steps:
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        reward_list = []
        while not done:
            action = agent.sample_action(obs)

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            done = terminated or truncated

        reward_list.append(wrapped_env.return_queue[-1])
        agent.update()

        episode += 1
        trainingSteps += wrapped_env.length_queue[-1]
        if episode % episode_batch == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            avg_episode_len = int(np.mean(wrapped_env.length_queue))
            rewards_to_plot.append(avg_reward)
            print("Episode:", episode, "Total Steps:", trainingSteps, "Average Reward:", avg_reward, "average episode len: ", avg_episode_len)

    rewards_over_episodes.append(reward_list)
    
    torch.save(agent.net.state_dict(), './models/hopper')

    plotRewards(rewards_to_plot, './charts/plot')
    return rewards_over_episodes

def replay_agent():
    env = gym.make('Hopper-v5', render_mode  = "human")
    env._max_episode_steps = 100000
    episode_batch = 100
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, episode_batch) 

    total_num_episodes = 100
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    agent = Agent(obs_space_dims, action_space_dims)
    agent.net.load_state_dict(torch.load("./models/hopper", weights_only=True))
    agent.net.eval()

    rewards_to_plot = []
    episode = 0
    while True:
        obs, info = wrapped_env.reset()

        done = False
        while not done:
            action = agent.sample_action(obs)

            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            done = terminated or truncated

        episode += 1
        if episode % episode_batch == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            rewards_to_plot.append(avg_reward)
            avg_episode_len = int(np.mean(wrapped_env.length_queue))
            print("Episode:", episode, "Average Reward:", avg_reward, "average episode len: ", avg_episode_len)

    plotRewards(rewards_to_plot, './charts/plot')

def movingAverage(x, window):
    cumSum = np.cumsum(x)
    ma = (cumSum[window:] - cumSum[:-window]) / window
    return ma

def plotRewards(rewards, savePath=None, show=True):
    plt.figure(figsize=(10, 5))
    window = 30
    trainMA = movingAverage(rewards, window)
    tLen = len(rewards)
    plt.scatter(range(tLen), rewards, alpha=0.5, c='tab:blue', linewidth=0, s=5)
    plt.plot(range(int(window/2), tLen-int(window/2)), trainMA, lw=2, c='b')
    plt.legend(['rewards', 'rewards moving average'])
    plt.xlabel("100 Episode")
    plt.ylabel("Discounted Reward in Episode")

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()