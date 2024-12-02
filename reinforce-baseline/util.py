
import gymnasium as gym
from agent import Agent
import torch

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

    # Reinitialize agent every seed
    agent = Agent(obs_space_dims, action_space_dims)
    print(f"discount factor = {agent.gamma}")
    rewards_over_episodes = []

    #for episode in range(max_episodes):
    trainingSteps = 0
    episode = 0
    while trainingSteps < max_steps:
        # gymnasium v26 requires users to set seed while resetting the environment
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        reward_list = []
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        reward_list.append(wrapped_env.return_queue[-1])
        agent.update()

        episode += 1
        trainingSteps += wrapped_env.length_queue[-1]
        if episode % episode_batch == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            avg_episode_len = int(np.mean(wrapped_env.length_queue))
            #print(f"return_queue = {wrapped_env.return_queue}")
            #print(f"length_queue = {wrapped_env.length_queue}")
            print("Episode:", episode, "Total Steps:", trainingSteps, "Average Reward:", avg_reward, "average episode len: ", avg_episode_len)

    rewards_over_episodes.append(reward_list)
    
    torch.save(agent.net.state_dict(), './model_saved_baseline')
    return rewards_over_episodes

def replay_agent():
    env = gym.make('Hopper-v5', render_mode  = "human")
    #env._max_episode_steps = 100000
    episode_batch = 100
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, episode_batch)  # Records episode-reward

    total_num_episodes = 100
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    rewards_over_seeds = []

    # Reinitialize agent every seed
    agent = Agent(obs_space_dims, action_space_dims)
    agent.net.load_state_dict(torch.load("./model_saved_baseline", weights_only=True))
    agent.net.eval()

    episode = 0
    while True:
        obs, info = wrapped_env.reset(seed=seed)

        done = False
        while not done:
            action = agent.sample_action(obs)

            # Step return type - `tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]`
            # These represent the next observation, the reward from the step,
            # if the episode is terminated, if the episode is truncated and
            # additional info from the step
            obs, reward, terminated, truncated, info = wrapped_env.step(action)
            agent.rewards.append(reward)

            # End the episode when either truncated or terminated is true
            #  - truncated: The episode duration reaches max number of timesteps
            #  - terminated: Any of the state space values is no longer finite.
            done = terminated or truncated

        episode += 1
        if episode % episode_batch == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            avg_episode_len = int(np.mean(wrapped_env.length_queue))
            print("Episode:", episode, "Average Reward:", avg_reward, "average episode len: ", avg_episode_len)
