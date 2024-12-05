import numpy as np
import gymnasium as gym
from matplotlib import pyplot as plt
import sys, argparse, random, json
import datetime, os, time

from invert_pendulum import InvertPendulumAgent

def movingAverage(x, window):
    cumSum = np.cumsum(x)
    ma = (cumSum[window:] - cumSum[:-window]) / window
    return ma

def plotRewards(trainRewards, evalRewards, savePath=None, show=True):
    plt.figure(figsize=(10, 5))
    window = 30
    trainMA = movingAverage(trainRewards, window)
    evalMA = movingAverage(evalRewards, window)
    tLen = len(trainRewards)
    eLen = len(evalRewards)
    plt.scatter(range(tLen), trainRewards, alpha=0.5, c='tab:blue', linewidth=0, s=5)
    plt.plot(range(int(window/2), tLen-int(window/2)), trainMA, lw=2, c='b')
    plt.scatter(range(tLen, tLen+eLen), evalRewards, alpha=0.5, c='tab:green', linewidth=0, s=5)
    plt.plot(range(tLen+int(window/2), tLen+eLen-int(window/2)), evalMA, lw=2, c='darkgreen')
    plt.legend(['train rewards', 'train moving average', 'eval rewards', 'eval moving average'])
    plt.xlabel("Episode")
    plt.ylabel("Discounted Reward in Episode")

    if savePath is not None:
        plt.savefig(savePath)
    if show:
        plt.show()

def train(agent, eval=False, totalIteration = 1000):
    env = agent.getGymEnv()
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 100)

    allRewards = []
    for episode in range(totalIteration):
        observation, info = wrapped_env.reset()
        episode_over = False
        while not episode_over:
            action = agent.getAction(observation, explore = not eval)
            next_observation, reward, terminated, truncated, info = wrapped_env.step(action)

            episode_over = terminated or truncated
            agent.incorporate_feedback(observation, action, reward, next_observation, episode_over)
            observation = next_observation

        allRewards.append(wrapped_env.return_queue[-1])
        
        if episode % 100 == 0:
            avg_reward = int(np.mean(wrapped_env.return_queue))
            avg_episode_len = int(np.mean(wrapped_env.length_queue))
            print(f"episode {episode}, average reward = {avg_reward}, average episode length = {avg_episode_len} exploreRate = {agent.exploreRate()}")           

    return allRewards

def trainAndEval(agent, trainIteration = 10000, evalIteration = 1000):
    print("exploration -----------------")
    exploreTrainReward = train(agent, False, trainIteration)
    print("greedy training -------------")
    greedyTrainReward = train(agent, True, int(trainIteration * 0.5))
    print("start evaluation ------------")
    evalReward = train(agent, True, evalIteration)
    return (exploreTrainReward + greedyTrainReward, evalReward)

def trainPendulum():
    train_iteration = 20000
    pendulumAgent = InvertPendulumAgent()
    print(f"training inverted pendulum with discrete factor of {pendulumAgent.action_discrete_factor}")
    trainReward, evalReward = trainAndEval(pendulumAgent, train_iteration, int(train_iteration * 0.1))
    os.mkdir(f"./{result_dir}")
    np.save(f"./{result_dir}/weights", pendulumAgent.getWeights())
    plotRewards(trainReward, evalReward, f"./{result_dir}/invert_pendulum")

def runPendulum(weights, totalIteration = 100):
    agent = InvertPendulumAgent(weights=weights, render = "human")

    env = agent.getGymEnv()
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 100)

    allRewards = []
    for episode in range(totalIteration):
        observation, info = wrapped_env.reset()
        episode_over = False
        while not episode_over:
            action = agent.getAction(observation, explore = False)
            next_observation, reward, terminated, truncated, info = wrapped_env.step(action)
            observation = next_observation

            episode_over = terminated or truncated
        print(f"reward: {wrapped_env.return_queue[-1]}")
        allRewards.append(wrapped_env.return_queue[-1])
        time.sleep(1)
        #if episode % 100 == 0:
        #    avg_reward = int(np.mean(wrapped_env.return_queue))
        #    avg_episode_len = int(np.mean(wrapped_env.length_queue))
        #    print(f"episode {episode}, average reward = {avg_reward}, average episode length = {avg_episode_len} exploreRate = {agent.exploreRate()}")
    plotRewards([], allRewards)

if __name__ == "__main__":
    """
    The main function called when train.py is run
    from the command line:

    > python train.py

    See the usage string for more details.

    > python train.py --help
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["pendulum", "pendulum2"],
        help="",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "play"],
        help="",        
    )
    parser.add_argument(
        "--weights",
        type=str,
        help="",        
    )
    args = parser.parse_args()

    mode = "train"
    if args.mode == "play":
        mode = "play"
    
    print(args.weights)
    weights = None
    if args.weights is not None:
        weights = np.load(f"./{args.weights}")
        print(f"./{weights}")

    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y-%m-%d-%H:%M:%S")
    result_dir = f"./training_results/{formatted_time}"

    if args.agent == "pendulum":
        if mode == "train":
            trainPendulum()
        else:
            if weights is None:
                print("--weights cannot be empty")
            runPendulum(weights)

