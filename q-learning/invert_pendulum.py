import gymnasium as gym
import numpy as np
from collections import defaultdict
import random, math, json
from feature import fourierFeatureExtractor, fourierFeatureExtractor2

class InvertPendulumAgent():
    def __init__(self, weights = None, render = None, epsilon = 0.2, eta = 0.001,  explore_total = 2e5, gamma = 0.95, 
                 feature_max_coef = 2, action_discrete_factor = 10, feature_extractor = fourierFeatureExtractor):
        self.eta = eta
        self.epsilon = epsilon
        self.gamma = gamma
        self.feature_max_coef = feature_max_coef
        self.action_discrete_factor = action_discrete_factor
        self.feature_extractor = feature_extractor
        self.discount = 0.99

        self.explore_total = explore_total
        self.explore_cnt = 0
        self.total_cnt = 0
        self.env = gym.make('InvertedPendulum-v5', reset_noise_scale=0.1, render_mode=render)

        observation_dim = 4
        self.feature_length = (1 + feature_max_coef) ** observation_dim
        if weights is None or len(weights) == 0:
            print("--- starting with zero weights --- ")
            self.W = np.random.standard_normal((self.feature_length, action_discrete_factor))
        else:
            print("---- starting with existing weights -----")
            self.W = weights
        self.actions = [i for i in range(action_discrete_factor)]

        self.min_action = -3
        self.max_action = 3

    def getGymEnv(self):
        return self.env 
    
    def getWeights(self):
        return self.W
    
    def bucketize_action(self, action): 
        return (action[0] - self.min_action) // ((self.max_action - self.min_action) / self.action_discrete_factor)
    
    def unbucketize_action(self, action_bucket):
        return [self.min_action + action_bucket * ((self.max_action - self.min_action) / self.action_discrete_factor)]

    def getQ(self, state: np.ndarray, action_bucket: int) -> float:
        features = self.feature_extractor(state, self.feature_max_coef)
        #f2 = fourierFeatureExtractor2(state, self.feature_max_coef)
        w = self.W[:, self.actions.index(action_bucket)]
        return np.dot(w, features)
    
    def exploreRate(self):
        return self.explore_cnt / self.total_cnt
    
    def getAction(self, obs, explore = True):
        self.total_cnt += 1
        exploreProb = self.epsilon
        if explore: 
            if self.explore_cnt < self.explore_total:
                exploreProb = 1
            elif self.explore_cnt > 5e5:
                exploreProb = self.epsilon / math.log(self.explore_cnt - 5e5 + 1)

        if explore and random.random() < exploreProb:
            self.explore_cnt += 1
            return self.env.action_space.sample()
        else:
            qOfAction = {action:self.getQ(obs, action) for action in self.actions}
            action_bucket = max(qOfAction, key=qOfAction.get)
            return self.unbucketize_action(action_bucket)
        
    def getStepSize(self) -> float:
        return 0.005 * (0.99)**(self.explore_cnt / 500)
    
    def incorporate_feedback(self, state, action, reward, next_state, terminal = False):
        action_bucket = self.bucketize_action(action)
        V_s_prime = 0 # for end state
        if not terminal:
            V_s_prime = max([self.getQ(next_state, bucket) for bucket in self.actions])
        features = self.feature_extractor(state, self.feature_max_coef)
        w = self.W[:, [self.actions.index(action_bucket)]]
        newW  = w.flatten() - self.getStepSize() * (self.getQ(state, action_bucket) - (reward + self.discount * V_s_prime) ) * features
        self.W[:, [self.actions.index(action_bucket)]] = newW.reshape((len(newW), 1))
    
if __name__ == "__main__":

    total_iteration = 10000

    agent = InvertPendulumAgent()

    env = agent.getGymEnv()

    total_reward = 0
    for iteration in range(total_iteration):
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = agent.getAction(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            episode_over = terminated or truncated
            agent.incorporate_feedback(observation, action, reward, next_observation, episode_over)
            observation = next_observation
    
        if episode_over and iteration % 100 == 0:
            print(f"total_reward = {total_reward}")
            total_reward = 0


    #eval
    print("starting eval")
    for iteration in range(2000):
        observation, info = env.reset()
        episode_over = False
        while not episode_over:
            action = agent.getAction(observation, False)
            next_observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            episode_over = terminated or truncated
        #agent.incorporate_feedback(observation, action, reward, next_observation, episode_over)
        #observation = next_observation
    
        if episode_over and iteration % 100 == 0:
            print(f"total_reward = {total_reward}")
            total_reward = 0       
    env.close()