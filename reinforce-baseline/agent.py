
from __future__ import annotations



#import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F


#plt.rcParams["figure.figsize"] = (10, 5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, state_space_dims: int, action_space_dims: int):
        super().__init__()

        hidden_space1 = 128
        hidden_space2 = 128

        self.inner_net = nn.Sequential(
            nn.Linear(state_space_dims, hidden_space1),
            nn.Tanh(),
            nn.Linear(hidden_space1, hidden_space2),
            nn.Tanh(),
        )

        self.action_mean_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
            nn.Tanh(),
        )

        self.action_stddev_net = nn.Sequential(
            nn.Linear(hidden_space2, action_space_dims),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_inner = self.inner_net(x.float())

        action_means = self.action_mean_net(x_inner)
        action_stddevs = torch.log(
            1 + torch.exp(self.action_stddev_net(x_inner))
        )

        return action_means, action_stddevs


class ValueNetwork(nn.Module):
    
    #Takes in state
    def __init__(self, observation_space):
        super(ValueNetwork, self).__init__()
        
        self.input_layer = nn.Linear(observation_space, 128)
        self.output_layer = nn.Linear(128, 1)
        
    def forward(self, x):
        #input layer
        x = self.input_layer(x)
        
        #activiation relu
        x = F.relu(x)
        
        #get state value
        state_value = self.output_layer(x)
        
        return state_value


class Agent:
    def __init__(self, state_space_dims: int, action_space_dims: int):
        self.learning_rate = 1e-4 
        self.gamma = 0.995
        self.eps = np.finfo(np.float32).eps.item()

        self.probs = []  
        self.rewards = [] 
        self.state_list = []

        self.net = PolicyNetwork(state_space_dims, action_space_dims)
        self.value_net = ValueNetwork(state_space_dims)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.learning_rate)
        self.value_optimizer = torch.optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

        self.trainCount = 0

    def sample_action(self, state: np.ndarray):
        self.state_list.append(state)

        action_means, action_stddevs = self.net(torch.tensor(np.array([state])))

        distrib = Normal(action_means[0] + self.eps, action_stddevs[0] + self.eps)
        action = distrib.sample()
        prob = distrib.log_prob(action)
        self.trainCount += 1
        if self.trainCount % 300000 == 0:
            print(f"action = {action}, log_prob = {prob}, mean = {action_means[0].detach()}, std = {action_stddevs[0].detach()}")

        self.probs.append(prob)

        return action.numpy()

    def update(self):
        G = []
        running_total = 0
        for R in self.rewards[::-1]:
            running_total = R + self.gamma * running_total
            G.insert(0, float(running_total))
        
        G = torch.tensor(G).to(device)
        G = (G - G.mean()) / (G.std() + self.eps)

        state_vals = []
        for state in self.state_list:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
            state_vals.append(self.value_net(state))
        
        state_vals = torch.stack(state_vals).squeeze()

        # value
        value_loss = F.mse_loss(state_vals, G)
        #print(f"value_loss = {value_loss}")
        self.value_optimizer.zero_grad()
        value_loss.backward(retain_graph=True)
        self.value_optimizer.step()

        # policy
        sum_probs = [torch.sum(x) for x in self.probs]
        #log_probs = torch.stack(self.probs)
        '''
        log_probs = torch.stack(sum_probs)
        loss = -torch.sum(log_probs * (G - state_vals))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        '''
        deltas = [gt - val for gt, val in zip(G, state_vals)]
        deltas = torch.tensor(deltas).to(device)

        policy_loss = []
        
        #calculate loss to be backpropagated
        for d, lp in zip(deltas, sum_probs):
            policy_loss.append(-d * lp)
        
        #Backpropagation
        self.optimizer.zero_grad()
        sum(policy_loss).backward()
        self.optimizer.step()

        self.probs = []
        self.rewards = []
        self.state_list = []


