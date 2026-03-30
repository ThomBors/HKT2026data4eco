import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


class Agent:
    def __init__(self, state_dim, action_dim):

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = DQN(state_dim, action_dim)
        self.target_model = DQN(state_dim, action_dim)
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.memory = deque(maxlen=10000)

        self.gamma = 0.99
        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, s, a, r, s_next, done):
        self.memory.append((s, a, r, s_next, done))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        q_values = self.model(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.target_model(next_states).max(1)[0]
            target = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(q_value, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update epsilon
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())