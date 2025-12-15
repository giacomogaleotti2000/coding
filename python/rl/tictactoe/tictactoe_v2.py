import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 9)
        )
        self._init()

    def _init(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        nn.init.uniform_(self.net[-1].weight, -1e-3, 1e-3)

    def forward(self, x):
        return self.net(x)

class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros(9, dtype=np.int8)
        self.agent = 1
        self.opponent = -1
        self.reset()

    def reset(self):
        self.board[:] = 0
        self.mover = np.random.choice([1, -1])
        return self.board.copy()

    def available_actions(self):
        return np.where(self.board == 0)[0]

    def reward_done(self):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for i,j,k in lines:
            s = self.board[i] + self.board[j] + self.board[k]
            if s == 3:  return 1.0, True
            if s == -3: return -1.0, True
        if np.all(self.board != 0):
            return 0.0, True
        return 0.0, False

    def step(self, action, opponent_policy=None):
        if self.board[action] != 0:
            return self.board.copy(), -1.0, True

        self.board[action] = self.agent
        r, d = self.reward_done()
        if d:
            return self.board.copy(), r, True

        if opponent_policy is None:
            opp_action = np.random.choice(self.available_actions())
        else:
            opp_action = opponent_policy(self.board.copy(), self.available_actions())

        self.board[opp_action] = self.opponent
        r, d = self.reward_done()

        if d and r == -1:
            return self.board.copy(), -1.0, True

        return self.board.copy(), r, d
    
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state), action, reward, np.array(next_state), done)

    def __len__(self):
        return len(self.buffer)
    
