import numpy as np

class TicTacToeEnv():
    def __init__(self) -> None:
        self.board = np.zeros(9, dtype=int)
        self.computer = 1
        self.opponent = -1
        self.reset()
    
    def reset(self):
        self.board[:] = 0
        self.mover = np.random.choice([-1,1])
        if self.mover == self.opponent:
            action = self.random_action()
            self.board[action] = self.opponent
        return self.board.copy()
    
    def available_actions_idx(self):
        "returns a np.array with the indexes of the available actions"
        return np.where(self.board == 0)[0]
    
    def random_action(self):
        """returns the action index to be taken"""
        return np.random.choice(self.available_actions_idx())
    
    def reward_done(self, state):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for (i,j,k) in lines:
            s = state[i] + state[j] + state[k]
            if s == 3:
                return 1, True
            if s == -3:
                return -1, True
        if np.all(state != 0):
            return 0, True  # draw
        return 0, False  # game not finished

    def step(self, action):
        # punishing illegal moves said by the model
        if self.board[action] != 0:
            return self.board.copy(), -5, True
        
        # model moves
        self.board[action] = self.computer
        new_state = self.board.copy()
        reward, done = self.reward_done(new_state)
        if done:
            return new_state, reward, done
        
        # opponent moves (for now randomly)
        action = self.random_action()
        self.board[action] = self.opponent
        new_state = self.board.copy()
        reward, done = self.reward_done(new_state)
        return new_state, reward, done

from torch import nn

class QNTicTacToe(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)