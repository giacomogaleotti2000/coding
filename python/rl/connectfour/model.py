import numpy as np
import torch

class ConnectFourEnv():
    def __init__(self) -> None:
        self.rows = 6
        self.cols = 7
        # 1D board of size 42
        self.board = np.zeros(shape=(self.rows * self.cols,), dtype=int)
        self.computer = 1
        self.opponent = -1
        self.reset()

    def reset(self):
        """Resets the board and handles who moves first."""
        self.board[:] = 0
        self.done = False
        self.winner = None
        
        # Randomly decide who goes first
        self.mover = np.random.choice([self.computer, self.opponent])
        
        # If opponent starts, they make a random move immediately
        if self.mover == self.opponent:
            action = self.random_action()
            self.apply_action(action, self.opponent)
            
        return self.board.copy()

    def available_actions_idx(self):
        """Returns a list of column indices (0-6) that are not full."""
        # Reshape to 2D to easily check the top row (row 0)
        board_2d = self.board.reshape(self.rows, self.cols)
        # If the top row (0) at column c is 0, the column is valid
        return [c for c in range(self.cols) if board_2d[0, c] == 0]

    def random_action(self):
        """Returns a random valid column."""
        possible_cols = self.available_actions_idx()
        if not possible_cols:
            return None # Draw/Full
        return np.random.choice(possible_cols)

    def apply_action(self, col_idx, player):
        """
        Simulates gravity: places the player's piece in the 
        lowest available row in the given column.
        """
        board_2d = self.board.reshape(self.rows, self.cols)
        
        # Find the lowest empty row in this column
        # We scan from bottom (row 5) to top (row 0)
        for r in range(self.rows - 1, -1, -1):
            if board_2d[r, col_idx] == 0:
                board_2d[r, col_idx] = player
                break
        
        # Flatten back to 1D to update self.board
        self.board = board_2d.flatten()

    def check_win(self, player):
        """Checks horizontal, vertical, and diagonal lines for 4 connected."""
        board_2d = self.board.reshape(self.rows, self.cols)

        # 1. Horizontal
        for r in range(self.rows):
            for c in range(self.cols - 3):
                if np.all(board_2d[r, c:c+4] == player):
                    return True

        # 2. Vertical
        for r in range(self.rows - 3):
            for c in range(self.cols):
                if np.all(board_2d[r:r+4, c] == player):
                    return True

        # 3. Diagonal (\)
        for r in range(self.rows - 3):
            for c in range(self.cols - 3):
                if np.all([board_2d[r+i, c+i] == player for i in range(4)]):
                    return True

        # 4. Anti-Diagonal (/)
        for r in range(3, self.rows):
            for c in range(self.cols - 3):
                if np.all([board_2d[r-i, c+i] == player for i in range(4)]):
                    return True

        return False
    
    def step(self, action, opponent_model=None): # <--- Check this argument
        # 1. Check Agent Valid Move
        if action not in self.available_actions_idx():
             return self.board.copy(), -10, True, {"result": "Error"}
        
        # 2. Agent Move
        self.apply_action(action, self.computer)
        if self.check_win(self.computer):
            return self.board.copy(), 1, True, {"result": "Win"}
        if len(self.available_actions_idx()) == 0:
            return self.board.copy(), 0, True, {"result": "Draw"}

        # 3. Opponent Move
        if opponent_model is None:
            # Default: Random
            opp_action = self.random_action()
        else:
            # Advanced: The Clone
            opp_action = self.get_opponent_action(opponent_model) # <--- Make sure this is called
            
        self.apply_action(opp_action, self.opponent)

        if self.check_win(self.opponent):
            return self.board.copy(), -1, True, {"result": "Loss"}
        if len(self.available_actions_idx()) == 0:
            return self.board.copy(), 0, True, {"result": "Draw"}

        return self.board.copy(), 0, False, {}

    def get_opponent_action(self, model):
        # 1. Prepare the board (Flip perspective)
        board_for_opp = self.board * -1 
        
        # 2. Create the tensor (defaults to CPU)
        state_t = torch.tensor(board_for_opp, dtype=torch.float32).unsqueeze(0).view(1, 1, 6, 7)
        
        # 3. CRITICAL FIX: Move tensor to the same device as the model (CPU or GPU)
        # We check the device of the first parameter of the model
        device = next(model.parameters()).device
        state_t = state_t.to(device)
        
        # 4. Get the move
        with torch.no_grad():
            q_vals = model(state_t)
            valid_moves = self.available_actions_idx()
            
            # Mask invalid moves
            mask = torch.full_like(q_vals, -float('inf'))
            mask[0, valid_moves] = q_vals[0, valid_moves]
            
            action = mask.max(1)[1].item()
            
        return action

    def render(self):
        """Visualizes the board."""
        board_2d = self.board.reshape(self.rows, self.cols)
        symbols = {0: '.', 1: 'X', -1: 'O'}
        print("\nBoard State:")
        for row in board_2d:
            print(" ".join([symbols[x] for x in row]))
        print("-" * 13)
        print("0 1 2 3 4 5 6\n")


import torch.nn as nn
import torch.nn.functional as F

class QNConnectFour(nn.Module):
    def __init__(self, output_dim=7):
        super(QNConnectFour, self).__init__()
        
        # --- Convolutional Block ---
        # We treat the board as an image: 1 channel (the values -1, 0, 1), 6 rows, 7 cols
        
        # Conv1: Expands features. looks for small local patterns
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64) # Normalization helps faster convergence
        
        # Conv2: Goes deeper
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        # Conv3: Refines features
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # --- Fully Connected Block ---
        # Flatten: 128 channels * 6 rows * 7 cols = 5376
        self.fc1 = nn.Linear(128 * 6 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_dim) # Output is 7 (one Q-value per column)

    def forward(self, x):
        # 1. Reshape Input
        # The environment gives us a flat vector (Batch, 42).
        # We must reshape it to (Batch, 1, 6, 7) for the CNN.
        x = x.view(-1, 1, 6, 7) 
        
        # 2. Convolutions + Activations
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 3. Flatten
        x = x.view(x.size(0), -1)
        
        # 4. Dense Layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # 5. Output (No activation here, raw Q-values)
        actions = self.fc3(x)
        
        return actions