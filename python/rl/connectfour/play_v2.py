import torch
import torch.nn as nn
import torch.nn.functional as F
from model_v2 import QNConnectFour, ConnectFourEnv
import numpy as np
import time


# --- CONFIGURATION ---
# Change MODEL_PATH to play against different AI opponents:
#   - connect4_dqn_v1.pth through v9.pth: Original models (USE_DUELING = False)
#   - connect4_master_*.pth: Master-level models (USE_DUELING = True)

MODEL_PATH = "connect4_dqn_optimal_new.pth"  # Change this to your desired model
USE_DUELING = False  # Set to True ONLY for master-level models from master_training.ipynb

HUMAN = 1
AI = -1

# --- Dueling DQN Architecture (for master-level models) ---
class DuelingQNetwork(nn.Module):
    def __init__(self, output_dim=7):
        super(DuelingQNetwork, self).__init__()

        # Shared Convolutional Layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        feature_size = 256 * 6 * 7

        # VALUE STREAM
        self.value_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # ADVANTAGE STREAM
        self.advantage_stream = nn.Sequential(
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.view(-1, 1, 6, 7)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        x = x.view(x.size(0), -1)

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values

def print_board(board_1d):
    """Pretty prints the board with colored circles."""
    board = board_1d.reshape(6, 7)
    print("\n  0  1  2  3  4  5  6")
    for row in board:
        row_str = " "
        for cell in row:
            if cell == 0:
                row_str += "⚪ "
            elif cell == 1: # Human
                row_str += "🔴 "
            elif cell == -1: # AI
                row_str += "🟡 "
        print(row_str)
    print()

def get_human_move(env):
    """Gets a valid integer input from the user."""
    valid_moves = env.available_actions_idx()
    while True:
        try:
            col = input(f"Your Turn (🔴). Choose column {valid_moves}: ")
            col = int(col)
            if col in valid_moves:
                return col
            else:
                print("Invalid column (full or out of range). Try again.")
        except ValueError:
            print("Please enter a number.")

def get_ai_move(model, board, env):
    """
    Gets the AI's move.
    CRITICAL: The model was trained to play as '1'. 
    Since AI is playing as '-1', we multiply board by -1.
    This 'flips' the perspective so the AI thinks it is Player 1.
    """
    # Flip perspective
    board_input = board * -1
    
    # Prepare Tensor
    state_t = torch.tensor(board_input, dtype=torch.float32).unsqueeze(0) # (1, 42)
    
    # Predict
    with torch.no_grad():
        q_values = model(state_t)
        
        # Mask invalid moves so AI doesn't cheat/fail
        valid_moves = env.available_actions_idx()
        mask = torch.full_like(q_values, -float('inf'))
        mask[0, valid_moves] = q_values[0, valid_moves]
        
        # Pick best move
        action = mask.max(1)[1].item()
        
    print(f"AI (🟡) chooses column: {action}")
    return action

def play():
    # 1. Load Environment
    env = ConnectFourEnv()

    # 2. Load Model
    print(f"Loading AI Brain from: {MODEL_PATH}")
    print(f"Architecture: {'Dueling DQN (Master)' if USE_DUELING else 'Standard QN'}")

    try:
        # Choose architecture based on configuration
        if USE_DUELING:
            model = DuelingQNetwork(output_dim=7)
        else:
            model = QNConnectFour(output_dim=7)

        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()  # Set to evaluation mode
        print("✓ AI loaded successfully!\n")
    except FileNotFoundError:
        print(f"✗ Error: Could not find {MODEL_PATH}")
        print("Available models:")
        import os
        for file in os.listdir("."):
            if file.endswith(".pth"):
                print(f"  - {file}")
        return
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        print("\nTip: If using a master-level model, set USE_DUELING = True")
        return

    # 3. Setup Game
    # Manually reset the board without automatic opponent move
    env.board[:] = 0
    env.done = False
    env.winner = None
    game_over = False

    # Randomly decide who starts
    turn = np.random.choice([HUMAN, AI])

    if turn == HUMAN:
        print("🎲 You go first!")
    else:
        print("🎲 AI goes first!")

    print("\n--- GAME START ---")
    print_board(env.board)

    while not game_over:
        if turn == HUMAN:
            col = get_human_move(env)
            env.apply_action(col, HUMAN)
        else:
            time.sleep(0.5) # Small delay to feel like "thinking"
            col = get_ai_move(model, env.board, env)
            env.apply_action(col, AI)
            
        # Show board
        print_board(env.board)
        
        # Check Win
        if env.check_win(turn):
            if turn == HUMAN:
                print("🎉 YOU WIN! 🎉")
            else:
                print("💀 AI WINS! 💀")
            game_over = True
            
        # Check Draw
        elif len(env.available_actions_idx()) == 0:
            print("It's a Draw!")
            game_over = True
            
        # Switch turn
        turn *= -1

if __name__ == "__main__":
    print("=" * 60)
    print("CONNECT FOUR - HUMAN VS AI")
    print("=" * 60)
    print("\nHow to use:")
    print("  1. Edit MODEL_PATH at the top to choose which AI to play")
    print("  2. Set USE_DUELING = True for master-level models")
    print("  3. Set USE_DUELING = False for v1-v9 models")
    print("\nExamples:")
    print("  - connect4_dqn_v9.pth (USE_DUELING = False)")
    print("  - connect4_master_final.pth (USE_DUELING = True)")
    print("=" * 60)
    print()

    play()

    # Ask to play again
    while True:
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again == 'y':
            print("\n" * 2)
            play()
        else:
            print("\nThanks for playing! 👋")
            break