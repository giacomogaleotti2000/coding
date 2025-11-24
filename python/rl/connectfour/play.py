import torch
from model import QNConnectFour, ConnectFourEnv
import numpy as np
import time


# --- CONFIGURATION ---
MODEL_PATH = "connect4_dqn_v8.pth" # Ensure this matches your save file
HUMAN = 1
AI = -1

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
    print("Loading Brain...")
    try:
        model = QNConnectFour(output_dim=7)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval() # Set to evaluation mode
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Did you train and save it?")
        return

    # 3. Setup Game
    board = env.reset()
    game_over = False
    
    # Randomly decide who starts
    # turn = np.random.choice([HUMAN, AI])
    turn = HUMAN # Let's force Human start for now
    
    print("--- GAME START ---")
    print_board(board)

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
    play()