from tictactoe_v2 import TicTacToeEnv, QNetwork
import torch
import numpy as np

# Load the trained model
try:
    model = QNetwork()
    model.load_state_dict(torch.load("model_v8.pth"))
    model.eval()
    print("🤖 Model 'model_v8.pth' loaded successfully.")
except FileNotFoundError:
    print("❌ ERROR: Could not find 'model_v8.pth'. Ensure the trained model file is in the current directory.")
    exit()
except Exception as e:
    print(f"❌ ERROR loading model: {e}")
    exit()

# --- Helper Functions (Provided by User) ---

def print_board(board):
    """Pretty print the board"""
    symbols = {0: ' ', 1: 'X', -1: 'O'}
    print("\n")
    print(f" {symbols[board[0]]} | {symbols[board[1]]} | {symbols[board[2]]} ")
    print("---+---+---")
    print(f" {symbols[board[3]]} | {symbols[board[4]]} | {symbols[board[5]]} ")
    print("---+---+---")
    print(f" {symbols[board[6]]} | {symbols[board[7]]} | {symbols[board[8]]} ")
    print("\n")

def print_board_positions():
    """Show position numbers"""
    print("\nPosition numbers:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 ")
    print()

def check_winner(board):
    """Returns 1 if X wins, -1 if O wins, 0 if draw, None if game continues"""
    lines = [
        (0,1,2), (3,4,5), (6,7,8),  # rows
        (0,3,6), (1,4,7), (2,5,8),  # columns
        (0,4,8), (2,4,6)             # diagonals
    ]
    for (i,j,k) in lines:
        s = board[i] + board[j] + board[k]
        if s == 3:
            return 1   # X wins
        if s == -3:
            return -1  # O wins
    if np.all(board != 0):
        return 0  # draw
    return None  # game continues

def model_move(board, model):
    """Model chooses best move by selecting the action with the highest Q-value"""
    # Convert numpy board state to a PyTorch tensor
    state_tensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        q_vals = model(state_tensor)
        
        # Mask illegal moves
        available_actions = np.where(board == 0)[0]
        
        # Set Q-values for occupied positions to negative infinity 
        # to ensure they are not selected, even if the model predicts a high Q-value.
        full_q = q_vals[0].numpy()
        mask = np.ones(9) * -np.inf
        mask[available_actions] = full_q[available_actions]
        
        action = int(np.argmax(mask))

    return action

def play_game():
    """Play one game against the model"""
    board = np.zeros(9, dtype=int)
    
    # Choose who goes first
    print("\n" + "="*40)
    print("NEW GAME!")
    print("="*40)
    print("You are 'O' (-1). Model is 'X' (1).")
    print_board_positions()
    
    choice = input("Do you want to go first? (y/n): ").strip().lower()
    human_turn = (choice == 'y')
    
    # If model starts, it makes the first move before the loop
    # if not human_turn:
    #     move = model_move(board, model)
    #     board[move] = 1
    #     print(f"Model played position {move}")
    
    print_board(board)
    
    while check_winner(board) is None:
        if human_turn:
            # Human's turn
            while True:
                try:
                    move = input("Your move (0-8): ")
                    if move.lower() == 'q':
                         return 'quit'
                    move = int(move)
                    
                    if move < 0 or move > 8:
                        print("Please enter a number between 0 and 8")
                        continue
                    if board[move] != 0:
                        print("That position is already taken!")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number (0-8) or 'q' to quit.")
            
            board[move] = -1  # Human is -1 (O)
            print(f"\nYou played position {move}")
            print_board(board)
            
            if check_winner(board) is not None:
                break
            
            human_turn = False
        
        else:
            # Model's turn
            move = model_move(board, model)
            board[move] = 1  # Model is 1 (X)
            print(f"Model played position {move}")
            print_board(board)
            
            if check_winner(board) is not None:
                break
            
            human_turn = True
    
    # Game over
    result = check_winner(board)
    if result == 1:
        print("🤖 Model (X) wins!")
    elif result == -1:
        print("🎉 You (O) win!")
    else:
        print("🤝 It's a draw!")
    
    return result

# --- Main Game Loop ---
if __name__ == "__main__":
    print("\n" + "="*40)
    print("TIC-TAC-TOE vs DEEP Q-NETWORK AI")
    print("="*40)

    wins = 0
    losses = 0
    draws = 0

    while True:
        result = play_game()
        
        if result == 'quit':
            break

        if result == -1:
            wins += 1
        elif result == 1:
            losses += 1
        else:
            draws += 1
        
        print(f"\nScoreboard: You: {wins}, Model: {losses}, Draws: {draws}")
        
        play_again = input("\nPlay again? (y/n/q): ").strip().lower()
        if play_again != 'y':
            break

    print("\n" + "="*40)
    print("Thanks for playing!")
    print(f"Final Score - You: {wins}, Model: {losses}, Draws: {draws}")
    print("="*40)