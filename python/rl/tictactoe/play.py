from tictactoe import TicTacToeEnv, QNTicTacToe
import torch
import numpy as np

model = QNTicTacToe()
model.load_state_dict(torch.load("model_1.pth"))
model.eval()

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
    """Model chooses best move"""
    with torch.no_grad():
        q_vals = model(torch.tensor(board, dtype=torch.float32))
        action = int(torch.argmax(q_vals).item())
    return action

def play_game():
    """Play one game against the model"""
    board = np.zeros(9, dtype=int)
    
    # Choose who goes first
    print("\n" + "="*40)
    print("NEW GAME!")
    print("="*40)
    print("You are 'O', Model is 'X'")
    print_board_positions()
    
    choice = input("Do you want to go first? (y/n): ").strip().lower()
    human_turn = (choice == 'y')
    
    print_board(board)
    
    while True:
        if human_turn:
            # Human's turn
            while True:
                try:
                    move = int(input("Your move (0-8): "))
                    if move < 0 or move > 8:
                        print("Please enter a number between 0 and 8")
                        continue
                    if board[move] != 0:
                        print("That position is already taken!")
                        continue
                    break
                except ValueError:
                    print("Please enter a valid number")
            
            board[move] = -1  # Human is -1 (O)
            print(f"\nYou played position {move}")
            print_board(board)
            
            result = check_winner(board)
            if result is not None:
                break
            
            human_turn = False
        
        else:
            # Model's turn
            move = model_move(board, model)
            board[move] = 1  # Model is 1 (X)
            print(f"Model played position {move}")
            print_board(board)
            
            result = check_winner(board)
            if result is not None:
                break
            
            human_turn = True
    
    # Game over
    if result == 1:
        print("🤖 Model (X) wins!")
    elif result == -1:
        print("🎉 You (O) win!")
    else:
        print("🤝 It's a draw!")
    
    return result

# Main game loop
print("\n" + "="*40)
print("TIC-TAC-TOE vs AI")
print("="*40)

wins = 0
losses = 0
draws = 0

while True:
    result = play_game()
    
    if result == -1:
        wins += 1
    elif result == 1:
        losses += 1
    else:
        draws += 1
    
    print(f"\nScore - You: {wins}, Model: {losses}, Draws: {draws}")
    
    play_again = input("\nPlay again? (y/n): ").strip().lower()
    if play_again != 'y':
        break

print("\n" + "="*40)
print("Thanks for playing!")
print(f"Final Score - You: {wins}, Model: {losses}, Draws: {draws}")
print("="*40)