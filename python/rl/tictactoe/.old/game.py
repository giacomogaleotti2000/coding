import numpy as np
import torch
from dqn_tictactoe import DQN, state_to_tensor, mask_invalid, TicTacToeEnv, DEVICE

MODEL_PATH = "dqn_tictactoe.pth"

def print_board(board):
    symbols = {1: 'X', -1: 'O', 0: ' '}
    b = [symbols[int(x)] for x in board]
    print("\n")
    print(f" {b[0]} | {b[1]} | {b[2]}")
    print("---+---+---")
    print(f" {b[3]} | {b[4]} | {b[5]}")
    print("---+---+---")
    print(f" {b[6]} | {b[7]} | {b[8]}")
    print("\n")


def human_move(board):
    valid = np.where(board == 0)[0]
    while True:
        print(f"Available cells: {valid}")
        try:
            pos = int(input("Your move (0-8): "))
            if pos in valid:
                return pos
            else:
                print("Invalid move, retry.")
        except ValueError:
            print("Insert a number between 0 and 8.")


def ai_move(net, board):
    state_tensor = state_to_tensor(board)
    with torch.no_grad():
        q = net(state_tensor)
        q_masked = mask_invalid(q, board)
        action = int(torch.argmax(q_masked).item())
    return action


def main():
    # Load model
    net = DQN().to(DEVICE)
    net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    net.eval()

    env = TicTacToeEnv(opponent_policy="random")  # not used in this script
    board = env.reset(agent_starts=False)  # human starts

    print("You're 'X'. AI is 'O'.")
    print_board(board)

    done = False

    while not done:
        # HUMAN
        a = human_move(board)
        board[a] = 1  # human = X
        print_board(board)

        winner = env._check_winner(board)
        if winner is not None:
            if winner == 1:
                print("You win!")
            elif winner == 0:
                print("Draw!")
            else:
                print("AI wins.")
            break

        # AI
        print("AI thinking...")
        a_ai = ai_move(net, board)
        print(f"AI chooses: {a_ai}")
        board[a_ai] = -1  # AI = O
        print_board(board)

        winner = env._check_winner(board)
        if winner is not None:
            if winner == -1:
                print("AI wins.")
            elif winner == 0:
                print("Draw!")
            else:
                print("You win!")
            break


if __name__ == "__main__":
    main()
