import numpy as np

from mcts import MCTS
from ttt_env import Game


def print_board(state: np.ndarray) -> None:
    symbols = {1: "X", -1: "O", 0: "."}
    print("\nBoard:")
    for row in state:
        print(" ".join(symbols[int(x)] for x in row))
    print("\nPositions:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    print()


def human_move(game: Game, state: np.ndarray) -> int:
    valid_moves = game.get_valid_moves(state)

    while True:
        try:
            action = int(input("Choose your move (0-8): ").strip())
        except ValueError:
            print("Please enter an integer from 0 to 8.")
            continue

        if action < 0 or action >= game.action_size:
            print("Move out of range. Choose a number from 0 to 8.")
            continue

        if valid_moves[action] == 0:
            print("That square is already occupied. Try again.")
            continue

        return action


def mcts_move(game: Game, mcts: MCTS, state: np.ndarray, player: int):
    # MCTS assumes the current player sees itself as +1.
    neutral_state = game.change_perspective(state.copy(), player)
    action_probs = mcts.search(neutral_state)
    action = int(np.argmax(action_probs))
    return action, action_probs


def main() -> None:
    game = Game()
    args = {
        "C": 1.41,
        "num_searches": 1000,
    }
    mcts = MCTS(game, args)

    state = game.get_initial_state()

    answer = input("Do you want to play first? [y/n]: ").strip().lower()
    human_player = 1 if answer in {"y", "yes"} else -1
    ai_player = game.get_opponent(human_player)
    player = 1  # X always starts

    print("\nYou are", "X" if human_player == 1 else "O")
    print("AI is", "X" if ai_player == 1 else "O")

    while True:
        print_board(state)

        if player == human_player:
            action = human_move(game, state)
            print(f"You played: {action}\n")
        else:
            action, probs = mcts_move(game, mcts, state, ai_player)
            print(f"AI played: {action}")
            print("AI move probabilities:")
            print(probs.reshape(3, 3))
            print()

        state = game.get_next_state(state.copy(), action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            print_board(state)
            if value == 1:
                if player == human_player:
                    print("You win!")
                else:
                    print("AI wins!")
            else:
                print("Draw!")
            break

        player = game.get_opponent(player)


if __name__ == "__main__":
    main()
