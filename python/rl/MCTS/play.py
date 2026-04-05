from pathlib import Path

import numpy as np
import torch

from AlphaMCTS import AlphaMCTS, ResNet
from ConnectFour import Game


def is_connect_four_like(game: Game) -> bool:
    return (
        game.action_size == game.column_count
        and game.action_size != game.row_count * game.column_count
    )


def get_latest_checkpoint() -> Path:
    checkpoints = sorted(
        Path(__file__).resolve().parent.glob("model_*.pt"),
        key=lambda path: int(path.stem.split("_")[1]),
    )
    if not checkpoints:
        raise FileNotFoundError(
            "No trained model checkpoints found. Train the model first."
        )
    return checkpoints[-1]


def load_trained_model(game: Game, checkpoint_path: Path) -> ResNet:
    model = ResNet(game, 4, 64)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def print_board(game: Game, state: np.ndarray) -> None:
    symbols = {1: "X", -1: "O", 0: "."}
    print("\nBoard:")
    for row in state:
        print(" ".join(symbols[int(x)] for x in row))

    print("\nActions:")
    if is_connect_four_like(game):
        print(" ".join(str(i) for i in range(game.action_size)))
    else:
        positions = np.arange(game.action_size).reshape(
            game.row_count, game.column_count
        )
        for row in positions:
            print(" ".join(str(x) for x in row))
    print()


def human_move(game: Game, state: np.ndarray) -> int:
    valid_moves = game.get_valid_moves(state)
    max_action = game.action_size - 1

    if is_connect_four_like(game):
        prompt = f"Choose your column (0-{max_action}): "
        invalid_msg = f"Please enter an integer from 0 to {max_action}."
        range_msg = f"Move out of range. Choose a number from 0 to {max_action}."
        occupied_msg = "That column is full. Try again."
    else:
        prompt = f"Choose your move (0-{max_action}): "
        invalid_msg = f"Please enter an integer from 0 to {max_action}."
        range_msg = f"Move out of range. Choose a number from 0 to {max_action}."
        occupied_msg = "That square is already occupied. Try again."

    while True:
        try:
            action = int(input(prompt).strip())
        except ValueError:
            print(invalid_msg)
            continue
        except EOFError as exc:
            raise SystemExit("\nInput closed. Exiting game.") from exc

        if action < 0 or action >= game.action_size:
            print(range_msg)
            continue

        if valid_moves[action] == 0:
            print(occupied_msg)
            continue

        return action


def ai_move(game: Game, mcts: AlphaMCTS, state: np.ndarray, player: int):
    neutral_state = game.change_perspective(state.copy(), player)
    action_probs = mcts.search(neutral_state)
    action = int(np.argmax(action_probs))
    return action, action_probs


def print_action_probs(game: Game, probs: np.ndarray) -> None:
    print("AI move probabilities:")
    if is_connect_four_like(game):
        print(probs)
    else:
        print(probs.reshape(game.row_count, game.column_count))
    print()


def main() -> None:
    game = Game()
    checkpoint_path = get_latest_checkpoint()
    model = load_trained_model(game, checkpoint_path)
    args = {
        "C": 2,
        "num_searches": 1000,
    }
    mcts = AlphaMCTS(game, args, model)

    state = game.get_initial_state()

    try:
        answer = input("Do you want to play first? [y/n]: ").strip().lower()
    except EOFError as exc:
        raise SystemExit("\nInput closed before the game started.") from exc
    human_player = 1 if answer in {"y", "yes"} else -1
    ai_player = game.get_opponent(human_player)
    player = 1

    print(f"\nLoaded checkpoint: {checkpoint_path.name}")
    print("You are", "X" if human_player == 1 else "O")
    print("AI is", "X" if ai_player == 1 else "O")

    while True:
        print_board(game, state)

        if player == human_player:
            action = human_move(game, state)
            if is_connect_four_like(game):
                print(f"You played column: {action}\n")
            else:
                print(f"You played: {action}\n")
        else:
            action, probs = ai_move(game, mcts, state, ai_player)
            if is_connect_four_like(game):
                print(f"AI played column: {action}")
            else:
                print(f"AI played: {action}")
            print_action_probs(game, probs)

        state = game.get_next_state(state.copy(), action, player)

        value, is_terminal = game.get_value_and_terminated(state, action)
        if is_terminal:
            print_board(game, state)
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
