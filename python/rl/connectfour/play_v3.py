import argparse

import torch

from model_v3 import ConnectFourEnvV3, ConnectFourQNetV3


def print_board(board) -> None:
    symbols = {0: ".", 1: "X", -1: "O"}
    print("\n0 1 2 3 4 5 6")
    for row in board:
        print(" ".join(symbols[int(cell)] for cell in row))
    print()


def select_model_action(model: ConnectFourQNetV3, env: ConnectFourEnvV3, player: int) -> int:
    state = torch.from_numpy(env.encode(player)).unsqueeze(0)
    valid_moves = env.valid_actions()

    with torch.no_grad():
        q_values = model(state).squeeze(0)
        q_values = q_values.clone()
        invalid = torch.ones_like(q_values, dtype=torch.bool)
        invalid[valid_moves] = False
        q_values[invalid] = -torch.inf
        return int(torch.argmax(q_values).item())


def get_human_action(env: ConnectFourEnvV3) -> int:
    valid_moves = env.valid_actions()
    while True:
        try:
            action = int(input(f"Choose a column {valid_moves}: "))
        except ValueError:
            print("Enter a number from 0 to 6.")
            continue

        if action in valid_moves:
            return action

        print("That column is full or out of range.")


def main(model_path: str) -> None:
    env = ConnectFourEnvV3()
    model = ConnectFourQNetV3()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    human = 1
    ai = -1
    current = human

    print(f"Loaded {model_path}")
    print_board(env.board)

    while True:
        if current == human:
            action = get_human_action(env)
        else:
            action = select_model_action(model, env, ai)
            print(f"AI chooses column {action}")

        result = env.step(action, current)
        print_board(env.board)

        if result.done:
            if result.info["winner"] == human:
                print("You win.")
            elif result.info["winner"] == ai:
                print("AI wins.")
            else:
                print("Draw.")
            break

        current *= -1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Connect Four against the v3 model")
    parser.add_argument("--model-path", default="connect4_dqn_v3.pth")
    args = parser.parse_args()
    main(args.model_path)
