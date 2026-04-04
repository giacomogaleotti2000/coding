import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectFourEnv:
    def __init__(self) -> None:
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.reset()

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        return self.board.copy()

    def copy(self) -> "ConnectFourEnv":
        env = ConnectFourEnv()
        env.board = self.board.copy()
        return env

    def valid_actions(self) -> list[int]:
        return [c for c in range(self.cols) if self.board[0, c] == 0]

    def apply_action(self, action: int, player: int) -> None:
        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                return
        raise ValueError(f"Column {action} is full")

    def check_win(self, player: int) -> bool:
        board = self.board
        for row in range(self.rows):
            for col in range(self.cols - 3):
                if np.all(board[row, col : col + 4] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.cols):
                if np.all(board[row : row + 4, col] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.cols - 3):
                if all(board[row + i, col + i] == player for i in range(4)):
                    return True
        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(board[row - i, col + i] == player for i in range(4)):
                    return True
        return False

    def is_draw(self) -> bool:
        return len(self.valid_actions()) == 0

    def step(self, action: int, player: int) -> tuple[bool, int]:
        self.apply_action(action, player)
        if self.check_win(player):
            return True, player
        if self.is_draw():
            return True, 0
        return False, 0

    def encode(self, player: int) -> np.ndarray:
        own = (self.board == player).astype(np.float32)
        opp = (self.board == -player).astype(np.float32)
        player_plane = np.full((1, self.rows, self.cols), 1.0 if player == 1 else 0.0, dtype=np.float32)
        return np.concatenate([own[None, ...], opp[None, ...], player_plane], axis=0)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class PolicyValueNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.body = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        flat_dim = 96 * 6 * 7
        self.policy_head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 7),
        )
        self.value_head = nn.Sequential(
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        return self.policy_head(x), self.value_head(x)


class MCTSNode:
    def __init__(self, prior: float) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, MCTSNode] = {}

    @property
    def mean_value(self) -> float:
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(self, model: PolicyValueNet, device: torch.device, simulations: int, c_puct: float) -> None:
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct

    def evaluate_state(self, env: ConnectFourEnv, player: int) -> tuple[np.ndarray, float]:
        state = torch.from_numpy(env.encode(player)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(state)
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        valid = env.valid_actions()
        mask = np.zeros(7, dtype=np.float32)
        mask[valid] = 1.0
        policy = policy * mask
        if policy.sum() <= 0:
            policy = mask / mask.sum()
        else:
            policy /= policy.sum()
        return policy.astype(np.float32), float(value.item())

    def expand(self, node: MCTSNode, priors: np.ndarray, valid_actions: list[int]) -> None:
        for action in valid_actions:
            node.children[action] = MCTSNode(float(priors[action]))

    def select_child(self, node: MCTSNode) -> tuple[int, MCTSNode]:
        sqrt_total = math.sqrt(max(1, node.visit_count))
        best_score = -float("inf")
        best_action = -1
        best_child = None
        for action, child in node.children.items():
            q = -child.mean_value
            u = self.c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            score = q + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        return best_action, best_child

    def run(self, env: ConnectFourEnv, player: int) -> np.ndarray:
        root = MCTSNode(prior=1.0)
        priors, _ = self.evaluate_state(env, player)
        valid_actions = env.valid_actions()
        self.expand(root, priors, valid_actions)

        for _ in range(self.simulations):
            sim_env = env.copy()
            sim_player = player
            node = root
            search_path = [node]

            while node.children:
                action, node = self.select_child(node)
                done, winner = sim_env.step(action, sim_player)
                search_path.append(node)
                if done:
                    value = 0.0 if winner == 0 else 1.0
                    break
                sim_player *= -1
            else:
                done = False

            if not done:
                priors, value = self.evaluate_state(sim_env, sim_player)
                self.expand(node, priors, sim_env.valid_actions())

            for back_node in reversed(search_path):
                back_node.visit_count += 1
                back_node.value_sum += value
                value = -value

        visits = np.zeros(7, dtype=np.float32)
        for action, child in root.children.items():
            visits[action] = child.visit_count
        if visits.sum() == 0:
            visits[valid_actions] = 1.0
        return visits / visits.sum()


def print_board(board: np.ndarray) -> None:
    symbols = {0: ".", 1: "X", -1: "O"}
    print("\n0 1 2 3 4 5 6")
    for row in board:
        print(" ".join(symbols[int(cell)] for cell in row))
    print()


def get_human_action(env: ConnectFourEnv) -> int:
    valid = env.valid_actions()
    while True:
        try:
            action = int(input(f"Choose column {valid}: "))
        except ValueError:
            print("Please enter a number.")
            continue
        if action in valid:
            return action
        print("Invalid move.")


def choose_ai_action(mcts: MCTS, env: ConnectFourEnv, ai_player: int) -> int:
    policy = mcts.run(env, ai_player)
    action = int(np.argmax(policy))
    print("AI visit distribution:", np.round(policy, 3))
    return action


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PolicyValueNet().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    env = ConnectFourEnv()
    mcts = MCTS(model=model, device=device, simulations=args.simulations, c_puct=args.c_puct)

    human_player = 1 if args.human_first else -1
    ai_player = -human_player
    current_player = 1

    print(f"Loaded model from {args.model_path}")
    print(f"You are {'X' if human_player == 1 else 'O'}")
    print(f"AI is {'X' if ai_player == 1 else 'O'}")
    print_board(env.board)

    while True:
        if current_player == human_player:
            action = get_human_action(env)
        else:
            action = choose_ai_action(mcts, env, ai_player)
            print(f"AI chooses column {action}")

        done, winner = env.step(action, current_player)
        print_board(env.board)

        if done:
            if winner == 0:
                print("Draw.")
            elif winner == human_player:
                print("You win.")
            else:
                print("AI wins.")
            break

        current_player *= -1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play against the MCTS-trained Connect Four v4 model")
    parser.add_argument("--model-path", default="connect4_alphazero_v4.pth")
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--c-puct", type=float, default=1.75)
    parser.add_argument("--human-first", action="store_true")
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
