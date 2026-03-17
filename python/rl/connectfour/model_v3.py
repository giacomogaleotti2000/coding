import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class StepResult:
    board: np.ndarray
    reward: float
    done: bool
    info: dict


class ConnectFourEnvV3:
    def __init__(self) -> None:
        self.rows = 6
        self.cols = 7
        self.board = np.zeros((self.rows, self.cols), dtype=np.int8)
        self.reset()

    def reset(self) -> np.ndarray:
        self.board.fill(0)
        return self.board.copy()

    def copy(self) -> "ConnectFourEnvV3":
        clone = ConnectFourEnvV3()
        clone.board = self.board.copy()
        return clone

    def valid_actions(self) -> list[int]:
        return [col for col in range(self.cols) if self.board[0, col] == 0]

    def is_full(self) -> bool:
        return not self.valid_actions()

    def apply_action(self, action: int, player: int) -> int:
        if action not in self.valid_actions():
            raise ValueError(f"Invalid action {action}")

        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                return row

        raise RuntimeError("No valid landing row found for action")

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
                if all(board[row + offset, col + offset] == player for offset in range(4)):
                    return True

        for row in range(3, self.rows):
            for col in range(self.cols - 3):
                if all(board[row - offset, col + offset] == player for offset in range(4)):
                    return True

        return False

    def step(self, action: int, player: int) -> StepResult:
        self.apply_action(action, player)

        if self.check_win(player):
            return StepResult(self.board.copy(), 1.0, True, {"result": "win", "winner": player})

        if self.is_full():
            return StepResult(self.board.copy(), 0.0, True, {"result": "draw", "winner": 0})

        return StepResult(self.board.copy(), 0.0, False, {"result": "continue", "winner": 0})

    def encode(self, perspective_player: int) -> np.ndarray:
        own = (self.board == perspective_player).astype(np.float32)
        opp = (self.board == -perspective_player).astype(np.float32)
        turn = np.full((1, self.rows, self.cols), 1.0 if perspective_player == 1 else 0.0, dtype=np.float32)
        return np.concatenate([own[None, ...], opp[None, ...], turn], axis=0)

    def render(self) -> None:
        symbols = {0: ".", 1: "X", -1: "O"}
        print()
        for row in self.board:
            print(" ".join(symbols[int(cell)] for cell in row))
        print("0 1 2 3 4 5 6")


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


class ConnectFourQNetV3(nn.Module):
    def __init__(self, output_dim: int = 7) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(96),
            ResidualBlock(96),
            ResidualBlock(96),
        )
        feature_dim = 96 * 6 * 7
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.advantage_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)
        value = self.value_head(x)
        advantage = self.advantage_head(x)
        return value + (advantage - advantage.mean(dim=1, keepdim=True))


def valid_action_mask(board_tensor: torch.Tensor) -> torch.Tensor:
    occupancy = board_tensor[:, 0] + board_tensor[:, 1]
    top_row = occupancy[:, 0, :]
    return top_row.eq(0)


def choose_heuristic_action(env: ConnectFourEnvV3, player: int) -> int:
    valid_moves = env.valid_actions()

    for action in valid_moves:
        probe = env.copy()
        probe.apply_action(action, player)
        if probe.check_win(player):
            return action

    for action in valid_moves:
        probe = env.copy()
        probe.apply_action(action, -player)
        if probe.check_win(-player):
            return action

    center_order = [3, 2, 4, 1, 5, 0, 6]
    for action in center_order:
        if action in valid_moves:
            return action

    return random.choice(valid_moves)
