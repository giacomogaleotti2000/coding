# ttt_env.py
#
# TicTacToe Environment Class

import numpy as np


class Game:
    def __init__(self) -> None:
        self.row_number = 3
        self.col_number = 3
        self.action_size = self.row_number * self.col_number

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_number, self.col_number))

    def get_valid_moves(self, state: np.ndarray):
        return (state.reshape(-1) == 0).astype(np.uint)

    def get_next_state(
        self, state: np.ndarray, action: np.uint, player: np.int8
    ) -> np.ndarray:
        row = action // self.row_number
        col = action % self.col_number
        state[row, col] = player
        return state

    def check_win(self, state: np.ndarray, action: np.uint) -> bool:
        if action is None:
            return False

        row = action // self.row_number
        col = action % self.col_number
        player = state[row, col]

        check = (
            np.sum(state[row, :]) == player * self.col_number
            or np.sum(state[:, col]) == player * self.row_number
            or np.sum(np.diag(state)) == player * self.row_number
            or np.sum(np.diag(np.flip(state, axis=0))) == player * self.row_number
        )

        return check

    def get_value_and_terminated(self, state: np.ndarray, action: np.uint):
        if self.check_win(state, action):
            return 1, True
        if sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: np.int8):
        return -player

    def get_opponent_value(self, value: np.uint):
        return -value

    def change_perspective(self, state: np.ndarray, player: np.int8):
        return state * player
