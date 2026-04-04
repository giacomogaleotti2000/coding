# TicTacToe.py
#
# TicTacToe Environment Class

import numpy as np


class Game:
    def __init__(self) -> None:
        self.row_number = 6
        self.col_number = 7
        self.action_size = self.col_number

    def get_initial_state(self) -> np.ndarray:
        return np.zeros((self.row_number, self.col_number))

    def get_valid_moves(self, state: np.ndarray):
        return (state[0, :] == 0).astype(int)

    def get_next_state(self, state: np.ndarray, action: int, player: int) -> np.ndarray:
        for row in range(self.row_number - 1, -1, -1):
            if state[row, action] == 0:
                state[row, action] = player
                break
        return state

    def check_win(self, state: np.ndarray, action: int) -> bool:
        if action is None:
            return False

        # Find the row of the last piece played in this column
        row = None
        for r in range(self.row_number):
            if state[r, action] != 0:
                row = r
                break

        if row is None:
            return False

        player = state[row, action]

        def count_direction(delta_row: int, delta_col: int) -> int:
            count = 1

            r, c = row + delta_row, action + delta_col
            while (
                0 <= r < self.row_number
                and 0 <= c < self.col_number
                and state[r, c] == player
            ):
                count += 1
                r += delta_row
                c += delta_col

            r, c = row - delta_row, action - delta_col
            while (
                0 <= r < self.row_number
                and 0 <= c < self.col_number
                and state[r, c] == player
            ):
                count += 1
                r -= delta_row
                c -= delta_col

            return count

        return (
            count_direction(0, 1) >= 4  # horizontal
            or count_direction(1, 0) >= 4  # vertical
            or count_direction(1, 1) >= 4  # diagonal \
            or count_direction(1, -1) >= 4  # diagonal /
        )

    def get_value_and_terminated(self, state: np.ndarray, action: int):
        if self.check_win(state, action):
            return 1, True
        if np.sum(self.get_valid_moves(state)) == 0:
            return 0, True
        return 0, False

    def get_opponent(self, player: int):
        return -player

    def get_opponent_value(self, value: int):
        return -value

    def change_perspective(self, state: np.ndarray, player: int):
        return state * player

    def get_encoded_state(self, state: np.ndarray):
        encoded_state = np.stack(
            (
                state == -1,
                state == 0,
                state == 1,
            )  # creating three layers, with just 1 and 0s, where we track each value (0,1,-1)
        ).astype(np.float32)

        return encoded_state
