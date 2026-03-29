# Node class indicating a single node of the tree search
class Node:
    def __init__(self, game, args, state, parent=None, action_taken=None) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken

        self.children = []
        self.expandible_moves = game.get_valid_moves(state)

        self.value_sum = 0
        self.visit_count = 0


class MCTS:
    def __init__(self, game, args) -> None:
        self.game = game
        self.args = args

    def search(self, state):
        root = Node(self.game, self.args, state)
