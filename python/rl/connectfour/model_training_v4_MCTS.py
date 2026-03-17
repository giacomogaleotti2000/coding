import argparse
import math
import random
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


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
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        return policy_logits, value


@dataclass
class SelfPlaySample:
    state: np.ndarray
    policy: np.ndarray
    value: float


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer = deque(maxlen=capacity)

    def extend(self, samples: list[SelfPlaySample]) -> None:
        self.buffer.extend(samples)

    def sample(self, batch_size: int) -> list[SelfPlaySample]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


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
    def __init__(
        self,
        model: PolicyValueNet,
        device: torch.device,
        simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_eps: float,
    ) -> None:
        self.model = model
        self.device = device
        self.simulations = simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_eps = dirichlet_eps

    def evaluate_state(self, env: ConnectFourEnv, player: int) -> tuple[np.ndarray, float]:
        state = torch.from_numpy(env.encode(player)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy_logits, value = self.model(state)
        policy = torch.softmax(policy_logits[0], dim=0).cpu().numpy()
        valid = env.valid_actions()
        mask = np.zeros(7, dtype=np.float32)
        mask[valid] = 1.0
        policy = policy * mask
        policy_sum = policy.sum()
        if policy_sum <= 0:
            policy = mask / mask.sum()
        else:
            policy /= policy_sum
        return policy.astype(np.float32), float(value.item())

    def add_exploration_noise(self, priors: np.ndarray, valid_actions: list[int]) -> np.ndarray:
        if not valid_actions:
            return priors
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions)).astype(np.float32)
        mixed = priors.copy()
        for idx, action in enumerate(valid_actions):
            mixed[action] = (1 - self.dirichlet_eps) * priors[action] + self.dirichlet_eps * noise[idx]
        mixed /= mixed.sum()
        return mixed

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

    def run(self, env: ConnectFourEnv, player: int, add_noise: bool) -> np.ndarray:
        root = MCTSNode(prior=1.0)
        priors, _ = self.evaluate_state(env, player)
        valid_actions = env.valid_actions()
        if add_noise:
            priors = self.add_exploration_noise(priors, valid_actions)
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
                    if winner == 0:
                        value = 0.0
                    else:
                        value = 1.0
                    break
                sim_player *= -1
            else:
                done = False
                winner = 0

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


def sample_action_from_policy(policy: np.ndarray, temperature: float, valid_actions: list[int]) -> int:
    if temperature <= 1e-6:
        return int(np.argmax(policy))
    tempered = np.zeros_like(policy)
    tempered[valid_actions] = np.power(policy[valid_actions], 1.0 / temperature)
    tempered_sum = tempered.sum()
    if tempered_sum <= 0:
        tempered[valid_actions] = 1.0 / len(valid_actions)
    else:
        tempered /= tempered_sum
    return int(np.random.choice(len(policy), p=tempered))


def play_self_play_game(
    env: ConnectFourEnv,
    mcts: MCTS,
    temperature_moves: int,
) -> tuple[list[SelfPlaySample], int]:
    env.reset()
    player = 1
    history: list[tuple[np.ndarray, np.ndarray, int]] = []
    move_count = 0

    while True:
        move_count += 1
        state = env.encode(player)
        policy = mcts.run(env, player, add_noise=True)
        valid_actions = env.valid_actions()
        temperature = 1.0 if move_count <= temperature_moves else 0.0
        action = sample_action_from_policy(policy, temperature, valid_actions)
        history.append((state, policy, player))
        done, winner = env.step(action, player)
        if done:
            samples = []
            for hist_state, hist_policy, hist_player in history:
                if winner == 0:
                    value = 0.0
                elif winner == hist_player:
                    value = 1.0
                else:
                    value = -1.0
                samples.append(SelfPlaySample(hist_state, hist_policy, value))
            return samples, move_count
        player *= -1


def train_on_buffer(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    buffer: ReplayBuffer,
    batch_size: int,
    batches_per_iteration: int,
    device: torch.device,
) -> tuple[float, float]:
    if len(buffer) < batch_size:
        return 0.0, 0.0

    model.train()
    policy_losses = []
    value_losses = []

    for _ in range(batches_per_iteration):
        batch = buffer.sample(batch_size)
        states = torch.tensor(np.stack([sample.state for sample in batch]), dtype=torch.float32, device=device)
        target_policy = torch.tensor(np.stack([sample.policy for sample in batch]), dtype=torch.float32, device=device)
        target_value = torch.tensor([sample.value for sample in batch], dtype=torch.float32, device=device)

        policy_logits, value_pred = model(states)
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()
        value_loss = F.mse_loss(value_pred.squeeze(1), target_value)
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        policy_losses.append(float(policy_loss.item()))
        value_losses.append(float(value_loss.item()))

    return float(np.mean(policy_losses)), float(np.mean(value_losses))


def evaluate_against_random(
    env: ConnectFourEnv,
    model: PolicyValueNet,
    device: torch.device,
    games: int,
    simulations: int,
    c_puct: float,
) -> tuple[int, int, int]:
    evaluator = MCTS(
        model=model,
        device=device,
        simulations=simulations,
        c_puct=c_puct,
        dirichlet_alpha=0.3,
        dirichlet_eps=0.0,
    )
    wins = 0
    losses = 0
    draws = 0

    for _ in range(games):
        env.reset()
        player = 1
        az_player = random.choice([1, -1])
        while True:
            if player == az_player:
                policy = evaluator.run(env, player, add_noise=False)
                action = int(np.argmax(policy))
            else:
                action = random.choice(env.valid_actions())
            done, winner = env.step(action, player)
            if done:
                if winner == 0:
                    draws += 1
                elif winner == az_player:
                    wins += 1
                else:
                    losses += 1
                break
            player *= -1
    return wins, losses, draws


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file Connect Four AlphaZero-style v4")
    parser.add_argument("--model-path", default="connect4_alphazero_v4.pth")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=35)
    parser.add_argument("--games-per-iteration", type=int, default=30)
    parser.add_argument("--simulations", type=int, default=80)
    parser.add_argument("--eval-simulations", type=int, default=40)
    parser.add_argument("--temperature-moves", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--batches-per-iteration", type=int, default=40)
    parser.add_argument("--buffer-size", type=int, default=75000)
    parser.add_argument("--lr", type=float, default=7.5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--c-puct", type=float, default=1.75)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--eval-games", type=int, default=24)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring notebook arguments: {' '.join(unknown)}")
    return args


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = ConnectFourEnv()
    model = PolicyValueNet().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    buffer = ReplayBuffer(args.buffer_size)
    best_score = -1.0
    start_time = time.time()

    for iteration in range(1, args.iterations + 1):
        model.eval()
        mcts = MCTS(
            model=model,
            device=device,
            simulations=args.simulations,
            c_puct=args.c_puct,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
        )

        generated = []
        game_lengths = []
        for _ in range(args.games_per_iteration):
            samples, move_count = play_self_play_game(env, mcts, args.temperature_moves)
            generated.extend(samples)
            game_lengths.append(move_count)
        buffer.extend(generated)

        policy_loss, value_loss = train_on_buffer(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            batch_size=args.batch_size,
            batches_per_iteration=args.batches_per_iteration,
            device=device,
        )

        wins, losses, draws = evaluate_against_random(
            env=env,
            model=model,
            device=device,
            games=args.eval_games,
            simulations=args.eval_simulations,
            c_puct=args.c_puct,
        )
        score = (wins + 0.5 * draws) / max(1, args.eval_games)
        elapsed = time.time() - start_time
        avg_len = float(np.mean(game_lengths)) if game_lengths else 0.0

        print(
            f"iter={iteration}/{args.iterations} "
            f"buffer={len(buffer)} "
            f"samples={len(generated)} "
            f"avg_moves={avg_len:.1f} "
            f"policy_loss={policy_loss:.4f} "
            f"value_loss={value_loss:.4f} "
            f"eval_random={wins}-{losses}-{draws} "
            f"elapsed={elapsed/60:.1f}m"
        )

        if score > best_score:
            best_score = score
            torch.save(model.state_dict(), args.model_path)
            print(f"saved best model to {args.model_path} with score={best_score:.3f}")


if __name__ == "__main__":
    train(parse_args())
