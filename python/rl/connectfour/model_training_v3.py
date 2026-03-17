import argparse
import copy
import random
from collections import deque, namedtuple
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


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

    def apply_action(self, action: int, player: int) -> None:
        if action not in self.valid_actions():
            raise ValueError(f"Invalid action {action}")

        for row in range(self.rows - 1, -1, -1):
            if self.board[row, action] == 0:
                self.board[row, action] = player
                return

        raise RuntimeError("No open slot found")

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


Transition = namedtuple(
    "Transition",
    ("state", "action", "reward", "next_state", "done", "next_valid_mask"),
)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory = deque(maxlen=capacity)

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


@dataclass
class EvalStats:
    wins: int = 0
    losses: int = 0
    draws: int = 0

    @property
    def score(self) -> float:
        total = self.wins + self.losses + self.draws
        return 0.0 if total == 0 else (self.wins + 0.5 * self.draws) / total


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


class DQNAgentV3:
    def __init__(
        self,
        lr: float = 3e-4,
        gamma: float = 0.995,
        batch_size: int = 256,
        replay_size: int = 200_000,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = ConnectFourQNetV3().to(self.device)
        self.target_net = ConnectFourQNetV3().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, weight_decay=1e-4)
        self.replay = ReplayBuffer(replay_size)

        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay_episodes = 40_000

    def epsilon(self, episode: int) -> float:
        progress = min(1.0, episode / self.eps_decay_episodes)
        return self.eps_end + (self.eps_start - self.eps_end) * (1.0 - progress) ** 2

    def select_action(
        self,
        state: torch.Tensor,
        valid_moves: list[int],
        epsilon: float,
        net: nn.Module | None = None,
    ) -> int:
        if random.random() < epsilon:
            return random.choice(valid_moves)

        model = self.policy_net if net is None else net
        with torch.no_grad():
            q_values = model(state.to(self.device)).squeeze(0).clone()
            invalid = torch.ones_like(q_values, dtype=torch.bool)
            invalid[valid_moves] = False
            q_values[invalid] = -torch.inf
            return int(torch.argmax(q_values).item())

    def optimize(self) -> float | None:
        if len(self.replay) < self.batch_size:
            return None

        batch = Transition(*zip(*self.replay.sample(self.batch_size)))
        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.tensor(batch.action, dtype=torch.long, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        next_valid_mask = torch.stack(batch.next_valid_mask).to(self.device)

        current_q = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)

        with torch.no_grad():
            next_policy_q = self.policy_net(next_state_batch).masked_fill(~next_valid_mask, -torch.inf)
            next_actions = next_policy_q.argmax(dim=1, keepdim=True)
            next_target_q = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            next_target_q = torch.where(done_batch.bool(), torch.zeros_like(next_target_q), next_target_q)
            target_q = reward_batch + self.gamma * next_target_q

        loss = F.smooth_l1_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


def board_to_tensor(env: ConnectFourEnvV3, player: int) -> torch.Tensor:
    return torch.from_numpy(env.encode(player)).unsqueeze(0).float()


def mirrored_state(state: torch.Tensor) -> torch.Tensor:
    return torch.flip(state, dims=[3])


def mirrored_mask(mask: torch.Tensor) -> torch.Tensor:
    return torch.flip(mask, dims=[0])


def play_opponent_turn(
    env: ConnectFourEnvV3,
    opponent_kind: str,
    opponent_net: nn.Module | None,
    agent: DQNAgentV3,
    player: int,
) -> tuple[bool, dict]:
    valid_moves = env.valid_actions()
    if opponent_kind == "random":
        action = random.choice(valid_moves)
    elif opponent_kind == "heuristic":
        action = choose_heuristic_action(env, player)
    elif opponent_kind == "snapshot":
        if opponent_net is None:
            raise ValueError("snapshot opponent requires a model")
        state = board_to_tensor(env, player)
        action = agent.select_action(state, valid_moves, epsilon=0.0, net=opponent_net)
    else:
        raise ValueError(f"Unsupported opponent kind: {opponent_kind}")

    result = env.step(action, player)
    return result.done, result.info


def run_training_episode(
    env: ConnectFourEnvV3,
    agent: DQNAgentV3,
    episode: int,
    opponent_kind: str,
    opponent_net: nn.Module | None,
    augmentation: bool,
) -> tuple[float, float, str]:
    env.reset()
    learner_player = random.choice([1, -1])
    current_player = 1
    done = False
    outcome = "draw"
    episode_reward = 0.0
    losses = []

    while not done:
        if current_player != learner_player:
            done, info = play_opponent_turn(env, opponent_kind, opponent_net, agent, current_player)
            if done:
                outcome = "loss" if info["winner"] == -learner_player else "draw"
            current_player *= -1
            continue

        state = board_to_tensor(env, learner_player)
        action = agent.select_action(state, env.valid_actions(), epsilon=agent.epsilon(episode))
        result = env.step(action, learner_player)
        reward = result.reward
        done = result.done

        if done:
            next_state = board_to_tensor(env, learner_player)
            next_mask = torch.zeros(env.cols, dtype=torch.bool)
            outcome = "win" if reward > 0 else "draw"
        else:
            opp_done, opp_info = play_opponent_turn(env, opponent_kind, opponent_net, agent, -learner_player)
            if opp_done:
                done = True
                reward = -1.0 if opp_info["winner"] == -learner_player else 0.0
                outcome = "loss" if reward < 0 else "draw"
            next_state = board_to_tensor(env, learner_player)
            next_mask = valid_action_mask(next_state).squeeze(0).cpu()

        agent.replay.push(state, action, reward, next_state, done, next_mask)

        if augmentation:
            agent.replay.push(
                mirrored_state(state),
                env.cols - 1 - action,
                reward,
                mirrored_state(next_state),
                done,
                mirrored_mask(next_mask),
            )

        episode_reward += reward
        loss = agent.optimize()
        if loss is not None:
            losses.append(loss)

        current_player = learner_player

    mean_loss = float(np.mean(losses)) if losses else 0.0
    return episode_reward, mean_loss, outcome


def evaluate_policy(
    env: ConnectFourEnvV3,
    agent: DQNAgentV3,
    games: int,
    opponent_kind: str,
    opponent_net: nn.Module | None,
) -> EvalStats:
    stats = EvalStats()

    for _ in range(games):
        env.reset()
        learner_player = random.choice([1, -1])
        current_player = 1

        while True:
            if current_player == learner_player:
                state = board_to_tensor(env, learner_player)
                action = agent.select_action(state, env.valid_actions(), epsilon=0.0)
                result = env.step(action, learner_player)
                if result.done:
                    if result.reward > 0:
                        stats.wins += 1
                    else:
                        stats.draws += 1
                    break
            else:
                done, info = play_opponent_turn(env, opponent_kind, opponent_net, agent, current_player)
                if done:
                    if info["winner"] == -learner_player:
                        stats.losses += 1
                    else:
                        stats.draws += 1
                    break

            current_player *= -1

    return stats


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = ConnectFourEnvV3()
    agent = DQNAgentV3(
        lr=args.lr,
        gamma=args.gamma,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
    )

    best_score = -1.0
    snapshot_net = None
    schedule = [
        ("random", args.random_episodes),
        ("heuristic", args.heuristic_episodes),
        ("snapshot", args.self_play_episodes),
    ]
    recent_outcomes = deque(maxlen=200)
    recent_losses = deque(maxlen=200)
    total_episodes = 0

    for phase_name, phase_episodes in schedule:
        if phase_episodes <= 0:
            continue

        print(f"\n=== Phase: {phase_name} ({phase_episodes} episodes) ===")
        if phase_name == "snapshot" and snapshot_net is None:
            snapshot_net = copy.deepcopy(agent.policy_net).to(agent.device)
            snapshot_net.eval()

        for _ in range(phase_episodes):
            total_episodes += 1
            _, loss, outcome = run_training_episode(
                env=env,
                agent=agent,
                episode=total_episodes,
                opponent_kind=phase_name,
                opponent_net=snapshot_net,
                augmentation=not args.no_augmentation,
            )
            recent_outcomes.append(outcome)
            recent_losses.append(loss)

            if total_episodes % args.target_sync == 0:
                agent.sync_target()

            if total_episodes % args.eval_every == 0:
                random_stats = evaluate_policy(env, agent, args.eval_games, "random", None)
                heuristic_stats = evaluate_policy(env, agent, args.eval_games, "heuristic", None)
                combined_score = 0.4 * random_stats.score + 0.6 * heuristic_stats.score
                outcomes = {name: recent_outcomes.count(name) for name in ["win", "loss", "draw"]}

                print(
                    f"ep={total_episodes} "
                    f"eps={agent.epsilon(total_episodes):.3f} "
                    f"loss={np.mean(recent_losses):.4f} "
                    f"train_wld={outcomes['win']}/{outcomes['loss']}/{outcomes['draw']} "
                    f"eval_random={random_stats.wins}-{random_stats.losses}-{random_stats.draws} "
                    f"eval_heuristic={heuristic_stats.wins}-{heuristic_stats.losses}-{heuristic_stats.draws}"
                )

                if combined_score > best_score:
                    best_score = combined_score
                    torch.save(agent.policy_net.state_dict(), args.model_path)
                    print(f"saved best model to {args.model_path} with score={best_score:.3f}")

                if phase_name == "snapshot" and combined_score >= args.snapshot_refresh_score:
                    snapshot_net = copy.deepcopy(agent.policy_net).to(agent.device)
                    snapshot_net.eval()
                    print("updated self-play snapshot opponent")

    if best_score < 0:
        torch.save(agent.policy_net.state_dict(), args.model_path)
        print(f"saved final model to {args.model_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-file Connect Four DQN v3 for Kaggle")
    parser.add_argument("--model-path", default="connect4_dqn_v3.pth")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--replay-size", type=int, default=200000)
    parser.add_argument("--random-episodes", type=int, default=8000)
    parser.add_argument("--heuristic-episodes", type=int, default=12000)
    parser.add_argument("--self-play-episodes", type=int, default=30000)
    parser.add_argument("--target-sync", type=int, default=1000)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-games", type=int, default=100)
    parser.add_argument("--snapshot-refresh-score", type=float, default=0.72)
    parser.add_argument("--no-augmentation", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    # Kaggle/Jupyter injects extra kernel arguments (for example `-f <connection.json>`).
    # We ignore unknown args so the script can be run unchanged in notebook cells.
    parser = build_parser()
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring notebook arguments: {' '.join(unknown)}")
    return args


if __name__ == "__main__":
    train(parse_args())
