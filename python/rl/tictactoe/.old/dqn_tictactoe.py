# dqn_tictactoe.py
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim

# --------- Hyperparams ---------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NUM_EPISODES = 30000
BATCH_SIZE = 64*4
GAMMA = 0.99
LR = 1e-4
REPLAY_SIZE = 50000
MIN_REPLAY_SIZE = 500
TARGET_UPDATE_EVERY = 500  # steps
EPS_START = 1.0
EPS_END = 0.
EPS_DECAY = 8000  # linear decay steps
MAX_STEPS_PER_EPISODE = 9
PRINT_EVERY = 500
SAVE_PATH = "dqn_tictactoe.pth"

Transition = namedtuple('Transition', ('s', 'a', 'r', 's2', 'done', 'mask_next'))

# --------- Environment ---------
class TicTacToeEnv:
    """
    Agent plays '1' (X). Opponent plays '-1' (O).
    State: numpy array shape (9,) with values in {-1,0,1}.
    step(action): agent plays action (0..8), then opponent plays (random).
    returns: next_state, reward, done, info
    reward: +1 win for agent, -1 loss, 0 otherwise (draw -> 0)
    """
    def __init__(self, opponent_policy="random"):
        self.board = np.zeros(9, dtype=int)
        self.current_player = 1  # agent starts by default
        self.opponent_policy = opponent_policy

    def reset(self, agent_starts=True):
        self.board[:] = 0
        self.current_player = 1 if agent_starts else -1
        # If opponent starts, let them play one move then return state to agent
        if not agent_starts:
            self._opponent_move()
        return self._get_state()

    def _get_state(self):
        return self.board.copy()

    def _available_actions(self):
        return np.where(self.board == 0)[0]

    def _check_winner(self, b):
        lines = [
            (0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)
        ]
        for (i,j,k) in lines:
            s = b[i] + b[j] + b[k]
            if s == 3:
                return 1
            if s == -3:
                return -1
        if np.all(b != 0):
            return 0  # draw
        return None  # game not finished

    def _opponent_move(self):
        avail = self._available_actions()
        if len(avail) == 0:
            return
        if self.opponent_policy == "random":
            a = np.random.choice(avail)
        else:
            a = np.random.choice(avail)
        self.board[a] = -1

    def step(self, action):
        """
        Agent action -> apply -> check -> opponent -> check.
        Returns: next_state, reward, done, info
        """
        info = {}
        # illegal move
        if self.board[action] != 0:
            # punish illegal moves harshly
            return self._get_state(), -1.0, True, {"illegal": True}

        # agent plays
        self.board[action] = 1
        winner = self._check_winner(self.board)
        if winner is not None:
            if winner == 1:
                return self._get_state(), 1.0, True, info
            elif winner == 0:  # draw
                return self._get_state(), 0.0, True, info

        # opponent plays
        self._opponent_move()
        winner = self._check_winner(self.board)
        if winner is not None:
            if winner == -1:
                return self._get_state(), -1.0, True, info
            elif winner == 0:
                return self._get_state(), 0.0, True, info

        # game continues
        return self._get_state(), 0.0, False, info

# --------- Replay Buffer ---------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)

# --------- DQN Model ---------
class DQN(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --------- Helpers ---------
def state_to_tensor(state):
    # state: numpy array shape (9,) with -1,0,1
    # convert to float tensor
    return torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)

def mask_invalid(q_values, state):
    # q_values: tensor shape (..., 9)
    # state: numpy array shape (9,) where 0 means available
    mask = (state != 0)  # True where not available
    q_values = q_values.clone()
    q_values[..., mask] = -1e9
    return q_values

# --------- Training loop ---------
def train():
    env = TicTacToeEnv(opponent_policy="random")
    policy_net = DQN().to(DEVICE)
    target_net = DQN().to(DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    replay = ReplayBuffer(REPLAY_SIZE)

    # populate initial replay with random games
    steps_done = 0
    while len(replay) < MIN_REPLAY_SIZE:
        state = env.reset(agent_starts=True)
        done = False
        while not done:
            avail = env._available_actions()
            a = np.random.choice(avail)
            s2, r, done, _ = env.step(a)
            mask_next = (s2 != 0)
            replay.push(state.copy(), a, r, s2.copy(), done, mask_next.copy())
            state = s2
    print(f"Replay buffer initialized with {len(replay)} transitions.")

    eps = EPS_START
    eps_decay_per_step = (EPS_START - EPS_END) / EPS_DECAY

    losses = []
    total_steps = 0
    win_count = 0
    draw_count = 0
    loss_count = 0

    for ep in range(1, NUM_EPISODES + 1):
        # alternate who starts occasionally to diversify
        agent_starts = True if random.random() < 0.5 else False
        state = env.reset(agent_starts=agent_starts)
        done = False
        steps = 0

        while not done and steps < MAX_STEPS_PER_EPISODE:
            steps += 1
            total_steps += 1
            # epsilon-greedy action selection with mask
            state_tensor = state_to_tensor(state)
            with torch.no_grad():
                qvals = policy_net(state_tensor)  # shape (1,9)
                qvals_masked = mask_invalid(qvals, state)
            if random.random() < eps:
                action = int(np.random.choice(np.where(state == 0)[0]))
            else:
                action = int(torch.argmax(qvals_masked).item())

            # step environment (this will apply opponent move inside)
            next_state, reward, done, info = env.step(action)

            # store transition
            mask_next = (next_state != 0)
            replay.push(state.copy(), action, reward, next_state.copy(), done, mask_next.copy())

            state = next_state

            # sample and train
            if len(replay) >= MIN_REPLAY_SIZE:
                batch = replay.sample(BATCH_SIZE)
                s_batch = torch.tensor(np.stack(batch.s), dtype=torch.float32, device=DEVICE)  # (B,9)
                a_batch = torch.tensor(batch.a, dtype=torch.long, device=DEVICE).unsqueeze(1)  # (B,1)
                r_batch = torch.tensor(batch.r, dtype=torch.float32, device=DEVICE).unsqueeze(1)  # (B,1)
                s2_batch = torch.tensor(np.stack(batch.s2), dtype=torch.float32, device=DEVICE)  # (B,9)
                done_batch = torch.tensor(batch.done, dtype=torch.bool, device=DEVICE).unsqueeze(1)
                mask_next_batch = torch.tensor(np.stack(batch.mask_next), dtype=torch.bool, device=DEVICE)

                # current Q(s,a)
                q_values = policy_net(s_batch)  # (B,9)
                q_s_a = q_values.gather(1, a_batch)  # (B,1)

                # compute target
                with torch.no_grad():
                    q_next_target = target_net(s2_batch)  # (B,9)
                    # mask invalid next actions (where mask_next_batch == True -> occupied => invalid)
                    invalid_mask = mask_next_batch  # True where occupied
                    q_next_target[invalid_mask] = -1e9
                    max_q_next, _ = q_next_target.max(dim=1, keepdim=True)  # (B,1)
                    y = r_batch + (GAMMA * max_q_next * (~done_batch))
                    # if done -> y = r (done mask handles it)

                loss = nn.MSELoss()(q_s_a, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            # update target network
            if total_steps % TARGET_UPDATE_EVERY == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # decay epsilon
            if eps > EPS_END:
                eps -= eps_decay_per_step
                if eps < EPS_END:
                    eps = EPS_END

            # stats
            if reward == 1.0:
                win_count += 1
            elif reward == -1.0:
                loss_count += 1
            else:
                draw_count += 1

        if ep % PRINT_EVERY == 0:
            total = win_count + loss_count + draw_count
            print(f"Ep {ep}/{NUM_EPISODES} | Steps {total_steps} | eps {eps:.3f} | avg loss {np.mean(losses[-200:]) if losses else 0:.4f}")
            print(f"  wins {win_count} ({win_count/total:.2f}) | draws {draw_count} ({draw_count/total:.2f}) | losses {loss_count} ({loss_count/total:.2f})")

    # save model
    torch.save(policy_net.state_dict(), SAVE_PATH)
    print("Training finished. Model saved to", SAVE_PATH)
    return policy_net

# --------- Evaluate ---------
def evaluate(net, episodes=1000):
    env = TicTacToeEnv(opponent_policy="random")
    net.eval()
    wins = draws = losses = 0
    for _ in range(episodes):
        state = env.reset(agent_starts=True)
        done = False
        while not done:
            with torch.no_grad():
                q = net(state_to_tensor(state))
                q_masked = mask_invalid(q, state)
                action = int(torch.argmax(q_masked).item())
            state, reward, done, _ = env.step(action)
            
            if reward == 1:
                wins += 1
            elif reward == -1:
                losses += 1
            else:
                draws += 1
    print(f"Eval over {episodes}: wins {wins}, draws {draws}, losses {losses}")
    return wins, draws, losses

if __name__ == "__main__":
    trained = train()
    evaluate(trained, episodes=1000)
