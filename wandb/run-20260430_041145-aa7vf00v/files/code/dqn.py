# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh
#
# Modular DQN: supports CartPole (MLP, raw vector state) and Atari (CNN, stacked frames).
# Switch via --env-type {cartpole, atari}. Same file, same agent, different branches
# in the network/preprocessor/buffer factories.

import argparse
import os
import random
from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------
def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# Q-networks: pick one via build_q_network()
# ---------------------------------------------------------------------------
class MLPQNet(nn.Module):
    """Fully-connected Q-net for low-dimensional vector states (CartPole)."""

    def __init__(self, input_dim, num_actions, hidden=(128, 128)):
        super().__init__()
        layers = []
        last = input_dim
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        layers.append(nn.Linear(last, num_actions))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CNNQNet(nn.Module):
    """Nature DQN CNN for stacked 84x84 frames (Atari)."""

    def __init__(self, num_actions, in_channels=4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        # Expect x in [0, 255]; normalize here so buffer can stay uint8.
        x = x / 255.0
        return self.head(self.features(x))


def build_q_network(env_type, env, frame_stack=4):
    if env_type == "cartpole":
        input_dim = env.observation_space.shape[0]
        return MLPQNet(input_dim, env.action_space.n)
    if env_type == "atari":
        return CNNQNet(env.action_space.n, in_channels=frame_stack)
    raise ValueError(f"unknown env_type={env_type}")


# ---------------------------------------------------------------------------
# Preprocessors: identity for CartPole, frame-stack+resize for Atari
# ---------------------------------------------------------------------------
class IdentityPreprocessor:
    """Pass-through for already-vectorized states (CartPole)."""

    def reset(self, obs):
        return np.asarray(obs, dtype=np.float32)

    def step(self, obs):
        return np.asarray(obs, dtype=np.float32)


class AtariPreprocessor:
    """Grayscale + resize to 84x84 + frame stacking. Outputs uint8 to save memory."""

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def _preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self._preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(self._preprocess(obs))
        return np.stack(self.frames, axis=0)


def build_preprocessor(env_type, frame_stack=4):
    if env_type == "cartpole":
        return IdentityPreprocessor()
    if env_type == "atari":
        return AtariPreprocessor(frame_stack=frame_stack)
    raise ValueError(f"unknown env_type={env_type}")


# ---------------------------------------------------------------------------
# Replay buffers: shared interface so DQNAgent.train() doesn't branch
#   add(transition, error=None)
#   sample(batch_size) -> (states, actions, rewards, next_states, dones,
#                          indices, weights)
#   update_priorities(indices, errors)  # no-op for uniform
# ---------------------------------------------------------------------------
class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # indices / weights are unused for uniform; return placeholders for a
        # consistent API with the prioritized buffer.
        weights = np.ones(batch_size, dtype=np.float32)
        return states, actions, rewards, next_states, dones, None, weights

    def update_priorities(self, indices, errors):
        pass  # no-op


class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al., 2016). Filled in for Task 3.

    The interface matches UniformReplayBuffer so the agent code is identical.
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, transition, error=None):
        ########## YOUR CODE HERE (for Task 3) ##########
        return
        ########## END OF YOUR CODE (for Task 3) ##########

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        return
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        return
        ########## END OF YOUR CODE (for Task 3) ##########


def build_replay_buffer(args):
    if args.use_per:
        return PrioritizedReplayBuffer(args.memory_size, alpha=args.per_alpha, beta=args.per_beta)
    return UniformReplayBuffer(args.memory_size)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------
class DQNAgent:
    def __init__(self, args):
        self.args = args
        self.env_type = args.env_type
        self.env = gym.make(args.env_name, render_mode="rgb_array")
        self.test_env = gym.make(args.env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.preprocessor = build_preprocessor(self.env_type, frame_stack=args.frame_stack)
        self.test_preprocessor = build_preprocessor(self.env_type, frame_stack=args.frame_stack)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Init] env={args.env_name} type={self.env_type} device={self.device}")

        self.q_net = build_q_network(self.env_type, self.env, args.frame_stack).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = build_q_network(self.env_type, self.env, args.frame_stack).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.memory = build_replay_buffer(args)

        # Hyperparams
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        # Counters / bookkeeping
        self.env_count = 0
        self.train_count = 0
        self.best_reward = args.best_reward_init
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _state_to_tensor(self, state):
        """Numpy state -> batched float tensor on device."""
        t = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
        return t

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            q_values = self.q_net(self._state_to_tensor(state))
        return q_values.argmax().item()

    # -----------------------------------------------------------------------
    # Main loop
    # -----------------------------------------------------------------------
    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self.preprocessor.reset(obs)
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = self.preprocessor.step(next_obs)

                self.memory.add((state, action, reward, next_state, float(done)))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                if self.env_count % 1000 == 0:
                    print(
                        f"[Collect] Ep:{ep} Step:{step_count} SC:{self.env_count} "
                        f"UC:{self.train_count} Eps:{self.epsilon:.4f}"
                    )
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })

            print(
                f"[Eval] Ep:{ep} TotalReward:{total_reward} SC:{self.env_count} "
                f"UC:{self.train_count} Eps:{self.epsilon:.4f}"
            )
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })

            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved checkpoint -> {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    model_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"New best -> {model_path} reward={eval_reward}")
                print(f"[TrueEval] Ep:{ep} EvalReward:{eval_reward:.2f}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                })

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.test_preprocessor.reset(obs)
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                action = self.q_net(self._state_to_tensor(state)).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.test_preprocessor.step(next_obs)
        return total_reward

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------
    def train(self):
        if len(self.memory) < self.replay_start_size:
            return
        if len(self.memory) < self.batch_size:
            return

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # Sample (uniform or PER, same interface)
        states, actions, rewards, next_states, dones, indices, weights = \
            self.memory.sample(self.batch_size)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device)

        # Q(s, a)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # TD target: r + gamma * max_a' Q_target(s', a') * (1 - done)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + self.gamma * next_q * (1.0 - dones)

        td_errors = target - q_values
        # Importance-sampling weighted Huber loss (weights are all 1 for uniform)
        loss = (weights * F.smooth_l1_loss(q_values, target, reduction="none")).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        # PER bookkeeping (no-op for uniform)
        if indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(
                f"[Train #{self.train_count}] Loss:{loss.item():.4f} "
                f"Qmean:{q_values.mean().item():.3f} Qstd:{q_values.std().item():.3f}"
            )
            wandb.log({
                "Loss": loss.item(),
                "Q mean": q_values.mean().item(),
                "Q std": q_values.std().item(),
                "TD error abs mean": td_errors.abs().mean().item(),
                "Update Count": self.train_count,
            })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    # Environment
    p.add_argument("--env-type", choices=["cartpole", "atari"], default="cartpole")
    p.add_argument("--env-name", type=str, default="CartPole-v1")
    p.add_argument("--frame-stack", type=int, default=4, help="Atari only")
    # Training
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--memory-size", type=int, default=10000)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--discount-factor", type=float, default=0.99)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-decay", type=float, default=0.995)
    p.add_argument("--epsilon-min", type=float, default=0.05)
    p.add_argument("--target-update-frequency", type=int, default=100)
    p.add_argument("--replay-start-size", type=int, default=1000)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--train-per-step", type=int, default=1)
    p.add_argument("--best-reward-init", type=float, default=0.0,
                   help="0 for CartPole, -21 for Pong")
    # Replay buffer
    p.add_argument("--use-per", action="store_true", help="Use Prioritized Experience Replay (Task 3)")
    p.add_argument("--per-alpha", type=float, default=0.6)
    p.add_argument("--per-beta", type=float, default=0.4)
    # Logging / saving
    p.add_argument("--save-dir", type=str, default="./results")
    p.add_argument("--wandb-project", type=str, default="DLP-Lab5-DQN-CartPole")
    p.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, save_code=True, config=vars(args))
    agent = DQNAgent(args)
    agent.run(episodes=args.episodes)
