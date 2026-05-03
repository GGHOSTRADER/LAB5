# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL - Enhanced DQN (Task 3)
# Flags: --double-dqn, --per, --n-step N, --huber-loss, --dueling-dqn, --reward-clip, --noisy-net

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
import math
from collections import deque
import wandb
import argparse

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DQN(nn.Module):
    def __init__(self, num_actions, input_channels=None, input_dim=None, dueling=False, noisy=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.noisy = noisy

        def linear_layer(in_f, out_f):
            return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)

        if input_channels is not None:
            self.features = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            feature_dim = 64 * 7 * 7
            self.is_cnn = True
        else:
            in_dim = input_dim if input_dim is not None else 4
            self.features = nn.Sequential(
                linear_layer(in_dim, 128),
                nn.ReLU(),
                linear_layer(128, 128),
                nn.ReLU(),
            )
            feature_dim = 128
            self.is_cnn = False

        if dueling:
            # Value stream: estimates V(s)
            self.value_stream = nn.Sequential(
                linear_layer(feature_dim, 512),
                nn.ReLU(),
                linear_layer(512, 1),
            )
            # Advantage stream: estimates A(s,a)
            self.advantage_stream = nn.Sequential(
                linear_layer(feature_dim, 512),
                nn.ReLU(),
                linear_layer(512, num_actions),
            )
        else:
            self.fc = nn.Sequential(
                linear_layer(feature_dim, 512),
                nn.ReLU(),
                linear_layer(512, num_actions),
            )

    def forward(self, x):
        if self.is_cnn:
            x = x / 255.0
        features = self.features(x)

        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            return self.fc(features)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()


class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        # Crop out scoreboard (top) and bottom border
        cropped = gray[34:194, :]   # keeps main gameplay area
        resized = cv2.resize(cropped, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class UniformReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        weights = np.ones(batch_size, dtype=np.float32)
        indices = None
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, indices, errors):
        pass

    def __len__(self):
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / 500_000
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0
        self.pos = 0

    def add(self, transition):
        priority = self.max_priority
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        priorities = self.priorities[:len(self.buffer)]
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=False)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return states, actions, rewards, next_states, dones, indices, weights.astype(np.float32)

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors):
            if idx < len(self.buffer):
                p = (abs(error) + 1e-5) ** self.alpha
                self.priorities[idx] = p
                self.max_priority = max(self.max_priority, p)

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, env_name="ALE/Pong-v5", args=None):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        obs_shape = self.env.observation_space.shape
        self.is_atari = len(obs_shape) == 3
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        # Enhancement flags
        self.use_double  = args.double_dqn
        self.use_per     = args.per
        self.n_step      = args.n_step
        self.use_huber   = args.huber_loss
        self.use_dueling = args.dueling_dqn
        self.reward_clip = args.reward_clip
        self.use_noisy   = args.noisy_net

        if self.is_atari:
            self.q_net      = DQN(self.num_actions, input_channels=4, dueling=self.use_dueling, noisy=self.use_noisy).to(self.device)
            self.target_net = DQN(self.num_actions, input_channels=4, dueling=self.use_dueling, noisy=self.use_noisy).to(self.device)
        else:
            self.q_net      = DQN(self.num_actions, input_dim=obs_shape[0], dueling=self.use_dueling, noisy=self.use_noisy).to(self.device)
            self.target_net = DQN(self.num_actions, input_dim=obs_shape[0], dueling=self.use_dueling, noisy=self.use_noisy).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Replay buffer
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(capacity=args.memory_size)
        else:
            self.memory = UniformReplayBuffer(capacity=args.memory_size)

        # N-step buffer
        self.n_step_buffer = deque(maxlen=self.n_step)

        self.batch_size              = args.batch_size
        self.gamma                   = args.discount_factor
        self.epsilon                 = args.epsilon_start
        self.epsilon_decay           = args.epsilon_decay
        self.epsilon_min             = args.epsilon_min
        self.env_count               = 0
        self.train_count             = 0
        self.best_reward             = -21
        self.max_episode_steps       = args.max_episode_steps
        self.replay_start_size       = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step          = args.train_per_step
        self.save_dir                = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Milestone checkpoints for Task 3
        self.milestones = {600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000}
        self.saved_milestones = set()

    def _obs_to_state(self, obs, reset=False):
        if self.is_atari:
            return self.preprocessor.reset(obs) if reset else self.preprocessor.step(obs)
        return np.asarray(obs, dtype=np.float32)

    def _get_n_step_transition(self):
        init_state, init_action = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
        n_step_return = sum(
            self.gamma ** i * self.n_step_buffer[i][2]
            for i in range(len(self.n_step_buffer))
        )
        final_next_state = self.n_step_buffer[-1][3]
        final_done       = self.n_step_buffer[-1][4]
        return (init_state, init_action, n_step_return, final_next_state, final_done)

    def select_action(self, state):
        if not self.use_noisy and random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def run(self, episodes=10000):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = self._obs_to_state(obs, reset=True)
            done = False
            total_reward = 0
            step_count = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                # Resample noise before making a step
                if self.use_noisy:
                    self.q_net.reset_noise()

                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                # Reward clipping
                if self.reward_clip:
                    reward = np.clip(reward, -1, 1)

                next_state = self._obs_to_state(next_obs)

                self.n_step_buffer.append((state, action, reward, next_state, float(done)))
                if len(self.n_step_buffer) == self.n_step:
                    self.memory.add(self._get_n_step_transition())

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # Milestone checkpoints
                for ms in self.milestones:
                    if self.env_count >= ms and ms not in self.saved_milestones:
                        path = os.path.join(self.save_dir, f"model_step{ms}.pt")
                        torch.save(self.q_net.state_dict(), path)
                        print(f"[Milestone] Saved {path}")
                        self.saved_milestones.add(ms)

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep:{ep} SC:{self.env_count} UC:{self.train_count} Eps:{self.epsilon:.4f}")
                    wandb.log({
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })

            # Flush remaining n-step buffer
            while len(self.n_step_buffer) > 0:
                self.memory.add(self._get_n_step_transition())
                self.n_step_buffer.popleft()

            print(f"[Eval] Ep:{ep} Reward:{total_reward} SC:{self.env_count} Eps:{self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
            })

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), path)
                    print(f"New best: {eval_reward} -> {path}")
                print(f"[TrueEval] Ep:{ep} EvalReward:{eval_reward:.2f} SC:{self.env_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Eval Reward": eval_reward,
                })

    def evaluate(self, num_episodes=10):
        total = 0
        self.q_net.eval() # Set to eval mode to use the deterministic weights from NoisyNet
        for _ in range(num_episodes):
            obs, _ = self.test_env.reset()
            state = self._obs_to_state(obs, reset=True)
            done = False
            ep_reward = 0
            while not done:
                state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                ep_reward += reward
                state = self._obs_to_state(next_obs)
            total += ep_reward
        self.q_net.train() # Revert to train mode
        return total / num_episodes

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        # Resample noise before making a train pass
        if self.use_noisy:
            self.q_net.reset_noise()
            self.target_net.reset_noise()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        states, actions, rewards, next_states, dones, indices, weights = self.memory.sample(self.batch_size)

        states      = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones, dtype=torch.float32).to(self.device)
        weights     = torch.tensor(weights, dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            else:
                next_q = self.target_net(next_states).max(dim=1)[0]
            target = rewards + (self.gamma ** self.n_step) * next_q * (1.0 - dones)

        td_errors = q_values - target

        if self.use_huber:
            # Huber loss (smooth L1) — clips large TD errors
            loss = (weights * F.smooth_l1_loss(q_values, target.detach(), reduction='none')).mean()
        else:
            # Weighted MSE
            loss = (weights * td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.abs().detach().cpu().numpy())

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss:{loss.item():.4f} Q mean:{q_values.mean().item():.3f}")
            wandb.log({
                "Loss": loss.item(),
                "Q mean": q_values.mean().item(),
                "Q std": q_values.std().item(),
                "TD error abs mean": td_errors.abs().mean().item(),
                "Update Count": self.train_count,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-name",    type=str,   default="ALE/Pong-v5")
    parser.add_argument("--save-dir",    type=str,   default="./results_enhanced")
    parser.add_argument("--wandb-run-name", type=str, default="pong-enhanced")
    parser.add_argument("--wandb-project",  type=str, default="DLP-Lab5-DQN-CartPole")
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--memory-size", type=int,   default=200_000)
    parser.add_argument("--lr",          type=float, default=0.00025)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start",   type=float, default=1.0)
    parser.add_argument("--epsilon-decay",   type=float, default=0.999975)
    parser.add_argument("--epsilon-min",     type=float, default=0.01)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size",        type=int, default=50_000)
    parser.add_argument("--max-episode-steps",        type=int, default=10_000)
    parser.add_argument("--train-per-step",           type=int, default=1)
    parser.add_argument("--episodes",                 type=int, default=10_000)

    # Enhancement flags
    parser.add_argument("--double-dqn",  action="store_true", default=False)
    parser.add_argument("--per",         action="store_true", default=False)
    parser.add_argument("--n-step",      type=int,            default=1)
    parser.add_argument("--huber-loss",  action="store_true", default=False)
    parser.add_argument("--dueling-dqn", action="store_true", default=False)
    parser.add_argument("--reward-clip", action="store_true", default=False)
    parser.add_argument("--noisy-net",   action="store_true", default=False)

    args = parser.parse_args()

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        save_code=True,
        config=vars(args)
    )
    agent = DQNAgent(env_name=args.env_name, args=args)
    agent.run(episodes=args.episodes)