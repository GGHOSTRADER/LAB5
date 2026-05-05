import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
import math
from collections import deque
import argparse

gym.register_envs(ale_py)

# --- 1. NoisyLinear Support ---
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

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def forward(self, x):
        return F.linear(x, self.weight_mu, self.bias_mu)

# --- 2. Enhanced DQN (Dueling & Noisy) ---
class DQN(nn.Module):
    def __init__(self, num_actions, input_channels=4, dueling=False, noisy=False):
        super(DQN, self).__init__()
        self.dueling = dueling
        self.noisy = noisy

        def linear_layer(in_f, out_f):
            return NoisyLinear(in_f, out_f) if noisy else nn.Linear(in_f, out_f)

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

        if dueling:
            self.value_stream = nn.Sequential(
                linear_layer(feature_dim, 512),
                nn.ReLU(),
                linear_layer(512, 1),
            )
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
        x = x / 255.0
        features = self.features(x)
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            return value + (advantage - advantage.mean(dim=1, keepdim=True))
        return self.fc(features)

# --- 3. Synchronized Preprocessing ---
class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        cropped = gray[34:194, :]
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

# --- 4. Evaluation Loop ---
def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    preprocessor = AtariPreprocessor()

    model = DQN(env.action_space.n, dueling=args.dueling, noisy=args.noisy).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    all_rewards = []

    for i in range(args.episodes):
        seed = args.seed_start + i
        obs, _ = env.reset(seed=seed)
        state = preprocessor.reset(obs)
        done = False
        total_reward = 0
        frames = []

        while not done:
            if args.save_video:
                frames.append(env.render())

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs)

        all_rewards.append(total_reward)
        print(f"Environment steps: {args.env_steps}, seed: {seed}, eval reward: {int(total_reward)}")

        if args.save_video:
            out_path = os.path.join(args.output_dir, f"eval_seed{seed}.mp4")
            imageio.mimsave(out_path, frames, fps=30)

    env.close()
    print(f"Average reward: {np.mean(all_rewards):.2f}")

# --- 5. Main Entry Point ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed; evaluates seeds [seed_start, seed_start + episodes - 1]")
    parser.add_argument("--env-steps", type=int, default=20000000,
                        help="Environment step count to display in log lines")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dueling", action="store_true")
    parser.add_argument("--noisy", action="store_true")

    args = parser.parse_args()
    evaluate(args)
