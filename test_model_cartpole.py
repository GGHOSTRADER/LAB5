import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import imageio
import os
import argparse


class DQN(nn.Module):
    def __init__(self, input_dim, num_actions):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    input_dim = env.observation_space.shape[0]
    num_actions = env.action_space.n

    model = DQN(input_dim, num_actions).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    if args.save_video:
        os.makedirs(args.output_dir, exist_ok=True)

    all_rewards = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        state = np.asarray(obs, dtype=np.float32)
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
            state = np.asarray(next_obs, dtype=np.float32)

        all_rewards.append(total_reward)

        if args.save_video:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}.mp4")
            with imageio.get_writer(out_path, fps=30) as video:
                for f in frames:
                    video.append_data(f)
            print(f"seed: {args.seed + ep}, eval reward: {total_reward}  ->  {out_path}")
        else:
            print(f"seed: {args.seed + ep}, eval reward: {total_reward}")

    env.close()
    print(f"Average reward: {np.mean(all_rewards):.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path",  type=str, required=True)
    parser.add_argument("--output-dir",  type=str, default="./eval_videos_cartpole")
    parser.add_argument("--episodes",    type=int, default=20)
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--save-video",  action="store_true")
    args = parser.parse_args()
    evaluate(args)
