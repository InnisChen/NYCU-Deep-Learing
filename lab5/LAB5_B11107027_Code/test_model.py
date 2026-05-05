import argparse
import os
import random

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import cv2
from collections import deque

from dqn import AtariPreprocessor, DQN, DQNMLP, _migrate_dqn_state_dict

gym.register_envs(ale_py)


class LegacyAtariPreprocessor:
    """Task 2 snapshots before Task 3 used plain frame stacking without max pooling."""
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if len(obs.shape) == 3 and obs.shape[2] == 3:
            gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        else:
            gray = obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame.copy())
        return np.stack(self.frames, axis=0)


class LegacyAtariDQN(nn.Module):
    """CNN layout used by early Task 2 checkpoints with network.* state_dict keys."""
    def __init__(self, input_channels, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )

    def forward(self, x):
        return self.network(x / 255.0)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make(args.env_name, render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    is_atari = "ALE" in args.env_name
    num_actions = env.action_space.n

    state_dict = torch.load(args.model_path, map_location=device, weights_only=True)
    if is_atari:
        legacy_task2 = any(k.startswith("network.") for k in state_dict)
        if legacy_task2:
            preprocessor = LegacyAtariPreprocessor()
            model = LegacyAtariDQN(4, num_actions).to(device)
        else:
            preprocessor = AtariPreprocessor()
            state_dict = _migrate_dqn_state_dict(state_dict)
            dueling = any(k.startswith("value_stream") for k in state_dict)
            model = DQN(num_actions, dueling=dueling).to(device)
    else:
        preprocessor = None
        state_dim = env.observation_space.shape[0]
        model = DQNMLP(num_actions, state_dim=state_dim).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.record:
        os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        obs, _ = env.reset(seed=ep_seed)
        state = preprocessor.reset(obs) if is_atari else np.asarray(obs, dtype=np.float32)
        done = False
        total_reward = 0
        frames = []

        while not done:
            if args.record:
                frames.append(env.render())

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)
            with torch.no_grad():
                action = model(state_tensor).argmax().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = preprocessor.step(next_obs) if is_atari else np.asarray(next_obs, dtype=np.float32)

        rewards.append(total_reward)

        if args.record:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}_seed{ep_seed}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"Saved eval episode {ep} (seed={ep_seed}) with reward {total_reward} -> {out_path}")
        else:
            print(f"Eval Episode {ep} (seed={ep_seed}) reward: {total_reward}")

    print(
        f"\nEval Reward Average ({args.episodes} episodes): {np.mean(rewards):.2f}  |  "
        f"Min: {np.min(rewards):.1f}  Max: {np.max(rewards):.1f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--env-name", type=str, default="ALE/Pong-v5")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for evaluation; episodes use seed, seed+1, ...")
    parser.add_argument("--record", action="store_true", help="Save mp4 videos for each evaluation episode")
    args = parser.parse_args()
    evaluate(args)
