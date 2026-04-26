import argparse
import os
import random

import ale_py
import gymnasium as gym
import imageio
import numpy as np
import torch

from dqn import AtariPreprocessor, DQN, _migrate_dqn_state_dict

gym.register_envs(ale_py)


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = gym.make("ALE/Pong-v5", render_mode="rgb_array")
    env.action_space.seed(args.seed)
    env.observation_space.seed(args.seed)

    preprocessor = AtariPreprocessor()
    num_actions = env.action_space.n

    state_dict = _migrate_dqn_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True)
    )
    dueling = any(k.startswith("value_stream") for k in state_dict)
    model = DQN(num_actions, dueling=dueling).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if args.record:
        os.makedirs(args.output_dir, exist_ok=True)

    rewards = []
    for ep in range(args.episodes):
        ep_seed = args.seed + ep
        obs, _ = env.reset(seed=ep_seed)
        state = preprocessor.reset(obs)
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
            state = preprocessor.step(next_obs)

        rewards.append(total_reward)

        if args.record:
            out_path = os.path.join(args.output_dir, f"eval_ep{ep}_seed{ep_seed}.mp4")
            with imageio.get_writer(out_path, fps=30, macro_block_size=1) as video:
                for frame in frames:
                    video.append_data(frame)
            print(f"Saved episode {ep} (seed={ep_seed}) with total reward {total_reward} -> {out_path}")
        else:
            print(f"Episode {ep} (seed={ep_seed}) total reward: {total_reward}")

    print(
        f"\nAverage ({args.episodes} episodes): {np.mean(rewards):.2f}  |  "
        f"Min: {np.min(rewards):.1f}  Max: {np.max(rewards):.1f}"
    )
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained .pt model")
    parser.add_argument("--output-dir", type=str, default="./eval_videos")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0, help="Base seed for evaluation; episodes use seed, seed+1, ...")
    parser.add_argument("--record", action="store_true", help="Save mp4 videos for each evaluation episode")
    args = parser.parse_args()
    evaluate(args)
