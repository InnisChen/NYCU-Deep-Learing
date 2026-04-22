# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

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
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class DQN(nn.Module):
    """
        Design the architecture of your deep Q network
        - Input size is the same as the state dimension; the output size is the same as the number of actions
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
    """
    def __init__(self, num_actions, state_dim=4, use_cnn=False):
        super(DQN, self).__init__()
        self.use_cnn = use_cnn
        ########## YOUR CODE HERE (5~10 lines) ##########
        if use_cnn:
            # Atari CNN: input (batch, 4, 84, 84)
            self.network = nn.Sequential(
                nn.Conv2d(state_dim, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
                nn.Linear(512, num_actions)
            )
        else:
            # MLP for low-dimensional state (e.g. CartPole: state_dim=4)
            self.network = nn.Sequential(
                nn.Linear(state_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
                nn.Linear(128, num_actions)
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.use_cnn:
            return self.network(x / 255.0)
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """    
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        if obs.ndim == 1:
            return obs  # CartPole: pass through raw state
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        if obs.ndim == 1:
            return frame  # CartPole: return (state_dim,) directly
        self.frames = deque([frame for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        if obs.ndim == 1:
            return frame  # CartPole: return (state_dim,) directly
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """ 
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return 
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ########## 
                    
        ########## END OF YOUR CODE (for Task 3) ########## 
        return
        

class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.preprocessor = AtariPreprocessor()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.is_atari = "ALE" in env_name
        if self.is_atari:
            self.q_net = DQN(self.num_actions, state_dim=4, use_cnn=True).to(self.device)
            self.target_net = DQN(self.num_actions, state_dim=4, use_cnn=True).to(self.device)
        else:
            state_dim = self.env.observation_space.shape[0]
            self.q_net = DQN(self.num_actions, state_dim=state_dim, use_cnn=False).to(self.device)
            self.target_net = DQN(self.num_actions, state_dim=state_dim, use_cnn=False).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        

        self.memory = deque(maxlen=args.memory_size)

        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min

        self.env_count = 0
        self.train_count = 0
        self.best_reward = -21 if self.is_atari else 0  # Initilized to 0 for CartPole and to -21 for Pong
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step
        self.save_dir = args.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.task = args.task
        self.student_id = args.student_id
        self.checkpoint_freq = args.checkpoint_freq
        self.task3_milestones = [600000, 1000000, 1500000, 2000000, 2500000]
        self.saved_milestones = set()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(state_tensor)
        return q_values.argmax().item()

    def save_checkpoint(self, ep):
        path = os.path.join(self.save_dir, "checkpoint.pt")
        torch.save({
            'q_net': self.q_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epsilon': self.epsilon,
            'env_count': self.env_count,
            'train_count': self.train_count,
            'best_reward': self.best_reward,
            'episode': ep,
            'saved_milestones': list(self.saved_milestones),
        }, path)
        # Sync to Google Drive if mounted
        drive_ckpt_dir = os.environ.get("DRIVE_CKPT_DIR", "")
        if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
            import shutil
            backup = os.path.join(drive_ckpt_dir, f"task{self.task}_checkpoint.pt")
            shutil.copy(path, backup)
            print(f"[Drive] Checkpoint backed up → {backup}")

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(ckpt['q_net'])
        self.target_net.load_state_dict(ckpt['target_net'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.epsilon = ckpt['epsilon']
        self.env_count = ckpt['env_count']
        self.train_count = ckpt['train_count']
        self.best_reward = ckpt['best_reward']
        self.saved_milestones = set(ckpt.get('saved_milestones', []))
        if 'scaler' in ckpt:
            self.scaler.load_state_dict(ckpt['scaler'])
        start_ep = ckpt['episode'] + 1
        print(f"Resumed from checkpoint: ep={ckpt['episode']} env_count={self.env_count} epsilon={self.epsilon:.4f}")
        return start_ep

    def run(self, episodes=1000, start_ep=0):
        for ep in range(start_ep, episodes):
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
                self.memory.append((state, action, reward, next_state, done))

                for _ in range(self.train_per_step):
                    self.train()

                state = next_state
                total_reward += reward
                self.env_count += 1
                step_count += 1

                # Task 3: save milestone snapshots
                if self.task == 3:
                    for milestone in self.task3_milestones:
                        if self.env_count >= milestone and milestone not in self.saved_milestones:
                            m_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_{milestone}.pt")
                            torch.save(self.q_net.state_dict(), m_path)
                            self.saved_milestones.add(milestone)
                            print(f"[Milestone] Saved {milestone} steps → {m_path}")
                            wandb.log({"Milestone Steps": milestone, "Env Step Count": self.env_count})

                # Periodic checkpoint for resume
                if self.checkpoint_freq > 0 and self.env_count % self.checkpoint_freq == 0:
                    self.save_checkpoint(ep)

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed 
                    
                    ########## END OF YOUR CODE ##########   
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            wandb.log({
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon
            })
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed 
            
            ########## END OF YOUR CODE ##########  
            if ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")

            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    if self.task == 3:
                        model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_best.pt")
                    else:
                        model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task{self.task}.pt")
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward}")
                    drive_ckpt_dir = os.environ.get("DRIVE_CKPT_DIR", "")
                    if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
                        import shutil
                        drive_best = os.path.join(os.path.dirname(drive_ckpt_dir), os.path.basename(model_path))
                        shutil.copy(model_path, drive_best)
                        print(f"[Drive] Best model backed up → {drive_best}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })

    def run_vectorized(self, episodes=1000, start_ep=0, num_envs=4):
        from gymnasium.vector import AsyncVectorEnv

        def make_env():
            def _init():
                return gym.make(self.env_name)
            return _init

        vec_env = AsyncVectorEnv([make_env() for _ in range(num_envs)])
        preprocessors = [AtariPreprocessor() for _ in range(num_envs)]

        obs_batch, _ = vec_env.reset()
        states = [preprocessors[i].reset(obs_batch[i]) for i in range(num_envs)]
        ep_rewards = [0.0] * num_envs
        ep_count = start_ep
        last_eval_ep = (start_ep // 20) * 20

        while ep_count < episodes:
            actions = [self.select_action(states[i]) for i in range(num_envs)]
            obs_batch, reward_batch, term_batch, trunc_batch, info_batch = vec_env.step(actions)

            for i in range(num_envs):
                done = bool(term_batch[i]) or bool(trunc_batch[i])

                if done and 'final_observation' in info_batch and info_batch['final_observation'][i] is not None:
                    next_obs = info_batch['final_observation'][i]
                else:
                    next_obs = obs_batch[i]
                next_state = preprocessors[i].step(next_obs)

                self.memory.append((states[i], actions[i], float(reward_batch[i]), next_state, done))
                ep_rewards[i] += reward_batch[i]
                self.env_count += 1

                if self.task == 3:
                    for milestone in self.task3_milestones:
                        if self.env_count >= milestone and milestone not in self.saved_milestones:
                            m_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_{milestone}.pt")
                            torch.save(self.q_net.state_dict(), m_path)
                            self.saved_milestones.add(milestone)
                            print(f"[Milestone] Saved {milestone} steps → {m_path}")
                            wandb.log({"Milestone Steps": milestone, "Env Step Count": self.env_count})

                if self.checkpoint_freq > 0 and self.env_count % self.checkpoint_freq == 0:
                    self.save_checkpoint(ep_count)

                if done:
                    ep_count += 1
                    total_reward = ep_rewards[i]
                    ep_rewards[i] = 0.0

                    print(f"[Eval] Ep: {ep_count} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep_count,
                        "Total Reward": total_reward,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })

                    if ep_count % 100 == 0:
                        model_path = os.path.join(self.save_dir, f"model_ep{ep_count}.pt")
                        torch.save(self.q_net.state_dict(), model_path)
                        print(f"Saved model checkpoint to {model_path}")

                    if ep_count % 20 == 0 and ep_count != last_eval_ep:
                        last_eval_ep = ep_count
                        eval_reward = self.evaluate()
                        if eval_reward > self.best_reward:
                            self.best_reward = eval_reward
                            if self.task == 3:
                                model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task3_best.pt")
                            else:
                                model_path = os.path.join(self.save_dir, f"LAB5_{self.student_id}_task{self.task}.pt")
                            torch.save(self.q_net.state_dict(), model_path)
                            print(f"Saved new best model to {model_path} with reward {eval_reward}")
                            drive_ckpt_dir = os.environ.get("DRIVE_CKPT_DIR", "")
                            if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
                                import shutil
                                drive_best = os.path.join(os.path.dirname(drive_ckpt_dir), os.path.basename(model_path))
                                shutil.copy(model_path, drive_best)
                                print(f"[Drive] Best model backed up → {drive_best}")
                        print(f"[TrueEval] Ep: {ep_count} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                        wandb.log({
                            "Env Step Count": self.env_count,
                            "Update Count": self.train_count,
                            "Eval Reward": eval_reward,
                        })

                    states[i] = preprocessors[i].reset(obs_batch[i])
                else:
                    states[i] = next_state

            if self.env_count % 1000 < num_envs:
                print(f"[Collect] SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Epsilon": self.epsilon,
                })

            for _ in range(self.train_per_step * num_envs):
                self.train()

        vec_env.close()

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state = self.preprocessor.reset(obs)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                action = self.q_net(state_tensor).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done = terminated or truncated
            total_reward += reward
            state = self.preprocessor.step(next_obs)

        return total_reward


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return 
        
        # Decay function for epsilin-greedy exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1
       
        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        ########## END OF YOUR CODE ##########

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        # NOTE: Enable this part after you finish the mini-batch sampling
        states = torch.from_numpy(np.array(states).astype(np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states).astype(np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]
                target = rewards + self.gamma * next_q * (1 - dones)
            loss = F.mse_loss(q_values, target)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # NOTE: Enable this part if "loss" is defined
        #if self.train_count % 1000 == 0:
        #    print(f"[Train #{self.train_count}] Loss: {loss.item():.4f} Q mean: {q_values.mean().item():.3f} std: {q_values.std().item():.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--student-id", type=str, default="B11107027")
    parser.add_argument("--env-name", type=str, default="CartPole-v1")
    parser.add_argument("--episodes", type=int, default=None, help="Override episode count")
    parser.add_argument("--save-dir", type=str, default="./results")
    parser.add_argument("--wandb-run-name", type=str, default="cartpole-run")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint.pt to resume from")
    parser.add_argument("--checkpoint-freq", type=int, default=50000, help="Save checkpoint every N env steps (0=disabled)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--memory-size", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.999999)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (>1 uses run_vectorized)")
    args = parser.parse_args()

    default_episodes = {1: 2000, 2: 10000, 3: 10000}
    episodes = args.episodes if args.episodes is not None else default_episodes[args.task]

    project_name = "DLP-Lab5-DQN-Atari" if "ALE" in args.env_name else "DLP-Lab5-DQN-CartPole"
    wandb.init(project=project_name, name=args.wandb_run_name, save_code=True)

    agent = DQNAgent(env_name=args.env_name, args=args)

    start_ep = 0
    if args.resume:
        start_ep = agent.load_checkpoint(args.resume)

    if args.num_envs > 1:
        agent.run_vectorized(episodes=episodes, start_ep=start_ep, num_envs=args.num_envs)
    else:
        agent.run(episodes=episodes, start_ep=start_ep)