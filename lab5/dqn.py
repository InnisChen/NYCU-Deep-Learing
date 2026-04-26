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
        Atari CNN DQN (matches test_model.py's architecture for Task 2/3).
        Input: (batch, 4, 84, 84); /255 normalization applied in forward.
    """
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        ########## YOUR CODE HERE (5~10 lines) ##########
        self.network = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512), nn.ReLU(),
            nn.Linear(512, num_actions)
        )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        return self.network(x / 255.0)


class DQNMLP(nn.Module):
    """
        MLP DQN for low-dimensional state environments (e.g., CartPole).
    """
    def __init__(self, num_actions, state_dim):
        super(DQNMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, num_actions)
        )

    def forward(self, x):
        return self.network(x)


class AtariPreprocessor:
    """
        Preprocesing the state input of DQN for Atari
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
        self.last_frame = None

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        return resized

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.last_frame = frame
        self.frames = deque([frame.copy() for _ in range(self.frame_stack)], maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        pooled = np.maximum(self.last_frame, frame) if self.last_frame is not None else frame
        self.last_frame = frame
        self.frames.append(pooled)
        return np.stack(self.frames, axis=0)


class SumTree:
    """
        Binary sum tree for O(log n) proportional sampling in PER.
        Leaves store per-transition priorities; internal nodes store subtree sums.
    """
    def __init__(self, capacity):
        self.capacity = int(capacity)
        size = 1
        while size < self.capacity:
            size *= 2
        self.tree_size = size
        # index 0 unused; internal [1, tree_size-1]; leaves [tree_size, 2*tree_size-1]
        self.tree = np.zeros(2 * self.tree_size, dtype=np.float64)

    def update(self, data_idx, priority):
        tree_idx = data_idx + self.tree_size
        self.tree[tree_idx] = priority
        tree_idx //= 2
        while tree_idx >= 1:
            self.tree[tree_idx] = self.tree[2 * tree_idx] + self.tree[2 * tree_idx + 1]
            tree_idx //= 2

    def total(self):
        return self.tree[1]

    def find(self, value):
        idx = 1
        while idx < self.tree_size:
            left = 2 * idx
            if self.tree[left] >= value:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        data_idx = idx - self.tree_size
        return data_idx, self.tree[idx]


class PrioritizedReplayBuffer:
    """
        Prioritizing the samples in the replay memory by the Bellman error
        See the paper (Schaul et al., 2016) at https://arxiv.org/abs/1511.05952
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = int(capacity)
        self.alpha = alpha
        self.beta = beta
        self.buffer = []                                               # template field (unused; kept for skeleton compat)
        self.priorities = np.zeros((capacity,), dtype=np.float32)      # template field (mirrors sum tree leaves for visibility)
        self.pos = 0

        ########## YOUR CODE HERE (for Task 3) ##########
        # Atari-only: fixed frame-stack state (4, 84, 84) uint8
        state_shape = (4, 84, 84)
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.size = 0
        self.max_priority = 1.0
        self.eps = 1e-6
        self.sum_tree = SumTree(self.capacity)
        ########## END OF YOUR CODE (for Task 3) ##########

    def __len__(self):
        return self.size

    def append(self, transition):
        """Convenience alias for Agent/NStepWrapper that don't pass a TD error."""
        self.add(transition, None)

    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########
        if len(transition) == 6:
            s, a, r, ns, d, disc = transition
        else:
            s, a, r, ns, d = transition
            disc = 0.99
        self.states[self.pos] = s
        self.next_states[self.pos] = ns
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.dones[self.pos] = float(d)
        self.discounts[self.pos] = float(disc)

        if error is None:
            priority = self.max_priority
        else:
            priority = (abs(float(error)) + self.eps) ** self.alpha

        self.sum_tree.update(self.pos, priority)
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        ########## END OF YOUR CODE (for Task 3) ##########
        return

    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        total = self.sum_tree.total()
        segment = total / batch_size
        indices = np.empty(batch_size, dtype=np.int64)
        probs = np.empty(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            v = np.random.uniform(lo, hi)
            if v <= 0:
                v = self.eps
            data_idx, prio = self.sum_tree.find(v)
            data_idx = min(max(0, data_idx), self.size - 1)
            indices[i] = data_idx
            probs[i] = max(prio, self.eps) / max(total, self.eps)

        # Importance-sampling weights
        weights = (self.size * probs) ** (-self.beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            self.discounts[indices],
            indices,
            weights,
        )
        ########## END OF YOUR CODE (for Task 3) ##########

    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        errors = np.abs(np.asarray(errors, dtype=np.float64)) + self.eps
        priorities = errors ** self.alpha
        for idx, p in zip(indices, priorities):
            ii = int(idx)
            self.sum_tree.update(ii, float(p))
            self.priorities[ii] = p
        self.max_priority = max(self.max_priority, float(priorities.max()))
        ########## END OF YOUR CODE (for Task 3) ##########
        return


class FrameStackReplayBuffer:
    """
        Uniform replay buffer for Atari. Stores stacked uint8 states in contiguous
        numpy arrays → O(1) sampling (vs deque's O(n)).
        Memory: 2 * capacity * 4 * 84 * 84 bytes
                ≈ 5.6 GB for capacity=100_000, ≈ 11.2 GB for capacity=200_000.
    """
    def __init__(self, capacity, state_shape=(4, 84, 84), gamma=0.99):
        self.capacity = int(capacity)
        self.default_discount = float(gamma)
        self.states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.next_states = np.zeros((self.capacity, *state_shape), dtype=np.uint8)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, transition):
        if len(transition) == 6:
            s, a, r, ns, d, disc = transition
        else:
            s, a, r, ns, d = transition
            disc = self.default_discount
        self.states[self.pos] = s
        self.next_states[self.pos] = ns
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.dones[self.pos] = float(d)
        self.discounts[self.pos] = float(disc)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.discounts[idx],
        )


class VectorReplayBuffer:
    """
        Uniform replay buffer for low-dim vector state (e.g., CartPole).
    """
    def __init__(self, capacity, state_dim, gamma=0.99):
        self.capacity = int(capacity)
        self.default_discount = float(gamma)
        self.states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.next_states = np.zeros((self.capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(self.capacity, dtype=np.int64)
        self.rewards = np.zeros(self.capacity, dtype=np.float32)
        self.dones = np.zeros(self.capacity, dtype=np.float32)
        self.discounts = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def append(self, transition):
        if len(transition) == 6:
            s, a, r, ns, d, disc = transition
        else:
            s, a, r, ns, d = transition
            disc = self.default_discount
        self.states[self.pos] = s
        self.next_states[self.pos] = ns
        self.actions[self.pos] = a
        self.rewards[self.pos] = r
        self.dones[self.pos] = float(d)
        self.discounts[self.pos] = float(disc)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.discounts[idx],
        )


class NStepWrapper:
    """
        Accumulates n-step returns before pushing to the underlying replay buffer.
        append(transition) takes a SINGLE-step (s, a, r, ns, d) and pushes
        (s_0, a_0, R^(n), s_n, done_within_n, gamma^actual_n) to the wrapped buffer.
        Delegates sample / update_priorities / __len__.
    """
    def __init__(self, buffer, n_step=3, gamma=0.99):
        self.buffer = buffer
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.pending = deque()

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.pending.append(transition)
        _, _, _, _, done = transition
        if len(self.pending) >= self.n_step:
            self._push_front()
        if done:
            while self.pending:
                self._push_front()

    def _push_front(self):
        R = 0.0
        last_ns = None
        last_done = False
        actual_n = 0
        for i, (_, _, r_i, ns_i, d_i) in enumerate(self.pending):
            R += (self.gamma ** i) * r_i
            last_ns = ns_i
            actual_n = i + 1
            if d_i:
                last_done = True
                break
            if actual_n >= self.n_step:
                break
        s0, a0, _, _, _ = self.pending[0]
        discount = self.gamma ** actual_n
        self.buffer.append((s0, a0, R, last_ns, last_done, discount))
        self.pending.popleft()

    def sample(self, batch_size):
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices, errors):
        if hasattr(self.buffer, 'update_priorities'):
            self.buffer.update_priorities(indices, errors)

    def reset_episode(self):
        self.pending.clear()


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name = env_name
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        self.is_atari = "ALE" in env_name

        # Task 3 enhancement flags
        self.use_per = bool(getattr(args, 'use_per', False))
        self.use_double = bool(getattr(args, 'use_double', False))
        self.n_step = int(getattr(args, 'n_step', 1))
        self.per_alpha = float(getattr(args, 'per_alpha', 0.6))
        self.per_beta_start = float(getattr(args, 'per_beta', 0.4))
        self.per_beta_anneal_steps = int(getattr(args, 'per_beta_anneal_steps', 1000000))
        self.epsilon_decay_type = getattr(args, 'epsilon_decay_type', 'exp')
        self.epsilon_decay_steps = int(getattr(args, 'epsilon_decay_steps', 250000))

        gamma = args.discount_factor

        if self.is_atari:
            self.preprocessor = AtariPreprocessor()
            self.q_net = DQN(self.num_actions).to(self.device)
            self.target_net = DQN(self.num_actions).to(self.device)
            if self.use_per:
                base_buffer = PrioritizedReplayBuffer(
                    args.memory_size, alpha=self.per_alpha, beta=self.per_beta_start
                )
            else:
                base_buffer = FrameStackReplayBuffer(
                    args.memory_size, state_shape=(4, 84, 84), gamma=gamma
                )
            if self.n_step > 1:
                self.memory = NStepWrapper(base_buffer, n_step=self.n_step, gamma=gamma)
            else:
                self.memory = base_buffer
        else:
            self.preprocessor = None
            state_dim = self.env.observation_space.shape[0]
            self.q_net = DQNMLP(self.num_actions, state_dim=state_dim).to(self.device)
            self.target_net = DQNMLP(self.num_actions, state_dim=state_dim).to(self.device)
            self.memory = VectorReplayBuffer(args.memory_size, state_dim=state_dim, gamma=gamma)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)
        self.scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

        self.batch_size = args.batch_size
        self.gamma = gamma
        self.epsilon_start_val = float(args.epsilon_start)
        self.epsilon = float(args.epsilon_start)
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
        self.seed = getattr(args, 'seed', None)
        self.noop_max = int(getattr(args, 'noop_max', 0))
        self.noop_step_count = 0
        self._seeded_single = False
        self.last_train_stats = {}

    def _reset_state(self, obs):
        if self.is_atari:
            return self.preprocessor.reset(obs)
        return np.asarray(obs, dtype=np.float32)

    def _step_state(self, obs):
        if self.is_atari:
            return self.preprocessor.step(obs)
        return np.asarray(obs, dtype=np.float32)

    def _per_buffer(self):
        b = self.memory
        if isinstance(b, NStepWrapper):
            b = b.buffer
        return b if isinstance(b, PrioritizedReplayBuffer) else None

    def _update_epsilon(self):
        if self.epsilon_decay_type == 'linear':
            start = self.replay_start_size
            denom = max(1, self.epsilon_decay_steps)
            frac = max(0.0, min(1.0, (self.env_count - start) / denom))
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon_start_val - frac * (self.epsilon_start_val - self.epsilon_min),
            )
        else:
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def _update_beta(self):
        per = self._per_buffer()
        if per is None:
            return
        frac = min(1.0, self.env_count / max(1, self.per_beta_anneal_steps))
        per.beta = self.per_beta_start + frac * (1.0 - self.per_beta_start)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
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
        if start_ep > 0:
            self._seeded_single = True

        for ep in range(start_ep, episodes):
            if not self._seeded_single and self.seed is not None:
                obs, _ = self.env.reset(seed=self.seed)
                self._seeded_single = True
            else:
                obs, _ = self.env.reset()

            if self.is_atari and self.noop_max > 0:
                for _ in range(random.randint(0, self.noop_max)):
                    obs, _, terminated, truncated, _ = self.env.step(0)
                    self.env_count += 1
                    self.noop_step_count += 1
                    if terminated or truncated:
                        obs, _ = self.env.reset()

            state = self._reset_state(obs)
            if isinstance(self.memory, NStepWrapper):
                self.memory.reset_episode()
            done = False
            total_reward = 0
            step_count = 0

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated

                next_state = self._step_state(next_obs)
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
                            drive_ckpt_dir = os.environ.get("DRIVE_CKPT_DIR", "")
                            if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
                                import shutil
                                drive_milestone = os.path.join(os.path.dirname(drive_ckpt_dir), os.path.basename(m_path))
                                shutil.copy(m_path, drive_milestone)
                                print(f"[Drive] Milestone backed up → {drive_milestone}")

                if self.checkpoint_freq > 0 and self.env_count % self.checkpoint_freq == 0:
                    self.save_checkpoint(ep)

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep: {ep} Step: {step_count} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    log_data = {
                        "Episode": ep,
                        "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                        "Noop Step Count": self.noop_step_count
                    }
                    log_data.update(self.last_train_stats)
                    wandb.log(log_data)
                    ########## YOUR CODE HERE  ##########
                    # Add additional wandb logs for debugging if needed

                    ########## END OF YOUR CODE ##########
            print(f"[Eval] Ep: {ep} Total Reward: {total_reward} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
            log_data = {
                "Episode": ep,
                "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
                "Noop Step Count": self.noop_step_count
            }
            log_data.update(self.last_train_stats)
            wandb.log(log_data)
            ########## YOUR CODE HERE  ##########
            # Add additional wandb logs for debugging if needed

            ########## END OF YOUR CODE ##########
            if self.task != 3 and ep % 100 == 0:
                model_path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), model_path)
                print(f"Saved model checkpoint to {model_path}")
                # Sync snapshot to Drive so it survives a Colab runtime restart
                drive_ckpt_dir = os.environ.get("DRIVE_CKPT_DIR", "")
                if drive_ckpt_dir and os.path.isdir(drive_ckpt_dir):
                    import shutil
                    snap_dir = os.path.join(os.path.dirname(drive_ckpt_dir), f"task{self.task}_snapshots")
                    os.makedirs(snap_dir, exist_ok=True)
                    shutil.copy(model_path, os.path.join(snap_dir, f"model_ep{ep}.pt"))
                    print(f"[Drive] Snapshot synced → {snap_dir}/model_ep{ep}.pt")

            # Match the starter-code training monitor: evaluate one episode every 20 episodes.
            eval_every = 20
            eval_n = 1
            if ep % eval_every == 0:
                eval_reward = self.evaluate(n_episodes=eval_n)
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
                    "Eval Reward": eval_reward,
                    "Eval Episodes": eval_n
                })

    def evaluate(self, n_episodes=1):
        """Return average reward over n_episodes evaluation rollouts."""
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.test_env.reset()
            state = self._reset_state(obs)
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    action = self.q_net(state_tensor).argmax().item()
                next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
                done = terminated or truncated
                total_reward += reward
                state = self._step_state(next_obs)
            rewards.append(total_reward)
        return float(np.mean(rewards))


    def train(self):

        if len(self.memory) < self.replay_start_size:
            return

        # Decay function for epsilin-greedy exploration
        self._update_epsilon()
        self._update_beta()
        self.train_count += 1

        ########## YOUR CODE HERE (<5 lines) ##########
        # Sample a mini-batch of (s,a,r,s',done) from the replay buffer
        sample_out = self.memory.sample(self.batch_size)
        ########## END OF YOUR CODE ##########

        if self.use_per:
            states, actions, rewards, next_states, dones, discounts, indices, is_weights = sample_out
        else:
            states, actions, rewards, next_states, dones, discounts = sample_out
            indices = None
            is_weights = None

        # Convert the states, actions, rewards, next_states, and dones into torch tensors
        states = torch.from_numpy(states).to(self.device, non_blocking=True).float()
        next_states = torch.from_numpy(next_states).to(self.device, non_blocking=True).float()
        actions = torch.from_numpy(actions).to(self.device, non_blocking=True)
        rewards = torch.from_numpy(rewards).to(self.device, non_blocking=True)
        dones = torch.from_numpy(dones).to(self.device, non_blocking=True)
        discounts = torch.from_numpy(discounts).to(self.device, non_blocking=True)
        if is_weights is not None:
            is_weights = torch.from_numpy(is_weights).to(self.device, non_blocking=True)

        ########## YOUR CODE HERE (~10 lines) ##########
        # Implement the loss function of DQN and the gradient updates
        with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                if self.use_double:
                    next_actions = self.q_net(next_states).argmax(1, keepdim=True)
                    next_q = self.target_net(next_states).gather(1, next_actions).squeeze(1)
                else:
                    next_q = self.target_net(next_states).max(1)[0]
                target = rewards + discounts * next_q * (1 - dones)
            td_errors = target - q_values
            if is_weights is not None:
                loss = (is_weights * F.smooth_l1_loss(q_values, target, reduction='none')).mean()
            else:
                loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        ########## END OF YOUR CODE ##########

        if self.use_per and indices is not None:
            td_np = td_errors.detach().abs().cpu().numpy()
            self.memory.update_priorities(indices, td_np)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 100 == 0 or not self.last_train_stats:
            per = self._per_buffer()
            self.last_train_stats = {
                "Train/Loss": float(loss.detach().cpu().item()),
                "Train/Q Mean": float(q_values.detach().mean().cpu().item()),
                "Train/Q Max": float(q_values.detach().max().cpu().item()),
                "Train/Target Mean": float(target.detach().mean().cpu().item()),
                "Train/TD Error Mean": float(td_errors.detach().abs().mean().cpu().item()),
                "Train/TD Error Max": float(td_errors.detach().abs().max().cpu().item()),
                "Train/Grad Norm": float(grad_norm.detach().cpu().item() if torch.is_tensor(grad_norm) else grad_norm),
                "Train/Replay Size": len(self.memory),
            }
            if per is not None:
                self.last_train_stats.update({
                    "Train/PER Beta": float(per.beta),
                    "Train/PER Max Priority": float(per.max_priority),
                })


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
    parser.add_argument("--epsilon-decay-type", type=str, default="exp", choices=["exp", "linear"])
    parser.add_argument("--epsilon-decay-steps", type=int, default=250000, help="Linear decay: env steps over which epsilon decays")
    parser.add_argument("--target-update-frequency", type=int, default=1000)
    parser.add_argument("--replay-start-size", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--train-per-step", type=int, default=1)
    # Task 3 enhancement flags (Double DQN / PER / n-step). Defaults off for Task 2 compat.
    parser.add_argument("--use-per", action="store_true", help="Task 3: enable Prioritized Experience Replay")
    parser.add_argument("--use-double", action="store_true", help="Task 3: enable Double DQN target")
    parser.add_argument("--n-step", type=int, default=1, help="Task 3: n-step return length (1=disabled)")
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta", type=float, default=0.4)
    parser.add_argument("--per-beta-anneal-steps", type=int, default=1000000)
    parser.add_argument("--noop-max", type=int, default=0, help="Atari training-only NoopReset max no-op actions after reset (0=disabled)")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()

    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    default_episodes = {1: 2000, 2: 10000, 3: 10000}
    episodes = args.episodes if args.episodes is not None else default_episodes[args.task]

    project_name = "DLP-Lab5-DQN-Atari" if "ALE" in args.env_name else "DLP-Lab5-DQN-CartPole"
    wandb.init(project=project_name, name=args.wandb_run_name, save_code=True)
    wandb.define_metric("Env Step Count")
    wandb.define_metric("*", step_metric="Env Step Count")

    agent = DQNAgent(env_name=args.env_name, args=args)

    start_ep = 0
    if args.resume:
        start_ep = agent.load_checkpoint(args.resume)

    agent.run(episodes=episodes, start_ep=start_ep)
