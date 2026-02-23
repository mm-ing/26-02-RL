from __future__ import annotations

import csv
import random
import time
from collections import deque, namedtuple
from pathlib import Path
from typing import Callable, Deque, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple("Transition", ("state", "action", "next_state", "reward", "done"))


class CliffWalkingEnv:
    rows: int = 4
    cols: int = 12
    start_state: int = 36
    goal_state: int = 47

    def __init__(self, slippery: bool = False, slippery_prob: float = 0.2) -> None:
        self.slippery = bool(slippery)
        self.slippery_prob = float(slippery_prob)
        self._rng = random.Random()
        self._gym_env = gym.make("CliffWalking-v1", render_mode="rgb_array")
        pygame.init()
        self.current_state = self.start_state

    def close(self) -> None:
        self._gym_env.close()

    def set_slippery(self, value: bool) -> None:
        self.slippery = bool(value)

    def reset(self) -> int:
        obs, _ = self._gym_env.reset()
        self.current_state = int(obs)
        return self.current_state

    def decode_state(self, state: int) -> Tuple[int, int]:
        return int(state) // self.cols, int(state) % self.cols

    def encode_state(self, row: int, col: int) -> int:
        return int(row) * self.cols + int(col)

    def state_to_vector(self, state: int) -> np.ndarray:
        row, col = self.decode_state(state)
        return np.array([row / (self.rows - 1), col / (self.cols - 1)], dtype=np.float32)

    def _effective_action(self, action: int) -> int:
        action = int(action)
        if not self.slippery:
            return action
        if self._rng.random() >= self.slippery_prob:
            return action
        if action in (0, 2):
            return self._rng.choice([1, 3])
        return self._rng.choice([0, 2])

    def step(self, action: int) -> Tuple[int, float, bool, dict]:
        effective = self._effective_action(action)
        next_obs, reward, terminated, truncated, info = self._gym_env.step(effective)
        done = bool(terminated or truncated)
        next_state = int(next_obs)
        self.current_state = next_state
        info = dict(info)
        info["effective_action"] = effective
        return next_state, float(reward), done, info

    def render_frame(self) -> np.ndarray:
        frame = self._gym_env.render()
        if frame is None:
            return np.zeros((200, 600, 3), dtype=np.uint8)
        return np.asarray(frame, dtype=np.uint8)


class ReplayBuffer:
    def __init__(self, capacity: int = 5000) -> None:
        self.capacity = int(capacity)
        self.memory: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool) -> None:
        self.memory.append(Transition(state, int(action), next_state, float(reward), bool(done)))

    def sample(self, batch_size: int) -> Sequence[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class _BaseDQN:
    def __init__(
        self,
        state_dim: int = 2,
        action_dim: int = 4,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        neurons: int = 64,
        activation: str = "relu",
        batch_size: int = 64,
        replay_buffer_size: int = 5000,
        target_update_interval: int = 200,
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.learning_rate = float(learning_rate)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.batch_size = int(batch_size)
        self.target_update_interval = int(target_update_interval)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._optimize_steps = 0

        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "elu": nn.ELU,
        }
        activation_cls = activation_map.get(activation.lower(), nn.ReLU)

        def build_model() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.state_dim, int(neurons)),
                activation_cls(),
                nn.Linear(int(neurons), int(neurons)),
                activation_cls(),
                nn.Linear(int(neurons), self.action_dim),
            )

        self.online_net = build_model().to(self.device)
        self.target_net = build_model().to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

    def select_action(self, state_vector: np.ndarray, epsilon: Optional[float] = None) -> int:
        eps = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < eps:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            tensor_state = torch.tensor(state_vector, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(tensor_state)
        return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state: np.ndarray, action: int, next_state: np.ndarray, reward: float, done: bool) -> None:
        self.replay_buffer.push(state, action, next_state, reward, done)

    def _next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def optimize(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._next_q_values(next_states)
            targets = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._optimize_steps += 1
        if self._optimize_steps % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


class DQNetwork(_BaseDQN):
    def _next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        return self.target_net(next_states).max(dim=1).values


class DDQNetwork(_BaseDQN):
    def _next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        next_actions = self.online_net(next_states).argmax(dim=1, keepdim=True)
        return self.target_net(next_states).gather(1, next_actions).squeeze(1)


class Trainer:
    def __init__(self, env: CliffWalkingEnv, output_dir: Optional[Path] = None) -> None:
        self.env = env
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
        self.results_dir = self.output_dir / "results_csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(
        self,
        policy: _BaseDQN,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[float, List[Transition]]:
        state = self.env.reset()
        total_reward = 0.0
        transitions: List[Transition] = []

        for step in range(1, int(max_steps) + 1):
            state_vec = self.env.state_to_vector(state)
            action = policy.select_action(state_vec, epsilon=epsilon)
            next_state, reward, done, _ = self.env.step(action)
            next_state_vec = self.env.state_to_vector(next_state)

            policy.remember(state_vec, action, next_state_vec, reward, done)
            policy.optimize()

            transition = Transition(state_vec, action, next_state_vec, reward, done)
            transitions.append(transition)
            total_reward += float(reward)
            state = next_state

            if progress_callback is not None:
                progress_callback(step)

            if done:
                break

        return total_reward, transitions

    def train(
        self,
        policy: _BaseDQN,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
    ) -> List[float]:
        rewards: List[float] = []
        sampled_transitions: List[Transition] = []

        for _ in range(int(num_episodes)):
            episode_reward, transitions = self.run_episode(
                policy=policy,
                epsilon=epsilon,
                max_steps=max_steps,
                progress_callback=None,
            )
            rewards.append(episode_reward)
            sampled_transitions.extend(transitions)

        if save_csv:
            csv_path = self.results_dir / f"{save_csv}_{int(time.time())}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["state", "action", "next_state", "reward", "done"])
                for transition in sampled_transitions:
                    writer.writerow(
                        [
                            np.asarray(transition.state).tolist(),
                            int(transition.action),
                            np.asarray(transition.next_state).tolist(),
                            float(transition.reward),
                            bool(transition.done),
                        ]
                    )

        return rewards
