from __future__ import annotations

import csv
import random
import time
from collections import deque, namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim


Transition = namedtuple("Transition", ["state", "action", "next_state", "reward", "done"])


def ensure_output_dirs(base_dir: Path) -> tuple[Path, Path]:
    results_dir = base_dir / "results_csv"
    plots_dir = base_dir / "plots"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, plots_dir


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000) -> None:
        self.capacity = max(1, int(capacity))
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Transition]:
        return random.sample(self.buffer, batch_size)


class GridWorld:
    def __init__(
        self,
        slippery: bool = False,
        slip_probability: float = 0.2,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.render_mode = render_mode
        self.env = gym.make("CliffWalking-v1", render_mode=render_mode)
        self.slippery = bool(slippery)
        self.slip_probability = float(max(0.0, min(1.0, slip_probability)))
        self.rng = random.Random(seed)

        self.n_states = self.env.observation_space.n
        self.n_actions = self.env.action_space.n
        self.start_state = 36
        self.goal_state = 47

        self.state, _ = self.env.reset(seed=seed)
        if not pygame.get_init():
            pygame.init()

    def set_slippery(self, slippery: bool) -> None:
        self.slippery = bool(slippery)

    def reset(self, seed: Optional[int] = None) -> int:
        self.state, _ = self.env.reset(seed=seed)
        return int(self.state)

    @staticmethod
    def state_to_row_col(state: int) -> tuple[int, int]:
        return int(state) // 12, int(state) % 12

    @staticmethod
    def row_col_to_state(row: int, col: int) -> int:
        return int(row) * 12 + int(col)

    @staticmethod
    def is_reachable(row: int, col: int) -> bool:
        return 0 <= row < 4 and 0 <= col < 12

    def _apply_slip(self, action: int) -> int:
        if not self.slippery or self.rng.random() >= self.slip_probability:
            return int(action)
        if action in (0, 2):
            return self.rng.choice([1, 3])
        return self.rng.choice([0, 2])

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        intended_action = int(action)
        actual_action = self._apply_slip(intended_action)

        next_state, reward, terminated, truncated, info = self.env.step(actual_action)
        done = bool(terminated or truncated)
        self.state = int(next_state)

        details = dict(info)
        details["intended_action"] = intended_action
        details["actual_action"] = actual_action
        details["is_cliff"] = reward <= -100
        return int(next_state), float(reward), done, details

    def render_frame(self):
        if self.render_mode != "rgb_array":
            return None
        frame = self.env.render()
        if frame is None:
            return None
        _ = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
        return frame

    def close(self) -> None:
        self.env.close()


class QMLP(nn.Module):
    def __init__(self, n_states: int, n_actions: int, hidden_neurons: int = 128, activation: str = "relu") -> None:
        super().__init__()
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "sigmoid": nn.Sigmoid,
        }
        activation_cls = activation_map.get(str(activation).lower(), nn.ReLU)
        self.model = nn.Sequential(
            nn.Linear(n_states, hidden_neurons),
            activation_cls(),
            nn.Linear(hidden_neurons, hidden_neurons),
            activation_cls(),
            nn.Linear(hidden_neurons, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@dataclass
class DQNDefaults:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 10_000
    batch_size: int = 64
    target_update_frequency: int = 50
    hidden_neurons: int = 128
    activation: str = "relu"


class DQNetwork:
    policy_name = "DQN"

    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = DQNDefaults.learning_rate,
        gamma: float = DQNDefaults.gamma,
        epsilon_start: float = DQNDefaults.epsilon_start,
        epsilon_end: float = DQNDefaults.epsilon_end,
        epsilon_decay: float = DQNDefaults.epsilon_decay,
        replay_buffer_size: int = DQNDefaults.replay_buffer_size,
        batch_size: int = DQNDefaults.batch_size,
        target_update_frequency: int = DQNDefaults.target_update_frequency,
        hidden_neurons: int = DQNDefaults.hidden_neurons,
        activation: str = DQNDefaults.activation,
        device: Optional[str] = None,
    ) -> None:
        self.n_states = int(n_states)
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon_start)
        self.epsilon_start = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)
        self.batch_size = max(1, int(batch_size))
        self.target_update_frequency = max(1, int(target_update_frequency))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = QMLP(self.n_states, self.n_actions, hidden_neurons, activation).to(self.device)
        self.target_net = QMLP(self.n_states, self.n_actions, hidden_neurons, activation).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.learn_steps = 0

    def _one_hot_states(self, states: list[int]) -> torch.Tensor:
        indices = torch.tensor(states, dtype=torch.int64, device=self.device)
        state_tensor = torch.zeros((len(states), self.n_states), dtype=torch.float32, device=self.device)
        state_tensor.scatter_(1, indices.unsqueeze(1), 1.0)
        return state_tensor

    def select_action(self, state: int, epsilon: Optional[float] = None) -> int:
        eps = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            state_vec = self._one_hot_states([int(state)])
            q_values = self.online_net(state_vec)
            return int(torch.argmax(q_values, dim=1).item())

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)

    def _compute_next_q(self, next_state_tensor: torch.Tensor) -> torch.Tensor:
        return self.target_net(next_state_tensor).max(1)[0]

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size)
        states = [item.state for item in batch]
        actions = torch.tensor([item.action for item in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        next_states = [item.next_state for item in batch]
        rewards = torch.tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([item.done for item in batch], dtype=torch.float32, device=self.device)

        state_tensor = self._one_hot_states(states)
        next_state_tensor = self._one_hot_states(next_states)

        q_values = self.online_net(state_tensor).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._compute_next_q(next_state_tensor)
            target = rewards + self.gamma * next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())


class DDQNetwork(DQNetwork):
    policy_name = "DDQN"

    def _compute_next_q(self, next_state_tensor: torch.Tensor) -> torch.Tensor:
        selected_actions = self.online_net(next_state_tensor).argmax(1, keepdim=True)
        return self.target_net(next_state_tensor).gather(1, selected_actions).squeeze(1)


class Trainer:
    def __init__(self, env: GridWorld, base_dir: Optional[Path] = None) -> None:
        self.env = env
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.results_dir, self.plots_dir = ensure_output_dirs(self.base_dir)

    def run_episode(
        self,
        policy: DQNetwork,
        epsilon: Optional[float] = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
        transition_callback: Optional[Callable[[Transition, int], None]] = None,
    ) -> dict:
        state = self.env.reset()
        total_reward = 0.0
        transitions: list[Transition] = []

        for step in range(1, int(max_steps) + 1):
            action = policy.select_action(state, epsilon=epsilon if epsilon is not None else None)
            next_state, reward, done, _ = self.env.step(action)

            transition = Transition(state, action, next_state, reward, float(done))
            policy.observe(transition)
            policy.learn()

            transitions.append(transition)
            total_reward += reward
            state = next_state

            if progress_callback:
                progress_callback(step)
            if transition_callback:
                transition_callback(transition, step)

            if done:
                break

        return {
            "total_reward": float(total_reward),
            "steps": len(transitions),
            "transitions": transitions,
        }

    def train(
        self,
        policy: DQNetwork,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[list[float], Optional[Path]]:
        rewards: list[float] = []
        rows: list[list] = []

        policy.epsilon = float(epsilon)

        for episode in range(1, int(num_episodes) + 1):
            result = self.run_episode(
                policy=policy,
                epsilon=None,
                max_steps=max_steps,
                progress_callback=(lambda step, ep=episode: progress_callback(ep, step)) if progress_callback else None,
            )
            rewards.append(float(result["total_reward"]))

            if save_csv:
                for step_idx, transition in enumerate(result["transitions"], start=1):
                    rows.append(
                        [
                            episode,
                            step_idx,
                            transition.state,
                            transition.action,
                            transition.next_state,
                            transition.reward,
                            transition.done,
                        ]
                    )

        csv_path: Optional[Path] = None
        if save_csv:
            timestamp = int(time.time())
            csv_path = self.results_dir / f"{save_csv}_{timestamp}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(["episode", "step", "state", "action", "next_state", "reward", "done"])
                writer.writerows(rows)

        return rewards, csv_path
