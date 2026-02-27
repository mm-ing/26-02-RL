from __future__ import annotations

import csv
import math
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


_DEVICE_PREFERENCE = "cpu"


def set_device_preference(use_gpu: bool) -> torch.device:
    global _DEVICE_PREFERENCE
    _DEVICE_PREFERENCE = "cuda" if bool(use_gpu) else "cpu"
    return get_device()


def get_device() -> torch.device:
    if _DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_hidden_layers(raw: str | Sequence[int] | int) -> List[int]:
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        values = [int(v) for v in raw if int(v) > 0]
        return values or [128]
    parts = [p.strip() for p in str(raw).split(",") if p.strip()]
    values = [int(v) for v in parts if int(v) > 0]
    return values or [128]


def make_activation(name: str) -> nn.Module:
    mapping = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
    }
    return mapping.get(name, nn.ReLU)()


class QNetwork(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_sizes: Sequence[int],
        activation: str = "ReLU",
        dueling: bool = False,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        input_dim = state_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(input_dim, size))
            layers.append(make_activation(activation))
            input_dim = size
        self.feature = nn.Sequential(*layers)
        self.dueling = dueling
        if dueling:
            self.value_head = nn.Linear(input_dim, 1)
            self.advantage_head = nn.Linear(input_dim, action_dim)
        else:
            self.q_head = nn.Linear(input_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        if self.dueling:
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage.mean(dim=1, keepdim=True)
        return self.q_head(features)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class ContinuousReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.float32),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )


class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, eps: float = 1e-5) -> None:
        super().__init__(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.eps = float(eps)
        self.priorities: Deque[float] = deque(maxlen=self.capacity)

    def add(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        super().add(transition)
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_prio)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled = priorities ** self.alpha
        probs = scaled / scaled.sum()
        indices = np.random.choice(len(self.buffer), size=batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*samples)
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            np.asarray(indices, dtype=np.int64),
            np.asarray(weights, dtype=np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        for idx, error in zip(indices, td_errors):
            self.priorities[int(idx)] = float(abs(error) + self.eps)


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_size: int = 50000
    batch_size: int = 64
    target_update: int = 100
    replay_warmup: int = 1000
    learning_cadence: int = 2
    activation_function: str = "ReLU"
    hidden_layers: str = "128"
    lr_strategy: str = "exponential"
    lr_decay: float = 0.1
    min_learning_rate: float = 1e-5
    gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2


POLICY_DEFAULTS: Dict[str, AgentConfig] = {
    "DuelingDQN": AgentConfig(gamma=0.99, learning_rate=3e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=2, activation_function="ReLU", hidden_layers="256,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "D3QN": AgentConfig(gamma=0.99, learning_rate=2.5e-4, replay_size=150000, batch_size=128, target_update=200, replay_warmup=8000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "DDQN+PER": AgentConfig(gamma=0.99, learning_rate=2e-4, replay_size=200000, batch_size=128, target_update=200, replay_warmup=10000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "PPO": AgentConfig(gamma=0.99, learning_rate=1e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=32, activation_function="ReLU", hidden_layers="256,256", lr_strategy="linear", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "A2C": AgentConfig(gamma=0.99, learning_rate=1.5e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=16, activation_function="ReLU", hidden_layers="256,256", lr_strategy="exponential", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=1.0, ppo_clip_range=0.2),
    "TRPO": AgentConfig(gamma=0.99, learning_rate=7.5e-5, replay_size=120000, batch_size=128, target_update=200, replay_warmup=6000, learning_cadence=32, activation_function="ReLU", hidden_layers="256,256", lr_strategy="linear", lr_decay=0.4, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "SAC": AgentConfig(gamma=0.99, learning_rate=1e-4, replay_size=200000, batch_size=128, target_update=200, replay_warmup=10000, learning_cadence=32, activation_function="ReLU", hidden_layers="256,256", lr_strategy="cosine", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
}

DISCRETE_POLICIES = ["DuelingDQN", "D3QN", "DDQN+PER"]
CONTINUOUS_POLICIES = ["PPO", "A2C", "TRPO", "SAC"]


class BaseDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, dueling: bool = False, prioritized: bool = False, planned_steps: int = 100000) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = get_device()
        hidden_sizes = parse_hidden_layers(config.hidden_layers)
        self.online_net = QNetwork(state_dim, action_dim, hidden_sizes, config.activation_function, dueling=dueling).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes, config.activation_function, dueling=dueling).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=config.learning_rate)
        self._base_lr = float(config.learning_rate)
        self._min_lr = max(0.0, float(config.min_learning_rate))
        self._target_lr = max(self._min_lr, self._base_lr * max(0.0, float(config.lr_decay)))
        self._lr_strategy = str(config.lr_strategy).strip().lower()
        self._planned_decay_steps = max(1, int(planned_steps))
        self._current_lr = self._base_lr
        self._best_loss = float("inf")
        self._loss_bad_steps = 0
        self._loss_patience = 200
        self._loss_tolerance = 1e-4
        self.prioritized = prioritized
        if prioritized:
            self.replay = PrioritizedReplayBuffer(config.replay_size)
        else:
            self.replay = ReplayBuffer(config.replay_size)
        self.learn_steps = 0
        self.total_steps = 0

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        return int(torch.argmax(q_values, dim=1).item())

    def store(self, transition: Tuple[np.ndarray, int, float, np.ndarray, bool]) -> None:
        self.replay.add(transition)
        self.total_steps += 1

    def can_learn(self) -> bool:
        warm = self.config.replay_warmup
        cadence = max(1, self.config.learning_cadence)
        return len(self.replay) >= warm and self.total_steps % cadence == 0 and len(self.replay) >= self.config.batch_size

    def learn_step(self) -> Optional[float]:
        if not self.can_learn():
            return None
        self.learn_steps += 1
        if self.prioritized:
            return self._learn_prioritized()
        return self._learn_uniform()

    def _set_optimizer_lr(self, value: float) -> None:
        lr = max(self._min_lr, float(value))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self._current_lr = lr

    def _schedule_progress(self) -> float:
        return min(1.0, max(0.0, float(self.learn_steps) / float(self._planned_decay_steps)))

    def _scheduled_lr(self) -> float:
        progress = self._schedule_progress()
        if self._lr_strategy == "linear":
            return self._base_lr + (self._target_lr - self._base_lr) * progress
        if self._lr_strategy == "cosine":
            return self._target_lr + 0.5 * (self._base_lr - self._target_lr) * (1.0 + math.cos(math.pi * progress))
        ratio = self._target_lr / max(self._base_lr, 1e-12)
        return self._base_lr * (ratio ** progress)

    def _step_lr_schedule(self, loss_value: float, grad_norm: float) -> None:
        strategy = self._lr_strategy
        if strategy == "loss-based":
            if loss_value + self._loss_tolerance < self._best_loss:
                self._best_loss = loss_value
                self._loss_bad_steps = 0
            else:
                self._loss_bad_steps += 1
            if self._loss_bad_steps >= self._loss_patience:
                factor = min(0.99, max(0.1, float(self.config.lr_decay)))
                self._set_optimizer_lr(self._current_lr * factor)
                self._loss_bad_steps = 0
            return

        if strategy == "guarded natural gradient":
            baseline = self._scheduled_lr()
            guard_strength = max(0.0, float(self.config.lr_decay))
            guarded_lr = baseline / (1.0 + guard_strength * max(0.0, grad_norm))
            self._set_optimizer_lr(guarded_lr)
            return

        self._set_optimizer_lr(self._scheduled_lr())

    def _double_q_targets(self, next_states_t: torch.Tensor, rewards_t: torch.Tensor, dones_t: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
            next_q_target = self.target_net(next_states_t).gather(1, next_actions).squeeze(1)
            return rewards_t + (1.0 - dones_t) * self.config.gamma * next_q_target

    def _learn_uniform(self) -> float:
        states, actions, rewards, next_states, dones = self.replay.sample(self.config.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)
        targets = self._double_q_targets(next_states_t, rewards_t, dones_t)
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0))
        self.optimizer.step()
        self._step_lr_schedule(float(loss.item()), grad_norm)

        if self.learn_steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())

    def _learn_prioritized(self) -> float:
        assert isinstance(self.replay, PrioritizedReplayBuffer)
        states, actions, rewards, next_states, dones, indices, weights = self.replay.sample(self.config.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        weights_t = torch.tensor(weights, dtype=torch.float32, device=self.device)

        q_values = self.online_net(states_t).gather(1, actions_t).squeeze(1)
        targets = self._double_q_targets(next_states_t, rewards_t, dones_t)
        td_errors = targets.detach() - q_values.detach()
        losses = nn.functional.smooth_l1_loss(q_values, targets, reduction="none")
        loss = (losses * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0))
        self.optimizer.step()
        self._step_lr_schedule(float(loss.item()), grad_norm)

        self.replay.update_priorities(indices, td_errors.cpu().numpy())

        if self.learn_steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


class DuelingDQN(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, dueling=True, prioritized=False, planned_steps=planned_steps)


class D3QN(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, dueling=True, prioritized=False, planned_steps=planned_steps)


class DDQNPER(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, dueling=False, prioritized=True, planned_steps=planned_steps)


def _make_mlp(input_dim: int, output_dim: int, hidden_sizes: Sequence[int], activation: str) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_dim = input_dim
    for size in hidden_sizes:
        layers.append(nn.Linear(in_dim, size))
        layers.append(make_activation(activation))
        in_dim = size
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class ContinuousQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_sizes: Sequence[int], activation: str = "ReLU") -> None:
        super().__init__()
        self.net = _make_mlp(state_dim + action_dim, 1, hidden_sizes, activation)

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([states, actions], dim=1)
        return self.net(x)


class BaseContinuousAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, algorithm: str, planned_steps: int = 100000) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.algorithm = algorithm
        self.device = get_device()

        hidden_sizes = parse_hidden_layers(config.hidden_layers)
        self.actor = _make_mlp(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(action_dim, device=self.device))
        self.critic = _make_mlp(state_dim, 1, hidden_sizes, config.activation_function).to(self.device)

        self.actor_optimizer = optim.Adam(list(self.actor.parameters()) + [self.log_std], lr=config.learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.learning_rate)

        self._base_lr = float(config.learning_rate)
        self._min_lr = max(0.0, float(config.min_learning_rate))
        self._target_lr = max(self._min_lr, self._base_lr * max(0.0, float(config.lr_decay)))
        self._lr_strategy = str(config.lr_strategy).strip().lower()
        self._planned_decay_steps = max(1, int(planned_steps))
        self._current_lr = self._base_lr
        self._best_loss = float("inf")
        self._loss_bad_steps = 0
        self._loss_patience = 50
        self._loss_tolerance = 1e-4
        self._max_grad_norm = 1.0
        self._gae_lambda = min(1.0, max(0.0, float(config.gae_lambda)))
        self._ppo_clip_range = min(0.5, max(0.01, float(config.ppo_clip_range)))
        self._minibatch_size = max(8, int(config.batch_size))
        self._min_rollout_steps = max(8, int(config.learning_cadence))
        self._entropy_coef = {
            "ppo": 5e-3,
            "a2c": 1e-2,
            "trpo": 2e-3,
            "sac": 2e-2,
        }.get(self.algorithm, 5e-3)

        self.learn_steps = 0
        self.total_steps = 0
        self._trajectory: List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = []

    def _set_optimizer_lr(self, value: float) -> None:
        lr = max(self._min_lr, float(value))
        for group in self.actor_optimizer.param_groups:
            group["lr"] = lr
        for group in self.critic_optimizer.param_groups:
            group["lr"] = lr
        self._current_lr = lr

    def _schedule_progress(self) -> float:
        return min(1.0, max(0.0, float(self.learn_steps) / float(self._planned_decay_steps)))

    def _scheduled_lr(self) -> float:
        progress = self._schedule_progress()
        if self._lr_strategy == "linear":
            return self._base_lr + (self._target_lr - self._base_lr) * progress
        if self._lr_strategy == "cosine":
            return self._target_lr + 0.5 * (self._base_lr - self._target_lr) * (1.0 + math.cos(math.pi * progress))
        ratio = self._target_lr / max(self._base_lr, 1e-12)
        return self._base_lr * (ratio ** progress)

    def _step_lr_schedule(self, loss_value: float, grad_norm: float) -> None:
        if self._lr_strategy == "loss-based":
            if loss_value + self._loss_tolerance < self._best_loss:
                self._best_loss = loss_value
                self._loss_bad_steps = 0
            else:
                self._loss_bad_steps += 1
            if self._loss_bad_steps >= self._loss_patience:
                factor = min(0.99, max(0.1, float(self.config.lr_decay)))
                self._set_optimizer_lr(self._current_lr * factor)
                self._loss_bad_steps = 0
            return

        if self._lr_strategy == "guarded natural gradient":
            baseline = self._scheduled_lr()
            guard_strength = max(0.0, float(self.config.lr_decay))
            guarded_lr = baseline / (1.0 + guard_strength * max(0.0, grad_norm))
            self._set_optimizer_lr(guarded_lr)
            return

        self._set_optimizer_lr(self._scheduled_lr())

    def _distribution(self, states_t: torch.Tensor) -> torch.distributions.Normal:
        safe_states = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = self.actor(safe_states)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        safe_log_std = torch.nan_to_num(self.log_std, nan=0.0, posinf=1.0, neginf=-5.0)
        std = torch.exp(safe_log_std).clamp(min=1e-3, max=2.0)
        return torch.distributions.Normal(mean, std)

    def _squashed_log_prob(self, dist: torch.distributions.Normal, actions_t: torch.Tensor) -> torch.Tensor:
        clipped = torch.clamp(actions_t, -0.999, 0.999)
        raw = 0.5 * torch.log((1.0 + clipped) / (1.0 - clipped))
        log_prob = dist.log_prob(raw) - torch.log(1.0 - clipped.pow(2) + 1e-6)
        return log_prob.sum(dim=1)

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=10.0, neginf=-10.0)
        state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            try:
                dist = self._distribution(state_t)
                raw = dist.sample()
                action = torch.tanh(raw).squeeze(0)
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
            except Exception:
                action = torch.zeros(self.action_dim, dtype=torch.float32, device=self.device)
        return action.cpu().numpy().astype(np.float32)

    def store(self, transition: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]) -> None:
        state, action, reward, next_state, done = transition
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim == 0:
            action_arr = np.asarray([float(action_arr)] * self.action_dim, dtype=np.float32)
        self._trajectory.append((state, action_arr, reward, next_state, done))
        self.total_steps += 1

    def can_learn(self) -> bool:
        return len(self._trajectory) >= self._min_rollout_steps

    def learn_step(self) -> Optional[float]:
        if not self.can_learn():
            return None
        return self._update_from_trajectory()

    def end_episode(self) -> Optional[float]:
        return self._update_from_trajectory()

    def _update_from_trajectory(self) -> Optional[float]:
        if not self._trajectory:
            return None

        states = np.asarray([t[0] for t in self._trajectory], dtype=np.float32)
        actions = np.asarray([t[1] for t in self._trajectory], dtype=np.float32)
        rewards = [float(t[2]) for t in self._trajectory]
        next_states = np.asarray([t[3] for t in self._trajectory], dtype=np.float32)
        dones = np.asarray([float(t[4]) for t in self._trajectory], dtype=np.float32)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        states_t = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        next_states_t = torch.nan_to_num(next_states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        actions_t = torch.nan_to_num(actions_t, nan=0.0, posinf=1.0, neginf=-1.0)

        with torch.no_grad():
            values_t = self.critic(states_t).squeeze(1)
            next_values_t = self.critic(next_states_t).squeeze(1)
            deltas_t = rewards_t + self.config.gamma * (1.0 - dones_t) * next_values_t - values_t

            advantages_t = torch.zeros_like(deltas_t)
            gae = torch.zeros(1, dtype=torch.float32, device=self.device)
            for idx in range(deltas_t.shape[0] - 1, -1, -1):
                mask = 1.0 - dones_t[idx]
                gae = deltas_t[idx] + self.config.gamma * self._gae_lambda * mask * gae
                advantages_t[idx] = gae

            returns_t = advantages_t + values_t
            adv_std = advantages_t.std(unbiased=False)
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-6)
            returns_t = torch.nan_to_num(returns_t, nan=0.0, posinf=1000.0, neginf=-1000.0)
            advantages_t = torch.nan_to_num(advantages_t, nan=0.0, posinf=1000.0, neginf=-1000.0)

            old_dist = self._distribution(states_t)
            old_log_prob = self._squashed_log_prob(old_dist, actions_t)

        epochs = 4 if self.algorithm == "ppo" else 3
        if self.algorithm == "trpo":
            epochs = 2
        if self.algorithm == "a2c":
            epochs = 1

        num_samples = int(states_t.shape[0])
        batch_size = min(num_samples, self._minibatch_size)
        total_loss = 0.0
        update_count = 0
        for _ in range(epochs):
            order = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                mb_idx = order[start : start + batch_size]
                states_mb = states_t[mb_idx]
                actions_mb = actions_t[mb_idx]
                returns_mb = returns_t[mb_idx]
                adv_mb = advantages_t[mb_idx]
                old_log_prob_mb = old_log_prob[mb_idx]

                dist = self._distribution(states_mb)
                log_prob = self._squashed_log_prob(dist, actions_mb)
                entropy = dist.entropy().sum(dim=1)
                values = self.critic(states_mb).squeeze(1)

                if self.algorithm == "ppo":
                    ratio = torch.exp(log_prob - old_log_prob_mb)
                    s1 = ratio * adv_mb
                    s2 = torch.clamp(ratio, 1.0 - self._ppo_clip_range, 1.0 + self._ppo_clip_range) * adv_mb
                    actor_loss = -torch.min(s1, s2).mean() - self._entropy_coef * entropy.mean()
                elif self.algorithm == "trpo":
                    ratio = torch.exp(log_prob - old_log_prob_mb)
                    approx_kl = (old_log_prob_mb - log_prob).mean()
                    actor_loss = -(ratio * adv_mb).mean() + 0.01 * approx_kl - self._entropy_coef * entropy.mean()
                elif self.algorithm == "sac":
                    actor_loss = -(log_prob * adv_mb).mean() - self._entropy_coef * entropy.mean()
                else:
                    actor_loss = -(log_prob * adv_mb).mean() - self._entropy_coef * entropy.mean()

                critic_loss = nn.functional.smooth_l1_loss(values, returns_mb)
                loss = actor_loss + 0.5 * critic_loss

                if not torch.isfinite(loss):
                    self._trajectory.clear()
                    return None

                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                actor_norm = float(torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std], self._max_grad_norm))
                critic_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm))
                grad_norm = max(actor_norm, critic_norm)
                self.actor_optimizer.step()
                self.critic_optimizer.step()

                self.learn_steps += 1
                self._step_lr_schedule(float(loss.item()), grad_norm)
                total_loss += float(loss.item())
                update_count += 1

        self._trajectory.clear()
        return total_loss / max(1, update_count)


class PPOAgent(BaseContinuousAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, algorithm="ppo", planned_steps=planned_steps)


class A2CAgent(BaseContinuousAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, algorithm="a2c", planned_steps=planned_steps)

    def _update_from_trajectory(self) -> Optional[float]:
        if not self._trajectory:
            return None

        states = np.asarray([t[0] for t in self._trajectory], dtype=np.float32)
        actions = np.asarray([t[1] for t in self._trajectory], dtype=np.float32)
        rewards = [float(t[2]) for t in self._trajectory]
        next_states = np.asarray([t[3] for t in self._trajectory], dtype=np.float32)
        dones = np.asarray([float(t[4]) for t in self._trajectory], dtype=np.float32)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        states_t = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        next_states_t = torch.nan_to_num(next_states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        actions_t = torch.nan_to_num(actions_t, nan=0.0, posinf=1.0, neginf=-1.0)

        with torch.no_grad():
            values_t = self.critic(states_t).squeeze(1)
            next_values_t = self.critic(next_states_t).squeeze(1)
            deltas_t = rewards_t + self.config.gamma * (1.0 - dones_t) * next_values_t - values_t

            advantages_t = torch.zeros_like(deltas_t)
            gae = torch.zeros(1, dtype=torch.float32, device=self.device)
            for idx in range(deltas_t.shape[0] - 1, -1, -1):
                mask = 1.0 - dones_t[idx]
                gae = deltas_t[idx] + self.config.gamma * self._gae_lambda * mask * gae
                advantages_t[idx] = gae

            returns_t = advantages_t + values_t
            adv_std = advantages_t.std(unbiased=False)
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-6)
            returns_t = torch.nan_to_num(returns_t, nan=0.0, posinf=1000.0, neginf=-1000.0)
            advantages_t = torch.nan_to_num(advantages_t, nan=0.0, posinf=1000.0, neginf=-1000.0)

        num_samples = int(states_t.shape[0])
        batch_size = min(num_samples, self._minibatch_size)
        order = torch.randperm(num_samples, device=self.device)
        total_loss = 0.0
        update_count = 0

        for start in range(0, num_samples, batch_size):
            mb_idx = order[start : start + batch_size]
            states_mb = states_t[mb_idx]
            actions_mb = actions_t[mb_idx]
            returns_mb = returns_t[mb_idx]
            adv_mb = advantages_t[mb_idx].detach()

            dist = self._distribution(states_mb)
            log_prob = self._squashed_log_prob(dist, actions_mb)
            entropy = dist.entropy().sum(dim=1)
            values = self.critic(states_mb).squeeze(1)

            actor_loss = -(log_prob * adv_mb).mean() - self._entropy_coef * entropy.mean()
            critic_loss = nn.functional.smooth_l1_loss(values, returns_mb)
            loss = actor_loss + 0.5 * critic_loss

            if not torch.isfinite(loss):
                self._trajectory.clear()
                return None

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            actor_norm = float(torch.nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std], self._max_grad_norm))
            critic_norm = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm))
            grad_norm = max(actor_norm, critic_norm)
            self.actor_optimizer.step()
            self.critic_optimizer.step()

            self.learn_steps += 1
            self._step_lr_schedule(float(loss.item()), grad_norm)
            total_loss += float(loss.item())
            update_count += 1

        self._trajectory.clear()
        return total_loss / max(1, update_count)


class TRPOAgent(BaseContinuousAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        super().__init__(state_dim, action_dim, config, algorithm="trpo", planned_steps=planned_steps)
        self._max_kl = 0.01
        self._cg_iters = 10
        self._cg_damping = 0.1
        self._line_search_fractions = [0.5**i for i in range(10)]

    def _policy_parameters(self) -> List[torch.Tensor]:
        return list(self.actor.parameters()) + [self.log_std]

    def _flatten_tensors(self, tensors: List[Optional[torch.Tensor]], params: List[torch.Tensor]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for grad, param in zip(tensors, params):
            if grad is None:
                chunks.append(torch.zeros_like(param).view(-1))
            else:
                chunks.append(grad.contiguous().view(-1))
        if not chunks:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        return torch.cat(chunks)

    def _flat_params(self) -> torch.Tensor:
        params = self._policy_parameters()
        if not params:
            return torch.tensor([], dtype=torch.float32, device=self.device)
        return torch.cat([p.data.view(-1) for p in params])

    def _set_flat_params(self, flat_params: torch.Tensor) -> None:
        params = self._policy_parameters()
        offset = 0
        for param in params:
            numel = param.numel()
            param.data.copy_(flat_params[offset : offset + numel].view_as(param))
            offset += numel

    def _flat_grad(self, loss: torch.Tensor, retain_graph: bool = False, create_graph: bool = False) -> torch.Tensor:
        params = self._policy_parameters()
        grads = torch.autograd.grad(
            loss,
            params,
            retain_graph=retain_graph,
            create_graph=create_graph,
            allow_unused=True,
        )
        return self._flatten_tensors(list(grads), params)

    def _fisher_vector_product(self, vector: torch.Tensor, states_t: torch.Tensor, old_mean: torch.Tensor, old_std: torch.Tensor) -> torch.Tensor:
        params = self._policy_parameters()
        dist = self._distribution(states_t)
        old_dist = torch.distributions.Normal(old_mean, old_std)
        kl = torch.distributions.kl_divergence(old_dist, dist).sum(dim=1).mean()
        grad_kl = torch.autograd.grad(kl, params, create_graph=True, allow_unused=True)
        flat_grad_kl = self._flatten_tensors(list(grad_kl), params)
        grad_vector = (flat_grad_kl * vector).sum()
        hessian_vector = torch.autograd.grad(grad_vector, params, retain_graph=True, allow_unused=True)
        flat_hessian_vector = self._flatten_tensors(list(hessian_vector), params)
        return flat_hessian_vector + self._cg_damping * vector

    def _conjugate_gradient(self, b: torch.Tensor, states_t: torch.Tensor, old_mean: torch.Tensor, old_std: torch.Tensor) -> torch.Tensor:
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.dot(r, r)

        for _ in range(self._cg_iters):
            hvp = self._fisher_vector_product(p, states_t, old_mean, old_std)
            denom = torch.dot(p, hvp) + 1e-8
            alpha = rs_old / denom
            x = x + alpha * p
            r = r - alpha * hvp
            rs_new = torch.dot(r, r)
            if torch.sqrt(rs_new) < 1e-10:
                break
            beta = rs_new / (rs_old + 1e-8)
            p = r + beta * p
            rs_old = rs_new
        return x

    def _fit_critic(self, states_t: torch.Tensor, returns_t: torch.Tensor) -> Tuple[float, float]:
        num_samples = int(states_t.shape[0])
        batch_size = min(num_samples, self._minibatch_size)
        critic_losses: List[float] = []
        max_grad = 0.0

        for _ in range(3):
            order = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, batch_size):
                mb_idx = order[start : start + batch_size]
                states_mb = states_t[mb_idx]
                returns_mb = returns_t[mb_idx]
                values = self.critic(states_mb).squeeze(1)
                critic_loss = nn.functional.smooth_l1_loss(values, returns_mb)
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_grad = float(torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self._max_grad_norm))
                self.critic_optimizer.step()
                critic_losses.append(float(critic_loss.item()))
                max_grad = max(max_grad, critic_grad)

        return float(np.mean(critic_losses)) if critic_losses else 0.0, max_grad

    def _update_from_trajectory(self) -> Optional[float]:
        if not self._trajectory:
            return None

        states = np.asarray([t[0] for t in self._trajectory], dtype=np.float32)
        actions = np.asarray([t[1] for t in self._trajectory], dtype=np.float32)
        rewards = [float(t[2]) for t in self._trajectory]
        next_states = np.asarray([t[3] for t in self._trajectory], dtype=np.float32)
        dones = np.asarray([float(t[4]) for t in self._trajectory], dtype=np.float32)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)
        states_t = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        next_states_t = torch.nan_to_num(next_states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        actions_t = torch.nan_to_num(actions_t, nan=0.0, posinf=1.0, neginf=-1.0)

        with torch.no_grad():
            values_t = self.critic(states_t).squeeze(1)
            next_values_t = self.critic(next_states_t).squeeze(1)
            deltas_t = rewards_t + self.config.gamma * (1.0 - dones_t) * next_values_t - values_t

            advantages_t = torch.zeros_like(deltas_t)
            gae = torch.zeros(1, dtype=torch.float32, device=self.device)
            for idx in range(deltas_t.shape[0] - 1, -1, -1):
                mask = 1.0 - dones_t[idx]
                gae = deltas_t[idx] + self.config.gamma * self._gae_lambda * mask * gae
                advantages_t[idx] = gae

            returns_t = advantages_t + values_t
            adv_std = advantages_t.std(unbiased=False)
            advantages_t = (advantages_t - advantages_t.mean()) / (adv_std + 1e-6)
            returns_t = torch.nan_to_num(returns_t, nan=0.0, posinf=1000.0, neginf=-1000.0)
            advantages_t = torch.nan_to_num(advantages_t, nan=0.0, posinf=1000.0, neginf=-1000.0)

            old_dist = self._distribution(states_t)
            old_log_prob = self._squashed_log_prob(old_dist, actions_t)
            old_mean = old_dist.loc.detach()
            old_std = old_dist.scale.detach()

        critic_loss, critic_grad_norm = self._fit_critic(states_t, returns_t)

        dist = self._distribution(states_t)
        log_prob = self._squashed_log_prob(dist, actions_t)
        ratio = torch.exp(log_prob - old_log_prob)
        surrogate = (ratio * advantages_t).mean()
        policy_loss = -surrogate

        grad = self._flat_grad(policy_loss, retain_graph=True)
        if grad.numel() == 0 or not torch.isfinite(grad).all():
            self._trajectory.clear()
            return critic_loss

        grad_norm = float(torch.norm(grad, p=2).item())
        if grad_norm < 1e-10:
            self.learn_steps += 1
            self._step_lr_schedule(critic_loss, max(grad_norm, critic_grad_norm))
            self._trajectory.clear()
            return critic_loss

        step_dir = self._conjugate_gradient(grad, states_t, old_mean, old_std)
        fisher_step = self._fisher_vector_product(step_dir, states_t, old_mean, old_std)
        shs = 0.5 * torch.dot(step_dir, fisher_step)
        if not torch.isfinite(shs) or shs.item() <= 0.0:
            self.learn_steps += 1
            self._step_lr_schedule(critic_loss, max(grad_norm, critic_grad_norm))
            self._trajectory.clear()
            return critic_loss

        step_scale = math.sqrt(self._max_kl / (shs.item() + 1e-12))
        full_step = step_dir * step_scale
        old_params = self._flat_params().detach().clone()
        old_surrogate = float(surrogate.detach().item())

        improved = False
        final_policy_loss = float(policy_loss.detach().item())
        for frac in self._line_search_fractions:
            candidate = old_params + frac * full_step
            self._set_flat_params(candidate)
            with torch.no_grad():
                cand_dist = self._distribution(states_t)
                cand_log_prob = self._squashed_log_prob(cand_dist, actions_t)
                cand_ratio = torch.exp(cand_log_prob - old_log_prob)
                cand_surrogate = (cand_ratio * advantages_t).mean()
                cand_kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(old_mean, old_std), cand_dist
                ).sum(dim=1).mean()

            improve = float(cand_surrogate.item()) - old_surrogate
            if torch.isfinite(cand_kl) and float(cand_kl.item()) <= self._max_kl and improve > 0.0:
                improved = True
                final_policy_loss = -float(cand_surrogate.item())
                break

        if not improved:
            self._set_flat_params(old_params)

        self.learn_steps += 1
        total_loss = final_policy_loss + critic_loss
        self._step_lr_schedule(total_loss, max(grad_norm, critic_grad_norm))
        self._trajectory.clear()
        return total_loss


class SACAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, planned_steps: int = 100000) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = get_device()

        hidden_sizes = parse_hidden_layers(config.hidden_layers)
        self.actor_mean = _make_mlp(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.actor_log_std = _make_mlp(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)

        self.q1 = ContinuousQNetwork(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.q2 = ContinuousQNetwork(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.target_q1 = ContinuousQNetwork(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.target_q2 = ContinuousQNetwork(state_dim, action_dim, hidden_sizes, config.activation_function).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        self.target_q1.eval()
        self.target_q2.eval()

        self.actor_optimizer = optim.Adam(
            list(self.actor_mean.parameters()) + list(self.actor_log_std.parameters()),
            lr=config.learning_rate,
        )
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=config.learning_rate)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=config.learning_rate)

        self.log_alpha = torch.tensor(math.log(0.2), dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=max(1e-5, config.learning_rate * 0.5))
        self.target_entropy = -float(action_dim)

        self._base_lr = float(config.learning_rate)
        self._min_lr = max(0.0, float(config.min_learning_rate))
        self._target_lr = max(self._min_lr, self._base_lr * max(0.0, float(config.lr_decay)))
        self._lr_strategy = str(config.lr_strategy).strip().lower()
        self._planned_decay_steps = max(1, int(planned_steps))
        self._current_lr = self._base_lr
        self._best_loss = float("inf")
        self._loss_bad_steps = 0
        self._loss_patience = 50
        self._loss_tolerance = 1e-4
        self._max_grad_norm = 1.0

        self._polyak = 0.005
        self.replay = ContinuousReplayBuffer(config.replay_size)
        self.learn_steps = 0
        self.total_steps = 0

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def _set_optimizer_lr(self, value: float) -> None:
        lr = max(self._min_lr, float(value))
        for optimizer in (self.actor_optimizer, self.q1_optimizer, self.q2_optimizer, self.alpha_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = lr
        self._current_lr = lr

    def _schedule_progress(self) -> float:
        return min(1.0, max(0.0, float(self.learn_steps) / float(self._planned_decay_steps)))

    def _scheduled_lr(self) -> float:
        progress = self._schedule_progress()
        if self._lr_strategy == "linear":
            return self._base_lr + (self._target_lr - self._base_lr) * progress
        if self._lr_strategy == "cosine":
            return self._target_lr + 0.5 * (self._base_lr - self._target_lr) * (1.0 + math.cos(math.pi * progress))
        ratio = self._target_lr / max(self._base_lr, 1e-12)
        return self._base_lr * (ratio ** progress)

    def _step_lr_schedule(self, loss_value: float, grad_norm: float) -> None:
        if self._lr_strategy == "loss-based":
            if loss_value + self._loss_tolerance < self._best_loss:
                self._best_loss = loss_value
                self._loss_bad_steps = 0
            else:
                self._loss_bad_steps += 1
            if self._loss_bad_steps >= self._loss_patience:
                factor = min(0.99, max(0.1, float(self.config.lr_decay)))
                self._set_optimizer_lr(self._current_lr * factor)
                self._loss_bad_steps = 0
            return

        if self._lr_strategy == "guarded natural gradient":
            baseline = self._scheduled_lr()
            guard_strength = max(0.0, float(self.config.lr_decay))
            guarded_lr = baseline / (1.0 + guard_strength * max(0.0, grad_norm))
            self._set_optimizer_lr(guarded_lr)
            return

        self._set_optimizer_lr(self._scheduled_lr())

    def _actor_distribution(self, states_t: torch.Tensor) -> torch.distributions.Normal:
        safe_states = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        mean = self.actor_mean(safe_states)
        log_std = self.actor_log_std(safe_states)
        mean = torch.nan_to_num(mean, nan=0.0, posinf=1.0, neginf=-1.0)
        log_std = torch.nan_to_num(log_std, nan=-1.0, posinf=1.0, neginf=-5.0)
        log_std = torch.clamp(log_std, min=-5.0, max=2.0)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def _sample_action_and_log_prob(self, states_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dist = self._actor_distribution(states_t)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        log_prob = torch.nan_to_num(log_prob, nan=0.0, posinf=100.0, neginf=-100.0)
        return action, log_prob

    def select_action(self, state: np.ndarray, epsilon: float) -> np.ndarray:
        state_arr = np.asarray(state, dtype=np.float32)
        state_arr = np.nan_to_num(state_arr, nan=0.0, posinf=10.0, neginf=-10.0)
        state_t = torch.tensor(state_arr, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, _ = self._sample_action_and_log_prob(state_t)
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def store(self, transition: Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]) -> None:
        state, action, reward, next_state, done = transition
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_arr.size == 1:
            action_arr = np.repeat(action_arr, self.action_dim)
        if action_arr.size != self.action_dim:
            action_arr = np.resize(action_arr, self.action_dim)
        action_arr = np.clip(action_arr, -1.0, 1.0)
        self.replay.add((state, action_arr, float(reward), next_state, bool(done)))
        self.total_steps += 1

    def can_learn(self) -> bool:
        warm = max(1, int(self.config.replay_warmup))
        cadence = max(1, int(self.config.learning_cadence))
        return len(self.replay) >= warm and len(self.replay) >= int(self.config.batch_size) and self.total_steps % cadence == 0

    def _soft_update_targets(self) -> None:
        with torch.no_grad():
            tau = float(self._polyak)
            for target_param, src_param in zip(self.target_q1.parameters(), self.q1.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * src_param.data)
            for target_param, src_param in zip(self.target_q2.parameters(), self.q2.parameters()):
                target_param.data.mul_(1.0 - tau).add_(tau * src_param.data)

    def learn_step(self) -> Optional[float]:
        if not self.can_learn():
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(int(self.config.batch_size))
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        states_t = torch.nan_to_num(states_t, nan=0.0, posinf=10.0, neginf=-10.0)
        actions_t = torch.nan_to_num(actions_t, nan=0.0, posinf=1.0, neginf=-1.0)
        next_states_t = torch.nan_to_num(next_states_t, nan=0.0, posinf=10.0, neginf=-10.0)

        with torch.no_grad():
            next_actions_t, next_log_prob_t = self._sample_action_and_log_prob(next_states_t)
            next_q1 = self.target_q1(next_states_t, next_actions_t)
            next_q2 = self.target_q2(next_states_t, next_actions_t)
            next_q = torch.min(next_q1, next_q2) - self.alpha.detach() * next_log_prob_t
            q_target = rewards_t + (1.0 - dones_t) * float(self.config.gamma) * next_q
            q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1000.0, neginf=-1000.0)

        q1_pred = self.q1(states_t, actions_t)
        q2_pred = self.q2(states_t, actions_t)
        q1_loss = nn.functional.mse_loss(q1_pred, q_target)
        q2_loss = nn.functional.mse_loss(q2_pred, q_target)

        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        q1_grad = float(torch.nn.utils.clip_grad_norm_(self.q1.parameters(), self._max_grad_norm))
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        q2_grad = float(torch.nn.utils.clip_grad_norm_(self.q2.parameters(), self._max_grad_norm))
        self.q2_optimizer.step()

        sampled_actions_t, log_prob_t = self._sample_action_and_log_prob(states_t)
        q1_pi = self.q1(states_t, sampled_actions_t)
        q2_pi = self.q2(states_t, sampled_actions_t)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob_t - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad = float(
            torch.nn.utils.clip_grad_norm_(
                list(self.actor_mean.parameters()) + list(self.actor_log_std.parameters()), self._max_grad_norm
            )
        )
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha * (log_prob_t + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        alpha_grad = float(torch.nn.utils.clip_grad_norm_([self.log_alpha], self._max_grad_norm))
        self.alpha_optimizer.step()

        self._soft_update_targets()
        self.learn_steps += 1
        total_loss = float((q1_loss + q2_loss + actor_loss + alpha_loss).item())
        grad_norm = max(q1_grad, q2_grad, actor_grad, alpha_grad)
        self._step_lr_schedule(total_loss, grad_norm)
        return total_loss

    def end_episode(self) -> Optional[float]:
        return None


class LunarLanderEnv:
    def __init__(
        self,
        gravity: float = -10.0,
        continuous: bool = False,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.gravity = float(gravity)
        self.continuous = bool(continuous)
        self.enable_wind = bool(enable_wind)
        self.wind_power = float(wind_power)
        self.turbulence_power = float(turbulence_power)
        self.render_mode = render_mode
        self._env_lock = threading.RLock()
        self._make_env()

    def _make_env(self) -> None:
        import gymnasium as gym

        with self._env_lock:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
            self.env = gym.make(
                "LunarLander-v3",
                continuous=self.continuous,
                gravity=self.gravity,
                enable_wind=self.enable_wind,
                wind_power=self.wind_power,
                turbulence_power=self.turbulence_power,
                render_mode=self.render_mode,
            )
            self.state_dim = int(self.env.observation_space.shape[0])
            if self.continuous:
                self.action_dim = int(self.env.action_space.shape[0])
            else:
                self.action_dim = int(self.env.action_space.n)
            self.last_obs = None
            self.last_frame = None

    def update_config(self, gravity: float, continuous: bool, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.gravity = float(gravity)
        self.continuous = bool(continuous)
        self.enable_wind = bool(enable_wind)
        self.wind_power = float(wind_power)
        self.turbulence_power = float(turbulence_power)
        self._make_env()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        with self._env_lock:
            obs, _ = self.env.reset(seed=seed)
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return self.last_obs.copy()

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        with self._env_lock:
            env_action: object
            if self.continuous:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if arr.size == 1:
                    arr = np.repeat(arr, self.action_dim)
                if arr.size != self.action_dim:
                    arr = np.resize(arr, self.action_dim)
                env_action = np.clip(arr, -1.0, 1.0)
            else:
                env_action = int(action)
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = bool(terminated or truncated)
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return self.last_obs.copy(), float(reward), done, info

    def render(self) -> Optional[np.ndarray]:
        acquired = self._env_lock.acquire(blocking=False)
        if not acquired:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()
        try:
            frame = self.env.render()
            if frame is None:
                if self.last_frame is None:
                    return None
                return self.last_frame.copy()
            self.last_frame = np.asarray(frame)
            return self.last_frame.copy()
        finally:
            self._env_lock.release()

    def is_reachable(self, obs: np.ndarray) -> bool:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.size < 2:
            return False
        return bool(np.isfinite(arr[0]) and np.isfinite(arr[1]) and abs(arr[0]) < 5.0 and abs(arr[1]) < 5.0)

    def close(self) -> None:
        with self._env_lock:
            if getattr(self, "env", None) is not None:
                self.env.close()
                self.env = None


class Trainer:
    def __init__(self, env: Optional[LunarLanderEnv] = None, env_factory: Optional[Callable[[], LunarLanderEnv]] = None) -> None:
        if env is not None:
            self.env = env
        elif env_factory is not None:
            self.env = env_factory()
        else:
            self.env = LunarLanderEnv()
        self.policy_configs: Dict[str, AgentConfig] = {k: AgentConfig(**vars(v)) for k, v in POLICY_DEFAULTS.items()}
        self.agents: Dict[str, BaseDQNAgent] = {}
        self._training_plan_steps: Dict[str, int] = {}
        self._csv_samples: List[Tuple[int, int, float, float, float, bool]] = []
        self._agent_classes = {
            "DuelingDQN": DuelingDQN,
            "D3QN": D3QN,
            "DDQN+PER": DDQNPER,
            "PPO": PPOAgent,
            "A2C": A2CAgent,
            "TRPO": TRPOAgent,
            "SAC": SACAgent,
        }

    def set_policy_config(self, policy: str, **kwargs: object) -> None:
        config = self.policy_configs[policy]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def get_policy_config(self, policy: str) -> AgentConfig:
        return AgentConfig(**vars(self.policy_configs[policy]))

    def get_current_learning_rate(self, policy: str) -> float:
        agent = self.agents.get(policy)
        if agent is None:
            return float(self.policy_configs[policy].learning_rate)
        return float(agent._current_lr)

    def _get_or_create_agent(self, policy: str) -> BaseDQNAgent:
        if policy not in self.agents:
            config = self.get_policy_config(policy)
            cls = self._agent_classes[policy]
            planned_steps = self._training_plan_steps.get(policy, 100000)
            self.agents[policy] = cls(self.env.state_dim, self.env.action_dim, config, planned_steps=planned_steps)
        return self.agents[policy]

    def set_training_plan(self, policy: str, episodes: int, max_steps: int) -> None:
        if policy in CONTINUOUS_POLICIES:
            cadence = max(1, int(self.policy_configs[policy].learning_cadence))
            updates_per_episode = max(1, int(math.ceil(max_steps / cadence)))
            self._training_plan_steps[policy] = max(1, int(episodes) * int(updates_per_episode))
            return

        cadence = max(1, int(self.policy_configs[policy].learning_cadence))
        total_steps = max(1, int(episodes) * int(max_steps))
        self._training_plan_steps[policy] = max(1, total_steps // cadence)

    def reset_policy_agent(self, policy: str) -> None:
        self.agents.pop(policy, None)

    def rebuild_environment(self, gravity: float, continuous: bool, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.env.update_config(gravity, continuous, enable_wind, wind_power, turbulence_power)
        self.agents.clear()

    def run_episode(
        self,
        policy: str,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, object]:
        agent = self._get_or_create_agent(policy)
        state = self.env.reset()
        total_reward = 0.0
        transitions: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        best_x = float(state[0]) if len(state) > 0 else float("nan")

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            best_x = max(best_x, float(next_state[0])) if len(next_state) > 0 else best_x
            transition = (state.copy(), action, float(reward), next_state.copy(), bool(done))
            transitions.append(transition)
            agent.store(transition)
            agent.learn_step()
            if progress_callback is not None:
                progress_callback(step)
            state = next_state
            if done:
                break

        end_episode = getattr(agent, "end_episode", None)
        if callable(end_episode):
            end_episode()

        return {
            "reward": float(total_reward),
            "steps": int(len(transitions)),
            "best_x": float(best_x),
            "final_state": state,
            "transitions": transitions,
        }

    def train(
        self,
        policy: str,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
    ) -> List[float]:
        rewards: List[float] = []
        csv_rows: List[Tuple[int, int, float, float, float, bool]] = []

        for episode_idx in range(1, num_episodes + 1):
            result = self.run_episode(policy, epsilon=epsilon, max_steps=max_steps)
            rewards.append(float(result["reward"]))
            if save_csv is not None:
                transitions = result["transitions"]
                for step_idx, transition in enumerate(transitions, start=1):
                    state, action, reward, next_state, done = transition
                    sx = float(state[0]) if len(state) > 0 else float("nan")
                    nx = float(next_state[0]) if len(next_state) > 0 else float("nan")
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    action_scalar = float(action_arr[0]) if action_arr.size > 0 else float("nan")
                    csv_rows.append((episode_idx, step_idx, sx, action_scalar, float(reward), bool(done)))
                    self._csv_samples.append((episode_idx, step_idx, sx, nx, float(reward), bool(done)))

        if save_csv is not None:
            base_name = os.path.basename(save_csv)
            if not base_name:
                base_name = f"samplings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = os.path.join(os.path.dirname(__file__), "results_csv")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base_name}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "step", "state_x", "action", "reward", "done"])
                writer.writerows(csv_rows)

        return rewards

    def save_plot_png(
        self,
        figure,
        policy: str,
        eps_max: float,
        eps_min: float,
        learning_rate: float,
        gamma: float,
        episodes: int,
        max_steps: int,
    ) -> str:
        def _sci_token(value: float) -> str:
            return f"{float(value):.2e}".replace("+", "")

        out_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_policy = policy.replace("+", "plus")
        lr_token = _sci_token(learning_rate)
        name = (
            f"{safe_policy}_eps{eps_max}_{eps_min}_lr{lr_token}_"
            f"g{gamma}_ep{episodes}_ms{max_steps}_{ts}.png"
        )
        path = os.path.join(out_dir, name)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def close(self) -> None:
        self.env.close()
