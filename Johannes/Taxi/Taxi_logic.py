from __future__ import annotations
"""Core RL logic for Taxi: environment wrapper, agents, replay buffers, and trainer."""

import csv
import os
import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.error import ResetNeeded


def _one_hot_state(state: int, n_states: int) -> np.ndarray:
    """Encode discrete Taxi state id into a one-hot vector for neural networks."""
    vec = np.zeros(n_states, dtype=np.float32)
    vec[state] = 1.0
    return vec


class QNetwork(nn.Module):
    """Standard feed-forward Q-network used by DQN variants."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DuelingQNetwork(nn.Module):
    """Dueling architecture separating value and advantage streams."""
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
        )
        self.advantage = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )
        self.value = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        adv = self.advantage(feat)
        val = self.value(feat)
        return val + (adv - adv.mean(dim=1, keepdim=True))


@dataclass
class Transition:
    """Single replay transition tuple."""
    state: int
    action: int
    reward: float
    next_state: int
    done: bool


class ReplayBuffer:
    """Uniform experience replay buffer."""
    def __init__(self, capacity: int = 50_000) -> None:
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, k=batch_size)


class PrioritizedReplayBuffer:
    """Prioritized replay buffer with proportional sampling."""
    def __init__(self, capacity: int = 50_000, alpha: float = 0.6) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition, priority: Optional[float] = None) -> None:
        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        prio = float(priority if priority is not None else max_prio)

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max(prio, 1e-6)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]

        # Convert priorities to a probability distribution (proportional prioritization).
        probs = prios ** self.alpha
        probs = probs / probs.sum()

        # Draw batch indices according to replay priorities.
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Importance-sampling weights reduce sampling bias during optimization.
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = max(float(prio), 1e-6)


class BaseDQNAgent:
    """Base DQN agent with target network and replay-based learning."""
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 1e-3,
        gamma: float = 0.99,
        batch_size: int = 64,
        buffer_size: int = 50_000,
        target_update_freq: int = 250,
        hidden_size: int = 128,
        device: Optional[str] = None,
    ) -> None:
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.learn_steps = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.q_network = self._build_network(hidden_size).to(self.device)
        self.target_network = self._build_network(hidden_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")
        self.replay_buffer = self._build_buffer(buffer_size)

    def _build_network(self, hidden_size: int) -> nn.Module:
        return QNetwork(self.state_size, self.action_size, hidden_size)

    def _build_buffer(self, buffer_size: int):
        return ReplayBuffer(buffer_size)

    def encode_state(self, state: int) -> torch.Tensor:
        vec = _one_hot_state(state, self.state_size)
        return torch.from_numpy(vec).float().unsqueeze(0).to(self.device)

    def select_action(self, state: int, epsilon: float = 0.1) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            q_values = self.q_network(self.encode_state(state))
        return int(torch.argmax(q_values, dim=1).item())

    def store_transition(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        self.replay_buffer.add(Transition(state, action, float(reward), next_state, done))

    def _compute_target_q(self, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.target_network(next_states).max(dim=1, keepdim=True)[0]

    def learn(self) -> Optional[float]:
        """Run one gradient update step from replay buffer if enough samples exist."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer):
            transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
            weight_t = torch.from_numpy(weights).float().unsqueeze(1).to(self.device)
        else:
            transitions = self.replay_buffer.sample(self.batch_size)
            indices = None
            weight_t = torch.ones((self.batch_size, 1), device=self.device)

        states = torch.from_numpy(
            np.stack([_one_hot_state(t.state, self.state_size) for t in transitions], axis=0)
        ).float().to(self.device)
        actions = torch.tensor([t.action for t in transitions], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.from_numpy(
            np.stack([_one_hot_state(t.next_state, self.state_size) for t in transitions], axis=0)
        ).float().to(self.device)
        dones = torch.tensor([t.done for t in transitions], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_pred = self.q_network(states).gather(1, actions)
        q_next = self._compute_target_q(next_states)
        # Standard bootstrapped Bellman target.
        q_target = rewards + self.gamma * q_next * (1.0 - dones)

        td_error = q_pred - q_target
        # Per-sample Huber losses allow prioritized replay re-weighting.
        losses = self.loss_fn(q_pred, q_target) * weight_t
        loss = losses.mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()

        if isinstance(self.replay_buffer, PrioritizedReplayBuffer) and indices is not None:
            # Update replay priorities from latest absolute TD errors.
            priorities = torch.abs(td_error).detach().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)

        self.learn_steps += 1
        if self.learn_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return float(loss.item())


class DQN(BaseDQNAgent):
    """Vanilla DQN policy."""
    pass


class DoubleDQN(BaseDQNAgent):
    """Double DQN using online argmax and target-network evaluation."""
    def _compute_target_q(self, next_states: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            next_actions = torch.argmax(self.q_network(next_states), dim=1, keepdim=True)
            return self.target_network(next_states).gather(1, next_actions)


class DuelingDQN(BaseDQNAgent):
    """DQN with dueling network head."""
    def _build_network(self, hidden_size: int) -> nn.Module:
        return DuelingQNetwork(self.state_size, self.action_size, hidden_size)


class PrioDQN(BaseDQNAgent):
    """DQN variant using prioritized replay sampling."""
    def __init__(self, *args, priority_alpha: float = 0.6, **kwargs) -> None:
        self.priority_alpha = priority_alpha
        super().__init__(*args, **kwargs)

    def _build_buffer(self, buffer_size: int):
        return PrioritizedReplayBuffer(buffer_size, alpha=self.priority_alpha)


class TaxiEnvironment:
    """Gymnasium Taxi-v3 wrapper with deterministic project-specific toggles."""
    def __init__(
        self,
        is_raining: bool = False,
        fickle_passenger: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.is_raining = is_raining
        self.fickle_passenger = fickle_passenger
        self.seed = seed
        self.render_mode = render_mode

        self.env = gym.make("Taxi-v3", render_mode=render_mode)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)

        self.episode_step = 0
        self._fickle_counter = 0

    def set_modes(self, is_raining: bool, fickle_passenger: bool) -> None:
        self.is_raining = is_raining
        self.fickle_passenger = fickle_passenger

    def reset(self, seed: Optional[int] = None) -> int:
        self.episode_step = 0
        self._fickle_counter = 0
        obs, _ = self.env.reset(seed=seed if seed is not None else self.seed)
        return int(obs)

    def _apply_rain_dynamics(self, action: int) -> int:
        """Simulate rain by deterministically perturbing movement actions."""
        if not self.is_raining:
            return action

        if action in (0, 1, 2, 3) and self.episode_step % 7 == 0:
            return (action + 1) % 4
        return action

    def _apply_fickle_reward(self, reward: float, action: int) -> float:
        """Shape rewards to model a fickle passenger behavior."""
        if not self.fickle_passenger:
            return reward

        self._fickle_counter += 1
        if action in (4, 5):
            return reward - 1.0
        if self._fickle_counter % 9 == 0:
            return reward - 1.0
        return reward

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        """Execute one environment step with optional mode effects."""
        self.episode_step += 1
        real_action = self._apply_rain_dynamics(action)
        next_state, reward, terminated, truncated, info = self.env.step(real_action)
        reward = self._apply_fickle_reward(float(reward), real_action)
        done = bool(terminated or truncated)
        info = dict(info)
        info["executed_action"] = int(real_action)
        return int(next_state), float(reward), done, info

    def render_rgb(self) -> Optional[np.ndarray]:
        """Render RGB frame; auto-reset if Gym requires reset before rendering."""
        try:
            frame = self.env.render()
        except ResetNeeded:
            self.reset()
            frame = self.env.render()
        if isinstance(frame, np.ndarray):
            return frame
        return None

    def render_text(self) -> str:
        prev_mode = self.render_mode
        if prev_mode == "ansi":
            rendered = self.env.render()
            return str(rendered) if rendered is not None else ""

        self.close()
        self.env = gym.make("Taxi-v3", render_mode="ansi")
        self.render_mode = "ansi"
        text = self.env.render()

        self.close()
        self.env = gym.make("Taxi-v3", render_mode=prev_mode)
        self.render_mode = prev_mode
        return str(text) if text is not None else ""

    def decode(self, state: int) -> Tuple[int, int, int, int]:
        return tuple(self.env.unwrapped.decode(int(state)))

    def is_reachable(self, state: int, target_state: int) -> bool:
        """Check one-step reachability from a given state via any action."""
        sim_env = gym.make("Taxi-v3")
        try:
            sim_env.reset(seed=0)
            sim_env.unwrapped.s = int(state)
            for action in range(sim_env.action_space.n):
                sim_env.unwrapped.s = int(state)
                nxt, _, _, _, _ = sim_env.step(action)
                if int(nxt) == int(target_state):
                    return True
            return False
        finally:
            sim_env.close()

    def close(self) -> None:
        self.env.close()


class Trainer:
    """Episode runner and multi-episode trainer with optional CSV transition export."""
    def __init__(
        self,
        environment: TaxiEnvironment,
        results_dir: str = "results_csv",
    ) -> None:
        self.environment = environment
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(
        self,
        policy: BaseDQNAgent,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict:
        """Run one episode and return reward, steps, and sampled transitions."""
        state = self.environment.reset()
        total_reward = 0.0
        transitions: List[Tuple[int, int, float, int, bool]] = []

        for step_idx in range(1, max_steps + 1):
            if progress_callback is not None:
                progress_callback(step_idx)

            action = policy.select_action(state, epsilon=epsilon)
            next_state, reward, done, _ = self.environment.step(action)

            policy.store_transition(state, action, reward, next_state, done)
            policy.learn()

            transitions.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            if done:
                return {
                    "total_reward": float(total_reward),
                    "steps": step_idx,
                    "transitions": transitions,
                }

        return {
            "total_reward": float(total_reward),
            "steps": max_steps,
            "transitions": transitions,
        }

    def train(
        self,
        policy: BaseDQNAgent,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
    ) -> List[float]:
        """Run multiple episodes and optionally persist sampled transitions to CSV."""
        rewards: List[float] = []
        csv_rows: List[Tuple[int, int, int, float, int, bool]] = []

        for ep in range(1, num_episodes + 1):
            result = self.run_episode(policy, epsilon=epsilon, max_steps=max_steps)
            rewards.append(float(result["total_reward"]))
            for (state, action, reward, next_state, done) in result["transitions"]:
                csv_rows.append((ep, state, action, reward, next_state, done))

        if save_csv:
            self._write_csv(save_csv, csv_rows)

        return rewards

    def _write_csv(self, base_name: str, rows: List[Tuple[int, int, int, float, int, bool]]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_base = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base_name)
        output_path = self.results_dir / f"{safe_base}_{timestamp}.csv"

        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["episode", "state", "action", "reward", "next_state", "done"])
            writer.writerows(rows)

        return output_path


def build_agent(policy_name: str, **kwargs) -> BaseDQNAgent:
    """Factory for constructing the selected DQN policy class."""
    policy_name = policy_name.strip().lower()
    registry = {
        "dqn": DQN,
        "doubledqn": DoubleDQN,
        "duelingdqn": DuelingDQN,
        "priodqn": PrioDQN,
    }
    if policy_name not in registry:
        raise ValueError(f"Unknown policy '{policy_name}'.")
    return registry[policy_name](**kwargs)
