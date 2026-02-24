from __future__ import annotations

import csv
import random
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
    def __init__(self, capacity: int = 20_000) -> None:
        self.capacity = max(1, int(capacity))
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, indices, weights

    def update_priorities(self, _indices: np.ndarray, _priorities: np.ndarray) -> None:
        return


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int = 20_000, alpha: float = 0.6, epsilon: float = 1e-6) -> None:
        self.capacity = max(1, int(capacity))
        self.alpha = float(max(0.0, alpha))
        self.epsilon = float(max(0.0, epsilon))
        self.buffer: deque[Transition] = deque(maxlen=self.capacity)
        self.priorities: deque[float] = deque(maxlen=self.capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def push(self, transition: Transition) -> None:
        self.buffer.append(transition)
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(float(max_prio))

    def sample(self, batch_size: int, beta: float = 0.4) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        priorities = np.asarray(self.priorities, dtype=np.float64)
        scaled = np.power(priorities + self.epsilon, self.alpha)
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False, p=probs)
        batch = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = np.power(total * probs[indices], -beta)
        weights /= weights.max()
        return batch, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = float(max(prio, self.epsilon))


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_size, 1)
        self.advantage_head = nn.Linear(hidden_size, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature(x)
        values = self.value_head(feats)
        advantages = self.advantage_head(feats)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


@dataclass
class DQNDefaults:
    learning_rate: float = 1e-3
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.995
    replay_buffer_size: int = 20_000
    batch_size: int = 128
    target_update_frequency: int = 100
    hidden_size: int = 128
    warmup_steps: int = 250


class FrozenLakeEnv:
    ACTIONS = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}  # left, down, right, up

    def __init__(
        self,
        is_slippery: bool = True,
        map_name: str = "4x4",
        success_rate: float = 1.0 / 3.0,
        seed: Optional[int] = None,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.is_slippery = bool(is_slippery)
        self.map_name = str(map_name)
        self.success_rate = float(success_rate)
        self.render_mode = render_mode
        self.seed = seed

        self.env = self._make_env(seed=seed)
        self.n_states = int(self.env.observation_space.n)
        self.n_actions = int(self.env.action_space.n)
        self.state, _ = self.env.reset(seed=seed)

        if not pygame.get_init():
            pygame.init()

    def _make_env(self, seed: Optional[int] = None):
        env = gym.make(
            "FrozenLake-v1",
            map_name=self.map_name,
            is_slippery=False,
            render_mode=self.render_mode,
        )
        self._apply_transition_model(env)
        env.reset(seed=seed)
        return env

    def _transition_actions(self, action: int) -> list[int]:
        if not self.is_slippery:
            return [action]
        return [(action - 1) % 4, action, (action + 1) % 4]

    def _transition_probs(self) -> list[float]:
        if not self.is_slippery:
            return [1.0]
        success = float(min(max(self.success_rate, 0.0), 1.0))
        side = (1.0 - success) / 2.0
        return [side, success, side]

    @staticmethod
    def _move(row: int, col: int, action: int, nrow: int, ncol: int) -> tuple[int, int]:
        if action == 0:
            col = max(0, col - 1)
        elif action == 1:
            row = min(nrow - 1, row + 1)
        elif action == 2:
            col = min(ncol - 1, col + 1)
        elif action == 3:
            row = max(0, row - 1)
        return row, col

    def _apply_transition_model(self, env) -> None:
        unwrapped = env.unwrapped
        desc = np.asarray(unwrapped.desc)
        nrow, ncol = desc.shape
        n_states = nrow * ncol

        probs = self._transition_probs()
        new_p = {s: {a: [] for a in range(4)} for s in range(n_states)}

        def state_index(row: int, col: int) -> int:
            return row * ncol + col

        for row in range(nrow):
            for col in range(ncol):
                s = state_index(row, col)
                tile = desc[row, col]
                terminal = tile in (b"H", b"G")
                for action in range(4):
                    if terminal:
                        new_p[s][action] = [(1.0, s, 0.0, True)]
                        continue

                    action_probs: dict[tuple[int, float, bool], float] = {}
                    for prob, sampled_action in zip(probs, self._transition_actions(action)):
                        nr, nc = self._move(row, col, sampled_action, nrow, ncol)
                        ns = state_index(nr, nc)
                        next_tile = desc[nr, nc]
                        done = bool(next_tile in (b"H", b"G"))
                        reward = 1.0 if next_tile == b"G" else 0.0
                        key = (ns, reward, done)
                        action_probs[key] = action_probs.get(key, 0.0) + float(prob)

                    new_p[s][action] = [
                        (float(prob), ns, float(reward), bool(done))
                        for (ns, reward, done), prob in action_probs.items()
                    ]

        unwrapped.P = new_p
        unwrapped.is_slippery = self.is_slippery

    def reset(self, seed: Optional[int] = None) -> int:
        self.state, _ = self.env.reset(seed=seed)
        return int(self.state)

    def update_config(self, is_slippery: bool, map_name: str, success_rate: float) -> None:
        reconfigure = (
            bool(is_slippery) != self.is_slippery
            or str(map_name) != self.map_name
            or abs(float(success_rate) - self.success_rate) > 1e-9
        )
        self.is_slippery = bool(is_slippery)
        self.map_name = str(map_name)
        self.success_rate = float(success_rate)
        if reconfigure:
            self.env.close()
            self.env = self._make_env(seed=self.seed)
            self.n_states = int(self.env.observation_space.n)
            self.n_actions = int(self.env.action_space.n)
            self.reset(seed=self.seed)

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        next_state, reward, terminated, truncated, info = self.env.step(int(action))
        done = bool(terminated or truncated)
        self.state = int(next_state)
        return int(next_state), float(reward), done, dict(info)

    def state_to_row_col(self, state: int) -> tuple[int, int]:
        desc = np.asarray(self.env.unwrapped.desc)
        nrow, ncol = desc.shape
        return int(state) // ncol, int(state) % ncol

    def is_reachable(self) -> bool:
        desc = np.asarray(self.env.unwrapped.desc)
        nrow, ncol = desc.shape

        start = None
        goal = None
        for r in range(nrow):
            for c in range(ncol):
                if desc[r, c] == b"S":
                    start = (r, c)
                elif desc[r, c] == b"G":
                    goal = (r, c)

        if start is None or goal is None:
            return False

        queue = deque([start])
        visited = {start}
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for action in range(4):
                nr, nc = self._move(r, c, action, nrow, ncol)
                if (nr, nc) in visited:
                    continue
                if desc[nr, nc] == b"H":
                    continue
                visited.add((nr, nc))
                queue.append((nr, nc))
        return False

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


class DQN:
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
        hidden_size: int = DQNDefaults.hidden_size,
        warmup_steps: int = DQNDefaults.warmup_steps,
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
        self.warmup_steps = max(0, int(warmup_steps))

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.online_net = self._create_network(hidden_size)
        self.target_net = self._create_network(hidden_size)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=float(learning_rate))
        self.loss_fn = nn.MSELoss(reduction="none")
        self.replay_buffer = ReplayBuffer(replay_buffer_size)

        self.learn_steps = 0
        self.total_steps = 0

    def _create_network(self, hidden_size: int) -> nn.Module:
        return QNetwork(self.n_states, self.n_actions, hidden_size=hidden_size).to(self.device)

    def _one_hot(self, states: list[int]) -> torch.Tensor:
        indices = torch.tensor(states, dtype=torch.int64, device=self.device)
        tensor = torch.zeros((len(states), self.n_states), dtype=torch.float32, device=self.device)
        tensor.scatter_(1, indices.unsqueeze(1), 1.0)
        return tensor

    def select_action(self, state: int, epsilon: Optional[float] = None) -> int:
        eps = self.epsilon if epsilon is None else float(epsilon)
        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        with torch.no_grad():
            q_values = self.online_net(self._one_hot([int(state)]))
            return int(torch.argmax(q_values, dim=1).item())

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.push(transition)

    def _target_q_values(self, next_state_tensor: torch.Tensor) -> torch.Tensor:
        return self.target_net(next_state_tensor).max(dim=1)[0]

    def _sample_batch(self) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        return self.replay_buffer.sample(self.batch_size)

    def _update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.replay_buffer.update_priorities(indices, td_errors)

    def learn(self) -> Optional[float]:
        if len(self.replay_buffer) < max(self.batch_size, self.warmup_steps):
            return None

        batch, indices, weights = self._sample_batch()
        states = [t.state for t in batch]
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=self.device).unsqueeze(1)
        next_states = [t.next_state for t in batch]
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=self.device)
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        state_tensor = self._one_hot(states)
        next_state_tensor = self._one_hot(next_states)

        current_q = self.online_net(state_tensor).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self._target_q_values(next_state_tensor)
            target_q = rewards + self.gamma * next_q * (1.0 - dones)

        td_errors = (target_q - current_q).detach().abs().cpu().numpy()
        per_item_loss = self.loss_fn(current_q, target_q)
        loss = (per_item_loss * weight_tensor).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        self._update_priorities(indices, td_errors)
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        return float(loss.item())


class DoubleDQN(DQN):
    policy_name = "DoubleDQN"

    def _target_q_values(self, next_state_tensor: torch.Tensor) -> torch.Tensor:
        best_actions = self.online_net(next_state_tensor).argmax(dim=1, keepdim=True)
        return self.target_net(next_state_tensor).gather(1, best_actions).squeeze(1)


class DuelingDQN(DQN):
    policy_name = "DuelingDQN"

    def _create_network(self, hidden_size: int) -> nn.Module:
        return DuelingQNetwork(self.n_states, self.n_actions, hidden_size=hidden_size).to(self.device)


class PrioDQN(DQN):
    policy_name = "PrioDQN"

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
        hidden_size: int = DQNDefaults.hidden_size,
        warmup_steps: int = DQNDefaults.warmup_steps,
        device: Optional[str] = None,
        prio_alpha: float = 0.6,
        prio_beta: float = 0.4,
        prio_beta_increment: float = 1e-4,
    ) -> None:
        self.prio_alpha = float(prio_alpha)
        self.prio_beta = float(prio_beta)
        self.prio_beta_increment = float(prio_beta_increment)
        super().__init__(
            n_states=n_states,
            n_actions=n_actions,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            replay_buffer_size=replay_buffer_size,
            batch_size=batch_size,
            target_update_frequency=target_update_frequency,
            hidden_size=hidden_size,
            warmup_steps=warmup_steps,
            device=device,
        )
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_size, alpha=self.prio_alpha)

    def _sample_batch(self) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        batch, indices, weights = self.replay_buffer.sample(self.batch_size, beta=self.prio_beta)
        self.prio_beta = min(1.0, self.prio_beta + self.prio_beta_increment)
        return batch, indices, weights

    def _update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.replay_buffer.update_priorities(indices, td_errors)


class Trainer:
    def __init__(self, env: FrozenLakeEnv, base_dir: Optional[Path] = None) -> None:
        self.env = env
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent
        self.results_dir, self.plots_dir = ensure_output_dirs(self.base_dir)

    def run_episode(
        self,
        policy: DQN,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
        transition_callback: Optional[Callable[[Transition, int], None]] = None,
    ) -> dict:
        state = self.env.reset()
        total_reward = 0.0
        transitions: list[Transition] = []

        for step in range(1, int(max_steps) + 1):
            policy.total_steps += 1
            action = policy.select_action(state, epsilon=epsilon)
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
        policy: DQN,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> tuple[list[float], Optional[Path]]:
        rewards: list[float] = []
        csv_rows: list[list] = []

        policy.epsilon = float(epsilon)

        for episode in range(1, int(num_episodes) + 1):
            result = self.run_episode(
                policy=policy,
                epsilon=policy.epsilon,
                max_steps=max_steps,
                progress_callback=(lambda step, ep=episode: progress_callback(ep, step)) if progress_callback else None,
            )
            rewards.append(float(result["total_reward"]))

            for step_idx, tr in enumerate(result["transitions"], start=1):
                csv_rows.append([episode, step_idx, tr.state, tr.action, tr.next_state, tr.reward, tr.done])

        csv_path = None
        if save_csv:
            base = Path(save_csv).stem
            csv_path = self.results_dir / f"{base}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp)
                writer.writerow(["episode", "step", "state", "action", "next_state", "reward", "done"])
                writer.writerows(csv_rows)

        return rewards, csv_path
