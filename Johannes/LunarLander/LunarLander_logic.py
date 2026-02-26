from __future__ import annotations

import csv
import os
import random
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


POLICY_DEFAULTS: Dict[str, AgentConfig] = {
    "DuelingDQN": AgentConfig(gamma=0.99, learning_rate=3e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=2, activation_function="ReLU", hidden_layers="256,256,128"),
    "D3QN": AgentConfig(gamma=0.99, learning_rate=2.5e-4, replay_size=150000, batch_size=128, target_update=200, replay_warmup=8000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128"),
    "DDQN+PER": AgentConfig(gamma=0.99, learning_rate=2e-4, replay_size=200000, batch_size=128, target_update=200, replay_warmup=10000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128"),
}


class BaseDQNAgent:
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig, dueling: bool = False, prioritized: bool = False) -> None:
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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

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
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        self.replay.update_priorities(indices, td_errors.cpu().numpy())

        if self.learn_steps % self.config.target_update == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


class DuelingDQN(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig) -> None:
        super().__init__(state_dim, action_dim, config, dueling=True, prioritized=False)


class D3QN(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig) -> None:
        super().__init__(state_dim, action_dim, config, dueling=True, prioritized=False)


class DDQNPER(BaseDQNAgent):
    def __init__(self, state_dim: int, action_dim: int, config: AgentConfig) -> None:
        super().__init__(state_dim, action_dim, config, dueling=False, prioritized=True)


class LunarLanderEnv:
    def __init__(
        self,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.gravity = float(gravity)
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
                continuous=False,
                gravity=self.gravity,
                enable_wind=self.enable_wind,
                wind_power=self.wind_power,
                turbulence_power=self.turbulence_power,
                render_mode=self.render_mode,
            )
            self.state_dim = int(self.env.observation_space.shape[0])
            self.action_dim = int(self.env.action_space.n)
            self.last_obs = None
            self.last_frame = None

    def update_config(self, gravity: float, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.gravity = float(gravity)
        self.enable_wind = bool(enable_wind)
        self.wind_power = float(wind_power)
        self.turbulence_power = float(turbulence_power)
        self._make_env()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        with self._env_lock:
            obs, _ = self.env.reset(seed=seed)
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return self.last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        with self._env_lock:
            obs, reward, terminated, truncated, info = self.env.step(int(action))
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
        self._csv_samples: List[Tuple[int, int, float, float, float, bool]] = []
        self._agent_classes = {
            "DuelingDQN": DuelingDQN,
            "D3QN": D3QN,
            "DDQN+PER": DDQNPER,
        }

    def set_policy_config(self, policy: str, **kwargs: object) -> None:
        config = self.policy_configs[policy]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def get_policy_config(self, policy: str) -> AgentConfig:
        return AgentConfig(**vars(self.policy_configs[policy]))

    def _get_or_create_agent(self, policy: str) -> BaseDQNAgent:
        if policy not in self.agents:
            config = self.get_policy_config(policy)
            cls = self._agent_classes[policy]
            self.agents[policy] = cls(self.env.state_dim, self.env.action_dim, config)
        return self.agents[policy]

    def rebuild_environment(self, gravity: float, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.env.update_config(gravity, enable_wind, wind_power, turbulence_power)
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
                    csv_rows.append((episode_idx, step_idx, sx, float(action), float(reward), bool(done)))
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
        out_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_policy = policy.replace("+", "plus")
        name = (
            f"{safe_policy}_eps{eps_max}_{eps_min}_lr{learning_rate}_"
            f"g{gamma}_ep{episodes}_ms{max_steps}_{ts}.png"
        )
        path = os.path.join(out_dir, name)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def close(self) -> None:
        self.env.close()
