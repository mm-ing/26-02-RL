from __future__ import annotations

import csv
import math
import random
import threading
from collections import deque, namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - handled at runtime
    gym = None

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class PolicyDefaults:
    gamma: float = 0.99
    learning_rate: float = 8e-4
    replay_size: int = 50_000
    batch_size: int = 64
    target_update: int = 100
    replay_warmup: int = 1_000
    train_every: int = 2
    activation_function: str = "ReLU"
    hidden_layer_size: int | Sequence[int] = 128


@dataclass
class PERDefaults(PolicyDefaults):
    alpha: float = 0.6
    beta_start: float = 0.4
    beta_frames: int = 50_000
    eps_prio: float = 1e-5


@dataclass
class RunSnapshot:
    policy_name: str
    params: Dict[str, float]
    rewards: List[float] = field(default_factory=list)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.buffer: Deque[Transition] = deque(maxlen=self.capacity)

    def push(self, *args) -> None:
        self.buffer.append(Transition(*args))

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, eps_prio: float = 1e-5) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.eps_prio = float(eps_prio)
        self.buffer: List[Transition] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def push(self, *args) -> None:
        max_prio = self.priorities.max() if self.buffer else 1.0
        transition = Transition(*args)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, beta: float) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        if len(self.buffer) == self.capacity:
            probs = self.priorities
        else:
            probs = self.priorities[: len(self.buffer)]

        probs = np.power(probs + self.eps_prio, self.alpha)
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = np.power(total * probs[indices], -beta)
        weights /= weights.max()
        return samples, indices, weights.astype(np.float32)

    def update_priorities(self, indices: Sequence[int], priorities: Sequence[float]) -> None:
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = float(abs(prio) + self.eps_prio)

    def __len__(self) -> int:
        return len(self.buffer)


class MLPQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int | Sequence[int], activation_function: str = "ReLU") -> None:
        super().__init__()
        hidden_layers = _normalize_hidden_layers(hidden_size)
        activation_name = _normalize_activation_name(activation_function)
        layers: List[nn.Module] = []
        in_features = state_dim
        for units in hidden_layers:
            layers.append(nn.Linear(in_features, units))
            layers.append(_make_activation(activation_name))
            in_features = units
        layers.append(nn.Linear(in_features, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int | Sequence[int], activation_function: str = "ReLU") -> None:
        super().__init__()
        hidden_layers = _normalize_hidden_layers(hidden_size)
        activation_name = _normalize_activation_name(activation_function)
        shared_layers: List[nn.Module] = []
        in_features = state_dim
        for units in hidden_layers:
            shared_layers.append(nn.Linear(in_features, units))
            shared_layers.append(_make_activation(activation_name))
            in_features = units
        self.shared = nn.Sequential(*shared_layers)

        stream_hidden = max(64, in_features // 2)
        self.value = nn.Sequential(
            nn.Linear(in_features, stream_hidden),
            _make_activation(activation_name),
            nn.Linear(stream_hidden, 1),
        )
        self.advantage = nn.Sequential(
            nn.Linear(in_features, stream_hidden),
            _make_activation(activation_name),
            nn.Linear(stream_hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.shared(x)
        values = self.value(features)
        advantages = self.advantage(features)
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class BaseDQNAgent:
    policy_name = "BaseDQN"

    def __init__(self, state_dim: int, action_dim: int, params: PolicyDefaults) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.params = params
        self.device = _device()
        self.steps_done = 0

        self.policy_net: nn.Module
        self.target_net: nn.Module

    def act(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def soft_or_hard_update(self) -> None:
        if self.steps_done % self.params.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


def _normalize_hidden_layers(hidden_layer_size: int | Sequence[int]) -> List[int]:
    if isinstance(hidden_layer_size, int):
        layers = [hidden_layer_size]
    else:
        layers = [int(v) for v in hidden_layer_size]
    layers = [max(1, int(v)) for v in layers]
    return layers or [128]


def _normalize_activation_name(name: str) -> str:
    text = str(name).strip().lower()
    mapping = {
        "relu": "relu",
        "tanh": "tanh",
        "leakyrelu": "leakyrelu",
        "leaky_relu": "leakyrelu",
        "elu": "elu",
    }
    return mapping.get(text, "relu")


def _make_activation(name: str) -> nn.Module:
    normalized = _normalize_activation_name(name)
    if normalized == "tanh":
        return nn.Tanh()
    if normalized == "leakyrelu":
        return nn.LeakyReLU(negative_slope=0.01)
    if normalized == "elu":
        return nn.ELU()
    return nn.ReLU()


class DuelingDQN(BaseDQNAgent):
    policy_name = "Dueling DQN"

    def __init__(self, state_dim: int, action_dim: int, params: PolicyDefaults) -> None:
        super().__init__(state_dim, action_dim, params)
        self.policy_net = DuelingQNetwork(state_dim, action_dim, params.hidden_layer_size, params.activation_function).to(self.device)
        self.target_net = DuelingQNetwork(state_dim, action_dim, params.hidden_layer_size, params.activation_function).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.learning_rate)
        self.buffer = ReplayBuffer(params.replay_size)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.params.batch_size:
            return None

        transitions = self.buffer.sample(self.params.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
            target = rewards + self.params.gamma * (1.0 - dones) * max_next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        self.soft_or_hard_update()
        return float(loss.item())


class D3QN(DuelingDQN):
    policy_name = "D3QN"

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.params.batch_size:
            return None

        transitions = self.buffer.sample(self.params.batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.params.gamma * (1.0 - dones) * next_q

        loss = nn.functional.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.steps_done += 1
        self.soft_or_hard_update()
        return float(loss.item())


class DDQN_PER(BaseDQNAgent):
    policy_name = "DDQN+PER"

    def __init__(self, state_dim: int, action_dim: int, params: PERDefaults) -> None:
        super().__init__(state_dim, action_dim, params)
        self.params = params
        self.policy_net = MLPQNetwork(state_dim, action_dim, params.hidden_layer_size, params.activation_function).to(self.device)
        self.target_net = MLPQNetwork(state_dim, action_dim, params.hidden_layer_size, params.activation_function).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=params.learning_rate)
        self.buffer = PrioritizedReplayBuffer(params.replay_size, alpha=params.alpha, eps_prio=params.eps_prio)
        self.frame_idx = 1

    def _beta(self) -> float:
        t = min(1.0, self.frame_idx / max(1, self.params.beta_frames))
        return self.params.beta_start + t * (1.0 - self.params.beta_start)

    def learn(self) -> Optional[float]:
        if len(self.buffer) < self.params.batch_size:
            return None

        beta = self._beta()
        transitions, indices, weights = self.buffer.sample(self.params.batch_size, beta=beta)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)
        is_weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.params.gamma * (1.0 - dones) * next_q

        td_errors = target - q_values
        loss = (is_weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.buffer.update_priorities(indices, np.abs(td_errors.detach().cpu().numpy().flatten()))

        self.steps_done += 1
        self.frame_idx += 1
        self.soft_or_hard_update()
        return float(loss.item())


class MountainCarEnvironment:
    def __init__(self, goal_velocity: float = 0.0, x_init: float = math.pi, y_init: float = 1.0, seed: Optional[int] = None):
        if gym is None:
            raise ImportError("gymnasium is required. Please install dependencies from requirements.txt")

        self.goal_velocity = float(goal_velocity)
        self.x_init = float(x_init)
        self.y_init = float(y_init)
        self.seed = seed
        self.env = gym.make("MountainCar-v0", render_mode="rgb_array", goal_velocity=self.goal_velocity)
        self._env_lock = threading.RLock()
        self.state: Optional[np.ndarray] = None

    @property
    def state_dim(self) -> int:
        return int(self.env.observation_space.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.env.action_space.n)

    def _reset_options(self) -> Dict[str, float]:
        options = {"x_init": self.x_init, "y_init": self.y_init}
        return options

    def reset(self) -> np.ndarray:
        with self._env_lock:
            try:
                state, _ = self.env.reset(seed=self.seed, options=self._reset_options())
            except TypeError:
                state, _ = self.env.reset(seed=self.seed)
            except Exception:
                state, _ = self.env.reset(seed=self.seed)
            self.state = np.asarray(state, dtype=np.float32)
            return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        with self._env_lock:
            next_state, reward, terminated, truncated, info = self.env.step(int(action))
            done = bool(terminated or truncated)
            self.state = np.asarray(next_state, dtype=np.float32)
            return self.state, float(reward), done, info

    def render_frame(self) -> Optional[np.ndarray]:
        with self._env_lock:
            if self.state is None:
                self.reset()
            frame = self.env.render()
            if frame is None:
                return None
            return np.asarray(frame)

    def is_reachable(self, x: float, y: float) -> bool:
        min_pos, max_pos = -1.2, 0.6
        min_vel, max_vel = -0.07, 0.07
        return (min_pos <= x <= max_pos) and (min_vel <= y <= max_vel)

    def close(self) -> None:
        with self._env_lock:
            self.env.close()


class Trainer:
    def __init__(self, environment: MountainCarEnvironment, output_dir: Optional[Path] = None) -> None:
        self.environment = environment
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent
        self.results_csv_dir = self.output_dir / "results_csv"
        self.results_csv_dir.mkdir(parents=True, exist_ok=True)

        self.policy_defaults: Dict[str, Dict[str, float]] = {
            "Dueling DQN": {
                "gamma": 0.99,
                "learning_rate": 1e-3,
                "replay_size": 50_000,
                "batch_size": 64,
                "target_update": 100,
                "replay_warmup": 1_000,
                "train_every": 2,
                "activation_function": "ReLU",
                "hidden_layer_size": [128, 128],
            },
            "D3QN": {
                "gamma": 0.99,
                "learning_rate": 8e-4,
                "replay_size": 50_000,
                "batch_size": 64,
                "target_update": 100,
                "replay_warmup": 1_000,
                "train_every": 2,
                "activation_function": "ReLU",
                "hidden_layer_size": [128, 128],
            },
            "DDQN+PER": {
                "gamma": 0.99,
                "learning_rate": 5e-4,
                "replay_size": 75_000,
                "batch_size": 64,
                "target_update": 100,
                "replay_warmup": 1_000,
                "train_every": 2,
                "activation_function": "ReLU",
                "hidden_layer_size": [128, 128],
                "alpha": 0.6,
                "beta_start": 0.4,
                "beta_frames": 80_000,
                "eps_prio": 1e-5,
            },
        }
        self._agents: Dict[str, BaseDQNAgent] = {}

    def create_agent(self, policy: str, overrides: Optional[Dict[str, float]] = None) -> BaseDQNAgent:
        merged = dict(self.policy_defaults[policy])
        if overrides:
            merged.update(overrides)
        self.policy_defaults[policy] = dict(merged)

        state_dim = self.environment.state_dim
        action_dim = self.environment.action_dim

        if policy == "Dueling DQN":
            agent = DuelingDQN(state_dim, action_dim, PolicyDefaults(**{k: merged[k] for k in PolicyDefaults.__annotations__.keys()}))
        elif policy == "D3QN":
            agent = D3QN(state_dim, action_dim, PolicyDefaults(**{k: merged[k] for k in PolicyDefaults.__annotations__.keys()}))
        elif policy == "DDQN+PER":
            agent = DDQN_PER(state_dim, action_dim, PERDefaults(**{k: merged[k] for k in PERDefaults.__annotations__.keys()}))
        else:
            raise ValueError(f"Unknown policy: {policy}")

        self._agents[policy] = agent
        return agent

    def get_or_create_agent(self, policy: str, overrides: Optional[Dict[str, float]] = None) -> BaseDQNAgent:
        if policy not in self._agents:
            return self.create_agent(policy, overrides)
        if overrides:
            return self.create_agent(policy, overrides)
        return self._agents[policy]

    def run_episode(
        self,
        policy: str,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Dict[str, object]:
        agent = self.get_or_create_agent(policy)
        state = self.environment.reset()
        total_reward = 0.0
        transitions: List[Transition] = []
        last_loss: Optional[float] = None
        max_position = float(state[0])
        reached_goal = False

        for step_idx in range(int(max_steps)):
            action = agent.act(state, float(epsilon))
            next_state, reward, done, _ = self.environment.step(action)

            if done and float(next_state[0]) >= 0.5:
                reached_goal = True
            max_position = max(max_position, float(next_state[0]))

            if hasattr(agent, "buffer"):
                agent.buffer.push(state, action, reward, next_state, done)

            loss = None
            if hasattr(agent, "learn"):
                replay_warmup = max(0, int(getattr(agent.params, "replay_warmup", 0)))
                train_every = max(1, int(getattr(agent.params, "train_every", 1)))
                buffer_ready = True
                if hasattr(agent, "buffer"):
                    capacity = max(1, int(getattr(agent.buffer, "capacity", replay_warmup if replay_warmup > 0 else 1)))
                    min_batch = max(1, int(getattr(agent.params, "batch_size", 1)))
                    effective_warmup = max(min_batch, min(replay_warmup, max(min_batch, capacity // 2)))
                    buffer_ready = len(agent.buffer) >= effective_warmup
                if buffer_ready and ((step_idx + 1) % train_every) == 0:
                    loss = agent.learn()
                    if loss is not None:
                        last_loss = float(loss)
            transitions.append(Transition(state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            if progress_callback:
                progress_callback(step_idx + 1)
            if done:
                break

        return {
            "total_reward": float(total_reward),
            "steps": len(transitions),
            "transitions": transitions,
            "last_loss": last_loss,
            "max_position": float(max_position),
            "reached_goal": bool(reached_goal),
        }

    def train(
        self,
        policy: str,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[float]:
        rewards: List[float] = []
        all_transitions: List[Transition] = []

        for episode in range(int(num_episodes)):
            result = self.run_episode(policy, epsilon=epsilon, max_steps=max_steps)
            rewards.append(float(result["total_reward"]))
            all_transitions.extend(result["transitions"])
            if progress_callback:
                progress_callback(episode + 1, int(result["steps"]))

        if save_csv:
            file_name = f"{save_csv}.csv" if not save_csv.endswith(".csv") else save_csv
            target = self.results_csv_dir / file_name
            self._write_transitions_csv(target, all_transitions)

        return rewards

    def _write_transitions_csv(self, file_path: Path, transitions: List[Transition]) -> None:
        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["s0", "s1", "action", "reward", "ns0", "ns1", "done"])
            for t in transitions:
                writer.writerow(
                    [
                        float(t.state[0]),
                        float(t.state[1]),
                        int(t.action),
                        float(t.reward),
                        float(t.next_state[0]),
                        float(t.next_state[1]),
                        int(t.done),
                    ]
                )


def make_default_trainer(
    goal_velocity: float = 0.0,
    x_init: float = math.pi,
    y_init: float = 1.0,
    seed: Optional[int] = None,
    output_dir: Optional[Path] = None,
) -> Trainer:
    environment = MountainCarEnvironment(goal_velocity=goal_velocity, x_init=x_init, y_init=y_init, seed=seed)
    return Trainer(environment=environment, output_dir=output_dir)
