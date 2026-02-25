import csv
import random
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.error import ResetNeeded


Transition = Tuple[np.ndarray, int, float, np.ndarray, bool]


class CartPoleEnvironment:
    """Environment wrapper for CartPole-v1.

    Sutton-Barto reward shaping in this project means:
    - reward = 0.0 for each non-terminal transition
    - reward = -1.0 when episode terminates/truncates

    Default behavior (`sutton_barto_reward=False`) keeps Gymnasium reward.
    """

    def __init__(self, sutton_barto_reward: bool = False, seed: int = 42) -> None:
        self.sutton_barto_reward = sutton_barto_reward
        self.seed = seed
        self._env = gym.make("CartPole-v1", render_mode="rgb_array")
        self._last_obs: Optional[np.ndarray] = None
        self.action_size = int(self._env.action_space.n)
        self.state_size = int(self._env.observation_space.shape[0])

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        chosen_seed = self.seed if seed is None else seed
        observation, _ = self._env.reset(seed=chosen_seed)
        self._last_obs = np.array(observation, dtype=np.float32)
        return self._last_obs.copy()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        observation, reward, terminated, truncated, info = self._env.step(action)
        observation = np.array(observation, dtype=np.float32)
        if self.sutton_barto_reward:
            reward = -1.0 if (terminated or truncated) else 0.0
        self._last_obs = observation
        return observation.copy(), float(reward), bool(terminated), bool(truncated), info

    def is_reachable(self, state: np.ndarray) -> bool:
        if state is None:
            return False
        if len(state) != self.state_size:
            return False
        bounds = [4.8, 5.0, 0.418, 5.0]
        return all(abs(float(value)) <= bound for value, bound in zip(state, bounds))

    def render_frame(self) -> np.ndarray:
        try:
            return self._env.render()
        except ResetNeeded:
            self.reset()
            return self._env.render()

    def close(self) -> None:
        self._env.close()


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50000
    min_replay_size: int = 1000
    target_update_every: int = 200
    hidden_layers: Tuple[int, ...] = (128,)


def _normalize_hidden_layers(hidden_layers: Union[int, Tuple[int, ...], List[int]]) -> Tuple[int, ...]:
    if isinstance(hidden_layers, int):
        return (max(1, int(hidden_layers)),)
    cleaned = tuple(max(1, int(value)) for value in hidden_layers if int(value) > 0)
    return cleaned if cleaned else (128,)


def _build_mlp(input_dim: int, hidden_layers: Tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: List[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_layers:
        layers.append(nn.Linear(current_dim, hidden_dim))
        layers.append(nn.ReLU())
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self.buffer)

    def add(self, transition: Transition) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, batch_size)


class DQNNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: Tuple[int, ...]) -> None:
        super().__init__()
        self.model = _build_mlp(state_size, hidden_layers, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DuelingNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_layers: Tuple[int, ...]) -> None:
        super().__init__()
        feature_layers = hidden_layers if len(hidden_layers) > 0 else (128,)
        feature_dim = feature_layers[-1]
        self.feature = _build_mlp(state_size, feature_layers[:-1], feature_dim)
        self.value_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
        )
        self.adv_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, action_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.feature(x)
        value = self.value_head(feat)
        advantage = self.adv_head(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class BaseDQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: Optional[AgentConfig] = None,
        use_dueling: bool = False,
        use_double: bool = False,
        device: Optional[str] = None,
    ) -> None:
        self.config = config or AgentConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.use_dueling = use_dueling
        self.use_double = use_double
        self.config.hidden_layers = _normalize_hidden_layers(self.config.hidden_layers)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        net_cls = DuelingNetwork if use_dueling else DQNNetwork
        self.online_net = net_cls(state_size, action_size, self.config.hidden_layers).to(self.device)
        self.target_net = net_cls(state_size, action_size, self.config.hidden_layers).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer(self.config.replay_size)
        self._learn_steps = 0

    def get_config_dict(self) -> Dict[str, float]:
        return {
            "gamma": self.config.gamma,
            "alpha": self.config.learning_rate,
            "batch_size": self.config.batch_size,
            "replay_size": self.config.replay_size,
            "target_update_every": self.config.target_update_every,
            "hidden_layers": self.config.hidden_layers,
        }

    def select_action(self, state: np.ndarray, epsilon: float = 0.1) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.online_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, transition: Transition) -> None:
        self.memory.add(transition)

    def learn(self) -> Optional[float]:
        if len(self.memory) < max(self.config.batch_size, self.config.min_replay_size):
            return None

        transitions = self.memory.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        current_q = self.online_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            if self.use_double:
                next_actions = self.online_net(next_states_t).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_states_t).gather(1, next_actions)
            else:
                next_q = self.target_net(next_states_t).max(dim=1, keepdim=True).values
            target_q = rewards_t + (1.0 - dones_t) * self.config.gamma * next_q

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self._learn_steps += 1
        if self._learn_steps % self.config.target_update_every == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return float(loss.item())


class DoubleDQN(BaseDQNAgent):
    def __init__(self, state_size: int, action_size: int, config: Optional[AgentConfig] = None) -> None:
        defaults = config or AgentConfig(
            gamma=0.99,
            learning_rate=1e-3,
            batch_size=64,
            replay_size=50000,
            min_replay_size=1000,
            target_update_every=200,
            hidden_layers=(128,),
        )
        super().__init__(state_size, action_size, defaults, use_dueling=False, use_double=True)


class DuelingDQN(BaseDQNAgent):
    def __init__(self, state_size: int, action_size: int, config: Optional[AgentConfig] = None) -> None:
        defaults = config or AgentConfig(
            gamma=0.99,
            learning_rate=7e-4,
            batch_size=64,
            replay_size=60000,
            min_replay_size=1000,
            target_update_every=250,
            hidden_layers=(128,),
        )
        super().__init__(state_size, action_size, defaults, use_dueling=True, use_double=False)


class D3QN(BaseDQNAgent):
    def __init__(self, state_size: int, action_size: int, config: Optional[AgentConfig] = None) -> None:
        defaults = config or AgentConfig(
            gamma=0.99,
            learning_rate=7e-4,
            batch_size=64,
            replay_size=70000,
            min_replay_size=1000,
            target_update_every=200,
            hidden_layers=(128,),
        )
        super().__init__(state_size, action_size, defaults, use_dueling=True, use_double=True)


class Trainer:
    def __init__(self, env: CartPoleEnvironment, output_dir: Optional[Path] = None) -> None:
        self.env = env
        self.output_dir = output_dir or Path(__file__).resolve().parent
        self.results_dir = self.output_dir / "results_csv"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_episode(
        self,
        policy: BaseDQNAgent,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
    ) -> Tuple[float, List[Transition]]:
        state = self.env.reset()
        total_reward = 0.0
        transitions: List[Transition] = []

        for step in range(1, max_steps + 1):
            action = policy.select_action(state, epsilon=epsilon)
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            transition = (state.copy(), action, reward, next_state.copy(), done)
            policy.remember(transition)
            policy.learn()
            transitions.append(transition)
            total_reward += reward
            state = next_state

            if progress_callback is not None:
                progress_callback(step)

            if done:
                break

        return total_reward, transitions

    def train(
        self,
        policy: BaseDQNAgent,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
    ) -> List[float]:
        rewards: List[float] = []
        all_transitions: List[Tuple[int, int, float, bool]] = []

        for ep in range(1, num_episodes + 1):
            ep_reward, transitions = self.run_episode(policy, epsilon=epsilon, max_steps=max_steps)
            rewards.append(ep_reward)
            for _, action, reward, _, done in transitions:
                all_transitions.append((ep, action, reward, done))

        if save_csv:
            base_name = f"{save_csv}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path = self.results_dir / base_name
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "action", "reward", "done"])
                writer.writerows(all_transitions)

        return rewards


def moving_average(values: List[float], window: int) -> List[float]:
    if window <= 1 or not values:
        return values.copy()
    out: List[float] = []
    running = deque(maxlen=window)
    for value in values:
        running.append(value)
        out.append(sum(running) / len(running))
    return out


def set_global_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
