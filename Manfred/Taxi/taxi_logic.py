"""
Taxi RL Workbench – Logic Layer
===============================
Environment wrapper/registry, algorithms, train loop, job scheduler,
event bus and checkpoint management.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor


# ---------------------------------------------------------------------------
# Environment wrapper + registry
# ---------------------------------------------------------------------------

ENV_ID = "Taxi-v3"


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = env.observation_space.n  # type: ignore[union-attr]
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec


class Environment:
    """Thin gymnasium wrapper with configurable env name."""

    def __init__(self, env_name: str = ENV_ID, **kwargs):
        self.env_name = env_name
        self.kwargs = kwargs
        self._env: Optional[gym.Env] = None
        self._build(render_mode=kwargs.pop("render_mode", None))

    def _build(self, render_mode: Optional[str] = None):
        build_kwargs = dict(self.kwargs)
        if render_mode is not None:
            build_kwargs["render_mode"] = render_mode
        try:
            env = gym.make(self.env_name, **build_kwargs)
        except TypeError:
            build_kwargs.pop("is_raining", None)
            build_kwargs.pop("fickle_passenger", None)
            env = gym.make(self.env_name, **build_kwargs)
        env = OneHotWrapper(env)
        env = Monitor(env)
        self._env = env

    def reset(self):
        return self._env.reset()  # type: ignore[union-attr]

    def step(self, action: int):
        return self._env.step(action)  # type: ignore[union-attr]

    def render(self):
        return self._env.render()  # type: ignore[union-attr]

    def close(self):
        if self._env is not None:
            self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space  # type: ignore[union-attr]

    @property
    def action_space(self):
        return self._env.action_space  # type: ignore[union-attr]

    @property
    def gym_env(self) -> gym.Env:
        return self._env  # type: ignore[return-value]


class EnvironmentRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., Environment]] = {}

    def register(self, name: str, factory: Callable[..., Environment]):
        self._registry[name] = factory

    def create(self, name: str, **kwargs) -> Environment:
        if name in self._registry:
            return self._registry[name](**kwargs)
        return Environment(env_name=name, **kwargs)


def make_env(
    env_id: str = ENV_ID,
    render_mode: Optional[str] = None,
    max_episode_steps: int = 500,
    is_raining: bool = False,
    fickle_passenger: bool = False,
) -> gym.Env:
    kwargs: Dict[str, Any] = {
        "is_raining": is_raining,
        "fickle_passenger": fickle_passenger,
        "max_episode_steps": max_episode_steps,
    }
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    try:
        env = gym.make(env_id, **kwargs)
    except TypeError:
        kwargs.pop("is_raining", None)
        kwargs.pop("fickle_passenger", None)
        env = gym.make(env_id, **kwargs)
    env = OneHotWrapper(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------


@dataclass
class TensorBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = max(1, int(capacity))
        self._data: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self._pos = 0

    def add(self, s, a, r, s2, done):
        transition = (np.asarray(s), int(a), float(r), np.asarray(s2), bool(done))
        if len(self._data) < self.capacity:
            self._data.append(transition)
        else:
            self._data[self._pos] = transition
        self._pos = (self._pos + 1) % self.capacity

    def sample(self, batch_size: int) -> TensorBatch:
        idx = np.random.randint(0, len(self._data), size=min(batch_size, len(self._data)))
        batch = [self._data[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return TensorBatch(
            states=np.stack(states).astype(np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_states=np.stack(next_states).astype(np.float32),
            dones=np.asarray(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self._data)


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class EventType(Enum):
    JOB_CREATED = "job_created"
    JOB_STATE_CHANGED = "job_state_changed"
    EPISODE_COMPLETED = "episode_completed"
    STEP_COMPLETED = "step_completed"
    TRAINING_DONE = "training_done"
    FRAME_RENDERED = "frame_rendered"
    ERROR = "error"
    RUN_STEP = "run_step"
    RUN_DONE = "run_done"


@dataclass
class Event:
    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    def __init__(self):
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: List[Callable[[Event], None]] = []

    def subscribe(self, listener: Callable[[Event], None]):
        self._listeners.append(listener)

    def publish(self, event: Event):
        self._queue.put(event)

    def process_events(self, max_events: int = 200):
        count = 0
        while count < max_events:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            for listener in self._listeners:
                try:
                    listener(event)
                except Exception:
                    pass
            count += 1


# ---------------------------------------------------------------------------
# Algorithm config + network helpers
# ---------------------------------------------------------------------------


ACTIVATION_MAP = {
    "ReLU": th.nn.ReLU,
    "Tanh": th.nn.Tanh,
    "LeakyReLU": th.nn.LeakyReLU,
    "ELU": th.nn.ELU,
    "GELU": th.nn.GELU,
}


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "ReLU"


@dataclass
class AlgorithmConfig:
    algorithm: str = "VDQN"
    learning_rate: float = 5e-4
    buffer_size: int = 5_000
    learning_starts: int = 64
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 1
    gradient_steps: int = 1
    target_update_interval: int = 200
    exploration_fraction: float = 0.8
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.02
    max_grad_norm: float = 10.0
    prioritized_alpha: float = 0.6
    prioritized_beta: float = 0.4
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig(hidden_layers=[128, 128], activation="ReLU"))

    episodes: int = 1000
    max_steps: int = 80
    moving_avg_window: int = 50

    env_name: str = ENV_ID
    is_raining: bool = False
    fickle_passenger: bool = False

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        dd = dict(d)
        network_dict = dd.pop("network", {}) or {}
        dd = {k: v for k, v in dd.items() if k in cls.__dataclass_fields__}
        obj = cls(**dd)
        obj.network = NetworkConfig(**{k: v for k, v in network_dict.items() if k in NetworkConfig.__dataclass_fields__})
        return obj


def build_mlp(network: NetworkConfig) -> Dict[str, Any]:
    activation_fn = ACTIVATION_MAP.get(network.activation, th.nn.ReLU)
    return {
        "net_arch": list(network.hidden_layers),
        "activation_fn": activation_fn,
    }


# ---------------------------------------------------------------------------
# Algorithm base + implementations
# ---------------------------------------------------------------------------


class AlgorithmBase(ABC):
    def __init__(self, config: AlgorithmConfig):
        self.config = config

    @abstractmethod
    def select_action(self, state, explore: bool = True) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: TensorBatch | None = None) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self):
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state_dict):
        raise NotImplementedError


class DoubleDQN(DQN):
    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions).squeeze(1)
                target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())
            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()
        self._workbench_last_loss = float(np.mean(losses)) if losses else 0.0
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", self._workbench_last_loss)


class VanillaDQN(DQN):
    """SB3 DQN with explicit last-loss tracking for UI status reporting."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            with th.no_grad():
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values, _ = next_q_values.max(dim=1)
                target_q_values = replay_data.rewards.flatten() + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long()).squeeze(1)

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._workbench_last_loss = float(np.mean(losses)) if losses else 0.0
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", self._workbench_last_loss)


class PrioritizedDQN(VanillaDQN):
    """Fallback implementation using SB3 DQN backend with configurable alpha/beta placeholders."""


class DuelingDQN(DoubleDQN):
    """Compatibility wrapper for Dueling option with SB3 backend limitations."""


class SB3DQNAlgorithm(AlgorithmBase):
    def __init__(self, config: AlgorithmConfig, env: gym.Env):
        super().__init__(config)
        self.env = env
        self.model: DQN = self._build_model()

    def _build_model(self) -> DQN:
        cfg = self.config
        policy_kwargs = build_mlp(cfg.network)
        cls = VanillaDQN
        if cfg.algorithm == "DDQN":
            cls = DoubleDQN
        elif cfg.algorithm == "Dueling DQN":
            cls = DuelingDQN
        elif cfg.algorithm == "Prioritized DQN":
            cls = PrioritizedDQN
        model = cls(
            "MlpPolicy",
            self.env,
            learning_rate=cfg.learning_rate,
            buffer_size=cfg.buffer_size,
            learning_starts=cfg.learning_starts,
            batch_size=cfg.batch_size,
            tau=cfg.tau,
            gamma=cfg.gamma,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_initial_eps=cfg.exploration_initial_eps,
            exploration_final_eps=cfg.exploration_final_eps,
            max_grad_norm=cfg.max_grad_norm,
            policy_kwargs=policy_kwargs,
            device="auto",
            verbose=0,
        )
        return model

    def select_action(self, state, explore: bool = True) -> int:
        action, _ = self.model.predict(state, deterministic=not explore)
        return int(action)

    def update(self, batch: TensorBatch | None = None) -> Dict[str, float]:
        try:
            self.model.train(gradient_steps=1, batch_size=self.config.batch_size)
        except Exception:
            return {"loss": 0.0}
        loss = getattr(self.model.logger, "name_to_value", {}).get("train/loss", 0.0)
        return {"loss": float(loss)}

    def get_state_dict(self):
        return self.model.get_parameters()

    def load_state_dict(self, state_dict):
        self.model.set_parameters(state_dict)


def build_model(config: AlgorithmConfig, env: gym.Env) -> DQN:
    return SB3DQNAlgorithm(config, env).model


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int
    duration: float
    loss: float = 0.0
    epsilon: float = 0.0


@dataclass
class StepResult:
    step: int
    reward: float
    done: bool


@dataclass
class TrainingResult:
    episodes_done: int
    returns: List[float]


class WorkbenchCallback(BaseCallback):
    def __init__(
        self,
        job: "TrainingJob",
        event_bus: EventBus,
        stop_event: threading.Event,
        pause_event: threading.Event,
        target_episodes: int,
        render_interval: float = 0.01,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.job = job
        self.event_bus = event_bus
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.target_episodes = target_episodes
        self.render_interval = render_interval
        self._episode_count = 0
        self._ep_start = time.time()
        self._last_render = 0.0
        self._last_loss: float = 0.0

    def _on_step(self) -> bool:
        if self.stop_event.is_set():
            return False
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.05)

        now = time.time()
        if self.job.visualization_enabled and (now - self._last_render) >= self.render_interval:
            try:
                envs = self.training_env.envs  # type: ignore[attr-defined]
                if envs:
                    frame = envs[0].render()
                    if frame is not None:
                        self.job.set_latest_frame(frame)
            except Exception:
                pass
            self._last_render = now

        if hasattr(self.model, "logger") and self.model.logger is not None:
            vals = getattr(self.model.logger, "name_to_value", {})
            if "train/loss" in vals:
                self._last_loss = vals["train/loss"]
        loss_attr = getattr(self.model, "_workbench_last_loss", None)
        if loss_attr is not None:
            self._last_loss = float(loss_attr)

        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is None:
                continue
            self._episode_count += 1
            ep_dur = time.time() - self._ep_start
            self._ep_start = time.time()
            eps = float(getattr(self.model, "exploration_rate", 0.0))
            result = EpisodeResult(
                episode=self.job.total_episodes_done + self._episode_count,
                total_reward=float(ep_info["r"]),
                steps=int(ep_info["l"]),
                duration=ep_dur,
                loss=self._last_loss,
                epsilon=eps,
            )
            self.event_bus.publish(Event(EventType.EPISODE_COMPLETED, {"job_id": self.job.job_id, "result": result}))
            if self._episode_count >= self.target_episodes:
                return False
        return True


# ---------------------------------------------------------------------------
# TrainLoop
# ---------------------------------------------------------------------------


class TrainLoop:
    def __init__(self, job: "TrainingJob", event_bus: EventBus):
        self.job = job
        self.event_bus = event_bus

    def run_step(self) -> Optional[StepResult]:
        if self.job.model is None or self.job._run_env is None:
            return None
        state = self.job._run_state
        if state is None:
            state, _ = self.job._run_env.reset()
            self.job._run_state = state
        action, _ = self.job.model.predict(state, deterministic=True)
        next_state, reward, terminated, truncated, _ = self.job._run_env.step(action)
        done = bool(terminated or truncated)
        self.job._run_state = None if done else next_state
        if self.job.visualization_enabled:
            try:
                frame = self.job._run_env.render()
                if frame is not None:
                    self.job.set_latest_frame(frame)
            except Exception:
                pass
        return StepResult(step=1, reward=float(reward), done=done)

    def run_episode(self) -> EpisodeResult:
        start = time.time()
        total_reward = 0.0
        steps = 0
        env = self.job._run_env
        assert env is not None and self.job.model is not None
        obs, _ = env.reset()
        while steps < self.job.config.max_steps and not self.job.stop_event.is_set():
            while self.job.pause_event.is_set() and not self.job.stop_event.is_set():
                time.sleep(0.05)
            action, _ = self.job.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
            if self.job.visualization_enabled:
                try:
                    frame = env.render()
                    if frame is not None:
                        self.job.set_latest_frame(frame)
                except Exception:
                    pass
            if terminated or truncated:
                break
        return EpisodeResult(
            episode=self.job.total_episodes_done + 1,
            total_reward=total_reward,
            steps=steps,
            duration=time.time() - start,
            loss=0.0,
            epsilon=0.0,
        )

    def run(self, n_episodes: int, callbacks: Optional[List[Callable[[EpisodeResult], None]]] = None) -> TrainingResult:
        callbacks = callbacks or []
        for _ in range(n_episodes):
            if self.job.stop_event.is_set():
                break
            result = self.run_episode()
            self.job.record_episode(result)
            for cb in callbacks:
                cb(result)
        return TrainingResult(episodes_done=n_episodes, returns=list(self.job.episode_returns))


# ---------------------------------------------------------------------------
# Training job + manager
# ---------------------------------------------------------------------------


class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RUN_MODE = "run_mode"


class TrainingJob:
    def __init__(self, config: AlgorithmConfig, name: Optional[str] = None):
        self.job_id = str(uuid.uuid4())[:8]
        self.config = config
        self.name = name or f"{config.algorithm}_{self.job_id}"
        self.status = JobStatus.PENDING

        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_durations: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_epsilons: List[float] = []

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self.visible = True
        self.visualization_enabled = True
        self.render_interval = 0.01
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        self.model: Optional[DQN] = None
        self._env: Optional[gym.Env] = None
        self._run_env: Optional[gym.Env] = None
        self._run_state = None
        self.train_loop: Optional[TrainLoop] = None

    @property
    def total_episodes_done(self) -> int:
        return len(self.episode_returns)

    @property
    def moving_avg(self) -> float:
        if not self.episode_returns:
            return 0.0
        w = max(1, self.config.moving_avg_window)
        return float(np.mean(self.episode_returns[-w:]))

    def set_latest_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

    def _ensure_model(self):
        if self.model is None:
            self._env = make_env(
                env_id=self.config.env_name,
                render_mode="rgb_array",
                max_episode_steps=self.config.max_steps,
                is_raining=self.config.is_raining,
                fickle_passenger=self.config.fickle_passenger,
            )
            self.model = build_model(self.config, self._env)
        elif self._env is None:
            self._env = make_env(
                env_id=self.config.env_name,
                render_mode="rgb_array",
                max_episode_steps=self.config.max_steps,
                is_raining=self.config.is_raining,
                fickle_passenger=self.config.fickle_passenger,
            )
            self.model.set_env(self._env)
        if self._run_env is None:
            self._run_env = make_env(
                env_id=self.config.env_name,
                render_mode="rgb_array",
                max_episode_steps=self.config.max_steps,
                is_raining=self.config.is_raining,
                fickle_passenger=self.config.fickle_passenger,
            )
        self.train_loop = TrainLoop(self, EventBus())

    def start_training(self, event_bus: EventBus, additional_episodes: Optional[int] = None):
        target_eps = additional_episodes or self.config.episodes
        self.stop_event.clear()
        self.pause_event.clear()
        self.status = JobStatus.RUNNING

        def _train():
            try:
                self._ensure_model()
                total_ts = target_eps * self.config.max_steps
                cb = WorkbenchCallback(
                    job=self,
                    event_bus=event_bus,
                    stop_event=self.stop_event,
                    pause_event=self.pause_event,
                    target_episodes=target_eps,
                    render_interval=self.render_interval,
                )
                self.model.learn(
                    total_timesteps=total_ts,
                    callback=cb,
                    reset_num_timesteps=(self.total_episodes_done == 0),
                    progress_bar=False,
                )
                self.status = JobStatus.CANCELLED if self.stop_event.is_set() else JobStatus.COMPLETED
            except Exception as exc:
                self.status = JobStatus.CANCELLED
                event_bus.publish(Event(EventType.ERROR, {"job_id": self.job_id, "error": str(exc)}))
            finally:
                event_bus.publish(Event(EventType.TRAINING_DONE, {"job_id": self.job_id}))
                event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

        self._thread = threading.Thread(target=_train, daemon=True)
        self._thread.start()
        event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

    def start_run(self, event_bus: EventBus):
        if self.model is None:
            return
        self.stop_event.clear()
        self.pause_event.clear()
        self.status = JobStatus.RUN_MODE

        def _run():
            try:
                self._ensure_model()
                total_reward = 0.0
                steps = 0
                result = self.train_loop.run_episode() if self.train_loop else EpisodeResult(0, 0.0, 0, 0.0)
                total_reward += result.total_reward
                steps += result.steps
                event_bus.publish(Event(EventType.RUN_DONE, {
                    "job_id": self.job_id,
                    "reward": total_reward,
                    "steps": steps,
                }))
            except Exception as exc:
                event_bus.publish(Event(EventType.ERROR, {"job_id": self.job_id, "error": str(exc)}))
            finally:
                self.status = JobStatus.COMPLETED if self.total_episodes_done > 0 else JobStatus.PENDING
                event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.set()
        self.status = JobStatus.PAUSED

    def resume(self):
        self.pause_event.clear()
        self.status = JobStatus.RUNNING

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def record_episode(self, result: EpisodeResult):
        self.episode_returns.append(result.total_reward)
        self.episode_lengths.append(result.steps)
        self.episode_durations.append(result.duration)
        self.episode_losses.append(result.loss)
        self.episode_epsilons.append(result.epsilon)

    def cleanup(self):
        self.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        for env in (self._env, self._run_env):
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        self._env = None
        self._run_env = None


class TrainingManager:
    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def add_job(self, config: AlgorithmConfig, name: Optional[str] = None) -> TrainingJob:
        job = TrainingJob(config, name)
        with self._lock:
            self.jobs[job.job_id] = job
        self.event_bus.publish(Event(EventType.JOB_CREATED, {"job_id": job.job_id}))
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def job_list(self) -> List[TrainingJob]:
        return list(self.jobs.values())

    def start_job(self, job_id: str, additional_episodes: Optional[int] = None):
        job = self.jobs.get(job_id)
        if job and job.status in (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.CANCELLED):
            job.start_training(self.event_bus, additional_episodes=additional_episodes or job.config.episodes)

    def start_all_pending(self):
        for job in self.job_list():
            if job.status in (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.CANCELLED):
                job.start_training(self.event_bus, additional_episodes=job.config.episodes)

    def pause(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.pause()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def resume(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.resume()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def cancel(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.is_alive():
            job.stop()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def cancel_all(self):
        for job in self.job_list():
            if job.is_alive():
                job.stop()

    def restart(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is None:
            return
        self.cancel(job_id)
        job.status = JobStatus.PENDING
        self.start_job(job_id)

    def remove(self, job_id: str):
        job = self.jobs.get(job_id)
        if job is None:
            return
        job.cleanup()
        with self._lock:
            self.jobs.pop(job_id, None)

    def run_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.model is not None and not job.is_alive():
            job.start_run(self.event_bus)

    def add_compare_jobs(self, base_config: AlgorithmConfig) -> List[TrainingJob]:
        jobs = []
        for algo in ("VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"):
            cfg = copy.deepcopy(base_config)
            cfg.algorithm = algo
            jobs.append(self.add_job(cfg, name=algo))
        return jobs

    def add_tuning_jobs(
        self,
        base_config: AlgorithmConfig,
        param_name: str,
        min_val: float,
        max_val: float,
        step_val: float,
    ) -> List[TrainingJob]:
        jobs: List[TrainingJob] = []
        if step_val <= 0:
            return jobs
        value = min_val
        while value <= max_val + 1e-9:
            cfg = copy.deepcopy(base_config)
            if param_name.startswith("network."):
                sub = param_name.split(".", 1)[1]
                if sub == "hidden_layers":
                    cfg.network.hidden_layers = [int(value)] * max(1, len(cfg.network.hidden_layers))
            elif hasattr(cfg, param_name):
                current = getattr(cfg, param_name)
                setattr(cfg, param_name, int(value) if isinstance(current, int) else float(value))
            name = f"{cfg.algorithm}_{param_name}={value:.4g}"
            jobs.append(self.add_job(cfg, name=name))
            value += step_val
        return jobs


# ---------------------------------------------------------------------------
# Checkpoint + metrics
# ---------------------------------------------------------------------------


class CheckpointManager:
    @staticmethod
    def save_job(job: TrainingJob, directory: str):
        os.makedirs(directory, exist_ok=True)
        data = {
            "job_id": job.job_id,
            "name": job.name,
            "config": job.config.to_dict(),
            "episode_returns": job.episode_returns,
            "episode_lengths": job.episode_lengths,
            "episode_durations": job.episode_durations,
            "episode_losses": job.episode_losses,
            "episode_epsilons": job.episode_epsilons,
            "visible": job.visible,
        }
        with open(os.path.join(directory, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        if job.model is not None:
            job.model.save(os.path.join(directory, "model"))

    @staticmethod
    def load_job(directory: str) -> TrainingJob:
        with open(os.path.join(directory, "metrics.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = AlgorithmConfig.from_dict(data["config"])
        job = TrainingJob(cfg, name=data.get("name"))
        job.job_id = data.get("job_id", job.job_id)
        job.episode_returns = data.get("episode_returns", [])
        job.episode_lengths = data.get("episode_lengths", [])
        job.episode_durations = data.get("episode_durations", [])
        job.episode_losses = data.get("episode_losses", [])
        job.episode_epsilons = data.get("episode_epsilons", [])
        job.visible = data.get("visible", True)
        job.status = JobStatus.COMPLETED if job.episode_returns else JobStatus.PENDING

        model_path = os.path.join(directory, "model.zip")
        if os.path.exists(model_path):
            env = make_env(
                env_id=cfg.env_name,
                render_mode="rgb_array",
                max_episode_steps=cfg.max_steps,
                is_raining=cfg.is_raining,
                fickle_passenger=cfg.fickle_passenger,
            )
            cls = DQN
            if cfg.algorithm == "DDQN":
                cls = DoubleDQN
            elif cfg.algorithm == "Prioritized DQN":
                cls = PrioritizedDQN
            job.model = cls.load(model_path, env=env)
            job._env = env
        return job

    @staticmethod
    def save_all(jobs: List[TrainingJob], directory: str):
        os.makedirs(directory, exist_ok=True)
        for job in jobs:
            sub = os.path.join(directory, job.job_id)
            CheckpointManager.save_job(job, sub)

    @staticmethod
    def load_all(directory: str) -> List[TrainingJob]:
        loaded: List[TrainingJob] = []
        if not os.path.isdir(directory):
            return loaded
        for entry in os.listdir(directory):
            sub = os.path.join(directory, entry)
            if os.path.isfile(os.path.join(sub, "metrics.json")):
                try:
                    loaded.append(CheckpointManager.load_job(sub))
                except Exception:
                    pass
        return loaded
