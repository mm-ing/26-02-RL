"""
CliffWalking RL Workbench – Logic Layer
=======================================
Environment wrapper, algorithm configs, training jobs,
training manager, event bus, checkpoint manager.
Uses Stable-Baselines3 for VDQN / DDQN.
"""

import copy
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
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
# Environment helpers
# ---------------------------------------------------------------------------

ENV_ID = "CliffWalking-v1"


class OneHotWrapper(gym.ObservationWrapper):
    """Convert Discrete observation to one-hot float32 vector."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        n = env.observation_space.n  # type: ignore[union-attr]
        self.observation_space = gym.spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        vec = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        vec[int(obs)] = 1.0
        return vec


def make_env(env_id: str = ENV_ID, render_mode: Optional[str] = None,
             max_episode_steps: int = 200) -> gym.Env:
    """Create wrapped CliffWalking env (OneHot + Monitor)."""
    env = gym.make(env_id, render_mode=render_mode,
                   max_episode_steps=max_episode_steps)
    env = OneHotWrapper(env)
    env = Monitor(env)
    return env


# ---------------------------------------------------------------------------
# Event system
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
    """Thread-safe event bus using a queue."""

    def __init__(self):
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: List[Callable[[Event], None]] = []

    def subscribe(self, listener: Callable[[Event], None]):
        self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[Event], None]):
        self._listeners = [l for l in self._listeners if l is not listener]

    def publish(self, event: Event):
        self._queue.put(event)

    def process_events(self, max_events: int = 200) -> List[Event]:
        """Drain queue and dispatch to listeners. Call from UI thread."""
        processed = []
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
            processed.append(event)
            count += 1
        return processed


# ---------------------------------------------------------------------------
# Algorithm configuration
# ---------------------------------------------------------------------------

ACTIVATION_MAP = {
    "ReLU": th.nn.ReLU,
    "Tanh": th.nn.Tanh,
    "LeakyReLU": th.nn.LeakyReLU,
    "ELU": th.nn.ELU,
    "GELU": th.nn.GELU,
}


@dataclass
class AlgorithmConfig:
    algorithm: str = "VDQN"  # "VDQN" or "DDQN"
    learning_rate: float = 1e-3
    buffer_size: int = 50_000
    learning_starts: int = 500
    batch_size: int = 64
    tau: float = 1.0
    gamma: float = 0.99
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    hidden_layers: List[int] = field(default_factory=lambda: [128, 128])
    activation: str = "ReLU"
    max_grad_norm: float = 10.0

    # Training schedule
    episodes: int = 300
    max_steps: int = 200

    # Moving average window for plotting
    moving_avg_window: int = 20

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AlgorithmConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Double DQN (subclass of SB3 DQN)
# ---------------------------------------------------------------------------

class DoubleDQN(DQN):
    """Double DQN: action selection with online net, evaluation with target net."""

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                # Double DQN: select best actions with ONLINE net
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                # Evaluate chosen actions with TARGET net
                next_q_target = self.q_net_target(replay_data.next_observations)
                next_q_values = th.gather(next_q_target, dim=1, index=next_actions).squeeze(1)
                target_q_values = (
                    replay_data.rewards.flatten()
                    + (1 - replay_data.dones.flatten()) * self.gamma * next_q_values
                )

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(
                current_q_values, dim=1, index=replay_data.actions.long()
            ).squeeze(1)

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None:
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", float(np.mean(losses)))


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(config: AlgorithmConfig, env: gym.Env) -> DQN:
    """Create SB3 DQN / DoubleDQN from config."""
    activation_fn = ACTIVATION_MAP.get(config.activation, th.nn.ReLU)
    policy_kwargs = dict(
        net_arch=list(config.hidden_layers),
        activation_fn=activation_fn,
    )
    cls = DoubleDQN if config.algorithm == "DDQN" else DQN
    model = cls(
        "MlpPolicy",
        env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        learning_starts=config.learning_starts,
        batch_size=config.batch_size,
        tau=config.tau,
        gamma=config.gamma,
        train_freq=config.train_freq,
        gradient_steps=config.gradient_steps,
        target_update_interval=config.target_update_interval,
        exploration_fraction=config.exploration_fraction,
        exploration_initial_eps=config.exploration_initial_eps,
        exploration_final_eps=config.exploration_final_eps,
        max_grad_norm=config.max_grad_norm,
        policy_kwargs=policy_kwargs,
        device="auto",
        verbose=0,
    )
    return model


# ---------------------------------------------------------------------------
# Episode / Step results
# ---------------------------------------------------------------------------

@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int
    duration: float
    loss: float = 0.0
    epsilon: float = 0.0


# ---------------------------------------------------------------------------
# SB3 training callback
# ---------------------------------------------------------------------------

class WorkbenchCallback(BaseCallback):
    """Integrates SB3 training loop with the workbench event bus."""

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
        # ---- stop / pause ----
        if self.stop_event.is_set():
            return False
        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.05)

        # ---- frame capture (rate-limited) ----
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

        # ---- track loss from logger ----
        if hasattr(self.model, "logger") and self.model.logger is not None:
            name_to_val = getattr(self.model.logger, "name_to_value", {})
            if "train/loss" in name_to_val:
                self._last_loss = name_to_val["train/loss"]

        # ---- episode tracking via Monitor info ----
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                self._episode_count += 1
                ep_dur = time.time() - self._ep_start
                self._ep_start = time.time()

                # Current exploration rate
                eps = 0.0
                if hasattr(self.model, "exploration_rate"):
                    eps = self.model.exploration_rate

                result = EpisodeResult(
                    episode=self.job.total_episodes_done + self._episode_count,
                    total_reward=float(ep_info["r"]),
                    steps=int(ep_info["l"]),
                    duration=ep_dur,
                    loss=self._last_loss,
                    epsilon=eps,
                )
                self.event_bus.publish(Event(EventType.EPISODE_COMPLETED, {
                    "job_id": self.job.job_id,
                    "result": result,
                }))

                if self._episode_count >= self.target_episodes:
                    return False
        return True


# ---------------------------------------------------------------------------
# Job status
# ---------------------------------------------------------------------------

class JobStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    RUN_MODE = "run_mode"  # inference only


# ---------------------------------------------------------------------------
# Training Job
# ---------------------------------------------------------------------------

class TrainingJob:
    """Represents a single training run."""

    def __init__(self, config: AlgorithmConfig, name: Optional[str] = None):
        self.job_id: str = str(uuid.uuid4())[:8]
        self.config = config
        self.name = name or f"{config.algorithm}_{self.job_id}"
        self.status = JobStatus.PENDING

        # Results
        self.episode_returns: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_durations: List[float] = []
        self.episode_losses: List[float] = []
        self.episode_epsilons: List[float] = []

        # Threading
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

        # Visualization
        self.visualization_enabled: bool = True
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None

        # UI visibility (show in plot)
        self.visible: bool = True

        # SB3 model (created at training start)
        self.model: Optional[DQN] = None
        self._env: Optional[gym.Env] = None

    @property
    def total_episodes_done(self) -> int:
        return len(self.episode_returns)

    @property
    def moving_avg(self) -> float:
        w = self.config.moving_avg_window
        if not self.episode_returns:
            return 0.0
        window = self.episode_returns[-w:]
        return float(np.mean(window))

    def set_latest_frame(self, frame: np.ndarray):
        with self._frame_lock:
            self._latest_frame = frame

    def get_latest_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._latest_frame

    def _ensure_model(self):
        """Create env + model if not yet created."""
        if self.model is None:
            self._env = make_env(ENV_ID, render_mode="rgb_array",
                                max_episode_steps=self.config.max_steps)
            self.model = build_model(self.config, self._env)
        elif self._env is None or not hasattr(self._env, "step"):
            # Recreate env for continued training
            self._env = make_env(ENV_ID, render_mode="rgb_array",
                                max_episode_steps=self.config.max_steps)
            self.model.set_env(self._env)

    def start_training(self, event_bus: EventBus, additional_episodes: Optional[int] = None):
        """Start or continue training in a background thread."""
        target_eps = additional_episodes or self.config.episodes
        self._stop_event.clear()
        self._pause_event.clear()
        self.status = JobStatus.RUNNING

        def _train():
            try:
                self._ensure_model()
                total_ts = target_eps * self.config.max_steps
                reset_timesteps = self.total_episodes_done == 0

                render_interval = getattr(self, "render_interval", 0.01)
                cb = WorkbenchCallback(
                    job=self,
                    event_bus=event_bus,
                    stop_event=self._stop_event,
                    pause_event=self._pause_event,
                    target_episodes=target_eps,
                    render_interval=render_interval,
                )
                self.model.learn(
                    total_timesteps=total_ts,
                    callback=cb,
                    reset_num_timesteps=reset_timesteps,
                    progress_bar=False,
                )
                # Append collected episode results
                # Results are tracked via events; this is just status update
                if not self._stop_event.is_set():
                    self.status = JobStatus.COMPLETED
                else:
                    self.status = JobStatus.CANCELLED
            except Exception as e:
                event_bus.publish(Event(EventType.ERROR, {"job_id": self.job_id, "error": str(e)}))
                self.status = JobStatus.CANCELLED
            finally:
                event_bus.publish(Event(EventType.TRAINING_DONE, {"job_id": self.job_id}))
                event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

        self._thread = threading.Thread(target=_train, daemon=True)
        self._thread.start()
        event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

    def start_run(self, event_bus: EventBus):
        """Run inference (no training) in background thread."""
        if self.model is None:
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self.status = JobStatus.RUN_MODE

        def _run():
            try:
                env = make_env(ENV_ID, render_mode="rgb_array",
                              max_episode_steps=self.config.max_steps)
                obs, _ = env.reset()
                done = False
                total_reward = 0.0
                steps = 0
                while not done and not self._stop_event.is_set():
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    total_reward += float(reward)
                    steps += 1
                    # Render frame
                    try:
                        frame = env.render()
                        if frame is not None:
                            self.set_latest_frame(frame)
                    except Exception:
                        pass
                    event_bus.publish(Event(EventType.RUN_STEP, {
                        "job_id": self.job_id,
                        "reward": total_reward,
                        "steps": steps,
                    }))
                    time.sleep(0.05)  # slow down for visualisation
                env.close()
                event_bus.publish(Event(EventType.RUN_DONE, {
                    "job_id": self.job_id,
                    "reward": total_reward,
                    "steps": steps,
                }))
            except Exception as e:
                event_bus.publish(Event(EventType.ERROR, {"job_id": self.job_id, "error": str(e)}))
            finally:
                self.status = JobStatus.COMPLETED if self.total_episodes_done > 0 else JobStatus.PENDING
                event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": self.job_id}))

    def stop(self):
        self._stop_event.set()

    def pause(self):
        self._pause_event.set()
        self.status = JobStatus.PAUSED

    def resume(self):
        self._pause_event.clear()
        self.status = JobStatus.RUNNING

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def record_episode(self, result: EpisodeResult):
        """Called from UI thread after receiving EPISODE_COMPLETED event."""
        self.episode_returns.append(result.total_reward)
        self.episode_lengths.append(result.steps)
        self.episode_durations.append(result.duration)
        self.episode_losses.append(result.loss)
        self.episode_epsilons.append(result.epsilon)

    def cleanup(self):
        """Stop training and release resources."""
        self.stop()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None


# ---------------------------------------------------------------------------
# Training Manager
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages multiple TrainingJobs. UI-agnostic."""

    def __init__(self, event_bus: EventBus):
        self.event_bus = event_bus
        self.jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def add_job(self, config: AlgorithmConfig, name: Optional[str] = None) -> TrainingJob:
        job = TrainingJob(config, name=name)
        with self._lock:
            self.jobs[job.job_id] = job
        self.event_bus.publish(Event(EventType.JOB_CREATED, {"job_id": job.job_id}))
        return job

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.jobs.get(job_id)

    def start_job(self, job_id: str, additional_episodes: Optional[int] = None):
        job = self.jobs.get(job_id)
        if job is None:
            return
        # For continuing training, calculate remaining episodes
        if job.status in (JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.PENDING):
            eps = additional_episodes or job.config.episodes
            job.start_training(self.event_bus, additional_episodes=eps)

    def start_all_pending(self):
        with self._lock:
            pending = [j for j in self.jobs.values()
                       if j.status in (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.CANCELLED)]
        for job in pending:
            job.start_training(self.event_bus, additional_episodes=job.config.episodes)

    def pause_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.RUNNING:
            job.pause()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def resume_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.status == JobStatus.PAUSED:
            job.resume()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def cancel_job(self, job_id: str):
        job = self.jobs.get(job_id)
        if job and job.is_alive():
            job.stop()
            self.event_bus.publish(Event(EventType.JOB_STATE_CHANGED, {"job_id": job_id}))

    def cancel_all(self):
        for job in list(self.jobs.values()):
            if job.is_alive():
                job.stop()

    def remove_job(self, job_id: str):
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

    def any_running(self) -> bool:
        return any(j.is_alive() for j in self.jobs.values())

    def job_list(self) -> List[TrainingJob]:
        return list(self.jobs.values())

    # --- Compare mode ---
    def add_compare_jobs(self, base_config: AlgorithmConfig) -> List[TrainingJob]:
        """Add one job per algorithm (VDQN, DDQN) with same hyperparams."""
        jobs = []
        for algo in ("VDQN", "DDQN"):
            cfg = copy.deepcopy(base_config)
            cfg.algorithm = algo
            job = self.add_job(cfg, name=algo)
            jobs.append(job)
        return jobs

    # --- Tuning mode ---
    def add_tuning_jobs(
        self,
        base_config: AlgorithmConfig,
        param_name: str,
        min_val: float,
        max_val: float,
        step_val: float,
    ) -> List[TrainingJob]:
        """Create variants with different values for *param_name*."""
        jobs = []
        val = min_val
        while val <= max_val + 1e-9:
            cfg = copy.deepcopy(base_config)
            if param_name == "hidden_layers":
                n = int(val)
                cfg.hidden_layers = [n] * len(cfg.hidden_layers)
            elif hasattr(cfg, param_name):
                field_type = type(getattr(cfg, param_name))
                if field_type == int:
                    setattr(cfg, param_name, int(val))
                else:
                    setattr(cfg, param_name, float(val))
            name = f"{cfg.algorithm}_{param_name}={val:.4g}"
            job = self.add_job(cfg, name=name)
            jobs.append(job)
            val += step_val
            if step_val <= 0:
                break
        return jobs


# ---------------------------------------------------------------------------
# Checkpoint manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save / load training jobs to disk."""

    @staticmethod
    def save_job(job: TrainingJob, directory: str):
        os.makedirs(directory, exist_ok=True)
        # Save metadata
        meta = {
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
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        # Save SB3 model
        if job.model is not None:
            model_path = os.path.join(directory, "model")
            job.model.save(model_path)

    @staticmethod
    def load_job(directory: str) -> TrainingJob:
        meta_path = os.path.join(directory, "meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        config = AlgorithmConfig.from_dict(meta["config"])
        job = TrainingJob(config, name=meta.get("name"))
        job.job_id = meta.get("job_id", job.job_id)
        job.episode_returns = meta.get("episode_returns", [])
        job.episode_lengths = meta.get("episode_lengths", [])
        job.episode_durations = meta.get("episode_durations", [])
        job.episode_losses = meta.get("episode_losses", [])
        job.episode_epsilons = meta.get("episode_epsilons", [])
        job.visible = meta.get("visible", True)
        job.status = JobStatus.COMPLETED if job.episode_returns else JobStatus.PENDING
        # Load SB3 model
        model_path = os.path.join(directory, "model.zip")
        if os.path.exists(model_path):
            env = make_env(ENV_ID, render_mode="rgb_array",
                          max_episode_steps=config.max_steps)
            cls = DoubleDQN if config.algorithm == "DDQN" else DQN
            job.model = cls.load(model_path, env=env)
            job._env = env
        return job

    @staticmethod
    def save_all(jobs: List[TrainingJob], directory: str):
        os.makedirs(directory, exist_ok=True)
        for job in jobs:
            job_dir = os.path.join(directory, job.job_id)
            CheckpointManager.save_job(job, job_dir)

    @staticmethod
    def load_all(directory: str) -> List[TrainingJob]:
        jobs = []
        if not os.path.isdir(directory):
            return jobs
        for entry in os.listdir(directory):
            sub = os.path.join(directory, entry)
            if os.path.isfile(os.path.join(sub, "meta.json")):
                try:
                    jobs.append(CheckpointManager.load_job(sub))
                except Exception:
                    pass
        return jobs
