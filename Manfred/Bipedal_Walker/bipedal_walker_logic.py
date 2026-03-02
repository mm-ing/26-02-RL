"""
bipedal_walker_logic.py
Business logic for the BipedalWalker RL Workbench.
Uses Stable-Baselines3 for PPO, A2C, SAC, TD3.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv


# ---------------------------------------------------------------------------
# Enums & Types
# ---------------------------------------------------------------------------

class AlgorithmType(str, Enum):
    PPO = "PPO"
    A2C = "A2C"
    SAC = "SAC"
    TD3 = "TD3"


class LRSchedule(str, Enum):
    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"


class JobStatus(str, Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    PAUSED   = "paused"
    DONE     = "done"
    CANCELLED = "cancelled"
    FAILED   = "failed"


# ---------------------------------------------------------------------------
# Configuration Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class EnvironmentConfig:
    env_name: str = "BipedalWalker-v3"
    hardcore: bool = False
    render_interval_ms: int = 10


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"          # "relu", "tanh", "elu"


@dataclass
class EpisodeConfig:
    n_episodes: int = 3000
    max_steps: int = 1600
    alpha: float = 3e-4               # learning rate
    gamma: float = 0.99
    lr_schedule: str = LRSchedule.CONSTANT.value


@dataclass
class PPOConfig:
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gae_lambda: float = 0.95
    max_grad_norm: float = 0.5
    normalize_advantage: bool = True
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig([256, 256]))


@dataclass
class A2CConfig:
    n_steps: int = 5
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    gae_lambda: float = 1.0
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5
    use_rms_prop: bool = True
    normalize_advantage: bool = False
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig([64, 64]))


@dataclass
class SACConfig:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 100
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    ent_coef: str = "auto"
    target_update_interval: int = 1
    target_entropy: str = "auto"
    use_sde: bool = False
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig([256, 256]))


@dataclass
class TD3Config:
    buffer_size: int = 1_000_000
    batch_size: int = 256
    learning_starts: int = 100
    train_freq: int = 1
    gradient_steps: int = 1
    tau: float = 0.005
    policy_delay: int = 2
    target_policy_noise: float = 0.2
    target_noise_clip: float = 0.5
    action_noise_std: float = 0.1
    network: NetworkConfig = field(default_factory=lambda: NetworkConfig([256, 256]))


@dataclass
class TuningConfig:
    enabled: bool = False
    parameter: str = "alpha"          # which parameter to tune
    min_value: float = 1e-4
    max_value: float = 1e-3
    step: float = 1e-4


@dataclass
class JobConfig:
    job_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = "Job"
    algorithm: str = AlgorithmType.PPO.value
    env_cfg: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ep_cfg: EpisodeConfig = field(default_factory=EpisodeConfig)
    ppo_cfg: PPOConfig = field(default_factory=PPOConfig)
    a2c_cfg: A2CConfig = field(default_factory=A2CConfig)
    sac_cfg: SACConfig = field(default_factory=SACConfig)
    td3_cfg: TD3Config = field(default_factory=TD3Config)
    tuning_cfg: TuningConfig = field(default_factory=TuningConfig)
    visible: bool = True
    render_enabled: bool = True  # False in compare mode to avoid GIL contention


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    JOB_CREATED       = "job_created"
    JOB_STARTED       = "job_started"
    JOB_PAUSED        = "job_paused"
    JOB_RESUMED       = "job_resumed"
    JOB_CANCELLED     = "job_cancelled"
    JOB_DONE          = "job_done"
    JOB_FAILED        = "job_failed"
    JOB_REMOVED       = "job_removed"
    EPISODE_COMPLETED = "episode_completed"
    STEP_COMPLETED    = "step_completed"
    FRAME_RENDERED    = "frame_rendered"
    TRAINING_DONE     = "training_done"
    METRICS_UPDATED   = "metrics_updated"
    ERROR             = "error"


@dataclass
class Event:
    type: EventType
    job_id: str = ""
    data: Any = None


class EventBus:
    """Thread-safe event bus using queue for Tkinter-safe UI updates."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: Dict[EventType, List[Callable[[Event], None]]] = {}

    def subscribe(self, event_type: EventType, listener: Callable[[Event], None]) -> None:
        self._listeners.setdefault(event_type, []).append(listener)

    def unsubscribe(self, event_type: EventType, listener: Callable[[Event], None]) -> None:
        if event_type in self._listeners:
            self._listeners[event_type] = [l for l in self._listeners[event_type] if l is not listener]

    def publish(self, event: Event) -> None:
        """Called from any thread – places event in queue."""
        self._queue.put_nowait(event)

    def drain(self) -> None:
        """Called from UI thread every N ms to dispatch queued events."""
        try:
            while True:
                event = self._queue.get_nowait()
                for listener in self._listeners.get(event.type, []):
                    try:
                        listener(event)
                    except Exception as exc:
                        print(f"[EventBus] Listener error for {event.type}: {exc}")
        except queue.Empty:
            pass


# ---------------------------------------------------------------------------
# Episode Tracking Callback
# ---------------------------------------------------------------------------

class EpisodeCallback(BaseCallback):
    """SB3 callback that fires events after each episode.
    Uses VecMonitor infos (SB3 wraps envs with VecMonitor automatically),
    which provides reliable episode statistics via infos[i]['episode'].
    """

    def __init__(
        self,
        job: "TrainingJob",
        stop_event: threading.Event,
        pause_event: threading.Event,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._job = job
        self._stop_event = stop_event
        self._pause_event = pause_event
        self._ep_start_time: float = time.perf_counter()

    # SB3 calls this after each env step across all vec envs
    def _on_step(self) -> bool:
        if self._stop_event.is_set():
            return False  # stop training

        # handle pause
        while self._pause_event.is_set() and not self._stop_event.is_set():
            time.sleep(0.05)

        # NaN guard: if any observation is NaN the model has diverged; stop cleanly
        new_obs = self.locals.get("new_obs")
        if new_obs is not None and np.any(~np.isfinite(new_obs)):
            self._job.status = JobStatus.FAILED
            self._job.bus.publish(Event(
                type=EventType.JOB_FAILED,
                job_id=self._job.job_id,
                data="NaN/Inf in observations — policy diverged (try a lower learning rate)",
            ))
            return False

        # SB3 wraps the env with VecMonitor which adds "episode" info
        # when an episode ends: infos[i]["episode"] = {"r": ret, "l": len, "t": time}
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" not in info:
                continue

            ep_info = info["episode"]
            ep_return = float(ep_info.get("r", 0.0))
            ep_len    = int(ep_info.get("l", 0))
            duration  = time.perf_counter() - self._ep_start_time
            ep_num    = len(self._job.returns) + 1

            # Compute moving average (window 100)
            self._job.returns.append(ep_return)
            window = min(100, len(self._job.returns))
            ma = float(np.mean(self._job.returns[-window:]))
            self._job.moving_avg.append(ma)
            self._job.ep_steps.append(ep_len)
            self._job.ep_durations.append(duration)

            # loss from logger (best-effort)
            loss_val = None
            try:
                nv = self.logger.name_to_value
                for key in ("train/loss", "train/policy_loss", "train/actor_loss"):
                    if key in nv:
                        loss_val = float(nv[key])
                        break
            except Exception:
                pass

            self._job.bus.publish(Event(
                type=EventType.EPISODE_COMPLETED,
                job_id=self._job.job_id,
                data={
                    "episode":         ep_num,
                    "total_episodes":  self._job.config.ep_cfg.n_episodes,
                    "return":          ep_return,
                    "moving_avg":      ma,
                    "steps":           ep_len,
                    "duration":        duration,
                    "loss":            loss_val,
                    "epsilon":         None,
                },
            ))
            self._ep_start_time = time.perf_counter()
            # Push a weight snapshot so the render thread shows updated behaviour.
            # Done here (after optimizer, before next rollout) – safe window.
            if self._job.model is not None:
                self._job.render_mgr.queue_snapshot(self._job.model)

        return True


# ---------------------------------------------------------------------------
# Render Environment Manager  (dedicated render thread)
# ---------------------------------------------------------------------------

class RenderEnvManager:
    """
    Owns a separate env + dedicated render thread.
    The render thread steps the env using the current policy and stores the
    latest RGB frame.  The UI thread only reads `latest_frame` – it never
    calls PyTorch operations, avoiding all threading races with the training
    thread.
    """

    def __init__(self) -> None:
        self._env:          Optional[gym.Env] = None
        self._obs:          Optional[np.ndarray] = None
        self._model:        Any = None          # live training model (read-only ref)
        self._render_model: Any = None          # deep-copied model owned by render thread
        self._env_cfg:      Optional[EnvironmentConfig] = None

        # Frame storage: always keep only the newest frame
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock   = threading.Lock()

        # Snapshot queue: training thread writes, render thread reads at safe boundary
        self._sd_lock      = threading.Lock()
        self._pending_sd:  Optional[dict] = None   # {param_name: cloned tensor}

        # Thread control
        self._thread:       Optional[threading.Thread] = None
        self._stop_evt      = threading.Event()
        self._active        = False

    def queue_snapshot(self, model: Any) -> None:
        """Called from training thread between optimizer steps (safe window).
        Clones the policy weights into a pending dict. The render thread
        loads this at episode boundaries so it always has recent weights
        without ever touching the live training model.
        """
        try:
            sd = {k: v.detach().clone()
                  for k, v in model.policy.state_dict().items()}
            with self._sd_lock:
                self._pending_sd = sd
        except Exception:
            pass

    # ------------------------------------------------------------------

    def set_model(self, model: Any, env_cfg: EnvironmentConfig,
                  algo: str = AlgorithmType.PPO.value) -> None:
        """Call from training thread once model is built.
        Creates an isolated render_model via SB3 save/load so that every
        parameter tensor is a fresh allocation — completely independent of
        the live training model.  The render thread owns _render_model
        exclusively; the training thread never calls predict() on it.
        """
        import tempfile, os
        self._model   = model
        self._env_cfg = env_cfg
        try:
            with tempfile.TemporaryDirectory() as td:
                snap_path = os.path.join(td, "render_snap")
                model.save(snap_path)
                self._render_model = type(model).load(snap_path)
        except Exception:
            # Fallback: deepcopy.  Should not happen for standard SB3 models.
            import copy
            self._render_model = copy.deepcopy(model)
        self._start_thread()

    def _start_thread(self) -> None:
        self._stop_evt.clear()
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._render_loop,
            daemon=True,
            name="render-env",
        )
        self._thread.start()

    def _render_loop(self) -> None:
        """Background thread: step env, store frame, repeat."""
        while not self._stop_evt.is_set():
            try:
                if self._model is None or self._env_cfg is None:
                    time.sleep(0.05)
                    continue

                # Lazy env creation
                if self._env is None:
                    self._env = gym.make(
                        self._env_cfg.env_name,
                        render_mode="rgb_array",
                        hardcore=self._env_cfg.hardcore,
                        max_episode_steps=1600,
                    )
                    self._obs, _ = self._env.reset()
                    term = trunc = False

                # Predict using the render-thread-owned copy of the model.
                # Training thread never touches _render_model, so this is
                # fully thread-safe for all algorithms including SAC/TD3.
                import torch
                with torch.no_grad():
                    action_batch, _ = self._render_model.predict(
                        self._obs[None], deterministic=True
                    )
                action = action_batch[0]

                self._obs, _, term, trunc, _ = self._env.step(action)

                # At episode boundaries, load any pending weight snapshot into
                # the render model.  This is the only place render_model weights
                # change and the training thread never touches render_model.
                if term or trunc:
                    with self._sd_lock:
                        sd = self._pending_sd
                        self._pending_sd = None
                    if sd is not None:
                        try:
                            self._render_model.policy.load_state_dict(sd)
                        except Exception:
                            pass
                    self._obs, _ = self._env.reset()

                frame = self._env.render()
                if frame is not None:
                    with self._frame_lock:
                        self._latest_frame = frame

            except Exception as exc:
                print(f"[RenderEnvManager] {exc}")
                # Reset env on error
                try:
                    if self._env is not None:
                        self._env.close()
                except Exception:
                    pass
                self._env = None
                self._obs = None
                time.sleep(0.1)

            # Limit to ~30 fps so BipedalWalker physics look fluid
            time.sleep(0.033)

    @property
    def latest_frame(self) -> Optional[np.ndarray]:
        """UI thread reads this – no PyTorch involved."""
        with self._frame_lock:
            return self._latest_frame

    def reset_env(self) -> None:
        self._stop_evt.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
        self._env  = None
        self._obs  = None
        with self._sd_lock:
            self._pending_sd = None
        with self._frame_lock:
            self._latest_frame = None

    def close(self) -> None:
        self.reset_env()
        self._model = None
        self._render_model = None



# ---------------------------------------------------------------------------
# Training Job
# ---------------------------------------------------------------------------

class TrainingJob:
    """Represents a single training run."""

    def __init__(self, config: JobConfig, bus: EventBus) -> None:
        self.job_id   = config.job_id
        self.name     = config.name
        self.config   = config
        self.bus      = bus
        self.status   = JobStatus.PENDING

        # Metrics
        self.returns:       List[float] = []
        self.moving_avg:    List[float] = []
        self.ep_steps:      List[int]   = []
        self.ep_durations:  List[float] = []

        # SB3 model
        self.model: Any = None

        # Thread control
        self._thread:      Optional[threading.Thread] = None
        self._stop_event   = threading.Event()
        self._pause_event  = threading.Event()

        # Render env
        self.render_mgr = RenderEnvManager()

    # ------------------------------------------------------------------
    # Build SB3 model
    # ------------------------------------------------------------------

    def _make_env(self, render_mode: Optional[str] = None) -> gym.Env:
        cfg    = self.config.env_cfg
        ep_cfg = self.config.ep_cfg
        kwargs: Dict[str, Any] = dict(hardcore=cfg.hardcore)
        if render_mode:
            kwargs["render_mode"] = render_mode
        return gym.make(cfg.env_name, max_episode_steps=ep_cfg.max_steps, **kwargs)

    def _build_model(self) -> Any:
        algo = self.config.algorithm
        ep   = self.config.ep_cfg
        dev  = "cuda" if torch.cuda.is_available() else "cpu"

        def make_env():
            return Monitor(self._make_env())

        vec_env = DummyVecEnv([make_env])

        def lr_schedule(progress: float) -> float:
            if ep.lr_schedule == LRSchedule.LINEAR.value:
                return ep.alpha * progress
            elif ep.lr_schedule == LRSchedule.COSINE.value:
                import math
                return ep.alpha * (1 + math.cos(math.pi * (1 - progress))) / 2
            return ep.alpha

        if algo == AlgorithmType.PPO.value:
            cfg = self.config.ppo_cfg
            net = cfg.network
            policy_kwargs = dict(
                net_arch=dict(pi=list(net.hidden_layers), vf=list(net.hidden_layers)),
                activation_fn=_get_activation(net.activation),
            )
            model = PPO(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_schedule,
                n_steps=cfg.n_steps,
                batch_size=cfg.batch_size,
                n_epochs=cfg.n_epochs,
                gamma=ep.gamma,
                gae_lambda=cfg.gae_lambda,
                clip_range=cfg.clip_range,
                ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                normalize_advantage=cfg.normalize_advantage,
                policy_kwargs=policy_kwargs,
                device=dev,
                verbose=0,
            )

        elif algo == AlgorithmType.A2C.value:
            cfg = self.config.a2c_cfg
            net = cfg.network
            policy_kwargs = dict(
                net_arch=dict(pi=list(net.hidden_layers), vf=list(net.hidden_layers)),
                activation_fn=_get_activation(net.activation),
            )
            model = A2C(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_schedule,
                n_steps=cfg.n_steps,
                gamma=ep.gamma,
                gae_lambda=cfg.gae_lambda,
                ent_coef=cfg.ent_coef,
                vf_coef=cfg.vf_coef,
                max_grad_norm=cfg.max_grad_norm,
                rms_prop_eps=cfg.rms_prop_eps,
                use_rms_prop=cfg.use_rms_prop,
                normalize_advantage=cfg.normalize_advantage,
                policy_kwargs=policy_kwargs,
                device=dev,
                verbose=0,
            )

        elif algo == AlgorithmType.SAC.value:
            cfg = self.config.sac_cfg
            net = cfg.network
            policy_kwargs = dict(
                net_arch=list(net.hidden_layers),
                activation_fn=_get_activation(net.activation),
            )
            model = SAC(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_schedule,
                buffer_size=cfg.buffer_size,
                batch_size=cfg.batch_size,
                learning_starts=cfg.learning_starts,
                train_freq=cfg.train_freq,
                gradient_steps=cfg.gradient_steps,
                gamma=ep.gamma,
                tau=cfg.tau,
                ent_coef=cfg.ent_coef,
                target_update_interval=cfg.target_update_interval,
                target_entropy=cfg.target_entropy,
                use_sde=cfg.use_sde,
                policy_kwargs=policy_kwargs,
                device=dev,
                verbose=0,
            )

        elif algo == AlgorithmType.TD3.value:
            cfg = self.config.td3_cfg
            net = cfg.network
            action_dim = vec_env.action_space.shape[0]
            action_noise = NormalActionNoise(
                mean=np.zeros(action_dim),
                sigma=cfg.action_noise_std * np.ones(action_dim),
            )
            policy_kwargs = dict(
                net_arch=list(net.hidden_layers),
                activation_fn=_get_activation(net.activation),
            )
            model = TD3(
                "MlpPolicy",
                vec_env,
                learning_rate=lr_schedule,
                buffer_size=cfg.buffer_size,
                batch_size=cfg.batch_size,
                learning_starts=cfg.learning_starts,
                train_freq=cfg.train_freq,
                gradient_steps=cfg.gradient_steps,
                gamma=ep.gamma,
                tau=cfg.tau,
                policy_delay=cfg.policy_delay,
                target_policy_noise=cfg.target_policy_noise,
                target_noise_clip=cfg.target_noise_clip,
                action_noise=action_noise,
                policy_kwargs=policy_kwargs,
                device=dev,
                verbose=0,
            )
        else:
            raise ValueError(f"Unknown algorithm: {algo}")

        return model

    # ------------------------------------------------------------------
    # Training Thread
    # ------------------------------------------------------------------

    def _train_thread(self, additional_episodes: int) -> None:
        try:
            # Build model here (background thread) so any construction
            # exception is caught and reported as JOB_FAILED, not silently
            # swallowed by Tkinter when called from the UI thread.
            if self.model is None:
                self.model = self._build_model()

            self.status = JobStatus.RUNNING
            self.bus.publish(Event(EventType.JOB_STARTED, self.job_id))

            ep_cfg = self.config.ep_cfg
            total_ts = additional_episodes * ep_cfg.max_steps

            cb = EpisodeCallback(self, self._stop_event, self._pause_event)

            # Start the render thread (disabled in compare mode to avoid
            # GIL contention that would starve SAC/TD3 training threads).
            if self.model is not None and self.config.render_enabled:
                self.render_mgr.set_model(
                    self.model,
                    self.config.env_cfg,
                    algo=self.config.algorithm,
                )

            self.model.learn(
                total_timesteps=total_ts,
                callback=cb,
                reset_num_timesteps=False,
                progress_bar=False,
            )

            if self._stop_event.is_set():
                self.status = JobStatus.CANCELLED
                self.bus.publish(Event(EventType.JOB_CANCELLED, self.job_id))
            else:
                self.status = JobStatus.DONE
                self.bus.publish(Event(EventType.JOB_DONE, self.job_id))
                self.bus.publish(Event(EventType.TRAINING_DONE, self.job_id))

        except Exception as exc:
            # Translate NaN distribution errors into a friendly message
            msg = str(exc)
            if "constraint Real()" in msg or "nan" in msg.lower() or "inf" in msg.lower():
                msg = "Policy network diverged (NaN/Inf) — try reducing the learning rate"
            elif "sizes of tensors must match" in msg.lower():
                msg = ("Tensor shape mismatch during training — "
                       "this can be caused by concurrent model access. "
                       f"Original: {exc}")
            # If NaN guard already marked this job as FAILED, skip re-publishing
            if self.status == JobStatus.FAILED:
                return
            self.status = JobStatus.FAILED
            self.bus.publish(Event(EventType.JOB_FAILED, self.job_id, msg))
            print(f"[TrainingJob {self.job_id}] Error: {exc}")
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start or continue training."""
        self._stop_event.clear()
        self._pause_event.clear()

        episodes_done = len(self.returns)
        target = self.config.ep_cfg.n_episodes
        remaining = max(1, target - episodes_done)

        self._thread = threading.Thread(
            target=self._train_thread,
            args=(remaining,),
            daemon=True,
            name=f"train-{self.job_id}",
        )
        self._thread.start()

    def pause(self) -> None:
        if self.status == JobStatus.RUNNING:
            self._pause_event.set()
            self.status = JobStatus.PAUSED
            self.bus.publish(Event(EventType.JOB_PAUSED, self.job_id))

    def resume(self) -> None:
        if self.status == JobStatus.PAUSED:
            self._pause_event.clear()
            self.status = JobStatus.RUNNING
            self.bus.publish(Event(EventType.JOB_RESUMED, self.job_id))

    def cancel(self) -> None:
        self._stop_event.set()
        self._pause_event.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self.status = JobStatus.CANCELLED
        self.bus.publish(Event(EventType.JOB_CANCELLED, self.job_id))

    def run_validation(self, n_episodes: int = 3) -> None:
        """Run current model in render env without training."""
        if self.model is None:
            return

        def _run():
            try:
                env = gym.make(
                    self.config.env_cfg.env_name,
                    render_mode="rgb_array",
                    hardcore=self.config.env_cfg.hardcore,
                    max_episode_steps=1600,
                )
                self.render_mgr.set_model(self.model)
                for _ in range(n_episodes):
                    obs, _ = env.reset()
                    done = False
                    while not done and not self._stop_event.is_set():
                        action, _ = self.model.predict(obs, deterministic=True)
                        obs, _, terminated, truncated, _ = env.step(action)
                        done = terminated or truncated
                        frame = env.render()
                        if frame is not None:
                            self.bus.publish(Event(EventType.FRAME_RENDERED, self.job_id, frame))
                        time.sleep(0.01)
                env.close()
            except Exception as exc:
                print(f"[RunValidation] {exc}")
                import traceback; traceback.print_exc()

        self._stop_event.clear()
        self.status = JobStatus.RUNNING
        self._thread = threading.Thread(target=_run, daemon=True, name=f"run-{self.job_id}")
        self._thread.start()

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def serialize(self) -> Dict[str, Any]:
        return {
            "job_id":       self.job_id,
            "name":         self.name,
            "config":       _serialize_config(self.config),
            "returns":      self.returns,
            "moving_avg":   self.moving_avg,
            "ep_steps":     self.ep_steps,
            "ep_durations": self.ep_durations,
        }


# ---------------------------------------------------------------------------
# Training Manager
# ---------------------------------------------------------------------------

class TrainingManager:
    """Manages multiple TrainingJobs. UI-agnostic."""

    def __init__(self, bus: EventBus) -> None:
        self.bus  = bus
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    @property
    def jobs(self) -> Dict[str, TrainingJob]:
        return self._jobs

    def add_job(self, config: JobConfig) -> TrainingJob:
        job = TrainingJob(config, self.bus)
        with self._lock:
            self._jobs[job.job_id] = job
        self.bus.publish(Event(EventType.JOB_CREATED, job.job_id, config))
        return job

    def start_job(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.start()

    def start_all_pending(self) -> None:
        for job in list(self._jobs.values()):
            if job.status in (JobStatus.PENDING, JobStatus.DONE, JobStatus.CANCELLED):
                job.start()

    def pause(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.pause()

    def resume(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.resume()

    def cancel(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            job.cancel()

    def cancel_all(self) -> None:
        for job in list(self._jobs.values()):
            if job.status == JobStatus.RUNNING:
                job.cancel()

    def remove(self, job_id: str) -> None:
        job = self._jobs.get(job_id)
        if job:
            if job.is_alive():
                job.cancel()
            job.render_mgr.close()
            with self._lock:
                del self._jobs[job_id]
            self.bus.publish(Event(EventType.JOB_REMOVED, job_id))

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    def get_active_job(self) -> Optional[TrainingJob]:
        """Return first running job, or None."""
        for job in self._jobs.values():
            if job.status == JobStatus.RUNNING:
                return job
        return None

    def build_tuning_jobs(self, base_config: JobConfig, tuning: TuningConfig) -> List[JobConfig]:
        """Create multiple job configs for parameter tuning."""
        values = np.arange(tuning.min_value, tuning.max_value + tuning.step * 0.5, tuning.step)
        configs: List[JobConfig] = []

        for val in values:
            import copy
            cfg = copy.deepcopy(base_config)
            cfg.job_id = str(uuid.uuid4())[:8]
            param = tuning.parameter
            _set_param(cfg, param, float(val))
            cfg.name = f"{base_config.algorithm} {param}={val:.4g}"
            configs.append(cfg)

        return configs


# ---------------------------------------------------------------------------
# Checkpoint Manager
# ---------------------------------------------------------------------------

class CheckpointManager:
    """Save/load TrainingJobs to/from disk."""

    @staticmethod
    def save(job: TrainingJob, save_dir: str) -> None:
        path = Path(save_dir) / job.job_id
        path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics = {
            "returns":      job.returns,
            "moving_avg":   job.moving_avg,
            "ep_steps":     job.ep_steps,
            "ep_durations": job.ep_durations,
        }
        with open(path / "metrics.json", "w") as f:
            json.dump(metrics, f)

        # Save config
        with open(path / "config.json", "w") as f:
            json.dump(job.serialize()["config"], f, indent=2, default=str)

        # Save model weights
        if job.model is not None:
            job.model.save(str(path / "model"))

    @staticmethod
    def load(load_dir: str, bus: EventBus) -> Optional[TrainingJob]:
        path = Path(load_dir)
        config_file = path / "config.json"
        metrics_file = path / "metrics.json"

        if not config_file.exists():
            return None

        with open(config_file) as f:
            raw = json.load(f)

        config = _deserialize_config(raw)
        config.job_id = path.name  # keep original job_id from folder name

        job = TrainingJob(config, bus)

        if metrics_file.exists():
            with open(metrics_file) as f:
                metrics = json.load(f)
            job.returns      = metrics.get("returns", [])
            job.moving_avg   = metrics.get("moving_avg", [])
            job.ep_steps     = metrics.get("ep_steps", [])
            job.ep_durations = metrics.get("ep_durations", [])

        model_file = path / "model.zip"
        if model_file.exists():
            algo = config.algorithm
            dev  = "cuda" if torch.cuda.is_available() else "cpu"

            def make_env():
                return Monitor(gym.make(
                    config.env_cfg.env_name,
                    hardcore=config.env_cfg.hardcore,
                    max_episode_steps=config.ep_cfg.max_steps,
                ))

            vec_env = DummyVecEnv([make_env])
            cls = {
                AlgorithmType.PPO.value: PPO,
                AlgorithmType.A2C.value: A2C,
                AlgorithmType.SAC.value: SAC,
                AlgorithmType.TD3.value: TD3,
            }[algo]
            job.model = cls.load(str(path / "model"), env=vec_env, device=dev)
            # Note: render thread is started when training resumes via _train_thread
            job.status = JobStatus.DONE

        return job


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_activation(name: str):
    import torch.nn as nn
    return {
        "relu":  nn.ReLU,
        "tanh":  nn.Tanh,
        "elu":   nn.ELU,
        "leaky": nn.LeakyReLU,
    }.get(name.lower(), nn.ReLU)


def _set_param(cfg: JobConfig, param: str, value: float) -> None:
    """Set a named parameter across config hierarchy."""
    param_lower = param.lower()
    if param_lower == "alpha":
        cfg.ep_cfg.alpha = value
    elif param_lower == "gamma":
        cfg.ep_cfg.gamma = value
    elif param_lower == "buffer_size":
        cfg.sac_cfg.buffer_size = int(value)
        cfg.td3_cfg.buffer_size = int(value)
    elif param_lower == "batch_size":
        cfg.ppo_cfg.batch_size = int(value)
        cfg.a2c_cfg.n_steps = int(value)
        cfg.sac_cfg.batch_size = int(value)
        cfg.td3_cfg.batch_size = int(value)
    elif param_lower == "clip_range":
        cfg.ppo_cfg.clip_range = value
    elif param_lower == "ent_coef":
        cfg.ppo_cfg.ent_coef = value
        cfg.a2c_cfg.ent_coef = value
    elif param_lower == "tau":
        cfg.sac_cfg.tau = value
        cfg.td3_cfg.tau = value
    elif param_lower == "gae_lambda":
        cfg.ppo_cfg.gae_lambda = value
        cfg.a2c_cfg.gae_lambda = value


def _serialize_config(cfg: JobConfig) -> Dict[str, Any]:
    return {
        "job_id":    cfg.job_id,
        "name":      cfg.name,
        "algorithm": cfg.algorithm,
        "env_cfg":   asdict(cfg.env_cfg),
        "ep_cfg":    asdict(cfg.ep_cfg),
        "ppo_cfg":   {**asdict(cfg.ppo_cfg), "network": asdict(cfg.ppo_cfg.network)},
        "a2c_cfg":   {**asdict(cfg.a2c_cfg), "network": asdict(cfg.a2c_cfg.network)},
        "sac_cfg":   {**asdict(cfg.sac_cfg), "network": asdict(cfg.sac_cfg.network)},
        "td3_cfg":   {**asdict(cfg.td3_cfg), "network": asdict(cfg.td3_cfg.network)},
        "tuning_cfg": asdict(cfg.tuning_cfg),
    }


def _deserialize_config(raw: Dict[str, Any]) -> JobConfig:
    def nc(d):
        return NetworkConfig(**{k: v for k, v in d.items()})

    ppo_raw  = raw.get("ppo_cfg",  {})
    a2c_raw  = raw.get("a2c_cfg",  {})
    sac_raw  = raw.get("sac_cfg",  {})
    td3_raw  = raw.get("td3_cfg",  {})

    ppo_net  = NetworkConfig(**ppo_raw.pop("network", {})) if "network" in ppo_raw else NetworkConfig()
    a2c_net  = NetworkConfig(**a2c_raw.pop("network", {})) if "network" in a2c_raw else NetworkConfig()
    sac_net  = NetworkConfig(**sac_raw.pop("network", {})) if "network" in sac_raw else NetworkConfig()
    td3_net  = NetworkConfig(**td3_raw.pop("network", {})) if "network" in td3_raw else NetworkConfig()

    return JobConfig(
        job_id    = raw.get("job_id", str(uuid.uuid4())[:8]),
        name      = raw.get("name", "Job"),
        algorithm = raw.get("algorithm", AlgorithmType.PPO.value),
        env_cfg   = EnvironmentConfig(**raw.get("env_cfg", {})),
        ep_cfg    = EpisodeConfig(**raw.get("ep_cfg", {})),
        ppo_cfg   = PPOConfig(**{**ppo_raw, "network": ppo_net}),
        a2c_cfg   = A2CConfig(**{**a2c_raw, "network": a2c_net}),
        sac_cfg   = SACConfig(**{**sac_raw, "network": sac_net}),
        td3_cfg   = TD3Config(**{**td3_raw, "network": td3_net}),
        tuning_cfg = TuningConfig(**raw.get("tuning_cfg", {})),
    )


# Global EventBus singleton
_global_bus = EventBus()


def get_bus() -> EventBus:
    return _global_bus
