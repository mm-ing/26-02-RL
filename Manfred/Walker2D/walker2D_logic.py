"""
Logic layer for the Walker2D RL Workbench.
SB3-based algorithm wrappers, job management, event bus, checkpoint I/O.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Events
# ─────────────────────────────────────────────────────────────────────────────

class EventType(Enum):
    JOB_CREATED       = auto()
    JOB_STARTED       = auto()
    JOB_PAUSED        = auto()
    JOB_RESUMED       = auto()
    JOB_CANCELLED     = auto()
    JOB_COMPLETED     = auto()
    JOB_REMOVED       = auto()
    JOB_ERROR         = auto()
    EPISODE_COMPLETED = auto()
    STEP_COMPLETED    = auto()
    FRAME_RENDERED    = auto()
    TRAINING_DONE     = auto()


@dataclass
class Event:
    type:   EventType
    job_id: str
    data:   Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Thread-safe event bus backed by a Queue for Tkinter polling."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: List[Callable[[Event], None]] = []

    def subscribe(self, cb: Callable[[Event], None]) -> None:
        self._listeners.append(cb)

    def publish(self, event: Event) -> None:
        self._queue.put_nowait(event)

    def drain(self) -> None:
        while True:
            try:
                ev = self._queue.get_nowait()
            except queue.Empty:
                break
            for cb in self._listeners:
                try:
                    cb(ev)
                except Exception:
                    import traceback
                    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Config dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    forward_reward_weight:                      float = 1.0
    ctrl_cost_weight:                           float = 0.001
    healthy_reward:                             float = 1.0
    terminate_when_unhealthy:                   bool  = True
    healthy_z_min:                              float = 0.8
    healthy_z_max:                              float = 2.0
    healthy_angle_min:                          float = -1.0
    healthy_angle_max:                          float = 1.0
    reset_noise_scale:                          float = 0.005
    exclude_current_positions_from_observation: bool  = True
    render_interval_ms:                         int   = 10
    visualize:                                  bool  = True


@dataclass
class EpisodeConfig:
    n_episodes:      int   = 3000
    max_steps:       int   = 1000
    alpha:           float = 3e-4
    gamma:           float = 0.99
    compare_methods: bool  = False


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation:    str       = "relu"


@dataclass
class PPOConfig:
    algo_name:     str           = "PPO"
    n_steps:       int           = 2048
    batch_size:    int           = 64
    n_epochs:      int           = 10
    clip_range:    float         = 0.2
    ent_coef:      float         = 0.0
    vf_coef:       float         = 0.5
    max_grad_norm: float         = 0.5
    gae_lambda:    float         = 0.95
    network:       NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class SACConfig:
    algo_name:              str           = "SAC"
    buffer_size:            int           = 300_000
    batch_size:             int           = 256
    learning_starts:        int           = 10_000
    train_freq:             int           = 1
    gradient_steps:         int           = 1
    tau:                    float         = 0.005
    ent_coef:               str           = "auto"
    target_update_interval: int           = 1
    network:                NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class TD3Config:
    algo_name:          str           = "TD3"
    buffer_size:        int           = 300_000
    batch_size:         int           = 256
    learning_starts:    int           = 10_000
    train_freq:         int           = 1
    gradient_steps:     int           = 1
    tau:                float         = 0.005
    policy_delay:       int           = 2
    target_noise:       float         = 0.2
    noise_clip:         float         = 0.5
    action_noise_sigma: float         = 0.1
    network:            NetworkConfig = field(default_factory=NetworkConfig)


ALGO_CONFIGS: Dict[str, Any] = {
    "PPO": PPOConfig,
    "SAC": SACConfig,
    "TD3": TD3Config,
}

ENV_NAME = "Walker2d-v5"


# ─────────────────────────────────────────────────────────────────────────────
# SB3 Callback
# ─────────────────────────────────────────────────────────────────────────────

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class WorkbenchCallback(BaseCallback):
        def __init__(
            self,
            job_id:         str,
            bus:            EventBus,
            n_episodes:     int,
            stop_event:     threading.Event,
            pause_event:    threading.Event,
            frame_queue:    queue.Queue,
            visualize_flag: Callable[[], bool],
            verbose:        int = 0,
        ) -> None:
            super().__init__(verbose)
            self.job_id         = job_id
            self.bus            = bus
            self.n_episodes     = n_episodes
            self.stop_event     = stop_event
            self.pause_event    = pause_event
            self.frame_queue    = frame_queue
            self.visualize_flag = visualize_flag
            self._ep_count      = 0
            self._ep_start      = time.time()
            self._returns: List[float] = []
            self._last_frame_t  = 0.0

        def _on_step(self) -> bool:
            # pause / stop
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)
            if self.stop_event.is_set():
                return False

            # render (throttled ~100 Hz)
            now = time.time()
            if self.visualize_flag() and now - self._last_frame_t >= 0.01:
                try:
                    frame = self.training_env.envs[0].render()
                    if frame is not None:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_queue.put_nowait(frame)
                        self._last_frame_t = now
                except Exception:
                    pass

            # episode-end detection
            dones = self.locals.get("dones")
            infos = self.locals.get("infos", [])
            if dones is not None:
                for done, info in zip(dones, infos):
                    if done:
                        ep_info = info.get("episode", {})
                        ret   = float(ep_info.get("r", 0.0))
                        steps = int(ep_info.get("l", 0))
                        dur   = time.time() - self._ep_start
                        self._ep_count += 1
                        self._returns.append(ret)
                        win  = min(50, len(self._returns))
                        mavg = float(np.mean(self._returns[-win:]))

                        loss = None
                        try:
                            loss = self.model.logger.name_to_value.get(
                                "train/loss",
                                self.model.logger.name_to_value.get(
                                    "train/value_loss"))
                        except Exception:
                            pass

                        self.bus.publish(Event(
                            type=EventType.EPISODE_COMPLETED,
                            job_id=self.job_id,
                            data={
                                "episode":    self._ep_count,
                                "n_episodes": self.n_episodes,
                                "return":     ret,
                                "moving_avg": mavg,
                                "steps":      steps,
                                "duration":   dur,
                                "loss":       loss,
                                "epsilon":    None,
                                "returns":    list(self._returns),
                            },
                        ))
                        self._ep_start = time.time()
                        if self._ep_count >= self.n_episodes:
                            return False
            return True

        def _on_training_end(self) -> None:
            self.bus.publish(Event(
                type=EventType.TRAINING_DONE,
                job_id=self.job_id,
                data={"returns": list(self._returns)},
            ))

        @property
        def returns(self) -> List[float]:
            return list(self._returns)

except ImportError:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Job Status
# ─────────────────────────────────────────────────────────────────────────────

class JobStatus(Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    PAUSED    = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR     = "error"


# ─────────────────────────────────────────────────────────────────────────────
# TrainingJob
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrainingJob:
    job_id:          str
    algo_name:       str
    algo_config:     Any
    env_config:      EnvConfig
    ep_config:       EpisodeConfig
    status:          JobStatus    = JobStatus.PENDING
    visible:         bool         = True
    returns:         List[float]  = field(default_factory=list)
    moving_avg:      List[float]  = field(default_factory=list)
    current_episode: int          = 0
    model:           Any          = field(default=None, repr=False)
    thread:          Optional[threading.Thread] = field(default=None, repr=False)
    stop_event:      threading.Event = field(default_factory=threading.Event, repr=False)
    pause_event:     threading.Event = field(default_factory=threading.Event, repr=False)
    frame_queue:     queue.Queue     = field(
        default_factory=lambda: queue.Queue(maxsize=1), repr=False)
    error_msg:       str          = ""
    label:           str          = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = f"{self.algo_name}-{self.job_id[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: build env / model
# ─────────────────────────────────────────────────────────────────────────────

def _activation_fn(name: str):
    import torch.nn as nn
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(name, nn.ReLU)


def _make_policy_kwargs(net: NetworkConfig) -> dict:
    return {
        "net_arch":      net.hidden_layers,
        "activation_fn": _activation_fn(net.activation),
    }


def _make_env(env_c: EnvConfig):
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    env = gym.make(
        ENV_NAME,
        render_mode="rgb_array",
        forward_reward_weight=env_c.forward_reward_weight,
        ctrl_cost_weight=env_c.ctrl_cost_weight,
        healthy_reward=env_c.healthy_reward,
        terminate_when_unhealthy=env_c.terminate_when_unhealthy,
        healthy_z_range=(env_c.healthy_z_min, env_c.healthy_z_max),
        healthy_angle_range=(env_c.healthy_angle_min, env_c.healthy_angle_max),
        reset_noise_scale=env_c.reset_noise_scale,
        exclude_current_positions_from_observation=(
            env_c.exclude_current_positions_from_observation),
    )
    return Monitor(env)


def _build_model(job: TrainingJob):
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.noise import NormalActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv

    ec  = job.ep_config
    ac  = job.algo_config
    envc= job.env_config

    vec_env = DummyVecEnv([lambda ec=envc: _make_env(ec)])
    pk      = _make_policy_kwargs(ac.network)

    if isinstance(ac, PPOConfig):
        return PPO(
            "MlpPolicy", vec_env,
            learning_rate = ec.alpha,
            gamma         = ec.gamma,
            n_steps       = ac.n_steps,
            batch_size    = ac.batch_size,
            n_epochs      = ac.n_epochs,
            clip_range    = ac.clip_range,
            ent_coef      = ac.ent_coef,
            vf_coef       = ac.vf_coef,
            max_grad_norm = ac.max_grad_norm,
            gae_lambda    = ac.gae_lambda,
            policy_kwargs = pk,
            verbose=0,
        )
    elif isinstance(ac, SACConfig):
        total = ec.n_episodes * ec.max_steps
        buf   = min(ac.buffer_size, max(total, ac.batch_size * 10))
        return SAC(
            "MlpPolicy", vec_env,
            learning_rate          = ec.alpha,
            gamma                  = ec.gamma,
            buffer_size            = buf,
            batch_size             = ac.batch_size,
            learning_starts        = min(ac.learning_starts, total // 2),
            train_freq             = ac.train_freq,
            gradient_steps         = ac.gradient_steps,
            tau                    = ac.tau,
            ent_coef               = ac.ent_coef,
            target_update_interval = ac.target_update_interval,
            policy_kwargs          = pk,
            verbose=0,
        )
    elif isinstance(ac, TD3Config):
        total    = ec.n_episodes * ec.max_steps
        buf      = min(ac.buffer_size, max(total, ac.batch_size * 10))
        n_acts   = vec_env.action_space.shape[0]
        act_noise = NormalActionNoise(
            mean=np.zeros(n_acts),
            sigma=ac.action_noise_sigma * np.ones(n_acts),
        )
        return TD3(
            "MlpPolicy", vec_env,
            learning_rate       = ec.alpha,
            gamma               = ec.gamma,
            buffer_size         = buf,
            batch_size          = ac.batch_size,
            learning_starts     = min(ac.learning_starts, total // 2),
            train_freq          = ac.train_freq,
            gradient_steps      = ac.gradient_steps,
            tau                 = ac.tau,
            policy_delay        = ac.policy_delay,
            target_policy_noise = ac.target_noise,
            target_noise_clip   = ac.noise_clip,
            action_noise        = act_noise,
            policy_kwargs       = pk,
            verbose=0,
        )
    raise ValueError(f"Unknown algo: {type(ac)}")


# ─────────────────────────────────────────────────────────────────────────────
# Thread workers
# ─────────────────────────────────────────────────────────────────────────────

def _run_job(job: TrainingJob, bus: EventBus) -> None:
    try:
        bus.publish(Event(EventType.JOB_STARTED, job.job_id))
        if job.model is None:
            job.model = _build_model(job)

        total_steps = job.ep_config.n_episodes * job.ep_config.max_steps
        cb = WorkbenchCallback(
            job_id        = job.job_id,
            bus           = bus,
            n_episodes    = job.ep_config.n_episodes,
            stop_event    = job.stop_event,
            pause_event   = job.pause_event,
            frame_queue   = job.frame_queue,
            visualize_flag= lambda: job.env_config.visualize,
        )
        cb._returns  = list(job.returns)
        cb._ep_count = len(job.returns)
        job.status   = JobStatus.RUNNING

        job.model.learn(
            total_timesteps     = total_steps,
            callback            = cb,
            reset_num_timesteps = (len(job.returns) == 0),
            progress_bar        = False,
        )

        job.returns         = cb.returns
        job.current_episode = cb._ep_count

        if not job.stop_event.is_set():
            job.status = JobStatus.COMPLETED
            bus.publish(Event(EventType.JOB_COMPLETED, job.job_id,
                              {"returns": job.returns}))
        else:
            job.status = JobStatus.CANCELLED
            bus.publish(Event(EventType.JOB_CANCELLED, job.job_id))

    except Exception as exc:
        import traceback as _tb
        job.status    = JobStatus.ERROR
        job.error_msg = str(exc)
        bus.publish(Event(EventType.JOB_ERROR, job.job_id,
                          {"error": str(exc), "traceback": _tb.format_exc()}))


def _run_inference(job: TrainingJob, bus: EventBus, n_episodes: int = 5) -> None:
    try:
        import gymnasium as gym
        env_c = job.env_config
        env = gym.make(
            ENV_NAME,
            render_mode="rgb_array",
            forward_reward_weight=env_c.forward_reward_weight,
            ctrl_cost_weight=env_c.ctrl_cost_weight,
            healthy_reward=env_c.healthy_reward,
            terminate_when_unhealthy=env_c.terminate_when_unhealthy,
            healthy_z_range=(env_c.healthy_z_min, env_c.healthy_z_max),
            healthy_angle_range=(env_c.healthy_angle_min, env_c.healthy_angle_max),
            reset_noise_scale=env_c.reset_noise_scale,
            exclude_current_positions_from_observation=(
                env_c.exclude_current_positions_from_observation),
        )
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done and not job.stop_event.is_set():
                while job.pause_event.is_set() and not job.stop_event.is_set():
                    time.sleep(0.05)
                action, _ = job.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                frame = env.render()
                if frame is not None:
                    try:
                        job.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    job.frame_queue.put_nowait(frame)
                time.sleep(0.01)
            if job.stop_event.is_set():
                break
        env.close()
    except Exception as exc:
        job.error_msg = str(exc)
    finally:
        job.status = JobStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# TrainingManager
# ─────────────────────────────────────────────────────────────────────────────

class TrainingManager:

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus   = bus or EventBus()
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create_job(self, algo_name: str, algo_config: Any,
                   env_config: EnvConfig, ep_config: EpisodeConfig,
                   label: str = "") -> TrainingJob:
        jid = str(uuid.uuid4())
        job = TrainingJob(
            job_id=jid, algo_name=algo_name, algo_config=algo_config,
            env_config=env_config, ep_config=ep_config,
            label=label or f"{algo_name}-{jid[:6]}",
        )
        with self._lock:
            self._jobs[jid] = job
        self.bus.publish(Event(EventType.JOB_CREATED, jid,
                               {"label": job.label, "algo": algo_name}))
        return job

    def start_job(self, job_id: str) -> None:
        job = self._get(job_id)
        if job.status in (JobStatus.PENDING, JobStatus.COMPLETED,
                          JobStatus.CANCELLED):
            job.stop_event.clear()
            job.pause_event.clear()
            t = threading.Thread(target=_run_job, args=(job, self.bus),
                                 daemon=True, name=f"train-{job_id[:8]}")
            job.thread = t
            t.start()

    def start_all_pending(self) -> None:
        for job in self.jobs:
            if job.status == JobStatus.PENDING:
                self.start_job(job.job_id)

    def pause(self, job_id: str) -> None:
        job = self._get(job_id)
        if job.status == JobStatus.RUNNING:
            job.pause_event.set()
            job.status = JobStatus.PAUSED
            self.bus.publish(Event(EventType.JOB_PAUSED, job_id))

    def resume(self, job_id: str) -> None:
        job = self._get(job_id)
        if job.status == JobStatus.PAUSED:
            job.pause_event.clear()
            job.status = JobStatus.RUNNING
            self.bus.publish(Event(EventType.JOB_RESUMED, job_id))

    def cancel(self, job_id: str) -> None:
        job = self._get(job_id)
        job.stop_event.set()
        job.pause_event.clear()
        job.status = JobStatus.CANCELLED
        self.bus.publish(Event(EventType.JOB_CANCELLED, job_id))

    def remove(self, job_id: str) -> None:
        job = self._get(job_id)
        self.cancel(job_id)
        if job.thread and job.thread.is_alive():
            job.thread.join(timeout=3)
        with self._lock:
            del self._jobs[job_id]
        self.bus.publish(Event(EventType.JOB_REMOVED, job_id))

    def run_inference(self, job_id: str, n_episodes: int = 5) -> None:
        job = self._get(job_id)
        if job.model is None:
            return
        job.stop_event.clear()
        job.pause_event.clear()
        job.status = JobStatus.RUNNING
        t = threading.Thread(target=_run_inference,
                             args=(job, self.bus, n_episodes),
                             daemon=True, name=f"run-{job_id[:8]}")
        job.thread = t
        t.start()

    def toggle_visibility(self, job_id: str) -> bool:
        job = self._get(job_id)
        job.visible = not job.visible
        return job.visible

    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self._jobs.get(job_id)

    @property
    def jobs(self) -> List[TrainingJob]:
        with self._lock:
            return list(self._jobs.values())

    def _get(self, job_id: str) -> TrainingJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(f"Job {job_id} not found")
        return job


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointManager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:

    @staticmethod
    def save(job: TrainingJob, base_dir: str) -> str:
        save_dir = os.path.join(base_dir, job.label.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)
        if job.model is not None:
            job.model.save(os.path.join(save_dir, "model"))
        meta: Dict[str, Any] = {
            "job_id":          job.job_id,
            "algo_name":       job.algo_name,
            "label":           job.label,
            "returns":         job.returns,
            "current_episode": job.current_episode,
            "algo_config":     asdict(job.algo_config)
                               if hasattr(job.algo_config, "__dataclass_fields__") else {},
            "env_config":      asdict(job.env_config),
            "ep_config":       asdict(job.ep_config),
        }
        with open(os.path.join(save_dir, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        return save_dir

    @staticmethod
    def load(save_dir: str, manager: TrainingManager) -> TrainingJob:
        from stable_baselines3 import PPO, SAC, TD3
        from stable_baselines3.common.vec_env import DummyVecEnv

        with open(os.path.join(save_dir, "meta.json")) as fh:
            meta = json.load(fh)

        algo_name = meta["algo_name"]
        ec  = EpisodeConfig(**{k: v for k, v in meta.get("ep_config",  {}).items()
                               if k in EpisodeConfig.__dataclass_fields__})
        env = EnvConfig(**{k: v for k, v in meta.get("env_config", {}).items()
                           if k in EnvConfig.__dataclass_fields__})
        ac_cls = ALGO_CONFIGS[algo_name]
        ac  = ac_cls(**{k: v for k, v in meta.get("algo_config", {}).items()
                        if k in ac_cls.__dataclass_fields__})

        job = manager.create_job(algo_name, ac, env, ec,
                                 label=meta.get("label", ""))
        j   = manager.get_job(job.job_id)
        j.returns         = meta.get("returns", [])
        j.current_episode = meta.get("current_episode", len(j.returns))
        j.status          = JobStatus.COMPLETED

        model_path = os.path.join(save_dir, "model.zip")
        if os.path.exists(model_path):
            cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}.get(algo_name)
            if cls:
                j.model = cls.load(model_path,
                                   env=DummyVecEnv([lambda ec=env: _make_env(ec)]))
        return j


# ─────────────────────────────────────────────────────────────────────────────
# Tuning helper
# ─────────────────────────────────────────────────────────────────────────────

def expand_tuning_values(raw: str) -> List[Any]:
    """Parse semicolon-separated values; comma-lists become int lists."""
    result = []
    for p in (p.strip() for p in raw.split(";") if p.strip()):
        if "," in p:
            sub = [s.strip() for s in p.split(",") if s.strip()]
            try:
                result.append([int(x) for x in sub])
            except ValueError:
                result.append(sub)
        else:
            try:
                v = float(p)
                result.append(int(v) if v == int(v) else v)
            except ValueError:
                result.append(p)
    return result
