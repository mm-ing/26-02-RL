"""
Logic layer for the Ant RL Workbench.
TQC / SAC via Stable-Baselines3 (sb3-contrib / SB3)
and CMA-ES via EvoTorch.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
import uuid
from copy import deepcopy
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
    """Thread-safe event bus backed by a single Queue for Tkinter polling."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Event] = queue.Queue()
        self._listeners: List[Callable[[Event], None]] = []

    def subscribe(self, callback: Callable[[Event], None]) -> None:
        self._listeners.append(callback)

    def publish(self, event: Event) -> None:
        self._queue.put_nowait(event)

    def drain(self) -> None:
        """Call from UI thread to dispatch all pending events."""
        while True:
            try:
                event = self._queue.get_nowait()
            except queue.Empty:
                break
            for listener in self._listeners:
                try:
                    listener(event)
                except Exception:
                    import traceback
                    traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Config Dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EnvConfig:
    forward_reward_weight:                      float = 1.0
    ctrl_cost_weight:                           float = 0.5
    contact_cost_weight:                        float = 5e-4
    healthy_reward:                             float = 1.0
    terminate_when_unhealthy:                   bool  = True
    healthy_z_min:                              float = 0.2
    healthy_z_max:                              float = 1.0
    contact_force_min:                          float = -1.0
    contact_force_max:                          float = 1.0
    reset_noise_scale:                          float = 0.1
    exclude_current_positions_from_observation: bool  = True
    include_cfrc_ext_in_observation:            bool  = True
    render_interval_ms: int  = 10
    visualize:          bool = True


@dataclass
class EpisodeConfig:
    n_episodes:      int   = 3000
    max_steps:       int   = 1000
    alpha:           float = 3e-4   # learning rate
    gamma:           float = 0.99
    compare_methods: bool  = False


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation:    str       = "relu"   # relu | tanh | elu


@dataclass
class TQCConfig:
    algo_name:              str           = "TQC"
    buffer_size:            int           = 300_000
    batch_size:             int           = 256
    learning_starts:        int           = 10_000
    train_freq:             int           = 1
    gradient_steps:         int           = 1
    tau:                    float         = 0.005
    ent_coef:               str           = "auto"
    target_update_interval: int           = 1
    top_quantiles_to_drop:  int           = 2
    n_quantiles:            int           = 25
    n_critics:              int           = 2
    network:                NetworkConfig = field(default_factory=NetworkConfig)


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
class CMAESConfig:
    algo_name:       str           = "CMA-ES"
    popsize:         int           = 16   # keep small; large values cause OOM with many params
    stdev_init:      float         = 0.5
    n_eval_episodes: int           = 3
    num_actors:      int           = 2    # parallel evaluation workers (1 = no Ray)
    max_memory_gb:   float         = 2.0  # hard cap on solution-tensor memory
    network:         NetworkConfig = field(default_factory=lambda: NetworkConfig(
        hidden_layers=[64, 64],
        activation="tanh",
    ))


ALGO_CONFIGS: Dict[str, Any] = {
    "TQC":    TQCConfig,
    "SAC":    SACConfig,
    "CMA-ES": CMAESConfig,
}

ENV_NAME = "Ant-v5"


# ─────────────────────────────────────────────────────────────────────────────
# Environment helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ant_env_kwargs(env_c: EnvConfig, render: bool = False) -> dict:
    kw: Dict[str, Any] = {
        "forward_reward_weight":                      env_c.forward_reward_weight,
        "ctrl_cost_weight":                           env_c.ctrl_cost_weight,
        "contact_cost_weight":                        env_c.contact_cost_weight,
        "healthy_reward":                             env_c.healthy_reward,
        "terminate_when_unhealthy":                   env_c.terminate_when_unhealthy,
        "healthy_z_range":                            (env_c.healthy_z_min, env_c.healthy_z_max),
        "contact_force_range":                        (env_c.contact_force_min, env_c.contact_force_max),
        "reset_noise_scale":                          env_c.reset_noise_scale,
        "exclude_current_positions_from_observation": env_c.exclude_current_positions_from_observation,
        "include_cfrc_ext_in_observation":            env_c.include_cfrc_ext_in_observation,
    }
    if render:
        kw["render_mode"] = "rgb_array"
    return kw


def _make_env(env_c: EnvConfig):
    """Create a single Ant-v5 env wrapped with Monitor (for SB3)."""
    import gymnasium as gym
    from stable_baselines3.common.monitor import Monitor
    env = gym.make(ENV_NAME, render_mode="rgb_array", **_ant_env_kwargs(env_c))
    return Monitor(env)


# ─────────────────────────────────────────────────────────────────────────────
# SB3 Callback (shared by TQC and SAC)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from stable_baselines3.common.callbacks import BaseCallback

    class WorkbenchCallback(BaseCallback):
        """Fires EventBus events per episode; handles pause/stop; captures frames."""

        def __init__(
            self,
            job_id:         str,
            bus:            EventBus,
            n_episodes:     int,
            stop_event:     threading.Event,
            pause_event:    threading.Event,
            frame_queue:    "queue.Queue",
            visualize_flag: Callable[[], bool],
            verbose:        int = 0,
        ) -> None:
            super().__init__(verbose)
            self.job_id          = job_id
            self.bus             = bus
            self.n_episodes      = n_episodes
            self.stop_event      = stop_event
            self.pause_event     = pause_event
            self.frame_queue     = frame_queue
            self.visualize_flag  = visualize_flag
            self._ep_count       = 0
            self._ep_start_time  = time.time()
            self._returns: List[float] = []
            self._last_frame_t   = 0.0

        def _on_step(self) -> bool:
            # ── pause / stop ──────────────────────────────────────────
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.05)
            if self.stop_event.is_set():
                return False

            # ── render frame (throttled ~100 Hz) ─────────────────────
            now = time.time()
            if self.visualize_flag() and now - self._last_frame_t >= 0.01:
                try:
                    env   = self.training_env.envs[0]
                    frame = env.render()
                    if frame is not None:
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.frame_queue.put_nowait(frame)
                        self._last_frame_t = now
                except Exception:
                    pass

            # ── episode end detection ─────────────────────────────────
            dones = self.locals.get("dones")
            infos = self.locals.get("infos", [])
            if dones is not None:
                for done, info in zip(dones, infos):
                    if done:
                        ep_info = info.get("episode", {})
                        ret     = float(ep_info.get("r", 0.0))
                        steps   = int(ep_info.get("l", 0))
                        dur     = time.time() - self._ep_start_time
                        self._ep_count += 1
                        self._returns.append(ret)
                        win  = min(50, len(self._returns))
                        mavg = float(np.mean(self._returns[-win:]))

                        loss = None
                        try:
                            loss = self.model.logger.name_to_value.get(
                                "train/loss",
                                self.model.logger.name_to_value.get("train/value_loss"),
                            )
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
                        self._ep_start_time = time.time()

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
    frame_queue:     "queue.Queue"   = field(
        default_factory=lambda: queue.Queue(maxsize=1), repr=False)
    error_msg:       str          = ""
    label:           str          = ""

    def __post_init__(self):
        if not self.label:
            self.label = f"{self.algo_name}-{self.job_id[:6]}"


# ─────────────────────────────────────────────────────────────────────────────
# Model building helpers – SB3 (TQC / SAC)
# ─────────────────────────────────────────────────────────────────────────────

def _activation_fn(name: str):
    import torch.nn as nn
    return {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(name, nn.ReLU)


def _make_policy_kwargs(net_cfg: NetworkConfig) -> dict:
    return {
        "net_arch":      net_cfg.hidden_layers,
        "activation_fn": _activation_fn(net_cfg.activation),
    }


def _build_sb3_model(job: TrainingJob):
    """Build (or re-instantiate) the SB3 model for TQC or SAC."""
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv

    ec    = job.ep_config
    ac    = job.algo_config
    env_c = job.env_config

    vec_env = DummyVecEnv([lambda ec=env_c: _make_env(ec)])
    pk      = _make_policy_kwargs(ac.network)

    if isinstance(ac, TQCConfig):
        try:
            from sb3_contrib import TQC
        except ImportError:
            raise ImportError(
                "sb3-contrib not installed. Install with: pip install sb3-contrib"
            )
        total_steps = ec.n_episodes * ec.max_steps
        buf = min(ac.buffer_size, max(total_steps, ac.batch_size * 10))
        tqc_pk = dict(pk)
        tqc_pk["n_quantiles"] = ac.n_quantiles
        tqc_pk["n_critics"]   = ac.n_critics
        model = TQC(
            "MlpPolicy", vec_env,
            learning_rate              = ec.alpha,
            gamma                      = ec.gamma,
            buffer_size                = buf,
            batch_size                 = ac.batch_size,
            learning_starts            = min(ac.learning_starts, total_steps // 2),
            train_freq                 = ac.train_freq,
            gradient_steps             = ac.gradient_steps,
            tau                        = ac.tau,
            ent_coef                   = ac.ent_coef,
            target_update_interval     = ac.target_update_interval,
            top_quantiles_to_drop_per_net = ac.top_quantiles_to_drop,
            policy_kwargs              = tqc_pk,
            verbose=0,
        )
    elif isinstance(ac, SACConfig):
        total_steps = ec.n_episodes * ec.max_steps
        buf = min(ac.buffer_size, max(total_steps, ac.batch_size * 10))
        model = SAC(
            "MlpPolicy", vec_env,
            learning_rate          = ec.alpha,
            gamma                  = ec.gamma,
            buffer_size            = buf,
            batch_size             = ac.batch_size,
            learning_starts        = min(ac.learning_starts, total_steps // 2),
            train_freq             = ac.train_freq,
            gradient_steps         = ac.gradient_steps,
            tau                    = ac.tau,
            ent_coef               = ac.ent_coef,
            target_update_interval = ac.target_update_interval,
            policy_kwargs          = pk,
            verbose=0,
        )
    else:
        raise ValueError(f"Unknown SB3 algo config: {type(ac)}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Thread workers – SB3 (TQC / SAC)
# ─────────────────────────────────────────────────────────────────────────────

def _run_sb3_job(job: TrainingJob, bus: EventBus) -> None:
    try:
        bus.publish(Event(EventType.JOB_STARTED, job.job_id))

        if job.model is None:
            job.model = _build_sb3_model(job)

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

        job.status = JobStatus.RUNNING

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
        tb_str = _tb.format_exc()
        job.status    = JobStatus.ERROR
        job.error_msg = str(exc)
        bus.publish(Event(EventType.JOB_ERROR, job.job_id,
                          {"error": str(exc), "traceback": tb_str}))


def _run_sb3_inference(job: TrainingJob, bus: EventBus, n_episodes: int = 5) -> None:
    try:
        import gymnasium as gym
        env_c = job.env_config
        env = gym.make(ENV_NAME, **_ant_env_kwargs(env_c, render=True))
        for _ in range(n_episodes):
            obs, _ = env.reset()
            done   = False
            while not done and not job.stop_event.is_set():
                while job.pause_event.is_set() and not job.stop_event.is_set():
                    time.sleep(0.05)
                action, _ = job.model.predict(obs, deterministic=True)
                obs, _r, terminated, truncated, _ = env.step(action)
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
# Thread workers – CMA-ES (EvoTorch)
# ─────────────────────────────────────────────────────────────────────────────

def _build_cmaes_net_class(obs_dim: int, act_dim: int, net_cfg: NetworkConfig):
    """Return a PolicyNet nn.Module class for the given architecture."""
    import torch.nn as nn

    layers   = list(net_cfg.hidden_layers)
    act_name = net_cfg.activation
    act_cls  = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU}.get(act_name, nn.Tanh)
    sizes    = [obs_dim] + layers + [act_dim]

    class PolicyNet(nn.Module):
        _sizes  = sizes
        _act_fn = act_cls

        def __init__(self):
            super().__init__()
            seq = []
            for i in range(len(self._sizes) - 1):
                seq.append(nn.Linear(self._sizes[i], self._sizes[i + 1]))
                if i < len(self._sizes) - 2:
                    seq.append(self._act_fn())
            seq.append(nn.Tanh())          # bound actions to [-1, 1]
            self.net = nn.Sequential(*seq)

        def forward(self, x):
            return self.net(x)

    return PolicyNet


def _make_cmaes_render_thread(
    job: TrainingJob,
    env_c: EnvConfig,
    net_holder: list,
    net_lock: threading.Lock,
    stop_event: threading.Event,
) -> threading.Thread:
    """Return a daemon thread that continuously renders the best CMA-ES policy.

    *net_holder* is a one-element list ``[net_or_None]`` shared with the
    training loop.  The lock protects writes to that list.
    """

    def _worker() -> None:
        import gymnasium as gym
        import torch
        try:
            renv = gym.make(ENV_NAME, **_ant_env_kwargs(env_c, render=True))
            obs, _ = renv.reset()
            while not stop_event.is_set() and not job.stop_event.is_set():
                if not env_c.visualize:
                    time.sleep(0.05)
                    continue
                with net_lock:
                    net = net_holder[0]
                if net is None:
                    time.sleep(0.05)
                    continue
                try:
                    with torch.no_grad():
                        x      = torch.tensor(obs, dtype=torch.float32)
                        action = net(x).numpy()
                    obs, _, term, trunc, _ = renv.step(action)
                    frame = renv.render()
                    if frame is not None:
                        try:
                            job.frame_queue.get_nowait()
                        except queue.Empty:
                            pass
                        job.frame_queue.put_nowait(frame)
                    if term or trunc:
                        obs, _ = renv.reset()
                except Exception:
                    obs, _ = renv.reset()
                time.sleep(0.01)   # ~100 Hz cap
        except Exception:
            pass
        finally:
            try:
                renv.close()
            except Exception:
                pass

    t = threading.Thread(target=_worker, daemon=True,
                         name=f"cmaes-render-{job.job_id[:8]}")
    return t


def _run_cmaes_job(job: TrainingJob, bus: EventBus) -> None:
    """Train using CMA-ES via EvoTorch GymNE with optional parallel actors
    and a continuous background render thread."""

    # ── shared state for the render thread ────────────────────────────
    net_holder: list          = [None]   # [nn.Module | None]
    net_lock:   threading.Lock = threading.Lock()
    render_stop = threading.Event()
    render_thread: Optional[threading.Thread] = None

    try:
        bus.publish(Event(EventType.JOB_STARTED, job.job_id))

        try:
            from evotorch.neuroevolution import GymNE
            from evotorch.algorithms import CMAES as EvoTorchCMAES
        except ImportError:
            raise ImportError(
                "EvoTorch not installed. Install with: pip install evotorch"
            )

        import gymnasium as gym
        import torch

        ac    = job.algo_config   # CMAESConfig
        ec    = job.ep_config
        env_c = job.env_config

        env_kwargs = _ant_env_kwargs(env_c)

        # Determine observation / action dims from a temporary env
        tmp = gym.make(ENV_NAME, **env_kwargs)
        obs_dim = int(np.prod(tmp.observation_space.shape))
        act_dim = int(np.prod(tmp.action_space.shape))
        tmp.close()

        PolicyNet = _build_cmaes_net_class(obs_dim, act_dim, ac.network)

        # ── count policy parameters ────────────────────────────────────
        _probe = PolicyNet()
        param_count = sum(p.numel() for p in _probe.parameters())
        del _probe

        # ── configure parallel actors ──────────────────────────────────
        # num_actors > 1 requires Ray; fall back to 1 if unavailable.
        num_actors = max(1, ac.num_actors)
        if num_actors > 1:
            try:
                import ray  # noqa: F401
            except ImportError:
                import warnings
                warnings.warn(
                    "Ray is not installed – falling back to num_actors=1. "
                    "Install with: pip install ray",
                    stacklevel=2,
                )
                num_actors = 1

        # ── memory guard ──────────────────────────────────────────────
        # EvoTorch allocates a [popsize, param_count] float32 tensor.
        # Clamp popsize so that tensor stays within max_memory_gb.
        bytes_per_float = 4
        max_bytes = int(ac.max_memory_gb * 1024 ** 3)
        safe_popsize = max(4, min(ac.popsize,
                                  max_bytes // max(1, param_count * bytes_per_float)))
        if safe_popsize < ac.popsize:
            import warnings
            warnings.warn(
                f"CMA-ES: popsize clamped {ac.popsize} → {safe_popsize} "
                f"(policy has {param_count:,} params; "
                f"limit {ac.max_memory_gb:.1f} GB). "
                "Reduce hidden layers or raise Max Memory GB to use a larger popsize.",
                stacklevel=2,
            )

        problem = GymNE(
            env=ENV_NAME,
            env_config=env_kwargs,
            network=PolicyNet,
            episode_length=ec.max_steps,
            num_episodes=ac.n_eval_episodes,
            observation_normalization=True,
            num_actors=num_actors,
        )

        searcher = EvoTorchCMAES(
            problem,
            stdev_init=ac.stdev_init,
            popsize=safe_popsize,
        )

        # ── start background render thread ─────────────────────────────
        render_thread = _make_cmaes_render_thread(
            job, env_c, net_holder, net_lock, render_stop
        )
        render_thread.start()

        n_gens     = ec.n_episodes   # treat generations as "episodes" for the plot
        start_gen  = len(job.returns)
        job.status = JobStatus.RUNNING

        for gen in range(start_gen, n_gens):
            if job.stop_event.is_set():
                break
            while job.pause_event.is_set() and not job.stop_event.is_set():
                time.sleep(0.05)

            t0 = time.time()
            searcher.step()
            dur = time.time() - t0

            status = searcher.status
            ret = float(
                status.get("pop_best_eval",
                           status.get("best_eval",
                                      status.get("mean_eval", 0.0)))
            )

            job.returns.append(ret)
            win  = min(50, len(job.returns))
            mavg = float(np.mean(job.returns[-win:]))

            bus.publish(Event(
                type=EventType.EPISODE_COMPLETED,
                job_id=job.job_id,
                data={
                    "episode":    gen + 1,
                    "n_episodes": n_gens,
                    "return":     ret,
                    "moving_avg": mavg,
                    "steps":      ac.popsize * ac.n_eval_episodes,
                    "duration":   dur,
                    "loss":       None,
                    "epsilon":    None,
                    "returns":    list(job.returns),
                },
            ))

            # Update best policy for render thread and job model
            try:
                best_sol = status.get("pop_best")
                if best_sol is not None:
                    best_net = problem.parameterize_net(best_sol.values.clone())
                    with net_lock:
                        net_holder[0] = best_net
                    job.model = {
                        "type":       "cmaes",
                        "net":        best_net,
                        "obs_dim":    obs_dim,
                        "act_dim":    act_dim,
                        "net_cfg":    ac.network,
                        "env_kwargs": env_kwargs,
                    }
            except Exception:
                pass

        if not job.stop_event.is_set():
            job.status = JobStatus.COMPLETED
            bus.publish(Event(EventType.JOB_COMPLETED, job.job_id,
                              {"returns": job.returns}))
        else:
            job.status = JobStatus.CANCELLED
            bus.publish(Event(EventType.JOB_CANCELLED, job.job_id))

    except Exception as exc:
        import traceback as _tb
        tb_str = _tb.format_exc()
        job.status    = JobStatus.ERROR
        job.error_msg = str(exc)
        bus.publish(Event(EventType.JOB_ERROR, job.job_id,
                          {"error": str(exc), "traceback": tb_str}))
    finally:
        # Always shut down the render thread cleanly
        render_stop.set()
        if render_thread is not None and render_thread.is_alive():
            render_thread.join(timeout=5)


def _run_cmaes_inference(job: TrainingJob, bus: EventBus, n_episodes: int = 5) -> None:
    """Run the saved CMA-ES best policy for visualization."""
    try:
        if not isinstance(job.model, dict) or "net" not in job.model:
            return

        import torch
        net        = job.model["net"]
        env_kwargs = job.model.get("env_kwargs", {})

        import gymnasium as gym
        env = gym.make(ENV_NAME, render_mode="rgb_array", **env_kwargs)

        for _ in range(n_episodes):
            if job.stop_event.is_set():
                break
            obs, _ = env.reset()
            done   = False
            while not done and not job.stop_event.is_set():
                while job.pause_event.is_set() and not job.stop_event.is_set():
                    time.sleep(0.05)
                with torch.no_grad():
                    x      = torch.tensor(obs, dtype=torch.float32)
                    action = net(x).numpy()
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

        env.close()
    except Exception as exc:
        job.error_msg = str(exc)
    finally:
        job.status = JobStatus.COMPLETED


# ─────────────────────────────────────────────────────────────────────────────
# Unified dispatch helpers
# ─────────────────────────────────────────────────────────────────────────────

def _run_job(job: TrainingJob, bus: EventBus) -> None:
    if isinstance(job.algo_config, CMAESConfig):
        _run_cmaes_job(job, bus)
    else:
        _run_sb3_job(job, bus)


def _run_job_inference(job: TrainingJob, bus: EventBus, n_episodes: int = 5) -> None:
    if isinstance(job.algo_config, CMAESConfig):
        _run_cmaes_inference(job, bus, n_episodes)
    else:
        _run_sb3_inference(job, bus, n_episodes)


# ─────────────────────────────────────────────────────────────────────────────
# Training Manager
# ─────────────────────────────────────────────────────────────────────────────

class TrainingManager:
    """Manages TrainingJobs and dispatches events via EventBus."""

    def __init__(self, bus: Optional[EventBus] = None) -> None:
        self.bus: EventBus = bus or EventBus()
        self._jobs: Dict[str, TrainingJob] = {}
        self._lock = threading.Lock()

    def create_job(
        self,
        algo_name:   str,
        algo_config: Any,
        env_config:  EnvConfig,
        ep_config:   EpisodeConfig,
        label:       str = "",
    ) -> TrainingJob:
        job_id = str(uuid.uuid4())
        job = TrainingJob(
            job_id      = job_id,
            algo_name   = algo_name,
            algo_config = algo_config,
            env_config  = env_config,
            ep_config   = ep_config,
            label       = label or f"{algo_name}-{job_id[:6]}",
        )
        with self._lock:
            self._jobs[job_id] = job
        self.bus.publish(Event(EventType.JOB_CREATED, job_id,
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
        t = threading.Thread(target=_run_job_inference,
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
# Checkpoint Manager
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointManager:

    @staticmethod
    def save(job: TrainingJob, base_dir: str) -> str:
        save_dir = os.path.join(base_dir, job.label.replace(" ", "_"))
        os.makedirs(save_dir, exist_ok=True)

        # Save SB3 model or CMA-ES network
        if job.model is not None:
            if isinstance(job.model, dict) and job.model.get("type") == "cmaes":
                import torch
                net = job.model.get("net")
                if net is not None:
                    torch.save(net.state_dict(), os.path.join(save_dir, "cmaes_net.pt"))
                    # Save network config so we can rebuild later
                    cfg = job.model.get("net_cfg")
                    if cfg:
                        net_meta = {
                            "obs_dim":      job.model.get("obs_dim"),
                            "act_dim":      job.model.get("act_dim"),
                            "hidden_layers":cfg.hidden_layers,
                            "activation":   cfg.activation,
                        }
                        with open(os.path.join(save_dir, "cmaes_net_meta.json"), "w") as fh:
                            json.dump(net_meta, fh, indent=2)
            else:
                try:
                    job.model.save(os.path.join(save_dir, "model"))
                except Exception:
                    pass

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
        with open(os.path.join(save_dir, "meta.json")) as fh:
            meta = json.load(fh)

        algo_name = meta["algo_name"]
        ec_d  = meta.get("ep_config",   {})
        env_d = meta.get("env_config",  {})
        ac_d  = meta.get("algo_config", {})

        ec  = EpisodeConfig(**{k: v for k, v in ec_d.items()
                               if k in EpisodeConfig.__dataclass_fields__})
        env = EnvConfig(**{k: v for k, v in env_d.items()
                           if k in EnvConfig.__dataclass_fields__})
        ac  = ALGO_CONFIGS[algo_name](**{k: v for k, v in ac_d.items()
                                         if k in ALGO_CONFIGS[algo_name].__dataclass_fields__})

        job = manager.create_job(algo_name, ac, env, ec, label=meta.get("label", ""))
        j   = manager.get_job(job.job_id)
        j.returns         = meta.get("returns", [])
        j.current_episode = meta.get("current_episode", len(j.returns))
        j.status          = JobStatus.COMPLETED

        model_path = os.path.join(save_dir, "model.zip")
        if os.path.exists(model_path):
            if algo_name == "TQC":
                try:
                    from sb3_contrib import TQC
                    from stable_baselines3.common.vec_env import DummyVecEnv
                    j.model = TQC.load(model_path,
                                       env=DummyVecEnv([lambda e=env: _make_env(e)]))
                except ImportError:
                    pass
            elif algo_name == "SAC":
                from stable_baselines3 import SAC
                from stable_baselines3.common.vec_env import DummyVecEnv
                j.model = SAC.load(model_path,
                                   env=DummyVecEnv([lambda e=env: _make_env(e)]))

        cmaes_net_path = os.path.join(save_dir, "cmaes_net.pt")
        cmaes_meta_path = os.path.join(save_dir, "cmaes_net_meta.json")
        if os.path.exists(cmaes_net_path) and os.path.exists(cmaes_meta_path):
            import torch
            with open(cmaes_meta_path) as fh:
                nm = json.load(fh)
            obs_dim = nm["obs_dim"]
            act_dim = nm["act_dim"]
            cfg = NetworkConfig(
                hidden_layers=nm["hidden_layers"],
                activation=nm["activation"],
            )
            PolicyNet = _build_cmaes_net_class(obs_dim, act_dim, cfg)
            net = PolicyNet()
            net.load_state_dict(torch.load(cmaes_net_path, map_location="cpu"))
            net.eval()
            j.model = {
                "type":       "cmaes",
                "net":        net,
                "obs_dim":    obs_dim,
                "act_dim":    act_dim,
                "net_cfg":    cfg,
                "env_kwargs": _ant_env_kwargs(env),
            }

        return j


# ─────────────────────────────────────────────────────────────────────────────
# Tuning helper
# ─────────────────────────────────────────────────────────────────────────────

def expand_tuning_values(raw: str) -> List[Any]:
    """
    Parse semicolon-separated tuning values.
      Parts without commas → scalar (int if whole number, else float)
      Parts with commas    → list of ints (hidden-layer config)
    """
    parts = [p.strip() for p in raw.split(";")]
    result = []
    for p in parts:
        if not p:
            continue
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
