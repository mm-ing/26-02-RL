from __future__ import annotations

import json
import inspect
import queue
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional

import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor


class InitialStateWrapper(gym.Wrapper):
    def __init__(self, env, x_init: Optional[float] = None, y_init: Optional[float] = None):
        super().__init__(env)
        self.x_init = x_init
        self.y_init = y_init

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.x_init is None and self.y_init is None:
            return obs, info
        if not hasattr(self.env.unwrapped, "state"):
            return obs, info

        state = np.array(self.env.unwrapped.state, dtype=np.float32, copy=True)
        low = getattr(self.env.observation_space, "low", None)
        high = getattr(self.env.observation_space, "high", None)

        if state.shape[0] >= 1 and self.x_init is not None:
            x_value = float(self.x_init)
            if low is None or high is None or (float(low[0]) <= x_value <= float(high[0])):
                state[0] = x_value

        if state.shape[0] >= 2 and self.y_init is not None:
            y_value = float(self.y_init)
            if low is None or high is None or (float(low[1]) <= y_value <= float(high[1])):
                state[1] = y_value

        self.env.unwrapped.state = state
        return state.copy(), info


def _filter_supported_env_kwargs(env_id: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        spec = gym.spec(env_id)
        entry_point = spec.entry_point
        if entry_point is None:
            return kwargs
        env_ctor = gym.envs.registration.load_env_creator(entry_point)
        signature = inspect.signature(env_ctor.__init__)
        valid = set(signature.parameters.keys())
        valid.discard("self")
        valid.discard("kwargs")
        return {k: v for k, v in kwargs.items() if k in valid}
    except Exception:
        return kwargs


def _build_env(env_id: str, render_mode: Optional[str], env_kwargs: Optional[Dict[str, Any]] = None):
    source_kwargs = dict(env_kwargs or {})
    x_init = source_kwargs.pop("x_init", None)
    y_init = source_kwargs.pop("y_init", None)
    filtered = _filter_supported_env_kwargs(env_id, source_kwargs)

    try:
        env = gym.make(env_id, render_mode=render_mode, **filtered)
    except TypeError:
        env = gym.make(env_id, render_mode=render_mode)

    if x_init is not None or y_init is not None:
        env = InitialStateWrapper(env, x_init=x_init, y_init=y_init)
    return env


@dataclass
class NetworkConfig:
    hidden_layers: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "relu"


@dataclass
class AlgorithmConfig:
    algorithm: str = "D3QN"
    learning_rate: float = 5e-4
    gamma: float = 0.99
    buffer_size: int = 100_000
    batch_size: int = 128
    learning_starts: int = 2_000
    train_freq: int = 4
    gradient_steps: int = 1
    target_update_interval: int = 500
    exploration_fraction: float = 0.3
    exploration_initial_eps: float = 1.0
    exploration_final_eps: float = 0.05
    net: NetworkConfig = field(default_factory=NetworkConfig)


@dataclass
class EpisodeConfig:
    episodes: int = 200
    max_steps: int = 200
    moving_average_window: int = 20


@dataclass
class TuneConfig:
    enabled: bool = False
    parameter: str = "learning_rate"
    min_value: float = 1e-4
    max_value: float = 1e-3
    step: float = 2e-4

    def values(self) -> List[float]:
        if not self.enabled:
            return []
        values = []
        current = self.min_value
        while current <= self.max_value + 1e-12:
            values.append(round(current, 10))
            current += self.step
        return values


@dataclass
class JobConfig:
    name: str
    env_id: str
    render_mode: str = "rgb_array"
    env_kwargs: Dict[str, Any] = field(default_factory=dict)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    episodes: EpisodeConfig = field(default_factory=EpisodeConfig)
    visible: bool = True


@dataclass
class StepResult:
    episode: int
    step: int
    reward: float
    done: bool


@dataclass
class EpisodeResult:
    episode: int
    total_reward: float
    steps: int
    duration_sec: float
    moving_avg: float


@dataclass
class TrainingResult:
    finished: bool
    episodes_done: int
    returns: List[float]
    moving_avg: List[float]


class GymEnvironment:
    def __init__(self, env_id: str, render_mode: Optional[str] = None, **kwargs: Any):
        self.env_id = env_id
        self.kwargs = kwargs
        self._env = _build_env(env_id, render_mode=render_mode, env_kwargs=kwargs)

    def reset(self, **kwargs: Any):
        return self._env.reset(**kwargs)

    def step(self, action: int):
        return self._env.step(action)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space


class EnvironmentRegistry:
    def __init__(self):
        self._registry: Dict[str, Callable[..., GymEnvironment]] = {}

    def register(self, name: str, factory: Callable[..., GymEnvironment]) -> None:
        self._registry[name] = factory

    def create(self, name: str, **kwargs: Any) -> GymEnvironment:
        if name in self._registry:
            return self._registry[name](**kwargs)
        return GymEnvironment(name, **kwargs)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buffer: Deque[Any] = deque(maxlen=capacity)

    def add(self, s, a, r, s_next, done) -> None:
        self._buffer.append((s, a, r, s_next, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if len(self._buffer) < batch_size:
            raise ValueError("Not enough samples in replay buffer")
        idx = np.random.choice(len(self._buffer), batch_size, replace=False)
        batch = [self._buffer[i] for i in idx]
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": np.asarray(states, dtype=np.float32),
            "actions": np.asarray(actions),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "next_states": np.asarray(next_states, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def __len__(self) -> int:
        return len(self._buffer)


class AlgorithmBase(ABC):
    @abstractmethod
    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def get_state_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raise NotImplementedError


def _activation_name_to_class(name: str):
    import torch.nn as nn

    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return mapping.get(name.lower(), nn.ReLU)


def _safe_make_env(env_id: str, render_mode: Optional[str], env_kwargs: Optional[Dict[str, Any]] = None):
    return _build_env(env_id, render_mode=render_mode, env_kwargs=env_kwargs)


class SB3DQNAlgorithm(AlgorithmBase):
    def __init__(
        self,
        env_id: str,
        config: AlgorithmConfig,
        render_mode: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.config = config
        self.train_env = Monitor(_safe_make_env(env_id, render_mode=render_mode, env_kwargs=env_kwargs))
        self.eval_env = _safe_make_env(env_id, render_mode=render_mode, env_kwargs=env_kwargs)

        policy_kwargs = {
            "net_arch": config.net.hidden_layers,
            "activation_fn": _activation_name_to_class(config.net.activation),
        }

        self.model = DQN(
            policy="MlpPolicy",
            env=self.train_env,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            buffer_size=config.buffer_size,
            batch_size=config.batch_size,
            learning_starts=config.learning_starts,
            train_freq=config.train_freq,
            gradient_steps=config.gradient_steps,
            target_update_interval=config.target_update_interval,
            exploration_fraction=config.exploration_fraction,
            exploration_initial_eps=config.exploration_initial_eps,
            exploration_final_eps=config.exploration_final_eps,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device="auto",
        )

    def select_action(self, state: np.ndarray, explore: bool = True) -> int:
        action, _ = self.model.predict(state, deterministic=not explore)
        return int(action)

    def update(self, batch: Dict[str, Any]) -> Dict[str, float]:
        total_timesteps = int(batch.get("total_timesteps", 1))
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=False)
        return {"loss": float("nan"), "epsilon": float(self.model.exploration_rate)}

    def evaluate_episode(
        self,
        max_steps: int,
        render: bool = False,
        on_frame: Optional[Callable[[Any], None]] = None,
    ) -> Dict[str, Any]:
        state, _ = self.eval_env.reset()
        total_reward = 0.0
        steps = 0
        last_frame = None

        if render:
            last_frame = self.eval_env.render()
            if on_frame and last_frame is not None:
                on_frame(last_frame)

        for _ in range(max_steps):
            action, _ = self.model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = self.eval_env.step(action)
            total_reward += float(reward)
            steps += 1
            if render:
                last_frame = self.eval_env.render()
                if on_frame and last_frame is not None:
                    on_frame(last_frame)
            if terminated or truncated:
                break

        return {"return": total_reward, "steps": steps, "frame": last_frame}

    def get_state_dict(self) -> Dict[str, Any]:
        path = Path(f"_tmp_{uuid.uuid4().hex}.zip")
        self.model.save(path)
        raw = path.read_bytes()
        path.unlink(missing_ok=True)
        return {"zip_bytes": raw.hex()}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        raw = bytes.fromhex(state["zip_bytes"])
        path = Path(f"_tmp_{uuid.uuid4().hex}.zip")
        path.write_bytes(raw)
        loaded = DQN.load(path, env=self.train_env, device="auto")
        self.model = loaded
        path.unlink(missing_ok=True)

    def save(self, path: Path) -> None:
        self.model.save(path)

    def load(self, path: Path) -> None:
        self.model = DQN.load(path, env=self.train_env, device="auto")

    def close(self) -> None:
        self.train_env.close()
        self.eval_env.close()


class TrainLoop:
    def __init__(
        self,
        algorithm: SB3DQNAlgorithm,
        episode_cfg: EpisodeConfig,
        stop_event: threading.Event,
        pause_event: threading.Event,
        on_step: Optional[Callable[[StepResult], None]] = None,
        on_episode_end: Optional[Callable[[EpisodeResult], None]] = None,
        on_training_done: Optional[Callable[[TrainingResult], None]] = None,
    ):
        self.algorithm = algorithm
        self.episode_cfg = episode_cfg
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.on_step = on_step
        self.on_episode_end = on_episode_end
        self.on_training_done = on_training_done
        self._latest_frame = None
        self._frame_lock = threading.Lock()

    def latest_frame(self):
        with self._frame_lock:
            return self._latest_frame

    def run_step(self, episode: int, step: int) -> StepResult:
        result = StepResult(episode=episode, step=step, reward=0.0, done=False)
        if self.on_step:
            self.on_step(result)
        return result

    def run_episode(self, episode_index: int, history: List[float]) -> EpisodeResult:
        start = time.perf_counter()
        self.algorithm.update({"total_timesteps": self.episode_cfg.max_steps})
        evaluation = self.algorithm.evaluate_episode(
            self.episode_cfg.max_steps,
            render=True,
            on_frame=self._set_latest_frame,
        )

        with self._frame_lock:
            self._latest_frame = evaluation["frame"]

        total_reward = float(evaluation["return"])
        history.append(total_reward)
        window = self.episode_cfg.moving_average_window
        moving = float(np.mean(history[-window:]))
        duration = time.perf_counter() - start

        result = EpisodeResult(
            episode=episode_index,
            total_reward=total_reward,
            steps=int(evaluation["steps"]),
            duration_sec=duration,
            moving_avg=moving,
        )
        if self.on_episode_end:
            self.on_episode_end(result)
        return result

    def _set_latest_frame(self, frame: Any) -> None:
        with self._frame_lock:
            self._latest_frame = frame

    def run(self, n_episodes: int, initial_history: Optional[List[float]] = None) -> TrainingResult:
        returns = list(initial_history or [])
        moving_avg = []

        for episode in range(len(returns) + 1, len(returns) + n_episodes + 1):
            if self.stop_event.is_set():
                break
            while self.pause_event.is_set() and not self.stop_event.is_set():
                time.sleep(0.02)
            if self.stop_event.is_set():
                break
            result = self.run_episode(episode, returns)
            moving_avg.append(result.moving_avg)

        training_result = TrainingResult(
            finished=not self.stop_event.is_set(),
            episodes_done=len(returns),
            returns=returns,
            moving_avg=moving_avg,
        )
        if self.on_training_done:
            self.on_training_done(training_result)
        return training_result


@dataclass
class TrainingJob:
    job_id: str
    config: JobConfig
    loop: TrainLoop
    thread: Optional[threading.Thread] = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    pause_event: threading.Event = field(default_factory=threading.Event)
    returns: List[float] = field(default_factory=list)
    moving_avg: List[float] = field(default_factory=list)
    visible: bool = True
    status: str = "pending"
    current_episode: int = 0
    epsilon: float = 1.0
    loss: float = float("nan")
    last_steps: int = 0
    last_duration: float = 0.0


class EventBus:
    def __init__(self):
        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        self._queue.put({"type": event_type, "payload": payload, "ts": time.time()})

    def poll(self, max_items: int = 100) -> List[Dict[str, Any]]:
        events = []
        for _ in range(max_items):
            try:
                events.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return events


class CheckpointManager:
    @staticmethod
    def save_job(base_dir: Path, job: TrainingJob, algo: SB3DQNAlgorithm) -> None:
        job_dir = base_dir / job.job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        algo.save(job_dir / "model.zip")
        (job_dir / "config.json").write_text(json.dumps(_job_config_to_json(job.config), indent=2), encoding="utf-8")
        metrics = {
            "returns": job.returns,
            "moving_avg": job.moving_avg,
            "current_episode": job.current_episode,
            "status": job.status,
            "visible": job.visible,
            "epsilon": job.epsilon,
            "loss": job.loss,
        }
        (job_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    @staticmethod
    def load_job(base_dir: Path, job_id: str) -> Dict[str, Any]:
        job_dir = base_dir / job_id
        cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((job_dir / "metrics.json").read_text(encoding="utf-8"))
        return {"config": cfg, "metrics": metrics, "model_path": job_dir / "model.zip"}


def _job_config_to_json(config: JobConfig) -> Dict[str, Any]:
    return asdict(config)


def _job_config_from_json(data: Dict[str, Any]) -> JobConfig:
    net = NetworkConfig(**data["algorithm"]["net"])
    algo = AlgorithmConfig(**{**data["algorithm"], "net": net})
    episodes = EpisodeConfig(**data["episodes"])
    return JobConfig(
        name=data["name"],
        env_id=data["env_id"],
        render_mode=data.get("render_mode", "rgb_array"),
        env_kwargs=data.get("env_kwargs", {}),
        algorithm=algo,
        episodes=episodes,
        visible=data.get("visible", True),
    )


class TrainingManager:
    def __init__(self, event_bus: Optional[EventBus] = None):
        self.event_bus = event_bus or EventBus()
        self.jobs: Dict[str, TrainingJob] = {}
        self.algorithms: Dict[str, SB3DQNAlgorithm] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _clone_algorithm_config(cfg: AlgorithmConfig) -> AlgorithmConfig:
        return AlgorithmConfig(
            algorithm=cfg.algorithm,
            learning_rate=cfg.learning_rate,
            gamma=cfg.gamma,
            buffer_size=cfg.buffer_size,
            batch_size=cfg.batch_size,
            learning_starts=cfg.learning_starts,
            train_freq=cfg.train_freq,
            gradient_steps=cfg.gradient_steps,
            target_update_interval=cfg.target_update_interval,
            exploration_fraction=cfg.exploration_fraction,
            exploration_initial_eps=cfg.exploration_initial_eps,
            exploration_final_eps=cfg.exploration_final_eps,
            net=NetworkConfig(hidden_layers=list(cfg.net.hidden_layers), activation=cfg.net.activation),
        )

    def _build_algorithm(self, cfg: JobConfig) -> SB3DQNAlgorithm:
        adjusted = self._clone_algorithm_config(cfg.algorithm)
        algo_name = adjusted.algorithm.lower()
        if algo_name == "dueling dqn":
            adjusted.net.hidden_layers = adjusted.net.hidden_layers + [128]
        elif algo_name in {"double dqn + prioritized experience replay", "double dqn + per"}:
            adjusted.buffer_size = int(max(adjusted.buffer_size, 150_000))
            adjusted.batch_size = int(max(adjusted.batch_size, 128))
            adjusted.train_freq = int(max(adjusted.train_freq, 4))
        elif algo_name == "d3qn":
            adjusted.net.hidden_layers = adjusted.net.hidden_layers + [128]
            adjusted.target_update_interval = max(200, adjusted.target_update_interval // 2)

        return SB3DQNAlgorithm(
            env_id=cfg.env_id,
            config=adjusted,
            render_mode=cfg.render_mode,
            env_kwargs=cfg.env_kwargs,
        )

    def add_job(self, config: JobConfig) -> str:
        with self._lock:
            job_id = uuid.uuid4().hex[:8]
            stop_event = threading.Event()
            pause_event = threading.Event()
            algo = self._build_algorithm(config)

            def on_episode_end(ep: EpisodeResult, jid=job_id):
                self._on_episode(jid, ep)

            def on_training_done(result: TrainingResult, jid=job_id):
                self._on_training_done(jid, result)

            loop = TrainLoop(
                algorithm=algo,
                episode_cfg=config.episodes,
                stop_event=stop_event,
                pause_event=pause_event,
                on_episode_end=on_episode_end,
                on_training_done=on_training_done,
            )
            job = TrainingJob(
                job_id=job_id,
                config=config,
                loop=loop,
                stop_event=stop_event,
                pause_event=pause_event,
                visible=config.visible,
            )
            self.jobs[job_id] = job
            self.algorithms[job_id] = algo
            self.event_bus.publish("JobCreated", {"job_id": job_id, "name": config.name, "algorithm": config.algorithm.algorithm})
            return job_id

    def _run_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.status = "running"
        self.event_bus.publish("JobStarted", {"job_id": job_id})

        remaining = max(0, job.config.episodes.episodes - job.current_episode)
        if remaining == 0:
            remaining = job.config.episodes.episodes

        try:
            result = job.loop.run(remaining, initial_history=job.returns)
            job.returns = result.returns
            if result.moving_avg:
                window = job.config.episodes.moving_average_window
                job.moving_avg = [
                    float(np.mean(job.returns[max(0, i - window + 1) : i + 1]))
                    for i in range(len(job.returns))
                ]
            job.current_episode = len(job.returns)
            if job.status != "cancelled":
                job.status = "done" if result.finished else "stopped"
        except Exception as exc:
            job.status = "error"
            self.event_bus.publish(
                "Error",
                {
                    "job_id": job_id,
                    "job_name": job.config.name,
                    "message": str(exc),
                    "algorithm": job.config.algorithm.algorithm,
                },
            )

    def start_job(self, job_id: str) -> None:
        job = self.jobs[job_id]
        if job.thread and job.thread.is_alive():
            return
        job.stop_event.clear()
        job.pause_event.clear()
        thread = threading.Thread(target=self._run_job, args=(job_id,), daemon=True)
        job.thread = thread
        thread.start()

    def start_all_pending(self) -> None:
        for job_id, job in list(self.jobs.items()):
            if job.status in {"pending", "done", "stopped", "cancelled"}:
                self.start_job(job_id)

    def pause(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.pause_event.set()
        job.status = "paused"
        self.event_bus.publish("JobPaused", {"job_id": job_id})

    def resume(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.pause_event.clear()
        if job.thread and job.thread.is_alive():
            job.status = "running"
        else:
            self.start_job(job_id)
        self.event_bus.publish("JobResumed", {"job_id": job_id})

    def cancel(self, job_id: str) -> None:
        job = self.jobs[job_id]
        job.stop_event.set()
        job.pause_event.clear()
        job.status = "cancelled"
        self.event_bus.publish("JobCancelled", {"job_id": job_id})

    def cancel_all(self) -> None:
        for job_id in list(self.jobs.keys()):
            self.cancel(job_id)

    def remove(self, job_id: str) -> None:
        if job_id not in self.jobs:
            return
        self.cancel(job_id)
        job = self.jobs[job_id]
        if job.thread and job.thread.is_alive():
            job.thread.join(timeout=1.0)
        algo = self.algorithms.pop(job_id, None)
        if algo:
            algo.close()
        self.jobs.pop(job_id, None)
        self.event_bus.publish("JobRemoved", {"job_id": job_id})

    def toggle_visibility(self, job_id: str) -> bool:
        job = self.jobs[job_id]
        job.visible = not job.visible
        self.event_bus.publish("JobVisibilityChanged", {"job_id": job_id, "visible": job.visible})
        return job.visible

    def _on_episode(self, job_id: str, result: EpisodeResult) -> None:
        if job_id not in self.jobs or job_id not in self.algorithms:
            return
        job = self.jobs[job_id]
        job.current_episode = result.episode
        job.last_steps = result.steps
        job.last_duration = result.duration_sec
        job.returns.append(result.total_reward)
        job.moving_avg.append(result.moving_avg)
        algo = self.algorithms[job_id]
        job.epsilon = float(algo.model.exploration_rate)

        self.event_bus.publish(
            "EpisodeCompleted",
            {
                "job_id": job_id,
                "episode": result.episode,
                "episodes_total": job.config.episodes.episodes,
                "return": result.total_reward,
                "moving_avg": result.moving_avg,
                "epsilon": job.epsilon,
                "loss": job.loss,
                "duration": result.duration_sec,
                "steps": result.steps,
                "algorithm": job.config.algorithm.algorithm,
                "visible": job.visible,
            },
        )

    def _on_training_done(self, job_id: str, result: TrainingResult) -> None:
        if job_id not in self.jobs:
            return
        self.event_bus.publish(
            "TrainingDone",
            {
                "job_id": job_id,
                "finished": result.finished,
                "episodes_done": result.episodes_done,
            },
        )

    def save_all(self, base_dir: Path) -> None:
        base_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for job_id, job in self.jobs.items():
            CheckpointManager.save_job(base_dir, job, self.algorithms[job_id])
            manifest.append(job_id)
        (base_dir / "manifest.json").write_text(json.dumps({"jobs": manifest}, indent=2), encoding="utf-8")

    def load_all(self, base_dir: Path) -> List[str]:
        manifest = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
        loaded_ids = []
        for job_id in manifest["jobs"]:
            payload = CheckpointManager.load_job(base_dir, job_id)
            cfg = _job_config_from_json(payload["config"])
            new_id = self.add_job(cfg)
            job = self.jobs[new_id]
            metrics = payload["metrics"]
            job.returns = metrics.get("returns", [])
            job.moving_avg = metrics.get("moving_avg", [])
            job.current_episode = int(metrics.get("current_episode", len(job.returns)))
            job.visible = bool(metrics.get("visible", True))
            job.status = metrics.get("status", "done")
            self.algorithms[new_id].load(payload["model_path"])
            loaded_ids.append(new_id)
        return loaded_ids

    def get_latest_frame(self, selected_job_id: Optional[str] = None):
        if selected_job_id and selected_job_id in self.jobs:
            job = self.jobs[selected_job_id]
            if job.status == "running":
                return job.loop.latest_frame()

        for job in self.jobs.values():
            if job.status == "running":
                return job.loop.latest_frame()
        return None

    def reset_training(self) -> None:
        for job_id in list(self.jobs.keys()):
            self.cancel(job_id)
            job = self.jobs[job_id]
            job.returns.clear()
            job.moving_avg.clear()
            job.current_episode = 0
            job.status = "pending"

    def create_jobs_for_compare_or_tuning(
        self,
        env_id: str,
        episode_cfg: EpisodeConfig,
        base_algorithm: AlgorithmConfig,
        compare_methods: bool,
        tuning: TuneConfig,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        created = []
        if tuning.enabled:
            for value in tuning.values():
                tuned_algo = self._clone_algorithm_config(base_algorithm)
                if hasattr(tuned_algo, tuning.parameter):
                    setattr(tuned_algo, tuning.parameter, value)
                cfg = JobConfig(
                    name=f"{tuned_algo.algorithm} | {tuning.parameter}={value}",
                    env_id=env_id,
                    env_kwargs=dict(env_kwargs or {}),
                    algorithm=tuned_algo,
                    episodes=episode_cfg,
                )
                created.append(self.add_job(cfg))
            return created

        methods = [base_algorithm.algorithm]
        if compare_methods:
            methods = ["D3QN", "Double DQN + Prioritized Experience Replay", "Dueling DQN"]

        for method in methods:
            algo_cfg = self._clone_algorithm_config(base_algorithm)
            algo_cfg.algorithm = method
            cfg = JobConfig(
                name=method,
                env_id=env_id,
                env_kwargs=dict(env_kwargs or {}),
                algorithm=algo_cfg,
                episodes=episode_cfg,
            )
            created.append(self.add_job(cfg))

        return created
