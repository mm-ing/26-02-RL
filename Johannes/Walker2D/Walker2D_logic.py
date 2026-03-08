from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import csv
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3


EventSink = Optional[Callable[[Dict[str, Any]], None]]


SHARED_DEFAULTS: Dict[str, Any] = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "batch_size": 256,
    "hidden_layer": 256,
    "lr_strategy": "constant",
    "min_lr": 1e-5,
    "lr_decay": 1.0,
}


POLICY_SHARED_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 128,
        "hidden_layer": 256,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
    },
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": 256,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
    },
    "TD3": {
        "gamma": 0.99,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "hidden_layer": 256,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
    },
}


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def parse_hidden_layers(value: Any, default: Any) -> List[int]:
    def parse_tokens(raw: Any) -> List[int]:
        if isinstance(raw, (list, tuple)):
            tokens = list(raw)
        elif isinstance(raw, str):
            cleaned = raw.strip()
            if not cleaned:
                return []
            tokens = [part.strip() for part in cleaned.split(",") if part.strip()]
        else:
            tokens = [raw]

        layers: List[int] = []
        for token in tokens:
            try:
                parsed = int(float(token))
            except Exception:
                continue
            layers.append(max(32, parsed))
        return layers

    layers = parse_tokens(value)
    if not layers:
        layers = parse_tokens(default)
    if not layers:
        layers = [256]
    if len(layers) == 1:
        return [layers[0], layers[0]]
    return layers


def make_lr_schedule(strategy: str, base_lr: float, min_lr: float, decay: float):
    strategy = str(strategy).lower().strip()
    if strategy == "constant":
        return base_lr

    def schedule(progress_remaining: float) -> float:
        progress = max(0.0, min(1.0, float(progress_remaining)))
        if strategy == "linear":
            return max(min_lr, min_lr + (base_lr - min_lr) * progress)
        if strategy == "exponential":
            steps = int((1.0 - progress) * 1000)
            return max(min_lr, base_lr * (decay ** steps))
        return base_lr

    return schedule


POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "n_steps": 2048,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    },
    "SAC": {
        "buffer_size": 500_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
    },
    "TD3": {
        "buffer_size": 500_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "policy_delay": 2,
        "train_freq": 1,
        "gradient_steps": 1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
    },
}


@dataclass
class Walker2DEnvConfig:
    env_id: str = "Walker2d-v5"
    render_mode: Optional[str] = None
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 1e-3
    healthy_reward: float = 1.0
    terminate_when_unhealthy: bool = True
    healthy_z_range: Tuple[float, float] = (0.8, 2.0)
    healthy_angle_range: Tuple[float, float] = (-1.0, 1.0)
    reset_noise_scale: float = 5e-3
    exclude_current_positions_from_observation: bool = True

    def gym_kwargs(self) -> Dict[str, Any]:
        return {
            "forward_reward_weight": self.forward_reward_weight,
            "ctrl_cost_weight": self.ctrl_cost_weight,
            "healthy_reward": self.healthy_reward,
            "terminate_when_unhealthy": self.terminate_when_unhealthy,
            "healthy_z_range": self.healthy_z_range,
            "healthy_angle_range": self.healthy_angle_range,
            "reset_noise_scale": self.reset_noise_scale,
            "exclude_current_positions_from_observation": self.exclude_current_positions_from_observation,
        }


@dataclass
class TrainConfig:
    policy_name: str = "PPO"
    episodes: int = 1000
    max_steps: int = 1000
    update_rate_episodes: int = 1
    frame_stride: int = 2
    moving_average_values: int = 20
    deterministic_eval_every: int = 10
    deterministic_eval_max_steps: int = 1000
    rollout_full_capture_steps: int = 1000
    low_overhead_animation: bool = False
    animation_on: bool = True
    collect_transitions: bool = False
    device: str = "CPU"
    shared_params: Dict[str, Any] = field(default_factory=lambda: dict(SHARED_DEFAULTS))
    specific_params: Dict[str, Any] = field(default_factory=dict)
    run_id: str = "single"


class Walker2DEnvironment:
    def __init__(self, config: Walker2DEnvConfig) -> None:
        self.config = config
        self._env: Optional[gym.Env] = None

    def make(self, render_mode: Optional[str] = None) -> gym.Env:
        return gym.make(
            self.config.env_id,
            render_mode=render_mode if render_mode is not None else self.config.render_mode,
            **self.config.gym_kwargs(),
        )

    def get_or_create(self) -> gym.Env:
        if self._env is None:
            self._env = self.make()
        return self._env

    def rebuild(self, render_mode: Optional[str] = None) -> gym.Env:
        self.close()
        self._env = self.make(render_mode=render_mode)
        return self._env

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


class Walker2DTrainer:
    def __init__(
        self,
        env_config: Optional[Walker2DEnvConfig] = None,
        train_config: Optional[TrainConfig] = None,
        event_sink: EventSink = None,
    ) -> None:
        self.env_config = env_config or Walker2DEnvConfig()
        self.train_config = train_config or TrainConfig()
        self.event_sink = event_sink
        self.environment = Walker2DEnvironment(self.env_config)
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self._model = None
        self._training_rewards: List[float] = []
        self._transition_buffer: List[Dict[str, Any]] = []

    @property
    def training_rewards(self) -> List[float]:
        return list(self._training_rewards)

    @property
    def transitions(self) -> List[Dict[str, Any]]:
        return list(self._transition_buffer)

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def _resolved_device(self) -> str:
        requested = str(self.train_config.device).strip().upper()
        if requested == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _build_model(self):
        policy = self.train_config.policy_name
        env = self.environment.get_or_create()
        shared = self.train_config.shared_params
        policy_shared = POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)
        specific = self.train_config.specific_params
        base_lr = _safe_float(shared.get("learning_rate", policy_shared["learning_rate"]), policy_shared["learning_rate"])
        min_lr = _safe_float(shared.get("min_lr", policy_shared["min_lr"]), policy_shared["min_lr"])
        lr_decay = _safe_float(shared.get("lr_decay", policy_shared["lr_decay"]), policy_shared["lr_decay"])
        lr_strategy = str(shared.get("lr_strategy", policy_shared["lr_strategy"])).lower().strip()
        learning_rate = make_lr_schedule(lr_strategy, base_lr, min_lr, lr_decay)
        gamma = _safe_float(shared.get("gamma", policy_shared["gamma"]), policy_shared["gamma"])
        batch_size = _safe_int(shared.get("batch_size", policy_shared["batch_size"]), policy_shared["batch_size"])
        hidden_layers = parse_hidden_layers(shared.get("hidden_layer", policy_shared["hidden_layer"]), policy_shared["hidden_layer"])
        device = self._resolved_device()
        policy_kwargs = {"net_arch": hidden_layers}

        if policy == "PPO":
            n_steps = int(specific.get("n_steps", POLICY_DEFAULTS["PPO"]["n_steps"]))
            if n_steps < batch_size:
                n_steps = batch_size
            if n_steps % batch_size != 0:
                n_steps = ((n_steps // batch_size) + 1) * batch_size
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
                n_steps=n_steps,
                gae_lambda=float(specific.get("gae_lambda", POLICY_DEFAULTS["PPO"]["gae_lambda"])),
                clip_range=float(specific.get("clip_range", POLICY_DEFAULTS["PPO"]["clip_range"])),
                ent_coef=float(specific.get("ent_coef", POLICY_DEFAULTS["PPO"]["ent_coef"])),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
            )
        elif policy == "SAC":
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(specific.get("buffer_size", POLICY_DEFAULTS["SAC"]["buffer_size"])),
                learning_starts=int(specific.get("learning_starts", POLICY_DEFAULTS["SAC"]["learning_starts"])),
                tau=float(specific.get("tau", POLICY_DEFAULTS["SAC"]["tau"])),
                train_freq=int(specific.get("train_freq", POLICY_DEFAULTS["SAC"]["train_freq"])),
                gradient_steps=int(specific.get("gradient_steps", POLICY_DEFAULTS["SAC"]["gradient_steps"])),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
            )
        elif policy == "TD3":
            model = TD3(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(specific.get("buffer_size", POLICY_DEFAULTS["TD3"]["buffer_size"])),
                learning_starts=int(specific.get("learning_starts", POLICY_DEFAULTS["TD3"]["learning_starts"])),
                tau=float(specific.get("tau", POLICY_DEFAULTS["TD3"]["tau"])),
                policy_delay=int(specific.get("policy_delay", POLICY_DEFAULTS["TD3"]["policy_delay"])),
                train_freq=int(specific.get("train_freq", POLICY_DEFAULTS["TD3"]["train_freq"])),
                gradient_steps=int(specific.get("gradient_steps", POLICY_DEFAULTS["TD3"]["gradient_steps"])),
                target_policy_noise=float(specific.get("target_policy_noise", POLICY_DEFAULTS["TD3"]["target_policy_noise"])),
                target_noise_clip=float(specific.get("target_noise_clip", POLICY_DEFAULTS["TD3"]["target_noise_clip"])),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=device,
            )
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        self._model = model
        return model

    def set_pause(self, paused: bool) -> None:
        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.set()

    def rebuild_environment(self, env_config: Optional[Walker2DEnvConfig] = None, render_mode: Optional[str] = None) -> None:
        if env_config is not None:
            self.env_config = env_config
            self.environment.config = env_config
        self.environment.rebuild(render_mode=render_mode)

    def update_environment(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.env_config, key):
                setattr(self.env_config, key, value)
        self.environment.rebuild(render_mode=self.env_config.render_mode)

    def run_episode(
        self,
        model=None,
        deterministic: bool = False,
        collect_transitions: bool = False,
        max_steps: Optional[int] = None,
        render: bool = False,
        frame_stride: int = 2,
        rollout_full_capture_steps: int = 120,
        emit_step_events: bool = True,
    ) -> Dict[str, Any]:
        model = model or self._model
        episode_env = self.environment.make(render_mode="rgb_array" if render else None)
        observation, _ = episode_env.reset()
        reward_total = 0.0
        steps = 0
        transitions: List[Dict[str, Any]] = []
        frames: List[np.ndarray] = []
        max_steps = int(max_steps or self.train_config.max_steps)
        stride = max(1, int(frame_stride))
        capture_limit = max_steps

        try:
            while steps < max_steps:
                self.pause_event.wait()
                if self.cancel_event.is_set():
                    break

                if model is None:
                    action = episode_env.action_space.sample()
                else:
                    action, _ = model.predict(observation, deterministic=deterministic)
                next_observation, reward, terminated, truncated, _ = episode_env.step(action)

                reward_total += float(reward)
                if collect_transitions:
                    transitions.append(
                        {
                            "step": steps,
                            "reward": float(reward),
                            "done": bool(terminated or truncated),
                        }
                    )

                if emit_step_events:
                    self._emit(
                        {
                            "type": "step",
                            "run_id": self.train_config.run_id,
                            "step": steps,
                            "max_steps": max_steps,
                            "reward": float(reward),
                            "done": bool(terminated or truncated),
                        }
                    )

                if render and steps < capture_limit and (steps == 0 or steps % stride == 0):
                    frame = episode_env.render()
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)

                observation = next_observation
                steps += 1

                if terminated or truncated:
                    break
        finally:
            episode_env.close()

        return {
            "reward": reward_total,
            "steps": steps,
            "transitions": transitions,
            "frames": frames,
        }

    def evaluate_policy(self, episodes: int = 1) -> float:
        episodes = max(1, int(episodes))
        if self._model is None:
            return 0.0
        values = []
        for _ in range(episodes):
            result = self.run_episode(
                model=self._model,
                deterministic=True,
                collect_transitions=False,
                max_steps=self.train_config.max_steps,
                render=False,
                frame_stride=self.train_config.frame_stride,
                rollout_full_capture_steps=self.train_config.rollout_full_capture_steps,
                emit_step_events=False,
            )
            values.append(float(result["reward"]))
        return float(np.mean(values)) if values else 0.0

    def train(self) -> Dict[str, Any]:
        cfg = self.train_config
        self.cancel_event.clear()
        self._training_rewards.clear()
        self._transition_buffer.clear()

        model = self._build_model()
        best_reward = float("-inf")
        eval_points: List[Tuple[int, float]] = []

        try:
            for episode in range(1, int(cfg.episodes) + 1):
                self.pause_event.wait()
                if self.cancel_event.is_set():
                    break

                model.learn(total_timesteps=int(cfg.max_steps), reset_num_timesteps=False, progress_bar=False)

                should_render = bool(cfg.animation_on) and (
                    episode == cfg.episodes or episode % max(1, int(cfg.update_rate_episodes)) == 0
                )
                result = self.run_episode(
                    model=model,
                    deterministic=False,
                    collect_transitions=bool(cfg.collect_transitions),
                    max_steps=cfg.max_steps,
                    render=should_render and not cfg.low_overhead_animation,
                    frame_stride=cfg.frame_stride,
                    rollout_full_capture_steps=cfg.rollout_full_capture_steps,
                    emit_step_events=False,
                )

                if cfg.collect_transitions and result["transitions"]:
                    self._transition_buffer.extend(result["transitions"])

                reward = float(result["reward"])
                best_reward = max(best_reward, reward)
                self._training_rewards.append(reward)
                window = max(1, int(cfg.moving_average_values))
                moving_average = float(np.mean(self._training_rewards[-window:]))

                if episode % max(1, int(cfg.deterministic_eval_every)) == 0:
                    eval_reward = self.evaluate_policy(episodes=1)
                    eval_points.append((episode, eval_reward))

                payload = {
                    "type": "episode",
                    "run_id": cfg.run_id,
                    "episode": episode,
                    "episodes": int(cfg.episodes),
                    "reward": reward,
                    "moving_average": moving_average,
                    "eval_points": list(eval_points),
                    "steps": int(result["steps"]),
                    "epsilon": "n/a",
                    "lr": float(cfg.shared_params.get("learning_rate", SHARED_DEFAULTS["learning_rate"])),
                    "best_reward": best_reward,
                    "render_state": "on" if should_render else "off",
                    "frames": result["frames"],
                    "frame": result["frames"][0] if result["frames"] else None,
                }
                self._emit(payload)

            status = "canceled" if self.cancel_event.is_set() else "completed"
            done_payload = {
                "type": "training_done",
                "run_id": cfg.run_id,
                "status": status,
                "episodes_done": len(self._training_rewards),
                "best_reward": best_reward if self._training_rewards else 0.0,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self._emit(done_payload)
            return done_payload
        except Exception as exc:
            error_payload = {
                "type": "error",
                "run_id": cfg.run_id,
                "message": str(exc),
            }
            self._emit(error_payload)
            raise
        finally:
            self.environment.close()

    def export_transitions_csv(self, output_dir: Path, filename_prefix: str = "walker2d_samples") -> Optional[Path]:
        if not self._transition_buffer:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"{filename_prefix}_{timestamp}.csv"
        fieldnames = sorted({key for item in self._transition_buffer for key in item.keys()})
        with output_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._transition_buffer)
        return output_path
