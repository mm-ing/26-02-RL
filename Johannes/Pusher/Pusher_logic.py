from __future__ import annotations

import csv
import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

try:
    import gymnasium as gym
except Exception as exc:  # pragma: no cover - imported by runtime users
    raise RuntimeError("gymnasium is required for Pusher logic") from exc

try:
    from stable_baselines3 import DDPG, PPO, SAC, TD3
    from stable_baselines3.common.callbacks import BaseCallback
except Exception:
    DDPG = PPO = SAC = TD3 = None  # type: ignore[assignment]

    class BaseCallback:  # type: ignore[no-redef]
        def __init__(self, verbose: int = 0) -> None:
            self.verbose = verbose

        def _on_step(self) -> bool:
            return True


ENV_ID = "Pusher-v5"

POLICIES_CONTINUOUS = ["PPO", "SAC", "TD3", "DDPG"]
DEFAULT_POLICY = "SAC"

SHARED_SPECIFIC_KEYS = [
    "gamma",
    "learning_rate",
    "batch_size",
    "hidden_layer",
    "activation",
    "lr_strategy",
    "min_lr",
    "lr_decay",
]

POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "relu",
        "lr_strategy": "linear",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "n_steps": 1024,
        "n_epochs": 10,
        "ent_coef": 0.0,
    },
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "relu",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 300000,
        "learning_starts": 10000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
    },
    "TD3": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "relu",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 300000,
        "learning_starts": 10000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
    },
    "DDPG": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "relu",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 200000,
        "learning_starts": 10000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
    },
}

POLICY_SPECIFIC_KEYS: Dict[str, List[str]] = {
    policy_name: [
        key for key in defaults.keys() if key not in SHARED_SPECIFIC_KEYS
    ]
    for policy_name, defaults in POLICY_DEFAULTS.items()
}

ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
    "DDPG": DDPG,
}

ACTIVATION_MAP = {
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "elu": torch.nn.ELU,
    "gelu": torch.nn.GELU,
}


@dataclass
class TrainerConfig:
    policy: str = DEFAULT_POLICY
    episodes: int = 3000
    max_steps: int = 200
    seed: Optional[int] = 42
    device: str = "CPU"
    update_rate: int = 10
    frame_stride: int = 2
    enable_animation: bool = True
    env_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "reward_near_weight": 0.5,
            "reward_dist_weight": 1.0,
            "reward_control_weight": 0.1,
        }
    )
    shared_params: Dict[str, Any] = field(default_factory=dict)
    policy_params: Dict[str, Any] = field(default_factory=dict)
    collect_transitions: bool = False
    export_transitions_csv: bool = False
    results_dir: str = "results_csv"
    plots_dir: str = "plots"
    run_label: str = ""
    cpu_threads: Optional[int] = None


class EpisodeFrameCallback(BaseCallback):
    def __init__(self, capture_enabled: bool, frame_stride: int, max_steps: int) -> None:
        super().__init__()
        self.capture_enabled = capture_enabled
        self.frame_stride = max(1, int(frame_stride))
        self.max_steps = max(1, int(max_steps))
        self._step = 0
        self.frames: List[np.ndarray] = []

    def _on_step(self) -> bool:
        self._step += 1
        if self.capture_enabled and (self._step == 1 or self._step % self.frame_stride == 0):
            try:
                frame = self.training_env.render()
                if frame is not None:
                    self.frames.append(np.array(frame))
            except Exception:
                pass
        return self._step < self.max_steps


def parse_hidden_layers(raw_value: Union[str, int, List[int], Tuple[int, ...]], fallback: Tuple[int, ...] = (256, 256)) -> Tuple[int, ...]:
    if isinstance(raw_value, int):
        return (raw_value, raw_value)
    if isinstance(raw_value, (list, tuple)) and raw_value:
        parsed = [int(v) for v in raw_value if int(v) > 0]
        return tuple(parsed) if parsed else fallback
    if not isinstance(raw_value, str):
        return fallback

    text = raw_value.strip()
    if not text:
        return fallback
    if "," not in text:
        try:
            width = int(text)
            return (width, width) if width > 0 else fallback
        except ValueError:
            return fallback

    values: List[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            width = int(part)
        except ValueError:
            return fallback
        if width <= 0:
            return fallback
        values.append(width)
    return tuple(values) if values else fallback


def resolve_device(requested: str) -> str:
    requested_upper = str(requested).strip().upper()
    if requested_upper == "GPU" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_learning_rate(
    base_lr: float,
    strategy: str,
    min_lr: float,
    lr_decay: float,
) -> Union[float, Callable[[float], float]]:
    base_lr = float(base_lr)
    min_lr = max(0.0, float(min_lr))
    lr_decay = float(lr_decay)
    strategy = str(strategy).strip().lower()

    if strategy == "constant":
        return max(base_lr, min_lr)

    if strategy == "linear":
        def linear_schedule(progress_remaining: float) -> float:
            value = progress_remaining * base_lr
            return max(value, min_lr)

        return linear_schedule

    if strategy == "exponential":
        def exp_schedule(progress_remaining: float) -> float:
            progress = max(0.0, min(1.0, 1.0 - progress_remaining))
            decay_multiplier = math.pow(max(1e-6, lr_decay), progress * 1000.0)
            value = base_lr * decay_multiplier
            return max(value, min_lr)

        return exp_schedule

    return max(base_lr, min_lr)


def build_compare_configs(
    base_config: TrainerConfig,
    compare_grid: Dict[str, List[Any]],
) -> List[TrainerConfig]:
    normalized_grid = {k: v for k, v in compare_grid.items() if v}
    if not normalized_grid:
        return [base_config]

    keys = list(normalized_grid.keys())
    values_product = list(product(*[normalized_grid[key] for key in keys]))
    policy_in_grid = "policy" in normalized_grid

    configs: List[TrainerConfig] = []
    for combo_values in values_product:
        combo = dict(zip(keys, combo_values))
        policy_name = str(combo.get("policy", base_config.policy))
        if policy_name not in POLICIES_CONTINUOUS:
            continue

        config = replace(base_config)
        config.policy = policy_name

        if policy_in_grid:
            # Policy compare baseline starts from each policy's defaults.
            config.shared_params = {
                key: POLICY_DEFAULTS[policy_name].get(key, base_config.shared_params.get(key))
                for key in SHARED_SPECIFIC_KEYS
            }
            config.policy_params = {
                key: POLICY_DEFAULTS[policy_name].get(key)
                for key in POLICY_SPECIFIC_KEYS.get(policy_name, [])
            }
        else:
            config.shared_params = dict(base_config.shared_params)
            config.policy_params = dict(base_config.policy_params)

        for key, value in combo.items():
            if key == "policy":
                continue
            if key in SHARED_SPECIFIC_KEYS:
                config.shared_params[key] = value
                continue
            if key in POLICY_SPECIFIC_KEYS.get(policy_name, []):
                config.policy_params[key] = value

        label_parts = [f"{k}={combo[k]}" for k in keys]
        config.run_label = " | ".join(label_parts)
        configs.append(config)

    return configs or [base_config]


class PusherEnvironment:
    def __init__(self, env_params: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None) -> None:
        self.env_params = dict(env_params or {})
        self.render_mode = render_mode

    def make(self):
        return gym.make(ENV_ID, render_mode=self.render_mode, **self.env_params)


class SB3PolicyAgent:
    @staticmethod
    def build_model(policy_name: str, env, shared_params: Dict[str, Any], policy_params: Dict[str, Any], device: str, seed: Optional[int]):
        algo_cls = ALGO_MAP.get(policy_name)
        if algo_cls is None:
            raise ImportError(
                f"Policy '{policy_name}' is unavailable. Install stable-baselines3/sb3-contrib first."
            )

        all_params = dict(POLICY_DEFAULTS[policy_name])
        all_params.update(shared_params)
        all_params.update(policy_params)

        hidden = parse_hidden_layers(all_params.get("hidden_layer", "256,256"))
        activation_name = str(all_params.get("activation", "relu")).strip().lower()
        activation_fn = ACTIVATION_MAP.get(activation_name, torch.nn.ReLU)
        lr_schedule = build_learning_rate(
            base_lr=float(all_params.get("learning_rate", 3e-4)),
            strategy=str(all_params.get("lr_strategy", "constant")),
            min_lr=float(all_params.get("min_lr", 1e-5)),
            lr_decay=float(all_params.get("lr_decay", 0.995)),
        )

        kwargs: Dict[str, Any] = {
            "learning_rate": lr_schedule,
            "gamma": float(all_params.get("gamma", 0.99)),
            "batch_size": int(all_params.get("batch_size", 256)),
            "policy_kwargs": {
                "net_arch": list(hidden),
                "activation_fn": activation_fn,
            },
            "device": device,
            "seed": seed,
            "verbose": 0,
        }

        if policy_name == "PPO":
            kwargs.update(
                {
                    "n_steps": int(all_params.get("n_steps", 1024)),
                    "n_epochs": int(all_params.get("n_epochs", 10)),
                    "ent_coef": float(all_params.get("ent_coef", 0.0)),
                }
            )
            kwargs["n_steps"] = max(kwargs["n_steps"], kwargs["batch_size"])  # keep PPO config valid
        else:
            kwargs.update(
                {
                    "buffer_size": int(all_params.get("buffer_size", 300000)),
                    "learning_starts": int(all_params.get("learning_starts", 10000)),
                    "tau": float(all_params.get("tau", 0.005)),
                    "train_freq": int(all_params.get("train_freq", 1)),
                    "gradient_steps": int(all_params.get("gradient_steps", 1)),
                }
            )
            if policy_name == "TD3":
                kwargs["policy_delay"] = int(all_params.get("policy_delay", 2))

        return algo_cls("MlpPolicy", env, **kwargs)


class PusherTrainer:
    def __init__(
        self,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        env_factory: Optional[Callable[[Optional[str], Dict[str, Any]], Any]] = None,
    ) -> None:
        self.event_sink = event_sink
        self.env_factory = env_factory
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self.eval_points: List[Tuple[int, float]] = []
        self._current_config: Optional[TrainerConfig] = None
        self._best_reward = -float("inf")

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def request_pause(self) -> None:
        self.pause_event.clear()

    def request_resume(self) -> None:
        self.pause_event.set()

    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.set()

    def update_environment(self, new_params: Dict[str, Any]) -> None:
        if self._current_config is None:
            return
        self._current_config.env_params.update(new_params)

    def _make_env(self, render_mode: Optional[str], env_params: Dict[str, Any]):
        if self.env_factory:
            return self.env_factory(render_mode, env_params)
        return PusherEnvironment(env_params=env_params, render_mode=render_mode).make()

    def _extract_current_lr(self, model) -> float:
        lr_schedule = getattr(model, "lr_schedule", None)
        if callable(lr_schedule):
            progress_remaining = float(getattr(model, "_current_progress_remaining", 1.0))
            return float(lr_schedule(progress_remaining))
        if isinstance(lr_schedule, (int, float)):
            return float(lr_schedule)
        return 0.0

    def run_episode(
        self,
        env,
        model,
        deterministic: bool = False,
        max_steps: int = 200,
        collect_transitions: bool = False,
        capture_frames: bool = False,
        frame_stride: int = 2,
    ) -> Dict[str, Any]:
        obs, _ = env.reset()
        total_reward = 0.0
        transitions: List[Dict[str, Any]] = []
        frames: List[np.ndarray] = []

        for step in range(1, max_steps + 1):
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)

            if collect_transitions:
                transitions.append(
                    {
                        "step": step,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "action": np.asarray(action).tolist(),
                        "observation": np.asarray(obs).tolist(),
                        "next_observation": np.asarray(next_obs).tolist(),
                    }
                )

            if capture_frames and (step == 1 or step % max(1, frame_stride) == 0):
                try:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.asarray(frame))
                except Exception:
                    pass

            obs = next_obs
            if terminated or truncated:
                return {
                    "reward": total_reward,
                    "steps": step,
                    "transitions": transitions,
                    "frames": frames,
                }

        return {
            "reward": total_reward,
            "steps": max_steps,
            "transitions": transitions,
            "frames": frames,
        }

    def evaluate_policy(self, model, episodes: int = 1, max_steps: int = 200) -> float:
        if self._current_config is None:
            return 0.0

        env = self._make_env(render_mode=None, env_params=self._current_config.env_params)
        try:
            rewards: List[float] = []
            for _ in range(max(1, episodes)):
                episode = self.run_episode(
                    env=env,
                    model=model,
                    deterministic=True,
                    max_steps=max_steps,
                    collect_transitions=False,
                    capture_frames=False,
                )
                rewards.append(float(episode["reward"]))
            return float(np.mean(rewards)) if rewards else 0.0
        finally:
            env.close()

    def _export_transitions_csv(self, transitions: List[Dict[str, Any]], run_id: str, output_dir: str) -> Optional[str]:
        if not transitions:
            return None
        os.makedirs(output_dir, exist_ok=True)
        file_name = f"{run_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path = os.path.join(output_dir, file_name)

        columns = [
            "run_id",
            "episode",
            "step",
            "reward",
            "terminated",
            "truncated",
            "action",
            "observation",
            "next_observation",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=columns)
            writer.writeheader()
            for row in transitions:
                writer.writerow(
                    {
                        "run_id": run_id,
                        "episode": row["episode"],
                        "step": row["step"],
                        "reward": row["reward"],
                        "terminated": row["terminated"],
                        "truncated": row["truncated"],
                        "action": json.dumps(row["action"]),
                        "observation": json.dumps(row["observation"]),
                        "next_observation": json.dumps(row["next_observation"]),
                    }
                )
        return csv_path

    def _snapshot_config(self, config: TrainerConfig) -> Dict[str, Any]:
        return {
            "policy": config.policy,
            "device": config.device,
            "episodes": int(config.episodes),
            "max_steps": int(config.max_steps),
            "env_params": dict(config.env_params),
            "shared_params": dict(config.shared_params),
            "policy_params": dict(config.policy_params),
            "run_label": config.run_label,
        }

    def train(self, config: TrainerConfig) -> Dict[str, Any]:
        self._current_config = config
        self.cancel_event.clear()
        self.pause_event.set()
        self.eval_points = []
        self._best_reward = -float("inf")

        run_id = f"{config.policy.lower()}_{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
        transitions_for_export: List[Dict[str, Any]] = []
        config_snapshot = self._snapshot_config(config)

        render_mode = "rgb_array" if config.enable_animation else None
        env = self._make_env(render_mode=render_mode, env_params=config.env_params)

        try:
            if resolve_device(config.device) == "cpu" and config.cpu_threads is not None:
                torch.set_num_threads(max(1, int(config.cpu_threads)))

            model = SB3PolicyAgent.build_model(
                policy_name=config.policy,
                env=env,
                shared_params=config.shared_params,
                policy_params=config.policy_params,
                device=resolve_device(config.device),
                seed=config.seed,
            )

            rewards: List[float] = []
            episodes_completed = 0

            for episode_idx in range(1, int(config.episodes) + 1):
                if self.cancel_event.is_set():
                    break
                self.pause_event.wait()

                capture_frames = bool(
                    config.enable_animation
                    and (episode_idx == config.episodes or episode_idx % max(1, config.update_rate) == 0)
                )
                callback = EpisodeFrameCallback(
                    capture_enabled=capture_frames,
                    frame_stride=config.frame_stride,
                    max_steps=config.max_steps,
                )
                model.learn(
                    total_timesteps=int(config.max_steps),
                    reset_num_timesteps=False,
                    callback=callback,
                    progress_bar=False,
                )

                episode_data = self.run_episode(
                    env=env,
                    model=model,
                    deterministic=False,
                    max_steps=int(config.max_steps),
                    collect_transitions=bool(config.collect_transitions or config.export_transitions_csv),
                    capture_frames=False,
                    frame_stride=config.frame_stride,
                )

                self._emit(
                    {
                        "type": "step",
                        "run_id": run_id,
                        "episode": episode_idx,
                        "step": int(episode_data["steps"]),
                        "max_steps": int(config.max_steps),
                    }
                )

                episode_reward = float(episode_data["reward"])
                rewards.append(episode_reward)
                moving_average = float(np.mean(rewards[-25:]))
                self._best_reward = max(self._best_reward, episode_reward)

                if config.collect_transitions or config.export_transitions_csv:
                    for tr in episode_data["transitions"]:
                        row = dict(tr)
                        row["episode"] = episode_idx
                        transitions_for_export.append(row)

                if episode_idx % 10 == 0 or episode_idx == config.episodes:
                    score = self.evaluate_policy(model, episodes=1, max_steps=int(config.max_steps))
                    self.eval_points.append((episode_idx, score))

                payload = {
                    "type": "episode",
                    "run_id": run_id,
                    "run_label": config.run_label,
                    "policy": config.policy,
                    "episode": episode_idx,
                    "episodes": int(config.episodes),
                    "reward": episode_reward,
                    "moving_average": moving_average,
                    "eval_points": list(self.eval_points),
                    "steps": int(episode_data["steps"]),
                    "epsilon": 0.0,
                    "lr": self._extract_current_lr(model),
                    "best_reward": self._best_reward,
                    "render_state": "updated" if callback.frames else "idle",
                    "frames": callback.frames,
                    "config_snapshot": config_snapshot,
                }
                if callback.frames:
                    payload["frame"] = callback.frames[-1]

                self._emit(payload)
                episodes_completed = episode_idx

            csv_path = None
            if config.export_transitions_csv:
                csv_path = self._export_transitions_csv(transitions_for_export, run_id, config.results_dir)

            final_payload = {
                "type": "training_done",
                "run_id": run_id,
                "run_label": config.run_label,
                "policy": config.policy,
                "episodes_completed": episodes_completed,
                "episodes": int(config.episodes),
                "best_reward": self._best_reward,
                "eval_points": list(self.eval_points),
                "canceled": self.cancel_event.is_set(),
                "csv_path": csv_path,
                "config_snapshot": config_snapshot,
            }
            self._emit(final_payload)
            return final_payload

        except Exception as exc:
            error_payload = {
                "type": "error",
                "run_id": run_id,
                "message": str(exc),
            }
            self._emit(error_payload)
            return error_payload
        finally:
            env.close()
