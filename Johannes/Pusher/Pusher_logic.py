from __future__ import annotations

import csv
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DDPG, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.type_aliases import Schedule

PROJECT_NAME = "Pusher"
ENV_ID = "Pusher-v5"

EventSink = Callable[[Dict[str, Any]], None]


@dataclass(frozen=True)
class EnvironmentConfig:
    reward_near_weight: float = 0.5
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1
    render_mode: Optional[str] = None


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layer: str = "256"
    activation: str = "relu"


@dataclass(frozen=True)
class LearningRateConfig:
    lr_strategy: str = "constant"
    learning_rate: float = 3e-4
    min_lr: float = 1e-5
    lr_decay: float = 0.999


@dataclass(frozen=True)
class TrainerConfig:
    policy_name: str = "SAC"
    episodes: int = 3000
    max_steps: int = 200
    update_rate: int = 5
    frame_stride: int = 2
    deterministic_eval_every: int = 10
    deterministic_eval_episodes: int = 1
    seed: Optional[int] = 42
    collect_transitions: bool = False
    export_csv: bool = False
    device: str = "CPU"
    session_id: str = "default-session"
    run_id: str = "default-run"
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    policy_params: Dict[str, Any] = field(default_factory=dict)


class PusherEnvWrapper:
    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config

    @property
    def config(self) -> EnvironmentConfig:
        return self._config

    def rebuild(self, config: EnvironmentConfig) -> None:
        self._config = config

    def make_env(self) -> gym.Env:
        kwargs: Dict[str, Any] = {
            "reward_near_weight": self._config.reward_near_weight,
            "reward_dist_weight": self._config.reward_dist_weight,
            "reward_control_weight": self._config.reward_control_weight,
        }
        if self._config.render_mode:
            kwargs["render_mode"] = self._config.render_mode
        return gym.make(ENV_ID, **kwargs)


class _TrainingProgressCallback(BaseCallback):
    def __init__(self, pause_event: threading.Event, cancel_event: threading.Event, step_budget: int) -> None:
        super().__init__()
        self.pause_event = pause_event
        self.cancel_event = cancel_event
        self.step_budget = max(1, int(step_budget))
        self.steps_seen = 0

    def _on_step(self) -> bool:
        if self.cancel_event.is_set():
            return False
        self.pause_event.wait()
        self.steps_seen += 1
        return self.steps_seen <= self.step_budget


class SB3PolicyFactory:
    POLICY_MAP = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
        "DDPG": DDPG,
    }

    ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
        "elu": torch.nn.ELU,
        "leaky_relu": torch.nn.LeakyReLU,
    }

    @staticmethod
    def policy_defaults(policy_name: str) -> Dict[str, Any]:
        defaults = {
            "PPO": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "n_steps": 1024,
                "ent_coef": 0.0,
            },
            "SAC": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "tau": 0.005,
            },
            "TD3": {
                "learning_rate": 3e-4,
                "gamma": 0.99,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "tau": 0.005,
                "policy_delay": 2,
            },
            "DDPG": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch_size": 256,
                "buffer_size": 200_000,
                "learning_starts": 5_000,
                "tau": 0.005,
            },
        }
        return dict(defaults.get(policy_name, defaults["SAC"]))

    @classmethod
    def _parse_net_arch(cls, hidden_layer: str) -> List[int]:
        try:
            chunks = [int(part.strip()) for part in str(hidden_layer).split(",") if part.strip()]
        except ValueError:
            chunks = []

        if not chunks:
            return [256, 256]
        if len(chunks) == 1:
            return [chunks[0], chunks[0]]
        return chunks

    @classmethod
    def _build_schedule(cls, cfg: LearningRateConfig) -> float | Schedule:
        strategy = cfg.lr_strategy.lower().strip()
        base_lr = float(cfg.learning_rate)
        min_lr = float(cfg.min_lr)
        decay = float(cfg.lr_decay)

        if strategy == "constant":
            return base_lr

        if strategy == "linear":
            def linear_schedule(progress_remaining: float) -> float:
                return max(min_lr, min_lr + (base_lr - min_lr) * progress_remaining)
            return linear_schedule

        def exp_schedule(progress_remaining: float) -> float:
            progress_done = 1.0 - progress_remaining
            value = base_lr * (decay ** (progress_done * 1000.0))
            return max(min_lr, value)

        return exp_schedule

    @classmethod
    def create_model(
        cls,
        policy_name: str,
        env: gym.Env,
        network_cfg: NetworkConfig,
        lr_cfg: LearningRateConfig,
        policy_params: Optional[Dict[str, Any]] = None,
        device: str = "cpu",
        seed: Optional[int] = None,
    ) -> Any:
        if policy_name not in cls.POLICY_MAP:
            raise ValueError(f"Unsupported policy '{policy_name}'.")

        algo_cls = cls.POLICY_MAP[policy_name]
        defaults = cls.policy_defaults(policy_name)
        merged: Dict[str, Any] = {**defaults, **(policy_params or {})}

        net_arch = cls._parse_net_arch(network_cfg.hidden_layer)
        activation_fn = cls.ACTIVATIONS.get(network_cfg.activation.lower().strip(), torch.nn.ReLU)

        policy_kwargs = {
            "activation_fn": activation_fn,
            "net_arch": net_arch,
        }

        merged["policy_kwargs"] = policy_kwargs
        merged["learning_rate"] = cls._build_schedule(lr_cfg)

        if policy_name == "PPO":
            n_steps = int(merged.get("n_steps", 1024))
            batch_size = int(merged.get("batch_size", 64))
            if n_steps < batch_size:
                n_steps = batch_size
            if n_steps % batch_size != 0:
                n_steps = ((n_steps // batch_size) + 1) * batch_size
            merged["n_steps"] = n_steps
            merged["batch_size"] = batch_size

        model = algo_cls(
            "MlpPolicy",
            env,
            seed=seed,
            device=device,
            verbose=0,
            **merged,
        )
        return model


class PusherTrainer:
    def __init__(self, env_wrapper: PusherEnvWrapper, event_sink: Optional[EventSink] = None) -> None:
        self.env_wrapper = env_wrapper
        self.event_sink = event_sink
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self._latest_render_state: Optional[np.ndarray] = None
        self._transitions: List[Dict[str, Any]] = []

    @staticmethod
    def _resolved_device(requested: str) -> str:
        if str(requested).upper() == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        self.pause_event.set()

    def update_environment(self, config: EnvironmentConfig) -> None:
        self.env_wrapper.rebuild(config)

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def _capture_frame(self, env: gym.Env, episode_frames: List[np.ndarray], stride: int, step_idx: int) -> None:
        if step_idx != 0 and step_idx % max(1, stride) != 0:
            return
        frame = env.render()
        if isinstance(frame, np.ndarray):
            self._latest_render_state = frame
            episode_frames.append(frame)

    def run_episode(
        self,
        model: Any,
        env: gym.Env,
        max_steps: int,
        deterministic: bool,
        frame_stride: int,
        collect_transitions: bool,
    ) -> Tuple[float, int, List[np.ndarray]]:
        obs, _ = env.reset()
        total_reward = 0.0
        episode_frames: List[np.ndarray] = []

        for step_idx in range(max_steps):
            self.pause_event.wait()
            if self.cancel_event.is_set():
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += float(reward)
            self._capture_frame(env, episode_frames, frame_stride, step_idx)

            if collect_transitions:
                self._transitions.append(
                    {
                        "step": step_idx,
                        "reward": float(reward),
                        "done": bool(done),
                    }
                )

            obs = next_obs
            if done:
                return total_reward, step_idx + 1, episode_frames

        return total_reward, max_steps, episode_frames

    def evaluate_policy(self, model: Any, episodes: int, max_steps: int) -> float:
        eval_env = self.env_wrapper.make_env()
        try:
            rewards: List[float] = []
            for _ in range(max(1, episodes)):
                reward, _, _ = self.run_episode(
                    model=model,
                    env=eval_env,
                    max_steps=max_steps,
                    deterministic=True,
                    frame_stride=1,
                    collect_transitions=False,
                )
                rewards.append(reward)
            return float(np.mean(rewards)) if rewards else 0.0
        finally:
            eval_env.close()

    def _export_transitions_csv(self, run_id: str) -> Optional[str]:
        if not self._transitions:
            return None

        os.makedirs("results_csv", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{PROJECT_NAME}_{run_id}_{timestamp}_transitions.csv"
        output_path = os.path.join("results_csv", filename)

        with open(output_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["step", "reward", "done"])
            writer.writeheader()
            writer.writerows(self._transitions)

        return output_path

    def train(self, config: TrainerConfig) -> Dict[str, Any]:
        self.cancel_event.clear()
        self.pause_event.set()
        self._transitions.clear()

        train_env = self.env_wrapper.make_env()
        model = SB3PolicyFactory.create_model(
            policy_name=config.policy_name,
            env=train_env,
            network_cfg=config.network,
            lr_cfg=config.lr,
            policy_params=config.policy_params,
            device=self._resolved_device(config.device),
            seed=config.seed,
        )

        rewards: List[float] = []
        moving_average: List[float] = []
        eval_points: List[Tuple[int, float]] = []
        best_reward = float("-inf")

        try:
            for episode in range(1, config.episodes + 1):
                self.pause_event.wait()
                if self.cancel_event.is_set():
                    break

                # Roll out one episode for metrics and optional animation frame capture.
                reward, steps, frames = self.run_episode(
                    model=model,
                    env=train_env,
                    max_steps=config.max_steps,
                    deterministic=False,
                    frame_stride=max(1, config.frame_stride),
                    collect_transitions=config.collect_transitions,
                )

                callback = _TrainingProgressCallback(
                    pause_event=self.pause_event,
                    cancel_event=self.cancel_event,
                    step_budget=max(1, steps),
                )
                model.learn(total_timesteps=max(1, steps), reset_num_timesteps=False, callback=callback)

                rewards.append(reward)
                window = rewards[-20:]
                moving = float(np.mean(window)) if window else reward
                moving_average.append(moving)
                best_reward = max(best_reward, reward)

                if episode % max(1, config.deterministic_eval_every) == 0:
                    eval_reward = self.evaluate_policy(
                        model=model,
                        episodes=config.deterministic_eval_episodes,
                        max_steps=config.max_steps,
                    )
                    eval_points.append((episode, eval_reward))

                should_emit_frames = episode % max(1, config.update_rate) == 0 or episode == config.episodes
                payload: Dict[str, Any] = {
                    "type": "episode",
                    "session_id": config.session_id,
                    "run_id": config.run_id,
                    "episode": episode,
                    "episodes": config.episodes,
                    "reward": reward,
                    "moving_average": moving,
                    "eval_points": list(eval_points),
                    "steps": steps,
                    "epsilon": None,
                    "lr": config.lr.learning_rate,
                    "best_reward": best_reward,
                    "render_state": self._latest_render_state,
                }
                if should_emit_frames:
                    payload["frames"] = frames
                    payload["frame"] = frames[-1] if frames else None
                self._emit(payload)

                if self.cancel_event.is_set():
                    break

            csv_path = self._export_transitions_csv(config.run_id) if config.export_csv else None
            final_payload = {
                "type": "training_done",
                "session_id": config.session_id,
                "run_id": config.run_id,
                "stopped": self.cancel_event.is_set(),
                "episodes_completed": len(rewards),
                "best_reward": best_reward if rewards else None,
                "csv_path": csv_path,
                "finished_at": time.time(),
            }
            self._emit(final_payload)
            return final_payload
        except Exception as exc:  # pragma: no cover - safety path for runtime GUI failures
            payload = {
                "type": "error",
                "session_id": config.session_id,
                "run_id": config.run_id,
                "message": str(exc),
            }
            self._emit(payload)
            raise
        finally:
            train_env.close()


def build_default_trainer(event_sink: Optional[EventSink] = None) -> PusherTrainer:
    wrapper = PusherEnvWrapper(EnvironmentConfig(render_mode="rgb_array"))
    return PusherTrainer(wrapper, event_sink=event_sink)
