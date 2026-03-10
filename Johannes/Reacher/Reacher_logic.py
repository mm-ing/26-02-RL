from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Event
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import csv
import math
import os
import random

try:
    import gymnasium as gym
except Exception:  # pragma: no cover - optional runtime dependency
    gym = None

try:
    import torch
except Exception:  # pragma: no cover - optional runtime dependency
    torch = None

try:
    from stable_baselines3 import PPO, SAC, TD3
except Exception:  # pragma: no cover - optional runtime dependency
    PPO = None
    SAC = None
    TD3 = None


POLICY_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "TD3": TD3,
}


@dataclass
class EnvironmentConfig:
    env_id: str = "Reacher-v5"
    reward_dist_weight: float = 1.0
    reward_control_weight: float = 0.1
    render_enabled: bool = True


@dataclass
class TrainingConfig:
    policy: str = "SAC"
    episodes: int = 3000
    max_steps: int = 200
    gamma: float = 0.99
    learning_rate: float = 3e-4
    batch_size: int = 256
    hidden_layer: str = "256,256"
    activation: str = "ReLU"
    lr_strategy: str = "constant"
    min_lr: float = 1e-5
    lr_decay: float = 0.995
    train_freq: int = 1
    gradient_steps: int = 1
    buffer_size: int = 100000
    learning_starts: int = 1000
    tau: float = 0.005
    policy_delay: int = 2
    update_rate_episodes: int = 1
    frame_stride: int = 2
    animation_fps: int = 30
    moving_average_window: int = 20
    eval_rollout_on: bool = False
    device: str = "CPU"
    run_id: str = "single"


def parse_hidden_layers(value: str, fallback: Sequence[int] = (256, 256)) -> List[int]:
    try:
        stripped = value.strip()
        if not stripped:
            return list(fallback)
        if "," not in stripped:
            width = int(stripped)
            if width <= 0:
                return list(fallback)
            return [width, width]
        parsed = [int(v.strip()) for v in stripped.split(",") if v.strip()]
        if not parsed or any(v <= 0 for v in parsed):
            return list(fallback)
        return parsed
    except Exception:
        return list(fallback)


def make_lr_schedule(base_lr: float, strategy: str, min_lr: float, decay: float) -> Callable[[float], float]:
    strategy = (strategy or "constant").lower()

    if strategy == "linear":
        def linear(progress_remaining: float) -> float:
            val = min_lr + (base_lr - min_lr) * max(0.0, progress_remaining)
            return float(max(min_lr, val))

        return linear

    if strategy == "exponential":
        def expo(progress_remaining: float) -> float:
            # Progress goes from 1 -> 0 in SB3.
            elapsed = 1.0 - max(0.0, min(1.0, progress_remaining))
            val = base_lr * (decay ** (elapsed * 100.0))
            return float(max(min_lr, val))

        return expo

    def constant(_: float) -> float:
        return float(max(min_lr, base_lr))

    return constant


class ReacherEnvironment:
    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.env = None
        self._build_env()

    def _build_env(self) -> None:
        if gym is None:
            self.env = None
            return
        render_mode = "rgb_array" if self.config.render_enabled else None
        kwargs = {
            "reward_dist_weight": self.config.reward_dist_weight,
            "reward_control_weight": self.config.reward_control_weight,
        }
        self.env = gym.make(self.config.env_id, render_mode=render_mode, **kwargs)

    def rebuild(self, config: EnvironmentConfig) -> None:
        self.close()
        self.config = config
        self._build_env()

    def reset(self, seed: Optional[int] = None) -> Tuple[Any, Dict[str, Any]]:
        if self.env is None:
            return [0.0, 0.0], {}
        return self.env.reset(seed=seed)

    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
        if self.env is None:
            reward = random.uniform(-1.0, 1.0)
            done = random.random() < 0.02
            return [0.0, 0.0], reward, done, False, {}
        return self.env.step(action)

    def render(self) -> Optional[Any]:
        if self.env is None:
            return None
        try:
            return self.env.render()
        except Exception:
            return None

    def close(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None


class SB3PolicyAgent:
    def __init__(self, config: TrainingConfig, env: Any):
        self.config = config
        self.env = env
        self.model = None
        self._build_model()

    def _effective_device(self) -> str:
        requested = (self.config.device or "CPU").upper()
        if requested == "GPU" and torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _activation(self):
        if torch is None:
            return None
        return torch.nn.Tanh if self.config.activation == "Tanh" else torch.nn.ReLU

    def _build_model(self) -> None:
        model_cls = POLICY_MAP.get(self.config.policy)
        if model_cls is None or self.env is None:
            self.model = None
            return

        net_arch = parse_hidden_layers(self.config.hidden_layer)
        policy_kwargs: Dict[str, Any] = {"net_arch": net_arch}
        activation_fn = self._activation()
        if activation_fn is not None:
            policy_kwargs["activation_fn"] = activation_fn

        learning_rate = make_lr_schedule(
            base_lr=self.config.learning_rate,
            strategy=self.config.lr_strategy,
            min_lr=self.config.min_lr,
            decay=self.config.lr_decay,
        )

        common_kwargs = {
            "policy": "MlpPolicy",
            "env": self.env,
            "gamma": self.config.gamma,
            "learning_rate": learning_rate,
            "batch_size": self.config.batch_size,
            "policy_kwargs": policy_kwargs,
            "device": self._effective_device(),
            "verbose": 0,
        }

        if self.config.policy == "PPO":
            # Keep PPO rollout size reasonable and divisible by batch size where possible.
            n_steps = max(self.config.batch_size, 2048)
            common_kwargs.update({"n_steps": n_steps})
        elif self.config.policy in {"SAC", "TD3"}:
            common_kwargs.update(
                {
                    "train_freq": self.config.train_freq,
                    "gradient_steps": self.config.gradient_steps,
                    "buffer_size": self.config.buffer_size,
                    "learning_starts": self.config.learning_starts,
                    "tau": self.config.tau,
                }
            )
            if self.config.policy == "TD3":
                common_kwargs["policy_delay"] = self.config.policy_delay

        self.model = model_cls(**common_kwargs)

    def learn(self, total_timesteps: int) -> None:
        if self.model is None:
            return
        self.model.learn(total_timesteps=total_timesteps, reset_num_timesteps=False, progress_bar=False)

    def predict(self, obs: Any, deterministic: bool = False) -> Any:
        if self.model is None:
            if isinstance(obs, (list, tuple)):
                return [0.0] * 2
            return 0
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return action


class ReacherTrainer:
    def __init__(
        self,
        env_config: EnvironmentConfig,
        train_config: TrainingConfig,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.env_config = env_config
        self.train_config = train_config
        self.seed = seed
        self.event_sink = event_sink
        self._first_reset_done = False

        self.pause_event = Event()
        self.pause_event.set()
        self.cancel_event = Event()

        self.env_wrapper = ReacherEnvironment(env_config)
        self.agent = SB3PolicyAgent(train_config, self.env_wrapper.env)

        self.transition_buffer: List[Dict[str, Any]] = []
        self.eval_points: List[Tuple[int, float]] = []
        self.reward_history: List[float] = []
        self.best_reward = float("-inf")

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(payload)

    def update_environment(self, env_config: EnvironmentConfig) -> None:
        self.env_config = env_config
        self.env_wrapper.rebuild(env_config)
        self.agent = SB3PolicyAgent(self.train_config, self.env_wrapper.env)

    def set_pause(self, paused: bool) -> None:
        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def cancel(self) -> None:
        self.cancel_event.set()
        # Ensure waiters unblock during shutdown/cancel.
        self.pause_event.set()

    def _moving_average(self) -> float:
        window = max(1, int(self.train_config.moving_average_window))
        values = self.reward_history[-window:]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _should_capture_episode(self, episode: int, episodes: int) -> bool:
        rate = max(1, int(self.train_config.update_rate_episodes))
        return (episode % rate == 0) or (episode == episodes)

    def run_episode(
        self,
        training: bool = True,
        collect_transitions: bool = True,
        deterministic: bool = False,
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        limit = int(max_steps if max_steps is not None else self.train_config.max_steps)
        # Seed only the first reset for reproducibility; later episodes should vary naturally.
        seed = self.seed if not self._first_reset_done else None
        obs, _ = self.env_wrapper.reset(seed=seed)
        self._first_reset_done = True

        total_reward = 0.0
        steps = 0
        frames: List[Any] = []

        for step_idx in range(limit):
            if self.cancel_event.is_set():
                break
            self.pause_event.wait()

            action = self.agent.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = self.env_wrapper.step(action)

            total_reward += float(reward)
            steps += 1

            if collect_transitions:
                self.transition_buffer.append(
                    {
                        "obs": str(obs),
                        "action": str(action),
                        "reward": float(reward),
                        "next_obs": str(next_obs),
                        "done": bool(terminated or truncated),
                    }
                )

            stride = max(1, int(self.train_config.frame_stride))
            if step_idx == 0 or (step_idx % stride == 0):
                frame = self.env_wrapper.render()
                if frame is not None:
                    frames.append(frame)

            self._emit(
                {
                    "type": "step",
                    "run_id": self.train_config.run_id,
                    "step": steps,
                    "max_steps": limit,
                }
            )

            obs = next_obs
            if terminated or truncated:
                break

        if training and self.agent.model is not None:
            self.agent.learn(total_timesteps=max(1, steps))

        return {
            "reward": float(total_reward),
            "steps": steps,
            "frames": frames,
        }

    def evaluate_policy(self, n_episodes: int = 1) -> float:
        rewards = []
        for _ in range(max(1, n_episodes)):
            episode = self.run_episode(training=False, collect_transitions=False, deterministic=True)
            rewards.append(episode["reward"])
        return float(sum(rewards) / len(rewards)) if rewards else 0.0

    def train(self) -> Dict[str, Any]:
        try:
            episodes = int(self.train_config.episodes)
            for episode_idx in range(1, episodes + 1):
                if self.cancel_event.is_set():
                    break
                self.pause_event.wait()

                episode_data = self.run_episode(training=True, collect_transitions=True, deterministic=False)
                reward = float(episode_data["reward"])
                steps = int(episode_data["steps"])
                self.reward_history.append(reward)
                self.best_reward = max(self.best_reward, reward)
                moving_avg = self._moving_average()

                current_lr = self.train_config.learning_rate
                if self.agent.model is not None and hasattr(self.agent.model, "lr_schedule"):
                    current_lr = float(self.agent.model.lr_schedule(0.0))

                render_state = "on" if self.env_config.render_enabled else "off"

                self._emit(
                    {
                        "type": "episode",
                        "run_id": self.train_config.run_id,
                        "episode": episode_idx,
                        "episodes": episodes,
                        "reward": reward,
                        "moving_average": moving_avg,
                        "eval_points": list(self.eval_points),
                        "steps": steps,
                        "epsilon": "n/a",
                        "lr": current_lr,
                        "best_reward": self.best_reward,
                        "render_state": render_state,
                    }
                )

                eval_score = None
                if self.train_config.eval_rollout_on and episode_idx % 10 == 0:
                    eval_score = self.evaluate_policy(n_episodes=1)
                    self.eval_points.append((episode_idx, eval_score))

                if self._should_capture_episode(episode_idx, episodes):
                    self._emit(
                        {
                            "type": "episode_aux",
                            "run_id": self.train_config.run_id,
                            "episode": episode_idx,
                            "frames": episode_data["frames"],
                            "eval_points": list(self.eval_points),
                            "eval_score": eval_score,
                        }
                    )

            done_payload = {
                "type": "training_done",
                "run_id": self.train_config.run_id,
                "cancelled": self.cancel_event.is_set(),
                "best_reward": self.best_reward,
                "episodes_completed": len(self.reward_history),
            }
            self._emit(done_payload)
            return done_payload
        except Exception as exc:
            payload = {
                "type": "error",
                "run_id": self.train_config.run_id,
                "error": str(exc),
            }
            self._emit(payload)
            return payload

    def export_transitions_csv(self, output_dir: str = "results_csv") -> Optional[Path]:
        if not self.transition_buffer:
            return None

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = out_dir / f"reacher_samples_{self.train_config.run_id}_{stamp}.csv"

        keys = ["obs", "action", "reward", "next_obs", "done"]
        with file_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.transition_buffer)

        return file_path

    def close(self) -> None:
        self.env_wrapper.close()
