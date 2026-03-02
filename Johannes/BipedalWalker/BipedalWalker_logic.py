from __future__ import annotations

import csv
import itertools
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3 import A2C, PPO, SAC, TD3
from stable_baselines3.common.noise import NormalActionNoise


EventCallback = Optional[Callable[[Dict[str, Any]], None]]


@dataclass
class BipedalWalkerConfig:
    hardcore: bool = False
    animation_on: bool = True
    animation_fps: int = 10
    update_rate_episodes: int = 5

    max_steps: int = 1600
    episodes: int = 50
    epsilon_max: float = 1.0
    epsilon_decay: float = 0.995
    epsilon_min: float = 0.05
    gamma: float = 0.99

    policy: str = "PPO"
    hidden_layer: str = "256,256"
    activation: str = "ReLU"
    learning_rate: float = 3e-4
    lr_strategy: str = "constant"
    min_lr: float = 1e-5
    lr_decay: float = 0.999
    replay_size: int = 300000
    batch_size: int = 64
    learning_start: int = 10000
    learning_frequency: int = 1
    target_update: int = 2
    short_episode_full_capture_steps: int = 120
    low_overhead_animation: bool = False

    moving_average_values: int = 20
    device: str = "cpu"


POLICY_DISPLAY_NAMES = ["PPO", "A2C", "SAC", "TD3"]


def get_policy_default_configs() -> Dict[str, Dict[str, Any]]:
    return {
        "PPO": {
            "hidden_layer": "256,256",
            "activation": "Tanh",
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "min_lr": 1e-5,
            "lr_decay": 0.999,
            "batch_size": 64,
            "gamma": 0.99,
        },
        "A2C": {
            "hidden_layer": "256,256",
            "activation": "Tanh",
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "min_lr": 1e-5,
            "lr_decay": 0.999,
            "batch_size": 64,
            "gamma": 0.99,
        },
        "SAC": {
            "hidden_layer": "256,256",
            "activation": "ReLU",
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "min_lr": 1e-5,
            "lr_decay": 0.999,
            "replay_size": 300000,
            "batch_size": 128,
            "learning_start": 5000,
            "learning_frequency": 4,
            "gamma": 0.99,
        },
        "TD3": {
            "hidden_layer": "256,256",
            "activation": "ReLU",
            "learning_rate": 1e-3,
            "lr_strategy": "constant",
            "min_lr": 1e-5,
            "lr_decay": 0.999,
            "replay_size": 300000,
            "batch_size": 128,
            "learning_start": 5000,
            "learning_frequency": 4,
            "target_update": 2,
            "gamma": 0.99,
        },
    }


def parse_hidden_layers(text: str) -> List[int]:
    values = [part.strip() for part in text.split(",") if part.strip()]
    layers = [int(value) for value in values]
    if not layers:
        return [256, 256]
    return layers


def resolve_activation(name: str) -> Any:
    mapping = {
        "ReLU": th.nn.ReLU,
        "Tanh": th.nn.Tanh,
        "ELU": th.nn.ELU,
        "LeakyReLU": th.nn.LeakyReLU,
    }
    return mapping.get(name, th.nn.ReLU)


def create_learning_rate_schedule(config: BipedalWalkerConfig):
    start = float(config.learning_rate)
    end = float(config.min_lr)

    if config.lr_strategy == "constant":
        return start

    if config.lr_strategy == "linear":
        def linear_schedule(progress_remaining: float) -> float:
            return max(end, end + (start - end) * progress_remaining)

        return linear_schedule

    def exponential_schedule(progress_remaining: float) -> float:
        elapsed = 1.0 - progress_remaining
        value = start * (config.lr_decay ** (elapsed * 1000.0))
        return max(end, value)

    return exponential_schedule


class BipedalWalkerEnvironment:
    ENV_ID = "BipedalWalker-v3"

    @classmethod
    def make(cls, *, hardcore: bool, render_mode: Optional[str] = None):
        return gym.make(cls.ENV_ID, hardcore=hardcore, render_mode=render_mode)


class SB3PolicyAgent:
    ALGO_MAP = {
        "PPO": PPO,
        "A2C": A2C,
        "SAC": SAC,
        "TD3": TD3,
    }

    def __init__(self, config: BipedalWalkerConfig):
        if config.policy not in self.ALGO_MAP:
            raise ValueError(f"Unsupported policy: {config.policy}")
        self.config = config
        self.model = None

    def _policy_kwargs(self) -> Dict[str, Any]:
        return {
            "activation_fn": resolve_activation(self.config.activation),
            "net_arch": parse_hidden_layers(self.config.hidden_layer),
        }

    def build(self, env):
        algo = self.ALGO_MAP[self.config.policy]
        policy_kwargs = self._policy_kwargs()
        learning_rate = create_learning_rate_schedule(self.config)

        common = {
            "policy": "MlpPolicy",
            "env": env,
            "gamma": self.config.gamma,
            "learning_rate": learning_rate,
            "verbose": 0,
            "device": self.config.device,
            "policy_kwargs": policy_kwargs,
        }

        if self.config.policy == "PPO":
            ppo_batch = max(16, self.config.batch_size)
            ppo_n_steps = max(ppo_batch, min(2048, max(64, self.config.max_steps)))
            common["n_steps"] = ppo_n_steps
            common["batch_size"] = ppo_batch
            common["gae_lambda"] = 0.95
            common["clip_range"] = 0.2

        if self.config.policy == "A2C":
            common["n_steps"] = max(16, min(128, self.config.max_steps))
            common["gae_lambda"] = 1.0

        if self.config.policy in {"SAC", "TD3"}:
            common["buffer_size"] = max(1000, self.config.replay_size)
            common["batch_size"] = max(16, self.config.batch_size)
            common["learning_starts"] = max(self.config.batch_size * 4, self.config.learning_start)
            common["train_freq"] = max(1, self.config.learning_frequency)
            common["gradient_steps"] = max(1, self.config.learning_frequency)

        if self.config.policy == "TD3":
            common["policy_delay"] = max(1, int(self.config.target_update))
            action_dim = env.action_space.shape[-1]
            common["action_noise"] = NormalActionNoise(mean=np.zeros(action_dim), sigma=0.1 * np.ones(action_dim))

        self.model = algo(**common)
        return self.model


class BipedalWalkerTrainer:
    def __init__(self, config: BipedalWalkerConfig, event_callback: EventCallback = None):
        self.config = config
        self.event_callback = event_callback

        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()

        self.agent: Optional[SB3PolicyAgent] = None
        self.model = None

        self.transitions: List[Dict[str, Any]] = []
        self.reward_history: List[float] = []
        self.moving_average_history: List[float] = []
        self.eval_history: List[float] = []

    def emit(self, event: Dict[str, Any]):
        if self.event_callback:
            self.event_callback(event)

    def stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def set_paused(self, paused: bool):
        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def update_environment(
        self,
        hardcore: bool,
        *,
        animation_on: Optional[bool] = None,
        animation_fps: Optional[int] = None,
        update_rate_episodes: Optional[int] = None,
    ):
        self.config.hardcore = hardcore
        if animation_on is not None:
            self.config.animation_on = bool(animation_on)
        if animation_fps is not None:
            self.config.animation_fps = max(1, int(animation_fps))
        if update_rate_episodes is not None:
            self.config.update_rate_episodes = max(1, int(update_rate_episodes))

    def _epsilon_for_episode(self, episode_idx: int) -> float:
        value = self.config.epsilon_max * (self.config.epsilon_decay ** max(0, episode_idx - 1))
        return max(self.config.epsilon_min, value)

    def _compute_moving_average(self, values: List[float], window: int) -> float:
        if not values:
            return 0.0
        size = max(1, min(window, len(values)))
        return float(np.mean(values[-size:]))

    def run_episode(
        self,
        env,
        *,
        deterministic: bool,
        max_steps: int,
        collect_transitions: bool = False,
        include_frame: bool = False,
        rollout_frame_stride: int = 1,
        max_rollout_frames: int = 300,
        short_episode_full_capture_steps: int = 120,
        frame_downsample: int = 1,
    ) -> Dict[str, Any]:
        observation, _ = env.reset()
        total_reward = 0.0
        steps = 0
        done = False
        truncated = False
        collected: List[Dict[str, Any]] = []
        latest_frame = None
        rollout_frames: List[Any] = []

        while not (done or truncated) and steps < max_steps:
            if self.model is None:
                action = env.action_space.sample()
            else:
                action, _ = self.model.predict(observation, deterministic=deterministic)

            next_observation, reward, done, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1

            if collect_transitions:
                collected.append(
                    {
                        "step": steps,
                        "reward": float(reward),
                        "done": bool(done),
                        "truncated": bool(truncated),
                    }
                )

            if include_frame:
                try:
                    frame = env.render()
                    if frame is not None:
                        ds = max(1, int(frame_downsample))
                        if ds > 1 and hasattr(frame, "ndim") and frame.ndim >= 2:
                            frame = frame[::ds, ::ds]
                        latest_frame = frame
                        should_capture = False
                        if steps <= max(1, short_episode_full_capture_steps):
                            should_capture = True
                        elif (
                            (steps - max(1, short_episode_full_capture_steps)) % max(1, rollout_frame_stride) == 0
                        ):
                            should_capture = True

                        if should_capture and len(rollout_frames) < max(1, max_rollout_frames):
                            rollout_frames.append(frame)
                except Exception:
                    pass

            observation = next_observation

        if include_frame and latest_frame is not None:
            if len(rollout_frames) == 0:
                rollout_frames.append(latest_frame)
            elif len(rollout_frames) < max(1, max_rollout_frames):
                rollout_frames.append(latest_frame)

        return {
            "reward": total_reward,
            "steps": steps,
            "transitions": collected,
            "frame": latest_frame,
            "rollout_frames": rollout_frames,
        }

    def evaluate_policy(self, episodes: int = 5) -> Dict[str, Any]:
        env = BipedalWalkerEnvironment.make(hardcore=self.config.hardcore, render_mode=None)
        rewards = []
        try:
            for _ in range(max(1, episodes)):
                result = self.run_episode(
                    env,
                    deterministic=True,
                    max_steps=self.config.max_steps,
                    collect_transitions=False,
                    include_frame=False,
                )
                rewards.append(float(result["reward"]))
        finally:
            env.close()

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if rewards else 0.0,
            "rewards": rewards,
        }

    def train(self, *, collect_transitions: bool = False, run_label: str = "run") -> Dict[str, Any]:
        self.reward_history = []
        self.moving_average_history = []
        self.eval_history = []
        self.transitions = []

        train_env = BipedalWalkerEnvironment.make(hardcore=self.config.hardcore, render_mode=None)
        eval_env = BipedalWalkerEnvironment.make(
            hardcore=self.config.hardcore,
            render_mode="rgb_array" if self.config.animation_on else None,
        )

        try:
            self.agent = SB3PolicyAgent(self.config)
            self.model = self.agent.build(train_env)
            best_reward = float("-inf")

            for episode in range(1, self.config.episodes + 1):
                if self.stop_event.is_set():
                    break

                while not self.pause_event.is_set() and not self.stop_event.is_set():
                    self.emit({"type": "paused", "episode": episode, "run_label": run_label})
                    time.sleep(0.05)

                if self.stop_event.is_set():
                    break

                self.model.learn(total_timesteps=self.config.max_steps, reset_num_timesteps=False, progress_bar=False)

                include_frame = (
                    self.config.animation_on
                    and (episode % max(1, self.config.update_rate_episodes) == 0 or episode == self.config.episodes)
                )

                if self.config.low_overhead_animation:
                    target_rollout_frames = max(40, min(180, self.config.animation_fps * 10))
                    short_capture_steps = max(
                        10,
                        min(self.config.max_steps, min(int(self.config.short_episode_full_capture_steps), 60)),
                    )
                    frame_downsample = 2
                else:
                    target_rollout_frames = max(60, min(400, self.config.animation_fps * 20))
                    short_capture_steps = max(10, min(self.config.max_steps, int(self.config.short_episode_full_capture_steps)))
                    frame_downsample = 1

                remaining_budget = max(1, target_rollout_frames - short_capture_steps)
                remaining_steps = max(1, self.config.max_steps - short_capture_steps)
                rollout_stride = max(1, remaining_steps // remaining_budget)

                episode_result = self.run_episode(
                    eval_env,
                    deterministic=True,
                    max_steps=self.config.max_steps,
                    collect_transitions=collect_transitions,
                    include_frame=include_frame,
                    rollout_frame_stride=rollout_stride,
                    max_rollout_frames=target_rollout_frames,
                    short_episode_full_capture_steps=short_capture_steps,
                    frame_downsample=frame_downsample,
                )

                reward = float(episode_result["reward"])
                self.reward_history.append(reward)
                moving_avg = self._compute_moving_average(self.reward_history, self.config.moving_average_values)
                self.moving_average_history.append(moving_avg)
                best_reward = max(best_reward, reward)

                if collect_transitions:
                    for row in episode_result["transitions"]:
                        row["episode"] = episode
                        row["policy"] = self.config.policy
                        row["hardcore"] = self.config.hardcore
                    self.transitions.extend(episode_result["transitions"])

                should_emit = (
                    episode % max(1, self.config.update_rate_episodes) == 0
                    or episode == self.config.episodes
                )
                eval_value = None
                if should_emit:
                    eval_value = reward
                    self.eval_history.append(eval_value)

                self.emit(
                    {
                        "type": "episode",
                        "episode": episode,
                        "episodes": self.config.episodes,
                        "steps": int(episode_result["steps"]),
                        "reward": reward,
                        "moving_average": moving_avg,
                        "eval_reward": eval_value,
                        "epsilon": self._epsilon_for_episode(episode),
                        "learning_rate": float(self.config.learning_rate),
                        "best_reward": best_reward,
                        "render_state": "on" if include_frame else "skipped",
                        "frame": episode_result["frame"],
                        "rollout_frames": episode_result.get("rollout_frames", []),
                        "plot_refresh": True,
                        "run_label": run_label,
                    }
                )

            self.emit(
                {
                    "type": "training_done",
                    "run_label": run_label,
                    "stopped": self.stop_event.is_set(),
                    "history": {
                        "reward": list(self.reward_history),
                        "moving_average": list(self.moving_average_history),
                        "eval": list(self.eval_history),
                    },
                    "config": asdict(self.config),
                }
            )
        except Exception as exc:
            self.emit({"type": "error", "run_label": run_label, "message": str(exc)})
            raise
        finally:
            train_env.close()
            eval_env.close()

        return {
            "reward": list(self.reward_history),
            "moving_average": list(self.moving_average_history),
            "eval": list(self.eval_history),
            "transitions": list(self.transitions),
        }

    def save_transitions_csv(self, output_dir: Path, run_name: str) -> Optional[Path]:
        if not self.transitions:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = output_dir / f"{run_name}_{stamp}.csv"

        keys = ["episode", "step", "reward", "done", "truncated", "policy", "hardcore"]
        with filepath.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in self.transitions:
                writer.writerow({key: row.get(key) for key in keys})

        return filepath


def build_compare_combinations(compare_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    active = {key: value for key, value in compare_values.items() if value}
    if not active:
        return [{}]
    keys = list(active.keys())
    combinations = []
    for combo in itertools.product(*(active[key] for key in keys)):
        combinations.append({key: value for key, value in zip(keys, combo)})
    return combinations


def make_run_label(config: BipedalWalkerConfig) -> str:
    return (
        f"Policy={config.policy}, hardcore={config.hardcore}, gamma={config.gamma}, "
        f"lr={config.learning_rate:.1e}, batch={config.batch_size}"
    )


def png_filename(config: BipedalWalkerConfig) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (
        f"BipedalWalker_{config.policy}_hardcore-{int(config.hardcore)}_"
        f"gamma-{config.gamma}_lr-{config.learning_rate:.1e}_{stamp}.png"
    )