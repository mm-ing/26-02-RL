from __future__ import annotations

import csv
import math
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import DQN, PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

try:
    from sb3_contrib import QRDQN
except Exception:
    QRDQN = None


POLICIES_CONTINUOUS = ("PPO", "SAC", "TD3")
POLICIES_DISCRETE = ("DDQN", "QR-DQN")
POLICIES_ALL = POLICIES_CONTINUOUS + POLICIES_DISCRETE

DEFAULT_ENV_CONFIG: Dict[str, Any] = {
    "env_id": "CarRacing-v3",
    "render_mode": "rgb_array",
    "lap_complete_percent": 0.95,
    "domain_randomize": False,
    "continuous": True,
}

DEFAULT_GENERAL: Dict[str, Any] = {
    "max_steps": 1000,
    "episodes": 3000,
}

DEFAULT_SPECIFIC: Dict[str, Dict[str, Any]] = {
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
    },
    "TD3": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_delay": 2,
    },
    "PPO": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "Tanh",
        "lr_strategy": "linear",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "n_steps": 1024,
        "ent_coef": 0.0,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
    },
    "DDQN": {
        "gamma": 0.99,
        "learning_rate": 2.5e-4,
        "batch_size": 128,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
    },
    "QR-DQN": {
        "gamma": 0.99,
        "learning_rate": 2.5e-4,
        "batch_size": 128,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "buffer_size": 200_000,
        "learning_starts": 10_000,
        "target_update_interval": 1000,
        "train_freq": 4,
        "gradient_steps": 1,
        "exploration_fraction": 0.20,
        "exploration_final_eps": 0.05,
        "n_quantiles": 50,
    },
}


def parse_hidden_layers(raw_value: Any, fallback: Sequence[int] = (256, 256)) -> List[int]:
    if raw_value is None:
        return list(fallback)
    if isinstance(raw_value, (list, tuple)):
        layers = [int(v) for v in raw_value if int(v) > 0]
        return layers if layers else list(fallback)
    text = str(raw_value).strip()
    if not text:
        return list(fallback)
    try:
        parts = [int(part.strip()) for part in text.split(",") if part.strip()]
        parts = [part for part in parts if part > 0]
        if not parts:
            return list(fallback)
        if len(parts) == 1:
            return [parts[0], parts[0]]
        return parts
    except Exception:
        return list(fallback)


def make_lr_schedule(
    base_lr: float,
    strategy: str = "constant",
    min_lr: float = 1e-5,
    lr_decay: float = 0.995,
) -> Callable[[float], float] | float:
    strategy = (strategy or "constant").strip().lower()
    base = max(float(base_lr), float(min_lr))
    min_value = float(min_lr)
    decay = min(max(float(lr_decay), 1e-6), 0.999999)

    if strategy == "constant":
        return base

    if strategy == "linear":
        def linear(progress_remaining: float) -> float:
            value = min_value + (base - min_value) * max(progress_remaining, 0.0)
            return max(min_value, float(value))

        return linear

    if strategy == "exponential":
        def exponential(progress_remaining: float) -> float:
            progressed = 1.0 - max(progress_remaining, 0.0)
            value = base * (decay ** (progressed * 1000.0))
            return max(min_value, float(value))

        return exponential

    return base


@dataclass
class EnvConfig:
    env_id: str = "CarRacing-v3"
    render_mode: Optional[str] = "rgb_array"
    lap_complete_percent: float = 0.95
    domain_randomize: bool = False
    continuous: bool = True


@dataclass
class TrainConfig:
    policy_name: str = "SAC"
    episodes: int = 3000
    max_steps: int = 1000
    params: Dict[str, Any] = field(default_factory=dict)
    env_config: EnvConfig = field(default_factory=EnvConfig)
    animation_on: bool = True
    animation_fps: int = 30
    update_rate: int = 1
    frame_stride: int = 2
    run_id: str = "run"
    session_id: str = "session"
    device: str = "cpu"
    collect_transitions: bool = False
    cpu_thread_budget: Optional[int] = None
    performance_mode: bool = False
    emit_meta_every: int = 0
    num_envs: int = 4


class CarRacingEnvWrapper:
    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()

    def update(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

    def make_env(self, render_mode: Optional[str] = None, continuous: Optional[bool] = None) -> gym.Env:
        cfg = self.config
        env = gym.make(
            cfg.env_id,
            render_mode=cfg.render_mode if render_mode is None else render_mode,
            lap_complete_percent=cfg.lap_complete_percent,
            domain_randomize=cfg.domain_randomize,
            continuous=cfg.continuous if continuous is None else continuous,
        )
        return env


class _PauseStopCallback(BaseCallback):
    def __init__(
        self,
        pause_event: threading.Event,
        stop_event: threading.Event,
        total_episodes: int,
        update_rate: int = 1,
        on_episode_done: Optional[Callable[[int, float, int], None]] = None,
    ):
        super().__init__(verbose=0)
        self.pause_event = pause_event
        self.stop_event = stop_event
        self.total_episodes = max(0, int(total_episodes))
        self.update_rate = max(1, int(update_rate))
        self.on_episode_done = on_episode_done
        self._episode_step = 0
        self._completed_episodes = 0

    def _on_step(self) -> bool:
        if not self.pause_event.is_set():
            self.pause_event.wait()
        if self.stop_event.is_set():
            return False
        self._episode_step += 1
        infos = self.locals.get("infos")
        info_list = infos if isinstance(infos, list) else [infos]
        for info in info_list:
            if not isinstance(info, dict):
                continue
            episode = info.get("episode")
            if not episode:
                final_info = info.get("final_info")
                if isinstance(final_info, dict):
                    episode = final_info.get("episode")
            if not episode:
                continue
            if self._completed_episodes >= self.total_episodes:
                return False
            self._completed_episodes += 1
            reward = float(episode.get("r", 0.0))
            length = int(episode.get("l", self._episode_step))
            if self.on_episode_done is not None:
                self.on_episode_done(self._completed_episodes, reward, length)
            self._episode_step = 0
            if self._completed_episodes >= self.total_episodes:
                return False
        return True


class SB3PolicyFactory:
    def create_model(self, env: gym.Env, policy_name: str, params: Dict[str, Any], device: str = "cpu"):
        hidden_layers = parse_hidden_layers(params.get("hidden_layer", "256,256"))
        activation_name = str(params.get("activation", "ReLU")).strip().lower()
        activation_fn = torch.nn.Tanh if activation_name == "tanh" else torch.nn.ReLU
        lr = make_lr_schedule(
            float(params.get("learning_rate", 3e-4)),
            strategy=str(params.get("lr_strategy", "constant")),
            min_lr=float(params.get("min_lr", 1e-5)),
            lr_decay=float(params.get("lr_decay", 0.995)),
        )
        gamma = float(params.get("gamma", 0.99))
        batch_size = int(params.get("batch_size", 256))
        effective_device = "cuda" if str(device).upper() == "GPU" and torch.cuda.is_available() else "cpu"

        if policy_name == "PPO":
            n_steps = int(params.get("n_steps", 1024))
            if n_steps < batch_size:
                n_steps = batch_size
            n_steps = int(math.ceil(n_steps / batch_size) * batch_size)
            return PPO(
                "CnnPolicy",
                env,
                verbose=0,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                n_steps=n_steps,
                ent_coef=float(params.get("ent_coef", 0.0)),
                gae_lambda=float(params.get("gae_lambda", 0.95)),
                clip_range=float(params.get("clip_range", 0.2)),
                policy_kwargs={"net_arch": hidden_layers, "activation_fn": activation_fn},
                device=effective_device,
            )

        if policy_name == "SAC":
            return SAC(
                "CnnPolicy",
                env,
                verbose=0,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(params.get("buffer_size", 200_000)),
                learning_starts=int(params.get("learning_starts", 10_000)),
                tau=float(params.get("tau", 0.005)),
                train_freq=int(params.get("train_freq", 1)),
                gradient_steps=int(params.get("gradient_steps", 1)),
                policy_kwargs={"net_arch": hidden_layers, "activation_fn": activation_fn},
                device=effective_device,
            )

        if policy_name == "TD3":
            return TD3(
                "CnnPolicy",
                env,
                verbose=0,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(params.get("buffer_size", 200_000)),
                learning_starts=int(params.get("learning_starts", 10_000)),
                tau=float(params.get("tau", 0.005)),
                train_freq=int(params.get("train_freq", 1)),
                gradient_steps=int(params.get("gradient_steps", 1)),
                policy_delay=int(params.get("policy_delay", 2)),
                policy_kwargs={"net_arch": hidden_layers, "activation_fn": activation_fn},
                device=effective_device,
            )

        if policy_name == "DDQN":
            return DQN(
                "CnnPolicy",
                env,
                verbose=0,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(params.get("buffer_size", 200_000)),
                learning_starts=int(params.get("learning_starts", 10_000)),
                target_update_interval=int(params.get("target_update_interval", 1000)),
                train_freq=int(params.get("train_freq", 4)),
                gradient_steps=int(params.get("gradient_steps", 1)),
                exploration_fraction=float(params.get("exploration_fraction", 0.20)),
                exploration_final_eps=float(params.get("exploration_final_eps", 0.05)),
                policy_kwargs={"net_arch": hidden_layers, "activation_fn": activation_fn},
                device=effective_device,
            )

        if policy_name == "QR-DQN":
            if QRDQN is None:
                raise RuntimeError("QR-DQN requires sb3-contrib. Please install sb3-contrib.")
            return QRDQN(
                "CnnPolicy",
                env,
                verbose=0,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                buffer_size=int(params.get("buffer_size", 200_000)),
                learning_starts=int(params.get("learning_starts", 10_000)),
                target_update_interval=int(params.get("target_update_interval", 1000)),
                train_freq=int(params.get("train_freq", 4)),
                gradient_steps=int(params.get("gradient_steps", 1)),
                exploration_fraction=float(params.get("exploration_fraction", 0.20)),
                exploration_final_eps=float(params.get("exploration_final_eps", 0.05)),
                n_quantiles=int(params.get("n_quantiles", 50)),
                policy_kwargs={"net_arch": hidden_layers, "activation_fn": activation_fn},
                device=effective_device,
            )

        raise ValueError(f"Unsupported policy: {policy_name}")


class CarRacingTrainer:
    def __init__(
        self,
        env_wrapper: Optional[CarRacingEnvWrapper] = None,
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.env_wrapper = env_wrapper or CarRacingEnvWrapper()
        self.event_sink = event_sink
        self.factory = SB3PolicyFactory()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.animation_event = threading.Event()
        self.animation_event.set()
        self.stop_event = threading.Event()
        self.last_model = None
        self.transition_samples: List[Dict[str, Any]] = []

    @staticmethod
    def policy_is_continuous(policy_name: str) -> bool:
        return policy_name in POLICIES_CONTINUOUS

    def set_event_sink(self, event_sink: Callable[[Dict[str, Any]], None]) -> None:
        self.event_sink = event_sink

    def _emit(self, payload: Dict[str, Any]) -> None:
        if self.event_sink:
            self.event_sink(payload)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause_event.clear()
        else:
            self.pause_event.set()

    def cancel(self) -> None:
        self.stop_event.set()
        self.pause_event.set()

    def reset_control_flags(self) -> None:
        self.stop_event.clear()
        self.pause_event.set()

    def set_animation_enabled(self, enabled: bool) -> None:
        if enabled:
            self.animation_event.set()
        else:
            self.animation_event.clear()

    def _read_optimizer_lr(self, model: Any) -> float:
        optimizer = getattr(getattr(model, "policy", None), "optimizer", None)
        if optimizer and optimizer.param_groups:
            return float(optimizer.param_groups[0].get("lr", 0.0))
        return float("nan")

    def run_episode(
        self,
        model: Any = None,
        max_steps: int = 1000,
        deterministic: bool = True,
        capture_frames: bool = False,
        frame_stride: int = 2,
        collect_transitions: bool = False,
    ) -> Dict[str, Any]:
        env = self.env_wrapper.make_env(
            render_mode="rgb_array" if capture_frames else None,
            continuous=self.env_wrapper.config.continuous,
        )
        obs, _ = env.reset()
        total_reward = 0.0
        executed_steps = 0
        frames: List[np.ndarray] = []
        transitions: List[Dict[str, Any]] = []

        while executed_steps < int(max_steps) and not self.stop_event.is_set():
            self.pause_event.wait()
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

            if collect_transitions:
                transitions.append(
                    {
                        "step": executed_steps,
                        "reward": float(reward),
                        "done": done,
                    }
                )

            if capture_frames and (executed_steps == 0 or executed_steps % max(1, int(frame_stride)) == 0):
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

            obs = next_obs
            executed_steps += 1
            if done:
                break

        env.close()
        return {
            "reward": total_reward,
            "steps": executed_steps,
            "frames": frames,
            "transitions": transitions,
        }

    def run_episode_on_env(
        self,
        env: gym.Env,
        model: Any,
        max_steps: int,
        deterministic: bool,
        capture_frames: bool,
        frame_stride: int,
    ) -> Dict[str, Any]:
        obs, _ = env.reset()
        total_reward = 0.0
        executed_steps = 0
        frames: List[np.ndarray] = []

        while executed_steps < int(max_steps) and not self.stop_event.is_set():
            self.pause_event.wait()
            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            total_reward += float(reward)

            if capture_frames and (executed_steps == 0 or executed_steps % max(1, int(frame_stride)) == 0):
                frame = env.render()
                if isinstance(frame, np.ndarray):
                    frames.append(frame)

            obs = next_obs
            executed_steps += 1
            if done:
                break

        return {"reward": total_reward, "steps": executed_steps, "frames": frames}

    def _make_single_train_env(self, continuous: bool, max_steps: int, with_monitor: bool = True) -> gym.Env:
        env = self.env_wrapper.make_env(render_mode=None, continuous=continuous)
        if max_steps > 0 and isinstance(env, gym.Env):
            env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        if with_monitor and isinstance(env, gym.Env):
            env = Monitor(env)
        return env

    def _build_train_env(self, continuous: bool, max_steps: int, num_envs: int):
        num_envs = max(1, int(num_envs))
        probe_env = self._make_single_train_env(
            continuous=continuous,
            max_steps=max_steps,
            with_monitor=(num_envs <= 1),
        )
        if num_envs <= 1 or not isinstance(probe_env, gym.Env):
            return probe_env
        probe_env.close()

        def _make_env_fn():
            return self._make_single_train_env(
                continuous=continuous,
                max_steps=max_steps,
                with_monitor=False,
            )

        env_fns = [_make_env_fn for _ in range(num_envs)]
        return VecMonitor(DummyVecEnv(env_fns))

    def evaluate_policy(self, model: Any, max_steps: int = 1000) -> Dict[str, Any]:
        result = self.run_episode(
            model=model,
            max_steps=max_steps,
            deterministic=True,
            capture_frames=False,
            frame_stride=1,
            collect_transitions=False,
        )
        return result

    def train(self, config: TrainConfig) -> Dict[str, Any]:
        self.reset_control_flags()
        self.transition_samples.clear()
        self.set_animation_enabled(bool(config.animation_on))

        if str(config.device).upper() != "GPU" and config.cpu_thread_budget is not None:
            try:
                torch.set_num_threads(max(1, int(config.cpu_thread_budget)))
            except Exception:
                pass

        continuous = self.policy_is_continuous(config.policy_name)
        self.env_wrapper.update(
            env_id=config.env_config.env_id,
            render_mode=config.env_config.render_mode,
            lap_complete_percent=config.env_config.lap_complete_percent,
            domain_randomize=config.env_config.domain_randomize,
            continuous=continuous,
        )

        train_env = self._build_train_env(
            continuous=continuous,
            max_steps=int(config.max_steps),
            num_envs=int(getattr(config, "num_envs", 1)),
        )
        model = self.factory.create_model(train_env, config.policy_name, config.params, device=config.device)
        self.last_model = model

        animation_env: Optional[gym.Env] = None
        if bool(config.animation_on):
            animation_env = self.env_wrapper.make_env(render_mode="rgb_array", continuous=continuous)
            if int(config.max_steps) > 0 and isinstance(animation_env, gym.Env):
                animation_env = gym.wrappers.TimeLimit(animation_env, max_episode_steps=int(config.max_steps))

        rewards: List[float] = []
        eval_points: List[Tuple[int, float]] = []
        best_reward = float("-inf")
        run_meta = {
            "policy": config.policy_name,
            "max_steps": int(config.max_steps),
            "gamma": config.params.get("gamma", "-"),
            "learning_rate": config.params.get("learning_rate", "-"),
            "lr_strategy": config.params.get("lr_strategy", "-"),
            "lr_decay": config.params.get("lr_decay", "-"),
            "epsilon": config.params.get("exploration_final_eps", "-"),
            "epsilon_decay": config.params.get("exploration_fraction", "-"),
            "epsilon_min": config.params.get("exploration_final_eps", "-"),
            "lap_complete_percent": config.env_config.lap_complete_percent,
            "domain_randomize": config.env_config.domain_randomize,
            "continuous": config.env_config.continuous,
            "num_envs": int(getattr(config, "num_envs", 1)),
            "params": dict(config.params),
        }

        try:
            target_episodes = max(0, int(config.episodes))

            def _on_episode_done(
                episode: int,
                episode_reward: float,
                episode_steps: int,
            ) -> None:
                nonlocal best_reward
                rewards.append(float(episode_reward))
                best_reward = max(best_reward, float(episode_reward))
                window = min(20, len(rewards))
                moving_average = float(np.mean(rewards[-window:]))

                emit_meta_every = max(0, int(getattr(config, "emit_meta_every", 0)))
                include_meta = episode == 1 or (emit_meta_every > 0 and episode % emit_meta_every == 0)

                should_capture_frames = (
                    bool(config.animation_on)
                    and (episode % max(1, int(config.update_rate)) == 0 or episode == target_episodes)
                    and self.animation_event.is_set()
                    and animation_env is not None
                )

                # Emit episode metrics immediately so live plot updates every episode
                # even when optional visualization/eval work is expensive.
                epsilon = float(getattr(model, "exploration_rate", float("nan")))
                lr = self._read_optimizer_lr(model)
                metric_payload = {
                    "type": "episode",
                    "session_id": config.session_id,
                    "run_id": config.run_id,
                    "episode": episode,
                    "episodes": target_episodes,
                    "reward": float(episode_reward),
                    "moving_average": moving_average,
                    "eval_points": list(eval_points),
                    "steps": int(episode_steps),
                    "epsilon": epsilon,
                    "lr": lr,
                    "best_reward": best_reward,
                    "render_state": "pending" if should_capture_frames else "off",
                }
                if include_meta:
                    metric_payload["meta"] = run_meta
                self._emit(metric_payload)

                if should_capture_frames:
                    animation_rollout = self.run_episode_on_env(
                        env=animation_env,
                        model=model,
                        max_steps=int(config.max_steps),
                        deterministic=True,
                        capture_frames=True,
                        frame_stride=max(1, int(config.frame_stride)),
                    )
                    frames = list(animation_rollout.get("frames", []))
                    render_state = "on" if frames else "skipped"
                else:
                    frames = []
                    render_state = "off"

                if config.collect_transitions:
                    rollout = self.run_episode(
                        model=model,
                        max_steps=int(config.max_steps),
                        deterministic=True,
                        capture_frames=False,
                        frame_stride=max(1, int(config.frame_stride)),
                        collect_transitions=True,
                    )
                    self.transition_samples.extend(rollout["transitions"])

                if episode % 10 == 0:
                    eval_result = self.evaluate_policy(model, max_steps=int(config.max_steps))
                    eval_points.append((episode, float(eval_result["reward"])))

                aux_payload = {
                    "type": "episode_aux",
                    "session_id": config.session_id,
                    "run_id": config.run_id,
                    "episode": episode,
                    "episodes": target_episodes,
                    "steps": int(episode_steps),
                    "epsilon": epsilon,
                    "lr": lr,
                    "best_reward": best_reward,
                    "render_state": render_state,
                    "eval_points": list(eval_points),
                }
                if frames:
                    aux_payload["frames"] = frames
                    aux_payload["frame"] = frames[-1]
                self._emit(aux_payload)

            if target_episodes > 0:
                callback = _PauseStopCallback(
                    self.pause_event,
                    self.stop_event,
                    total_episodes=target_episodes,
                    on_episode_done=_on_episode_done,
                )
                total_timesteps = max(1, target_episodes * max(1, int(config.max_steps)))
                model.learn(
                    total_timesteps=total_timesteps,
                    reset_num_timesteps=False,
                    callback=callback,
                    progress_bar=False,
                )

            summary = {
                "type": "training_done",
                "session_id": config.session_id,
                "run_id": config.run_id,
                "episodes_done": len(rewards),
                "best_reward": best_reward if rewards else float("nan"),
                "rewards": rewards,
                "eval_points": eval_points,
                "cancelled": self.stop_event.is_set(),
                "meta": run_meta,
            }
            self._emit(summary)
            return summary

        except Exception as exc:
            error_payload = {
                "type": "error",
                "session_id": config.session_id,
                "run_id": config.run_id,
                "message": str(exc),
            }
            self._emit(error_payload)
            raise

        finally:
            train_env.close()
            if animation_env is not None:
                animation_env.close()

    def export_sampled_transitions_csv(
        self,
        output_dir: str | os.PathLike[str] = "results_csv",
        run_label: str = "run",
    ) -> Optional[Path]:
        if not self.transition_samples:
            return None

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = output_path / f"{run_label}_samples_{timestamp}.csv"

        with file_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=["step", "reward", "done"])
            writer.writeheader()
            writer.writerows(self.transition_samples)

        return file_path
