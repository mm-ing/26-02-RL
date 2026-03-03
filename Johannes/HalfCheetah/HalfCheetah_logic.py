from __future__ import annotations

import csv
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
except Exception:
    gym = None

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None

try:
    from stable_baselines3 import PPO, SAC, TD3
    from stable_baselines3.common.vec_env import DummyVecEnv
except Exception:
    PPO = None
    SAC = None
    TD3 = None
    DummyVecEnv = None


ENV_ID = "HalfCheetah-v5"
EXPOSED_POLICIES = ("PPO", "SAC", "TD3")

ENV_DEFAULTS: Dict[str, Any] = {
    "forward_reward_weight": 1.0,
    "ctrl_cost_weight": 0.1,
    "reset_noise_scale": 0.1,
    "exclude_current_positions_from_observation": True,
}

GENERAL_DEFAULTS: Dict[str, Any] = {
    "max_steps": 1000,
    "episodes": 60,
    "epsilon_max": 0.0,
    "epsilon_decay": 1.0,
    "epsilon_min": 0.0,
    "gamma": 0.99,
}

POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "hidden_layer": 256,
        "activation": "Tanh",
        "lr": 3e-4,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
        "replay_size": 100000,
        "batch_size": 64,
        "learning_start": 0,
        "learning_frequency": 1,
        "target_update": 1,
        "n_steps": 2048,
    },
    "SAC": {
        "hidden_layer": 256,
        "activation": "ReLU",
        "lr": 3e-4,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
        "replay_size": 500000,
        "batch_size": 256,
        "learning_start": 5000,
        "learning_frequency": 1,
        "target_update": 1,
    },
    "TD3": {
        "hidden_layer": 256,
        "activation": "ReLU",
        "lr": 1e-3,
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 1.0,
        "replay_size": 500000,
        "batch_size": 256,
        "learning_start": 5000,
        "learning_frequency": 1,
        "target_update": 2,
    },
}


def _required_packages_present() -> bool:
    return gym is not None and PPO is not None and SAC is not None and TD3 is not None and DummyVecEnv is not None


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


def _activation(name: str):
    if nn is None:
        return None
    mapping = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "elu": nn.ELU,
        "gelu": nn.GELU,
    }
    return mapping.get(str(name).lower(), nn.ReLU)


def make_lr_schedule(strategy: str, base_lr: float, min_lr: float, decay: float):
    strategy = str(strategy).lower()
    if strategy == "constant":
        return base_lr

    def schedule(progress_remaining: float) -> float:
        p = max(0.0, min(1.0, float(progress_remaining)))
        if strategy == "linear":
            return max(min_lr, min_lr + (base_lr - min_lr) * p)
        if strategy == "exponential":
            steps = int((1.0 - p) * 1000)
            return max(min_lr, base_lr * (decay ** steps))
        return base_lr

    return schedule


@dataclass
class TrainerConfig:
    policy: str = "PPO"
    device: str = "CPU"
    episodes: int = GENERAL_DEFAULTS["episodes"]
    max_steps: int = GENERAL_DEFAULTS["max_steps"]
    gamma: float = GENERAL_DEFAULTS["gamma"]
    epsilon_max: float = GENERAL_DEFAULTS["epsilon_max"]
    epsilon_decay: float = GENERAL_DEFAULTS["epsilon_decay"]
    epsilon_min: float = GENERAL_DEFAULTS["epsilon_min"]
    moving_average_values: int = 20
    update_rate_episodes: int = 5
    animation_on: bool = True
    rollout_full_capture_steps: int = 120
    low_overhead_animation: bool = False
    eval_interval: int = 10
    eval_episodes: int = 1
    run_id: str = ""
    seed: int = 42
    env_params: Optional[Dict[str, Any]] = None
    specific_params: Optional[Dict[str, Any]] = None


class HalfCheetahEnvironment:
    def __init__(self, env_id: str = ENV_ID, env_params: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = None):
        self.env_id = env_id
        self.env_params = dict(ENV_DEFAULTS)
        if env_params:
            self.env_params.update(env_params)
        self.render_mode = render_mode
        self.env = None

    def _build(self):
        if gym is None:
            raise RuntimeError("gymnasium is not available.")
        kwargs = dict(self.env_params)
        if self.render_mode:
            kwargs["render_mode"] = self.render_mode
        self.env = gym.make(self.env_id, **kwargs)
        return self.env

    def ensure(self):
        if self.env is None:
            return self._build()
        return self.env

    def update(self, env_params: Dict[str, Any], render_mode: Optional[str] = None):
        self.close()
        self.env_params.update(env_params or {})
        self.render_mode = render_mode
        return self._build()

    def reset(self):
        env = self.ensure()
        return env.reset()

    def step(self, action):
        env = self.ensure()
        return env.step(action)

    def render(self):
        env = self.ensure()
        return env.render()

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None

    def make_vectorized(self):
        if DummyVecEnv is None:
            raise RuntimeError("stable_baselines3 is not available.")

        env_id = self.env_id
        kwargs = dict(self.env_params)

        def make_env():
            return gym.make(env_id, **kwargs)

        return DummyVecEnv([make_env])


class SB3PolicyAgent:
    POLICY_MAP = {
        "PPO": PPO,
        "SAC": SAC,
        "TD3": TD3,
    }

    def __init__(self, policy_name: str, device: str = "CPU", seed: int = 42):
        if policy_name not in EXPOSED_POLICIES:
            raise ValueError(f"Unsupported policy '{policy_name}'. Allowed: {EXPOSED_POLICIES}")
        self.policy_name = policy_name
        self.seed = seed
        self.device = self._resolve_device(device)

    def _resolve_device(self, device: str) -> str:
        if torch is None:
            return "cpu"
        if str(device).upper() == "GPU" and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def create_model(self, vec_env, gamma: float, specific_params: Dict[str, Any]):
        policy_class = self.POLICY_MAP[self.policy_name]
        if policy_class is None:
            raise RuntimeError("Stable-Baselines3 is not available.")

        hidden_layer = _safe_int(specific_params.get("hidden_layer", 256), 256)
        activation_name = str(specific_params.get("activation", "ReLU"))
        activation_fn = _activation(activation_name)
        net_arch = [hidden_layer, hidden_layer]

        base_lr = _safe_float(specific_params.get("lr", 3e-4), 3e-4)
        min_lr = _safe_float(specific_params.get("min_lr", 1e-5), 1e-5)
        lr_decay = _safe_float(specific_params.get("lr_decay", 0.99), 0.99)
        lr_strategy = str(specific_params.get("lr_strategy", "constant"))
        lr = make_lr_schedule(lr_strategy, base_lr, min_lr, lr_decay)

        policy_kwargs = {"net_arch": net_arch}
        if activation_fn is not None:
            policy_kwargs["activation_fn"] = activation_fn

        common = {
            "env": vec_env,
            "policy": "MlpPolicy",
            "gamma": _safe_float(gamma, 0.99),
            "learning_rate": lr,
            "device": self.device,
            "seed": self.seed,
            "verbose": 0,
            "policy_kwargs": policy_kwargs,
        }

        if self.policy_name == "PPO":
            n_steps = max(128, _safe_int(specific_params.get("n_steps", 2048), 2048))
            batch_size = max(32, _safe_int(specific_params.get("batch_size", 256), 256))
            if n_steps < batch_size:
                n_steps = batch_size
            common.update({"n_steps": n_steps, "batch_size": batch_size})

        if self.policy_name in ("SAC", "TD3"):
            common.update(
                {
                    "buffer_size": max(10000, _safe_int(specific_params.get("replay_size", 1000000), 1000000)),
                    "batch_size": max(32, _safe_int(specific_params.get("batch_size", 256), 256)),
                    "learning_starts": max(100, _safe_int(specific_params.get("learning_start", 10000), 10000)),
                    "train_freq": max(1, _safe_int(specific_params.get("learning_frequency", 1), 1)),
                }
            )

        return policy_class(**common)


class HalfCheetahTrainer:
    def __init__(self, base_dir: Optional[str] = None, event_callback: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parent)
        self.event_callback = event_callback
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.transitions: List[Dict[str, Any]] = []
        self.current_env_wrapper: Optional[HalfCheetahEnvironment] = None

    def _emit(self, event_type: str, payload: Dict[str, Any]):
        event = {"type": event_type}
        event.update(payload)
        if self.event_callback:
            self.event_callback(event)

    def request_stop(self):
        self.stop_event.set()
        self.pause_event.set()

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def rebuild_environment(self, env_params: Optional[Dict[str, Any]] = None, render_mode: Optional[str] = "rgb_array"):
        if self.current_env_wrapper is None:
            self.current_env_wrapper = HalfCheetahEnvironment(env_params=env_params, render_mode=render_mode)
            self.current_env_wrapper.ensure()
            return self.current_env_wrapper
        self.current_env_wrapper.update(env_params or {}, render_mode=render_mode)
        return self.current_env_wrapper

    def run_episode(
        self,
        model,
        env_wrapper: HalfCheetahEnvironment,
        max_steps: int,
        deterministic: bool = True,
        collect_transitions: bool = False,
        animation_on: bool = False,
        capture_replay_frames: bool = True,
        rollout_full_capture_steps: int = 120,
        low_overhead_animation: bool = False,
    ) -> Dict[str, Any]:
        env = env_wrapper.ensure()
        obs, _ = env.reset()
        total_reward = 0.0
        executed_steps = 0
        render_state = "off"
        latest_frame = None
        frames: List[Any] = []

        full_capture = max(5, int(rollout_full_capture_steps))

        def _frame_stride(step: int) -> int:
            if step <= full_capture:
                return 1
            if bool(low_overhead_animation):
                return 10
            if step <= full_capture * 2:
                return 2
            if step <= full_capture * 3:
                return 3
            return 5

        for step_idx in range(1, max_steps + 1):
            if self.stop_event.is_set():
                break
            self.pause_event.wait()
            if self.stop_event.is_set():
                break

            action, _ = model.predict(obs, deterministic=deterministic)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            executed_steps = step_idx

            if collect_transitions:
                self.transitions.append(
                    {
                        "step": step_idx,
                        "reward": float(reward),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                    }
                )

            if animation_on:
                try:
                    if capture_replay_frames:
                        stride = _frame_stride(step_idx)
                        if step_idx <= full_capture or step_idx % stride == 0:
                            latest_frame = env.render()
                            if latest_frame is not None:
                                frames.append(latest_frame)
                                render_state = "on"
                            else:
                                render_state = "skipped"
                except Exception:
                    render_state = "skipped"

            obs = next_obs
            if terminated or truncated:
                break

        if not animation_on:
            render_state = "off"
        elif not capture_replay_frames and latest_frame is None:
            try:
                latest_frame = env.render()
                render_state = "on" if latest_frame is not None else "skipped"
            except Exception:
                render_state = "skipped"

        return {
            "reward": float(total_reward),
            "steps": int(executed_steps),
            "render_state": render_state,
            "frame": latest_frame,
            "frames": frames,
        }

    def evaluate_policy(self, model, env_params: Dict[str, Any], episodes: int = 1, max_steps: int = 1000) -> float:
        eval_wrapper = HalfCheetahEnvironment(env_params=env_params, render_mode=None)
        rewards = []
        try:
            for _ in range(max(1, episodes)):
                result = self.run_episode(
                    model=model,
                    env_wrapper=eval_wrapper,
                    max_steps=max_steps,
                    deterministic=True,
                    collect_transitions=False,
                    animation_on=False,
                )
                rewards.append(result["reward"])
        finally:
            eval_wrapper.close()
        return float(np.mean(rewards)) if rewards else 0.0

    def train(self, config_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not _required_packages_present():
            raise RuntimeError("Missing required packages: gymnasium, torch, stable-baselines3.")

        config = TrainerConfig(**config_dict)
        run_id = config.run_id or str(uuid.uuid4())

        env_params = dict(ENV_DEFAULTS)
        env_params.update(config.env_params or {})

        specific = dict(POLICY_DEFAULTS[config.policy])
        specific.update(config.specific_params or {})

        self.stop_event.clear()
        self.pause_event.set()
        self.transitions.clear()

        rollout_render_mode = "rgb_array" if bool(config.animation_on) else None
        train_env_wrapper = HalfCheetahEnvironment(env_params=env_params, render_mode=rollout_render_mode)
        vec_env = train_env_wrapper.make_vectorized()
        agent = SB3PolicyAgent(config.policy, device=config.device, seed=config.seed)
        model = agent.create_model(vec_env=vec_env, gamma=config.gamma, specific_params=specific)

        rewards: List[float] = []
        eval_points: List[Tuple[int, float]] = []
        best_reward = -float("inf")
        epsilon = float(config.epsilon_max)
        last_steps = 0

        try:
            for episode in range(1, config.episodes + 1):
                if self.stop_event.is_set():
                    break
                self.pause_event.wait()
                if self.stop_event.is_set():
                    break

                model.learn(total_timesteps=config.max_steps, reset_num_timesteps=False, progress_bar=False)
                should_replay = bool(config.animation_on and (episode % max(1, config.update_rate_episodes) == 0 or episode == config.episodes))

                rollout = self.run_episode(
                    model=model,
                    env_wrapper=train_env_wrapper,
                    max_steps=config.max_steps,
                    deterministic=False,
                    collect_transitions=False,
                    animation_on=bool(config.animation_on),
                    capture_replay_frames=should_replay,
                    rollout_full_capture_steps=int(config.rollout_full_capture_steps),
                    low_overhead_animation=bool(config.low_overhead_animation),
                )
                reward = float(rollout["reward"])
                steps = int(rollout["steps"])
                last_steps = steps
                rewards.append(reward)
                epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
                best_reward = max(best_reward, reward)

                window = max(1, int(config.moving_average_values))
                moving_average = float(np.mean(rewards[-window:]))

                if episode % max(1, config.eval_interval) == 0:
                    eval_reward = self.evaluate_policy(
                        model=model,
                        env_params=env_params,
                        episodes=max(1, config.eval_episodes),
                        max_steps=config.max_steps,
                    )
                    eval_points.append((episode, eval_reward))

                lr = specific.get("lr", 0.0)
                if hasattr(model, "policy") and getattr(model.policy, "optimizer", None) is not None:
                    try:
                        lr = float(model.policy.optimizer.param_groups[0].get("lr", lr))
                    except Exception:
                        pass

                self._emit(
                    "episode",
                    {
                        "run_id": run_id,
                        "episode": episode,
                        "episodes": config.episodes,
                        "reward": reward,
                        "moving_average": moving_average,
                        "eval_points": list(eval_points),
                        "steps": steps,
                        "epsilon": epsilon,
                        "lr": lr,
                        "best_reward": best_reward,
                        "render_state": rollout["render_state"],
                        "frame": rollout.get("frame"),
                        "frames": list(rollout.get("frames", [])),
                        "policy": config.policy,
                    },
                )

            result = {
                "run_id": run_id,
                "policy": config.policy,
                "episodes_done": len(rewards),
                "rewards": rewards,
                "eval_points": eval_points,
                "best_reward": best_reward if rewards else 0.0,
                "steps": last_steps,
                "stopped": self.stop_event.is_set(),
            }
            self._emit("training_done", result)
            return result
        except Exception as exc:
            self._emit("error", {"run_id": run_id, "message": str(exc)})
            raise
        finally:
            try:
                train_env_wrapper.close()
            except Exception:
                pass
            try:
                vec_env.close()
            except Exception:
                pass

    def export_transitions_csv(self, run_id: Optional[str] = None) -> Optional[Path]:
        if not self.transitions:
            return None
        output_dir = self.base_dir / "results_csv"
        output_dir.mkdir(parents=True, exist_ok=True)
        tag = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"halfcheetah_transitions_{tag}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.transitions[0].keys()))
            writer.writeheader()
            writer.writerows(self.transitions)
        return path


def build_compare_runs(base_config: Dict[str, Any], compare_map: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not compare_map:
        return [base_config]

    keys = list(compare_map.keys())
    values = [compare_map[k] for k in keys]
    runs: List[Dict[str, Any]] = []
    for combo in product(*values):
        cfg = dict(base_config)
        specific = dict(cfg.get("specific_params", {}))
        for key, value in zip(keys, combo):
            if key in GENERAL_DEFAULTS or key in ("policy", "device"):
                cfg[key] = value
            else:
                specific[key] = value
        cfg["specific_params"] = specific
        cfg["run_id"] = str(uuid.uuid4())
        runs.append(cfg)
    return runs


def cap_torch_cpu_threads(max_threads: int = 1):
    if torch is None:
        return
    try:
        torch.set_num_threads(max(1, int(max_threads)))
        torch.set_num_interop_threads(1)
    except Exception:
        pass


__all__ = [
    "ENV_ID",
    "ENV_DEFAULTS",
    "GENERAL_DEFAULTS",
    "POLICY_DEFAULTS",
    "EXPOSED_POLICIES",
    "HalfCheetahEnvironment",
    "HalfCheetahTrainer",
    "SB3PolicyAgent",
    "build_compare_runs",
    "cap_torch_cpu_threads",
]
