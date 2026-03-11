from __future__ import annotations

import csv
import itertools
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable in minimal envs
    torch = None  # type: ignore[assignment]

try:
    from stable_baselines3 import SAC
except Exception:  # pragma: no cover - optional dependency for tests
    SAC = None  # type: ignore[assignment]

try:
    from sb3_contrib import TQC
except Exception:  # pragma: no cover - optional dependency for tests
    TQC = None  # type: ignore[assignment]

try:
    from evotorch import Problem
    from evotorch.algorithms import CMAES
except Exception:  # pragma: no cover - optional dependency for tests
    Problem = None  # type: ignore[assignment]
    CMAES = None  # type: ignore[assignment]

EventSink = Callable[[Dict[str, Any]], None]

ACTIVATION_MAP: Dict[str, Any] = {
    "ReLU": "relu",
    "Tanh": "tanh",
}

POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 200_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1_000,
        "hidden_layer": "256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
    },
    "TQC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 300_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "learning_starts": 1_000,
        "top_quantiles_to_drop_per_net": 2,
        "hidden_layer": "256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
    },
    "CMA-ES": {
        "sigma": 0.18,
        "popsize": 48,
        "number_of_agents": 48,
        "iterations_per_episode": 6,
        "cmaes_eval_horizon": 192,
        "cmaes_rollouts": 2,
        "cmaes_hidden_units": 32,
        "cmaes_init_spread": 0.25,
        "cmaes_action_noise": 0.20,
        "cmaes_elite_k": 5,
        "cmaes_center_momentum": 0.7,
        "cmaes_stagnation_patience": 25,
        "cmaes_restart_spread": 0.5,
        "hidden_layer": "256",
        "activation": "Tanh",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
    },
}


@dataclass
class EnvironmentConfig:
    env_id: str = "Ant-v5"
    max_steps: int = 1_000
    render_mode: str = "rgb_array"
    forward_reward_weight: float = 1.0
    ctrl_cost_weight: float = 0.5
    contact_cost_weight: float = 5e-4


@dataclass
class TrainerConfig:
    policy: str = "SAC"
    episodes: int = 3_000
    device: str = "CPU"
    update_rate: int = 1
    frame_stride: int = 2
    moving_average_window: int = 20
    evaluation_rollout_on: bool = False
    render_enabled: bool = True
    seed: Optional[int] = None
    specific_params: Dict[str, Any] = field(default_factory=dict)


class AntEnvironment:
    def __init__(self, config: EnvironmentConfig):
        self.config = config

    def build(self, render_mode: Optional[str] = None) -> gym.Env:
        kwargs = {
            "max_episode_steps": self.config.max_steps,
            "forward_reward_weight": self.config.forward_reward_weight,
            "ctrl_cost_weight": self.config.ctrl_cost_weight,
            "contact_cost_weight": self.config.contact_cost_weight,
        }
        return gym.make(
            self.config.env_id,
            render_mode=render_mode if render_mode is not None else self.config.render_mode,
            **kwargs,
        )


def parse_hidden_layer(value: Any, fallback: Sequence[int] = (256, 256)) -> List[int]:
    if value is None:
        return list(fallback)
    if isinstance(value, (int, float)):
        width = int(value)
        return [width, width] if width > 0 else list(fallback)
    text = str(value).strip()
    if not text:
        return list(fallback)
    parts = [p.strip() for p in text.split(",") if p.strip()]
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return list(fallback)
    if not nums or any(n <= 0 for n in nums):
        return list(fallback)
    return nums if len(nums) > 1 else [nums[0], nums[0]]


def build_lr_schedule(
    initial_lr: float,
    strategy: str = "constant",
    min_lr: float = 1e-5,
    lr_decay: float = 0.995,
) -> Callable[[float], float]:
    initial_lr = float(initial_lr)
    min_lr = float(min_lr)
    lr_decay = float(lr_decay)
    strategy = (strategy or "constant").lower()

    def _constant(_: float) -> float:
        return max(initial_lr, min_lr)

    def _linear(progress_remaining: float) -> float:
        # SB3 passes progress_remaining from 1.0 down to 0.0
        val = min_lr + (initial_lr - min_lr) * float(progress_remaining)
        return max(val, min_lr)

    def _exponential(progress_remaining: float) -> float:
        steps_done = max(0.0, 1.0 - float(progress_remaining))
        val = initial_lr * (lr_decay ** (steps_done * 1000.0))
        return max(val, min_lr)

    if strategy == "linear":
        return _linear
    if strategy == "exponential":
        return _exponential
    return _constant


class AntPolicyAgent:
    def __init__(self, policy_name: str, env: gym.Env, config: TrainerConfig):
        self.policy_name = policy_name
        self.env = env
        self.config = config
        self.model = None

    def _resolved_device(self) -> str:
        request = (self.config.device or "CPU").upper()
        if request == "GPU" and torch is not None and getattr(torch.cuda, "is_available", lambda: False)():
            return "cuda"
        return "cpu"

    def _shared_kwargs(self) -> Dict[str, Any]:
        p = self.config.specific_params
        if torch is not None:
            cpu_threads = int(p.get("cpu_threads", 0) or 0)
            if cpu_threads > 0:
                torch.set_num_threads(max(1, cpu_threads))
        net_arch = parse_hidden_layer(p.get("hidden_layer", "256"))
        activation_name = p.get("activation", "ReLU")
        activation_fn = activation_name
        if torch is not None:
            activation_fn = {
                "ReLU": torch.nn.ReLU,
                "Tanh": torch.nn.Tanh,
            }.get(activation_name, torch.nn.ReLU)

        learning_rate = build_lr_schedule(
            initial_lr=float(p.get("learning_rate", 3e-4)),
            strategy=str(p.get("lr_strategy", "constant")),
            min_lr=float(p.get("min_lr", 1e-5)),
            lr_decay=float(p.get("lr_decay", 0.995)),
        )
        return {
            "gamma": float(p.get("gamma", 0.99)),
            "learning_rate": learning_rate,
            "batch_size": int(p.get("batch_size", 256)),
            "policy_kwargs": {
                "net_arch": net_arch,
                "activation_fn": activation_fn,
            },
            "device": self._resolved_device(),
            "verbose": 0,
        }

    def create(self) -> Any:
        if self.policy_name == "SAC":
            if SAC is None:
                raise RuntimeError("stable-baselines3 is not available")
            p = self.config.specific_params
            kwargs = self._shared_kwargs()
            kwargs.update(
                {
                    "buffer_size": int(p.get("buffer_size", 200_000)),
                    "tau": float(p.get("tau", 0.005)),
                    "train_freq": int(p.get("train_freq", 1)),
                    "gradient_steps": int(p.get("gradient_steps", 1)),
                    "learning_starts": int(p.get("learning_starts", 1000)),
                }
            )
            self.model = SAC("MlpPolicy", self.env, **kwargs)
            return self.model

        if self.policy_name == "TQC":
            if TQC is None:
                raise RuntimeError("sb3-contrib is not available")
            p = self.config.specific_params
            kwargs = self._shared_kwargs()
            kwargs.update(
                {
                    "buffer_size": int(p.get("buffer_size", 300_000)),
                    "tau": float(p.get("tau", 0.005)),
                    "train_freq": int(p.get("train_freq", 1)),
                    "gradient_steps": int(p.get("gradient_steps", 1)),
                    "learning_starts": int(p.get("learning_starts", 1000)),
                    "top_quantiles_to_drop_per_net": int(
                        p.get("top_quantiles_to_drop_per_net", 2)
                    ),
                }
            )
            self.model = TQC("MlpPolicy", self.env, **kwargs)
            return self.model

        if self.policy_name == "CMA-ES":
            self.model = "cma-es"
            return self.model

        raise ValueError(f"Unsupported policy: {self.policy_name}")


class AntTrainer:
    def __init__(
        self,
        env_config: EnvironmentConfig,
        trainer_config: TrainerConfig,
        event_sink: Optional[EventSink] = None,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        stop_event: Optional[Any] = None,
        pause_event: Optional[Any] = None,
    ):
        self.env_config = env_config
        self.trainer_config = trainer_config
        self.event_sink = event_sink
        self.session_id = session_id or "session"
        self.run_id = run_id or f"run-{int(time.time() * 1000)}"

        self._stop_event = stop_event or threading.Event()
        self._pause_event = pause_event or threading.Event()
        try:
            if not self._pause_event.is_set():
                self._pause_event.set()
        except Exception:
            # Fallback for event-like objects without is_set.
            self._pause_event.set()

        self.rewards: List[float] = []
        self.eval_points: List[Tuple[int, float]] = []
        self.sampled_transitions: List[Dict[str, Any]] = []
        self._cmaes_center: Optional[np.ndarray] = None
        self._cma_archive: List[Tuple[float, np.ndarray]] = []
        self._cma_best_score: float = -float("inf")
        self._cma_no_improve: int = 0

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        self._stop_event.set()
        self._pause_event.set()

    def rebuild_environment(self, updates: Dict[str, Any]) -> None:
        for key, value in updates.items():
            if hasattr(self.env_config, key):
                setattr(self.env_config, key, value)

    def _emit(self, event: Dict[str, Any]) -> None:
        event.setdefault("session_id", self.session_id)
        event.setdefault("run_id", self.run_id)
        if self.event_sink is not None:
            self.event_sink(event)

    def _collect_frames(self, env: gym.Env, frame_stride: int) -> List[np.ndarray]:
        frame_stride = max(1, int(frame_stride))
        frames: List[np.ndarray] = []
        frame = env.render()
        if frame is not None:
            frames.append(np.array(frame))
        if frame_stride <= 1:
            return frames
        return frames[::frame_stride] if len(frames) > 1 else frames

    def run_episode(
        self,
        collect_transitions: bool = False,
        render_frames: bool = False,
        deterministic: bool = False,
    ) -> Dict[str, Any]:
        policy = self.trainer_config.policy
        max_steps = int(self.env_config.max_steps)

        if policy in ("SAC", "TQC"):
            env = self.env_config_wrapper.build(render_mode="rgb_array" if render_frames else None)
            agent = AntPolicyAgent(policy, env, self.trainer_config)
            model = agent.create()
            model.learn(total_timesteps=max_steps, progress_bar=False)

            obs, _ = env.reset(seed=self.trainer_config.seed)
            total_reward = 0.0
            transitions: List[Dict[str, Any]] = []
            frames: List[np.ndarray] = []
            executed_steps = 0
            for step in range(max_steps):
                action, _ = model.predict(obs, deterministic=deterministic)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                if render_frames and step % max(1, int(self.trainer_config.frame_stride)) == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.array(frame))
                if collect_transitions:
                    transitions.append(
                        {
                            "step": step,
                            "reward": float(reward),
                            "obs_mean": float(np.asarray(obs).mean()),
                            "next_obs_mean": float(np.asarray(next_obs).mean()),
                        }
                    )
                total_reward += float(reward)
                executed_steps = step + 1
                obs = next_obs
                if terminated or truncated:
                    break
            env.close()
            return {
                "reward": total_reward,
                "steps": executed_steps,
                "frames": frames,
                "transitions": transitions,
            }

        # CMA-ES path (EvoTorch or safe fallback).
        # Use a non-render env for optimization and a separate env for rollout playback.
        opt_env = self.env_config_wrapper.build(render_mode=None)

        obs0, _ = opt_env.reset(seed=self.trainer_config.seed)
        obs0_vec = np.asarray(obs0, dtype=np.float32).reshape(-1)

        action_shape = opt_env.action_space.shape
        action_dim = int(np.prod(action_shape)) if action_shape else 1
        obs_dim = int(obs0_vec.size)
        hidden_units = max(4, int(self.trainer_config.specific_params.get("cmaes_hidden_units", 32)))
        param_dim = int((obs_dim * hidden_units) + hidden_units + (hidden_units * action_dim) + action_dim)
        action_low = np.asarray(getattr(opt_env.action_space, "low", -1.0), dtype=np.float32).reshape(-1)
        action_high = np.asarray(getattr(opt_env.action_space, "high", 1.0), dtype=np.float32).reshape(-1)
        if action_low.size != action_dim:
            action_low = np.full(action_dim, -1.0, dtype=np.float32)
        if action_high.size != action_dim:
            action_high = np.full(action_dim, 1.0, dtype=np.float32)
        finite_mask = np.isfinite(action_low) & np.isfinite(action_high) & (action_high > action_low)
        if not np.all(finite_mask):
            action_low = np.where(finite_mask, action_low, -1.0).astype(np.float32)
            action_high = np.where(finite_mask, action_high, 1.0).astype(np.float32)

        eval_horizon = max(1, int(self.trainer_config.specific_params.get("cmaes_eval_horizon", 192)))
        eval_rollouts = max(1, int(self.trainer_config.specific_params.get("cmaes_rollouts", 2)))
        action_noise = max(0.0, float(self.trainer_config.specific_params.get("cmaes_action_noise", 0.08)))
        action_span = np.maximum(1e-6, action_high - action_low)
        rng = np.random.default_rng(self.trainer_config.seed)

        def _decode_controller(flat_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            vec = np.asarray(flat_values, dtype=np.float32).reshape(-1)
            if vec.size < param_dim:
                pad = np.zeros(param_dim - vec.size, dtype=np.float32)
                vec = np.concatenate([vec, pad], axis=0)
            elif vec.size > param_dim:
                vec = vec[:param_dim]

            idx = 0
            s1 = obs_dim * hidden_units
            w1 = vec[idx:idx + s1].reshape(obs_dim, hidden_units)
            idx += s1
            b1 = vec[idx:idx + hidden_units]
            idx += hidden_units
            s2 = hidden_units * action_dim
            w2 = vec[idx:idx + s2].reshape(hidden_units, action_dim)
            idx += s2
            b2 = vec[idx:idx + action_dim]
            return w1, b1, w2, b2

        def _policy_action(
            obs_value: np.ndarray,
            w1: np.ndarray,
            b1: np.ndarray,
            w2: np.ndarray,
            b2: np.ndarray,
            stochastic: bool = False,
        ) -> np.ndarray:
            obs_vec = np.asarray(obs_value, dtype=np.float32).reshape(-1)
            if obs_vec.size < obs_dim:
                pad = np.zeros(obs_dim - obs_vec.size, dtype=np.float32)
                obs_vec = np.concatenate([obs_vec, pad], axis=0)
            elif obs_vec.size > obs_dim:
                obs_vec = obs_vec[:obs_dim]

            hid = np.tanh(np.matmul(obs_vec, w1) + b1)
            raw = np.matmul(hid, w2) + b2
            squashed = np.tanh(raw)
            scaled = action_low + ((squashed + 1.0) * 0.5) * (action_high - action_low)
            if stochastic and action_noise > 0.0:
                scaled = scaled + rng.normal(0.0, action_noise * action_span, size=scaled.shape)
            return np.clip(scaled, action_low, action_high).reshape(action_shape)

        def _evaluate_controller(
            w1: np.ndarray,
            b1: np.ndarray,
            w2: np.ndarray,
            b2: np.ndarray,
            stochastic_eval: bool = True,
        ) -> float:
            returns: List[float] = []
            base_seed = self.trainer_config.seed
            for rollout_idx in range(eval_rollouts):
                seed = None if base_seed is None else int(base_seed + rollout_idx)
                obs_eval, _ = opt_env.reset(seed=seed)
                total_reward = 0.0
                for _ in range(eval_horizon):
                    act = _policy_action(obs_eval, w1, b1, w2, b2, stochastic=stochastic_eval)
                    next_obs, reward, terminated, truncated, _ = opt_env.step(act)
                    total_reward += float(reward)
                    obs_eval = next_obs
                    if terminated or truncated:
                        break
                returns.append(total_reward)
            return float(np.mean(returns)) if returns else 0.0

        def objective(solution: Any) -> Any:
            flat = solution.detach().cpu().numpy()
            w1, b1, w2, b2 = _decode_controller(flat)
            score = _evaluate_controller(w1, b1, w2, b2)
            return torch.tensor(-score, dtype=torch.float32)

        iterations = int(self.trainer_config.specific_params.get("iterations_per_episode", 2))
        sigma = float(self.trainer_config.specific_params.get("sigma", 0.25))
        popsize = int(
            self.trainer_config.specific_params.get(
                "number_of_agents",
                self.trainer_config.specific_params.get("popsize", 32),
            )
        )

        best_vector: Optional[np.ndarray] = None
        has_prior_center = self._cmaes_center is not None and int(self._cmaes_center.size) == param_dim

        # Episode reward should reflect the current policy; optimization updates next episode.
        if has_prior_center:
            rollout_vec = np.asarray(self._cmaes_center, dtype=np.float32)
            rollout_w1, rollout_b1, rollout_w2, rollout_b2 = _decode_controller(rollout_vec)
        elif not deterministic:
            init_spread = max(1e-3, float(self.trainer_config.specific_params.get("cmaes_init_spread", 0.25)))
            rollout_vec = rng.normal(0.0, init_spread, size=(param_dim,)).astype(np.float32)
            rollout_w1, rollout_b1, rollout_w2, rollout_b2 = _decode_controller(rollout_vec)
        else:
            rollout_w1 = np.zeros((obs_dim, hidden_units), dtype=np.float32)
            rollout_b1 = np.zeros((hidden_units,), dtype=np.float32)
            rollout_w2 = np.zeros((hidden_units, action_dim), dtype=np.float32)
            rollout_b2 = np.zeros((action_dim,), dtype=np.float32)

        if deterministic and has_prior_center:
            best_vector = np.asarray(self._cmaes_center, dtype=np.float32)
            best_w1, best_b1, best_w2, best_b2 = _decode_controller(best_vector)
        elif deterministic:
            best_w1 = np.zeros((obs_dim, hidden_units), dtype=np.float32)
            best_b1 = np.zeros((hidden_units,), dtype=np.float32)
            best_w2 = np.zeros((hidden_units, action_dim), dtype=np.float32)
            best_b2 = np.zeros((action_dim,), dtype=np.float32)

        elif Problem is not None and CMAES is not None and torch is not None:
            # Controller parameters: [W(obs_dim x action_dim), b(action_dim)].
            if has_prior_center:
                spread = max(1e-3, float(self.trainer_config.specific_params.get("cmaes_init_spread", 0.25)))
                init_lower = np.asarray(self._cmaes_center, dtype=np.float32) - spread
                init_upper = np.asarray(self._cmaes_center, dtype=np.float32) + spread
            else:
                init_lower = np.full((param_dim,), -0.5, dtype=np.float32)
                init_upper = np.full((param_dim,), 0.5, dtype=np.float32)
            problem = Problem(
                "min",
                objective,
                solution_length=param_dim,
                initial_bounds=(init_lower, init_upper),
                device="cpu",
            )
            sigma_scale = 1.0 + (0.1 * min(10, self._cma_no_improve))
            sigma_eff = max(1e-3, float(sigma) * sigma_scale)
            searcher = CMAES(problem, stdev_init=sigma_eff, popsize=popsize)
            for _ in range(max(1, iterations)):
                searcher.step()
            best = searcher.status["best"]
            best_vec = best.values.detach().cpu().numpy().astype(np.float32)
            best_vector = best_vec
            best_w1, best_b1, best_w2, best_b2 = _decode_controller(best_vec)
        else:
            # Fallback random search keeps project usable even without EvoTorch.
            best_reward = -float("inf")
            rng = np.random.default_rng(self.trainer_config.seed)
            for _ in range(max(1, iterations * popsize)):
                cand = rng.uniform(low=-1.0, high=1.0, size=(param_dim,))
                cand_w1, cand_b1, cand_w2, cand_b2 = _decode_controller(cand)
                total_reward = _evaluate_controller(cand_w1, cand_b1, cand_w2, cand_b2)
                if total_reward > best_reward:
                    best_reward = float(total_reward)
                    best_w1 = cand_w1
                    best_b1 = cand_b1
                    best_w2 = cand_w2
                    best_b2 = cand_b2
                    best_vector = np.asarray(cand, dtype=np.float32)

        if not all(k in locals() for k in ["best_w1", "best_b1", "best_w2", "best_b2"]):
            best_w1 = np.zeros((obs_dim, hidden_units), dtype=np.float32)
            best_b1 = np.zeros((hidden_units,), dtype=np.float32)
            best_w2 = np.zeros((hidden_units, action_dim), dtype=np.float32)
            best_b2 = np.zeros((action_dim,), dtype=np.float32)

        if best_vector is None:
            best_vector = np.concatenate(
                [best_w1.reshape(-1), best_b1.reshape(-1), best_w2.reshape(-1), best_b2.reshape(-1)],
                axis=0,
            ).astype(np.float32)

        best_score = _evaluate_controller(best_w1, best_b1, best_w2, best_b2, stochastic_eval=False)
        elite_k = max(1, int(self.trainer_config.specific_params.get("cmaes_elite_k", 5)))
        self._cma_archive.append((best_score, np.asarray(best_vector, dtype=np.float32)))
        self._cma_archive.sort(key=lambda x: x[0], reverse=True)
        self._cma_archive = self._cma_archive[:elite_k]

        elite_vectors = [vec for _score, vec in self._cma_archive]
        elite_center = np.mean(np.stack(elite_vectors, axis=0), axis=0).astype(np.float32)

        momentum = float(self.trainer_config.specific_params.get("cmaes_center_momentum", 0.7))
        momentum = max(0.0, min(0.99, momentum))
        if self._cmaes_center is not None and int(self._cmaes_center.size) == int(elite_center.size):
            center = (momentum * self._cmaes_center) + ((1.0 - momentum) * elite_center)
        else:
            center = elite_center

        if best_score > (self._cma_best_score + 1e-6):
            self._cma_best_score = float(best_score)
            self._cma_no_improve = 0
        else:
            self._cma_no_improve += 1

        patience = max(1, int(self.trainer_config.specific_params.get("cmaes_stagnation_patience", 25)))
        if self._cma_no_improve >= patience:
            restart_spread = max(1e-3, float(self.trainer_config.specific_params.get("cmaes_restart_spread", 0.5)))
            center = center + rng.normal(0.0, restart_spread, size=center.shape).astype(np.float32)
            self._cma_no_improve = 0

        self._cmaes_center = np.asarray(center, dtype=np.float32)

        opt_env.close()

        env = self.env_config_wrapper.build(render_mode="rgb_array" if render_frames else None)
        obs, _ = env.reset(seed=self.trainer_config.seed)
        total_reward = 0.0
        transitions: List[Dict[str, Any]] = []
        frames: List[np.ndarray] = []
        executed_steps = 0
        frame_stride = max(1, int(self.trainer_config.frame_stride))
        for step in range(max_steps):
            best_action = _policy_action(
                obs,
                rollout_w1,
                rollout_b1,
                rollout_w2,
                rollout_b2,
                stochastic=not deterministic,
            )
            next_obs, reward, terminated, truncated, _ = env.step(best_action)
            if render_frames and step % frame_stride == 0:
                frame = env.render()
                if frame is not None:
                    frames.append(np.array(frame))
            if collect_transitions:
                transitions.append(
                    {
                        "step": step,
                        "reward": float(reward),
                        "obs_mean": float(np.asarray(obs).mean()),
                        "next_obs_mean": float(np.asarray(next_obs).mean()),
                    }
                )
            total_reward += float(reward)
            executed_steps = step + 1

            obs = next_obs
            if terminated or truncated:
                break

        # Keep animation fluid after early termination without altering episode reward.
        if render_frames and executed_steps < max_steps:
            vis_obs, _ = env.reset(seed=None)
            for vis_step in range(executed_steps, max_steps):
                vis_action = _policy_action(
                    vis_obs,
                    rollout_w1,
                    rollout_b1,
                    rollout_w2,
                    rollout_b2,
                    stochastic=not deterministic,
                )
                vis_next_obs, _reward, vis_terminated, vis_truncated, _ = env.step(vis_action)
                if vis_step % frame_stride == 0:
                    frame = env.render()
                    if frame is not None:
                        frames.append(np.array(frame))
                if vis_terminated or vis_truncated:
                    vis_obs, _ = env.reset(seed=None)
                else:
                    vis_obs = vis_next_obs
        env.close()

        return {
            "reward": total_reward,
            "steps": executed_steps,
            "frames": frames,
            "transitions": transitions,
        }

    @property
    def env_config_wrapper(self) -> AntEnvironment:
        return AntEnvironment(self.env_config)

    def evaluate_policy(self) -> float:
        result = self.run_episode(
            collect_transitions=False,
            render_frames=False,
            deterministic=True,
        )
        return float(result["reward"])

    def train(self, collect_transitions: bool = False) -> Dict[str, Any]:
        episodes = int(self.trainer_config.episodes)
        update_rate = max(1, int(self.trainer_config.update_rate))
        ma_window = max(1, int(self.trainer_config.moving_average_window))
        rewards_window: Deque[float] = deque(maxlen=ma_window)
        best_reward = -float("inf")

        self.rewards.clear()
        self.eval_points.clear()
        self.sampled_transitions.clear()

        try:
            for ep in range(1, episodes + 1):
                if self._stop_event.is_set():
                    break
                self._pause_event.wait()

                render_this = bool(self.trainer_config.render_enabled) and (ep % update_rate == 0 or ep == episodes)
                result = self.run_episode(
                    collect_transitions=collect_transitions,
                    render_frames=render_this,
                    deterministic=False,
                )
                reward = float(result["reward"])
                steps = int(result["steps"])
                self.rewards.append(reward)
                rewards_window.append(reward)
                moving_average = float(np.mean(rewards_window))
                best_reward = max(best_reward, reward)

                if collect_transitions and result["transitions"]:
                    self.sampled_transitions.extend(result["transitions"])

                if self.trainer_config.evaluation_rollout_on and ep % 10 == 0:
                    eval_reward = self.evaluate_policy()
                    self.eval_points.append((ep, eval_reward))

                lr_value = float(self.trainer_config.specific_params.get("learning_rate", 0.0))
                event = {
                    "type": "episode",
                    "episode": ep,
                    "episodes": episodes,
                    "reward": reward,
                    "moving_average": moving_average,
                    "steps": steps,
                    "epsilon": 0.0,
                    "lr": lr_value,
                    "best_reward": best_reward,
                    "render_state": "on" if render_this else "off",
                }
                self._emit(event)

                aux = {
                    "type": "episode_aux",
                    "episode": ep,
                    "episodes": episodes,
                    "eval_points": list(self.eval_points),
                    "frames": result["frames"],
                }
                self._emit(aux)

            done = {
                "type": "training_done",
                "episode": min(len(self.rewards), episodes),
                "episodes": episodes,
                "status": "canceled" if self._stop_event.is_set() else "completed",
                "rewards": list(self.rewards),
                "eval_points": list(self.eval_points),
            }
            self._emit(done)
            return done
        except Exception as exc:
            self._emit({"type": "error", "message": str(exc)})
            raise

    def export_sampled_transitions_csv(self, output_dir: Path) -> Optional[Path]:
        if not self.sampled_transitions:
            return None
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = output_dir / f"ant_transitions_{self.run_id}_{ts}.csv"
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=["step", "reward", "obs_mean", "next_obs_mean"],
            )
            writer.writeheader()
            writer.writerows(self.sampled_transitions)
        return out_path


def expand_compare_runs(base: Dict[str, Any], compare_lists: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    if not compare_lists:
        return [dict(base)]
    keys = sorted(compare_lists.keys())
    values = [compare_lists[k] for k in keys]
    runs: List[Dict[str, Any]] = []
    for combo in itertools.product(*values):
        row = dict(base)
        for k, v in zip(keys, combo):
            row[k] = v
        runs.append(row)
    return runs
