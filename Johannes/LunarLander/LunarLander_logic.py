from __future__ import annotations

import csv
import math
import os
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import A2C, DQN, PPO, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

try:
    from sb3_contrib import TRPO
except Exception:  # pragma: no cover
    TRPO = None


_DEVICE_PREFERENCE = "cpu"


def set_device_preference(use_gpu: bool) -> torch.device:
    global _DEVICE_PREFERENCE
    _DEVICE_PREFERENCE = "cuda" if bool(use_gpu) else "cpu"
    return get_device()


def get_device() -> torch.device:
    if _DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_hidden_layers(raw: str | Sequence[int] | int) -> List[int]:
    if isinstance(raw, int):
        return [raw]
    if isinstance(raw, (list, tuple)):
        vals = [int(v) for v in raw if int(v) > 0]
        return vals or [128]
    vals = [int(v.strip()) for v in str(raw).split(",") if v.strip()]
    vals = [v for v in vals if v > 0]
    return vals or [128]


def make_activation(name: str) -> nn.Module:
    mapping = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
    }
    return mapping.get(name, nn.ReLU)()


def _activation_class(name: str):
    mapping = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "LeakyReLU": nn.LeakyReLU,
        "ELU": nn.ELU,
    }
    return mapping.get(name, nn.ReLU)


@dataclass
class AgentConfig:
    gamma: float = 0.99
    learning_rate: float = 1e-3
    replay_size: int = 50000
    batch_size: int = 64
    target_update: int = 100
    replay_warmup: int = 1000
    learning_cadence: int = 2
    activation_function: str = "ReLU"
    hidden_layers: str = "128"
    lr_strategy: str = "exponential"
    lr_decay: float = 0.1
    min_learning_rate: float = 1e-5
    gae_lambda: float = 0.95
    ppo_clip_range: float = 0.2


POLICY_DEFAULTS: Dict[str, AgentConfig] = {
    "DuelingDQN": AgentConfig(gamma=0.99, learning_rate=3e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=2, activation_function="ReLU", hidden_layers="256,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "D3QN": AgentConfig(gamma=0.99, learning_rate=2.5e-4, replay_size=150000, batch_size=128, target_update=200, replay_warmup=8000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "DDQN+PER": AgentConfig(gamma=0.99, learning_rate=2e-4, replay_size=200000, batch_size=128, target_update=200, replay_warmup=10000, learning_cadence=2, activation_function="ReLU", hidden_layers="512,256,128", lr_strategy="exponential", lr_decay=0.1, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "PPO": AgentConfig(gamma=0.99, learning_rate=3e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=256, activation_function="ReLU", hidden_layers="256,256", lr_strategy="linear", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "A2C": AgentConfig(gamma=0.99, learning_rate=3e-4, replay_size=100000, batch_size=128, target_update=200, replay_warmup=5000, learning_cadence=64, activation_function="ReLU", hidden_layers="256,256", lr_strategy="exponential", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=1.0, ppo_clip_range=0.2),
    "TRPO": AgentConfig(gamma=0.99, learning_rate=1e-4, replay_size=120000, batch_size=64, target_update=200, replay_warmup=6000, learning_cadence=256, activation_function="ReLU", hidden_layers="256,256", lr_strategy="linear", lr_decay=0.4, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
    "SAC": AgentConfig(gamma=0.99, learning_rate=1e-4, replay_size=200000, batch_size=128, target_update=200, replay_warmup=10000, learning_cadence=32, activation_function="ReLU", hidden_layers="256,256", lr_strategy="cosine", lr_decay=0.3, min_learning_rate=1e-5, gae_lambda=0.95, ppo_clip_range=0.2),
}

DISCRETE_POLICIES = ["DuelingDQN", "D3QN", "DDQN+PER"]
CONTINUOUS_POLICIES = ["PPO", "A2C", "TRPO", "SAC"]


class _SB3ProgressCallback(BaseCallback):
    def __init__(self, progress_callback: Optional[Callable[[int], None]], offset: int = 0) -> None:
        super().__init__(verbose=0)
        self.progress_callback = progress_callback
        self.offset = int(offset)

    def _on_step(self) -> bool:
        if self.progress_callback is not None:
            self.progress_callback(int(self.num_timesteps) + self.offset)
        return True


class SB3PolicyAgent:
    def __init__(
        self,
        policy_name: str,
        action_dim: int,
        config: AgentConfig,
        env_builder: Callable[[], gym.Env],
        planned_steps: int = 100000,
    ) -> None:
        self.policy_name = str(policy_name)
        self.action_dim = int(action_dim)
        self.config = config
        self.device = str(get_device())

        self._base_lr = float(config.learning_rate)
        self._min_lr = max(0.0, float(config.min_learning_rate))
        self._target_lr = max(self._min_lr, self._base_lr * max(0.0, float(config.lr_decay)))
        self._lr_strategy = str(config.lr_strategy).strip().lower()
        self._planned_decay_steps = max(1, int(planned_steps))
        self._current_lr = self._base_lr
        self._best_loss = float("inf")
        self._loss_bad_steps = 0
        self._loss_patience = 100
        self._loss_tolerance = 1e-4

        self._gae_lambda = float(config.gae_lambda)
        self._ppo_clip_range = float(config.ppo_clip_range)
        self.learn_steps = 0

        self._vec_env = DummyVecEnv([env_builder])
        self.model = self._build_model()

    def _build_model(self):
        hidden = parse_hidden_layers(self.config.hidden_layers)
        policy_kwargs = {
            "activation_fn": _activation_class(self.config.activation_function),
            "net_arch": hidden,
        }

        if self.policy_name in DISCRETE_POLICIES:
            return DQN(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self._base_lr,
                gamma=float(self.config.gamma),
                batch_size=max(1, int(self.config.batch_size)),
                buffer_size=max(1000, int(self.config.replay_size)),
                learning_starts=max(0, int(self.config.replay_warmup)),
                train_freq=max(1, int(self.config.learning_cadence)),
                target_update_interval=max(1, int(self.config.target_update)),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=self.device,
            )

        if self.policy_name == "SAC":
            return SAC(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self._base_lr,
                gamma=float(self.config.gamma),
                batch_size=max(1, int(self.config.batch_size)),
                buffer_size=max(1000, int(self.config.replay_size)),
                learning_starts=max(0, int(self.config.replay_warmup)),
                train_freq=max(1, int(self.config.learning_cadence)),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=self.device,
            )

        if self.policy_name == "A2C":
            return A2C(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self._base_lr,
                gamma=float(self.config.gamma),
                gae_lambda=float(self.config.gae_lambda),
                n_steps=max(8, int(self.config.learning_cadence)),
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=self.device,
            )

        if self.policy_name == "TRPO" and TRPO is not None:
            trpo_n_steps = max(32, int(self.config.learning_cadence))
            trpo_batch_size = max(16, min(int(self.config.batch_size), trpo_n_steps))
            return TRPO(
                "MlpPolicy",
                self._vec_env,
                learning_rate=self._base_lr,
                gamma=float(self.config.gamma),
                gae_lambda=float(self.config.gae_lambda),
                n_steps=trpo_n_steps,
                batch_size=trpo_batch_size,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=self.device,
            )

        ppo_n_steps = max(32, int(self.config.learning_cadence))
        ppo_batch_size = max(16, min(int(self.config.batch_size), ppo_n_steps))
        return PPO(
            "MlpPolicy",
            self._vec_env,
            learning_rate=self._base_lr,
            gamma=float(self.config.gamma),
            gae_lambda=float(self.config.gae_lambda),
            clip_range=float(self.config.ppo_clip_range),
            n_steps=ppo_n_steps,
            batch_size=ppo_batch_size,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=self.device,
        )

    def _schedule_progress(self) -> float:
        return min(1.0, max(0.0, float(self.learn_steps) / float(self._planned_decay_steps)))

    def _scheduled_lr(self) -> float:
        progress = self._schedule_progress()
        if self._lr_strategy == "linear":
            return self._base_lr + (self._target_lr - self._base_lr) * progress
        if self._lr_strategy == "cosine":
            return self._target_lr + 0.5 * (self._base_lr - self._target_lr) * (1.0 + math.cos(math.pi * progress))
        ratio = self._target_lr / max(self._base_lr, 1e-12)
        return self._base_lr * (ratio ** progress)

    def _iter_optimizers(self):
        policy_opt = getattr(getattr(self.model, "policy", None), "optimizer", None)
        actor_opt = getattr(getattr(self.model, "actor", None), "optimizer", None)
        critic_opt = getattr(getattr(self.model, "critic", None), "optimizer", None)
        ent_opt = getattr(self.model, "ent_coef_optimizer", None)
        return [opt for opt in [policy_opt, actor_opt, critic_opt, ent_opt] if opt is not None]

    def _set_optimizer_lr(self, value: float) -> None:
        lr = max(self._min_lr, float(value))
        self._current_lr = lr
        for opt in self._iter_optimizers():
            for group in opt.param_groups:
                group["lr"] = lr

    def _step_lr_schedule(self, synthetic_loss: float) -> None:
        if self._lr_strategy == "loss-based":
            if synthetic_loss + self._loss_tolerance < self._best_loss:
                self._best_loss = synthetic_loss
                self._loss_bad_steps = 0
            else:
                self._loss_bad_steps += 1
            if self._loss_bad_steps >= self._loss_patience:
                factor = min(0.99, max(0.1, float(self.config.lr_decay)))
                self._set_optimizer_lr(self._current_lr * factor)
                self._loss_bad_steps = 0
            return

        if self._lr_strategy == "guarded natural gradient":
            baseline = self._scheduled_lr()
            guard_strength = max(0.0, float(self.config.lr_decay))
            self._set_optimizer_lr(baseline / (1.0 + guard_strength))
            return

        self._set_optimizer_lr(self._scheduled_lr())

    def select_action(self, state: np.ndarray, _epsilon: float, deterministic: bool = False):
        action, _ = self.model.predict(np.asarray(state, dtype=np.float32), deterministic=deterministic)
        if self.policy_name in DISCRETE_POLICIES:
            return int(action)
        return np.asarray(action, dtype=np.float32)

    def learn(self, total_timesteps: int, progress_callback: Optional[Callable[[int], None]] = None, offset: int = 0) -> None:
        self.learn_steps += 1
        synthetic_loss = 1.0 / float(self.learn_steps + 1)
        self._step_lr_schedule(synthetic_loss)
        callback = _SB3ProgressCallback(progress_callback, offset=offset) if progress_callback is not None else None
        self.model.learn(
            total_timesteps=max(1, int(total_timesteps)),
            reset_num_timesteps=False,
            progress_bar=False,
            callback=callback,
        )

    def close(self) -> None:
        try:
            self._vec_env.close()
        except Exception:
            pass


class LunarLanderEnv:
    def __init__(
        self,
        gravity: float = -10.0,
        continuous: bool = False,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        render_mode: Optional[str] = "rgb_array",
    ) -> None:
        self.gravity = float(gravity)
        self.continuous = bool(continuous)
        self.enable_wind = bool(enable_wind)
        self.wind_power = float(wind_power)
        self.turbulence_power = float(turbulence_power)
        self.render_mode = render_mode
        self._env_lock = threading.RLock()
        self._make_env()

    def make_gym_env(self, render_mode: Optional[str] = None) -> gym.Env:
        return gym.make(
            "LunarLander-v3",
            continuous=self.continuous,
            gravity=self.gravity,
            enable_wind=self.enable_wind,
            wind_power=self.wind_power,
            turbulence_power=self.turbulence_power,
            render_mode=render_mode,
        )

    def _make_env(self) -> None:
        with self._env_lock:
            if hasattr(self, "env") and self.env is not None:
                self.env.close()
            self.env = self.make_gym_env(render_mode=self.render_mode)
            self.observation_space = self.env.observation_space
            self.action_space = self.env.action_space
            self.state_dim = int(self.env.observation_space.shape[0])
            if self.continuous:
                self.action_dim = int(self.env.action_space.shape[0])
            else:
                self.action_dim = int(self.env.action_space.n)
            self.last_obs = None
            self.last_frame = None

    def update_config(self, gravity: float, continuous: bool, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.gravity = float(gravity)
        self.continuous = bool(continuous)
        self.enable_wind = bool(enable_wind)
        self.wind_power = float(wind_power)
        self.turbulence_power = float(turbulence_power)
        self._make_env()

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        with self._env_lock:
            obs, _ = self.env.reset(seed=seed)
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return self.last_obs.copy()

    def step(self, action: Union[int, np.ndarray]) -> Tuple[np.ndarray, float, bool, dict]:
        with self._env_lock:
            if self.continuous:
                arr = np.asarray(action, dtype=np.float32).reshape(-1)
                if arr.size == 1:
                    arr = np.repeat(arr, self.action_dim)
                if arr.size != self.action_dim:
                    arr = np.resize(arr, self.action_dim)
                env_action = np.clip(arr, -1.0, 1.0)
            else:
                env_action = int(action)
            obs, reward, terminated, truncated, info = self.env.step(env_action)
            done = bool(terminated or truncated)
            self.last_obs = np.asarray(obs, dtype=np.float32)
            return self.last_obs.copy(), float(reward), done, info

    def render(self) -> Optional[np.ndarray]:
        acquired = self._env_lock.acquire(blocking=False)
        if not acquired:
            if self.last_frame is None:
                return None
            return self.last_frame.copy()
        try:
            frame = self.env.render()
            if frame is None:
                if self.last_frame is None:
                    return None
                return self.last_frame.copy()
            self.last_frame = np.asarray(frame)
            return self.last_frame.copy()
        finally:
            self._env_lock.release()

    def is_reachable(self, obs: np.ndarray) -> bool:
        arr = np.asarray(obs, dtype=np.float32)
        if arr.size < 2:
            return False
        return bool(np.isfinite(arr[0]) and np.isfinite(arr[1]) and abs(arr[0]) < 5.0 and abs(arr[1]) < 5.0)

    def close(self) -> None:
        with self._env_lock:
            if getattr(self, "env", None) is not None:
                self.env.close()
                self.env = None


class Trainer:
    def __init__(self, env: Optional[LunarLanderEnv] = None, env_factory: Optional[Callable[[], LunarLanderEnv]] = None) -> None:
        if env is not None:
            self.env = env
        elif env_factory is not None:
            self.env = env_factory()
        else:
            self.env = LunarLanderEnv()
        self.policy_configs: Dict[str, AgentConfig] = {k: AgentConfig(**vars(v)) for k, v in POLICY_DEFAULTS.items()}
        self.agents: Dict[str, SB3PolicyAgent] = {}
        self._training_plan_steps: Dict[str, int] = {}
        self._csv_samples: List[Tuple[int, int, float, float, float, bool]] = []

    def set_policy_config(self, policy: str, **kwargs: object) -> None:
        config = self.policy_configs[policy]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)

    def get_policy_config(self, policy: str) -> AgentConfig:
        return AgentConfig(**vars(self.policy_configs[policy]))

    def get_current_learning_rate(self, policy: str) -> float:
        agent = self.agents.get(policy)
        if agent is None:
            return float(self.policy_configs[policy].learning_rate)
        return float(agent._current_lr)

    def _ensure_policy_environment_mode(self, policy: str) -> None:
        if not isinstance(self.env, LunarLanderEnv):
            return
        desired_continuous = policy in CONTINUOUS_POLICIES
        if bool(self.env.continuous) == bool(desired_continuous):
            return
        self.rebuild_environment(
            gravity=float(self.env.gravity),
            continuous=bool(desired_continuous),
            enable_wind=bool(self.env.enable_wind),
            wind_power=float(self.env.wind_power),
            turbulence_power=float(self.env.turbulence_power),
        )

    def _make_sb3_env_builder(self, policy: str) -> Callable[[], gym.Env]:
        if isinstance(self.env, LunarLanderEnv):
            return lambda: self.env.make_gym_env(render_mode=None)

        base_env = self.env
        obs_shape = tuple(getattr(getattr(base_env, "observation_space", None), "shape", (int(getattr(base_env, "state_dim", 8)),)))
        act_space_obj = getattr(base_env, "action_space", None)
        is_continuous_policy = policy in CONTINUOUS_POLICIES
        fallback_action_dim = int(getattr(base_env, "action_dim", 2 if is_continuous_policy else 4))

        if is_continuous_policy:
            if isinstance(act_space_obj, spaces.Box):
                action_space = act_space_obj
            else:
                action_space = spaces.Box(low=-1.0, high=1.0, shape=(fallback_action_dim,), dtype=np.float32)
        elif isinstance(act_space_obj, spaces.Space):
            action_space = act_space_obj
        elif hasattr(act_space_obj, "n") and act_space_obj.n is not None:
            action_space = spaces.Discrete(int(act_space_obj.n))
        elif hasattr(act_space_obj, "shape") and act_space_obj.shape is not None:
            action_space = spaces.Box(low=-1.0, high=1.0, shape=tuple(act_space_obj.shape), dtype=np.float32)
        else:
            action_space = spaces.Discrete(int(getattr(base_env, "action_dim", 4)))

        observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32)

        class _AdapterEnv(gym.Env):
            metadata = {"render_modes": [None]}

            def __init__(self) -> None:
                super().__init__()
                self._wrapped = base_env
                self.observation_space = observation_space
                self.action_space = action_space

            def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
                obs = self._wrapped.reset(seed=seed)
                return np.asarray(obs, dtype=np.float32), {}

            def step(self, action):
                result = self._wrapped.step(action)
                if len(result) == 5:
                    obs, reward, terminated, truncated, info = result
                    return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), bool(truncated), info
                obs, reward, done, info = result
                done_bool = bool(done)
                return np.asarray(obs, dtype=np.float32), float(reward), done_bool, False, info

            def render(self):
                if hasattr(self._wrapped, "render"):
                    return self._wrapped.render()
                return None

            def close(self):
                if hasattr(self._wrapped, "close"):
                    self._wrapped.close()

        return _AdapterEnv

    def _get_or_create_agent(self, policy: str) -> SB3PolicyAgent:
        self._ensure_policy_environment_mode(policy)

        if policy in self.agents:
            return self.agents[policy]

        config = self.get_policy_config(policy)
        planned_steps = self._training_plan_steps.get(policy, 100000)
        agent = SB3PolicyAgent(
            policy_name=policy,
            action_dim=int(self.env.action_dim),
            config=config,
            env_builder=self._make_sb3_env_builder(policy),
            planned_steps=planned_steps,
        )
        self.agents[policy] = agent
        return agent

    def set_training_plan(self, policy: str, episodes: int, max_steps: int) -> None:
        cadence = max(1, int(self.policy_configs[policy].learning_cadence))
        total_steps = max(1, int(episodes) * int(max_steps))
        self._training_plan_steps[policy] = max(1, total_steps // cadence)

    def reset_policy_agent(self, policy: str) -> None:
        agent = self.agents.pop(policy, None)
        if agent is not None:
            agent.close()

    def rebuild_environment(self, gravity: float, continuous: bool, enable_wind: bool, wind_power: float, turbulence_power: float) -> None:
        self.env.update_config(gravity, continuous, enable_wind, wind_power, turbulence_power)
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()

    def run_episode(
        self,
        policy: str,
        epsilon: float = 0.1,
        max_steps: int = 1000,
        progress_callback: Optional[Callable[[int], None]] = None,
        learn: bool = True,
        deterministic: bool = False,
        seed: Optional[int] = None,
        report_learning_progress: bool = False,
        collect_transitions: bool = False,
    ) -> Dict[str, object]:
        agent = self._get_or_create_agent(policy)

        if learn:
            cadence = max(1, int(self.policy_configs[policy].learning_cadence))
            requested_steps = max(1, int(max_steps))
            train_steps = (requested_steps // cadence) * cadence
            if train_steps <= 0:
                train_steps = requested_steps
            learn_progress_callback = progress_callback if report_learning_progress else None
            agent.learn(total_timesteps=train_steps, progress_callback=learn_progress_callback, offset=0)

        state = self.env.reset(seed=seed)
        total_reward = 0.0
        transitions: List[Tuple[np.ndarray, Union[int, np.ndarray], float, np.ndarray, bool]] = [] if collect_transitions else []
        best_x = float(state[0]) if len(state) > 0 else float("nan")

        for step in range(1, max_steps + 1):
            action = agent.select_action(state, epsilon, deterministic=deterministic)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += float(reward)
            best_x = max(best_x, float(next_state[0])) if len(next_state) > 0 else best_x
            if collect_transitions:
                transition = (state.copy(), action, float(reward), next_state.copy(), bool(done))
                transitions.append(transition)
            if progress_callback is not None:
                progress_callback(step)
            state = next_state
            if done:
                break

        return {
            "reward": float(total_reward),
            "steps": int(len(transitions)),
            "best_x": float(best_x),
            "final_state": state,
            "transitions": transitions,
        }

    def evaluate_policy(
        self,
        policy: str,
        max_steps: int,
        episodes: int = 1,
        seed_base: Optional[int] = None,
    ) -> Dict[str, object]:
        eval_rewards: List[float] = []
        eval_steps: List[int] = []
        for idx in range(max(1, int(episodes))):
            seed = None if seed_base is None else int(seed_base) + idx
            result = self.run_episode(
                policy=policy,
                epsilon=0.0,
                max_steps=max_steps,
                progress_callback=None,
                learn=False,
                deterministic=True,
                seed=seed,
            )
            eval_rewards.append(float(result["reward"]))
            eval_steps.append(int(result["steps"]))

        return {
            "rewards": eval_rewards,
            "steps": eval_steps,
            "mean_reward": float(np.mean(eval_rewards)) if eval_rewards else float("nan"),
            "median_reward": float(np.median(eval_rewards)) if eval_rewards else float("nan"),
        }

    def train(
        self,
        policy: str,
        num_episodes: int,
        max_steps: int,
        epsilon: float,
        save_csv: Optional[str] = None,
    ) -> List[float]:
        rewards: List[float] = []
        csv_rows: List[Tuple[int, int, float, float, float, bool]] = []

        for episode_idx in range(1, num_episodes + 1):
            collect_transitions = save_csv is not None
            result = self.run_episode(
                policy,
                epsilon=epsilon,
                max_steps=max_steps,
                learn=True,
                collect_transitions=collect_transitions,
            )
            rewards.append(float(result["reward"]))
            if save_csv is not None:
                transitions = result["transitions"]
                for step_idx, transition in enumerate(transitions, start=1):
                    state, action, reward, next_state, done = transition
                    sx = float(state[0]) if len(state) > 0 else float("nan")
                    nx = float(next_state[0]) if len(next_state) > 0 else float("nan")
                    action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
                    action_scalar = float(action_arr[0]) if action_arr.size > 0 else float("nan")
                    csv_rows.append((episode_idx, step_idx, sx, action_scalar, float(reward), bool(done)))
                    self._csv_samples.append((episode_idx, step_idx, sx, nx, float(reward), bool(done)))

        if save_csv is not None:
            base_name = os.path.basename(save_csv)
            if not base_name:
                base_name = f"samplings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            out_dir = os.path.join(os.path.dirname(__file__), "results_csv")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{base_name}.csv")
            with open(out_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["episode", "step", "state_x", "action", "reward", "done"])
                writer.writerows(csv_rows)

        return rewards

    def save_plot_png(
        self,
        figure,
        policy: str,
        eps_max: float,
        eps_min: float,
        learning_rate: float,
        gamma: float,
        episodes: int,
        max_steps: int,
    ) -> str:
        def _sci_token(value: float) -> str:
            return f"{float(value):.2e}".replace("+", "")

        out_dir = os.path.join(os.path.dirname(__file__), "plots")
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_policy = policy.replace("+", "plus")
        lr_token = _sci_token(learning_rate)
        name = (
            f"{safe_policy}_eps{eps_max}_{eps_min}_lr{lr_token}_"
            f"g{gamma}_ep{episodes}_ms{max_steps}_{ts}.png"
        )
        path = os.path.join(out_dir, name)
        figure.savefig(path, dpi=150, bbox_inches="tight")
        return path

    def close(self) -> None:
        for agent in self.agents.values():
            agent.close()
        self.agents.clear()
        self.env.close()
