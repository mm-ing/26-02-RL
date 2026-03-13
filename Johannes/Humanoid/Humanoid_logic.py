from __future__ import annotations

import csv
import itertools
import os
import threading
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.type_aliases import Schedule

# GLFW teardown warnings are benign during env shutdown on Windows.
warnings.filterwarnings("ignore", message=r".*The GLFW library is not initialized.*", category=Warning)

PROJECT_NAME = "Humanoid"
ENV_ID = "Humanoid-v5"

EventSink = Callable[[Dict[str, Any]], None]

POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 64,
        "n_steps": 2048,
        "ent_coef": 0.0,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.999,
    },
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "learning_starts": 5_000,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": 1,
        "ent_coef": "auto",
        "use_sde": True,
        "sde_sample_freq": 4,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.999,
    },
    "TD3": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "buffer_size": 1_000_000,
        "learning_starts": 5_000,
        "tau": 0.005,
        "policy_delay": 2,
        "train_freq": (1, "step"),
        "gradient_steps": 1,
        "target_policy_noise": 0.2,
        "target_noise_clip": 0.5,
        "action_noise_sigma": 0.1,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.999,
    },
}


@dataclass(frozen=True)
class EnvironmentConfig:
    forward_reward_weight: float = 1.25
    ctrl_cost_weight: float = 0.1
    contact_cost_weight: float = 5e-7
    contact_cost_range_low: float = float("-inf")
    contact_cost_range_high: float = 10.0
    healthy_reward: float = 5.0
    terminate_when_unhealthy: bool = True
    healthy_z_range_low: float = 1.0
    healthy_z_range_high: float = 2.0
    render_mode: Optional[str] = None


@dataclass(frozen=True)
class NetworkConfig:
    hidden_layer: str = "256,256"
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
    episodes: int = 4000
    max_steps: int = 1000
    update_rate: int = 1
    frame_stride: int = 2
    deterministic_eval_every: int = 10
    deterministic_eval_episodes: int = 1
    seed: Optional[int] = 42
    collect_transitions: bool = False
    export_csv: bool = False
    split_aux_events: bool = False
    device: str = "CPU"
    session_id: str = "default-session"
    run_id: str = "default-run"
    env: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    lr: LearningRateConfig = field(default_factory=LearningRateConfig)
    policy_params: Dict[str, Any] = field(default_factory=dict)


class HumanoidEnvWrapper:
    """Light wrapper around gymnasium.make to support runtime env updates."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self._config = config

    @property
    def config(self) -> EnvironmentConfig:
        return self._config

    def rebuild(self, config: EnvironmentConfig) -> None:
        self._config = config

    def make_env_with_render_mode(self, render_mode: Optional[str]) -> gym.Env:
        kwargs: Dict[str, Any] = {
            "forward_reward_weight": self._config.forward_reward_weight,
            "ctrl_cost_weight": self._config.ctrl_cost_weight,
            "contact_cost_weight": self._config.contact_cost_weight,
            "contact_cost_range": (
                self._config.contact_cost_range_low,
                self._config.contact_cost_range_high,
            ),
            "healthy_reward": self._config.healthy_reward,
            "terminate_when_unhealthy": self._config.terminate_when_unhealthy,
            "healthy_z_range": (
                self._config.healthy_z_range_low,
                self._config.healthy_z_range_high,
            ),
        }
        if render_mode is not None:
            kwargs["render_mode"] = render_mode
        return gym.make(ENV_ID, **kwargs)

    def make_env(self) -> gym.Env:
        return self.make_env_with_render_mode(self._config.render_mode)


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
    }

    ACTIVATIONS = {
        "relu": torch.nn.ReLU,
        "tanh": torch.nn.Tanh,
    }

    @staticmethod
    def policy_defaults(policy_name: str) -> Dict[str, Any]:
        return dict(POLICY_DEFAULTS.get(policy_name, POLICY_DEFAULTS["SAC"]))

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

    @staticmethod
    def _normalize_train_freq(value: Any) -> Any:
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (int(value[0]), str(value[1]).strip().lower())
        if isinstance(value, str):
            compact = value.strip().strip("()[]").replace(" ", "")
            if "," in compact:
                count, unit = compact.split(",", 1)
                return (int(count), unit.strip("\"'").lower())
            return int(compact)
        return value

    @staticmethod
    def _repair_sac_cuda_sde_state(model: Any, env: gym.Env) -> None:
        """Ensure SAC gSDE tensors are on CUDA before first rollout/predict call."""
        if not getattr(model, "use_sde", False):
            return

        device = getattr(model, "device", None)
        if device is None or str(device).lower() == "cpu":
            return

        actor = getattr(model, "actor", None)
        if actor is None:
            return

        actor.to(device)

        # gSDE exploration tensors may be created on CPU initially in some SB3/PyTorch
        # combinations. Reset noise once on target device to materialize CUDA tensors.
        try:
            actor.reset_noise(batch_size=1)
        except TypeError:
            actor.reset_noise(1)

        action_dist = getattr(actor, "action_dist", None)
        if action_dist is not None:
            for name in ("exploration_mat", "exploration_matrices", "mean_actions", "log_std"):
                value = getattr(action_dist, name, None)
                if torch.is_tensor(value) and value.device != device:
                    setattr(action_dist, name, value.to(device))

        # Fail fast if the first stochastic prediction still triggers mixed-device errors.
        obs, _ = env.reset()
        try:
            model.predict(obs, deterministic=False)
        except RuntimeError as exc:
            message = str(exc).lower()
            if "cuda" in message and "cpu" in message:
                raise RuntimeError(
                    "SAC CUDA gSDE device mismatch after repair. "
                    "Please check stable-baselines3/torch version compatibility."
                ) from exc
            raise

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

        # Remove GUI/shared-control fields that are not valid SB3 constructor kwargs.
        for ui_only_key in ("hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"):
            merged.pop(ui_only_key, None)

        net_arch = cls._parse_net_arch(network_cfg.hidden_layer)
        activation_fn = cls.ACTIVATIONS.get(network_cfg.activation.lower().strip(), torch.nn.ReLU)

        if policy_name == "PPO":
            policy_kwargs = {
                "activation_fn": activation_fn,
                "net_arch": {
                    "pi": net_arch,
                    "vf": net_arch,
                },
            }
        else:
            policy_kwargs = {
                "activation_fn": activation_fn,
                "net_arch": net_arch,
            }

        merged["policy_kwargs"] = policy_kwargs
        merged["learning_rate"] = cls._build_schedule(lr_cfg)

        if policy_name in {"SAC", "TD3"} and "train_freq" in merged:
            merged["train_freq"] = cls._normalize_train_freq(merged["train_freq"])

        if policy_name == "TD3" and "action_noise" not in merged:
            # TD3 uses deterministic actions; explicit action noise is needed for exploration.
            if isinstance(getattr(env, "action_space", None), gym.spaces.Box):
                action_dim = int(np.prod(env.action_space.shape))
                sigma_value = float(merged.pop("action_noise_sigma", 0.1))
                sigma = sigma_value * np.ones(action_dim, dtype=np.float32)
                merged["action_noise"] = NormalActionNoise(
                    mean=np.zeros(action_dim, dtype=np.float32),
                    sigma=sigma,
                )
        elif policy_name == "TD3":
            # action_noise_sigma is a GUI helper field, not an SB3 kwarg.
            merged.pop("action_noise_sigma", None)

        if policy_name == "PPO":
            n_steps = int(merged.get("n_steps", 2048))
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

        if policy_name == "SAC" and str(device).lower().startswith("cuda"):
            cls._repair_sac_cuda_sde_state(model, env)

        return model


class HumanoidTrainer:
    """Training orchestrator with UI-independent event emission."""

    def __init__(
        self,
        env_wrapper: HumanoidEnvWrapper,
        event_sink: Optional[EventSink] = None,
        render_enabled_fn: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.env_wrapper = env_wrapper
        self.event_sink = event_sink
        self.render_enabled_fn = render_enabled_fn
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.cancel_event = threading.Event()
        self._latest_render_state: Optional[np.ndarray] = None
        self._transitions: List[Dict[str, Any]] = []
        self._render_available = True

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

    @staticmethod
    def _safe_close_env(env: gym.Env) -> None:
        """Close env while suppressing benign GLFW teardown warnings on Windows."""
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r".*The GLFW library is not initialized.*",
                category=Warning,
            )
            try:
                env.close()
            except Exception:
                # Ignore teardown-only errors to keep reset/shutdown paths clean.
                pass

    def _capture_frame(self, env: gym.Env, episode_frames: List[np.ndarray], stride: int, step_idx: int) -> None:
        if not self._render_available:
            return
        if step_idx != 0 and step_idx % max(1, stride) != 0:
            return
        try:
            frame = env.render()
        except Exception:
            # Disable rendering for the rest of this run when GL context becomes invalid.
            self._render_available = False
            return
        if isinstance(frame, np.ndarray):
            # Copy render buffers because some envs reuse the same underlying array.
            # Without copying, async GUI playback can show black/stale frames.
            frozen = np.array(frame, copy=True)
            self._latest_render_state = frozen
            episode_frames.append(frozen)

    def run_episode(
        self,
        model: Any,
        env: gym.Env,
        max_steps: int,
        deterministic: bool,
        frame_stride: int,
        collect_transitions: bool,
        capture_frames: bool,
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
            if capture_frames:
                self._capture_frame(env, episode_frames, frame_stride, step_idx)

            if collect_transitions:
                self._transitions.append({"step": step_idx, "reward": float(reward), "done": bool(done)})

            obs = next_obs
            if done:
                return total_reward, step_idx + 1, episode_frames

        return total_reward, max_steps, episode_frames

    def evaluate_policy(self, model: Any, episodes: int, max_steps: int) -> float:
        # Evaluation does not need rendering; keep it headless for performance.
        eval_env = self.env_wrapper.make_env_with_render_mode(None)
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
                    capture_frames=False,
                )
                rewards.append(reward)
            return float(np.mean(rewards)) if rewards else 0.0
        finally:
            self._safe_close_env(eval_env)

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
        self._render_available = True

        # Keep a separate learning env for SB3 updates.
        # Preview env is recreated per episode to avoid long-lived render-context stalls.
        preview_env: Optional[gym.Env] = None
        preview_env_mode: Optional[str] = None
        train_env = self.env_wrapper.make_env_with_render_mode(None)
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

                render_enabled = bool(config.env.render_mode)
                if render_enabled and self.render_enabled_fn is not None:
                    try:
                        render_enabled = bool(self.render_enabled_fn())
                    except Exception:
                        render_enabled = True
                preview_mode = config.env.render_mode if render_enabled else None

                # Fast path for performance mode: keep a non-render preview env alive
                # across episodes. Recreate only when render mode changes.
                if preview_env is None:
                    preview_env = self.env_wrapper.make_env_with_render_mode(preview_mode)
                    preview_env_mode = preview_mode
                elif preview_mode != preview_env_mode:
                    self._safe_close_env(preview_env)
                    preview_env = self.env_wrapper.make_env_with_render_mode(preview_mode)
                    preview_env_mode = preview_mode
                elif preview_mode is not None:
                    # Render mode stays on: recreate each episode to avoid long-lived
                    # context instability observed with MuJoCo+GL in GUI workflows.
                    self._safe_close_env(preview_env)
                    preview_env = self.env_wrapper.make_env_with_render_mode(preview_mode)
                    preview_env_mode = preview_mode

                reward, steps, frames = self.run_episode(
                    model=model,
                    env=preview_env,
                    max_steps=config.max_steps,
                    deterministic=False,
                    frame_stride=max(1, config.frame_stride),
                    collect_transitions=config.collect_transitions,
                    capture_frames=bool(preview_mode),
                )

                train_timesteps = max(1, steps)
                if config.policy_name == "PPO":
                    # PPO needs at least one complete rollout buffer to trigger policy updates.
                    # If train timesteps are below n_steps, learning can stall.
                    ppo_n_steps = int(config.policy_params.get("n_steps", 2048))
                    train_timesteps = max(train_timesteps, ppo_n_steps)

                callback = _TrainingProgressCallback(
                    pause_event=self.pause_event,
                    cancel_event=self.cancel_event,
                    step_budget=train_timesteps,
                )
                model.learn(total_timesteps=train_timesteps, reset_num_timesteps=False, callback=callback)

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
                    "steps": steps,
                    "epsilon": None,
                    "lr": config.lr.learning_rate,
                    "best_reward": best_reward,
                    "render_state": self._latest_render_state,
                }
                if not config.split_aux_events:
                    payload["eval_points"] = list(eval_points)
                    if should_emit_frames:
                        payload["frames"] = frames
                        payload["frame"] = frames[-1] if frames else None
                self._emit(payload)

                if config.split_aux_events:
                    aux_payload: Dict[str, Any] = {
                        "type": "episode_aux",
                        "session_id": config.session_id,
                        "run_id": config.run_id,
                        "episode": episode,
                        "eval_points": list(eval_points),
                    }
                    if should_emit_frames:
                        aux_payload["frames"] = frames
                        aux_payload["frame"] = frames[-1] if frames else None
                    self._emit(aux_payload)

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
        except Exception as exc:  # pragma: no cover
            payload = {
                "type": "error",
                "session_id": config.session_id,
                "run_id": config.run_id,
                "message": str(exc),
            }
            self._emit(payload)
            raise
        finally:
            if preview_env is not None:
                self._safe_close_env(preview_env)
            self._safe_close_env(train_env)


def build_default_trainer(
    event_sink: Optional[EventSink] = None,
    render_enabled_fn: Optional[Callable[[], bool]] = None,
) -> HumanoidTrainer:
    wrapper = HumanoidEnvWrapper(EnvironmentConfig(render_mode="rgb_array"))
    return HumanoidTrainer(wrapper, event_sink=event_sink, render_enabled_fn=render_enabled_fn)


def run_trainer_subprocess(
    env_cfg: EnvironmentConfig,
    trainer_cfg: TrainerConfig,
    outbound_queue: Any,
    pause_event: Any = None,
    cancel_event: Any = None,
) -> None:
    """Process entrypoint: train in an isolated process and stream events to queue."""

    def _sink(payload: Dict[str, Any]) -> None:
        outbound_queue.put(dict(payload))

    try:
        wrapper = HumanoidEnvWrapper(env_cfg)
        trainer = HumanoidTrainer(wrapper, event_sink=_sink)
        if pause_event is not None:
            trainer.pause_event = pause_event
        if cancel_event is not None:
            trainer.cancel_event = cancel_event
        trainer.train(trainer_cfg)
    except Exception as exc:  # pragma: no cover - process safety
        outbound_queue.put(
            {
                "type": "error",
                "session_id": trainer_cfg.session_id,
                "run_id": trainer_cfg.run_id,
                "message": f"subprocess worker error: {exc}",
            }
        )


def expand_compare_runs(base: Dict[str, Any], compare_values: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Build Cartesian compare runs from a base config and compare value lists."""
    if not compare_values:
        return [dict(base)]

    keys = list(compare_values.keys())
    value_sets = [compare_values[key] for key in keys]
    runs: List[Dict[str, Any]] = []
    for combo in itertools.product(*value_sets):
        run = dict(base)
        for key, value in zip(keys, combo):
            run[key] = value
        runs.append(run)
    return runs