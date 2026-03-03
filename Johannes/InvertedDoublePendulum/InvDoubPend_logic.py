import csv
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

try:
    from stable_baselines3 import PPO, SAC, TD3
except Exception:  # pragma: no cover - dependency/runtime guard
    PPO = SAC = TD3 = None


ENV_ID = "InvertedDoublePendulum-v5"
ALLOWED_POLICIES = ("PPO", "SAC", "TD3")


def _activation_from_name(name: str):
    import torch.nn as nn

    table = {
        "ReLU": nn.ReLU,
        "Tanh": nn.Tanh,
        "ELU": nn.ELU,
        "LeakyReLU": nn.LeakyReLU,
    }
    return table.get(name, nn.ReLU)


def _resolve_torch_device(requested: Any) -> str:
    requested_text = str(requested or "CPU").strip().upper()
    if requested_text != "GPU":
        return "cpu"

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@dataclass
class EnvironmentConfig:
    healthy_reward: float = 10.0
    reset_noise_scale: float = 0.1
    render_mode: Optional[str] = None


class InvertedDoublePendulumEnvironment:
    def __init__(self, config: Optional[EnvironmentConfig] = None):
        self.config = config or EnvironmentConfig()
        self.env = self._build_env()

    def _build_env(self):
        kwargs = {
            "healthy_reward": float(self.config.healthy_reward),
            "reset_noise_scale": float(self.config.reset_noise_scale),
        }
        if self.config.render_mode:
            kwargs["render_mode"] = self.config.render_mode
        return gym.make(ENV_ID, **kwargs)

    def update(self, healthy_reward: float, reset_noise_scale: float, render_mode: Optional[str] = None):
        self.close()
        self.config = EnvironmentConfig(
            healthy_reward=float(healthy_reward),
            reset_noise_scale=float(reset_noise_scale),
            render_mode=render_mode,
        )
        self.env = self._build_env()

    def close(self):
        if self.env is not None:
            self.env.close()


class SB3PolicyFactory:
    def __init__(self):
        self.policy_defaults: Dict[str, Dict[str, Any]] = {
            "PPO": {
                "hidden_layer": 256,
                "activation": "Tanh",
                "lr": 3e-4,
                "lr_strategy": "constant",
                "min_lr": 1e-5,
                "lr_decay": 0.995,
                "replay_size": 100000,
                "batch_size": 128,
                "learning_start": 0,
                "learning_frequency": 1,
                "target_update": 1,
                "gamma": 0.99,
                "n_steps": 2048,
            },
            "SAC": {
                "hidden_layer": 256,
                "activation": "ReLU",
                "lr": 3e-4,
                "lr_strategy": "constant",
                "min_lr": 1e-5,
                "lr_decay": 0.995,
                "replay_size": 500000,
                "batch_size": 256,
                "learning_start": 10000,
                "learning_frequency": 1,
                "target_update": 1,
                "gamma": 0.99,
            },
            "TD3": {
                "hidden_layer": 256,
                "activation": "ReLU",
                "lr": 3e-4,
                "lr_strategy": "constant",
                "min_lr": 1e-5,
                "lr_decay": 0.995,
                "replay_size": 500000,
                "batch_size": 256,
                "learning_start": 10000,
                "learning_frequency": 1,
                "target_update": 1,
                "gamma": 0.99,
            },
        }

    def get_defaults(self, policy_name: str) -> Dict[str, Any]:
        if policy_name not in self.policy_defaults:
            raise ValueError(f"Unsupported policy: {policy_name}")
        return dict(self.policy_defaults[policy_name])

    def build_model(self, policy_name: str, env, params: Dict[str, Any]):
        if policy_name not in ALLOWED_POLICIES:
            raise ValueError(f"Unsupported policy: {policy_name}")
        if PPO is None or SAC is None or TD3 is None:
            raise RuntimeError("stable-baselines3 is not available in this environment")

        hidden_layer = int(params.get("hidden_layer", 256))
        activation_name = str(params.get("activation", "ReLU"))
        activation_fn = _activation_from_name(activation_name)
        learning_rate = float(params.get("lr", 3e-4))
        gamma = float(params.get("gamma", 0.99))
        torch_device = _resolve_torch_device(params.get("device", "CPU"))

        policy_kwargs = dict(
            net_arch=[hidden_layer, hidden_layer],
            activation_fn=activation_fn,
        )

        if policy_name == "PPO":
            batch_size = int(params.get("batch_size", 128))
            n_steps = int(params.get("n_steps", 2048))
            if n_steps < batch_size:
                n_steps = batch_size
            return PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_steps=n_steps,
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=torch_device,
            )

        if policy_name == "SAC":
            return SAC(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                buffer_size=int(params.get("replay_size", 500000)),
                learning_starts=int(params.get("learning_start", 10000)),
                batch_size=int(params.get("batch_size", 256)),
                train_freq=int(params.get("learning_frequency", 1)),
                gamma=gamma,
                policy_kwargs=policy_kwargs,
                verbose=0,
                device=torch_device,
            )

        return TD3(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=int(params.get("replay_size", 500000)),
            learning_starts=int(params.get("learning_start", 10000)),
            batch_size=int(params.get("batch_size", 256)),
            train_freq=int(params.get("learning_frequency", 1)),
            gamma=gamma,
            policy_kwargs=policy_kwargs,
            verbose=0,
            device=torch_device,
        )


class InvertedDoublePendulumTrainer:
    def __init__(self):
        self.policy_factory = SB3PolicyFactory()
        self.stop_event = threading.Event()
        self.pause_event = threading.Event()
        self.pause_event.set()
        self.model = None
        self.transitions: List[Dict[str, Any]] = []
        self.animation_on = True
        self.update_rate = 5
        self.rollout_full_capture_steps = 120
        self.low_overhead_animation = False

    def set_runtime_animation(self, animation_on: bool, update_rate: int, rollout_full_capture_steps: int, low_overhead_animation: bool):
        self.animation_on = bool(animation_on)
        self.update_rate = max(1, int(update_rate))
        self.rollout_full_capture_steps = max(5, int(rollout_full_capture_steps))
        self.low_overhead_animation = bool(low_overhead_animation)

    def stop(self):
        self.stop_event.set()

    def pause(self):
        self.pause_event.clear()

    def resume(self):
        self.pause_event.set()

    def _emit(self, event_sink: Optional[Callable[[Dict[str, Any]], None]], event: Dict[str, Any]):
        if event_sink:
            event_sink(event)

    def _effective_lr(self, params: Dict[str, Any], episode_idx: int) -> float:
        lr = float(params.get("lr", 3e-4))
        strategy = str(params.get("lr_strategy", "constant"))
        if strategy.lower() != "decay":
            return lr
        min_lr = float(params.get("min_lr", 1e-5))
        decay = float(params.get("lr_decay", 0.995))
        return max(min_lr, lr * (decay ** max(0, episode_idx - 1)))

    def _should_capture_rollout(self, episode_idx: int, episodes: int) -> bool:
        return self.animation_on and (episode_idx % self.update_rate == 0 or episode_idx == episodes)

    def _stride_for_step(self, step: int) -> int:
        if step <= self.rollout_full_capture_steps:
            return 1
        if self.low_overhead_animation:
            return 10
        if step <= self.rollout_full_capture_steps * 2:
            return 2
        if step <= self.rollout_full_capture_steps * 3:
            return 3
        return 5

    def run_episode(
        self,
        env,
        max_steps: int,
        epsilon: float,
        deterministic: bool,
        collect_transitions: bool,
        capture_rollout: bool,
    ) -> Dict[str, Any]:
        obs, _ = env.reset()
        total_reward = 0.0
        frames: List[np.ndarray] = []
        transitions: List[Dict[str, Any]] = []
        executed_steps = 0
        render_failed = False

        for step in range(1, max_steps + 1):
            self.pause_event.wait()
            if self.stop_event.is_set():
                break

            executed_steps = step
            action = None
            if self.model is None or np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = self.model.predict(obs, deterministic=deterministic)

            next_obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)

            if collect_transitions:
                transitions.append(
                    {
                        "state": np.array(obs, copy=True),
                        "action": np.array(action, copy=True),
                        "reward": float(reward),
                        "next_state": np.array(next_obs, copy=True),
                        "done": bool(terminated or truncated),
                    }
                )

            if capture_rollout:
                stride = self._stride_for_step(step)
                if step <= self.rollout_full_capture_steps or step % stride == 0:
                    try:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                    except Exception:
                        render_failed = True
                        capture_rollout = False

            obs = next_obs
            if terminated or truncated:
                break

        if capture_rollout and executed_steps > 0 and len(frames) == 0:
            try:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            except Exception:
                render_failed = True

        return {
            "reward": total_reward,
            "steps": executed_steps,
            "frames": frames,
            "transitions": transitions,
            "render_failed": render_failed,
        }

    def evaluate_policy(self, env, episodes: int = 3) -> float:
        rewards = []
        for _ in range(max(1, episodes)):
            rollout = self.run_episode(
                env=env,
                max_steps=1000,
                epsilon=0.0,
                deterministic=True,
                collect_transitions=False,
                capture_rollout=False,
            )
            rewards.append(float(rollout["reward"]))
        return float(np.mean(rewards))

    def train(
        self,
        run_id: str,
        policy_name: str,
        env_params: Dict[str, Any],
        general_params: Dict[str, Any],
        specific_params: Dict[str, Any],
        event_sink: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        self.stop_event.clear()
        self.pause_event.set()
        self.transitions = []

        eval_env_wrapper = InvertedDoublePendulumEnvironment(
            EnvironmentConfig(
                healthy_reward=float(env_params.get("healthy_reward", 10.0)),
                reset_noise_scale=float(env_params.get("reset_noise_scale", 0.1)),
                render_mode=None,
            )
        )
        train_env_wrapper = InvertedDoublePendulumEnvironment(
            EnvironmentConfig(
                healthy_reward=float(env_params.get("healthy_reward", 10.0)),
                reset_noise_scale=float(env_params.get("reset_noise_scale", 0.1)),
                render_mode=None,
            )
        )
        render_env_wrapper: Optional[InvertedDoublePendulumEnvironment] = None

        episodes = int(general_params.get("episodes", 50))
        max_steps = int(general_params.get("max_steps", 1000))
        epsilon = float(general_params.get("epsilon_max", 0.0))
        epsilon_min = float(general_params.get("epsilon_min", 0.0))
        epsilon_decay = float(general_params.get("epsilon_decay", 0.995))

        model_params = dict(specific_params)
        model_params["gamma"] = float(general_params.get("gamma", model_params.get("gamma", 0.99)))

        self.model = self.policy_factory.build_model(policy_name, train_env_wrapper.env, model_params)

        rewards: List[float] = []
        moving_average: List[float] = []
        eval_points: List[Tuple[int, float]] = []
        best_reward = float("-inf")

        try:
            for episode_idx in range(1, episodes + 1):
                if self.stop_event.is_set():
                    break

                self.pause_event.wait()
                lr_now = self._effective_lr(model_params, episode_idx)
                if hasattr(self.model, "learning_rate"):
                    self.model.learning_rate = lr_now

                self.model.learn(total_timesteps=max_steps, reset_num_timesteps=False, progress_bar=False)

                capture_rollout = self._should_capture_rollout(episode_idx, episodes)
                rollout = self.run_episode(
                    env=train_env_wrapper.env,
                    max_steps=max_steps,
                    epsilon=0.0,
                    deterministic=True,
                    collect_transitions=False,
                    capture_rollout=False,
                )

                frames: List[np.ndarray] = []
                render_failed = False
                if capture_rollout and self.animation_on:
                    try:
                        if render_env_wrapper is None:
                            render_env_wrapper = InvertedDoublePendulumEnvironment(
                                EnvironmentConfig(
                                    healthy_reward=float(env_params.get("healthy_reward", 10.0)),
                                    reset_noise_scale=float(env_params.get("reset_noise_scale", 0.1)),
                                    render_mode="rgb_array",
                                )
                            )
                    except Exception:
                        render_failed = True

                    if render_env_wrapper is not None:
                        vis_rollout = self.run_episode(
                            env=render_env_wrapper.env,
                            max_steps=max_steps,
                            epsilon=0.0,
                            deterministic=True,
                            collect_transitions=False,
                            capture_rollout=True,
                        )
                        if vis_rollout.get("render_failed", False):
                            render_failed = True
                        else:
                            frames = vis_rollout["frames"]

                episode_reward = float(rollout["reward"])
                rewards.append(episode_reward)
                best_reward = max(best_reward, episode_reward)
                ma_window = max(1, int(general_params.get("moving_average", 20)))
                moving_average.append(float(np.mean(rewards[-ma_window:])))

                if episode_idx % 10 == 0 or episode_idx == episodes:
                    eval_reward = self.evaluate_policy(eval_env_wrapper.env, episodes=2)
                    eval_points.append((episode_idx, eval_reward))

                self._emit(
                    event_sink,
                    {
                        "type": "episode",
                        "run_id": run_id,
                        "episode": episode_idx,
                        "episodes": episodes,
                        "reward": episode_reward,
                        "moving_average": moving_average[-1],
                        "eval_points": list(eval_points),
                        "steps": int(rollout["steps"]),
                        "frames": frames,
                        "epsilon": epsilon,
                        "lr": lr_now,
                        "best_reward": best_reward,
                        "render_state": (
                            "skipped"
                            if render_failed
                            else ("on" if capture_rollout and len(frames) > 0 else ("off" if not self.animation_on else "skipped"))
                        ),
                    },
                )

                epsilon = max(epsilon_min, epsilon * epsilon_decay)

            status = "stopped" if self.stop_event.is_set() else "completed"
            result = {
                "status": status,
                "run_id": run_id,
                "rewards": rewards,
                "moving_average": moving_average,
                "eval_points": eval_points,
                "best_reward": best_reward if rewards else None,
                "episodes_completed": len(rewards),
                "policy": policy_name,
            }
            self._emit(event_sink, {"type": "training_done", **result})
            return result
        except Exception as exc:
            self._emit(event_sink, {"type": "error", "run_id": run_id, "message": str(exc)})
            raise
        finally:
            train_env_wrapper.close()
            if render_env_wrapper is not None:
                render_env_wrapper.close()
            eval_env_wrapper.close()

    def export_transitions_csv(self, filename_prefix: str = "samplings") -> Optional[str]:
        if not self.transitions:
            return None
        os.makedirs("results_csv", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join("results_csv", f"{filename_prefix}_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["state", "action", "reward", "next_state", "done"])
            for row in self.transitions:
                writer.writerow([
                    np.array2string(row["state"], precision=6),
                    np.array2string(row["action"], precision=6),
                    row["reward"],
                    np.array2string(row["next_state"], precision=6),
                    row["done"],
                ])
        return path


def build_run_label(policy_name: str, env_params: Dict[str, Any], general_params: Dict[str, Any], specific_params: Dict[str, Any]) -> str:
    base = (
        f"{policy_name} | hr={env_params.get('healthy_reward')} | rn={env_params.get('reset_noise_scale')} | "
        f"ep={general_params.get('episodes')} | ms={general_params.get('max_steps')} | "
        f"lr={specific_params.get('lr'):.1e} | gamma={general_params.get('gamma')}"
    )
    return base


def timestamp_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time() * 1000)}"
