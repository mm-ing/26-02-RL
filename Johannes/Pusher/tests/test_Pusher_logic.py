from __future__ import annotations

import csv
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from Pusher_logic import (
    PusherTrainer,
    TrainerConfig,
    build_compare_configs,
    build_learning_rate,
    parse_hidden_layers,
)


class DummyEnv:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return np.zeros(3, dtype=float), {}

    def step(self, action):
        self.step_count += 1
        obs = np.ones(3, dtype=float) * self.step_count
        reward = 1.0
        terminated = self.step_count >= self.max_steps
        truncated = False
        return obs, reward, terminated, truncated, {}

    def render(self):
        return np.zeros((8, 8, 3), dtype=np.uint8)

    def close(self):
        return None


class DummyModel:
    def __init__(self):
        self._current_progress_remaining = 1.0
        self.lr_schedule = 0.001

    def predict(self, obs, deterministic=False):
        return np.zeros(1, dtype=float), None

    def learn(self, total_timesteps: int, reset_num_timesteps: bool, callback, progress_bar: bool):
        for _ in range(total_timesteps):
            callback._on_step()
        self._current_progress_remaining = 0.5
        time.sleep(0.01)
        return self


class DummyAgentBuilder:
    @staticmethod
    def build_model(policy_name: str, env, shared_params: Dict[str, Any], policy_params: Dict[str, Any], device: str, seed):
        return DummyModel()


def test_parse_hidden_layers_variants():
    assert parse_hidden_layers("256") == (256, 256)
    assert parse_hidden_layers("256,128,64") == (256, 128, 64)
    assert parse_hidden_layers("bad", fallback=(64, 64)) == (64, 64)


def test_exponential_learning_rate_respects_floor():
    schedule = build_learning_rate(3e-4, "exponential", 1e-5, 0.995)
    assert callable(schedule)
    assert schedule(0.0) >= 1e-5


def test_run_episode_returns_actual_step_count():
    trainer = PusherTrainer()
    env = DummyEnv(max_steps=4)
    model = DummyModel()
    out = trainer.run_episode(env=env, model=model, deterministic=False, max_steps=10, collect_transitions=False)
    assert out["steps"] == 4


def test_training_emits_step_episode_and_eval_points(monkeypatch):
    events: List[Dict[str, Any]] = []
    trainer = PusherTrainer(event_sink=events.append, env_factory=lambda render_mode, env_params: DummyEnv(max_steps=2))
    monkeypatch.setattr("Pusher_logic.SB3PolicyAgent", DummyAgentBuilder)

    config = TrainerConfig(policy="SAC", episodes=12, max_steps=2, update_rate=2, enable_animation=False)
    result = trainer.train(config)
    assert result["type"] == "training_done"

    step_events = [e for e in events if e.get("type") == "step"]
    episode_events = [e for e in events if e.get("type") == "episode"]
    assert len(step_events) == 12
    assert len(episode_events) == 12
    assert [pt[0] for pt in episode_events[-1]["eval_points"]] == [10, 12]


def test_compare_policy_baseline_and_incompatible_override_ignore():
    base = TrainerConfig(
        policy="SAC",
        shared_params={"learning_rate": 9e-4, "batch_size": 999, "gamma": 0.98},
        policy_params={"learning_starts": 2222},
    )
    grid = {"policy": ["SAC", "PPO"], "learning_rate": [3e-4], "learning_starts": [12345]}
    configs = build_compare_configs(base, grid)

    sac_cfg = [cfg for cfg in configs if cfg.policy == "SAC"][0]
    ppo_cfg = [cfg for cfg in configs if cfg.policy == "PPO"][0]

    assert sac_cfg.policy_params["learning_starts"] == 12345
    assert ppo_cfg.shared_params["batch_size"] == 256
    assert "learning_starts" not in ppo_cfg.policy_params


def test_pause_resume_cancel_transitions(monkeypatch):
    events: List[Dict[str, Any]] = []
    trainer = PusherTrainer(event_sink=events.append, env_factory=lambda render_mode, env_params: DummyEnv(max_steps=2))
    monkeypatch.setattr("Pusher_logic.SB3PolicyAgent", DummyAgentBuilder)

    config = TrainerConfig(policy="SAC", episodes=40, max_steps=2, enable_animation=False)

    t = threading.Thread(target=lambda: trainer.train(config), daemon=True)
    t.start()
    time.sleep(0.05)
    trainer.request_pause()
    assert trainer.pause_event.is_set() is False
    time.sleep(0.02)
    trainer.request_resume()
    assert trainer.pause_event.is_set() is True
    trainer.request_cancel()
    t.join(timeout=4)

    done = [e for e in events if e.get("type") == "training_done"]
    assert done
    assert done[-1].get("canceled") is True


def test_transition_csv_export_non_empty(monkeypatch, tmp_path: Path):
    events: List[Dict[str, Any]] = []
    trainer = PusherTrainer(event_sink=events.append, env_factory=lambda render_mode, env_params: DummyEnv(max_steps=2))
    monkeypatch.setattr("Pusher_logic.SB3PolicyAgent", DummyAgentBuilder)

    config = TrainerConfig(
        policy="SAC",
        episodes=3,
        max_steps=2,
        enable_animation=False,
        export_transitions_csv=True,
        results_dir=str(tmp_path),
    )
    out = trainer.train(config)
    csv_path = out.get("csv_path")
    assert csv_path
    assert Path(csv_path).exists()

    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert "run_id" in rows[0]
    assert "observation" in rows[0]
