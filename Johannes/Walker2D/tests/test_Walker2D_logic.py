from __future__ import annotations

import csv
from typing import Any, Tuple

import numpy as np
import pytest

from Walker2D_logic import POLICY_SHARED_DEFAULTS, TrainConfig, Walker2DEnvConfig, Walker2DTrainer


class DummyActionSpace:
    def sample(self) -> np.ndarray:
        return np.array([0.0], dtype=np.float32)


class DummyEnv:
    def __init__(self, render_mode=None, **kwargs):
        self.render_mode = render_mode
        self.kwargs = kwargs
        self.action_space = DummyActionSpace()
        self._steps = 0

    def reset(self) -> Tuple[np.ndarray, dict]:
        self._steps = 0
        return np.array([0.0], dtype=np.float32), {}

    def step(self, action: Any):
        self._steps += 1
        done = self._steps >= 3
        return np.array([0.0], dtype=np.float32), 1.0, done, False, {}

    def render(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        return None


class DummyModel:
    def predict(self, observation, deterministic=False):
        return np.array([0.0], dtype=np.float32), None

    def learn(self, total_timesteps: int, reset_num_timesteps: bool = False, progress_bar: bool = False):
        return self


def test_run_episode_returns_step_count(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=1, max_steps=10, animation_on=True),
    )

    result = trainer.run_episode(
        model=DummyModel(),
        deterministic=True,
        collect_transitions=False,
        max_steps=10,
        render=True,
        frame_stride=1,
        rollout_full_capture_steps=10,
    )

    assert result["steps"] == 3
    assert result["reward"] == 3.0
    assert len(result["frames"]) >= 1


def test_training_emits_episode_and_done(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))
    monkeypatch.setattr("Walker2D_logic.PPO", lambda *args, **kwargs: DummyModel())

    events = []

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=2, max_steps=3, deterministic_eval_every=1),
        event_sink=events.append,
    )

    done = trainer.train()

    assert done["type"] == "training_done"
    assert any(e.get("type") == "episode" for e in events)
    assert any(e.get("type") == "training_done" for e in events)


def test_deterministic_eval_cadence_points(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))
    monkeypatch.setattr("Walker2D_logic.PPO", lambda *args, **kwargs: DummyModel())

    events = []
    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(
            policy_name="PPO",
            episodes=5,
            max_steps=3,
            deterministic_eval_every=2,
        ),
        event_sink=events.append,
    )

    trainer.train()

    episode_events = [event for event in events if event.get("type") == "episode"]
    assert len(episode_events) == 5
    assert episode_events[0]["eval_points"] == []
    assert episode_events[1]["eval_points"][-1][0] == 2
    assert episode_events[3]["eval_points"][-1][0] == 4


def test_export_transitions_csv_contains_expected_columns(monkeypatch, tmp_path):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))
    monkeypatch.setattr("Walker2D_logic.PPO", lambda *args, **kwargs: DummyModel())

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(
            policy_name="PPO",
            episodes=1,
            max_steps=3,
            collect_transitions=True,
        ),
    )

    trainer.train()
    output_path = trainer.export_transitions_csv(tmp_path)

    assert output_path is not None
    assert output_path.exists()

    with output_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert {"step", "reward", "done"}.issubset(set(reader.fieldnames or []))
    assert len(rows) >= 1


def test_evaluate_policy_uses_max_steps(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=1, max_steps=1000, deterministic_eval_max_steps=123),
    )

    trainer._model = DummyModel()
    captured = {"max_steps": None}

    original_run_episode = trainer.run_episode

    def capturing_run_episode(*args, **kwargs):
        captured["max_steps"] = kwargs.get("max_steps")
        return original_run_episode(*args, **kwargs)

    monkeypatch.setattr(trainer, "run_episode", capturing_run_episode)

    _ = trainer.evaluate_policy(episodes=1)
    assert captured["max_steps"] == 1000


def test_model_uses_hidden_layer_and_lr_schedule(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))

    captured = {}

    class CapturingModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def predict(self, observation, deterministic=False):
            return np.array([0.0], dtype=np.float32), None

        def learn(self, total_timesteps: int, reset_num_timesteps: bool = False, progress_bar: bool = False):
            return self

    monkeypatch.setattr("Walker2D_logic.PPO", CapturingModel)

    shared = dict(POLICY_SHARED_DEFAULTS["PPO"])
    shared.update(
        {
            "learning_rate": 3e-4,
            "lr_strategy": "linear",
            "min_lr": 1e-5,
            "lr_decay": 0.99,
            "hidden_layer": 320,
        }
    )

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=1, max_steps=3, shared_params=shared),
    )

    trainer._build_model()

    policy_kwargs = captured.get("policy_kwargs", {})
    assert policy_kwargs.get("net_arch") == [320, 320]
    lr = captured.get("learning_rate")
    assert callable(lr)
    assert float(lr(1.0)) == pytest.approx(3e-4)
    assert float(lr(0.0)) == pytest.approx(1e-5)


def test_model_uses_comma_separated_hidden_layers(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))

    captured = {}

    class CapturingModel:
        def __init__(self, *args, **kwargs):
            captured.update(kwargs)

        def predict(self, observation, deterministic=False):
            return np.array([0.0], dtype=np.float32), None

        def learn(self, total_timesteps: int, reset_num_timesteps: bool = False, progress_bar: bool = False):
            return self

    monkeypatch.setattr("Walker2D_logic.PPO", CapturingModel)

    shared = dict(POLICY_SHARED_DEFAULTS["PPO"])
    shared["hidden_layer"] = "512, 256,128"

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=1, max_steps=3, shared_params=shared),
    )

    trainer._build_model()

    policy_kwargs = captured.get("policy_kwargs", {})
    assert policy_kwargs.get("net_arch") == [512, 256, 128]


def test_run_episode_capture_horizon_follows_max_steps_not_rollout_limit(monkeypatch):
    monkeypatch.setattr("Walker2D_logic.gym.make", lambda *args, **kwargs: DummyEnv(**kwargs))

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(policy_name="PPO", episodes=1, max_steps=10, animation_on=True),
    )

    result = trainer.run_episode(
        model=DummyModel(),
        deterministic=True,
        collect_transitions=False,
        max_steps=10,
        render=True,
        frame_stride=1,
        rollout_full_capture_steps=1,
    )

    assert result["steps"] == 3
    assert len(result["frames"]) == 3
