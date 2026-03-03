from typing import Any, Dict

import pytest

from InvDoubPend_logic import (
    InvertedDoublePendulumTrainer,
    SB3PolicyFactory,
    build_run_label,
)


class DummyEnv:
    def __init__(self):
        self.action_space = self
        self._step = 0

    def sample(self):
        return [0.0]

    def reset(self):
        self._step = 0
        return [0.0], {}

    def step(self, _action):
        self._step += 1
        terminated = self._step >= 5
        return [0.0], 1.0, terminated, False, {}

    def render(self):
        return None


def test_policy_defaults_exposed():
    factory = SB3PolicyFactory()
    for policy in ("PPO", "SAC", "TD3"):
        defaults = factory.get_defaults(policy)
        assert "lr" in defaults
        assert "gamma" in defaults


def test_build_run_label_has_core_fields():
    label = build_run_label(
        "PPO",
        {"healthy_reward": 10, "reset_noise_scale": 0.1},
        {"episodes": 50, "max_steps": 1000, "gamma": 0.99},
        {"lr": 3e-4},
    )
    assert "PPO" in label
    assert "hr=10" in label
    assert "rn=0.1" in label


def test_run_episode_reports_executed_steps_without_transitions():
    trainer = InvertedDoublePendulumTrainer()
    env = DummyEnv()
    out = trainer.run_episode(
        env=env,
        max_steps=100,
        epsilon=1.0,
        deterministic=False,
        collect_transitions=False,
        capture_rollout=False,
    )
    assert out["steps"] == 5
    assert out["transitions"] == []


def test_runtime_animation_settings_update():
    trainer = InvertedDoublePendulumTrainer()
    trainer.set_runtime_animation(animation_on=False, update_rate=7, rollout_full_capture_steps=99, low_overhead_animation=True)
    assert trainer.animation_on is False
    assert trainer.update_rate == 7
    assert trainer.rollout_full_capture_steps == 99
    assert trainer.low_overhead_animation is True
