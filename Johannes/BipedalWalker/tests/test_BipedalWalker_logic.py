from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from BipedalWalker_logic import (  # noqa: E402
    BipedalWalkerConfig,
    BipedalWalkerTrainer,
    build_compare_combinations,
    get_policy_default_configs,
)


def test_policy_defaults_expose_required_policies():
    defaults = get_policy_default_configs()
    assert {"PPO", "A2C", "SAC", "TD3"}.issubset(defaults.keys())


def test_build_compare_combinations_cartesian_product():
    combinations = build_compare_combinations({"Policy": ["PPO", "A2C"], "gamma": [0.95, 0.99]})
    assert len(combinations) == 4
    assert {combo["Policy"] for combo in combinations} == {"PPO", "A2C"}


def test_run_episode_reports_executed_steps_without_transition_collection():
    class DummySpace:
        def sample(self):
            return np.array([0.0])

    class DummyEnv:
        def __init__(self):
            self.action_space = DummySpace()
            self.count = 0

        def reset(self):
            self.count = 0
            return np.array([0.0]), {}

        def step(self, _action):
            self.count += 1
            done = self.count >= 5
            return np.array([0.0]), 1.0, done, False, {}

        def render(self):
            return None

    trainer = BipedalWalkerTrainer(BipedalWalkerConfig())
    result = trainer.run_episode(
        DummyEnv(),
        deterministic=True,
        max_steps=20,
        collect_transitions=False,
        include_frame=False,
    )

    assert result["steps"] == 5
    assert result["reward"] == 5.0
    assert result["transitions"] == []


def test_environment_dependent_training_path_or_skip():
    gym = pytest.importorskip("gymnasium", reason="gymnasium unavailable")
    pytest.importorskip("Box2D", reason="Box2D backend unavailable")

    _ = gym
    config = BipedalWalkerConfig(episodes=1, max_steps=8)
    trainer = BipedalWalkerTrainer(config)
    output = trainer.train(collect_transitions=False, run_label="smoke")
    assert "reward" in output
