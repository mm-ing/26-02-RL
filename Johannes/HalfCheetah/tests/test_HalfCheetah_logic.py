from __future__ import annotations

from HalfCheetah_logic import HalfCheetahTrainer, build_compare_runs


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return 0, None


class _FakeEnv:
    def __init__(self, horizon=5):
        self.horizon = horizon
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        return 0, {}

    def step(self, action):
        self.step_count += 1
        terminated = self.step_count >= self.horizon
        return 0, 1.0, terminated, False, {}

    def render(self):
        return None


class _FakeWrapper:
    def __init__(self):
        self.env = _FakeEnv()

    def ensure(self):
        return self.env


def test_build_compare_runs_cartesian_product():
    base = {"policy": "PPO", "max_steps": 100, "specific_params": {"activation": "Tanh"}}
    compare = {"policy": ["PPO", "SAC"], "gamma": [0.95, 0.99]}
    runs = build_compare_runs(base, compare)
    assert len(runs) == 4
    assert {run["policy"] for run in runs} == {"PPO", "SAC"}


def test_run_episode_returns_executed_steps_without_transition_collection():
    trainer = HalfCheetahTrainer(event_callback=None)
    result = trainer.run_episode(
        model=_FakeModel(),
        env_wrapper=_FakeWrapper(),
        max_steps=10,
        deterministic=True,
        collect_transitions=False,
        animation_on=False,
    )
    assert result["steps"] == 5
    assert result["reward"] == 5.0
    assert trainer.transitions == []


def test_pause_resume_and_stop_transitions():
    trainer = HalfCheetahTrainer(event_callback=None)
    trainer.pause()
    assert not trainer.pause_event.is_set()
    trainer.resume()
    assert trainer.pause_event.is_set()
    trainer.request_stop()
    assert trainer.stop_event.is_set()
