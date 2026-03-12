from __future__ import annotations

from typing import Any, Dict

import numpy as np

import Ant_logic
from Ant_logic import (
    EnvironmentConfig,
    TrainerConfig,
    AntTrainer,
    build_lr_schedule,
    expand_compare_runs,
    parse_hidden_layer,
)


def test_parse_hidden_layer_variants() -> None:
    assert parse_hidden_layer("256") == [256, 256]
    assert parse_hidden_layer("256,128,64") == [256, 128, 64]
    assert parse_hidden_layer("bad", fallback=(64, 64)) == [64, 64]


def test_lr_schedule_floor() -> None:
    schedule = build_lr_schedule(1e-3, "exponential", min_lr=1e-5, lr_decay=0.8)
    assert schedule(0.0) >= 1e-5
    assert schedule(1.0) >= 1e-5


def test_expand_compare_runs_cartesian() -> None:
    runs = expand_compare_runs({"Policy": "SAC"}, {"Policy": ["SAC", "TQC"], "gamma": [0.95, 0.99]})
    assert len(runs) == 4


def test_train_emits_episode_and_done(monkeypatch) -> None:
    events = []

    cfg = TrainerConfig(policy="CMA-ES", episodes=3, update_rate=1, specific_params={"iterations_per_episode": 1, "popsize": 2})
    trainer = AntTrainer(EnvironmentConfig(max_steps=5), cfg, event_sink=events.append, session_id="s1", run_id="r1")

    def fake_run_episode(self: AntTrainer, collect_transitions: bool = False, render_frames: bool = False, deterministic: bool = False) -> Dict[str, Any]:
        return {
            "reward": 1.0,
            "steps": 5,
            "frames": [],
            "transitions": [{"step": 0, "reward": 1.0, "obs_mean": 0.0, "next_obs_mean": 0.0}] if collect_transitions else [],
        }

    monkeypatch.setattr(AntTrainer, "run_episode", fake_run_episode)

    result = trainer.train(collect_transitions=True)
    assert result["type"] == "training_done"
    assert any(e.get("type") == "episode" for e in events)
    assert any(e.get("type") == "episode_aux" for e in events)
    assert any(e.get("type") == "training_done" for e in events)
    assert trainer.sampled_transitions


def test_render_enabled_false_disables_render_state(monkeypatch) -> None:
    events = []
    cfg = TrainerConfig(policy="CMA-ES", episodes=1, update_rate=1, render_enabled=False, specific_params={"iterations_per_episode": 1, "popsize": 2})
    trainer = AntTrainer(EnvironmentConfig(max_steps=5), cfg, event_sink=events.append, session_id="s2", run_id="r2")

    def fake_run_episode(self: AntTrainer, collect_transitions: bool = False, render_frames: bool = False, deterministic: bool = False) -> Dict[str, Any]:
        return {
            "reward": 1.0,
            "steps": 1,
            "frames": [1] if render_frames else [],
            "transitions": [],
        }

    monkeypatch.setattr(AntTrainer, "run_episode", fake_run_episode)
    trainer.train()

    ep = next(e for e in events if e.get("type") == "episode")
    aux = next(e for e in events if e.get("type") == "episode_aux")
    assert ep["render_state"] == "off"
    assert aux["frames"] == []


def test_cmaes_searcher_path_is_exercised(monkeypatch) -> None:
    cfg = TrainerConfig(policy="CMA-ES", episodes=1, specific_params={"iterations_per_episode": 1, "popsize": 2})
    trainer = AntTrainer(EnvironmentConfig(max_steps=3), cfg, session_id="s3", run_id="r3")

    class _FakeEnv:
        class _Space:
            shape = (2,)

        action_space = _Space()

        def reset(self, seed: Any = None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: Any):
            obs = np.zeros(2, dtype=np.float32)
            reward = float(np.sum(action))
            terminated = False
            truncated = False
            info: Dict[str, Any] = {}
            return obs, reward, terminated, truncated, info

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    monkeypatch.setattr(Ant_logic.AntEnvironment, "build", lambda self, render_mode=None: _FakeEnv())

    class _FakeBest:
        def __init__(self, values):
            self.values = values

    class _FakeCMAES:
        def __init__(self, problem, stdev_init: float, popsize: int):
            assert stdev_init > 0
            assert popsize >= 1
            self.status = {"best": _FakeBest(Ant_logic.torch.zeros(2))}

        def step(self) -> None:
            return None

    monkeypatch.setattr(Ant_logic, "Problem", lambda *args, **kwargs: object())
    monkeypatch.setattr(Ant_logic, "CMAES", _FakeCMAES)

    result = trainer.run_episode(collect_transitions=False, render_frames=False, deterministic=False)
    assert "reward" in result
    assert result["steps"] == 3


def test_cmaes_problem_gets_initial_bounds_from_action_space(monkeypatch) -> None:
    cfg = TrainerConfig(policy="CMA-ES", episodes=1, specific_params={"iterations_per_episode": 1, "popsize": 2})
    trainer = AntTrainer(EnvironmentConfig(max_steps=1), cfg, session_id="s4", run_id="r4")

    class _FakeEnv:
        class _Space:
            shape = (2,)
            low = np.array([-2.0, -0.5], dtype=np.float32)
            high = np.array([2.0, 0.5], dtype=np.float32)

        action_space = _Space()

        def reset(self, seed: Any = None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: Any):
            obs = np.zeros(2, dtype=np.float32)
            reward = float(np.sum(action))
            terminated = True
            truncated = False
            info: Dict[str, Any] = {}
            return obs, reward, terminated, truncated, info

        def render(self):
            return np.zeros((2, 2, 3), dtype=np.uint8)

        def close(self) -> None:
            return None

    monkeypatch.setattr(Ant_logic.AntEnvironment, "build", lambda self, render_mode=None: _FakeEnv())

    captured: Dict[str, Any] = {}

    def _fake_problem(*args, **kwargs):
        captured["kwargs"] = kwargs
        return object()

    class _FakeBest:
        def __init__(self, values):
            self.values = values

    class _FakeCMAES:
        def __init__(self, problem, stdev_init: float, popsize: int):
            self.status = {"best": _FakeBest(Ant_logic.torch.zeros(2))}

        def step(self) -> None:
            return None

    monkeypatch.setattr(Ant_logic, "Problem", _fake_problem)
    monkeypatch.setattr(Ant_logic, "CMAES", _FakeCMAES)

    trainer.run_episode(collect_transitions=False, render_frames=False, deterministic=False)

    kwargs = captured["kwargs"]
    assert "initial_bounds" in kwargs
    assert "bounds" not in kwargs
    lb, ub = kwargs["initial_bounds"]
    lb_arr = np.asarray(lb, dtype=np.float32)
    ub_arr = np.asarray(ub, dtype=np.float32)
    assert lb_arr.shape == ub_arr.shape
    assert lb_arr.size == int(kwargs["solution_length"])
    assert np.all(lb_arr < ub_arr)


def test_cmaes_rollout_keeps_running_after_termination(monkeypatch) -> None:
    cfg = TrainerConfig(policy="CMA-ES", episodes=1, frame_stride=1, specific_params={"iterations_per_episode": 1, "popsize": 2})
    trainer = AntTrainer(EnvironmentConfig(max_steps=6), cfg, session_id="s5", run_id="r5")

    class _FakeOptEnv:
        class _Space:
            shape = (2,)
            low = np.array([-1.0, -1.0], dtype=np.float32)
            high = np.array([1.0, 1.0], dtype=np.float32)

        action_space = _Space()

        def reset(self, seed: Any = None):
            return np.zeros(2, dtype=np.float32), {}

        def step(self, action: Any):
            _ = action
            return np.zeros(2, dtype=np.float32), 0.1, True, False, {}

        def close(self) -> None:
            return None

    class _FakeRenderEnv(_FakeOptEnv):
        def __init__(self):
            self.reset_calls = 0

        def reset(self, seed: Any = None):
            self.reset_calls += 1
            return np.zeros(2, dtype=np.float32), {}

        def render(self):
            return np.zeros((4, 4, 3), dtype=np.uint8)

    opt_env = _FakeOptEnv()
    render_env = _FakeRenderEnv()

    build_calls = {"count": 0}

    def _fake_build(self, render_mode=None):
        build_calls["count"] += 1
        return opt_env if build_calls["count"] == 1 else render_env

    monkeypatch.setattr(Ant_logic.AntEnvironment, "build", _fake_build)

    monkeypatch.setattr(Ant_logic, "Problem", None)
    monkeypatch.setattr(Ant_logic, "CMAES", None)

    result = trainer.run_episode(collect_transitions=False, render_frames=True, deterministic=False)
    # Episode accounting should stop at first termination.
    assert result["steps"] == 1
    # Visual playback can still continue with reset segments.
    assert len(result["frames"]) == 6
    assert render_env.reset_calls > 1
