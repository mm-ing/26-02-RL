from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from Humanoid_logic import (
    EnvironmentConfig,
    HumanoidEnvWrapper,
    HumanoidTrainer,
    TrainerConfig,
    expand_compare_runs,
)


class FakeEnv:
    def __init__(self) -> None:
        self.step_idx = 0

    def reset(self):
        self.step_idx = 0
        return np.zeros((4,), dtype=float), {}

    def step(self, action):
        self.step_idx += 1
        done = self.step_idx >= 5
        return np.zeros((4,), dtype=float), 1.0, done, False, {}

    def render(self):
        return np.zeros((32, 32, 3), dtype=np.uint8)

    def close(self):
        return None


class FakeModel:
    def predict(self, obs, deterministic=False):
        return 0, None

    def learn(self, total_timesteps, reset_num_timesteps=False, callback=None):
        if callback is not None:
            for _ in range(int(total_timesteps)):
                if not callback._on_step():
                    break
        return self


def test_run_episode_reports_executed_steps():
    wrapper = HumanoidEnvWrapper(EnvironmentConfig(render_mode="rgb_array"))
    trainer = HumanoidTrainer(wrapper, event_sink=None)

    reward, steps, frames = trainer.run_episode(
        model=FakeModel(),
        env=FakeEnv(),
        max_steps=20,
        deterministic=False,
        frame_stride=2,
        collect_transitions=False,
    )

    assert reward == 5.0
    assert steps == 5
    assert len(frames) >= 3


def test_train_emits_episode_and_done_events(monkeypatch):
    events: List[Dict[str, Any]] = []
    wrapper = HumanoidEnvWrapper(EnvironmentConfig(render_mode="rgb_array"))

    trainer = HumanoidTrainer(wrapper, event_sink=events.append)

    monkeypatch.setattr(wrapper, "make_env", lambda: FakeEnv())

    import Humanoid_logic

    monkeypatch.setattr(
        Humanoid_logic.SB3PolicyFactory,
        "create_model",
        lambda **kwargs: FakeModel(),
    )

    cfg = TrainerConfig(
        episodes=3,
        max_steps=8,
        update_rate=1,
        deterministic_eval_every=2,
        deterministic_eval_episodes=1,
        session_id="session-test",
        run_id="run-test",
    )

    result = trainer.train(cfg)

    assert result["type"] == "training_done"
    episode_events = [e for e in events if e.get("type") == "episode"]
    done_events = [e for e in events if e.get("type") == "training_done"]

    assert len(episode_events) == 3
    assert len(done_events) == 1
    assert episode_events[-1]["session_id"] == "session-test"
    assert episode_events[-1]["run_id"] == "run-test"


def test_pause_resume_cancel_transitions():
    wrapper = HumanoidEnvWrapper(EnvironmentConfig())
    trainer = HumanoidTrainer(wrapper, event_sink=None)

    trainer.pause()
    assert not trainer.pause_event.is_set()

    trainer.resume()
    assert trainer.pause_event.is_set()

    trainer.cancel()
    assert trainer.cancel_event.is_set()
    assert trainer.pause_event.is_set()


def test_expand_compare_runs_cartesian_product():
    base = {"Policy": "SAC", "episodes": 100}
    compare = {"Policy": ["SAC", "TD3"], "gamma": [0.95, 0.99]}
    runs = expand_compare_runs(base, compare)

    assert len(runs) == 4
    combos = {(r["Policy"], r["gamma"]) for r in runs}
    assert combos == {("SAC", 0.95), ("SAC", 0.99), ("TD3", 0.95), ("TD3", 0.99)}
