import math
from pathlib import Path

import pytest

from MountainCar_logic import Trainer, make_default_trainer


gym = pytest.importorskip("gymnasium")


@pytest.fixture
def trainer(tmp_path: Path):
    t = make_default_trainer(output_dir=tmp_path)
    yield t
    t.environment.close()


def test_environment_step_returns_expected_shape(trainer: Trainer):
    state = trainer.environment.reset()
    assert len(state) == 2

    next_state, reward, done, info = trainer.environment.step(1)
    assert len(next_state) == 2
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)


def test_is_reachable(trainer: Trainer):
    assert trainer.environment.is_reachable(0.0, 0.0)
    assert not trainer.environment.is_reachable(2.0, 0.0)
    assert not trainer.environment.is_reachable(0.0, 1.0)


@pytest.mark.parametrize("policy", ["Dueling DQN", "D3QN", "DDQN+PER"])
def test_run_episode_for_policies(trainer: Trainer, policy: str):
    params = trainer.policy_defaults[policy]
    params["batch_size"] = 2
    params["replay_size"] = 64
    trainer.get_or_create_agent(policy, overrides=params)

    progress_calls = []

    def cb(step: int):
        progress_calls.append(step)

    result = trainer.run_episode(policy=policy, epsilon=0.5, max_steps=15, progress_callback=cb)

    assert "total_reward" in result
    assert "steps" in result
    assert "transitions" in result
    assert result["steps"] <= 15
    assert len(progress_calls) == result["steps"]


def test_basic_learning_update(trainer: Trainer):
    policy = "D3QN"
    params = dict(trainer.policy_defaults[policy])
    params["batch_size"] = 4
    params["replay_size"] = 256
    trainer.get_or_create_agent(policy, overrides=params)

    losses = []
    for _ in range(8):
        result = trainer.run_episode(policy=policy, epsilon=0.8, max_steps=30)
        if result["last_loss"] is not None:
            losses.append(result["last_loss"])

    assert losses, "Expected at least one learning step with non-null loss"
    assert all(math.isfinite(v) for v in losses)


def test_train_and_save_csv(trainer: Trainer, tmp_path: Path):
    rewards = trainer.train(policy="Dueling DQN", num_episodes=3, max_steps=10, epsilon=0.3, save_csv="sample_run")
    assert len(rewards) == 3
    output_csv = tmp_path / "results_csv" / "sample_run.csv"
    assert output_csv.exists()


def _can_start_tk() -> bool:
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.update_idletasks()
        root.destroy()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _can_start_tk(), reason="Tk display unavailable")
def test_gui_smoke_clear_and_reset_do_not_raise():
    import tkinter as tk

    from MountainCar_gui import MountainCarGUI

    root = tk.Tk()
    root.withdraw()
    gui = MountainCarGUI(root)
    gui.clear_plot()
    gui.reset_all()
    root.update_idletasks()
    gui._on_close()


@pytest.mark.skipif(not _can_start_tk(), reason="Tk display unavailable")
def test_gui_smoke_legend_toggle_after_plot():
    import tkinter as tk

    from MountainCar_gui import MountainCarGUI

    root = tk.Tk()
    root.withdraw()
    gui = MountainCarGUI(root)
    snapshot = gui._snapshot_training_params()
    label = gui._build_run_label(snapshot, "Dueling DQN")
    gui._append_run_plot(label, [-200.0, -180.0, -160.0], 2)

    assert gui._line_registry
    root.update_idletasks()
    gui._on_close()
