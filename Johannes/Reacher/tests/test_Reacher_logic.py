from __future__ import annotations

from pathlib import Path

from Reacher_logic import EnvironmentConfig, ReacherTrainer, TrainingConfig, make_lr_schedule, parse_hidden_layers


def test_hidden_layer_parser() -> None:
    assert parse_hidden_layers("256") == [256, 256]
    assert parse_hidden_layers("256,128,64") == [256, 128, 64]
    assert parse_hidden_layers("", fallback=(32, 32)) == [32, 32]


def test_lr_schedule_variants() -> None:
    linear = make_lr_schedule(1e-3, "linear", 1e-5, 0.99)
    expo = make_lr_schedule(1e-3, "exponential", 1e-5, 0.9)
    const = make_lr_schedule(1e-3, "constant", 1e-5, 0.9)

    assert linear(1.0) >= linear(0.0)
    assert expo(1.0) >= expo(0.0)
    assert const(0.0) == const(1.0)


def test_run_episode_returns_real_step_count() -> None:
    trainer = ReacherTrainer(
        EnvironmentConfig(render_enabled=False),
        TrainingConfig(episodes=1, max_steps=8),
    )
    data = trainer.run_episode(training=False, collect_transitions=False)
    assert data["steps"] > 0
    assert data["steps"] <= 8


def test_eval_cadence_and_csv_export(tmp_path: Path) -> None:
    events = []

    trainer = ReacherTrainer(
        EnvironmentConfig(render_enabled=False),
        TrainingConfig(episodes=12, max_steps=2, eval_rollout_on=True),
        event_sink=events.append,
    )
    out = trainer.train()

    assert out["type"] == "training_done"
    assert trainer.eval_points and trainer.eval_points[0][0] == 10

    csv_path = trainer.export_transitions_csv(str(tmp_path))
    assert csv_path is not None
    assert csv_path.exists()


def test_pause_resume_and_cancel_transitions() -> None:
    trainer = ReacherTrainer(
        EnvironmentConfig(render_enabled=False),
        TrainingConfig(episodes=2, max_steps=4),
    )

    trainer.set_pause(True)
    assert not trainer.pause_event.is_set()
    trainer.set_pause(False)
    assert trainer.pause_event.is_set()

    trainer.cancel()
    assert trainer.cancel_event.is_set()
    assert trainer.pause_event.is_set()
