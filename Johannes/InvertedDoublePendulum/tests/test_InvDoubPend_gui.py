import tkinter as tk

import pytest


class _DummyPreviewEnv:
    def __init__(self, *_args, **_kwargs):
        self.env = self

    def reset(self):
        return [0.0], {}

    def render(self):
        return None

    def update(self, *_args, **_kwargs):
        return None

    def close(self):
        return None


def _make_root_or_skip():
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except Exception as exc:
        pytest.skip(f"Tk runtime unavailable: {exc}")


def test_gui_smoke_startup_and_clear(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    app.clear_plot()
    app.reset_all()
    assert app.training_active is False

    root.update_idletasks()
    root.destroy()


def test_animation_toggle_runtime_hotfix(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    app.playback_frames = [1, 2, 3]
    app.playback_index = 2
    app.var_animation_on.set(False)
    app._on_animation_toggle()

    assert app.playback_frames == []
    assert app.playback_index == 0
    assert "Render: off" in app.status_var.get()

    root.update_idletasks()
    root.destroy()


def test_pause_unhighlights_train_and_train_restarts_fresh(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    class _DummyThread:
        def __init__(self, target=None, daemon=None):
            self.target = target
            self.daemon = daemon

        def start(self):
            return None

    monkeypatch.setattr(gui_mod.threading, "Thread", _DummyThread)

    class _DummyWorker:
        def __init__(self):
            self.pause_called = 0
            self.resume_called = 0
            self.stop_called = 0

        def pause(self):
            self.pause_called += 1

        def resume(self):
            self.resume_called += 1

        def stop(self):
            self.stop_called += 1

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    worker = _DummyWorker()
    app.current_workers = [worker]
    app.training_active = True
    app.paused = False
    app.btn_train.configure(style="Accent.TButton")

    app.toggle_pause()

    assert app.paused is True
    assert worker.pause_called == 1
    assert str(app.btn_train.cget("style")) == "Control.TButton"
    assert str(app.btn_pause.cget("style")) == "Amber.TButton"
    assert str(app.btn_pause.cget("text")) == "Run"

    old_session = app.current_training_session_id
    app.var_episodes.set("120")
    app.train_and_run()

    assert worker.resume_called == 1
    assert worker.stop_called == 1
    assert app.training_active is True
    assert app.paused is False
    assert app.current_training_session_id == old_session + 1
    assert str(app.btn_train.cget("style")) == "Accent.TButton"
    assert str(app.btn_pause.cget("style")) == "Control.TButton"
    assert str(app.btn_pause.cget("text")) == "Pause"

    root.update_idletasks()
    root.destroy()


def test_stale_session_events_are_ignored(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    app.current_training_session_id = 5
    app.training_active = True
    app.paused = False
    app.btn_train.configure(style="Accent.TButton")
    app.btn_pause.configure(text="Pause", style="Control.TButton")

    app.event_queue.put({"type": "all_done", "session_id": 4})
    app._consume_events()

    assert app.training_active is True
    assert app.paused is False
    assert str(app.btn_train.cget("style")) == "Accent.TButton"
    assert str(app.btn_pause.cget("style")) == "Control.TButton"
    assert str(app.btn_pause.cget("text")) == "Pause"

    root.update_idletasks()
    root.destroy()


def test_stale_episode_event_does_not_update_progress_or_status(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    app.current_training_session_id = 7
    app.render_run_id = "run-current"
    app.episodes_progress["value"] = 0.0
    app.status_var.set("Epsilon: - | LR: - | Best reward: - | Render: idle")

    stale_episode = {
        "type": "episode",
        "session_id": 6,
        "run_id": "run-current",
        "episode": 4,
        "episodes": 10,
        "reward": 1.0,
        "moving_average": 1.0,
        "eval_points": [],
        "steps": 10,
        "epsilon": 0.0,
        "lr": 3e-4,
        "best_reward": 1.0,
        "render_state": "on",
        "frames": [],
    }
    app.event_queue.put(stale_episode)
    app._consume_events()

    assert float(app.episodes_progress["value"]) == 0.0
    assert app.status_var.get() == "Epsilon: - | LR: - | Best reward: - | Render: idle"
    assert "run-current" not in app.plot_series

    root.update_idletasks()
    root.destroy()


def test_reset_all_resumes_then_stops_workers(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    class _DummyWorker:
        def __init__(self):
            self.calls = []

        def resume(self):
            self.calls.append("resume")

        def stop(self):
            self.calls.append("stop")

    worker = _DummyWorker()
    app.current_workers = [worker]
    app.training_active = True
    app.paused = True

    app.reset_all()

    assert worker.calls == ["resume", "stop"]
    assert app.training_active is False
    assert app.paused is False

    root.update_idletasks()
    root.destroy()


def test_compare_build_runs_have_unique_run_ids(monkeypatch):
    try:
        import InvDoubPend_gui as gui_mod
    except Exception as exc:
        pytest.skip(f"GUI import skipped: {exc}")

    monkeypatch.setattr(gui_mod, "InvertedDoublePendulumEnvironment", _DummyPreviewEnv)
    monkeypatch.setattr(gui_mod, "timestamp_run_id", lambda _prefix="cmp": "cmp_fixed")

    root = _make_root_or_skip()
    app = gui_mod.InvDoubPendGUI(root)

    app.var_compare_on.set(True)
    app.compare_entries = {
        "Policy": ["PPO", "SAC"],
        "Gamma": ["0.99", "0.995"],
    }

    runs = app._build_compare_runs()
    run_ids = [r["run_id"] for r in runs]

    assert len(runs) == 4
    assert len(set(run_ids)) == len(run_ids)

    root.update_idletasks()
    root.destroy()
