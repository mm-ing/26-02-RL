from pathlib import Path
import sys

import pytest

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _can_create_tk_root():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        root.destroy()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _can_create_tk_root(), reason="Tk display not available")


def _make_tk_root_or_skip():
    import tkinter as tk

    try:
        root = tk.Tk()
    except Exception as exc:
        pytest.skip(f"Tk unavailable in runtime environment: {exc}")
    return root


def test_gui_startup_and_safe_actions():
    from BipedalWalker_gui import BipedalWalkerGUI

    root = _make_tk_root_or_skip()
    root.withdraw()
    gui = BipedalWalkerGUI(root)

    gui._clear_plot()
    gui._clear_compare()
    gui._reset_all()

    root.update_idletasks()
    root.destroy()


def test_compare_toggle_disables_animation():
    from BipedalWalker_gui import BipedalWalkerGUI

    root = _make_tk_root_or_skip()
    root.withdraw()
    gui = BipedalWalkerGUI(root)

    gui.var_animation_on.set(True)
    gui.var_compare_on.set(True)
    gui._on_compare_toggle()

    assert gui.var_animation_on.get() is False
    root.destroy()


def test_animation_toggle_off_clears_replay_queue():
    from BipedalWalker_gui import BipedalWalkerGUI

    root = _make_tk_root_or_skip()
    root.withdraw()
    gui = BipedalWalkerGUI(root)

    gui._rollout_playback_frames = [object(), object()]
    gui._rollout_playback_index = 1
    gui._rollout_playback_total = 2
    gui.steps_progress["value"] = 50
    gui.render_state = "on"

    gui.var_animation_on.set(False)
    gui._on_animation_toggle_changed()

    assert gui._rollout_playback_frames == []
    assert gui._rollout_playback_index == 0
    assert gui._rollout_playback_total == 0
    assert float(gui.steps_progress["value"]) == 0.0
    assert gui.render_state == "off"
    assert "Render: off" in gui.var_status.get()

    root.destroy()


def test_compare_parallel_max_workers_constant_is_4():
    from BipedalWalker_gui import BipedalWalkerGUI

    assert BipedalWalkerGUI.COMPARE_MAX_WORKERS == 4


def test_compare_mode_uses_bounded_parallel_executor(monkeypatch):
    from BipedalWalker_gui import BipedalWalkerGUI
    from BipedalWalker_logic import BipedalWalkerConfig

    root = _make_tk_root_or_skip()
    root.withdraw()
    gui = BipedalWalkerGUI(root)

    class FakeTrainer:
        def __init__(self, config, event_callback=None):
            self.config = config
            self.event_callback = event_callback
            self.stop_event = type("_Stop", (), {"is_set": lambda _self: False})()

        def train(self, collect_transitions=False, run_label="run"):
            return

        def set_paused(self, paused):
            return

        def stop(self):
            return

        def update_environment(self, *args, **kwargs):
            return

    class FakeFuture:
        def __init__(self, result_value=False):
            self._result = result_value

        def result(self):
            return self._result

    captured = {"max_workers": None, "submit_count": 0}

    class FakeExecutor:
        def __init__(self, max_workers):
            captured["max_workers"] = max_workers

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def submit(self, fn, *args, **kwargs):
            captured["submit_count"] += 1
            value = fn(*args, **kwargs)
            return FakeFuture(value)

    def fake_as_completed(futures):
        for future in futures:
            yield future

    monkeypatch.setattr("BipedalWalker_gui.BipedalWalkerTrainer", FakeTrainer)
    monkeypatch.setattr("BipedalWalker_gui.ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr("BipedalWalker_gui.as_completed", fake_as_completed)

    configs = [BipedalWalkerConfig(policy="PPO", episodes=1, max_steps=1) for _ in range(6)]
    gui._start_worker(configs, collect_transitions=False)
    gui.worker_thread.join(timeout=5)

    assert captured["max_workers"] == 4
    assert captured["submit_count"] == 6

    root.destroy()
