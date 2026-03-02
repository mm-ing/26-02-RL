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
