from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from HalfCheetah_gui import HalfCheetahGUI


@pytest.fixture
def tk_root():
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk/Tcl assets unavailable: {exc}")
    root.withdraw()
    yield root
    try:
        root.update_idletasks()
        root.destroy()
    except Exception:
        pass


def test_gui_smoke_startup(tk_root):
    gui = HalfCheetahGUI(tk_root)
    assert gui is not None
    assert gui.policy_var.get() in {"PPO", "SAC", "TD3"}


def test_compare_add_and_clear(tk_root):
    gui = HalfCheetahGUI(tk_root)
    gui.compare_param_var.set("policy")
    gui.compare_values_var.set("PPO,SAC")
    gui._compare_add()
    assert "policy" in gui.compare_items
    gui._compare_clear()
    assert gui.compare_items == {}


def test_reset_and_clear_plot_safety(tk_root):
    gui = HalfCheetahGUI(tk_root)
    gui.clear_plot()
    gui.reset_all()
    assert gui.steps_progress_var.get() == 0
    assert gui.episodes_progress_var.get() == 0


def test_redraw_plot_with_history_data(tk_root):
    gui = HalfCheetahGUI(tk_root)
    run_id = "run-1"
    gui.run_history[run_id] = {
        "episodes": [1, 2, 3],
        "rewards": [10.0, 11.0, 12.0],
        "ma": [10.0, 10.5, 11.0],
        "eval": [(1, 9.5), (3, 11.5)],
    }
    gui._redraw_plot()
    assert run_id in gui.run_colors
