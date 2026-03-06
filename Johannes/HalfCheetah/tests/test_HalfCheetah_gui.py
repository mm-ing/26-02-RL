from __future__ import annotations

import pytest

tk = pytest.importorskip("tkinter")

from HalfCheetah_gui import HalfCheetahGUI
from HalfCheetah_logic import GENERAL_DEFAULTS, POLICY_DEFAULTS


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


def test_policy_specific_visibility_switches_with_policy(tk_root):
    gui = HalfCheetahGUI(tk_root)

    gui.policy_var.set("PPO")
    gui._apply_policy_defaults()
    assert gui.ppo_n_steps_entry.winfo_manager() == "grid"
    assert gui.off_replay_size_entry.winfo_manager() == ""

    gui.policy_var.set("SAC")
    gui._apply_policy_defaults()
    assert gui.ppo_n_steps_entry.winfo_manager() == ""
    assert gui.off_replay_size_entry.winfo_manager() == "grid"
    assert gui.off_learning_frequency_entry.winfo_manager() == "grid"


def test_compare_param_options_follow_selected_policy(tk_root):
    gui = HalfCheetahGUI(tk_root)

    gui.policy_var.set("PPO")
    gui._apply_policy_defaults()
    ppo_options = set(gui.compare_param_combo.cget("values"))
    assert "n_steps" in ppo_options
    assert "replay_size" not in ppo_options

    gui.policy_var.set("TD3")
    gui._apply_policy_defaults()
    td3_options = set(gui.compare_param_combo.cget("values"))
    assert "n_steps" not in td3_options
    assert "replay_size" in td3_options
    assert "learning_start" in td3_options


def test_legend_includes_compare_specific_values(tk_root):
    gui = HalfCheetahGUI(tk_root)
    run_id = "run-compare-1"
    gui.run_history[run_id] = {
        "episodes": [1, 2],
        "rewards": [1.0, 2.0],
        "ma": [1.0, 1.5],
        "eval": [],
    }
    gui.run_metadata[run_id] = {
        "policy": "TD3",
        "max_steps": 1000,
        "gamma": 0.99,
        "epsilon_max": 0.0,
        "epsilon_decay": 1.0,
        "epsilon_min": 0.0,
        "lr": 0.0003,
        "lr_strategy": "constant",
        "lr_decay": 1.0,
        "env": {
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 0.1,
            "reset_noise_scale": 0.1,
        },
        "compare_values": {
            "learning_start": 5000,
            "batch_size": 256,
            "gamma": 0.99,
        },
    }

    gui._redraw_plot()
    reward_labels = [ln.get_label() for ln in gui.ax.lines if "| steps=" in ln.get_label()]
    assert reward_labels
    reward_label = reward_labels[0]
    assert "compare:" in reward_label
    assert "learning_start=5000" in reward_label
    assert "batch_size=256" in reward_label
    assert "compare: gamma=" not in reward_label


def test_policy_switch_preserves_current_values(tk_root):
    gui = HalfCheetahGUI(tk_root)

    gui.policy_var.set("PPO")
    gui._on_policy_changed()
    gui.gamma_var.set(0.975)
    gui.batch_size_var.set(192)
    gui.lr_var.set("2e-4")
    gui.lr_decay_var.set(0.93)
    gui.n_steps_var.set(1024)

    gui.policy_var.set("TD3")
    gui._on_policy_changed()

    assert float(gui.gamma_var.get()) == pytest.approx(0.975)
    assert int(gui.batch_size_var.get()) == 192
    assert str(gui.lr_var.get()) == "2e-4"
    assert float(gui.lr_decay_var.get()) == pytest.approx(0.93)
    assert int(gui.n_steps_var.get()) == 1024


def test_reset_all_restores_general_and_policy_defaults(tk_root):
    gui = HalfCheetahGUI(tk_root)

    gui.policy_var.set("TD3")
    gui._on_policy_changed()
    gui.max_steps_var.set(321)
    gui.episodes_var.set(17)
    gui.gamma_var.set(0.91)
    gui.batch_size_var.set(111)
    gui.lr_var.set("9e-4")
    gui.lr_strategy_var.set("linear")
    gui.n_steps_var.set(777)

    gui.reset_all()

    ppo_defaults = POLICY_DEFAULTS["PPO"]
    assert gui.policy_var.get() == "PPO"
    assert int(gui.max_steps_var.get()) == int(GENERAL_DEFAULTS["max_steps"])
    assert int(gui.episodes_var.get()) == int(GENERAL_DEFAULTS["episodes"])
    assert float(gui.gamma_var.get()) == pytest.approx(float(GENERAL_DEFAULTS["gamma"]))
    assert int(gui.batch_size_var.get()) == int(ppo_defaults["batch_size"])
    assert float(gui.lr_var.get()) == pytest.approx(float(ppo_defaults["lr"]))
    assert str(gui.lr_strategy_var.get()) == str(ppo_defaults["lr_strategy"])
    assert int(gui.n_steps_var.get()) == int(ppo_defaults["n_steps"])
