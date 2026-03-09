from __future__ import annotations

import tkinter as tk

import pytest

from Pusher_gui import PusherGUI


def _tk_root_or_skip():
    try:
        root = tk.Tk()
        root.withdraw()
        return root
    except tk.TclError:
        pytest.skip("Tk is not available in this environment")


def test_gui_smoke_startup_and_required_panels():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)
    assert gui.environment_panel.winfo_exists()
    assert gui.parameters_panel.winfo_exists()
    assert gui.controls_panel.winfo_exists()
    assert gui.current_run_panel.winfo_exists()
    assert gui.plot_panel.winfo_exists()
    root.destroy()


def test_controls_row_contains_8_controls_in_order():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)
    labels = [
        gui.run_single_btn.cget("text"),
        gui.train_btn.cget("text"),
        gui.pause_run_btn.cget("text"),
        gui.reset_btn.cget("text"),
        gui.clear_plot_btn.cget("text"),
        gui.save_csv_btn.cget("text"),
        gui.save_plot_btn.cget("text"),
    ]
    assert labels == [
        "Run single episode",
        "Train and Run",
        "Pause",
        "Reset All",
        "Clear Plot",
        "Save samplings CSV",
        "Save Plot PNG",
    ]
    assert isinstance(gui.device_combo, tk.Widget)
    root.destroy()


def test_policy_switch_preserves_independent_values():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui.policy_var.set("PPO")
    gui._on_policy_selected()
    gui.shared_param_vars["learning_rate"].set("7e-4")
    gui.policy_specific_vars["PPO"]["n_steps"].set(2048)
    gui._capture_policy_values("PPO")

    gui.policy_var.set("SAC")
    gui._on_policy_selected()
    gui.shared_param_vars["learning_rate"].set("2e-4")
    gui.policy_specific_vars["SAC"]["learning_starts"].set(777)
    gui._capture_policy_values("SAC")

    gui.policy_var.set("PPO")
    gui._on_policy_selected()
    assert gui.shared_param_vars["learning_rate"].get() == "7e-4"
    assert gui.policy_specific_vars["PPO"]["n_steps"].get() == 2048

    gui.policy_var.set("SAC")
    gui._on_policy_selected()
    assert gui.shared_param_vars["learning_rate"].get() == "2e-4"
    assert gui.policy_specific_vars["SAC"]["learning_starts"].get() == 777
    root.destroy()


def test_activation_and_lr_strategy_are_readonly_selectors():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    values = list(gui.compare_param_combo.cget("values"))
    assert "activation" in values
    assert "lr_strategy" in values
    root.destroy()


def test_compare_toggle_disables_animation_by_default():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)
    gui.animate_var.set(True)
    gui.compare_on_var.set(True)
    gui._on_compare_toggle()
    assert gui.animate_var.get() is False
    root.destroy()


def test_compare_tab_completion_places_caret_at_end_and_hint_shows():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui.compare_param_var.set("activation")
    gui.compare_values_var.set("ta")
    gui._update_completion_hint()
    assert "Tab -> tanh".lower() in gui.completion_hint.cget("text").lower()

    gui._accept_completion()
    assert gui.compare_values_var.get().lower() == "tanh"
    assert gui.compare_values_entry.index(tk.INSERT) == len(gui.compare_values_var.get())
    root.destroy()


def test_compare_add_and_clear_summary_lines():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui.compare_param_var.set("policy")
    gui.compare_values_var.set("PPO,SAC")
    gui._add_compare_param()
    assert "policy: [PPO, SAC]" in gui.compare_summary_var.get()

    gui._clear_compare_params()
    assert gui.compare_summary_var.get() == ""
    root.destroy()


def test_reset_all_restores_defaults():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui.episodes_var.set(42)
    gui.max_steps_var.set(12)
    gui.env_reward_near_weight_var.set(0.9)
    gui.policy_var.set("PPO")
    gui._on_policy_selected()
    gui.shared_param_vars["learning_rate"].set("9e-4")

    gui.reset_all()
    assert gui.episodes_var.get() == 3000
    assert gui.max_steps_var.get() == 200
    assert gui.env_reward_near_weight_var.get() == pytest.approx(0.5)
    assert gui.policy_var.get() == "SAC"
    root.destroy()


def test_stale_session_event_is_ignored():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui.active_session_id = "live"
    before = len(gui._run_series)
    gui._handle_worker_event({"session_id": "old", "type": "episode", "run_id": "r1", "episode": 1, "episodes": 2, "reward": 1.0, "moving_average": 1.0, "eval_points": [], "epsilon": 0.0, "lr": 0.0, "best_reward": 1.0})
    after = len(gui._run_series)
    assert before == after
    root.destroy()


def test_animation_queue_latest_wins_behavior():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui._animation_active = True
    gui._queue_animation_frames([1, 2])
    assert gui._animation_pending == [1, 2]
    gui._queue_animation_frames([9])
    assert gui._animation_pending == [9]
    root.destroy()


def test_legend_visibility_toggle_persists_across_redraws():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui._run_series = {
        "run1": {
            "episodes": [1, 2],
            "rewards": [1.0, 1.5],
            "ma": [1.0, 1.25],
            "eval": [(2, 1.4)],
            "label": "run1",
            "color": "#1f77b4",
            "policy": "SAC",
        }
    }
    gui._run_visibility = {"run1": True}
    gui._redraw_plot()

    gui._toggle_run_visibility("run1")
    assert gui._run_visibility["run1"] is False

    gui._redraw_plot()
    assert gui._run_visibility["run1"] is False
    root.destroy()


def test_plot_motion_cursor_changes_without_crash():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    class E:
        x = None
        y = None

    gui._on_plot_motion(E())
    assert gui.canvas.get_tk_widget().cget("cursor") in {"", "arrow", "hand2"}
    root.destroy()


def test_on_close_cancels_training_and_destroys_root(monkeypatch):
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    called = {"cancel": False, "destroy": False}

    def _fake_cancel():
        called["cancel"] = True

    def _fake_destroy():
        called["destroy"] = True

    monkeypatch.setattr(gui, "cancel_training", _fake_cancel)
    monkeypatch.setattr(gui.master, "destroy", _fake_destroy)

    gui._on_close()
    assert called == {"cancel": True, "destroy": True}


def test_control_highlight_contract_for_idle_run_pause_states():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    gui._training_active = False
    gui._paused = False
    gui._update_control_highlights()
    assert str(gui.pause_run_btn.cget("state")) == tk.DISABLED
    assert gui.pause_run_btn.cget("text") == "Pause"

    gui._training_active = True
    gui._paused = False
    gui._update_control_highlights()
    assert str(gui.pause_run_btn.cget("state")) == tk.NORMAL
    assert gui.pause_run_btn.cget("text") == "Pause"
    assert gui.train_btn.cget("style") == "Train.TButton"

    gui._training_active = True
    gui._paused = True
    gui._update_control_highlights()
    assert str(gui.pause_run_btn.cget("state")) == tk.NORMAL
    assert gui.pause_run_btn.cget("text") == "Run"
    assert gui.pause_run_btn.cget("style") == "Pause.TButton"

    root.destroy()


def test_pause_or_run_calls_trainer_pause_and_resume():
    root = _tk_root_or_skip()
    gui = PusherGUI(root)

    class _DummyTrainer:
        def __init__(self):
            self.pause_calls = 0
            self.resume_calls = 0

        def request_pause(self):
            self.pause_calls += 1

        def request_resume(self):
            self.resume_calls += 1

    trainer = _DummyTrainer()
    gui._training_active = True
    gui._paused = False
    gui.worker_trainers = {"r1": trainer}

    gui.pause_or_run()
    assert gui._paused is True
    assert trainer.pause_calls == 1

    gui.pause_or_run()
    assert gui._paused is False
    assert trainer.resume_calls == 1

    root.destroy()
