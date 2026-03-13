from __future__ import annotations

import tkinter as tk

import pytest


class DummyTrainer:
    def __init__(self, event_sink=None):
        self.event_sink = event_sink
        self.pause_event = type("PauseEvent", (), {"is_set": lambda self: True})()

    def update_environment(self, config):
        return None

    def train(self, config):
        return None

    def pause(self):
        return None

    def resume(self):
        return None

    def cancel(self):
        return None


def test_policy_snapshot_isolation(monkeypatch):
    import Humanoid_gui

    monkeypatch.setattr(Humanoid_gui, "build_default_trainer", lambda event_sink=None: DummyTrainer(event_sink=event_sink))

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("Tk is not available in this environment.")

    try:
        root.withdraw()
        gui = Humanoid_gui.HumanoidGUI(root)

        gui.policy_var.set("SAC")
        gui._populate_specific_panel()
        gui.policy_param_vars["SAC"]["learning_rate"].set("0.0005")
        gui.policy_param_vars["SAC"]["buffer_size"].set("12345")

        gui.policy_var.set("PPO")
        gui._populate_specific_panel()
        gui.policy_param_vars["PPO"]["learning_rate"].set("0.001")
        gui.policy_param_vars["PPO"]["n_steps"].set("2048")

        gui.policy_var.set("SAC")
        gui._populate_specific_panel()
        assert gui.policy_param_vars["SAC"]["learning_rate"].get() == "0.0005"
        assert gui.policy_param_vars["SAC"]["buffer_size"].get() == "12345"

        gui.policy_var.set("PPO")
        gui._populate_specific_panel()
        assert gui.policy_param_vars["PPO"]["learning_rate"].get() == "0.001"
        assert gui.policy_param_vars["PPO"]["n_steps"].get() == "2048"
    finally:
        root.destroy()
