from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import pytest

from Ant_gui import AntGUI


def test_gui_smoke_startup_and_clear() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.clear_plot()
    gui.reset_all()
    root.update_idletasks()
    root.destroy()


def test_policy_switch_preserves_values() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.policy_var.set("SAC")
    gui._populate_specific_panel()
    gui.policy_param_vars["SAC"]["gamma"].set("0.91")

    gui.policy_var.set("TQC")
    gui._populate_specific_panel()
    gui.policy_param_vars["TQC"]["gamma"].set("0.88")

    gui.policy_var.set("SAC")
    gui._populate_specific_panel()
    assert gui.policy_param_vars["SAC"]["gamma"].get() == "0.91"

    root.destroy()


def test_strict_visual_update_toggle_exists() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    assert gui.strict_update_var.get() is False
    gui.strict_update_var.set(True)
    assert gui.strict_update_var.get() is True
    root.destroy()


def test_legend_text_pick_toggles_visibility() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui._run_history["run-1"] = {
        "rewards": [1.0, 2.0],
        "ma": [1.0, 1.5],
        "eval": [],
        "visible": True,
        "label": "label-run-1",
    }

    class _Artist:
        def get_label(self) -> str:
            return ""

        def get_text(self) -> str:
            return "label-run-1"

    class _Event:
        artist = _Artist()

    gui._on_legend_pick(_Event())
    assert gui._run_history["run-1"]["visible"] is False
    root.destroy()


def test_paused_train_and_run_starts_fresh(monkeypatch) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui._training_active = True
    gui._paused = True

    called = {"cancel": 0, "start": 0}

    def _fake_cancel() -> None:
        called["cancel"] += 1

    def _fake_start(single_episode: bool) -> None:
        assert single_episode is False
        called["start"] += 1

    monkeypatch.setattr(gui, "_cancel_active_trainers", _fake_cancel)
    monkeypatch.setattr(gui, "_start_training", _fake_start)

    gui.train_and_run()

    assert called["cancel"] == 1
    assert called["start"] == 1
    root.destroy()


def test_stale_session_event_is_ignored() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    current = gui._session_id
    gui._event_queue.put({"type": "episode", "session_id": "old-session", "run_id": "r", "episode": 1, "episodes": 1, "reward": 1.0, "moving_average": 1.0, "lr": 1e-3, "best_reward": 1.0, "render_state": "off"})
    gui._event_queue.put({"type": "episode", "session_id": current, "run_id": "r", "episode": 1, "episodes": 1, "reward": 1.0, "moving_average": 1.0, "lr": 1e-3, "best_reward": 1.0, "render_state": "off"})

    gui._pump_events()
    assert "r" in gui._run_history
    assert len(gui._run_history["r"]["rewards"]) == 1
    root.destroy()


def test_episode_aux_updates_eval_without_reward_dup() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui._render_run_id = "run-1"

    gui._handle_event({"type": "episode", "session_id": gui._session_id, "run_id": "run-1", "episode": 1, "episodes": 2, "reward": 2.0, "moving_average": 2.0, "lr": 1e-3, "best_reward": 2.0, "render_state": "off"})
    before = len(gui._run_history["run-1"]["rewards"])
    gui._handle_event({"type": "episode_aux", "session_id": gui._session_id, "run_id": "run-1", "episode": 1, "episodes": 2, "eval_points": [(1, 3.0)], "frames": []})

    assert len(gui._run_history["run-1"]["rewards"]) == before
    assert gui._run_history["run-1"]["eval"] == [(1, 3.0)]
    root.destroy()


def test_compare_render_index_prefers_selected_policy() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.compare_on_var.set(True)
    gui.policy_var.set("TQC")
    idx = gui._select_render_run_index([
        {"Policy": "SAC"},
        {"Policy": "TQC"},
    ])
    assert idx == 1
    root.destroy()


def test_combobox_mousewheel_is_blocked() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    assert gui.device_combo.bind("<MouseWheel>")
    assert gui.policy_combo.bind("<MouseWheel>")
    assert gui.compare_param_combo.bind("<MouseWheel>")
    root.destroy()


def test_cancel_active_trainers_resumes_before_cancel() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    calls = []

    class _FakeTrainer:
        def resume(self) -> None:
            calls.append("resume")

        def cancel(self) -> None:
            calls.append("cancel")

    with gui._trainers_lock:
        gui._active_trainers = {"r1": _FakeTrainer()}  # type: ignore[assignment]

    gui._cancel_active_trainers()
    assert calls == ["resume", "cancel"]
    root.destroy()


def test_live_plot_uses_dark_mode_theme() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    fig_bg = gui.figure.get_facecolor()
    ax_bg = gui.ax.get_facecolor()
    assert fig_bg[:3] == pytest.approx((30 / 255, 30 / 255, 30 / 255), rel=1e-3)
    assert ax_bg[:3] == pytest.approx((37 / 255, 37 / 255, 38 / 255), rel=1e-3)
    root.destroy()


def test_env_specific_parameters_are_two_column_below_update() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    root.update_idletasks()

    # Reacher-style pair layout: label/input | label/input in env group.
    label_cols = {}
    for w in gui.env_params_group.winfo_children():
        if isinstance(w, ttk.Label):
            text = str(w.cget("text"))
            if text in {"forward_reward_weight", "ctrl_cost_weight", "contact_cost_weight"}:
                label_cols[text] = int(w.grid_info()["column"])

    assert label_cols["forward_reward_weight"] == 0
    assert label_cols["ctrl_cost_weight"] == 2
    assert label_cols["contact_cost_weight"] == 0
    root.destroy()


def test_step_event_updates_steps_progress() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.max_steps_var.set("100")
    gui._handle_event({"type": "step", "steps": 25, "session_id": gui._session_id})
    assert float(gui.steps_progress["value"]) == pytest.approx(25.0)
    root.destroy()


def test_shared_label_width_is_enforced() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    labels = [w for w in gui.general_params_group.winfo_children() if isinstance(w, ttk.Label)]
    assert labels
    assert all(int(lbl.cget("width")) == gui.LABEL_WIDTH_CHARS for lbl in labels)
    root.destroy()


def test_legend_hover_changes_label_style() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui._run_history["run-1"] = {
        "rewards": [1.0],
        "ma": [1.0],
        "eval": [],
        "visible": True,
        "label": "label-run-1",
    }
    gui._refresh_plot()
    if gui._legend is None:
        pytest.skip("Legend not available in this Tk/mpl setup")

    gui._set_legend_hover_state(True)
    txt = gui._legend.get_texts()[0]
    assert txt.get_color() == "#ffffff"
    root.destroy()


def test_specific_shared_parameters_use_pair_columns() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.policy_var.set("SAC")
    gui._populate_specific_panel()
    root.update_idletasks()

    label_cols = {}
    for w in gui.specific_shared_frame.winfo_children():
        if isinstance(w, ttk.Label):
            text = str(w.cget("text"))
            if text in {"gamma", "learning_rate", "batch_size", "hidden_layer"}:
                label_cols[text] = int(w.grid_info()["column"])

    assert label_cols["gamma"] == 0
    assert label_cols["learning_rate"] == 2
    assert label_cols["batch_size"] == 0
    assert label_cols["hidden_layer"] == 2
    root.destroy()


def test_comboboxes_use_dark_style() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    assert str(gui.device_combo.cget("style")) == "Dark.TCombobox"
    assert str(gui.policy_combo.cget("style")) == "Dark.TCombobox"
    assert str(gui.compare_param_combo.cget("style")) == "Dark.TCombobox"
    root.destroy()


def test_parameter_groups_stretch_with_panel_width() -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    root.geometry("1700x900")
    root.update_idletasks()

    canvas_w = int(gui.param_canvas.winfo_width())
    group_w = int(gui.env_params_group.winfo_width())
    assert group_w >= int(canvas_w * 0.9)
    root.destroy()


def test_train_start_invalid_input_does_not_crash(monkeypatch) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.max_steps_var.set("not-a-number")

    shown = {"called": 0}

    def _fake_showerror(title: str, msg: str) -> None:
        shown["called"] += 1

    monkeypatch.setattr("Ant_gui.messagebox.showerror", _fake_showerror)

    gui.train_and_run()

    assert shown["called"] == 1
    assert gui._training_active is False
    root.destroy()


def test_worker_render_safety_policy(monkeypatch) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    monkeypatch.delenv("ANT_DISABLE_WORKER_RENDER", raising=False)
    assert gui._is_worker_render_safe() is True
    monkeypatch.setenv("ANT_DISABLE_WORKER_RENDER", "1")
    assert gui._is_worker_render_safe() is False
    root.destroy()


def test_single_animated_run_uses_process_mode(monkeypatch) -> None:
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk unavailable: {exc}")

    gui = AntGUI(root)
    gui.animation_on_var.set(True)
    gui.compare_on_var.set(False)

    called = {"process": 0}

    class _FakeEvent:
        def __init__(self) -> None:
            self._flag = False

        def set(self) -> None:
            self._flag = True

        def clear(self) -> None:
            self._flag = False

        def is_set(self) -> bool:
            return self._flag

    class _FakeProcess:
        def __init__(self, *args, **kwargs):
            self.exitcode = None

        def start(self) -> None:
            called["process"] += 1

        def is_alive(self) -> bool:
            return False

        def join(self, timeout: float | None = None) -> None:
            _ = timeout

    class _FakeCtx:
        def Event(self) -> _FakeEvent:
            return _FakeEvent()

        def Process(self, *args, **kwargs) -> _FakeProcess:
            return _FakeProcess(*args, **kwargs)

    gui._mp_ctx = _FakeCtx()  # type: ignore[assignment]
    monkeypatch.setattr(gui, "_is_worker_render_safe", lambda: True)

    gui._start_training(single_episode=True)
    root.update()

    assert called["process"] == 1
    root.destroy()
