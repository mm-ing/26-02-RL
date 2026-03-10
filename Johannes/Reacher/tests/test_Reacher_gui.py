from __future__ import annotations

import pytest

pytest.importorskip("tkinter")

from Reacher_gui import POLICY_DEFAULTS, ReacherGUI


@pytest.fixture
def root():
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk/Tcl not available: {exc}")
    root.withdraw()
    yield root
    root.destroy()


def test_gui_smoke(root) -> None:
    gui = ReacherGUI(root)
    assert gui.var_policy.get() in {"PPO", "SAC", "TD3"}
    assert gui.var_device.get() == "CPU"
    assert gui.var_eval_rollout_on.get() is False


def test_eval_rollout_toggle_propagates_to_train_config(root) -> None:
    gui = ReacherGUI(root)
    gui.var_eval_rollout_on.set(True)
    cfg = gui._build_train_config("run_eval_toggle")
    assert cfg.eval_rollout_on is True


def test_policy_switch_keeps_cache(root) -> None:
    gui = ReacherGUI(root)
    gui.var_policy.set("SAC")
    gui._render_specific_fields()
    gui._specific_vars["gamma"].set("0.93")
    gui._cache_policy_values()

    gui.var_policy.set("PPO")
    gui._on_policy_change()
    gui._specific_vars["gamma"].set("0.91")
    gui._cache_policy_values()

    gui.var_policy.set("SAC")
    gui._on_policy_change()
    assert gui._specific_vars["gamma"].get() == "0.93"


def test_compare_tab_completion(root) -> None:
    gui = ReacherGUI(root)
    gui.var_compare_param.set("activation")
    gui.var_compare_values.set("Ta")

    class Event:
        pass

    gui._tab_complete_compare(Event())
    assert gui.var_compare_values.get() == "Tanh"


def test_compare_values_include_general_fields(root) -> None:
    gui = ReacherGUI(root)
    values = gui._compare_param_values()
    assert "max_steps" in values
    assert "episodes" in values


def test_compare_allowed_keys_are_policy_specific(root) -> None:
    gui = ReacherGUI(root)
    sac_allowed = gui._allowed_compare_keys("SAC")
    ppo_allowed = gui._allowed_compare_keys("PPO")

    assert "train_freq" in sac_allowed
    assert "train_freq" not in ppo_allowed


def test_render_index_prefers_selected_policy(root) -> None:
    gui = ReacherGUI(root)
    grid = [{"Policy": "TD3"}, {"Policy": "SAC"}, {"Policy": "PPO"}]
    assert gui._select_render_combo_index(grid, "SAC") == 1
    assert gui._select_render_combo_index(grid, "A2C") == 0


def test_steps_progress_updates_from_playback_only(root) -> None:
    gui = ReacherGUI(root)
    initial = gui.steps_var.get()
    assert initial.startswith("Playback frames:")
    gui._handle_step_event({"step": 3, "max_steps": 10})
    assert gui.steps_var.get() == initial

    gui._start_playback([None, None])
    assert gui.steps_var.get() == "Playback frames: 1/2"


def test_policy_compare_uses_policy_defaults_baseline(root) -> None:
    gui = ReacherGUI(root)
    gui.var_policy.set("SAC")
    gui._render_specific_fields()
    gui._specific_vars["gamma"].set("0.5")

    cfg = gui._build_train_config("run_test", policy_override="SAC", use_policy_cache=False)
    assert cfg.gamma == POLICY_DEFAULTS["SAC"]["gamma"]


def test_parameter_inputs_use_two_columns(root) -> None:
    gui = ReacherGUI(root)

    # General group: Max steps and Episodes should sit on the same row in two columns.
    max_steps_entry = None
    episodes_entry = None
    for frame in gui.param_frame.winfo_children():
        if getattr(frame, "cget", None) and frame.cget("text") == "General":
            for w in frame.winfo_children():
                if isinstance(w, type(gui.entry_reward_dist)):
                    text_var = w.cget("textvariable")
                    if text_var == str(gui.var_max_steps):
                        max_steps_entry = w
                    if text_var == str(gui.var_episodes):
                        episodes_entry = w

    assert max_steps_entry is not None
    assert episodes_entry is not None
    assert int(max_steps_entry.grid_info()["row"]) == 0
    assert int(episodes_entry.grid_info()["row"]) == 0
    assert int(max_steps_entry.grid_info()["column"]) == 1
    assert int(episodes_entry.grid_info()["column"]) == 3

    # Environment group representative check: Animation FPS and Update rate share row in two columns.
    animation_fps_entry = None
    update_rate_entry = None
    for frame in gui.param_frame.winfo_children():
        if getattr(frame, "cget", None) and frame.cget("text") == "Environment":
            for w in frame.winfo_children():
                if isinstance(w, type(gui.entry_reward_dist)):
                    text_var = w.cget("textvariable")
                    if text_var == str(gui.var_animation_fps):
                        animation_fps_entry = w
                    if text_var == str(gui.var_update_rate):
                        update_rate_entry = w

    assert animation_fps_entry is not None
    assert update_rate_entry is not None
    assert int(animation_fps_entry.grid_info()["row"]) == 1
    assert int(update_rate_entry.grid_info()["row"]) == 1
    assert int(animation_fps_entry.grid_info()["column"]) == 1
    assert int(update_rate_entry.grid_info()["column"]) == 3


def test_stale_session_events_are_ignored(root) -> None:
    gui = ReacherGUI(root)
    gui.current_session_id = 2

    stale_episode = {
        "type": "episode",
        "session_id": 1,
        "run_id": "run_stale",
        "episode": 1,
        "episodes": 2,
        "reward": 1.0,
        "moving_average": 1.0,
        "steps": 1,
        "epsilon": "n/a",
        "lr": 1e-3,
        "best_reward": 1.0,
        "render_state": "off",
    }
    active_episode = {
        "type": "episode",
        "session_id": 2,
        "run_id": "run_active",
        "episode": 1,
        "episodes": 2,
        "reward": 2.0,
        "moving_average": 2.0,
        "steps": 1,
        "epsilon": "n/a",
        "lr": 1e-3,
        "best_reward": 2.0,
        "render_state": "off",
    }

    gui.event_queue.put(stale_episode)
    gui.event_queue.put(active_episode)
    gui._ui_pump()

    assert "run_stale" not in gui.run_plot_data
    assert "run_active" in gui.run_plot_data
    assert gui.run_plot_data["run_active"]["reward"] == [2.0]


def test_split_episode_aux_updates_without_reward_duplication(root) -> None:
    gui = ReacherGUI(root)
    gui.current_session_id = 3

    episode = {
        "type": "episode",
        "session_id": 3,
        "run_id": "run_split",
        "episode": 1,
        "episodes": 3,
        "reward": 5.0,
        "moving_average": 5.0,
        "steps": 10,
        "epsilon": "n/a",
        "lr": 1e-3,
        "best_reward": 5.0,
        "render_state": "off",
    }
    episode_aux = {
        "type": "episode_aux",
        "session_id": 3,
        "run_id": "run_split",
        "episode": 1,
        "frames": [],
        "eval_points": [(1, 4.5)],
        "eval_score": 4.5,
    }

    gui.event_queue.put(episode)
    gui.event_queue.put(episode_aux)
    gui._ui_pump()

    data = gui.run_plot_data["run_split"]
    assert data["reward"] == [5.0]
    assert data["eval_x"] == [1]
    assert data["eval_y"] == [4.5]
