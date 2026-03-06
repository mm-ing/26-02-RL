from __future__ import annotations

import os
import tkinter as tk
import threading
import numpy as np
from pathlib import Path

import pytest
import torch

from Walker2D_gui import Walker2DGUI
from Walker2D_logic import TrainConfig, Walker2DEnvConfig, Walker2DTrainer


class _DummyTrainer:
    def __init__(self):
        self.canceled = False

    def cancel(self):
        self.canceled = True

    def set_pause(self, paused: bool):
        _ = paused


class _OrderTrainer:
    def __init__(self):
        self.calls = []

    def set_pause(self, paused: bool):
        self.calls.append(("set_pause", paused))

    def cancel(self):
        self.calls.append(("cancel", None))


class _TrainStub:
    def __init__(self):
        self.train_called = False
        self.train_config = TrainConfig(device="CPU")

    def train(self):
        self.train_called = True
        return {"ok": True}


@pytest.mark.gui
def test_gui_smoke_startup_and_reset():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)
    app._clear_plot()
    app._reset_all()
    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_legend_interaction_click_and_scroll():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    for idx in range(8):
        run_id = f"run_{idx}"
        app.run_meta_snapshots[run_id] = {
            "policy": "PPO",
            "max_steps": 1000,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "env": {"healthy_reward": 1.0, "reset_noise_scale": 5e-3},
        }
        app._append_plot_data(
            run_id=run_id,
            episode=1,
            reward=float(idx),
            moving_average=float(idx),
            eval_points=[(1, float(idx))],
        )

    app.canvas.draw()
    assert app._legend is not None
    assert len(app._legend_entries) > app._legend_max_visible

    legend_handle, _, artists = app._legend_items[0]
    before_visible = artists[0].get_visible()

    class PickEvent:
        def __init__(self, artist):
            self.artist = artist

    app._on_plot_pick(PickEvent(legend_handle))
    after_visible = artists[0].get_visible()
    assert after_visible is not before_visible

    renderer = app.figure.canvas.get_renderer()
    bbox = app._legend.get_texts()[0].get_window_extent(renderer)
    x_center = (bbox.x0 + bbox.x1) / 2
    y_center = (bbox.y0 + bbox.y1) / 2

    class ScrollEvent:
        def __init__(self, inaxes, x, y, button):
            self.inaxes = inaxes
            self.x = x
            self.y = y
            self.button = button

    old_scroll_index = app._legend_scroll_index
    app._on_plot_scroll(ScrollEvent(app.ax, x_center, y_center, "down"))
    assert app._legend_scroll_index >= old_scroll_index

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_parameter_options_follow_policy_and_general_fields():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    initial_options = list(app.compare_param_combo.cget("values"))
    assert "max_steps" in initial_options
    assert "episodes" in initial_options
    assert "clip_range" in initial_options
    assert "hidden_layer" in initial_options
    assert "lr_strategy" in initial_options
    assert "min_lr" in initial_options
    assert "lr_decay" in initial_options

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    td3_options = list(app.compare_param_combo.cget("values"))
    assert "policy_delay" in td3_options
    assert "target_policy_noise" in td3_options
    assert "clip_range" not in td3_options

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_policy_specific_values_are_isolated_per_policy():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.var_policy.set("SAC")
    app._on_policy_changed(None)
    app.current_specific_vars["tau"].set(0.02)

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    assert app.current_specific_vars["tau"].get() == pytest.approx(0.005)
    app.current_specific_vars["tau"].set(0.03)

    app.var_policy.set("SAC")
    app._on_policy_changed(None)
    assert app.current_specific_vars["tau"].get() == pytest.approx(0.02)

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    assert app.current_specific_vars["tau"].get() == pytest.approx(0.03)

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_shared_specific_group_values_are_isolated_per_policy():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.var_policy.set("PPO")
    app.var_gamma.set(0.95)
    app.var_learning_rate.set("1e-4")
    app.var_batch_size.set(128)
    app.var_hidden_layer.set("384")
    app.var_lr_strategy.set("linear")
    app.var_min_lr.set("1e-6")
    app.var_lr_decay.set(0.97)

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    assert app.var_gamma.get() == pytest.approx(0.99)
    assert float(app.var_learning_rate.get()) == pytest.approx(1e-3)
    assert app.var_batch_size.get() == 256
    assert app.var_hidden_layer.get() == "256"
    assert app.var_lr_strategy.get() == "constant"
    assert float(app.var_min_lr.get()) == pytest.approx(1e-5)
    assert app.var_lr_decay.get() == pytest.approx(1.0)

    app.var_gamma.set(0.90)
    app.var_learning_rate.set("5e-4")
    app.var_batch_size.set(64)
    app.var_hidden_layer.set("128,64")
    app.var_lr_strategy.set("exponential")
    app.var_min_lr.set("2e-6")
    app.var_lr_decay.set(0.99)
    app.var_policy.set("PPO")
    app._on_policy_changed(None)
    assert app.var_gamma.get() == pytest.approx(0.95)
    assert float(app.var_learning_rate.get()) == pytest.approx(1e-4)
    assert app.var_batch_size.get() == 128
    assert app.var_hidden_layer.get() == "384"
    assert app.var_lr_strategy.get() == "linear"
    assert float(app.var_min_lr.get()) == pytest.approx(1e-6)
    assert app.var_lr_decay.get() == pytest.approx(0.97)

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    assert app.var_gamma.get() == pytest.approx(0.90)
    assert float(app.var_learning_rate.get()) == pytest.approx(5e-4)
    assert app.var_batch_size.get() == 64
    assert app.var_hidden_layer.get() == "128,64"
    assert app.var_lr_strategy.get() == "exponential"
    assert float(app.var_min_lr.get()) == pytest.approx(2e-6)
    assert app.var_lr_decay.get() == pytest.approx(0.99)

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_legend_enrichment_avoids_duplicate_base_fields():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    run_id = "cmp_dedup"
    app.run_meta_snapshots[run_id] = {
        "policy": "TD3",
        "max_steps": 1000,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "env": {"healthy_reward": 1.0, "reset_noise_scale": 5e-3},
        "compare": {
            "Policy": "TD3",
            "max_steps": 1200,
            "gamma": 0.95,
            "learning_rate": 1e-4,
            "policy_delay": 3,
        },
    }

    app._append_plot_data(
        run_id=run_id,
        episode=1,
        reward=1.0,
        moving_average=1.0,
        eval_points=[(1, 1.0)],
    )

    label = app.run_plots[run_id].label
    assert "policy_delay=3" in label
    assert "compare:" in label
    assert "compare: Policy=TD3" not in label
    assert "compare: max_steps=1200" not in label
    assert "compare: gamma=0.95" not in label
    assert "compare: learning_rate=0.0001" not in label

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_thread_budgets_distribute_cpu_cores(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    monkeypatch.setattr(os, "cpu_count", lambda: 10)
    budgets = app._compute_compare_thread_budgets(4)
    assert len(budgets) == 4
    assert sum(budgets) == 10
    assert max(budgets) - min(budgets) <= 1

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_train_and_run_while_paused_starts_fresh_run(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    old_trainer = _DummyTrainer()
    app.training_active = True
    app.training_paused = True
    with app.active_trainers_lock:
        app.active_trainers = {"old_run": old_trainer}

    monkeypatch.setattr(threading.Thread, "start", lambda self: None)
    app._start_training()

    assert old_trainer.canceled is True
    assert app.training_active is True
    assert app.training_paused is False
    with app.active_trainers_lock:
        assert len(app.active_trainers) == 1
        assert "old_run" not in app.active_trainers

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_poll_worker_events_ignores_stale_session(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.session_id = "current_session"
    calls = {"episode": 0}

    def count_episode(payload):
        _ = payload
        calls["episode"] += 1

    monkeypatch.setattr(app, "after", lambda delay, callback: None)
    monkeypatch.setattr(app, "_handle_episode_event", count_episode)

    app.event_queue.put({"type": "episode", "session_id": "old_session", "episode": 1, "episodes": 1, "steps": 1, "reward": 0.0, "moving_average": 0.0, "eval_points": []})
    app.event_queue.put({"type": "episode", "session_id": "current_session", "episode": 1, "episodes": 1, "steps": 1, "reward": 0.0, "moving_average": 0.0, "eval_points": []})

    app._poll_worker_events()
    assert calls["episode"] == 1

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_render_selection_and_done_finalization():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.var_policy.set("TD3")
    app.run_meta_snapshots["run_ppo"] = {"policy": "PPO"}
    app.run_meta_snapshots["run_td3"] = {"policy": "TD3"}
    selected = app._select_render_run_id(["run_ppo", "run_td3"])
    assert selected == "run_td3"

    app.training_active = True
    app.compare_active = True
    app.compare_expected_done = 2
    app.compare_done_count = 0
    with app.active_trainers_lock:
        app.active_trainers = {"run_a": _DummyTrainer(), "run_b": _DummyTrainer()}

    app._handle_training_done({"type": "training_done", "run_id": "run_a"})
    assert app.training_active is True
    assert app.compare_done_count == 1

    app._handle_training_done({"type": "training_done", "run_id": "run_b"})
    assert app.training_active is False
    assert app.compare_active is False
    assert app.compare_done_count == 0
    with app.active_trainers_lock:
        assert len(app.active_trainers) == 0

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_close_path_resumes_paused_workers_before_cancel(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    trainer = _OrderTrainer()
    app.training_active = True
    app.training_paused = True
    with app.active_trainers_lock:
        app.active_trainers = {"run": trainer}

    destroyed = {"value": False}

    def fake_destroy():
        destroyed["value"] = True

    monkeypatch.setattr(root, "destroy", fake_destroy)
    app._on_close()

    assert destroyed["value"] is True
    assert trainer.calls == [("set_pause", False), ("cancel", None)]

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_policy_uses_policy_defaults_with_explicit_overrides():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    base_env_cfg = app._make_env_config(render_mode=None)
    base_train_cfg = app._make_train_config(run_id="base")

    app.compare_items = {
        "Policy": ["PPO", "TD3"],
        "n_steps": ["1024"],
        "tau": ["0.02"],
    }

    run_specs = app._build_compare_run_configs(base_env_cfg, base_train_cfg)
    by_policy = {cfg.policy_name: cfg for _, _, cfg in run_specs}

    assert "PPO" in by_policy
    assert "TD3" in by_policy

    ppo_cfg = by_policy["PPO"]
    td3_cfg = by_policy["TD3"]

    assert ppo_cfg.specific_params["n_steps"] == 1024
    assert "tau" not in ppo_cfg.specific_params
    assert "policy_delay" not in ppo_cfg.specific_params

    assert td3_cfg.specific_params["policy_delay"] == 2
    assert td3_cfg.specific_params["target_policy_noise"] == 0.2
    assert td3_cfg.specific_params["tau"] == 0.02
    assert "n_steps" not in td3_cfg.specific_params

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_gpu_selected_without_cuda_uses_cpu_thread_budget(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    trainer = Walker2DTrainer(
        env_config=Walker2DEnvConfig(env_id="Dummy-v0"),
        train_config=TrainConfig(device="GPU"),
    )
    trainer.train = lambda: {"ok": True}

    calls = {"set_threads": []}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls["set_threads"].append(n))

    app._run_trainer_with_thread_budget(trainer, cpu_threads=2)
    assert calls["set_threads"] == [2, 8]

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_legend_visibility_persists_after_plot_redraw():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    run_id = "persist_run"
    app.run_meta_snapshots[run_id] = {
        "policy": "PPO",
        "max_steps": 1000,
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "env": {"healthy_reward": 1.0, "reset_noise_scale": 5e-3},
    }

    app._append_plot_data(run_id, 1, 1.0, 1.0, [(1, 1.0)])
    app.canvas.draw()

    first_handle, _, artists = app._legend_items[0]

    class PickEvent:
        def __init__(self, artist):
            self.artist = artist

    app._on_plot_pick(PickEvent(first_handle))
    assert artists[0].get_visible() is False

    app._append_plot_data(run_id, 2, 1.5, 1.25, [(1, 1.0)])
    app.canvas.draw()

    _, _, artists_after = app._legend_items[0]
    assert artists_after[0].get_visible() is False

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_animation_queue_keeps_latest_pending_payload():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.ones((8, 8, 3), dtype=np.uint8)
    frame_c = np.full((8, 8, 3), 2, dtype=np.uint8)

    app._playback_active = True
    app._enqueue_playback([frame_a])
    assert app._playback_pending is not None
    assert np.array_equal(app._playback_pending[0], frame_a)

    app._enqueue_playback([frame_b])
    assert np.array_equal(app._playback_pending[0], frame_b)

    app._enqueue_playback([frame_c])
    assert np.array_equal(app._playback_pending[0], frame_c)

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_compare_hint_updates_with_parameter_and_prefix():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.var_compare_parameter.set("Policy")
    app.var_compare_values.set("")
    app._update_compare_hint()
    assert app.compare_hint.cget("text") == ""

    app.var_compare_parameter.set("Policy")
    app.var_compare_values.set("t")
    app._update_compare_hint()
    assert app.compare_hint.cget("text") == "Tab -> TD3"

    app.var_compare_parameter.set("gamma")
    app.var_compare_values.set("0")
    app._update_compare_hint()
    assert app.compare_hint.cget("text") == ""

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_controls_use_neutral_style_when_inactive_and_grid_is_enabled():
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    assert app.btn_train.cget("style") == "Neutral.TButton"
    assert app.btn_pause.cget("style") == "Neutral.TButton"

    app.canvas.draw()
    x_grid_visible = any(line.get_visible() for line in app.ax.get_xgridlines())
    y_grid_visible = any(line.get_visible() for line in app.ax.get_ygridlines())
    assert x_grid_visible or y_grid_visible

    root.update_idletasks()
    root.destroy()


@pytest.mark.gui
def test_save_plot_png_filename_includes_core_params(monkeypatch):
    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    saved = {"path": None}
    monkeypatch.setattr(app.figure, "savefig", lambda file_path, dpi=120: saved.update({"path": str(file_path)}))
    monkeypatch.setattr("Walker2D_gui.messagebox.showinfo", lambda *args, **kwargs: None)

    app.var_policy.set("PPO")
    app.var_max_steps.set(1234)
    app.var_gamma.set(0.97)
    app.var_learning_rate.set("1e-4")
    app._save_plot_png()

    path = saved["path"]
    assert path is not None
    name = Path(path).name
    assert "walker2d_PPO" in name
    assert "steps-1234" in name
    assert "gamma-0.97" in name
    assert "lr-1e-04" in name
    assert name.endswith(".png")

    root.update_idletasks()
    root.destroy()
