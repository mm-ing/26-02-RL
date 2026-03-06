import pytest


tk = pytest.importorskip("tkinter")
from tkinter import ttk

from CarRacing_gui import CarRacingGUI
from CarRacing_logic import DEFAULT_SPECIFIC


def test_gui_smoke_startup_and_close():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)
    assert gui.param_vars["Policy"].get() == "SAC"
    assert gui.param_vars["Episodes"].get() == 3000
    gui._on_close()


def test_compare_tab_completion_uses_last_comma_token_and_keeps_caret_at_end():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    gui.param_vars["compare_parameter"].set("Policy")
    gui.param_vars["compare_values"].set("SAC, T")
    gui._on_compare_key_release()
    assert gui.compare_hint.cget("text") == "Tab -> TD3"

    class DummyWidget:
        def __init__(self):
            self.cursor = None

        def icursor(self, value):
            self.cursor = value

    class DummyEvent:
        def __init__(self):
            self.widget = DummyWidget()

    event = DummyEvent()
    response = gui._on_compare_tab_complete(event)
    assert response == "break"
    assert gui.param_vars["compare_values"].get() == "SAC, TD3"
    assert event.widget.cursor == tk.END
    gui._on_close()


def test_compare_toggle_turns_animation_off():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    gui.param_vars["Animation on"].set(True)
    gui.param_vars["compare_on"].set(True)
    gui._on_compare_toggle()
    assert gui.param_vars["Animation on"].get() is False
    gui._on_close()


def test_lr_strategy_and_activation_are_readonly_comboboxes():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    lr_widget = gui.param_widgets["lr_strategy"]
    activation_widget = gui.param_widgets["activation"]

    assert isinstance(lr_widget, ttk.Combobox)
    assert isinstance(activation_widget, ttk.Combobox)
    assert str(lr_widget.cget("state")) == "readonly"
    assert str(activation_widget.cget("state")) == "readonly"

    assert list(lr_widget.cget("values")) == ["constant", "linear", "exponential"]
    assert list(activation_widget.cget("values")) == ["ReLU", "Tanh"]
    gui._on_close()


def test_specific_group_separator_is_below_shared_fields():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    separators = [child for child in gui.specific_grid.winfo_children() if isinstance(child, ttk.Separator)]
    assert len(separators) == 1

    separator = separators[0]
    separator_row = int(separator.grid_info()["row"])
    expected_row = (len(gui.shared_specific_fields) + 1) // 2
    assert separator_row == expected_row
    gui._on_close()


def test_legend_visibility_persists_across_plot_redraws():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    payload = {
        "run_id": "run_1",
        "episode": 1,
        "episodes": 10,
        "reward": 12.0,
        "moving_average": 12.0,
        "eval_points": [(1, 10.0)],
        "steps": 100,
        "epsilon": 0.1,
        "lr": 3e-4,
        "best_reward": 12.0,
        "render_state": "off",
        "frames": [],
        "meta": {
            "policy": "SAC",
            "max_steps": 1000,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "lr_decay": 0.995,
            "epsilon": "-",
            "epsilon_decay": "-",
            "epsilon_min": "-",
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": True,
            "params": {"gamma": 0.99},
        },
    }

    gui._on_episode(payload)
    assert gui.run_visibility["run_1"] is True

    gui._toggle_run_visibility("run_1")
    assert gui.run_visibility["run_1"] is False
    assert gui.run_history["run_1"]["lines"]["reward"].get_visible() is False

    gui._redraw_plot()
    assert gui.run_visibility["run_1"] is False
    assert gui.run_history["run_1"]["lines"]["reward"].get_visible() is False
    gui._on_close()


def test_legend_hover_sets_hand_cursor():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    payload = {
        "run_id": "run_hover",
        "episode": 1,
        "episodes": 10,
        "reward": 8.0,
        "moving_average": 8.0,
        "eval_points": [],
        "steps": 50,
        "epsilon": 0.1,
        "lr": 3e-4,
        "best_reward": 8.0,
        "render_state": "off",
        "frames": [],
        "meta": {
            "policy": "SAC",
            "max_steps": 1000,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "lr_decay": 0.995,
            "epsilon": "-",
            "epsilon_decay": "-",
            "epsilon_min": "-",
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": True,
            "params": {"gamma": 0.99},
        },
    }

    gui._on_episode(payload)
    assert gui.legend is not None

    text_item = gui.legend.get_texts()[0]
    renderer = gui.figure.canvas.get_renderer()
    bbox = text_item.get_window_extent(renderer=renderer)
    x_center = (bbox.x0 + bbox.x1) / 2
    y_center = (bbox.y0 + bbox.y1) / 2

    class DummyHoverEvent:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    gui._on_plot_hover(DummyHoverEvent(x_center, y_center))
    assert gui.plot_canvas.get_tk_widget().cget("cursor") == "hand2"
    gui._on_close()


def test_legend_scroll_center_is_bounded():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    payload = {
        "run_id": "run_scroll",
        "episode": 1,
        "episodes": 10,
        "reward": 5.0,
        "moving_average": 5.0,
        "eval_points": [],
        "steps": 40,
        "epsilon": 0.1,
        "lr": 3e-4,
        "best_reward": 5.0,
        "render_state": "off",
        "frames": [],
        "meta": {
            "policy": "SAC",
            "max_steps": 1000,
            "gamma": 0.99,
            "learning_rate": 3e-4,
            "lr_strategy": "constant",
            "lr_decay": 0.995,
            "epsilon": "-",
            "epsilon_decay": "-",
            "epsilon_min": "-",
            "lap_complete_percent": 0.95,
            "domain_randomize": False,
            "continuous": True,
            "params": {"gamma": 0.99},
        },
    }

    gui._on_episode(payload)
    renderer = gui.figure.canvas.get_renderer()
    bbox = gui.legend.get_window_extent(renderer=renderer)
    x_center = (bbox.x0 + bbox.x1) / 2
    y_center = (bbox.y0 + bbox.y1) / 2

    class DummyScrollEvent:
        def __init__(self, x, y, step):
            self.x = x
            self.y = y
            self.step = step

    for _ in range(200):
        gui._on_plot_scroll(DummyScrollEvent(x_center, y_center, -1))
    assert 0.0 <= gui.legend_scroll_center <= 1.0

    for _ in range(200):
        gui._on_plot_scroll(DummyScrollEvent(x_center, y_center, 1))
    assert 0.0 <= gui.legend_scroll_center <= 1.0
    gui._on_close()


def test_policy_switch_preserves_per_policy_specific_values():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    gui.specific_entries["gamma"].set("0.91")
    gui.policy_dropdown.set("TD3")
    gui._on_policy_changed()
    gui.specific_entries["gamma"].set("0.77")

    gui.policy_dropdown.set("SAC")
    gui._on_policy_changed()
    assert gui.specific_entries["gamma"].get() == "0.91"

    gui.policy_dropdown.set("TD3")
    gui._on_policy_changed()
    assert gui.specific_entries["gamma"].get() == "0.77"
    gui._on_close()


def test_compare_policy_defaults_and_incompatible_keys_are_handled():
    root = tk.Tk()
    root.withdraw()
    gui = CarRacingGUI(root)

    gui.specific_entries["gamma"].set("0.5")
    gui.compare_items = {
        "Policy": ["SAC", "DDQN"],
        "tau": ["0.02"],
        "n_steps": ["2048"],
    }
    configs = gui._expand_compare_configs(session_id="s1")

    assert len(configs) == 2
    by_policy = {cfg.policy_name: cfg for cfg in configs}

    sac_cfg = by_policy["SAC"]
    ddqn_cfg = by_policy["DDQN"]

    assert sac_cfg.params["gamma"] == DEFAULT_SPECIFIC["SAC"]["gamma"]
    assert sac_cfg.params["tau"] == 0.02
    assert "n_steps" not in sac_cfg.params

    assert ddqn_cfg.params["gamma"] == DEFAULT_SPECIFIC["DDQN"]["gamma"]
    assert "tau" not in ddqn_cfg.params
    assert "n_steps" not in ddqn_cfg.params
    gui._on_close()
