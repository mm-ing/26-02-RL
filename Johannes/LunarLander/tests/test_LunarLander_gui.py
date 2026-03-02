from LunarLander_gui import LunarLanderGUI
from LunarLander_logic import POLICY_DEFAULTS


class _DummyVar:
    def __init__(self):
        self.value = None

    def set(self, value):
        self.value = value


class _DummyGetVar:
    def __init__(self, value):
        self._value = value

    def get(self):
        return self._value


class _DummyButton:
    def configure(self, **kwargs):
        return None


def test_consume_pending_keeps_run_meta_immutable_for_preview_and_finalize():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)

    gui._compare_mode_active = False
    gui._latest_step = 0
    gui._latest_episode = 0
    gui._current_x = None
    gui._best_x = None
    gui._latest_rewards_snapshot = None
    gui._last_plot_update = 0.0

    gui.steps_progress_var = _DummyVar()
    gui.episodes_progress_var = _DummyVar()
    gui.btn_pause = _DummyButton()

    gui._set_status_text = lambda epsilon, current_x, best_x: None
    gui._update_control_highlights = lambda: None

    captured = {}

    def _capture_update_live_plot(run_meta=None):
        captured["preview_meta"] = run_meta

    def _capture_finalize_run(rewards, meta=None):
        captured["finalize_meta"] = meta
        captured["finalize_rewards"] = list(rewards)

    gui._update_live_plot = _capture_update_live_plot
    gui._finalize_run = _capture_finalize_run

    run_meta = {
        "policy": "PPO",
        "eps_max": 1.0,
        "eps_min": 0.05,
        "learning_rate": 1e-4,
        "lr_strategy": "linear",
        "lr_decay": 0.3,
        "min_learning_rate": 1e-5,
        "moving_avg": 20,
    }

    pending = {
        "step": 10,
        "episode": 2,
        "epsilon": 0.9,
        "rewards_snapshot": [1.0, 2.0, 3.0],
        "run_meta": run_meta,
        "finalize_run": True,
        "finished": True,
    }

    LunarLanderGUI._consume_pending(gui, pending)

    assert captured["preview_meta"] == run_meta
    assert captured["finalize_meta"] == run_meta
    assert captured["finalize_rewards"] == [1.0, 2.0, 3.0]


def test_finalize_compare_run_uses_policy_meta_not_current_ui_values():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)

    gui._run_counter = 0
    gui._plot_runs = []
    gui._compare_run_meta = {
        "PPO": {
            "policy": "PPO",
            "eps_max": 1.0,
            "eps_min": 0.05,
            "learning_rate": 1e-4,
            "lr_strategy": "linear",
            "lr_decay": 0.3,
            "min_learning_rate": 1e-5,
            "moving_avg": 20,
        }
    }

    gui.epsilon_max_var = _DummyGetVar(9.9)
    gui.epsilon_min_var = _DummyGetVar(9.9)
    gui.learning_rate_var = _DummyGetVar("9.90e-01")
    gui.lr_strategy_var = _DummyGetVar("cosine")
    gui.lr_decay_var = _DummyGetVar(0.99)
    gui.min_learning_rate_var = _DummyGetVar("9.90e-01")
    gui.moving_avg_var = _DummyGetVar(3)

    gui._remove_compare_preview_policy = lambda run_key: None
    gui._redraw_plot = lambda *args, **kwargs: None

    expected_base = LunarLanderGUI._build_base_label(gui, "PPO", 1.0, 0.05, 1e-4, "linear", 0.3, 1e-5)

    LunarLanderGUI._finalize_run_for_policy(gui, "PPO", [1.0, 2.0, 3.0])

    assert len(gui._plot_runs) == 1
    assert gui._plot_runs[0]["base"] == expected_base


def test_compare_param_parsing_and_combo_generation():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)
    gui._format_legend_number = lambda value: f"{float(value):.6f}".rstrip("0").rstrip(".")
    gui.compare_param_var = _DummyGetVar("Learning rate")
    gui.compare_values_var = _DummyGetVar("1e-4, 1e-3")
    gui.compare_summary_var = _DummyVar()
    gui._compare_param_lists = {"Policy": ["PPO", "A2C"]}
    gui._compare_raw_lists = {"Policy": "PPO, A2C"}

    snap = {
        "policy": "PPO",
        "max_steps": 100,
        "episodes": 10,
        "epsilon_max": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.05,
        "gamma": 0.99,
        "learning_rate": 1e-4,
        "replay_size": 1000,
        "batch_size": 32,
        "target_update": 10,
        "replay_warmup": 10,
        "learning_cadence": 1,
        "activation_function": "ReLU",
        "hidden_layers": "128",
        "lr_strategy": "linear",
        "lr_decay": 0.3,
        "min_learning_rate": 1e-5,
        "moving_avg": 20,
        "continous": True,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    }

    configs = LunarLanderGUI._build_compare_run_configs(gui, snap)

    assert len(configs) == 4
    policies = sorted({cfg["policy"] for cfg in configs})
    learning_rates = sorted({cfg["learning_rate"] for cfg in configs})
    assert policies == ["A2C", "PPO"]
    assert learning_rates == [1e-4, 1e-3]


def test_compare_policy_uses_policy_defaults_for_non_compared_fields():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)
    gui._format_legend_number = lambda value: f"{float(value):.6f}".rstrip("0").rstrip(".")
    gui.compare_param_var = _DummyGetVar("Policy")
    gui.compare_values_var = _DummyGetVar("PPO, A2C")
    gui.compare_summary_var = _DummyVar()
    gui._compare_param_lists = {}
    gui._compare_raw_lists = {}

    snap = {
        "policy": "PPO",
        "max_steps": 100,
        "episodes": 10,
        "epsilon_max": 1.0,
        "epsilon_decay": 0.99,
        "epsilon_min": 0.05,
        "gamma": 0.99,
        "learning_rate": 1e-4,
        "replay_size": 1000,
        "batch_size": 32,
        "target_update": 10,
        "replay_warmup": 10,
        "learning_cadence": 1,
        "activation_function": "ReLU",
        "hidden_layers": "128",
        "lr_strategy": "linear",
        "lr_decay": 0.3,
        "min_learning_rate": 1e-5,
        "gae_lambda": 0.95,
        "ppo_clip_range": 0.2,
        "moving_avg": 20,
        "continous": True,
        "gravity": -10.0,
        "enable_wind": False,
        "wind_power": 15.0,
        "turbulence_power": 1.5,
    }

    gui.compare_values_var = _DummyGetVar("PPO, A2C")
    gui.compare_param_var = _DummyGetVar("Policy")
    gui._compare_param_lists = {"Policy": ["PPO", "A2C"]}

    configs = LunarLanderGUI._build_compare_run_configs(gui, snap)
    by_policy = {cfg["policy"]: cfg for cfg in configs}

    assert by_policy["PPO"]["lr_strategy"] == POLICY_DEFAULTS["PPO"].lr_strategy
    assert by_policy["A2C"]["lr_strategy"] == POLICY_DEFAULTS["A2C"].lr_strategy
    assert by_policy["A2C"]["learning_cadence"] == POLICY_DEFAULTS["A2C"].learning_cadence
