import queue
import threading

from LunarLander_gui import LunarLanderGUI, _UiEventType, _WorkerUiEvent
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


class _DummyProgressbar:
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

    gui._set_status_text = lambda epsilon: None
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
    gui.compare_param_var = _DummyGetVar("LR")
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


def test_flush_event_queue_consumes_finalize_instead_of_dropping_it():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)
    gui._event_queue = queue.Queue()

    captured = {}

    def _capture_single(pending):
        captured["single"] = dict(pending)

    def _capture_policy(pending_policy):
        captured["policy"] = dict(pending_policy)

    gui._consume_pending = _capture_single
    gui._consume_policy_pending = _capture_policy

    run_meta = {"policy": "PPO", "moving_avg": 20}
    gui._event_queue.put(
        _WorkerUiEvent(
            channel="single",
            event_type=_UiEventType.FINALIZE.value,
            payload={
                "finalize_run": True,
                "rewards_snapshot": [1.0, 2.0, 3.0],
                "run_meta": run_meta,
                "finished": True,
                "epsilon": 0.5,
            },
        )
    )

    LunarLanderGUI._flush_event_queue(gui)

    assert "single" in captured
    assert captured["single"]["finalize_run"] is True
    assert captured["single"]["rewards_snapshot"] == [1.0, 2.0, 3.0]
    assert captured["single"]["run_meta"] == run_meta
    assert gui._event_queue.empty()


def test_start_worker_calls_flush_event_queue_before_new_run():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)

    called = {"flush": 0}

    gui._flush_event_queue = lambda: called.__setitem__("flush", called["flush"] + 1)
    gui._snapshot_ui = lambda: {
        "policy": "PPO",
        "compare_on": False,
        "epsilon_max": 1.0,
        "episodes": 2,
        "max_steps": 10,
        "animation_on": False,
        "moving_avg": 20,
        "epsilon_min": 0.05,
        "learning_rate": 1e-4,
        "lr_strategy": "linear",
        "lr_decay": 0.3,
        "min_learning_rate": 1e-5,
    }
    gui._enforce_policy_environment_mode = lambda policy, show_info: False
    gui._start_compare_workers = lambda snap: None
    gui._close_aux_policy_trainers = lambda: None
    gui._apply_snapshot_to_trainer = lambda snap: None
    gui._update_control_highlights = lambda: None
    gui._set_status_text = lambda epsilon: None
    gui._worker_loop = lambda snap, single_episode: None

    class _TrainerStub:
        def set_training_plan(self, policy, episodes, max_steps):
            return None

        def reset_policy_agent(self, policy):
            return None

    gui.trainer = _TrainerStub()
    gui._workers = {}
    gui._policy_trainers = {}
    gui._compare_mode_active = False
    gui._compare_selected_run_key = None
    gui._selected_render_trainer = gui.trainer
    gui._latest_eval_snapshot = []
    gui._single_run_meta = {}

    gui._stop_requested = threading.Event()
    gui._pause_requested = threading.Event()
    gui.btn_pause = _DummyButton()
    gui.steps_bar = _DummyProgressbar()
    gui.episodes_bar = _DummyProgressbar()
    gui.steps_progress_var = _DummyVar()
    gui.episodes_progress_var = _DummyVar()

    gui._ui_lag_last_ms = 0.0
    gui._ui_lag_max_ms = 0.0
    gui._last_ui_pump_time = 0.0
    gui._animate_current_episode = False
    gui._best_reward = None
    gui._last_episode_steps = 0

    LunarLanderGUI._start_worker(gui, single_episode=False)
    gui._worker.join(timeout=1.0)

    assert called["flush"] == 1


def test_train_and_run_after_pause_preserves_queued_finalize_plot_run():
    gui = LunarLanderGUI.__new__(LunarLanderGUI)

    class _TrainerStub:
        def set_training_plan(self, policy, episodes, max_steps):
            return None

        def reset_policy_agent(self, policy):
            return None

    gui.trainer = _TrainerStub()
    gui._policy_trainers = {}
    gui._workers = {}
    gui._worker = None

    gui._event_queue = queue.Queue()
    gui._event_queue.put(
        _WorkerUiEvent(
            channel="single",
            event_type=_UiEventType.FINALIZE.value,
            payload={
                "finalize_run": True,
                "rewards_snapshot": [10.0, 20.0],
                "eval_snapshot": [],
                "finished": True,
                "epsilon": 0.5,
                "run_meta": {
                    "policy": "PPO",
                    "eps_max": 1.0,
                    "eps_min": 0.05,
                    "learning_rate": 1e-4,
                    "lr_strategy": "linear",
                    "lr_decay": 0.3,
                    "min_learning_rate": 1e-5,
                    "moving_avg": 20,
                },
            },
        )
    )

    gui._compare_mode_active = False
    gui._compare_selected_run_key = None
    gui._latest_compare_rewards = {}
    gui._latest_compare_eval_snapshots = {}
    gui._compare_finalized_policies = set()
    gui._compare_run_meta = {}
    gui._selected_render_trainer = gui.trainer

    gui._single_run_meta = {}
    gui._plot_runs = []
    gui._run_counter = 0
    gui._latest_rewards_snapshot = None
    gui._latest_eval_snapshot = []
    gui._preview_single_lines = None
    gui._preview_compare_lines = {}

    gui._last_plot_update = 0.0
    gui._ui_lag_last_ms = 0.0
    gui._ui_lag_max_ms = 0.0
    gui._last_ui_pump_time = 0.0

    gui.steps_progress_var = _DummyVar()
    gui.episodes_progress_var = _DummyVar()
    gui.steps_bar = _DummyProgressbar()
    gui.episodes_bar = _DummyProgressbar()
    gui.btn_pause = _DummyButton()
    gui.policy_var = _DummyGetVar("PPO")
    gui.epsilon_max_var = _DummyGetVar(1.0)
    gui.epsilon_min_var = _DummyGetVar(0.05)
    gui.learning_rate_var = _DummyGetVar("1.00e-04")
    gui.lr_strategy_var = _DummyGetVar("linear")
    gui.lr_decay_var = _DummyGetVar(0.3)
    gui.min_learning_rate_var = _DummyGetVar("1.00e-05")
    gui.moving_avg_var = _DummyGetVar(20)

    gui._stop_requested = threading.Event()
    gui._pause_requested = threading.Event()
    gui._pause_requested.set()

    state = {"active": True}
    gui._has_active_workers = lambda: state["active"]

    gui._start_after_worker_stops = lambda single_episode: (state.__setitem__("active", False), LunarLanderGUI._start_worker(gui, single_episode))

    gui._close_aux_policy_trainers = lambda: None
    gui._apply_snapshot_to_trainer = lambda snap: None
    gui._enforce_policy_environment_mode = lambda policy, show_info: False
    gui._update_control_highlights = lambda: None
    gui._set_status_text = lambda epsilon: None
    gui._redraw_plot = lambda *args, **kwargs: None
    gui._update_live_plot = lambda run_meta=None: None
    gui._consume_policy_pending = lambda pending_policy: None
    gui._worker_loop = lambda snap, single_episode: None

    gui._snapshot_ui = lambda: {
        "policy": "PPO",
        "compare_on": False,
        "animation_on": False,
        "episodes": 3,
        "max_steps": 10,
        "epsilon_max": 1.0,
        "epsilon_min": 0.05,
        "learning_rate": 1e-4,
        "lr_strategy": "linear",
        "lr_decay": 0.3,
        "min_learning_rate": 1e-5,
        "moving_avg": 20,
    }

    LunarLanderGUI.train_and_run(gui)
    gui._worker.join(timeout=1.0)

    assert len(gui._plot_runs) == 1
    assert gui._plot_runs[0]["rewards"] == [10.0, 20.0]
