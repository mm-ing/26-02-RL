from __future__ import annotations

import os
import queue
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import tkinter as tk
from tkinter import ttk, messagebox

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from HalfCheetah_logic import (
    ENV_DEFAULTS,
    EXPOSED_POLICIES,
    GENERAL_DEFAULTS,
    POLICY_DEFAULTS,
    HalfCheetahEnvironment,
    HalfCheetahTrainer,
    build_compare_runs,
    cap_torch_cpu_threads,
)

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


class _Tooltip:
    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)
        self.widget.bind("<ButtonPress>", self._hide)

    def _show(self, _event=None):
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#2d2d30",
            foreground="#e6e6e6",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class HalfCheetahGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("HalfCheetah (SB3)")
        self.base_dir = Path(__file__).resolve().parent

        self.event_queue: "queue.Queue[dict]" = queue.Queue()
        self.current_session_id = ""
        self.current_run_id = ""
        self.render_run_id: str | None = None
        self.worker_thread: threading.Thread | None = None
        self.is_training = False
        self.is_paused = False
        self.current_workers: List[HalfCheetahTrainer] = []
        self._workers_lock = threading.Lock()
        self.latest_transitions_run_id = ""
        self.compare_items: Dict[str, List[Any]] = {}
        self.run_colors: Dict[str, Any] = {}
        self.run_history: Dict[str, Dict[str, Any]] = {}
        self.run_metadata: Dict[str, Dict[str, Any]] = {}
        self.latest_frame = None
        self.latest_image_tk = None
        self.playback_frames: List[Any] = []
        self.pending_playback_frames: List[Any] = []
        self.playback_index = 0
        self.playback_total_frames = 0
        self.playback_after_id: str | None = None
        self._tooltips: List[_Tooltip] = []

        self.trainer = HalfCheetahTrainer(base_dir=str(self.base_dir), event_callback=self._push_event)

        self._configure_style()
        self._create_vars()
        self._build_layout()
        self._apply_policy_defaults()
        self._on_plot_options_toggle()
        self._refresh_compare_hint()
        self._set_control_highlight()
        self._pump_events()

        self.legend_scroll_y = 1.0
        self.legend_scroll_min_y = 1.0
        self.legend_pick_cid = None
        self.legend_hover_cid = None
        self.legend_scroll_cid = None

    def _configure_style(self):
        self.root.configure(bg="#1e1e1e")
        self.root.option_add("*Entry.insertBackground", "#ffffff")
        self.root.option_add("*TEntry*insertBackground", "#ffffff")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            try:
                style.theme_use("vista")
            except Exception:
                pass

        style.configure(".", font=("Segoe UI", 10), background="#1e1e1e", foreground="#e6e6e6")
        style.configure("Group.TLabelframe", background="#252526", foreground="#e6e6e6")
        style.configure("Group.TLabelframe.Label", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10, "bold"))
        style.configure("TFrame", background="#1e1e1e")
        style.configure("Panel.TFrame", background="#252526")
        style.configure("TLabel", background="#252526", foreground="#e6e6e6")
        style.configure("TCheckbutton", background="#252526", foreground="#e6e6e6")
        style.configure("TEntry", fieldbackground="#2d2d30", foreground="#e6e6e6")
        style.configure("TCombobox", fieldbackground="#2d2d30", background="#2d2d30", foreground="#e6e6e6")
        style.map("TCombobox", fieldbackground=[("readonly", "#2d2d30")], selectbackground=[("readonly", "#0e639c")])
        style.configure("Neutral.TButton", font=("Segoe UI", 10, "bold"), background="#3a3d41", foreground="#e6e6e6")
        style.map("Neutral.TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")])
        style.configure("Train.TButton", font=("Segoe UI", 10, "bold"), background="#0e639c", foreground="#ffffff")
        style.map("Train.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.configure("Pause.TButton", font=("Segoe UI", 10, "bold"), background="#a66a00", foreground="#ffffff")
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])
        style.configure("TProgressbar", troughcolor="#343434", background="#0e639c")

    def _create_vars(self):
        self.animation_on_var = tk.BooleanVar(value=True)
        self.animation_fps_var = tk.IntVar(value=30)
        self.update_rate_var = tk.IntVar(value=1)
        self.frame_stride_var = tk.IntVar(value=2)

        self.env_forward_reward_weight_var = tk.DoubleVar(value=ENV_DEFAULTS["forward_reward_weight"])
        self.env_ctrl_cost_weight_var = tk.DoubleVar(value=ENV_DEFAULTS["ctrl_cost_weight"])
        self.env_reset_noise_scale_var = tk.DoubleVar(value=ENV_DEFAULTS["reset_noise_scale"])
        self.env_exclude_positions_var = tk.BooleanVar(value=ENV_DEFAULTS["exclude_current_positions_from_observation"])

        self.compare_on_var = tk.BooleanVar(value=False)
        self.compare_param_var = tk.StringVar(value="policy")
        self.compare_values_var = tk.StringVar(value="")
        self.compare_hint_var = tk.StringVar(value="")

        self.max_steps_var = tk.IntVar(value=GENERAL_DEFAULTS["max_steps"])
        self.episodes_var = tk.IntVar(value=GENERAL_DEFAULTS["episodes"])
        self.epsilon_max_var = tk.DoubleVar(value=GENERAL_DEFAULTS["epsilon_max"])
        self.epsilon_decay_var = tk.DoubleVar(value=GENERAL_DEFAULTS["epsilon_decay"])
        self.epsilon_min_var = tk.DoubleVar(value=GENERAL_DEFAULTS["epsilon_min"])
        self.gamma_var = tk.DoubleVar(value=GENERAL_DEFAULTS["gamma"])

        self.policy_var = tk.StringVar(value="PPO")
        self.hidden_layer_var = tk.IntVar(value=256)
        self.activation_var = tk.StringVar(value="Tanh")
        self.lr_var = tk.StringVar(value="3e-4")
        self.lr_strategy_var = tk.StringVar(value="constant")
        self.min_lr_var = tk.StringVar(value="1e-5")
        self.lr_decay_var = tk.DoubleVar(value=1.0)
        self.replay_size_var = tk.IntVar(value=100000)
        self.batch_size_var = tk.IntVar(value=64)
        self.n_steps_var = tk.IntVar(value=int(POLICY_DEFAULTS["PPO"].get("n_steps", 2048)))
        self.learning_start_var = tk.IntVar(value=0)
        self.learning_frequency_var = tk.IntVar(value=1)
        self.target_update_var = tk.IntVar(value=1)

        self.moving_average_var = tk.IntVar(value=20)
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.rollout_capture_var = tk.IntVar(value=120)
        self.low_overhead_var = tk.BooleanVar(value=False)

        self.device_var = tk.StringVar(value="CPU")

        self.steps_progress_var = tk.DoubleVar(value=0)
        self.episodes_progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="Epsilon: - | LR: - | Best reward: - | Render: idle")

    def _build_layout(self):
        outer = ttk.Frame(self.root, style="TFrame", padding=10)
        outer.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        outer.columnconfigure(0, weight=2)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=3)
        outer.rowconfigure(1, weight=0)
        outer.rowconfigure(2, weight=0)
        outer.rowconfigure(3, weight=2)

        self._build_environment_panel(outer)
        self._build_parameters_panel(outer)
        self._build_controls_row(outer)
        self._build_current_run_panel(outer)
        self._build_plot_panel(outer)

    def _build_environment_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Environment", style="Group.TLabelframe", padding=6)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(frame, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew")
        self.render_canvas.bind("<Configure>", lambda _e: self._render_latest_frame())

    def _build_parameters_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="Parameters", style="Group.TLabelframe", padding=0)
        panel.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        container = ttk.Frame(panel, style="Panel.TFrame")
        container.grid(row=0, column=0, sticky="nsew")
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

        self.param_canvas = tk.Canvas(container, background="#252526", highlightthickness=0)
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        self.param_scroll = ttk.Scrollbar(container, orient="vertical", command=self.param_canvas.yview)
        self.param_canvas.configure(yscrollcommand=self._on_scrollset)
        self.param_scroll.grid(row=0, column=1, sticky="ns")

        self.params_inner = ttk.Frame(self.param_canvas, style="Panel.TFrame")
        self.params_window = self.param_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        self.params_inner.bind("<Configure>", self._on_params_configure)
        self.param_canvas.bind("<Configure>", self._on_canvas_configure)
        self.param_canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.param_canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        self._build_environment_group(self.params_inner)
        self._build_compare_group(self.params_inner)
        self._build_general_group(self.params_inner)
        self._build_specific_group(self.params_inner)
        self._build_live_plot_group(self.params_inner)

    def _build_group(self, parent, title: str):
        group = ttk.LabelFrame(parent, text=title, style="Group.TLabelframe", padding=6)
        group.pack(fill="x", padx=6, pady=(6, 0))
        return group

    def _build_environment_group(self, parent):
        group = self._build_group(parent, "Environment")
        group.columnconfigure(1, weight=1)
        group.columnconfigure(3, weight=1)

        self._add_pair_row(
            group,
            0,
            "Animation on",
            ttk.Checkbutton(group, variable=self.animation_on_var, command=self._on_animation_toggle),
            "Animation FPS",
            ttk.Entry(group, textvariable=self.animation_fps_var, width=9),
        )
        self._add_pair_row(
            group,
            1,
            "Update rate (episodes)",
            ttk.Entry(group, textvariable=self.update_rate_var, width=9),
            "Frame stride",
            ttk.Entry(group, textvariable=self.frame_stride_var, width=9),
        )

        btn = ttk.Button(group, text="Update", style="Neutral.TButton", command=self._update_environment_only)
        btn.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(2, 4))

        self._add_pair_row(
            group,
            3,
            "forward_reward_weight",
            ttk.Entry(group, textvariable=self.env_forward_reward_weight_var, width=9),
            "ctrl_cost_weight",
            ttk.Entry(group, textvariable=self.env_ctrl_cost_weight_var, width=9),
        )
        self._add_pair_row(
            group,
            4,
            "reset_noise_scale",
            ttk.Entry(group, textvariable=self.env_reset_noise_scale_var, width=9),
            "exclude_current_positions",
            ttk.Checkbutton(group, variable=self.env_exclude_positions_var),
        )

    def _build_compare_group(self, parent):
        group = self._build_group(parent, "Compare")
        group.columnconfigure(0, weight=1)
        group.columnconfigure(1, weight=1)

        top = ttk.Frame(group, style="Panel.TFrame")
        top.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)

        ttk.Checkbutton(top, text="Compare on", variable=self.compare_on_var, command=self._on_compare_toggle).grid(row=0, column=0, sticky="w")
        btns = ttk.Frame(top, style="Panel.TFrame")
        btns.grid(row=0, column=1, sticky="e")
        ttk.Button(btns, text="Clear", style="Neutral.TButton", command=self._compare_clear).pack(side="left", padx=(0, 4))
        ttk.Button(btns, text="Add", style="Neutral.TButton", command=self._compare_add).pack(side="left")

        param_options = self._compare_param_options()
        self.compare_param_combo = ttk.Combobox(group, textvariable=self.compare_param_var, values=param_options, state="readonly", width=16)
        self.compare_param_combo.grid(row=1, column=0, sticky="ew", padx=(0, 4))
        self._disable_combobox_mousewheel(self.compare_param_combo)

        self.compare_values_entry = ttk.Entry(group, textvariable=self.compare_values_var)
        self.compare_values_entry.grid(row=1, column=1, sticky="ew")
        self.compare_values_entry.bind("<Return>", lambda _e: self._compare_add())
        self.compare_values_entry.bind("<Tab>", self._complete_compare_value)
        self.compare_values_var.trace_add("write", lambda *_: self._refresh_compare_hint())

        ttk.Label(group, textvariable=self.compare_hint_var, style="TLabel").grid(row=2, column=1, sticky="w")

        self.compare_summary = tk.Text(group, height=3, bg="#2d2d30", fg="#e6e6e6", insertbackground="#e6e6e6", relief="flat")
        self.compare_summary.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        self.compare_summary.configure(state="disabled")

    def _build_general_group(self, parent):
        group = self._build_group(parent, "General")
        group.columnconfigure(1, weight=1)
        group.columnconfigure(3, weight=1)

        self._add_pair_row(
            group,
            0,
            "Max steps",
            ttk.Entry(group, textvariable=self.max_steps_var, width=9),
            "Episodes",
            ttk.Entry(group, textvariable=self.episodes_var, width=9),
        )

    def _build_specific_group(self, parent):
        group = self._build_group(parent, "Specific")
        group.columnconfigure(1, weight=1)
        group.columnconfigure(3, weight=1)
        self.specific_group = group

        policy_label = ttk.Label(group, text="Policy")
        policy_label.grid(row=0, column=0, sticky="w", pady=2)
        policy = ttk.Combobox(group, textvariable=self.policy_var, values=list(EXPOSED_POLICIES), state="readonly", width=18)
        policy.grid(row=0, column=1, columnspan=3, sticky="ew", pady=2)
        policy.bind("<<ComboboxSelected>>", lambda _e: self._on_policy_changed())
        self._disable_combobox_mousewheel(policy)

        hidden_layer_entry = ttk.Entry(group, textvariable=self.hidden_layer_var, width=9)
        activation_combo = ttk.Combobox(group, textvariable=self.activation_var, values=["ReLU", "Tanh", "ELU", "GELU"], state="readonly", width=9)
        self._disable_combobox_mousewheel(activation_combo)
        self._add_pair_row(group, 1, "Hidden layer", hidden_layer_entry, "Activation", activation_combo)

        gamma_entry = ttk.Entry(group, textvariable=self.gamma_var, width=9)
        batch_size_entry = ttk.Entry(group, textvariable=self.batch_size_var, width=9)
        self._add_pair_row(group, 2, "Gamma", gamma_entry, "Batch size", batch_size_entry)

        lr_entry = ttk.Entry(group, textvariable=self.lr_var, width=9)
        lr_strategy_combo = ttk.Combobox(group, textvariable=self.lr_strategy_var, values=["constant", "linear", "exponential"], state="readonly", width=9)
        self._disable_combobox_mousewheel(lr_strategy_combo)
        self._add_pair_row(group, 3, "LR", lr_entry, "LR strategy", lr_strategy_combo)

        min_lr_entry = ttk.Entry(group, textvariable=self.min_lr_var, width=9)
        lr_decay_entry = ttk.Entry(group, textvariable=self.lr_decay_var, width=9)
        self._add_pair_row(group, 4, "Min LR", min_lr_entry, "LR decay", lr_decay_entry)

        self.policy_specific_separator = ttk.Separator(group, orient="horizontal")
        self.policy_specific_separator.grid(row=5, column=0, columnspan=4, sticky="ew", pady=(6, 4))

        self.ppo_n_steps_label = ttk.Label(group, text="n_steps")
        self.ppo_n_steps_entry = ttk.Entry(group, textvariable=self.n_steps_var, width=9)
        self.ppo_empty_label = ttk.Label(group, text="")
        self.ppo_empty_value = ttk.Label(group, text="")

        self.off_replay_size_label = ttk.Label(group, text="Replay size")
        self.off_replay_size_entry = ttk.Entry(group, textvariable=self.replay_size_var, width=9)
        self.off_learning_start_label = ttk.Label(group, text="Learning start")
        self.off_learning_start_entry = ttk.Entry(group, textvariable=self.learning_start_var, width=9)

        self.off_learning_frequency_label = ttk.Label(group, text="Learning frequency")
        self.off_learning_frequency_entry = ttk.Entry(group, textvariable=self.learning_frequency_var, width=9)
        self.off_empty_label = ttk.Label(group, text="")
        self.off_empty_value = ttk.Label(group, text="")

        self._add_tooltip(policy, "Selects the algorithm family; this switches optimizer dynamics and which policy-specific controls are active.")
        self._add_tooltip(hidden_layer_entry, "Sets network width. Larger values increase capacity and compute cost.")
        self._add_tooltip(activation_combo, "Changes network nonlinearity, affecting optimization smoothness and representation shape.")
        self._add_tooltip(gamma_entry, "Discount factor for future rewards. Higher values favor long-term return.")
        self._add_tooltip(batch_size_entry, "Samples per gradient step. Larger batches reduce gradient noise but are slower.")
        self._add_tooltip(lr_entry, "Base learning rate. Higher values learn faster but can destabilize training.")
        self._add_tooltip(lr_strategy_combo, "Schedules how learning rate changes over training (constant/linear/exponential).")
        self._add_tooltip(min_lr_entry, "Lower bound for scheduled learning rate to avoid vanishing update steps.")
        self._add_tooltip(lr_decay_entry, "Decay strength for exponential LR schedule; lower values reduce LR faster.")
        self._add_tooltip(self.ppo_n_steps_entry, "PPO rollout length before an update. Larger values improve estimate quality but delay updates.")
        self._add_tooltip(self.off_replay_size_entry, "Replay buffer capacity. Larger buffers improve diversity but increase memory use.")
        self._add_tooltip(self.off_learning_start_entry, "Number of steps collected before updates begin. Higher values improve early stability.")
        self._add_tooltip(self.off_learning_frequency_entry, "How often gradient updates run. Higher frequency increases learning speed and compute load.")

        self._set_policy_specific_visibility()

    def _set_policy_specific_visibility(self):
        is_ppo = str(self.policy_var.get()) == "PPO"

        self.ppo_n_steps_label.grid_remove()
        self.ppo_n_steps_entry.grid_remove()
        self.ppo_empty_label.grid_remove()
        self.ppo_empty_value.grid_remove()
        self.off_replay_size_label.grid_remove()
        self.off_replay_size_entry.grid_remove()
        self.off_learning_start_label.grid_remove()
        self.off_learning_start_entry.grid_remove()
        self.off_learning_frequency_label.grid_remove()
        self.off_learning_frequency_entry.grid_remove()
        self.off_empty_label.grid_remove()
        self.off_empty_value.grid_remove()

        if is_ppo:
            self.ppo_n_steps_label.grid(row=6, column=0, sticky="w", pady=2)
            self.ppo_n_steps_entry.grid(row=6, column=1, sticky="ew", pady=2, padx=(0, 6))
            self.ppo_empty_label.grid(row=6, column=2, sticky="w", pady=2)
            self.ppo_empty_value.grid(row=6, column=3, sticky="ew", pady=2)
            return

        self.off_replay_size_label.grid(row=6, column=0, sticky="w", pady=2)
        self.off_replay_size_entry.grid(row=6, column=1, sticky="ew", pady=2, padx=(0, 6))
        self.off_learning_start_label.grid(row=6, column=2, sticky="w", pady=2)
        self.off_learning_start_entry.grid(row=6, column=3, sticky="ew", pady=2)
        self.off_learning_frequency_label.grid(row=7, column=0, sticky="w", pady=2)
        self.off_learning_frequency_entry.grid(row=7, column=1, sticky="ew", pady=2, padx=(0, 6))
        self.off_empty_label.grid(row=7, column=2, sticky="w", pady=2)
        self.off_empty_value.grid(row=7, column=3, sticky="ew", pady=2)

    def _build_live_plot_group(self, parent):
        group = self._build_group(parent, "Live Plot")
        self._add_row(group, "Moving average values", ttk.Entry(group, textvariable=self.moving_average_var, width=9))
        self._add_row(group, "Show Advanced", ttk.Checkbutton(group, variable=self.show_advanced_var, command=self._on_plot_options_toggle))

        self.advanced_frame = ttk.Frame(group, style="Panel.TFrame")
        self.advanced_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self._add_row(self.advanced_frame, "Rollout full-capture steps", ttk.Entry(self.advanced_frame, textvariable=self.rollout_capture_var, width=9), row=0)
        self._add_row(self.advanced_frame, "Low-overhead animation", ttk.Checkbutton(self.advanced_frame, variable=self.low_overhead_var), row=1)

    def _build_controls_row(self, parent):
        frame = ttk.LabelFrame(parent, text="Controls", style="Group.TLabelframe", padding=6)
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for col in range(8):
            frame.columnconfigure(col, weight=1)

        self.btn_run_episode = ttk.Button(frame, text="Run single episode", style="Neutral.TButton", command=self.run_single_episode)
        self.btn_run_episode.grid(row=0, column=0, sticky="ew", padx=2)
        self.btn_train = ttk.Button(frame, text="Train and Run", style="Neutral.TButton", command=self.train_and_run)
        self.btn_train.grid(row=0, column=1, sticky="ew", padx=2)
        self.btn_pause = ttk.Button(frame, text="Pause", style="Neutral.TButton", command=self.pause_or_resume)
        self.btn_pause.grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(frame, text="Reset All", style="Neutral.TButton", command=self.reset_all).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(frame, text="Clear Plot", style="Neutral.TButton", command=self.clear_plot).grid(row=0, column=4, sticky="ew", padx=2)
        ttk.Button(frame, text="Save samplings CSV", style="Neutral.TButton", command=self.save_samplings_csv).grid(row=0, column=5, sticky="ew", padx=2)
        ttk.Button(frame, text="Save Plot PNG", style="Neutral.TButton", command=self.save_plot_png).grid(row=0, column=6, sticky="ew", padx=2)
        device_combo = ttk.Combobox(frame, textvariable=self.device_var, values=["CPU", "GPU"], state="readonly", width=8)
        device_combo.grid(row=0, column=7, sticky="ew", padx=2)
        self._disable_combobox_mousewheel(device_combo)

    def _build_current_run_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Current Run", style="Group.TLabelframe", padding=6)
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Steps").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.steps_bar = ttk.Progressbar(frame, maximum=100, variable=self.steps_progress_var)
        self.steps_bar.grid(row=0, column=1, sticky="ew")

        ttk.Label(frame, text="Episodes").grid(row=1, column=0, sticky="w", padx=(0, 6))
        self.episodes_bar = ttk.Progressbar(frame, maximum=100, variable=self.episodes_progress_var)
        self.episodes_bar.grid(row=1, column=1, sticky="ew")

        ttk.Label(frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

    def _build_plot_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Live Plot", style="Group.TLabelframe", padding=6)
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew")
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.fig = Figure(figsize=(10, 4), facecolor="#1e1e1e")
        self.ax = self.fig.add_subplot(111)
        self.fig.subplots_adjust(left=0.04, right=0.78, bottom=0.13, top=0.95)
        self.ax.set_xlabel("Episodes", color="#d0d0d0")
        self.ax.set_ylabel("Reward", color="#d0d0d0")
        self.ax.set_facecolor("#252526")
        self.ax.grid(True, alpha=0.2)
        for spine in self.ax.spines.values():
            spine.set_color("#d0d0d0")
        self.ax.tick_params(colors="#d0d0d0")

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.legend_obj = None
        self._legend_artists = {}

    def _add_row(self, parent, label: str, widget, row: int | None = None):
        if row is None:
            row = parent.grid_size()[1]
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        widget.grid(row=row, column=1, sticky="ew", pady=2)
        parent.columnconfigure(1, weight=1)

    def _add_pair_row(self, parent, row, l1, w1, l2, w2):
        ttk.Label(parent, text=l1).grid(row=row, column=0, sticky="w", pady=2)
        w1.grid(row=row, column=1, sticky="ew", pady=2, padx=(0, 6))
        ttk.Label(parent, text=l2).grid(row=row, column=2, sticky="w", pady=2)
        w2.grid(row=row, column=3, sticky="ew", pady=2)

    def _on_params_configure(self, _event):
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        self.param_canvas.itemconfigure(self.params_window, width=event.width)

    def _bind_mousewheel(self):
        self.param_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self.param_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_scrollset(self, first, last):
        if float(first) <= 0.0 and float(last) >= 1.0:
            self.param_scroll.grid_remove()
        else:
            self.param_scroll.grid()
        self.param_scroll.set(first, last)

    def _on_plot_options_toggle(self):
        if self.show_advanced_var.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def _on_compare_toggle(self):
        if self.compare_on_var.get():
            self.animation_on_var.set(False)
            self._on_animation_toggle()

    def _on_animation_toggle(self):
        animation_enabled = bool(self.animation_on_var.get())
        for worker in self._active_workers():
            worker.set_animation_enabled(animation_enabled)
        if not animation_enabled:
            self._stop_frame_playback()

    def _compare_param_options(self):
        general = ["max_steps", "episodes", "gamma", "device"]
        policy_specific = ["n_steps"] if str(self.policy_var.get()) == "PPO" else ["replay_size", "learning_start", "learning_frequency"]
        specific = ["policy", "hidden_layer", "activation", "batch_size", "lr", "lr_strategy", "min_lr", "lr_decay"] + policy_specific
        return general + specific

    def _refresh_compare_param_options(self):
        options = self._compare_param_options()
        self.compare_param_combo.configure(values=options)
        if self.compare_param_var.get() not in options:
            self.compare_param_var.set(options[0])
            self._refresh_compare_hint()

    def _refresh_compare_hint(self):
        param = self.compare_param_var.get()
        text = self.compare_values_var.get().split(",")[-1].strip()
        options_map = {
            "policy": list(EXPOSED_POLICIES),
            "activation": ["ReLU", "Tanh", "ELU", "GELU"],
            "lr_strategy": ["constant", "linear", "exponential"],
            "device": ["CPU", "GPU"],
        }
        opts = options_map.get(param, [])
        suggestion = next((o for o in opts if o.lower().startswith(text.lower()) and text), "")
        self.compare_hint_var.set(f"Tab -> {suggestion}" if suggestion else "")

    def _complete_compare_value(self, event):
        param = self.compare_param_var.get()
        options_map = {
            "policy": list(EXPOSED_POLICIES),
            "activation": ["ReLU", "Tanh", "ELU", "GELU"],
            "lr_strategy": ["constant", "linear", "exponential"],
            "device": ["CPU", "GPU"],
        }
        opts = options_map.get(param, [])
        parts = [p.strip() for p in self.compare_values_var.get().split(",")]
        prefix = parts[-1] if parts else ""
        suggestion = next((o for o in opts if o.lower().startswith(prefix.lower()) and prefix), None)
        if suggestion:
            parts[-1] = suggestion
            self.compare_values_var.set(", ".join([p for p in parts if p]))
            self.compare_values_entry.icursor("end")
        return "break"

    def _compare_add(self):
        key = self.compare_param_var.get().strip()
        raw_values = [v.strip() for v in self.compare_values_var.get().split(",") if v.strip()]
        if not key or not raw_values:
            return
        converted = [self._coerce_value(key, v) for v in raw_values]
        self.compare_items[key] = converted
        self.compare_values_var.set("")
        self._refresh_compare_summary()

    def _compare_clear(self):
        self.compare_items.clear()
        self._refresh_compare_summary()

    def _refresh_compare_summary(self):
        self.compare_summary.configure(state="normal")
        self.compare_summary.delete("1.0", "end")
        for key, values in self.compare_items.items():
            self.compare_summary.insert("end", f"{key}: {values}\n")
        self.compare_summary.configure(state="disabled")

    def _coerce_value(self, key: str, value: str):
        if key in {"max_steps", "episodes", "hidden_layer", "batch_size", "n_steps", "replay_size", "learning_start", "learning_frequency"}:
            return int(float(value))
        if key in {"gamma", "lr", "min_lr", "lr_decay"}:
            return float(value)
        return value

    def _apply_policy_defaults(self):
        defaults = dict(POLICY_DEFAULTS.get(self.policy_var.get(), POLICY_DEFAULTS["PPO"]))
        self.hidden_layer_var.set(int(defaults["hidden_layer"]))
        self.activation_var.set(str(defaults["activation"]))
        self.n_steps_var.set(int(defaults.get("n_steps", POLICY_DEFAULTS["PPO"].get("n_steps", 2048))))
        self.lr_var.set(f"{defaults['lr']:.2e}".replace("e-0", "e-").replace("e+0", "e+"))
        self.lr_strategy_var.set(str(defaults["lr_strategy"]))
        self.min_lr_var.set(f"{defaults['min_lr']:.2e}".replace("e-0", "e-").replace("e+0", "e+"))
        self.lr_decay_var.set(float(defaults["lr_decay"]))
        self.replay_size_var.set(int(defaults["replay_size"]))
        self.batch_size_var.set(int(defaults["batch_size"]))
        self.learning_start_var.set(int(defaults["learning_start"]))
        self.learning_frequency_var.set(int(defaults["learning_frequency"]))
        self.target_update_var.set(int(defaults["target_update"]))
        self._set_policy_specific_visibility()
        if hasattr(self, "compare_param_combo"):
            self._refresh_compare_param_options()

    def _on_policy_changed(self):
        self._set_policy_specific_visibility()
        self._refresh_compare_param_options()
        self._refresh_compare_hint()

    def _disable_combobox_mousewheel(self, combo: ttk.Combobox):
        combo.bind("<MouseWheel>", lambda _e: "break")
        combo.bind("<Button-4>", lambda _e: "break")
        combo.bind("<Button-5>", lambda _e: "break")

    def _add_tooltip(self, widget, text: str):
        self._tooltips.append(_Tooltip(widget, text))

    def _set_control_highlight(self):
        if self.is_paused:
            self.btn_train.configure(style="Neutral.TButton")
            self.btn_pause.configure(style="Pause.TButton", text="Run")
        elif self.is_training:
            self.btn_train.configure(style="Train.TButton")
            self.btn_pause.configure(style="Neutral.TButton", text="Pause")
        else:
            self.btn_train.configure(style="Neutral.TButton")
            self.btn_pause.configure(style="Neutral.TButton", text="Pause")

    def _push_event(self, event: Dict[str, Any]):
        self.event_queue.put(event)

    def _flush_event_queue(self):
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def _snapshot_config(self) -> Dict[str, Any]:
        env_params = {
            "forward_reward_weight": float(self.env_forward_reward_weight_var.get()),
            "ctrl_cost_weight": float(self.env_ctrl_cost_weight_var.get()),
            "reset_noise_scale": float(self.env_reset_noise_scale_var.get()),
            "exclude_current_positions_from_observation": bool(self.env_exclude_positions_var.get()),
        }

        policy = str(self.policy_var.get())
        specific = {
            "hidden_layer": int(self.hidden_layer_var.get()),
            "activation": str(self.activation_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "lr": float(self.lr_var.get()),
            "lr_strategy": str(self.lr_strategy_var.get()),
            "min_lr": float(self.min_lr_var.get()),
            "lr_decay": float(self.lr_decay_var.get()),
        }

        if policy == "PPO":
            specific["n_steps"] = int(self.n_steps_var.get())
        else:
            specific["replay_size"] = int(self.replay_size_var.get())
            specific["learning_start"] = int(self.learning_start_var.get())
            specific["learning_frequency"] = int(self.learning_frequency_var.get())

        return {
            "policy": policy,
            "device": str(self.device_var.get()),
            "episodes": int(self.episodes_var.get()),
            "max_steps": int(self.max_steps_var.get()),
            "gamma": float(self.gamma_var.get()),
            "epsilon_max": float(self.epsilon_max_var.get()),
            "epsilon_decay": float(self.epsilon_decay_var.get()),
            "epsilon_min": float(self.epsilon_min_var.get()),
            "moving_average_values": int(self.moving_average_var.get()),
            "update_rate_episodes": int(self.update_rate_var.get()),
            "frame_capture_stride": int(self.frame_stride_var.get()),
            "animation_on": bool(self.animation_on_var.get()),
            "rollout_full_capture_steps": int(self.rollout_capture_var.get()),
            "low_overhead_animation": bool(self.low_overhead_var.get()),
            "eval_interval": 10,
            "eval_episodes": 1,
            "run_id": str(uuid.uuid4()),
            "env_params": env_params,
            "specific_params": specific,
        }

    def _build_run_metadata(self, cfg: Dict[str, Any]) -> Dict[str, Any]:
        env = dict(cfg.get("env_params", {}))
        specific = dict(cfg.get("specific_params", {}))
        compare_values: Dict[str, Any] = {}
        compare_keys = list(cfg.get("compare_keys", []))
        for key in compare_keys:
            if key in cfg:
                compare_values[key] = cfg.get(key)
            elif key in specific:
                compare_values[key] = specific.get(key)
            elif key in env:
                compare_values[key] = env.get(key)
        return {
            "policy": str(cfg.get("policy", "?")),
            "max_steps": int(cfg.get("max_steps", 0)),
            "gamma": float(cfg.get("gamma", 0.0)),
            "epsilon_max": float(cfg.get("epsilon_max", 0.0)),
            "epsilon_decay": float(cfg.get("epsilon_decay", 0.0)),
            "epsilon_min": float(cfg.get("epsilon_min", 0.0)),
            "lr": specific.get("lr", "?"),
            "lr_strategy": specific.get("lr_strategy", "?"),
            "lr_decay": specific.get("lr_decay", "?"),
            "env": env,
            "compare_values": compare_values,
        }

    def _update_environment_only(self):
        try:
            self.trainer.rebuild_environment(env_params=self._snapshot_config()["env_params"], render_mode="rgb_array")
            self.status_var.set("Epsilon: - | LR: - | Best reward: - | Render: idle")
        except Exception as exc:
            messagebox.showerror("Environment update failed", str(exc))

    def run_single_episode(self):
        cfg = self._snapshot_config()
        self.steps_progress_var.set(0)
        try:
            tmp_trainer = HalfCheetahTrainer(base_dir=str(self.base_dir), event_callback=None)
            env_wrap = HalfCheetahEnvironment(env_params=cfg["env_params"], render_mode="rgb_array")
            vec_env = env_wrap.make_vectorized()
            from HalfCheetah_logic import SB3PolicyAgent

            model = SB3PolicyAgent(cfg["policy"], device=cfg["device"]).create_model(vec_env, cfg["gamma"], cfg["specific_params"])
            result = tmp_trainer.run_episode(
                model=model,
                env_wrapper=env_wrap,
                max_steps=cfg["max_steps"],
                deterministic=True,
                collect_transitions=True,
                animation_on=bool(cfg["animation_on"]),
                frame_capture_stride=max(1, int(cfg["frame_capture_stride"])),
                rollout_full_capture_steps=int(cfg["rollout_full_capture_steps"]),
                low_overhead_animation=bool(cfg["low_overhead_animation"]),
            )
            frames = list(result.get("frames", []))
            if frames:
                self._enqueue_or_start_playback(frames)
            elif result.get("frame") is not None:
                self.latest_frame = result["frame"]
                self._render_latest_frame()
            self.status_var.set(f"Epsilon: {cfg['epsilon_max']:.4f} | LR: {cfg['specific_params']['lr']:.3e} | Best reward: {result['reward']:.3f} | Render: {result['render_state']}")
            self.trainer.transitions = tmp_trainer.transitions
            self.latest_transitions_run_id = cfg["run_id"]
            env_wrap.close()
            vec_env.close()
        except Exception as exc:
            messagebox.showerror("Run failed", str(exc))

    def train_and_run(self):
        if self.is_paused:
            self._cancel_current_run()
        elif self.is_training:
            return

        cfg = self._snapshot_config()
        self.current_session_id = str(uuid.uuid4())
        self.current_run_id = cfg["run_id"]
        self.render_run_id = cfg["run_id"] if bool(cfg.get("animation_on", False)) else None
        with self._workers_lock:
            self.current_workers = []
        self._stop_frame_playback()
        self._flush_event_queue()
        self.is_training = True
        self.is_paused = False
        self._set_control_highlight()

        self.worker_thread = threading.Thread(target=self._worker_train, args=(self.current_session_id, cfg), daemon=True)
        self.worker_thread.start()

    def _worker_train(self, session_id: str, cfg: Dict[str, Any]):
        try:
            if self.compare_on_var.get() and self.compare_items:
                runs = build_compare_runs(cfg, self.compare_items)
                compare_keys = list(self.compare_items.keys())
                if runs:
                    selected_policy = str(self.policy_var.get())
                    selected = next((r for r in runs if str(r.get("policy")) == selected_policy), None)
                    self.render_run_id = str((selected or runs[0]).get("run_id"))
                else:
                    self.render_run_id = None

                workers = min(4, len(runs))
                if workers > 1:
                    cpu_count = max(1, int(os.cpu_count() or workers))
                    threads_per_worker = max(1, cpu_count // workers)
                    cap_torch_cpu_threads(max(1, min(8, threads_per_worker)))

                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futures = []
                    for run_cfg in runs:
                        run_cfg_local = dict(run_cfg)
                        run_cfg_local["compare_keys"] = list(compare_keys)
                        run_cfg_local["animation_on"] = bool(run_cfg_local.get("run_id") == self.render_run_id)
                        run_cfg_local["rollout_full_capture_steps"] = int(cfg.get("rollout_full_capture_steps", 120))
                        run_cfg_local["low_overhead_animation"] = bool(cfg.get("low_overhead_animation", False))
                        self.run_metadata[str(run_cfg_local.get("run_id"))] = self._build_run_metadata(run_cfg_local)

                        trainer = HalfCheetahTrainer(base_dir=str(self.base_dir), event_callback=self._make_session_event_sink(session_id))
                        self._register_worker(trainer)
                        futures.append(pool.submit(self._execute_single_run, trainer, run_cfg_local, True))
                    for fut in as_completed(futures):
                        fut.result()
            else:
                cfg_single = dict(cfg)
                cfg_single["animation_on"] = True
                self.run_metadata[str(cfg_single.get("run_id"))] = self._build_run_metadata(cfg_single)
                trainer = HalfCheetahTrainer(base_dir=str(self.base_dir), event_callback=self._make_session_event_sink(session_id))
                self._register_worker(trainer)
                self._execute_single_run(trainer, cfg_single, False)
        except Exception as exc:
            self._push_event({"type": "error", "run_id": self.current_run_id, "session_id": session_id, "message": str(exc)})
        finally:
            self._push_event({"type": "worker_done", "session_id": session_id})

    def _make_session_event_sink(self, session_id: str):
        def _session_event_sink(event: Dict[str, Any]):
            event_with_session = dict(event)
            event_with_session["session_id"] = session_id
            self._push_event(event_with_session)

        return _session_event_sink

    def _execute_single_run(self, local_trainer: HalfCheetahTrainer, cfg: Dict[str, Any], compare_mode: bool):
        try:
            local_trainer.train(cfg)
        finally:
            self._unregister_worker(local_trainer)

    def _register_worker(self, worker: HalfCheetahTrainer):
        with self._workers_lock:
            self.current_workers.append(worker)
        worker.set_animation_enabled(bool(self.animation_on_var.get()))
        if self.is_paused:
            worker.pause()
        else:
            worker.resume()

    def _unregister_worker(self, worker: HalfCheetahTrainer):
        with self._workers_lock:
            try:
                self.current_workers.remove(worker)
            except ValueError:
                pass

    def _active_workers(self) -> List[HalfCheetahTrainer]:
        with self._workers_lock:
            return list(self.current_workers)

    def pause_or_resume(self):
        if not self.is_training:
            return
        if self.is_paused:
            self.trainer.resume()
            for worker in self._active_workers():
                worker.resume()
            self.is_paused = False
        else:
            self.trainer.pause()
            for worker in self._active_workers():
                worker.pause()
            self.is_paused = True
        self._set_control_highlight()

    def _cancel_current_run(self):
        for worker in self._active_workers():
            worker.resume()
            worker.request_stop()
        self.is_paused = False
        self.is_training = False
        with self._workers_lock:
            self.current_workers = []
        self._set_control_highlight()

    def reset_all(self):
        self._cancel_current_run()
        self._stop_frame_playback()
        self._flush_event_queue()
        self.compare_on_var.set(False)
        self._compare_clear()
        self.animation_on_var.set(True)
        self.max_steps_var.set(GENERAL_DEFAULTS["max_steps"])
        self.episodes_var.set(GENERAL_DEFAULTS["episodes"])
        self.epsilon_max_var.set(GENERAL_DEFAULTS["epsilon_max"])
        self.epsilon_decay_var.set(GENERAL_DEFAULTS["epsilon_decay"])
        self.epsilon_min_var.set(GENERAL_DEFAULTS["epsilon_min"])
        self.gamma_var.set(GENERAL_DEFAULTS["gamma"])
        self.policy_var.set("PPO")
        self._apply_policy_defaults()
        self.steps_progress_var.set(0)
        self.episodes_progress_var.set(0)
        self.status_var.set("Epsilon: - | LR: - | Best reward: - | Render: idle")
        self.latest_frame = None
        self.render_canvas.delete("all")

    def clear_plot(self):
        self.run_history.clear()
        self.run_colors.clear()
        self.run_metadata.clear()
        self.ax.clear()
        self.ax.set_xlabel("Episodes", color="#d0d0d0")
        self.ax.set_ylabel("Reward", color="#d0d0d0")
        self.ax.set_facecolor("#252526")
        self.ax.grid(True, alpha=0.2)
        for spine in self.ax.spines.values():
            spine.set_color("#d0d0d0")
        self.ax.tick_params(colors="#d0d0d0")
        self.canvas_plot.draw_idle()

    def save_samplings_csv(self):
        path = self.trainer.export_transitions_csv(run_id=self.latest_transitions_run_id or None)
        if path is None:
            messagebox.showinfo("Save samplings CSV", "No sampled transitions available yet.")
            return
        messagebox.showinfo("Save samplings CSV", f"Saved: {path}")

    def save_plot_png(self):
        output_dir = self.base_dir / "plots"
        output_dir.mkdir(parents=True, exist_ok=True)
        cfg = self._snapshot_config()
        env_part = "_".join(
            [
                f"frw-{cfg['env_params']['forward_reward_weight']}",
                f"ccw-{cfg['env_params']['ctrl_cost_weight']}",
                f"rns-{cfg['env_params']['reset_noise_scale']}",
            ]
        )
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"halfcheetah_{cfg['policy']}_steps-{cfg['max_steps']}_gamma-{cfg['gamma']}_{env_part}_{ts}.png"
        path = output_dir / filename
        self.fig.savefig(path, dpi=140)
        messagebox.showinfo("Save Plot PNG", f"Saved: {path}")

    def _pump_events(self):
        try:
            while True:
                event = self.event_queue.get_nowait()
                event_type = event.get("type")
                event_session_id = event.get("session_id")
                if event_session_id is not None and event_session_id != self.current_session_id:
                    continue
                if event_type in {"episode", "training_done", "error"}:
                    run_id = event.get("run_id")
                    if not run_id:
                        continue
                if event_type == "episode":
                    self._handle_episode_event(event)
                elif event_type == "training_done":
                    self._handle_training_done(event)
                elif event_type == "error":
                    self._handle_error_event(event)
                elif event_type == "worker_done":
                    if event.get("session_id") == self.current_session_id:
                        self.is_training = False
                        self.is_paused = False
                        with self._workers_lock:
                            self.current_workers = []
                        self._set_control_highlight()
        except queue.Empty:
            pass
        self.root.after(100, self._pump_events)

    def _handle_episode_event(self, event: Dict[str, Any]):
        run_id = event["run_id"]
        episode = int(event["episode"])
        episodes_total = int(event["episodes"])
        reward = float(event["reward"])
        ma = float(event["moving_average"])
        eval_points = list(event.get("eval_points", []))
        frames = list(event.get("frames", []))
        frame = event.get("frame")

        should_render_run = (self.render_run_id is None) or (run_id == self.render_run_id)
        if should_render_run and frames and self.animation_on_var.get():
            self._enqueue_or_start_playback(frames)
        elif should_render_run and frame is not None and self.playback_after_id is None:
            self.latest_frame = frame
            self._render_latest_frame()

        store = self.run_history.setdefault(run_id, {"episodes": [], "rewards": [], "ma": [], "eval": []})
        store["episodes"].append(episode)
        store["rewards"].append(reward)
        store["ma"].append(ma)
        store["eval"] = eval_points

        self.episodes_progress_var.set(100.0 * episode / max(1, episodes_total))
        self.status_var.set(
            f"Epsilon: {float(event.get('epsilon', 0.0)):.4f} | "
            f"LR: {float(event.get('lr', 0.0)):.3e} | "
            f"Best reward: {float(event.get('best_reward', 0.0)):.3f} | "
            f"Render: {event.get('render_state', 'idle')}"
        )
        self._redraw_plot()

    def _handle_training_done(self, event: Dict[str, Any]):
        if event.get("run_id") == self.current_run_id:
            self.latest_transitions_run_id = self.current_run_id
            self.status_var.set(
                f"Epsilon: {self.epsilon_min_var.get():.4f} | LR: {float(self.lr_var.get()):.3e} | "
                f"Best reward: {float(event.get('best_reward', 0.0)):.3f} | Render: idle"
            )

    def _handle_error_event(self, event: Dict[str, Any]):
        message = event.get("message", "Unknown error")
        messagebox.showerror("Training error", message)

    def _redraw_plot(self):
        self.ax.clear()
        self.ax.set_xlabel("Episodes", color="#d0d0d0")
        self.ax.set_ylabel("Reward", color="#d0d0d0")
        self.ax.set_facecolor("#252526")
        self.ax.grid(True, alpha=0.2)
        for spine in self.ax.spines.values():
            spine.set_color("#d0d0d0")
        self.ax.tick_params(colors="#d0d0d0")

        palette = matplotlib.rcParams.get("axes.prop_cycle", None)
        colors = []
        if palette is not None:
            try:
                colors = palette.by_key().get("color", [])
            except Exception:
                colors = []
        if not colors:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        for run_id, data in self.run_history.items():
            if run_id not in self.run_colors:
                color_index = len(self.run_colors) % len(colors)
                self.run_colors[run_id] = colors[color_index]
            color = self.run_colors[run_id]
            eps = data["episodes"]
            rewards = data["rewards"]
            ma = data["ma"]
            ev = data["eval"]
            if not eps:
                continue

            meta = self.run_metadata.get(run_id, {})
            env_meta = dict(meta.get("env", {}))
            compare_meta = dict(meta.get("compare_values", {}))
            label = (
                f"{meta.get('policy', self.policy_var.get())} | steps={meta.get('max_steps', self.max_steps_var.get())} | gamma={meta.get('gamma', self.gamma_var.get())}\n"
                f"epsilon={meta.get('epsilon_max', self.epsilon_max_var.get())} | epsilon_decay={meta.get('epsilon_decay', self.epsilon_decay_var.get())} | epsilon_min={meta.get('epsilon_min', self.epsilon_min_var.get())}\n"
                f"LR={meta.get('lr', self.lr_var.get())} | LR strategy={meta.get('lr_strategy', self.lr_strategy_var.get())} | LR decay={meta.get('lr_decay', self.lr_decay_var.get())}\n"
                f"forward_reward_weight={env_meta.get('forward_reward_weight', self.env_forward_reward_weight_var.get())} | ctrl_cost_weight={env_meta.get('ctrl_cost_weight', self.env_ctrl_cost_weight_var.get())} | reset_noise_scale={env_meta.get('reset_noise_scale', self.env_reset_noise_scale_var.get())}"
            )
            if compare_meta:
                already_stated = {
                    "policy",
                    "max_steps",
                    "gamma",
                    "epsilon_max",
                    "epsilon_decay",
                    "epsilon_min",
                    "lr",
                    "lr_strategy",
                    "lr_decay",
                    "forward_reward_weight",
                    "ctrl_cost_weight",
                    "reset_noise_scale",
                }
                extra_parts = []
                for key, value in compare_meta.items():
                    if key in already_stated:
                        continue
                    extra_parts.append(f"{key}={value}")
                if extra_parts:
                    label = f"{label}\ncompare: {' | '.join(extra_parts)}"
            reward_line, = self.ax.plot(eps, rewards, color=color, alpha=0.60, linewidth=1.5, label=label)
            ma_line, = self.ax.plot(eps, ma, color=color, alpha=1.0, linewidth=3.0, linestyle="--", label="moving average")
            if ev:
                x = [p[0] for p in ev]
                y = [p[1] for p in ev]
                eval_line, = self.ax.plot(x, y, color=color, alpha=1.0, linewidth=3.0, linestyle=":", marker="o", label="evaluation rollout")
            else:
                eval_line = None

            self._legend_artists[run_id] = [reward_line, ma_line, eval_line]

        self.legend_obj = self.ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, labelcolor="#d0d0d0")
        self.legend_scroll_y = 1.0
        self.legend_scroll_min_y = 1.0
        if self.legend_obj:
            self.legend_obj.set_draggable(False)
            for text in self.legend_obj.get_texts():
                text.set_picker(True)
            for handle in self.legend_obj.legend_handles:
                handle.set_picker(True)
            self.canvas_plot.draw()
            renderer = self.canvas_plot.get_renderer()
            legend_bbox = self.legend_obj.get_window_extent(renderer=renderer)
            axis_bbox = self.ax.get_window_extent(renderer=renderer)
            if axis_bbox.height > 0 and legend_bbox.height > axis_bbox.height:
                overflow_fraction = (legend_bbox.height - axis_bbox.height) / axis_bbox.height
                self.legend_scroll_min_y = 1.0 - overflow_fraction

            if self.legend_pick_cid is not None:
                self.canvas_plot.mpl_disconnect(self.legend_pick_cid)
            if self.legend_hover_cid is not None:
                self.canvas_plot.mpl_disconnect(self.legend_hover_cid)
            if self.legend_scroll_cid is not None:
                self.canvas_plot.mpl_disconnect(self.legend_scroll_cid)
            self.legend_pick_cid = self.canvas_plot.mpl_connect("pick_event", self._on_legend_pick)
            self.legend_hover_cid = self.canvas_plot.mpl_connect("motion_notify_event", self._on_legend_hover)
            self.legend_scroll_cid = self.canvas_plot.mpl_connect("scroll_event", self._on_legend_scroll)

        self.canvas_plot.draw_idle()

    def _on_legend_scroll(self, event):
        if not self.legend_obj or event is None:
            return
        if event.inaxes != self.ax:
            return
        contains, _ = self.legend_obj.contains(event)
        if not contains:
            return
        direction = 1 if getattr(event, "step", 0) > 0 else -1
        step = 0.06 * direction
        new_y = max(self.legend_scroll_min_y, min(1.0, self.legend_scroll_y + step))
        if new_y == self.legend_scroll_y:
            return
        self.legend_scroll_y = new_y
        self.legend_obj.set_bbox_to_anchor((1.01, self.legend_scroll_y))
        self.canvas_plot.draw_idle()

    def _on_legend_pick(self, event):
        artist = event.artist
        if not self.legend_obj:
            return
        labels = [txt.get_text() for txt in self.legend_obj.get_texts()]
        index = None
        if artist in self.legend_obj.get_texts():
            index = self.legend_obj.get_texts().index(artist)
        elif artist in self.legend_obj.legend_handles:
            index = self.legend_obj.legend_handles.index(artist)
        if index is None:
            return
        label = labels[index]
        lines = [ln for ln in self.ax.lines if ln.get_label() == label]
        if not lines:
            return
        visible = not lines[0].get_visible()
        for line in lines:
            line.set_visible(visible)
        self.canvas_plot.draw_idle()

    def _on_legend_hover(self, event):
        if not self.legend_obj:
            return
        over = False
        for text in self.legend_obj.get_texts():
            contains, _ = text.contains(event)
            if contains:
                text.set_alpha(0.8)
                over = True
            else:
                text.set_alpha(1.0)
        widget = self.canvas_plot.get_tk_widget()
        widget.configure(cursor="hand2" if over else "")
        if over:
            self.canvas_plot.draw_idle()

    def _render_latest_frame(self):
        if self.latest_frame is None:
            return
        if Image is None or ImageTk is None:
            return
        frame = self.latest_frame
        if not hasattr(frame, "shape"):
            return
        h, w = frame.shape[0], frame.shape[1]
        cw = max(1, self.render_canvas.winfo_width())
        ch = max(1, self.render_canvas.winfo_height())
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        image = Image.fromarray(frame)
        image = image.resize((nw, nh))
        self.latest_image_tk = ImageTk.PhotoImage(image)
        self.render_canvas.delete("all")
        self.render_canvas.create_image(cw // 2, ch // 2, image=self.latest_image_tk)

    def _start_frame_playback(self, frames: List[Any]):
        if not frames:
            return
        self._stop_frame_playback()
        self.playback_frames = list(frames)
        self.playback_index = 0
        self.playback_total_frames = len(self.playback_frames)
        self._play_next_frame()

    def _enqueue_or_start_playback(self, frames: List[Any]):
        if not frames:
            return
        if self.playback_after_id is None:
            self._start_frame_playback(frames)
            return
        self.pending_playback_frames = list(frames)

    def _play_next_frame(self):
        if not self.playback_frames:
            self.playback_after_id = None
            return
        if self.playback_index >= len(self.playback_frames):
            self.playback_after_id = None
            self.playback_frames = []
            self.playback_index = 0
            self.playback_total_frames = 0
            self.steps_progress_var.set(0)
            if self.pending_playback_frames:
                queued = list(self.pending_playback_frames)
                self.pending_playback_frames = []
                self._start_frame_playback(queued)
            return

        self.latest_frame = self.playback_frames[self.playback_index]
        self._render_latest_frame()

        played = self.playback_index + 1
        total = max(1, int(self.playback_total_frames))
        self.steps_progress_var.set(100.0 * min(total, played) / total)

        self.playback_index += 1
        fps = max(1, int(self.animation_fps_var.get()))
        delay_ms = max(1, int(1000 / fps))
        self.playback_after_id = self.root.after(delay_ms, self._play_next_frame)

    def _stop_frame_playback(self):
        if self.playback_after_id is not None:
            try:
                self.root.after_cancel(self.playback_after_id)
            except Exception:
                pass
        self.playback_after_id = None
        self.playback_frames = []
        self.pending_playback_frames = []
        self.playback_index = 0
        self.playback_total_frames = 0
        self.steps_progress_var.set(0)


__all__ = ["HalfCheetahGUI"]
