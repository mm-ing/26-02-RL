from __future__ import annotations

import os
import queue
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib as mpl
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

from Pusher_logic import (
    DEFAULT_POLICY,
    POLICY_DEFAULTS,
    POLICIES_CONTINUOUS,
    SHARED_SPECIFIC_KEYS,
    PusherTrainer,
    TrainerConfig,
    build_compare_configs,
)


POLICY_SPECIFIC_KEYS: Dict[str, List[str]] = {
    "PPO": ["n_steps", "n_epochs", "ent_coef"],
    "SAC": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps"],
    "TD3": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps", "policy_delay"],
    "DDPG": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps"],
}

CATEGORY_COMPLETION = {
    "policy": ["PPO", "SAC", "TD3", "DDPG"],
    "activation": ["relu", "tanh"],
    "lr_strategy": ["constant", "linear", "exponential"],
}

PARAM_TOOLTIPS = {
    "gamma": "Higher gamma values favor long-term reward but may slow convergence.",
    "learning_rate": "Higher learning rates speed updates but can destabilize training.",
    "batch_size": "Larger batches smooth gradients but increase compute and memory cost.",
    "hidden_layer": "Wider/deeper networks can model harder policies but train more slowly.",
    "activation": "Activation changes nonlinearity and can affect optimization stability.",
    "lr_strategy": "Schedule shape controls how aggressively learning rate decays over time.",
    "min_lr": "Lower floor allows finer late training adjustments but may slow adaptation.",
    "lr_decay": "Lower decay values reduce learning rate faster during training.",
}


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None) -> None:
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        win = tk.Toplevel(self.widget)
        win.wm_overrideredirect(True)
        win.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            win,
            text=self.text,
            bg="#2d2d30",
            fg="#e6e6e6",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=4,
            justify="left",
            wraplength=300,
        )
        label.pack(fill="both", expand=True)
        self.tip_window = win

    def _hide(self, _event=None) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class PusherGUI(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.master = master
        self.master.title("Pusher RL - SB3")
        self.pack(fill=tk.BOTH, expand=True)

        self._configure_style()

        self.event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.worker_lock = threading.Lock()
        self.worker_trainers: Dict[str, PusherTrainer] = {}
        self.worker_thread: Optional[threading.Thread] = None
        self.active_session_id: Optional[str] = None
        self._session_expected_done = 0
        self._session_done_count = 0

        self._animation_active = False
        self._animation_pending: Optional[List[Any]] = None
        self._animation_image_ref = None
        self._active_playback_steps = 0
        self._selected_render_policy = DEFAULT_POLICY

        self._run_series: Dict[str, Dict[str, Any]] = {}
        self._run_visibility: Dict[str, bool] = {}
        self._run_colors: Dict[str, Any] = {}
        self._color_index = 0
        self._legend = None
        self._legend_items: List[Tuple[Any, Any, List[Any]]] = []
        self._legend_entries: List[Tuple[Any, str, List[Any]]] = []
        self._legend_scroll_index = 0
        self._legend_max_visible = 12
        self._legend_hover_index: Optional[int] = None
        self._hidden_run_ids: set[str] = set()
        self._line_run_id_map: Dict[Any, str] = {}
        self._line_group_map: Dict[Any, List[Any]] = {}

        self._last_csv_path: Optional[str] = None
        self._force_export_csv_next_run = False
        self._paused = False
        self._training_active = False

        self._build_variables()
        self._build_layout()
        self._load_policy_values(DEFAULT_POLICY)
        self._refresh_compare_param_options()
        self._refresh_compare_summary()
        self._update_control_highlights()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(100, self._poll_worker_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        self.master.option_add("*Entry.insertBackground", "#ffffff")
        self.master.option_add("*TEntry*insertBackground", "#ffffff")
        try:
            style.theme_use("clam")
        except tk.TclError:
            try:
                style.theme_use("vista")
            except tk.TclError:
                pass

        self.master.configure(bg="#1e1e1e")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("TLabelframe", background="#252526", foreground="#e6e6e6")
        style.configure("TLabelframe.Label", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10))
        style.configure("Neutral.TButton", font=("Segoe UI", 10, "bold"), background="#3a3d41", foreground="#e6e6e6")
        style.map("Neutral.TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")])
        style.configure("Train.TButton", font=("Segoe UI", 10, "bold"), background="#0e639c", foreground="white")
        style.map("Train.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.configure("Pause.TButton", font=("Segoe UI", 10, "bold"), background="#a66a00", foreground="white")
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])

        style.configure("Dark.TEntry", fieldbackground="#2d2d30", foreground="#e6e6e6")
        style.configure("Dark.TCombobox", fieldbackground="#2d2d30", foreground="#e6e6e6")
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#2d2d30"), ("!disabled", "#2d2d30")],
            foreground=[("readonly", "#e6e6e6"), ("!disabled", "#e6e6e6")],
            selectbackground=[("readonly", "#2d2d30")],
            selectforeground=[("readonly", "#e6e6e6")],
        )
        style.configure("Dark.Horizontal.TProgressbar", troughcolor="#343434", background="#0e639c")

    def _build_variables(self) -> None:
        self.policy_var = tk.StringVar(value=DEFAULT_POLICY)
        self.device_var = tk.StringVar(value="CPU")
        self.animate_var = tk.BooleanVar(value=True)
        self.animation_fps_var = tk.IntVar(value=30)
        self.update_rate_var = tk.IntVar(value=1)
        self.frame_stride_var = tk.IntVar(value=2)

        self.compare_on_var = tk.BooleanVar(value=False)
        self.compare_param_var = tk.StringVar(value="policy")
        self.compare_values_var = tk.StringVar(value="")

        self.max_steps_var = tk.IntVar(value=200)
        self.episodes_var = tk.IntVar(value=3000)
        self.moving_avg_window_var = tk.IntVar(value=20)

        self.env_reward_near_weight_var = tk.DoubleVar(value=0.5)
        self.env_reward_dist_weight_var = tk.DoubleVar(value=1.0)
        self.env_reward_control_weight_var = tk.DoubleVar(value=0.1)

        self.shared_param_vars: Dict[str, tk.Variable] = {
            "gamma": tk.DoubleVar(value=0.99),
            "learning_rate": tk.StringVar(value="3e-4"),
            "batch_size": tk.IntVar(value=256),
            "hidden_layer": tk.StringVar(value="256,256"),
            "activation": tk.StringVar(value="relu"),
            "lr_strategy": tk.StringVar(value="constant"),
            "min_lr": tk.StringVar(value="1e-5"),
            "lr_decay": tk.DoubleVar(value=0.995),
        }

        self.policy_specific_vars: Dict[str, Dict[str, tk.Variable]] = {}
        for policy_name in POLICIES_CONTINUOUS:
            defaults = POLICY_DEFAULTS[policy_name]
            policy_vars: Dict[str, tk.Variable] = {}
            for key in POLICY_SPECIFIC_KEYS.get(policy_name, []):
                value = defaults.get(key)
                if isinstance(value, int):
                    policy_vars[key] = tk.IntVar(value=value)
                elif isinstance(value, float):
                    policy_vars[key] = tk.DoubleVar(value=value)
                else:
                    policy_vars[key] = tk.StringVar(value=str(value))
            self.policy_specific_vars[policy_name] = policy_vars

        self.policy_snapshots: Dict[str, Dict[str, Any]] = {
            name: dict(POLICY_DEFAULTS[name]) for name in POLICIES_CONTINUOUS
        }

        self.compare_params: Dict[str, List[Any]] = {}

        self.steps_progress_var = tk.DoubleVar(value=0.0)
        self.episodes_progress_var = tk.DoubleVar(value=0.0)
        self.status_var = tk.StringVar(value="Epsilon: 0.000 | LR: 0.000000 | Best reward: 0.00 | Render: idle")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(3, weight=1)

        self.environment_panel = ttk.LabelFrame(self, text="Environment")
        self.environment_panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.environment_panel.rowconfigure(0, weight=1)
        self.environment_panel.columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(self.environment_panel, bg="#111111", highlightthickness=0, height=720)
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.render_canvas.bind("<Configure>", self._on_canvas_resize)

        self.parameters_panel = ttk.LabelFrame(self, text="Parameters")
        self.parameters_panel.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        self.parameters_panel.rowconfigure(0, weight=1)
        self.parameters_panel.columnconfigure(0, weight=1)
        self._build_scrollable_parameters()

        self.controls_panel = ttk.LabelFrame(self, text="Controls")
        self.controls_panel.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.controls_panel.columnconfigure(tuple(range(8)), weight=1)
        self._build_controls_row()

        self.current_run_panel = ttk.LabelFrame(self, text="Current Run")
        self.current_run_panel.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.current_run_panel.columnconfigure(1, weight=1)

        ttk.Label(self.current_run_panel, text="Steps").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.steps_progress = ttk.Progressbar(
            self.current_run_panel,
            style="Dark.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            variable=self.steps_progress_var,
            maximum=100,
        )
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(self.current_run_panel, text="Episodes").grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.episodes_progress = ttk.Progressbar(
            self.current_run_panel,
            style="Dark.Horizontal.TProgressbar",
            orient="horizontal",
            mode="determinate",
            variable=self.episodes_progress_var,
            maximum=100,
        )
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(self.current_run_panel, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=4)

        self.plot_panel = ttk.LabelFrame(self, text="Live Plot")
        self.plot_panel.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.plot_panel.rowconfigure(0, weight=1)
        self.plot_panel.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 4), dpi=100, facecolor="#252526")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes", color="#e6e6e6")
        self.ax.set_ylabel("Reward", color="#e6e6e6")
        self.ax.tick_params(axis="x", colors="#d0d0d0")
        self.ax.tick_params(axis="y", colors="#d0d0d0")
        self.ax.grid(True, alpha=0.25, color="#777777")
        self.figure.subplots_adjust(left=0.04, right=0.78, bottom=0.14, top=0.95)

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_panel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._connect_plot_interactions()
        self._redraw_plot()

    def _connect_plot_interactions(self) -> None:
        self.figure.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.figure.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

    def _build_scrollable_parameters(self) -> None:
        self.params_canvas = tk.Canvas(self.parameters_panel, bg="#1e1e1e", highlightthickness=0)
        self.params_canvas.grid(row=0, column=0, sticky="nsew")
        self.params_scrollbar = ttk.Scrollbar(self.parameters_panel, orient="vertical", command=self.params_canvas.yview)
        self.params_scrollbar.grid(row=0, column=1, sticky="ns")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)

        self.params_inner = ttk.Frame(self.params_canvas)
        self.params_window = self.params_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        self.params_inner.columnconfigure(0, weight=1)
        self.params_canvas.itemconfigure(self.params_window, width=self.params_canvas.winfo_width())

        self.params_inner.bind("<Configure>", self._on_params_configure)
        self.params_canvas.bind("<Configure>", self._on_params_canvas_configure)
        self.params_canvas.bind("<Enter>", lambda _e: self.params_canvas.bind_all("<MouseWheel>", self._on_params_mousewheel))
        self.params_canvas.bind("<Leave>", lambda _e: self.params_canvas.unbind_all("<MouseWheel>"))

        self._build_environment_group(self.params_inner)
        self._build_compare_group(self.params_inner)
        self._build_general_group(self.params_inner)
        self._build_specific_group(self.params_inner)
        self._build_liveplot_group(self.params_inner)

    def _group_container(self, parent: tk.Widget, title: str, row: int) -> ttk.LabelFrame:
        grp = ttk.LabelFrame(parent, text=title)
        grp.grid(row=row, column=0, sticky="nsew", padx=6, pady=6)
        grp.columnconfigure(0, minsize=92)
        grp.columnconfigure(1, weight=1)
        return grp

    def _build_environment_group(self, parent: tk.Widget) -> None:
        group = self._group_container(parent, "Environment", 0)
        ttk.Checkbutton(group, text="Animation on", variable=self.animate_var, command=self._on_animation_toggle).grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=2)

        ttk.Label(group, text="Animation FPS").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.animation_fps_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(group, text="Update rate (episodes)").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.update_rate_var).grid(row=2, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(group, text="Frame stride").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.frame_stride_var).grid(row=3, column=1, sticky="ew", padx=4, pady=2)

        ttk.Button(group, text="Update", style="Neutral.TButton", command=self._apply_environment_update).grid(
            row=4, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )

        env_grid = ttk.Frame(group)
        env_grid.grid(row=5, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        env_grid.columnconfigure(0, weight=1)
        env_grid.columnconfigure(1, weight=1)
        env_grid.columnconfigure(2, weight=1)
        env_grid.columnconfigure(3, weight=1)

        ttk.Label(env_grid, text="reward_near_weight").grid(row=0, column=0, sticky="w", padx=2, pady=2)
        self._entry(env_grid, self.env_reward_near_weight_var).grid(row=0, column=1, sticky="ew", padx=2, pady=2)
        ttk.Label(env_grid, text="reward_dist_weight").grid(row=0, column=2, sticky="w", padx=2, pady=2)
        self._entry(env_grid, self.env_reward_dist_weight_var).grid(row=0, column=3, sticky="ew", padx=2, pady=2)
        ttk.Label(env_grid, text="reward_control_weight").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        self._entry(env_grid, self.env_reward_control_weight_var).grid(row=1, column=1, sticky="ew", padx=2, pady=2)

    def _build_compare_group(self, parent: tk.Widget) -> None:
        group = self._group_container(parent, "Compare", 1)

        top = ttk.Frame(group)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)

        ttk.Checkbutton(top, text="Compare on", variable=self.compare_on_var, command=self._on_compare_toggle).grid(
            row=0, column=0, sticky="w", padx=2
        )
        ttk.Button(top, text="Clear", style="Neutral.TButton", command=self._clear_compare_params).grid(row=0, column=1, padx=2)
        ttk.Button(top, text="Add", style="Neutral.TButton", command=self._add_compare_param).grid(row=0, column=2, padx=2)

        row2 = ttk.Frame(group)
        row2.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        row2.columnconfigure(0, weight=1)
        row2.columnconfigure(1, weight=1)

        self.compare_param_combo = self._combobox(row2, self.compare_param_var, values=["policy"], width=16)
        self.compare_param_combo.grid(row=0, column=0, sticky="ew", padx=2)
        self.compare_values_entry = self._entry(row2, self.compare_values_var)
        self.compare_values_entry.grid(row=0, column=1, sticky="ew", padx=2)
        self.compare_values_entry.bind("<Return>", lambda _e: self._add_compare_param())
        self.compare_values_entry.bind("<KeyRelease>", self._update_completion_hint)
        self.compare_values_entry.bind("<Tab>", self._accept_completion)

        self.completion_hint = ttk.Label(group, text="", foreground="#d0d0d0")
        self.completion_hint.grid(row=2, column=1, sticky="w", padx=8, pady=1)

        self.compare_summary_var = tk.StringVar(value="")
        ttk.Label(group, textvariable=self.compare_summary_var, justify="left").grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=2)

    def _build_general_group(self, parent: tk.Widget) -> None:
        group = self._group_container(parent, "General", 2)
        ttk.Label(group, text="Max steps").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.max_steps_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(group, text="Episodes").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.episodes_var).grid(row=1, column=1, sticky="ew", padx=4, pady=2)

    def _build_specific_group(self, parent: tk.Widget) -> None:
        group = self._group_container(parent, "Specific", 3)

        ttk.Label(group, text="Policy").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        policy_combo = self._combobox(group, self.policy_var, values=POLICIES_CONTINUOUS, width=16)
        policy_combo.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        policy_combo.bind("<<ComboboxSelected>>", self._on_policy_selected)

        shared_panel = ttk.Frame(group)
        shared_panel.grid(row=1, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        shared_panel.columnconfigure(0, minsize=92)
        shared_panel.columnconfigure(1, weight=1)
        shared_panel.columnconfigure(2, minsize=92)
        shared_panel.columnconfigure(3, weight=1)

        row_map = {
            "gamma": (0, 0),
            "learning_rate": (0, 2),
            "batch_size": (1, 0),
            "hidden_layer": (1, 2),
            "activation": (2, 0),
            "lr_strategy": (2, 2),
            "min_lr": (3, 0),
            "lr_decay": (3, 2),
        }
        self.shared_widgets: Dict[str, tk.Widget] = {}
        for key in SHARED_SPECIFIC_KEYS:
            r, c = row_map[key]
            label = ttk.Label(shared_panel, text=key)
            label.grid(row=r, column=c, sticky="w", padx=4, pady=2)
            if key == "activation":
                widget = self._combobox(shared_panel, self.shared_param_vars[key], ["relu", "tanh"], width=16)
            elif key == "lr_strategy":
                widget = self._combobox(shared_panel, self.shared_param_vars[key], ["constant", "linear", "exponential"], width=16)
            else:
                widget = self._entry(shared_panel, self.shared_param_vars[key])
            widget.grid(row=r, column=c + 1, sticky="ew", padx=4, pady=2)
            self.shared_widgets[key] = widget
            if key in PARAM_TOOLTIPS:
                ToolTip(label, PARAM_TOOLTIPS[key])
                ToolTip(widget, PARAM_TOOLTIPS[key])

        ttk.Separator(group, orient="horizontal").grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=6)
        self.policy_specific_panel = ttk.Frame(group)
        self.policy_specific_panel.grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        self.policy_specific_panel.columnconfigure(0, minsize=92)
        self.policy_specific_panel.columnconfigure(1, weight=1)
        self._render_policy_specific_inputs()

    def _build_liveplot_group(self, parent: tk.Widget) -> None:
        group = self._group_container(parent, "Live Plot", 4)
        ttk.Label(group, text="Moving average values").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self._entry(group, self.moving_avg_window_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)

    def _build_controls_row(self) -> None:
        self.run_single_btn = ttk.Button(self.controls_panel, text="Run single episode", style="Neutral.TButton", command=self.run_single_episode)
        self.train_btn = ttk.Button(self.controls_panel, text="Train and Run", style="Neutral.TButton", command=self.start_training)
        self.pause_run_btn = ttk.Button(self.controls_panel, text="Pause", style="Neutral.TButton", command=self.pause_or_run)
        self.reset_btn = ttk.Button(self.controls_panel, text="Reset All", style="Neutral.TButton", command=self.reset_all)
        self.clear_plot_btn = ttk.Button(self.controls_panel, text="Clear Plot", style="Neutral.TButton", command=self.clear_plot)
        self.save_csv_btn = ttk.Button(self.controls_panel, text="Save samplings CSV", style="Neutral.TButton", command=self.save_samplings_csv)
        self.save_plot_btn = ttk.Button(self.controls_panel, text="Save Plot PNG", style="Neutral.TButton", command=self.save_plot_png)
        self.device_combo = self._combobox(self.controls_panel, self.device_var, ["CPU", "GPU"], width=10)

        controls: List[tk.Widget] = [
            self.run_single_btn,
            self.train_btn,
            self.pause_run_btn,
            self.reset_btn,
            self.clear_plot_btn,
            self.save_csv_btn,
            self.save_plot_btn,
            self.device_combo,
        ]
        for idx, widget in enumerate(controls):
            widget.grid(row=0, column=idx, sticky="ew", padx=4, pady=6)
        self.pause_run_btn.configure(state=tk.DISABLED)

    def _entry(self, parent: tk.Widget, var: tk.Variable) -> ttk.Entry:
        entry = ttk.Entry(parent, textvariable=var, width=9, style="Dark.TEntry")
        entry.configure(cursor="xterm")
        entry.bind("<FocusIn>", lambda _e: entry.configure(takefocus=True))
        return entry

    def _combobox(self, parent: tk.Widget, var: tk.Variable, values: List[Any], width: int = 12) -> ttk.Combobox:
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=width, style="Dark.TCombobox")
        combo.bind("<MouseWheel>", lambda _e: "break")
        return combo

    def _on_params_configure(self, _event=None) -> None:
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        bbox = self.params_canvas.bbox("all")
        if bbox is None:
            self.params_scrollbar.grid_remove()
            return
        needs_scroll = bbox[3] > self.params_canvas.winfo_height() + 2
        if needs_scroll:
            self.params_scrollbar.grid()
        else:
            self.params_scrollbar.grid_remove()

    def _on_params_canvas_configure(self, event) -> None:
        self.params_canvas.itemconfig(self.params_window, width=event.width)

    def _on_params_mousewheel(self, event) -> None:
        self.params_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_canvas_resize(self, _event=None) -> None:
        if self._animation_image_ref is not None:
            self.render_canvas.itemconfig("frame_image", image=self._animation_image_ref)

    def _render_frame_on_canvas(self, frame: Any) -> None:
        if Image is None or ImageTk is None:
            return
        try:
            pil_img = Image.fromarray(frame)
            cw = max(1, self.render_canvas.winfo_width())
            ch = max(1, self.render_canvas.winfo_height())
            iw, ih = pil_img.size
            scale = min(cw / max(1, iw), ch / max(1, ih))
            nw = max(1, int(iw * scale))
            nh = max(1, int(ih * scale))
            pil_img = pil_img.resize((nw, nh))
            tk_img = ImageTk.PhotoImage(pil_img)
            self._animation_image_ref = tk_img
            self.render_canvas.delete("frame_image")
            self.render_canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center", tags="frame_image")
        except Exception:
            pass

    def _apply_environment_update(self) -> None:
        params = self._read_env_params()
        with self.worker_lock:
            trainers = list(self.worker_trainers.values())
        for trainer in trainers:
            trainer.update_environment(params)
        messagebox.showinfo("Pusher", "Environment parameters will be used for the next run.")

    def _read_env_params(self) -> Dict[str, Any]:
        return {
            "reward_near_weight": float(self.env_reward_near_weight_var.get()),
            "reward_dist_weight": float(self.env_reward_dist_weight_var.get()),
            "reward_control_weight": float(self.env_reward_control_weight_var.get()),
        }

    def _capture_policy_values(self, policy_name: str) -> None:
        snapshot: Dict[str, Any] = {}
        for key, var in self.shared_param_vars.items():
            snapshot[key] = var.get()
        for key in POLICY_SPECIFIC_KEYS.get(policy_name, []):
            snapshot[key] = self.policy_specific_vars[policy_name][key].get()
        self.policy_snapshots[policy_name] = snapshot

    def _load_policy_values(self, policy_name: str) -> None:
        snapshot = dict(POLICY_DEFAULTS[policy_name])
        snapshot.update(self.policy_snapshots.get(policy_name, {}))
        for key in SHARED_SPECIFIC_KEYS:
            if key in snapshot:
                self.shared_param_vars[key].set(snapshot[key])
        for key in POLICY_SPECIFIC_KEYS.get(policy_name, []):
            self.policy_specific_vars[policy_name][key].set(snapshot.get(key, POLICY_DEFAULTS[policy_name].get(key)))

    def _on_policy_selected(self, _event=None) -> None:
        previous = getattr(self, "_last_policy", DEFAULT_POLICY)
        self._capture_policy_values(previous)
        selected = self.policy_var.get()
        self._render_policy_specific_inputs()
        self._load_policy_values(selected)
        self._last_policy = selected
        self._refresh_compare_param_options()

    def _render_policy_specific_inputs(self) -> None:
        for child in self.policy_specific_panel.winfo_children():
            child.destroy()
        policy_name = self.policy_var.get()
        for row, key in enumerate(POLICY_SPECIFIC_KEYS.get(policy_name, [])):
            ttk.Label(self.policy_specific_panel, text=key).grid(row=row, column=0, sticky="w", padx=4, pady=2)
            self._entry(self.policy_specific_panel, self.policy_specific_vars[policy_name][key]).grid(row=row, column=1, sticky="ew", padx=4, pady=2)

    def _refresh_compare_param_options(self) -> None:
        options = ["policy", "max_steps", "episodes"]
        options.extend(SHARED_SPECIFIC_KEYS)
        options.extend(POLICY_SPECIFIC_KEYS.get(self.policy_var.get(), []))
        self.compare_param_combo.configure(values=options)
        if self.compare_param_var.get() not in options:
            self.compare_param_var.set(options[0])

    def _on_compare_toggle(self) -> None:
        if self.compare_on_var.get():
            self.animate_var.set(False)
            self._on_animation_toggle()

    def _on_animation_toggle(self) -> None:
        if not self.animate_var.get():
            self._animation_pending = None
            self._animation_active = False
            self.steps_progress_var.set(0.0)
            status = self.status_var.get()
            if "| Render:" in status:
                self.status_var.set(status.rsplit("| Render:", 1)[0] + "| Render: off")

    def _coerce_compare_value(self, text: str) -> Any:
        raw = text.strip()
        low = raw.lower()
        if low in {"true", "false"}:
            return low == "true"
        try:
            if "." in raw or "e" in low:
                return float(raw)
            return int(raw)
        except ValueError:
            return raw

    def _add_compare_param(self) -> None:
        key = self.compare_param_var.get().strip()
        values_raw = self.compare_values_var.get().strip()
        if not key or not values_raw:
            return
        values = [self._coerce_compare_value(chunk) for chunk in values_raw.split(",") if chunk.strip()]
        if not values:
            return
        self.compare_params[key] = values
        self.compare_values_var.set("")
        self._refresh_compare_summary()

    def _clear_compare_params(self) -> None:
        self.compare_params = {}
        self._refresh_compare_summary()

    def _refresh_compare_summary(self) -> None:
        lines = [f"{k}: [{', '.join([str(v) for v in vals])}]" for k, vals in self.compare_params.items()]
        self.compare_summary_var.set("\n".join(lines))

    def _completion_candidate(self) -> Optional[str]:
        param = self.compare_param_var.get().strip().lower()
        choices = CATEGORY_COMPLETION.get(param)
        if not choices:
            return None

        typed = self.compare_values_var.get()
        token = typed.split(",")[-1].strip().lower()
        if not token:
            return None
        for choice in choices:
            if choice.lower().startswith(token):
                return choice
        return None

    def _update_completion_hint(self, _event=None) -> None:
        candidate = self._completion_candidate()
        if candidate is None:
            self.completion_hint.configure(text="")
        else:
            self.completion_hint.configure(text=f"Tab -> {candidate}")

    def _accept_completion(self, _event=None):
        candidate = self._completion_candidate()
        if candidate is None:
            return "break"
        text = self.compare_values_var.get()
        parts = [chunk for chunk in text.split(",")]
        parts[-1] = candidate
        new_text = ",".join(parts)
        self.compare_values_var.set(new_text)
        self.compare_values_entry.icursor(len(new_text))
        self._update_completion_hint()
        return "break"

    def _build_trainer_config(self) -> TrainerConfig:
        policy_name = self.policy_var.get()
        shared_params = {k: self.shared_param_vars[k].get() for k in SHARED_SPECIFIC_KEYS}
        shared_params["learning_rate"] = float(shared_params["learning_rate"])
        shared_params["min_lr"] = float(shared_params["min_lr"])
        policy_params = {
            key: self.policy_specific_vars[policy_name][key].get()
            for key in POLICY_SPECIFIC_KEYS.get(policy_name, [])
        }
        return TrainerConfig(
            policy=policy_name,
            episodes=max(1, int(self.episodes_var.get())),
            max_steps=max(1, int(self.max_steps_var.get())),
            update_rate=max(1, int(self.update_rate_var.get())),
            frame_stride=max(1, int(self.frame_stride_var.get())),
            device=self.device_var.get(),
            enable_animation=bool(self.animate_var.get()),
            env_params=self._read_env_params(),
            shared_params=shared_params,
            policy_params=policy_params,
            collect_transitions=True,
            export_transitions_csv=bool(self._force_export_csv_next_run),
            results_dir="results_csv",
            plots_dir="plots",
        )

    def _flush_queue(self) -> None:
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def _register_trainers_for_session(self, session_id: str, trainers: Dict[str, PusherTrainer]) -> None:
        with self.worker_lock:
            self.worker_trainers.update(trainers)
        if self._paused:
            for trainer in trainers.values():
                trainer.request_pause()

    def _remove_session_trainers(self, session_id: str) -> None:
        with self.worker_lock:
            keys = [k for k in self.worker_trainers.keys() if str(k).startswith(session_id)]
            for key in keys:
                self.worker_trainers.pop(key, None)

    def run_single_episode(self) -> None:
        cfg = self._build_trainer_config()
        cfg.episodes = 1
        self._launch_training([cfg])

    def start_training(self) -> None:
        if self._training_active and self._paused:
            self.cancel_training()
        if self._training_active:
            return

        base = self._build_trainer_config()
        if self.compare_on_var.get() and self.compare_params:
            configs = build_compare_configs(base, dict(self.compare_params))
        elif self.compare_on_var.get():
            configs = build_compare_configs(base, {"policy": list(POLICIES_CONTINUOUS)})
        else:
            configs = [base]
        self._launch_training(configs)

    def _launch_training(self, configs: List[TrainerConfig]) -> None:
        self._capture_policy_values(self.policy_var.get())
        self._flush_queue()
        self._session_done_count = 0
        self._session_expected_done = len(configs)
        self.steps_progress_var.set(0.0)
        self.episodes_progress_var.set(0.0)
        self._paused = False
        self._training_active = True
        self._update_control_highlights()

        session_id = uuid.uuid4().hex
        self.active_session_id = session_id
        self._selected_render_policy = self.policy_var.get()

        if len(configs) > 1:
            render_index = 0
            for idx, cfg in enumerate(configs):
                if cfg.policy == self._selected_render_policy:
                    render_index = idx
                    break
            for idx, cfg in enumerate(configs):
                cfg.enable_animation = bool(self.animate_var.get() and idx == render_index)

        self.worker_thread = threading.Thread(
            target=self._worker_session,
            args=(session_id, configs),
            daemon=True,
        )
        self.worker_thread.start()
        self._update_control_highlights()

    def _worker_session(self, session_id: str, configs: List[TrainerConfig]) -> None:
        workers = max(1, min(4, len(configs)))
        cores = max(1, int(os.cpu_count() or 1))
        per_worker_threads = max(1, cores // workers)

        trainers: Dict[str, PusherTrainer] = {}
        for idx, cfg in enumerate(configs):
            cfg.cpu_threads = per_worker_threads
            key = f"{session_id}:{idx}"
            trainers[key] = PusherTrainer(event_sink=lambda payload, sid=session_id: self._enqueue_worker_event(sid, payload))

        self._register_trainers_for_session(session_id, trainers)

        try:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = []
                for idx, cfg in enumerate(configs):
                    key = f"{session_id}:{idx}"
                    futures.append(executor.submit(trainers[key].train, replace(cfg)))
                for fut in futures:
                    fut.result()
        finally:
            self._remove_session_trainers(session_id)

    def _enqueue_worker_event(self, session_id: str, payload: Dict[str, Any]) -> None:
        event = dict(payload)
        event["session_id"] = session_id
        self.event_queue.put(event)

    def pause_or_run(self) -> None:
        if not self._training_active:
            return
        if not self._paused:
            self.pause_training()
        else:
            self.resume_training()

    def pause_training(self) -> None:
        if not self._training_active:
            return
        with self.worker_lock:
            trainers = list(self.worker_trainers.values())
        for trainer in trainers:
            trainer.request_pause()
        self._paused = True
        self._update_control_highlights()

    def resume_training(self) -> None:
        if not self._training_active:
            return
        with self.worker_lock:
            trainers = list(self.worker_trainers.values())
        for trainer in trainers:
            trainer.request_resume()
        self._paused = False
        self._update_control_highlights()

    def cancel_training(self) -> None:
        with self.worker_lock:
            trainers = list(self.worker_trainers.values())
        for trainer in trainers:
            trainer.request_cancel()
            trainer.request_resume()
        self._training_active = False
        self._paused = False
        self._update_control_highlights()

    def reset_all(self) -> None:
        self.cancel_training()
        self.policy_var.set(DEFAULT_POLICY)
        self.device_var.set("CPU")
        self.animate_var.set(True)
        self.animation_fps_var.set(30)
        self.update_rate_var.set(1)
        self.frame_stride_var.set(2)
        self.compare_on_var.set(False)
        self.compare_values_var.set("")
        self.compare_params = {}
        self._refresh_compare_summary()
        self.max_steps_var.set(200)
        self.episodes_var.set(3000)
        self.moving_avg_window_var.set(20)
        self.env_reward_near_weight_var.set(0.5)
        self.env_reward_dist_weight_var.set(1.0)
        self.env_reward_control_weight_var.set(0.1)

        self.policy_snapshots = {name: dict(POLICY_DEFAULTS[name]) for name in POLICIES_CONTINUOUS}
        self._load_policy_values(self.policy_var.get())
        self._render_policy_specific_inputs()

    def clear_plot(self) -> None:
        self._run_series = {}
        self._run_visibility = {}
        self._run_colors = {}
        self._legend_entries = []
        self._legend_items = []
        self._legend = None
        self._legend_scroll_index = 0
        self._legend_hover_index = None
        self._hidden_run_ids = set()
        self._line_run_id_map = {}
        self._line_group_map = {}
        self._redraw_plot()

    def save_samplings_csv(self) -> None:
        if self._last_csv_path and os.path.exists(self._last_csv_path):
            self.status_var.set(
                f"Epsilon: 0.000 | LR: 0.000000 | Best reward: 0.00 | Render: idle"
            )
        else:
            self._force_export_csv_next_run = True

    def _safe_name(self, text: str) -> str:
        compact = re.sub(r"\s+", "_", text.strip())
        compact = re.sub(r"[^a-zA-Z0-9_\-]", "", compact)
        return compact[:120] if compact else "run"

    def save_plot_png(self) -> None:
        os.makedirs("plots", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy = self.policy_var.get()
        max_steps = int(self.max_steps_var.get())
        gamma = float(self.shared_param_vars["gamma"].get())
        learning_rate = float(self.shared_param_vars["learning_rate"].get())
        path = os.path.join(
            "plots",
            f"pusher_{policy}_steps-{max_steps}_gamma-{gamma:g}_lr-{learning_rate:.0e}_{ts}.png",
        )
        self.figure.savefig(path, dpi=140)
        messagebox.showinfo("Pusher", f"Saved plot: {path}")

    def _poll_worker_events(self) -> None:
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_worker_event(event)
        self.after(100, self._poll_worker_events)

    def _handle_worker_event(self, event: Dict[str, Any]) -> None:
        if event.get("session_id") != self.active_session_id:
            return

        etype = event.get("type")
        if etype == "episode":
            self._handle_episode_event(event)
        elif etype == "training_done":
            self._session_done_count += 1
            self._last_csv_path = event.get("csv_path") or self._last_csv_path
            self._force_export_csv_next_run = False
            self._export_plot_png(event)
            if self._session_done_count >= self._session_expected_done:
                self._training_active = False
                self._paused = False
                self._update_control_highlights()
        elif etype == "error":
            self._training_active = False
            self._paused = False
            self._update_control_highlights()

    def _run_color(self, run_id: str):
        if run_id not in self._run_colors:
            colors = mpl.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            self._run_colors[run_id] = colors[self._color_index % len(colors)]
            self._color_index += 1
        return self._run_colors[run_id]

    def _handle_episode_event(self, event: Dict[str, Any]) -> None:
        run_id = str(event.get("run_id", ""))
        if not run_id:
            return
        color = self._run_color(run_id)

        reward = float(event.get("reward", 0.0))
        episode = int(event.get("episode", 0))
        episodes = max(1, int(event.get("episodes", 1)))
        moving_avg = float(event.get("moving_average", reward))
        eval_points = list(event.get("eval_points", []))

        if run_id not in self._run_series:
            snapshot = event.get("config_snapshot", {})
            shared = snapshot.get("shared_params", {}) if isinstance(snapshot, dict) else {}
            envp = snapshot.get("env_params", {}) if isinstance(snapshot, dict) else {}
            label = (
                f"{event.get('policy', 'policy')} | steps={snapshot.get('max_steps', 'na')} | gamma={shared.get('gamma', 'na')}\n"
                f"epsilon=0.0 | epsilon_decay=na | epsilon_min=na\n"
                f"LR={shared.get('learning_rate', 'na')} | LR strategy={shared.get('lr_strategy', 'na')} | LR decay={shared.get('lr_decay', 'na')}\n"
                f"reward_near_weight={envp.get('reward_near_weight', 'na')} | reward_dist_weight={envp.get('reward_dist_weight', 'na')} | reward_control_weight={envp.get('reward_control_weight', 'na')}"
            )
            self._run_series[run_id] = {
                "episodes": [],
                "rewards": [],
                "ma": [],
                "eval": eval_points,
                "label": label,
                "color": color,
                "policy": event.get("policy", ""),
            }
            self._run_visibility[run_id] = True

        self._run_series[run_id]["episodes"].append(episode)
        self._run_series[run_id]["rewards"].append(reward)
        self._run_series[run_id]["ma"].append(moving_avg)
        self._run_series[run_id]["eval"] = eval_points

        self.episodes_progress_var.set((episode / episodes) * 100.0)

        frames = event.get("frames") or []
        render_state = str(event.get("render_state", "idle"))
        if event.get("policy") == self._selected_render_policy and frames and self.animate_var.get():
            self._queue_animation_frames(frames)
            render_state = "on"
        elif not self.animate_var.get():
            render_state = "off"

        self.status_var.set(
            f"Epsilon: {float(event.get('epsilon', 0.0)):.3f} | "
            f"LR: {float(event.get('lr', 0.0)):.6f} | "
            f"Best reward: {float(event.get('best_reward', 0.0)):.2f} | "
            f"Render: {render_state}"
        )

        self._redraw_plot()

    def _redraw_plot(self) -> None:
        self.ax.clear()
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes", color="#e6e6e6")
        self.ax.set_ylabel("Reward", color="#e6e6e6")
        self.ax.tick_params(axis="x", colors="#d0d0d0")
        self.ax.tick_params(axis="y", colors="#d0d0d0")
        self.ax.grid(True, alpha=0.25, color="#777777")

        self._line_run_id_map.clear()
        self._line_group_map.clear()
        legend_entries: List[Tuple[Any, str, List[Any]]] = []
        for run_id, series in self._run_series.items():
            if self._run_visibility.get(run_id, True):
                self._hidden_run_ids.discard(run_id)
            else:
                self._hidden_run_ids.add(run_id)
            x = series["episodes"]
            y = series["rewards"]
            ma = series["ma"]
            eval_points = series["eval"]
            c = series["color"]

            reward_line, = self.ax.plot(x, y, color=c, alpha=0.60, linewidth=1.2, label=series["label"])
            ma_line, = self.ax.plot(x, ma, color=c, alpha=1.0, linewidth=2.4, linestyle="--")
            run_group = [reward_line, ma_line]
            legend_entries.append((reward_line, series["label"], [reward_line]))
            legend_entries.append((ma_line, "moving average", [ma_line]))
            if eval_points:
                ex = [p[0] for p in eval_points]
                ey = [p[1] for p in eval_points]
                eval_line, = self.ax.plot(ex, ey, color=c, alpha=1.0, linewidth=2.4, linestyle=":", marker="o", markersize=3)
                run_group.append(eval_line)
                legend_entries.append((eval_line, "evaluation rollout", [eval_line]))

            is_hidden = run_id in self._hidden_run_ids
            for artist in run_group:
                artist.set_visible(not is_hidden)

            self._line_run_id_map[reward_line] = run_id
            self._line_group_map[reward_line] = run_group

        self._legend_entries = legend_entries
        self._rebuild_interactive_legend()

        for spine in self.ax.spines.values():
            spine.set_color("#777777")

        self.figure.subplots_adjust(left=0.04, right=0.78, bottom=0.14, top=0.95)
        self.canvas.draw_idle()

    def _rebuild_interactive_legend(self) -> None:
        self._legend_items.clear()
        if not self._legend_entries:
            self._legend = None
            return

        if self._legend is not None:
            self._legend.remove()
            self._legend = None

        if len(self._legend_entries) <= self._legend_max_visible:
            self._legend_scroll_index = 0
        else:
            max_start = max(0, len(self._legend_entries) - self._legend_max_visible)
            self._legend_scroll_index = min(max(0, self._legend_scroll_index), max_start)

        start = self._legend_scroll_index
        end = start + self._legend_max_visible
        visible_entries = self._legend_entries[start:end]
        handles = [entry[0] for entry in visible_entries]
        labels = [entry[1] for entry in visible_entries]

        self._legend = self.ax.legend(
            handles,
            labels,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=False,
            labelcolor="#e6e6e6",
            fontsize=8,
        )

        if self._legend is None:
            return

        legend_handles = getattr(self._legend, "legendHandles", getattr(self._legend, "legend_handles", []))
        for idx, (legend_handle, legend_text) in enumerate(zip(legend_handles, self._legend.get_texts())):
            entry = visible_entries[idx]
            artists = entry[2]
            legend_handle.set_picker(True)
            legend_text.set_picker(True)
            self._legend_items.append((legend_handle, legend_text, artists))
            self._set_legend_item_visual(idx)

    def _set_legend_item_visual(self, index: int) -> None:
        if index < 0 or index >= len(self._legend_items):
            return
        legend_handle, legend_text, artists = self._legend_items[index]
        visible = any(artist.get_visible() for artist in artists)
        base_alpha = 1.0 if visible else 0.3
        if self._legend_hover_index == index:
            base_alpha = 1.0
            legend_text.set_color("#0e639c")
        else:
            legend_text.set_color("#e6e6e6")
        legend_handle.set_alpha(base_alpha)
        legend_text.set_alpha(base_alpha)

    def _event_hits_legend(self, event) -> bool:
        for legend_handle, legend_text, _ in self._legend_items:
            handle_hit = legend_handle.contains(event)[0]
            text_hit = legend_text.contains(event)[0]
            if handle_hit or text_hit:
                return True
        return False

    def _toggle_run_visibility(self, run_id: str) -> None:
        if run_id not in self._run_series:
            return
        now_visible = run_id in self._hidden_run_ids
        self._run_visibility[run_id] = bool(now_visible)
        if now_visible:
            self._hidden_run_ids.discard(run_id)
        else:
            self._hidden_run_ids.add(run_id)
        self._redraw_plot()

    def _on_legend_pick(self, event) -> None:
        for idx, (legend_handle, legend_text, artists) in enumerate(self._legend_items):
            if event.artist in (legend_handle, legend_text):
                run_id = self._line_run_id_map.get(artists[0]) if artists else None
                targets = self._line_group_map.get(artists[0], artists)
                make_visible = not any(artist.get_visible() for artist in targets)
                for artist in targets:
                    artist.set_visible(make_visible)
                if run_id is not None:
                    self._run_visibility[run_id] = bool(make_visible)
                    if make_visible:
                        self._hidden_run_ids.discard(run_id)
                    else:
                        self._hidden_run_ids.add(run_id)
                self._set_legend_item_visual(idx)
                self.canvas.draw_idle()
                return

    def _on_plot_motion(self, event) -> None:
        if self._legend is None or event.inaxes != self.ax:
            if self._legend_hover_index is not None:
                old = self._legend_hover_index
                self._legend_hover_index = None
                self._set_legend_item_visual(old)
                self.canvas.get_tk_widget().configure(cursor="")
                self.canvas.draw_idle()
            return

        hover_index = None
        for idx, (legend_handle, legend_text, _) in enumerate(self._legend_items):
            if legend_handle.contains(event)[0] or legend_text.contains(event)[0]:
                hover_index = idx
                break

        if hover_index != self._legend_hover_index:
            if self._legend_hover_index is not None:
                self._set_legend_item_visual(self._legend_hover_index)
            self._legend_hover_index = hover_index
            if self._legend_hover_index is not None:
                self._set_legend_item_visual(self._legend_hover_index)
                self.canvas.get_tk_widget().configure(cursor="hand2")
            else:
                self.canvas.get_tk_widget().configure(cursor="")
            self.canvas.draw_idle()

    def _on_plot_scroll(self, event) -> None:
        if self._legend is None or not self._event_hits_legend(event):
            return
        if len(self._legend_entries) <= self._legend_max_visible:
            return

        max_start = max(0, len(self._legend_entries) - self._legend_max_visible)
        old_index = self._legend_scroll_index
        if getattr(event, "button", "") == "up" or getattr(event, "step", 0) > 0:
            self._legend_scroll_index = max(0, self._legend_scroll_index - 1)
        else:
            self._legend_scroll_index = min(max_start, self._legend_scroll_index + 1)

        if self._legend_scroll_index != old_index:
            self._legend_hover_index = None
            self._rebuild_interactive_legend()
            self.canvas.get_tk_widget().configure(cursor="")
            self.canvas.draw_idle()

    def _export_plot_png(self, event: Dict[str, Any]) -> None:
        run_id = str(event.get("run_id", "run"))
        snapshot = event.get("config_snapshot", {})
        shared = snapshot.get("shared_params", {}) if isinstance(snapshot, dict) else {}
        policy = self._safe_name(str(event.get("policy", "policy")))
        lr = shared.get("learning_rate", "na")
        bs = shared.get("batch_size", "na")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("plots", exist_ok=True)
        filename = f"{policy}__lr-{lr}__bs-{bs}__{self._safe_name(run_id)}__{timestamp}.png"
        self.figure.savefig(os.path.join("plots", filename), dpi=140)

    def _queue_animation_frames(self, frames: List[Any]) -> None:
        if self._animation_active:
            self._animation_pending = frames
            return
        self._start_animation(frames)

    def _start_animation(self, frames: List[Any]) -> None:
        self._animation_active = True
        self._active_playback_steps = len(frames)
        self.steps_progress_var.set(0.0)
        self._play_frames(frames, 0)

    def _play_frames(self, frames: List[Any], index: int) -> None:
        if index >= len(frames):
            if self._animation_pending is not None:
                pending = self._animation_pending
                self._animation_pending = None
                self._start_animation(pending)
                return
            self._animation_active = False
            return

        self._render_frame_on_canvas(frames[index])
        denom = max(1, self._active_playback_steps)
        self.steps_progress_var.set(((index + 1) / denom) * 100.0)
        delay = int(1000 / max(1, int(self.animation_fps_var.get())))
        self.after(delay, lambda: self._play_frames(frames, index + 1))

    def _update_control_highlights(self) -> None:
        if self._training_active and not self._paused:
            self.train_btn.configure(style="Train.TButton")
            self.pause_run_btn.configure(style="Neutral.TButton")
            self.pause_run_btn.configure(text="Pause")
            self.pause_run_btn.configure(state=tk.NORMAL)
        elif self._training_active and self._paused:
            self.train_btn.configure(style="Neutral.TButton")
            self.pause_run_btn.configure(style="Pause.TButton")
            self.pause_run_btn.configure(text="Run")
            self.pause_run_btn.configure(state=tk.NORMAL)
        else:
            self.train_btn.configure(style="Neutral.TButton")
            self.pause_run_btn.configure(style="Neutral.TButton")
            self.pause_run_btn.configure(text="Pause")
            self.pause_run_btn.configure(state=tk.DISABLED)

    def _on_close(self) -> None:
        self.cancel_training()
        self.master.destroy()
