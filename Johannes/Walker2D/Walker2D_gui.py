from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import itertools
import os
import queue
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import tkinter as tk
from tkinter import ttk, messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib
import numpy as np
import torch

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

from Walker2D_logic import (
    POLICY_DEFAULTS,
    POLICY_SHARED_DEFAULTS,
    SHARED_DEFAULTS,
    TrainConfig,
    Walker2DEnvConfig,
    Walker2DTrainer,
)


matplotlib.use("TkAgg")


@dataclass
class RunPlotState:
    rewards_x: List[int]
    rewards_y: List[float]
    moving_average_x: List[int]
    moving_average_y: List[float]
    eval_x: List[int]
    eval_y: List[float]
    color: str
    label: str


class _SimpleTooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        tip = tk.Toplevel(self.widget)
        tip.wm_overrideredirect(True)
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip,
            text=self.text,
            bg="#2d2d30",
            fg="#e6e6e6",
            relief=tk.SOLID,
            borderwidth=1,
            padx=6,
            pady=3,
            font=("Segoe UI", 9),
            justify=tk.LEFT,
        )
        label.pack()
        self.tip_window = tip

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class Walker2DGUI(ttk.Frame):
    PALETTE = {
        "main_bg": "#1e1e1e",
        "panel_bg": "#252526",
        "input_bg": "#2d2d30",
        "text": "#e6e6e6",
        "muted": "#d0d0d0",
        "accent": "#0e639c",
        "neutral_btn": "#3a3d41",
        "neutral_active": "#4a4f55",
        "neutral_pressed": "#2f3338",
        "pause_btn": "#a66a00",
    }

    POLICY_SPECIFIC_FIELDS: Dict[str, List[Tuple[str, Any]]] = {
        "PPO": [
            ("n_steps", 2048),
            ("gae_lambda", 0.95),
            ("clip_range", 0.2),
            ("ent_coef", 0.0),
        ],
        "SAC": [
            ("buffer_size", 300000),
            ("learning_starts", 10000),
            ("tau", 0.005),
            ("train_freq", 1),
            ("gradient_steps", 1),
        ],
        "TD3": [
            ("buffer_size", 300000),
            ("learning_starts", 10000),
            ("tau", 0.005),
            ("policy_delay", 2),
            ("train_freq", 1),
            ("gradient_steps", 1),
            ("target_policy_noise", 0.2),
            ("target_noise_clip", 0.5),
        ],
    }

    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master = master
        self.master.title("Walker2D")
        self.master.configure(bg=self.PALETTE["main_bg"])
        self.pack(fill=tk.BOTH, expand=True)

        self.event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.session_id: Optional[str] = None
        self.active_trainers: Dict[str, Walker2DTrainer] = {}
        self.active_trainers_lock = threading.Lock()
        self.training_thread: Optional[threading.Thread] = None
        self.training_active = False
        self.training_paused = False
        self.compare_active = False
        self.compare_expected_done = 0
        self.compare_done_count = 0
        self.render_run_id: Optional[str] = None
        self.last_completed_trainer: Optional[Walker2DTrainer] = None

        self.last_render_image = None
        self.last_canvas_size = (640, 360)
        self._playback_active = False
        self._playback_pending: Optional[List[np.ndarray]] = None

        self.run_plots: Dict[str, RunPlotState] = {}
        self.run_meta_snapshots: Dict[str, Dict[str, Any]] = {}
        self._legend = None
        self._legend_items: List[Tuple[Any, Any, List[Any]]] = []
        self._legend_entries: List[Tuple[Any, str, List[Any]]] = []
        self._legend_scroll_index = 0
        self._legend_max_visible = 12
        self._legend_hover_index: Optional[int] = None
        self._hidden_run_ids: set[str] = set()
        self._line_run_id_map: Dict[Any, str] = {}
        self._line_group_map: Dict[Any, List[Any]] = {}
        self._tooltips: List[_SimpleTooltip] = []

        self._init_vars()
        self._init_style()
        self._build_layout()
        self._configure_plot()
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

        self.after(100, self._poll_worker_events)

    def _init_vars(self) -> None:
        self.var_animation_on = tk.BooleanVar(value=True)
        self.var_animation_fps = tk.IntVar(value=30)
        self.var_update_rate = tk.IntVar(value=1)
        self.var_frame_stride = tk.IntVar(value=2)

        self.var_compare_on = tk.BooleanVar(value=False)
        self.var_compare_parameter = tk.StringVar(value="Policy")
        self.var_compare_values = tk.StringVar(value="")

        self.var_max_steps = tk.IntVar(value=1000)
        self.var_episodes = tk.IntVar(value=1000)

        self.var_policy = tk.StringVar(value="PPO")
        ppo_shared = POLICY_SHARED_DEFAULTS.get("PPO", SHARED_DEFAULTS)
        self.var_gamma = tk.DoubleVar(value=float(ppo_shared["gamma"]))
        self.var_learning_rate = tk.StringVar(value=f"{float(ppo_shared['learning_rate']):.1e}")
        self.var_batch_size = tk.IntVar(value=int(ppo_shared["batch_size"]))
        self.var_hidden_layer = tk.StringVar(value=str(int(ppo_shared["hidden_layer"])))
        self.var_lr_strategy = tk.StringVar(value=str(ppo_shared["lr_strategy"]))
        self.var_min_lr = tk.StringVar(value=f"{float(ppo_shared['min_lr']):.1e}")
        self.var_lr_decay = tk.DoubleVar(value=float(ppo_shared["lr_decay"]))

        self.var_moving_average_values = tk.IntVar(value=20)
        self.var_show_advanced = tk.BooleanVar(value=False)
        self.var_rollout_capture_steps = tk.IntVar(value=120)
        self.var_low_overhead_animation = tk.BooleanVar(value=False)

        self.var_device = tk.StringVar(value="CPU")

        self.var_status = tk.StringVar(value="Epsilon: n/a | LR: 3.0e-04 | Best reward: 0.0 | Render: idle")
        self.var_steps_text = tk.StringVar(value="Steps: 0 / 0")
        self.var_episodes_text = tk.StringVar(value="Episodes: 0 / 0")

        self.env_param_vars: Dict[str, Any] = {
            "forward_reward_weight": tk.DoubleVar(value=1.0),
            "ctrl_cost_weight": tk.DoubleVar(value=1e-3),
            "healthy_reward": tk.DoubleVar(value=1.0),
            "terminate_when_unhealthy": tk.BooleanVar(value=True),
            "healthy_z_range_low": tk.DoubleVar(value=0.8),
            "healthy_z_range_high": tk.DoubleVar(value=2.0),
            "healthy_angle_range_low": tk.DoubleVar(value=-1.0),
            "healthy_angle_range_high": tk.DoubleVar(value=1.0),
            "reset_noise_scale": tk.DoubleVar(value=5e-3),
            "exclude_current_positions_from_observation": tk.BooleanVar(value=True),
        }

        self.compare_lines: List[str] = []
        self.compare_items: Dict[str, List[str]] = {}
        self.current_specific_vars: Dict[str, Any] = {}
        self.previous_policy = "PPO"
        self._suspend_shared_cache_sync = False
        self.cached_shared_values: Dict[str, Dict[str, Any]] = {
            name: {
                "gamma": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["gamma"]),
                "learning_rate": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["learning_rate"]),
                "batch_size": int(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["batch_size"]),
                "hidden_layer": str(int(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["hidden_layer"])),
                "lr_strategy": str(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["lr_strategy"]),
                "min_lr": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["min_lr"]),
                "lr_decay": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["lr_decay"]),
            }
            for name in self.POLICY_SPECIFIC_FIELDS
        }
        self.cached_specific_values: Dict[str, Dict[str, Any]] = {
            name: dict(POLICY_DEFAULTS[name]) for name in self.POLICY_SPECIFIC_FIELDS
        }

    def _init_style(self) -> None:
        self.master.option_add("*Entry.insertBackground", "#ffffff")
        self.master.option_add("*TEntry*insertBackground", "#ffffff")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            try:
                style.theme_use("vista")
            except tk.TclError:
                pass

        style.configure(".", font=("Segoe UI", 10), background=self.PALETTE["main_bg"], foreground=self.PALETTE["text"])
        style.configure("TFrame", background=self.PALETTE["main_bg"])
        style.configure("Panel.TFrame", background=self.PALETTE["panel_bg"])
        style.configure("Panel.TLabelframe", background=self.PALETTE["panel_bg"], foreground=self.PALETTE["text"])
        style.configure("Panel.TLabelframe.Label", background=self.PALETTE["panel_bg"], foreground=self.PALETTE["text"], font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=self.PALETTE["panel_bg"], foreground=self.PALETTE["text"], font=("Segoe UI", 10))
        style.configure("TCheckbutton", background=self.PALETTE["panel_bg"], foreground=self.PALETTE["text"], font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground=self.PALETTE["input_bg"], foreground=self.PALETTE["text"])
        style.configure("TCombobox", fieldbackground=self.PALETTE["input_bg"], background=self.PALETTE["input_bg"], foreground=self.PALETTE["text"])
        style.map("TCombobox", fieldbackground=[("readonly", self.PALETTE["input_bg"])], selectbackground=[("readonly", self.PALETTE["accent"])])
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=4)
        style.configure("Neutral.TButton", font=("Segoe UI", 10, "bold"), background=self.PALETTE["neutral_btn"], foreground=self.PALETTE["text"])
        style.map("Neutral.TButton", background=[("active", self.PALETTE["neutral_active"]), ("pressed", self.PALETTE["neutral_pressed"])])
        style.configure("Accent.TButton", background=self.PALETTE["accent"], foreground="white")
        style.map("Accent.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.configure("Pause.TButton", background=self.PALETTE["pause_btn"], foreground="white")
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])
        style.configure("Horizontal.TProgressbar", troughcolor="#343434", background=self.PALETTE["accent"])

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=2)
        self.rowconfigure(3, weight=3)

        self.environment_group = ttk.LabelFrame(self, text="Environment", style="Panel.TLabelframe")
        self.environment_group.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.parameters_group = ttk.LabelFrame(self, text="Parameters", style="Panel.TLabelframe")
        self.parameters_group.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.controls_group = ttk.LabelFrame(self, text="Controls", style="Panel.TLabelframe")
        self.controls_group.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.current_run_group = ttk.LabelFrame(self, text="Current Run", style="Panel.TLabelframe")
        self.current_run_group.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        self.plot_group = ttk.LabelFrame(self, text="Live Plot", style="Panel.TLabelframe")
        self.plot_group.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        self._build_environment_panel()
        self._build_parameters_panel()
        self._build_controls_row()
        self._build_current_run_panel()
        self._build_plot_panel()

    def _build_environment_panel(self) -> None:
        self.environment_group.rowconfigure(0, weight=1)
        self.environment_group.columnconfigure(0, weight=1)
        self.render_canvas = tk.Canvas(
            self.environment_group,
            bg="#111111",
            highlightthickness=0,
            bd=0,
        )
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.render_canvas.bind("<Configure>", self._on_canvas_resize)

    def _build_parameters_panel(self) -> None:
        self.parameters_group.rowconfigure(0, weight=1)
        self.parameters_group.columnconfigure(0, weight=1)

        self.params_canvas = tk.Canvas(
            self.parameters_group,
            bg=self.PALETTE["panel_bg"],
            highlightthickness=0,
            bd=0,
        )
        self.params_canvas.grid(row=0, column=0, sticky="nsew")
        self.params_scrollbar = ttk.Scrollbar(self.parameters_group, orient=tk.VERTICAL, command=self.params_canvas.yview)
        self.params_scrollbar.grid(row=0, column=1, sticky="ns")
        self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)

        self.params_inner = ttk.Frame(self.params_canvas)
        self.params_window = self.params_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
        self.params_inner.columnconfigure(0, weight=1)
        self.params_inner.bind("<Configure>", self._on_params_configure)
        self.params_canvas.bind("<Configure>", self._on_params_canvas_configure)
        self.params_canvas.bind("<Enter>", lambda _: self._bind_mousewheel())
        self.params_canvas.bind("<Leave>", lambda _: self._unbind_mousewheel())

        self.env_params_frame = ttk.LabelFrame(self.params_inner, text="Environment", style="Panel.TLabelframe")
        self.compare_frame = ttk.LabelFrame(self.params_inner, text="Compare", style="Panel.TLabelframe")
        self.general_frame = ttk.LabelFrame(self.params_inner, text="General", style="Panel.TLabelframe")
        self.specific_frame = ttk.LabelFrame(self.params_inner, text="Specific", style="Panel.TLabelframe")
        self.live_plot_frame = ttk.LabelFrame(self.params_inner, text="Live Plot", style="Panel.TLabelframe")

        for idx, frame in enumerate(
            [self.env_params_frame, self.compare_frame, self.general_frame, self.specific_frame, self.live_plot_frame]
        ):
            frame.grid(row=idx, column=0, sticky="ew", padx=6, pady=4)
            frame.columnconfigure(0, weight=1)
            frame.columnconfigure(1, weight=1)
            frame.columnconfigure(2, weight=1)
            frame.columnconfigure(3, weight=1)

        self._build_env_param_group()
        self._build_compare_group()
        self._build_general_group()
        self._build_specific_group()
        self._build_live_plot_group()

    def _build_env_param_group(self) -> None:
        ttk.Checkbutton(self.env_params_frame, text="Animation on", variable=self.var_animation_on, command=self._on_animation_toggle).grid(
            row=0, column=0, sticky="w", padx=4, pady=2
        )
        ttk.Label(self.env_params_frame, text="Animation FPS").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        animation_fps_entry = ttk.Entry(self.env_params_frame, textvariable=self.var_animation_fps, width=9)
        animation_fps_entry.grid(row=0, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(animation_fps_entry, "Higher FPS makes replay smoother but increases rendering load.")

        ttk.Label(self.env_params_frame, text="Update rate (episodes)").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        update_rate_entry = ttk.Entry(self.env_params_frame, textvariable=self.var_update_rate, width=9)
        update_rate_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(update_rate_entry, "Higher values reduce replay frequency and lower GUI overhead.")
        ttk.Label(self.env_params_frame, text="Frame stride").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        frame_stride_entry = ttk.Entry(self.env_params_frame, textvariable=self.var_frame_stride, width=9)
        frame_stride_entry.grid(row=1, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(frame_stride_entry, "Larger stride samples fewer rollout frames and speeds up replay.")

        ttk.Button(self.env_params_frame, text="Update", style="Neutral.TButton", command=self._apply_environment_update).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 6)
        )

        fields = [
            ("forward_reward_weight", self.env_param_vars["forward_reward_weight"]),
            ("ctrl_cost_weight", self.env_param_vars["ctrl_cost_weight"]),
            ("healthy_reward", self.env_param_vars["healthy_reward"]),
            ("terminate_when_unhealthy", self.env_param_vars["terminate_when_unhealthy"]),
            ("healthy_z_min", self.env_param_vars["healthy_z_range_low"]),
            ("healthy_z_max", self.env_param_vars["healthy_z_range_high"]),
            ("healthy_angle_min", self.env_param_vars["healthy_angle_range_low"]),
            ("healthy_angle_max", self.env_param_vars["healthy_angle_range_high"]),
            ("reset_noise_scale", self.env_param_vars["reset_noise_scale"]),
            (
                "exclude_current_positions_from_observation",
                self.env_param_vars["exclude_current_positions_from_observation"],
            ),
        ]
        start_row = 3
        for idx, (name, var) in enumerate(fields):
            row = start_row + (idx // 2)
            col = (idx % 2) * 2
            if isinstance(var, tk.BooleanVar):
                ttk.Checkbutton(self.env_params_frame, text=name, variable=var).grid(
                    row=row, column=col, columnspan=2, sticky="w", padx=4, pady=2
                )
            else:
                ttk.Label(self.env_params_frame, text=name).grid(row=row, column=col, sticky="w", padx=4, pady=2)
                param_entry = ttk.Entry(self.env_params_frame, textvariable=var, width=9)
                param_entry.grid(
                    row=row, column=col + 1, sticky="ew", padx=4, pady=2
                )
                if name == "healthy_reward":
                    self._bind_tooltip(param_entry, "Higher healthy reward encourages stable upright walking behavior.")
                elif name == "reset_noise_scale":
                    self._bind_tooltip(param_entry, "Higher reset noise increases start variation and can improve robustness.")

    def _build_compare_group(self) -> None:
        top_row = ttk.Frame(self.compare_frame)
        top_row.grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        top_row.columnconfigure(0, weight=1)
        top_row.columnconfigure(1, weight=0)
        top_row.columnconfigure(2, weight=0)

        ttk.Checkbutton(top_row, text="Compare on", variable=self.var_compare_on, command=self._on_compare_toggle).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(top_row, text="Clear", style="Neutral.TButton", command=self._clear_compare).grid(row=0, column=1, sticky="e", padx=(4, 2))
        ttk.Button(top_row, text="Add", style="Neutral.TButton", command=self._add_compare_line).grid(row=0, column=2, sticky="e")

        self.compare_param_combo = ttk.Combobox(
            self.compare_frame,
            textvariable=self.var_compare_parameter,
            values=[],
            state="readonly",
            width=16,
        )
        self.compare_param_combo.grid(row=1, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        self._disable_combobox_wheel(self.compare_param_combo)
        self.compare_param_combo.bind("<<ComboboxSelected>>", self._on_compare_parameter_changed)

        self.compare_values_entry = ttk.Entry(self.compare_frame, textvariable=self.var_compare_values)
        self.compare_values_entry.grid(row=1, column=2, columnspan=2, sticky="ew", padx=4, pady=2)
        self.compare_values_entry.bind("<Return>", lambda _: self._add_compare_line())
        self.compare_values_entry.bind("<Tab>", self._on_compare_tab_complete)
        self.compare_values_entry.bind("<KeyRelease>", self._on_compare_values_changed)

        self.compare_hint = ttk.Label(self.compare_frame, text="", foreground=self.PALETTE["muted"])
        self.compare_hint.grid(row=2, column=2, columnspan=2, sticky="w", padx=4, pady=(0, 2))
        self._refresh_compare_parameter_options()
        self._update_compare_hint()

        self.compare_summary = tk.Text(
            self.compare_frame,
            height=4,
            bg=self.PALETTE["input_bg"],
            fg=self.PALETTE["text"],
            insertbackground="white",
            relief=tk.FLAT,
        )
        self.compare_summary.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        self.compare_summary.configure(state=tk.DISABLED)

    def _build_general_group(self) -> None:
        ttk.Label(self.general_frame, text="Max steps").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        max_steps_entry = ttk.Entry(self.general_frame, textvariable=self.var_max_steps, width=9)
        max_steps_entry.grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(max_steps_entry, "Higher max steps allows longer rollouts but increases compute per episode.")
        ttk.Label(self.general_frame, text="Episodes").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        episodes_entry = ttk.Entry(self.general_frame, textvariable=self.var_episodes, width=9)
        episodes_entry.grid(row=0, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(episodes_entry, "More episodes improve convergence potential but extend training time.")

    def _build_specific_group(self) -> None:
        ttk.Label(self.specific_frame, text="Policy").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.policy_combo = ttk.Combobox(
            self.specific_frame,
            textvariable=self.var_policy,
            values=["PPO", "SAC", "TD3"],
            state="readonly",
            width=16,
        )
        self.policy_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=2)
        self.policy_combo.bind("<<ComboboxSelected>>", self._on_policy_changed)
        self._disable_combobox_wheel(self.policy_combo)
        self._bind_tooltip(self.policy_combo, "Select the SB3 algorithm used for training and evaluation.")

        ttk.Label(self.specific_frame, text="gamma").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        gamma_entry = ttk.Entry(self.specific_frame, textvariable=self.var_gamma, width=9)
        gamma_entry.grid(row=1, column=1, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(gamma_entry, "Higher gamma values prioritize long-term rewards over immediate gains.")
        ttk.Label(self.specific_frame, text="learning_rate").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        learning_rate_entry = ttk.Entry(self.specific_frame, textvariable=self.var_learning_rate, width=9)
        learning_rate_entry.grid(row=1, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(learning_rate_entry, "Higher learning rates learn faster but can destabilize optimization.")

        ttk.Label(self.specific_frame, text="batch_size").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        batch_size_entry = ttk.Entry(self.specific_frame, textvariable=self.var_batch_size, width=9)
        batch_size_entry.grid(row=2, column=1, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(batch_size_entry, "Larger batches smooth updates but require more memory and time.")

        ttk.Label(self.specific_frame, text="hidden_layer").grid(row=2, column=2, sticky="w", padx=4, pady=2)
        hidden_layer_entry = ttk.Entry(self.specific_frame, textvariable=self.var_hidden_layer, width=9)
        hidden_layer_entry.grid(row=2, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(hidden_layer_entry, "Hidden layer widths for policy/value networks (for example 256 or 256,128,64); larger values increase capacity and compute cost.")

        ttk.Label(self.specific_frame, text="lr_strategy").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        lr_strategy_combo = ttk.Combobox(
            self.specific_frame,
            textvariable=self.var_lr_strategy,
            values=["constant", "linear", "exponential"],
            state="readonly",
            width=9,
        )
        lr_strategy_combo.grid(row=3, column=1, sticky="ew", padx=4, pady=2)
        self._disable_combobox_wheel(lr_strategy_combo)
        self._bind_tooltip(lr_strategy_combo, "Learning-rate schedule strategy used during training.")

        ttk.Label(self.specific_frame, text="min_lr").grid(row=3, column=2, sticky="w", padx=4, pady=2)
        min_lr_entry = ttk.Entry(self.specific_frame, textvariable=self.var_min_lr, width=9)
        min_lr_entry.grid(row=3, column=3, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(min_lr_entry, "Minimum learning-rate floor used by schedule strategies.")

        ttk.Label(self.specific_frame, text="lr_decay").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        lr_decay_entry = ttk.Entry(self.specific_frame, textvariable=self.var_lr_decay, width=9)
        lr_decay_entry.grid(row=4, column=1, sticky="ew", padx=4, pady=2)
        self._bind_tooltip(lr_decay_entry, "Decay factor used by exponential learning-rate schedule.")

        self.var_gamma.trace_add("write", self._on_shared_specific_value_changed)
        self.var_learning_rate.trace_add("write", self._on_shared_specific_value_changed)
        self.var_batch_size.trace_add("write", self._on_shared_specific_value_changed)
        self.var_hidden_layer.trace_add("write", self._on_shared_specific_value_changed)
        self.var_lr_strategy.trace_add("write", self._on_shared_specific_value_changed)
        self.var_min_lr.trace_add("write", self._on_shared_specific_value_changed)
        self.var_lr_decay.trace_add("write", self._on_shared_specific_value_changed)

        self.specific_separator = ttk.Separator(self.specific_frame, orient=tk.HORIZONTAL)
        self.specific_separator.grid(row=5, column=0, columnspan=4, sticky="ew", padx=4, pady=4)

        self.specific_dynamic_frame = ttk.Frame(self.specific_frame)
        self.specific_dynamic_frame.grid(row=6, column=0, columnspan=4, sticky="ew", padx=0, pady=0)
        self._render_specific_fields(self.var_policy.get())

    def _build_live_plot_group(self) -> None:
        ttk.Label(self.live_plot_frame, text="Moving average values").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        moving_avg_entry = ttk.Entry(self.live_plot_frame, textvariable=self.var_moving_average_values, width=9)
        moving_avg_entry.grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )
        self._bind_tooltip(moving_avg_entry, "Higher window values smooth reward curves but hide short-term changes.")
        ttk.Checkbutton(self.live_plot_frame, text="Show Advanced", variable=self.var_show_advanced, command=self._toggle_advanced).grid(
            row=0, column=2, columnspan=2, sticky="w", padx=4, pady=2
        )

        self.advanced_frame = ttk.Frame(self.live_plot_frame)
        self.advanced_frame.grid(row=1, column=0, columnspan=4, sticky="ew")
        ttk.Label(self.advanced_frame, text="Rollout full-capture steps").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        capture_steps_entry = ttk.Entry(self.advanced_frame, textvariable=self.var_rollout_capture_steps, width=9)
        capture_steps_entry.grid(
            row=0, column=1, sticky="ew", padx=4, pady=2
        )
        self._bind_tooltip(capture_steps_entry, "Limits how many early rollout steps are captured for animation playback.")
        ttk.Checkbutton(self.advanced_frame, text="Low-overhead animation", variable=self.var_low_overhead_animation).grid(
            row=0, column=2, columnspan=2, sticky="w", padx=4, pady=2
        )
        self.advanced_frame.grid_remove()

    def _build_controls_row(self) -> None:
        for column in range(8):
            self.controls_group.columnconfigure(column, weight=1)

        self.btn_run_episode = ttk.Button(self.controls_group, text="Run single episode", command=self._run_single_episode)
        self.btn_train = ttk.Button(self.controls_group, text="Train and Run", style="Neutral.TButton", command=self._start_training)
        self.btn_pause = ttk.Button(self.controls_group, text="Pause", style="Neutral.TButton", command=self._toggle_pause)
        self.btn_reset = ttk.Button(self.controls_group, text="Reset All", style="Neutral.TButton", command=self._reset_all)
        self.btn_clear_plot = ttk.Button(self.controls_group, text="Clear Plot", style="Neutral.TButton", command=self._clear_plot)
        self.btn_save_csv = ttk.Button(self.controls_group, text="Save samplings CSV", style="Neutral.TButton", command=self._save_sampling_csv)
        self.btn_save_png = ttk.Button(self.controls_group, text="Save Plot PNG", style="Neutral.TButton", command=self._save_plot_png)
        self.device_combo = ttk.Combobox(
            self.controls_group,
            values=["CPU", "GPU"],
            textvariable=self.var_device,
            state="readonly",
            width=8,
        )
        self._disable_combobox_wheel(self.device_combo)
        self._bind_tooltip(self.device_combo, "GPU can accelerate training when CUDA is available, otherwise CPU is used.")

        controls = [
            self.btn_run_episode,
            self.btn_train,
            self.btn_pause,
            self.btn_reset,
            self.btn_clear_plot,
            self.btn_save_csv,
            self.btn_save_png,
            self.device_combo,
        ]
        for idx, widget in enumerate(controls):
            widget.grid(row=0, column=idx, sticky="ew", padx=4, pady=6)
        self.btn_run_episode.configure(style="Neutral.TButton")

        self.btn_pause.configure(state=tk.DISABLED)

    def _build_current_run_panel(self) -> None:
        self.current_run_group.columnconfigure(1, weight=1)
        ttk.Label(self.current_run_group, textvariable=self.var_steps_text).grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.steps_progress = ttk.Progressbar(self.current_run_group, orient=tk.HORIZONTAL, mode="determinate")
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(self.current_run_group, textvariable=self.var_episodes_text).grid(row=1, column=0, sticky="w", padx=6, pady=4)
        self.episodes_progress = ttk.Progressbar(self.current_run_group, orient=tk.HORIZONTAL, mode="determinate")
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(self.current_run_group, textvariable=self.var_status).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 6)
        )

    def _build_plot_panel(self) -> None:
        self.plot_group.columnconfigure(0, weight=1)
        self.plot_group.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 5), dpi=100, facecolor=self.PALETTE["panel_bg"])
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.ax.set_facecolor(self.PALETTE["panel_bg"])

        self.figure.subplots_adjust(left=0.04, right=0.78, top=0.97, bottom=0.11)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_group)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.canvas.mpl_connect("pick_event", self._on_plot_pick)
        self.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

    def _configure_plot(self) -> None:
        self.ax.grid(True, color=self.PALETTE["muted"], alpha=0.35)
        self.ax.tick_params(colors=self.PALETTE["muted"])
        self.ax.xaxis.label.set_color(self.PALETTE["text"])
        self.ax.yaxis.label.set_color(self.PALETTE["text"])
        for spine in self.ax.spines.values():
            spine.set_color(self.PALETTE["muted"])

    def _on_canvas_resize(self, event: tk.Event) -> None:
        self.last_canvas_size = (max(1, event.width), max(1, event.height))
        if self.last_render_image is not None:
            self._draw_frame(self.last_render_image)

    def _on_params_configure(self, _: tk.Event) -> None:
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        required_height = self.params_inner.winfo_reqheight()
        current_height = self.params_canvas.winfo_height()
        if required_height > current_height:
            self.params_scrollbar.grid()
        else:
            self.params_scrollbar.grid_remove()

    def _on_params_canvas_configure(self, event: tk.Event) -> None:
        self.params_canvas.itemconfig(self.params_window, width=event.width)

    def _bind_mousewheel(self) -> None:
        self.params_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self) -> None:
        self.params_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event: tk.Event) -> None:
        delta = int(-1 * (event.delta / 120))
        self.params_canvas.yview_scroll(delta, "units")

    def _on_compare_toggle(self) -> None:
        if self.var_compare_on.get():
            self.var_animation_on.set(False)
            self._on_animation_toggle()

    def _on_animation_toggle(self) -> None:
        if not self.var_animation_on.get():
            self._playback_pending = None
            self._playback_active = False
            self.var_status.set(self.var_status.get().rsplit("| Render:", 1)[0] + "| Render: off")
            self.steps_progress["value"] = 0

    def _apply_environment_update(self) -> None:
        messagebox.showinfo("Walker2D", "Environment parameters will be used for the next run.")

    def _add_compare_line(self) -> None:
        key = self.var_compare_parameter.get().strip()
        raw_values = self.var_compare_values.get().strip()
        if not key or not raw_values:
            return
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not values:
            return
        self.compare_items[key] = values
        self.compare_lines = [f"{name}: [{', '.join(items)}]" for name, items in self.compare_items.items()]
        self.compare_summary.configure(state=tk.NORMAL)
        self.compare_summary.delete("1.0", tk.END)
        self.compare_summary.insert("1.0", "\n".join(self.compare_lines))
        self.compare_summary.configure(state=tk.DISABLED)

    def _clear_compare(self) -> None:
        self.compare_items.clear()
        self.compare_lines.clear()
        self.compare_summary.configure(state=tk.NORMAL)
        self.compare_summary.delete("1.0", tk.END)
        self.compare_summary.configure(state=tk.DISABLED)

    def _on_compare_tab_complete(self, event: tk.Event) -> str:
        key = self.var_compare_parameter.get().strip()
        options = self._get_compare_categorical_options(key)
        text = self.var_compare_values.get().strip()
        if not text:
            self._update_compare_hint()
            return "break"
        lower = text.lower()
        for item in options:
            if item.lower().startswith(lower):
                self.var_compare_values.set(item)
                self.compare_values_entry.icursor(tk.END)
                self._update_compare_hint()
                break
        self._update_compare_hint()
        return "break"

    def _on_compare_parameter_changed(self, _: tk.Event) -> None:
        self._update_compare_hint()

    def _on_compare_values_changed(self, _: tk.Event) -> None:
        self._update_compare_hint()

    def _on_policy_changed(self, _: tk.Event) -> None:
        self._sync_current_policy_shared_values()
        self._sync_current_policy_specific_values()
        current_policy = self.var_policy.get()
        self._load_policy_shared_values(current_policy)
        self._render_specific_fields(current_policy)
        self._refresh_compare_parameter_options()
        self.previous_policy = current_policy

    def _sync_current_policy_shared_values(self) -> None:
        self._cache_policy_shared_values(self.previous_policy)

    def _cache_policy_shared_values(self, policy: str) -> None:
        store = self.cached_shared_values.setdefault(
            policy,
            {
                "gamma": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["gamma"]),
                "learning_rate": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["learning_rate"]),
                "batch_size": int(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["batch_size"]),
                "hidden_layer": str(int(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["hidden_layer"])),
                "lr_strategy": str(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["lr_strategy"]),
                "min_lr": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["min_lr"]),
                "lr_decay": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["lr_decay"]),
            },
        )
        try:
            store["gamma"] = float(self.var_gamma.get())
        except (ValueError, tk.TclError):
            pass
        try:
            store["learning_rate"] = float(str(self.var_learning_rate.get()).strip())
        except (ValueError, tk.TclError):
            pass
        try:
            store["batch_size"] = int(self.var_batch_size.get())
        except (ValueError, tk.TclError):
            pass
        try:
            hidden_layer_text = str(self.var_hidden_layer.get()).strip()
            if hidden_layer_text:
                store["hidden_layer"] = hidden_layer_text
        except (ValueError, tk.TclError):
            pass
        store["lr_strategy"] = str(self.var_lr_strategy.get()).strip().lower() or "constant"
        try:
            store["min_lr"] = float(str(self.var_min_lr.get()).strip())
        except (ValueError, tk.TclError):
            pass
        try:
            store["lr_decay"] = float(self.var_lr_decay.get())
        except (ValueError, tk.TclError):
            pass

    def _on_shared_specific_value_changed(self, *_args: Any) -> None:
        if self._suspend_shared_cache_sync:
            return
        active_policy = self.var_policy.get().strip() or self.previous_policy
        self._cache_policy_shared_values(active_policy)

    def _load_policy_shared_values(self, policy: str) -> None:
        values = self.cached_shared_values.setdefault(
            policy,
            {
                "gamma": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["gamma"]),
                "learning_rate": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["learning_rate"]),
                "batch_size": int(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["batch_size"]),
                "hidden_layer": str(int(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["hidden_layer"])),
                "lr_strategy": str(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["lr_strategy"]),
                "min_lr": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["min_lr"]),
                "lr_decay": float(POLICY_SHARED_DEFAULTS.get(policy, SHARED_DEFAULTS)["lr_decay"]),
            },
        )
        self._suspend_shared_cache_sync = True
        try:
            self.var_gamma.set(float(values.get("gamma", SHARED_DEFAULTS["gamma"])))
            self.var_learning_rate.set(f"{float(values.get('learning_rate', SHARED_DEFAULTS['learning_rate'])):.1e}")
            self.var_batch_size.set(int(values.get("batch_size", SHARED_DEFAULTS["batch_size"])))
            self.var_hidden_layer.set(str(values.get("hidden_layer", SHARED_DEFAULTS["hidden_layer"])))
            self.var_lr_strategy.set(str(values.get("lr_strategy", SHARED_DEFAULTS["lr_strategy"])))
            self.var_min_lr.set(f"{float(values.get('min_lr', SHARED_DEFAULTS['min_lr'])):.1e}")
            self.var_lr_decay.set(float(values.get("lr_decay", SHARED_DEFAULTS["lr_decay"])))
        finally:
            self._suspend_shared_cache_sync = False

    def _sync_current_policy_specific_values(self) -> None:
        policy = self.previous_policy
        store = self.cached_specific_values.setdefault(policy, {})
        for key, var in self.current_specific_vars.items():
            store[key] = self._var_value(var)

    def _cache_specific_var_update(self, policy: str, key: str, var: Any) -> None:
        self.cached_specific_values.setdefault(policy, {})[key] = self._var_value(var)

    def _render_specific_fields(self, policy: str) -> None:
        for child in self.specific_dynamic_frame.winfo_children():
            child.destroy()
        self.current_specific_vars.clear()

        defaults = self.cached_specific_values.get(policy, dict(POLICY_DEFAULTS.get(policy, {})))
        fields = self.POLICY_SPECIFIC_FIELDS.get(policy, [])
        tooltip_map = {
            "n_steps": "More rollout steps can improve stability but increase update latency.",
            "gae_lambda": "Higher GAE lambda lowers bias but may increase variance.",
            "clip_range": "Smaller clipping makes PPO updates safer but slower.",
            "ent_coef": "Higher entropy bonus encourages broader exploration.",
            "buffer_size": "Larger replay buffers improve diversity but use more memory.",
            "learning_starts": "More warmup delays updates until buffer coverage is broader.",
            "tau": "Lower tau makes target updates smoother and more stable.",
            "train_freq": "Higher train frequency increases update cadence and compute load.",
            "gradient_steps": "More gradient steps per update can improve fit but cost more compute.",
            "policy_delay": "Delaying policy updates can stabilize TD3 critic learning.",
            "target_policy_noise": "Target noise improves robustness of TD3 target actions.",
            "target_noise_clip": "Noise clipping bounds target action perturbations for stability.",
        }
        for idx, (name, default) in enumerate(fields):
            row = idx // 2
            col = (idx % 2) * 2
            value = defaults.get(name, default)
            if isinstance(default, int):
                var = tk.IntVar(value=int(value))
            else:
                var = tk.DoubleVar(value=float(value))
            self.current_specific_vars[name] = var
            self.cached_specific_values.setdefault(policy, {})[name] = self._var_value(var)
            var.trace_add(
                "write",
                lambda *_args, current_policy=policy, current_name=name, current_var=var: self._cache_specific_var_update(
                    current_policy,
                    current_name,
                    current_var,
                ),
            )
            ttk.Label(self.specific_dynamic_frame, text=name).grid(row=row, column=col, sticky="w", padx=4, pady=2)
            specific_entry = ttk.Entry(self.specific_dynamic_frame, textvariable=var, width=9)
            specific_entry.grid(
                row=row, column=col + 1, sticky="ew", padx=4, pady=2
            )
            if name in tooltip_map:
                self._bind_tooltip(specific_entry, tooltip_map[name])

        self.specific_dynamic_frame.columnconfigure(1, weight=1)
        self.specific_dynamic_frame.columnconfigure(3, weight=1)

    def _toggle_advanced(self) -> None:
        if self.var_show_advanced.get():
            self.advanced_frame.grid()
        else:
            self.advanced_frame.grid_remove()

    def _make_env_config(self, render_mode: Optional[str] = None) -> Walker2DEnvConfig:
        return Walker2DEnvConfig(
            env_id="Walker2d-v5",
            render_mode=render_mode,
            forward_reward_weight=float(self.env_param_vars["forward_reward_weight"].get()),
            ctrl_cost_weight=float(self.env_param_vars["ctrl_cost_weight"].get()),
            healthy_reward=float(self.env_param_vars["healthy_reward"].get()),
            terminate_when_unhealthy=bool(self.env_param_vars["terminate_when_unhealthy"].get()),
            healthy_z_range=(
                float(self.env_param_vars["healthy_z_range_low"].get()),
                float(self.env_param_vars["healthy_z_range_high"].get()),
            ),
            healthy_angle_range=(
                float(self.env_param_vars["healthy_angle_range_low"].get()),
                float(self.env_param_vars["healthy_angle_range_high"].get()),
            ),
            reset_noise_scale=float(self.env_param_vars["reset_noise_scale"].get()),
            exclude_current_positions_from_observation=bool(
                self.env_param_vars["exclude_current_positions_from_observation"].get()
            ),
        )

    def _make_train_config(self, run_id: str) -> TrainConfig:
        shared = {
            "gamma": float(self.var_gamma.get()),
            "learning_rate": float(self.var_learning_rate.get()),
            "batch_size": int(self.var_batch_size.get()),
            "hidden_layer": str(self.var_hidden_layer.get()).strip(),
            "lr_strategy": str(self.var_lr_strategy.get()).strip().lower(),
            "min_lr": float(self.var_min_lr.get()),
            "lr_decay": float(self.var_lr_decay.get()),
        }
        specific = {key: self._var_value(var) for key, var in self.current_specific_vars.items()}
        return TrainConfig(
            policy_name=self.var_policy.get(),
            episodes=int(self.var_episodes.get()),
            max_steps=int(self.var_max_steps.get()),
            update_rate_episodes=max(1, int(self.var_update_rate.get())),
            frame_stride=max(1, int(self.var_frame_stride.get())),
            moving_average_values=max(1, int(self.var_moving_average_values.get())),
            deterministic_eval_every=10,
            rollout_full_capture_steps=max(1, int(self.var_rollout_capture_steps.get())),
            low_overhead_animation=bool(self.var_low_overhead_animation.get()),
            animation_on=bool(self.var_animation_on.get()),
            collect_transitions=True,
            device=self.var_device.get(),
            shared_params=shared,
            specific_params=specific,
            run_id=run_id,
        )

    def _run_single_episode(self) -> None:
        run_id = f"single_episode_{datetime.now().strftime('%H%M%S')}"
        env_cfg = self._make_env_config(render_mode="rgb_array")
        train_cfg = self._make_train_config(run_id=run_id)
        trainer = Walker2DTrainer(env_config=env_cfg, train_config=train_cfg)
        result = trainer.run_episode(
            model=None,
            deterministic=False,
            collect_transitions=True,
            max_steps=train_cfg.max_steps,
            render=self.var_animation_on.get(),
            frame_stride=train_cfg.frame_stride,
            rollout_full_capture_steps=train_cfg.rollout_full_capture_steps,
            emit_step_events=False,
        )
        self.var_steps_text.set(f"Steps: {result['steps']} / {train_cfg.max_steps}")
        self.var_status.set(
            f"Epsilon: n/a | LR: {float(self.var_learning_rate.get()):.1e} | Best reward: {result['reward']:.3f} | Render: {'on' if self.var_animation_on.get() else 'off'}"
        )
        if result["transitions"]:
            trainer._transition_buffer.extend(result["transitions"])
        self.last_completed_trainer = trainer
        if result["frames"]:
            self._enqueue_playback(result["frames"])

    def _start_training(self) -> None:
        if self.training_active and self.training_paused:
            self._cancel_active_training()

        if self.training_active:
            return

        self.session_id = uuid.uuid4().hex
        self._drain_event_queue()
        self.training_active = True
        self.training_paused = False
        self._set_control_styles()

        run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.var_episodes_text.set(f"Episodes: 0 / {int(self.var_episodes.get())}")
        self.var_steps_text.set(f"Steps: 0 / {int(self.var_max_steps.get())}")
        self.episodes_progress.configure(maximum=max(1, int(self.var_episodes.get())), value=0)

        env_cfg = self._make_env_config(render_mode=None)
        train_cfg = self._make_train_config(run_id=run_id)

        if self.var_compare_on.get() and self.compare_items:
            self._start_compare_training(base_env_cfg=env_cfg, base_train_cfg=train_cfg)
            return

        self.run_meta_snapshots[run_id] = {
            "policy": train_cfg.policy_name,
            "max_steps": train_cfg.max_steps,
            "gamma": train_cfg.shared_params["gamma"],
            "learning_rate": train_cfg.shared_params["learning_rate"],
            "hidden_layer": train_cfg.shared_params.get("hidden_layer"),
            "lr_strategy": train_cfg.shared_params.get("lr_strategy"),
            "min_lr": train_cfg.shared_params.get("min_lr"),
            "lr_decay": train_cfg.shared_params.get("lr_decay"),
            "env": {
                "healthy_reward": env_cfg.healthy_reward,
                "reset_noise_scale": env_cfg.reset_noise_scale,
            },
        }

        def sink(event_payload: Dict[str, Any]) -> None:
            event_payload["session_id"] = self.session_id
            self.event_queue.put(event_payload)

        trainer = Walker2DTrainer(env_config=env_cfg, train_config=train_cfg, event_sink=sink)
        with self.active_trainers_lock:
            self.active_trainers = {run_id: trainer}
        self.render_run_id = run_id
        self.compare_active = False

        self.training_thread = threading.Thread(target=self._train_worker, args=(trainer,), daemon=True)
        self.training_thread.start()

    def _train_worker(self, trainer: Walker2DTrainer) -> None:
        try:
            trainer.train()
        except Exception as exc:
            self.event_queue.put({"type": "error", "session_id": self.session_id, "message": str(exc)})

    def _start_compare_training(self, base_env_cfg: Walker2DEnvConfig, base_train_cfg: TrainConfig) -> None:
        run_specs = self._build_compare_run_configs(base_env_cfg, base_train_cfg)
        if not run_specs:
            self.training_active = False
            self._set_control_styles()
            return

        self.compare_active = True
        self.compare_expected_done = len(run_specs)
        self.compare_done_count = 0
        self.render_run_id = self._select_render_run_id([run_id for run_id, _, _ in run_specs])
        for run_id, _, train_cfg in run_specs:
            train_cfg.animation_on = bool(base_train_cfg.animation_on) and run_id == self.render_run_id

        with self.active_trainers_lock:
            self.active_trainers.clear()
            for run_id, env_cfg, train_cfg in run_specs:
                def sink_factory(current_session: str):
                    def sink(payload: Dict[str, Any]) -> None:
                        payload["session_id"] = current_session
                        self.event_queue.put(payload)
                    return sink

                trainer = Walker2DTrainer(
                    env_config=env_cfg,
                    train_config=train_cfg,
                    event_sink=sink_factory(self.session_id or ""),
                )
                self.active_trainers[run_id] = trainer

        self.training_thread = threading.Thread(target=self._run_compare_workers, daemon=True)
        self.training_thread.start()

    def _run_compare_workers(self) -> None:
        session = self.session_id
        with self.active_trainers_lock:
            trainers = list(self.active_trainers.values())
        try:
            max_workers = min(4, max(1, len(trainers)))
            thread_budgets = self._compute_compare_thread_budgets(max_workers)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(self._run_trainer_with_thread_budget, trainer, thread_budgets[idx % len(thread_budgets)])
                    for idx, trainer in enumerate(trainers)
                ]
                for future in as_completed(futures):
                    _ = future.result()
        except Exception as exc:
            self.event_queue.put({"type": "error", "session_id": session, "message": str(exc)})
        finally:
            self.event_queue.put({"type": "training_done", "session_id": session, "run_id": "compare", "status": "completed"})

    def _compute_compare_thread_budgets(self, worker_count: int) -> List[int]:
        workers = max(1, int(worker_count))
        cpu_cores = max(1, int(os.cpu_count() or 1))
        base = max(1, cpu_cores // workers)
        remainder = max(0, cpu_cores - (base * workers))
        budgets = [base for _ in range(workers)]
        for index in range(remainder):
            budgets[index % workers] += 1
        return budgets

    def _run_trainer_with_thread_budget(self, trainer: Walker2DTrainer, cpu_threads: int) -> Dict[str, Any]:
        requested_device = str(trainer.train_config.device).strip().upper()
        effective_cpu_run = not (requested_device == "GPU" and torch.cuda.is_available())
        if not effective_cpu_run:
            return trainer.train()

        previous_threads = torch.get_num_threads()
        safe_threads = max(1, int(cpu_threads))
        try:
            torch.set_num_threads(safe_threads)
            return trainer.train()
        finally:
            torch.set_num_threads(previous_threads)

    def _build_compare_run_configs(
        self,
        base_env_cfg: Walker2DEnvConfig,
        base_train_cfg: TrainConfig,
    ) -> List[Tuple[str, Walker2DEnvConfig, TrainConfig]]:
        keys = list(self.compare_items.keys())
        values_per_key = [self.compare_items[key] for key in keys]
        combinations = list(itertools.product(*values_per_key))
        run_specs: List[Tuple[str, Walker2DEnvConfig, TrainConfig]] = []

        for index, combo in enumerate(combinations, start=1):
            compare_map = {key: self._convert_compare_value(key, value) for key, value in zip(keys, combo)}
            policy_name = str(compare_map.get("Policy", base_train_cfg.policy_name))

            shared_params = dict(base_train_cfg.shared_params)
            if "Policy" in compare_map:
                specific_params = dict(POLICY_DEFAULTS.get(policy_name, {}))
            else:
                specific_params = dict(base_train_cfg.specific_params)
            for key, value in compare_map.items():
                if key in {"Policy"}:
                    continue
                if key in {"gamma", "learning_rate", "batch_size", "hidden_layer", "lr_strategy", "min_lr", "lr_decay"}:
                    shared_params[key] = value
                elif key in specific_params or key in {f[0] for f in self.POLICY_SPECIFIC_FIELDS.get(policy_name, [])}:
                    specific_params[key] = value

            compare_episodes = int(compare_map.get("episodes", base_train_cfg.episodes))
            compare_max_steps = int(compare_map.get("max_steps", base_train_cfg.max_steps))

            run_id = f"cmp_{index}_{datetime.now().strftime('%H%M%S%f')}"
            env_cfg = Walker2DEnvConfig(**base_env_cfg.__dict__)
            train_cfg = TrainConfig(
                policy_name=policy_name,
                episodes=compare_episodes,
                max_steps=compare_max_steps,
                update_rate_episodes=base_train_cfg.update_rate_episodes,
                frame_stride=base_train_cfg.frame_stride,
                moving_average_values=base_train_cfg.moving_average_values,
                deterministic_eval_every=base_train_cfg.deterministic_eval_every,
                rollout_full_capture_steps=base_train_cfg.rollout_full_capture_steps,
                low_overhead_animation=base_train_cfg.low_overhead_animation,
                animation_on=False,
                collect_transitions=base_train_cfg.collect_transitions,
                device=base_train_cfg.device,
                shared_params=shared_params,
                specific_params=specific_params,
                run_id=run_id,
            )
            self.run_meta_snapshots[run_id] = {
                "policy": policy_name,
                "max_steps": train_cfg.max_steps,
                "gamma": train_cfg.shared_params.get("gamma"),
                "learning_rate": train_cfg.shared_params.get("learning_rate"),
                "hidden_layer": train_cfg.shared_params.get("hidden_layer"),
                "lr_strategy": train_cfg.shared_params.get("lr_strategy"),
                "min_lr": train_cfg.shared_params.get("min_lr"),
                "lr_decay": train_cfg.shared_params.get("lr_decay"),
                "env": {
                    "healthy_reward": env_cfg.healthy_reward,
                    "reset_noise_scale": env_cfg.reset_noise_scale,
                },
                "compare": compare_map,
            }
            run_specs.append((run_id, env_cfg, train_cfg))

        return run_specs

    def _convert_compare_value(self, key: str, raw_value: str) -> Any:
        value = raw_value.strip()
        if key == "Policy":
            return value
        if key in {
            "episodes",
            "max_steps",
            "batch_size",
            "hidden_layer",
            "n_steps",
            "buffer_size",
            "learning_starts",
            "train_freq",
            "gradient_steps",
            "policy_delay",
        }:
            return int(float(value))
        try:
            return float(value)
        except ValueError:
            return value

    def _select_render_run_id(self, run_ids: List[str]) -> Optional[str]:
        if not run_ids:
            return None
        selected_policy = self.var_policy.get()
        for run_id in run_ids:
            if self.run_meta_snapshots.get(run_id, {}).get("policy") == selected_policy:
                return run_id
        return run_ids[0]

    def _toggle_pause(self) -> None:
        if not self.training_active:
            return
        self.training_paused = not self.training_paused
        with self.active_trainers_lock:
            for trainer in self.active_trainers.values():
                trainer.set_pause(self.training_paused)
        self._set_control_styles()

    def _cancel_active_training(self) -> None:
        with self.active_trainers_lock:
            trainers = list(self.active_trainers.values())
        for trainer in trainers:
            trainer.set_pause(False)
            trainer.cancel()
        self.training_active = False
        self.training_paused = False
        with self.active_trainers_lock:
            self.active_trainers.clear()
        self.compare_active = False
        self.compare_expected_done = 0
        self.compare_done_count = 0
        self.render_run_id = None
        self._set_control_styles()

    def _reset_all(self) -> None:
        self._cancel_active_training()
        self.var_animation_on.set(True)
        self.var_animation_fps.set(30)
        self.var_update_rate.set(1)
        self.var_frame_stride.set(2)
        self.var_max_steps.set(1000)
        self.var_episodes.set(1000)
        self.var_policy.set("PPO")
        self.cached_shared_values = {
            name: {
                "gamma": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["gamma"]),
                "learning_rate": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["learning_rate"]),
                "batch_size": int(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["batch_size"]),
                "hidden_layer": str(int(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["hidden_layer"])),
                "lr_strategy": str(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["lr_strategy"]),
                "min_lr": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["min_lr"]),
                "lr_decay": float(POLICY_SHARED_DEFAULTS.get(name, SHARED_DEFAULTS)["lr_decay"]),
            }
            for name in self.POLICY_SPECIFIC_FIELDS
        }
        self._load_policy_shared_values("PPO")
        self.previous_policy = "PPO"
        self.var_moving_average_values.set(20)
        self.var_show_advanced.set(False)
        self.var_rollout_capture_steps.set(120)
        self.var_low_overhead_animation.set(False)
        self.var_device.set("CPU")
        self._render_specific_fields("PPO")
        self._toggle_advanced()

    def _clear_plot(self) -> None:
        self.run_plots.clear()
        self._legend_entries.clear()
        self._legend_items.clear()
        self._legend = None
        self._legend_scroll_index = 0
        self._legend_hover_index = None
        self._hidden_run_ids.clear()
        self._line_run_id_map.clear()
        self._line_group_map.clear()
        self.ax.clear()
        self._configure_plot()
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.canvas.draw_idle()

    def _save_sampling_csv(self) -> None:
        with self.active_trainers_lock:
            trainers = list(self.active_trainers.values())
        trainer = trainers[0] if trainers else self.last_completed_trainer
        if trainer is None:
            messagebox.showinfo("Walker2D", "No sampled transitions available.")
            return
        output_path = trainer.export_transitions_csv(Path("results_csv"), filename_prefix="walker2d_samples")
        if output_path is None:
            messagebox.showinfo("Walker2D", "No transitions collected yet.")
        else:
            messagebox.showinfo("Walker2D", f"Saved CSV: {output_path}")

    def _save_plot_png(self) -> None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy = self.var_policy.get()
        max_steps = int(self.var_max_steps.get())
        gamma = float(self.var_gamma.get())
        learning_rate = float(self.var_learning_rate.get())
        file_path = Path("plots") / f"walker2d_{policy}_steps-{max_steps}_gamma-{gamma:g}_lr-{learning_rate:.0e}_{ts}.png"
        self.figure.savefig(file_path, dpi=120)
        messagebox.showinfo("Walker2D", f"Saved plot: {file_path}")

    def _set_control_styles(self) -> None:
        self.btn_train.configure(style="Accent.TButton" if self.training_active and not self.training_paused else "Neutral.TButton")
        self.btn_pause.configure(style="Pause.TButton" if self.training_paused else "Neutral.TButton")
        self.btn_pause.configure(text="Run" if self.training_paused else "Pause")
        self.btn_pause.configure(state=tk.NORMAL if self.training_active else tk.DISABLED)

    def _poll_worker_events(self) -> None:
        while True:
            try:
                payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            payload_session = payload.get("session_id")
            if self.session_id is not None and payload_session != self.session_id:
                continue

            event_type = payload.get("type")
            if event_type == "step":
                self._handle_step_event(payload)
            elif event_type == "episode":
                self._handle_episode_event(payload)
            elif event_type == "training_done":
                self._handle_training_done(payload)
            elif event_type == "error":
                self._handle_error(payload)

        self.after(100, self._poll_worker_events)

    def _handle_episode_event(self, payload: Dict[str, Any]) -> None:
        episode = int(payload["episode"])
        total_episodes = int(payload["episodes"])
        steps = int(payload["steps"])
        reward = float(payload["reward"])
        moving_average = float(payload["moving_average"])
        run_id = str(payload.get("run_id", "single"))
        eval_points = list(payload.get("eval_points", []))

        self.var_episodes_text.set(f"Episodes: {episode} / {total_episodes}")
        self.episodes_progress.configure(maximum=max(1, total_episodes), value=episode)
        if not self.compare_active or run_id == self.render_run_id:
            self.var_status.set(
                f"Epsilon: {payload.get('epsilon', 'n/a')} | LR: {float(payload.get('lr', 0.0)):.1e} | Best reward: {float(payload.get('best_reward', reward)):.3f} | Render: {payload.get('render_state', 'idle')}"
            )

        frames = payload.get("frames") or []
        if self.var_animation_on.get() and frames and (not self.compare_active or run_id == self.render_run_id):
            self._enqueue_playback(frames, max_steps=max(1, int(self.var_max_steps.get())), steps=steps)

        self._append_plot_data(run_id, episode, reward, moving_average, eval_points)

    def _handle_step_event(self, payload: Dict[str, Any]) -> None:
        _ = payload

    def _handle_training_done(self, payload: Dict[str, Any]) -> None:
        run_id = payload.get("run_id")
        with self.active_trainers_lock:
            if run_id and run_id in self.active_trainers:
                self.last_completed_trainer = self.active_trainers.get(run_id)

        if self.compare_active:
            if run_id == "compare":
                pass
            else:
                self.compare_done_count += 1
                if self.compare_done_count < self.compare_expected_done:
                    return

        self.training_active = False
        self.training_paused = False
        with self.active_trainers_lock:
            if self.active_trainers and self.last_completed_trainer is None:
                self.last_completed_trainer = next(iter(self.active_trainers.values()))
            self.active_trainers.clear()
        self.compare_active = False
        self.compare_expected_done = 0
        self.compare_done_count = 0
        self.render_run_id = None
        self._set_control_styles()

    def _handle_error(self, payload: Dict[str, Any]) -> None:
        self.training_active = False
        self.training_paused = False
        with self.active_trainers_lock:
            self.active_trainers.clear()
        self.compare_active = False
        self.compare_expected_done = 0
        self.compare_done_count = 0
        self.render_run_id = None
        self._set_control_styles()
        messagebox.showerror("Walker2D Error", payload.get("message", "Unknown error"))

    def _append_plot_data(
        self,
        run_id: str,
        episode: int,
        reward: float,
        moving_average: float,
        eval_points: List[Tuple[int, float]],
    ) -> None:
        if run_id not in self.run_plots:
            cycle = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["#1f77b4"])
            color = cycle[len(self.run_plots) % len(cycle)]
            snapshot = self.run_meta_snapshots.get(run_id, {})
            compare_text = self._build_compare_legend_suffix(snapshot)
            base_label = (
                f"{snapshot.get('policy', self.var_policy.get())} | steps={snapshot.get('max_steps', self.var_max_steps.get())} | gamma={snapshot.get('gamma', self.var_gamma.get())}\n"
                f"epsilon=n/a | epsilon_decay=n/a | epsilon_min=n/a\n"
                f"LR={snapshot.get('learning_rate', self.var_learning_rate.get())} | LR strategy={snapshot.get('lr_strategy', self.var_lr_strategy.get())} | LR decay={snapshot.get('lr_decay', self.var_lr_decay.get())} | min_lr={snapshot.get('min_lr', self.var_min_lr.get())}\n"
                f"healthy_reward={snapshot.get('env', {}).get('healthy_reward', self.env_param_vars['healthy_reward'].get())} | reset_noise_scale={snapshot.get('env', {}).get('reset_noise_scale', self.env_param_vars['reset_noise_scale'].get())}"
                f"{compare_text}"
            )
            self.run_plots[run_id] = RunPlotState([], [], [], [], [], [], color, base_label)

        data = self.run_plots[run_id]
        data.rewards_x.append(episode)
        data.rewards_y.append(reward)
        data.moving_average_x.append(episode)
        data.moving_average_y.append(moving_average)

        if eval_points:
            latest_ep, latest_value = eval_points[-1]
            if not data.eval_x or data.eval_x[-1] != latest_ep:
                data.eval_x.append(int(latest_ep))
                data.eval_y.append(float(latest_value))

        self.ax.clear()
        self._configure_plot()
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self._line_run_id_map.clear()
        self._line_group_map.clear()

        legend_entries: List[Tuple[Any, str, List[Any]]] = []
        for current_run_id, state in self.run_plots.items():
            reward_line, = self.ax.plot(
                state.rewards_x,
                state.rewards_y,
                color=state.color,
                alpha=0.60,
                linewidth=1.5,
                label=state.label,
            )
            ma_line, = self.ax.plot(
                state.moving_average_x,
                state.moving_average_y,
                color=state.color,
                alpha=1.0,
                linewidth=3,
                linestyle="--",
                label="moving average",
            )
            run_group = [reward_line, ma_line]
            legend_entries.append((reward_line, state.label, [reward_line]))
            legend_entries.append((ma_line, "moving average", [ma_line]))
            if state.eval_x:
                eval_line, = self.ax.plot(
                    state.eval_x,
                    state.eval_y,
                    color=state.color,
                    alpha=1.0,
                    linewidth=3,
                    linestyle=":",
                    marker="o",
                    markersize=3,
                    label="evaluation rollout",
                )
                run_group.append(eval_line)
                legend_entries.append((eval_line, "evaluation rollout", [eval_line]))

            is_hidden = current_run_id in self._hidden_run_ids
            for artist in run_group:
                artist.set_visible(not is_hidden)

            self._line_run_id_map[reward_line] = current_run_id
            self._line_group_map[reward_line] = run_group

        self._legend_entries = legend_entries
        self._rebuild_interactive_legend()
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
            labelcolor=self.PALETTE["text"],
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
            legend_text.set_color(self.PALETTE["accent"])
        else:
            legend_text.set_color(self.PALETTE["text"])
        legend_handle.set_alpha(base_alpha)
        legend_text.set_alpha(base_alpha)

    def _event_hits_legend(self, event: Any) -> bool:
        for legend_handle, legend_text, _ in self._legend_items:
            handle_hit = legend_handle.contains(event)[0]
            text_hit = legend_text.contains(event)[0]
            if handle_hit or text_hit:
                return True
        return False

    def _on_plot_pick(self, event: Any) -> None:
        for idx, (legend_handle, legend_text, artists) in enumerate(self._legend_items):
            if event.artist in (legend_handle, legend_text):
                run_id = self._line_run_id_map.get(artists[0]) if artists else None
                targets = self._line_group_map.get(artists[0], artists)
                make_visible = not any(artist.get_visible() for artist in targets)
                for artist in targets:
                    artist.set_visible(make_visible)
                if run_id is not None:
                    if make_visible:
                        self._hidden_run_ids.discard(run_id)
                    else:
                        self._hidden_run_ids.add(run_id)
                self._set_legend_item_visual(idx)
                self.canvas.draw_idle()
                return

    def _on_plot_motion(self, event: Any) -> None:
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

    def _on_plot_scroll(self, event: Any) -> None:
        if self._legend is None or not self._event_hits_legend(event):
            return
        if len(self._legend_entries) <= self._legend_max_visible:
            return

        max_start = max(0, len(self._legend_entries) - self._legend_max_visible)
        old_index = self._legend_scroll_index
        if getattr(event, "button", "") == "up":
            self._legend_scroll_index = max(0, self._legend_scroll_index - 1)
        else:
            self._legend_scroll_index = min(max_start, self._legend_scroll_index + 1)

        if self._legend_scroll_index != old_index:
            self._legend_hover_index = None
            self._rebuild_interactive_legend()
            self.canvas.get_tk_widget().configure(cursor="")
            self.canvas.draw_idle()

    def _enqueue_playback(self, frames: List[np.ndarray], max_steps: int = 1, steps: int = 0) -> None:
        if not frames:
            return
        if self._playback_active:
            self._playback_pending = frames
            return
        self._playback_active = True
        self.steps_progress.configure(maximum=max_steps, value=0)
        self.var_steps_text.set(f"Steps: 0 / {max_steps}")
        self._playback_frames(frames, 0, max_steps, steps)

    def _playback_frames(self, frames: List[np.ndarray], index: int, max_steps: int, steps: int) -> None:
        if not self.var_animation_on.get():
            self._playback_active = False
            return
        if index >= len(frames):
            self.steps_progress["value"] = min(max_steps, steps)
            self.var_steps_text.set(f"Steps: {min(max_steps, steps)} / {max_steps}")
            self._playback_active = False
            if self._playback_pending is not None:
                next_frames = self._playback_pending
                self._playback_pending = None
                self._enqueue_playback(next_frames, max_steps=max_steps, steps=steps)
            return

        frame = frames[index]
        self._draw_frame(frame)
        progress = int((index + 1) / max(1, len(frames)) * min(max_steps, steps))
        self.steps_progress["value"] = progress
        self.var_steps_text.set(f"Steps: {progress} / {max_steps}")

        fps = max(1, int(self.var_animation_fps.get()))
        delay = int(1000 / fps)
        self.after(delay, lambda: self._playback_frames(frames, index + 1, max_steps, steps))

    def _draw_frame(self, frame: np.ndarray) -> None:
        self.last_render_image = frame
        if Image is None or ImageTk is None:
            return

        image = Image.fromarray(frame)
        canvas_w, canvas_h = self.last_canvas_size
        image_ratio = image.width / max(1, image.height)
        canvas_ratio = canvas_w / max(1, canvas_h)

        if image_ratio > canvas_ratio:
            new_w = canvas_w
            new_h = int(canvas_w / image_ratio)
        else:
            new_h = canvas_h
            new_w = int(canvas_h * image_ratio)

        resized = image.resize((max(1, new_w), max(1, new_h)), resample=Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(resized)
        self.render_canvas.delete("all")
        self.render_canvas.create_image(canvas_w // 2, canvas_h // 2, image=photo, anchor="center")
        self.render_canvas.image = photo

    def _drain_event_queue(self) -> None:
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    @staticmethod
    def _var_value(var: Any) -> Any:
        try:
            return var.get()
        except Exception:
            return var

    def _disable_combobox_wheel(self, combo: ttk.Combobox) -> None:
        combo.bind("<MouseWheel>", lambda e: "break")

    def _on_close(self) -> None:
        self._cancel_active_training()
        self.master.destroy()

    def _build_compare_legend_suffix(self, snapshot: Dict[str, Any]) -> str:
        compare = snapshot.get("compare", {}) or {}
        if not compare:
            return ""

        represented_keys = {
            "Policy",
            "policy",
            "max_steps",
            "steps",
            "gamma",
            "learning_rate",
            "hidden_layer",
            "lr_strategy",
            "min_lr",
            "lr_decay",
            "LR",
            "LR strategy",
            "LR decay",
            "epsilon",
            "epsilon_decay",
            "epsilon_min",
            "healthy_reward",
            "reset_noise_scale",
        }

        extras: List[str] = []
        for key, value in compare.items():
            if key in represented_keys:
                continue
            extras.append(f"{key}={value}")

        if not extras:
            return ""
        return f"\ncompare: {' | '.join(extras)}"

    def _bind_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(_SimpleTooltip(widget, text))

    def _refresh_compare_parameter_options(self) -> None:
        current_policy = self.var_policy.get()
        specific_keys = [name for name, _ in self.POLICY_SPECIFIC_FIELDS.get(current_policy, [])]
        values = [
            "Policy",
            "max_steps",
            "episodes",
            "gamma",
            "learning_rate",
            "batch_size",
            "hidden_layer",
            "lr_strategy",
            "min_lr",
            "lr_decay",
            *specific_keys,
        ]
        unique_values = list(dict.fromkeys(values))
        selected = self.var_compare_parameter.get()
        self.compare_param_combo.configure(values=unique_values)
        if selected in unique_values:
            self.var_compare_parameter.set(selected)
        else:
            self.var_compare_parameter.set(unique_values[0])
        self._update_compare_hint()

    def _get_compare_categorical_options(self, key: str) -> List[str]:
        categorical = {
            "Policy": ["PPO", "SAC", "TD3"],
            "Activation": ["ReLU", "Tanh", "Linear"],
            "LR strategy": ["constant", "linear", "exponential"],
            "lr_strategy": ["constant", "linear", "exponential"],
        }
        return categorical.get(key, [])

    def _update_compare_hint(self) -> None:
        if not hasattr(self, "compare_hint"):
            return
        key = self.var_compare_parameter.get().strip()
        options = self._get_compare_categorical_options(key)
        typed = self.var_compare_values.get().strip().lower()
        suggestion = ""
        if options and typed:
            for option in options:
                if option.lower().startswith(typed):
                    suggestion = option
                    break
        self.compare_hint.configure(text=f"Tab -> {suggestion}" if suggestion else "")
