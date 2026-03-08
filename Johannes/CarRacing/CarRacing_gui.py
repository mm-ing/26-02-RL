from __future__ import annotations

import itertools
import os
import queue
import threading
import time
from collections import deque
import tkinter as tk
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from tkinter import ttk
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
from matplotlib.lines import Line2D

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None

from CarRacing_logic import (
    DEFAULT_ENV_CONFIG,
    DEFAULT_GENERAL,
    DEFAULT_SPECIFIC,
    CarRacingEnvWrapper,
    CarRacingTrainer,
    EnvConfig,
    TrainConfig,
)


class _Tooltip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        if not self.text:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + 20
        self.tip_window = tk.Toplevel(self.widget)
        self.tip_window.wm_overrideredirect(True)
        self.tip_window.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip_window,
            text=self.text,
            bg="#2d2d30",
            fg="#e6e6e6",
            relief="solid",
            borderwidth=1,
            padx=6,
            pady=3,
            justify="left",
        )
        label.pack()

    def _hide(self, _event=None):
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class CarRacingGUI(ttk.Frame):
    input_width = 12
    shared_specific_fields = [
        "gamma",
        "learning_rate",
        "batch_size",
        "hidden_layer",
        "activation",
        "lr_strategy",
        "min_lr",
        "lr_decay",
    ]

    policy_specific_fields = {
        "SAC": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps"],
        "TD3": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps", "policy_delay"],
        "PPO": ["n_steps", "ent_coef", "gae_lambda", "clip_range"],
        "DDQN": [
            "buffer_size",
            "learning_starts",
            "target_update_interval",
            "train_freq",
            "gradient_steps",
            "exploration_fraction",
            "exploration_final_eps",
        ],
        "QR-DQN": [
            "buffer_size",
            "learning_starts",
            "target_update_interval",
            "train_freq",
            "gradient_steps",
            "exploration_fraction",
            "exploration_final_eps",
            "n_quantiles",
        ],
    }

    # Strict visual mode: consume one event per pump so episode progress is
    # rendered one-by-one instead of jumping over batches.
    EVENT_PUMP_INTERVAL_MS = 16
    EVENT_PUMP_MAX_BATCH = 1

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.pack(fill="both", expand=True)

        self._setup_style()

        self.env_wrapper = CarRacingEnvWrapper(EnvConfig(**DEFAULT_ENV_CONFIG))
        self.trainer = CarRacingTrainer(self.env_wrapper, event_sink=self._enqueue_event)
        self.worker_lock = threading.Lock()
        self.active_workers: Dict[str, threading.Thread] = {}
        self.active_trainers: Dict[str, CarRacingTrainer] = {}
        self.current_session_id = "idle"
        self.event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        self.run_history: Dict[str, Dict[str, Any]] = {}
        self.run_visibility: Dict[str, bool] = {}
        self.run_meta: Dict[str, Dict[str, Any]] = {}
        self.run_compare_meta: Dict[str, Dict[str, Any]] = {}
        self.run_label_cache: Dict[str, str] = {}
        self.playback_active = False
        self.pending_playback = deque(maxlen=2)
        self.current_frames: List[Any] = []
        self.current_frame_idx = 0
        self.current_playback_steps = 0
        self.batch_compare_active = False
        self.render_photo = None
        self.legend = None
        self.legend_scroll_center = 0.5
        self._legend_item_to_run: Dict[Any, str] = {}
        self._hovered_legend_item = None

        self.param_vars: Dict[str, tk.Variable] = {}
        self.param_widgets: Dict[str, tk.Widget] = {}
        self.active_policy = "SAC"
        self.policy_param_cache = {policy: dict(values) for policy, values in DEFAULT_SPECIFIC.items()}
        self._build_ui()

        self.after(80, self._pump_events)
        self.master.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        style = ttk.Style(self.master)
        themes = style.theme_names()
        if "clam" in themes:
            style.theme_use("clam")
        elif "vista" in themes:
            style.theme_use("vista")

        palette = {
            "bg": "#1e1e1e",
            "panel": "#252526",
            "input": "#2d2d30",
            "fg": "#e6e6e6",
            "muted": "#d0d0d0",
            "accent": "#0e639c",
        }
        self.master.configure(bg=palette["bg"])

        style.configure("TFrame", background=palette["bg"])
        style.configure("Panel.TLabelframe", background=palette["panel"], foreground=palette["fg"])
        style.configure("Panel.TLabelframe.Label", background=palette["panel"], foreground=palette["fg"], font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background=palette["bg"], foreground=palette["fg"], font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=palette["panel"], foreground=palette["fg"], font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground=palette["input"], foreground=palette["fg"], insertcolor="white")
        style.configure("TCombobox", fieldbackground=palette["input"], background=palette["input"], foreground=palette["fg"])
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", palette["input"])],
            foreground=[("readonly", palette["fg"])],
            selectbackground=[("readonly", palette["accent"])],
            selectforeground=[("readonly", "white")],
        )
        style.configure("Neutral.TButton", font=("Segoe UI", 10, "bold"), background="#3a3d41", foreground=palette["fg"])
        style.map("Neutral.TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")])
        style.configure("Train.TButton", font=("Segoe UI", 10, "bold"), background="#0e639c", foreground="white")
        style.map("Train.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.configure("Pause.TButton", font=("Segoe UI", 10, "bold"), background="#a66a00", foreground="white")
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])
        style.configure("Horizontal.TProgressbar", troughcolor="#343434", background="#0e639c")

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=3)

        self._build_environment_panel()
        self._build_parameters_panel()
        self._build_controls_row()
        self._build_current_run_panel()
        self._build_plot_panel()

    def _build_environment_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Environment", style="Panel.TLabelframe")
        frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 6))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(frame, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.render_canvas.bind("<Configure>", lambda _e: self._draw_last_frame())

    def _build_parameters_panel(self) -> None:
        outer = ttk.LabelFrame(self, text="Parameters", style="Panel.TLabelframe")
        outer.grid(row=0, column=1, sticky="nsew", padx=(6, 10), pady=(10, 6))
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(0, weight=1)

        self.params_canvas = tk.Canvas(outer, bg="#252526", highlightthickness=0)
        self.params_canvas.grid(row=0, column=0, sticky="nsew", padx=(6, 0), pady=6)
        self.params_scroll = ttk.Scrollbar(outer, orient="vertical", command=self.params_canvas.yview)
        self.params_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        self.params_canvas.configure(yscrollcommand=self.params_scroll.set)

        self.params_frame = ttk.Frame(self.params_canvas)
        self.params_window = self.params_canvas.create_window((0, 0), window=self.params_frame, anchor="nw")

        self.params_frame.bind("<Configure>", self._on_params_configure)
        self.params_canvas.bind("<Configure>", self._on_params_canvas_configure)
        self.params_canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.params_canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        self._build_group_environment(self.params_frame)
        self._build_group_compare(self.params_frame)
        self._build_group_general(self.params_frame)
        self._build_group_specific(self.params_frame)
        self._build_group_liveplot(self.params_frame)

    def _build_group_environment(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Environment", style="Panel.TLabelframe")
        group.pack(fill="x", padx=6, pady=6)
        for col in range(2):
            group.columnconfigure(col, weight=1)

        self._add_labeled_var(group, "Animation on", tk.BooleanVar(value=True), 0, 0)
        self.param_vars["Animation on"].trace_add("write", self._on_animation_runtime_toggle)
        self._add_labeled_var(group, "Animation FPS", tk.IntVar(value=30), 0, 1)
        self._add_labeled_var(group, "Update rate (episodes)", tk.IntVar(value=1), 1, 0)
        self._add_labeled_var(group, "Frame stride", tk.IntVar(value=2), 1, 1)

        update_btn = ttk.Button(group, text="Update", style="Neutral.TButton", command=self._apply_environment_update)
        update_btn.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 6))

        self._add_labeled_var(group, "lap_complete_percent", tk.DoubleVar(value=0.95), 3, 0)
        self._add_labeled_var(group, "domain_randomize", tk.BooleanVar(value=False), 3, 1)
        self._add_labeled_var(group, "continuous", tk.BooleanVar(value=True), 4, 0)

    def _build_group_compare(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Compare", style="Panel.TLabelframe")
        group.pack(fill="x", padx=6, pady=6)
        group.columnconfigure(0, weight=1)
        group.columnconfigure(1, weight=1)

        top = ttk.Frame(group)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)
        top.columnconfigure(3, weight=0)

        self.param_vars["compare_on"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Compare on", variable=self.param_vars["compare_on"], command=self._on_compare_toggle).grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Clear", style="Neutral.TButton", command=self._clear_compare).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Add", style="Neutral.TButton", command=self._add_compare).grid(row=0, column=2)
        self.param_vars["Batch compare mode"] = tk.BooleanVar(value=False)
        ttk.Checkbutton(top, text="Batch compare", variable=self.param_vars["Batch compare mode"]).grid(row=0, column=3, sticky="e", padx=(8, 0))

        self.param_vars["compare_parameter"] = tk.StringVar(value="Policy")
        self.param_vars["compare_values"] = tk.StringVar(value="")
        self.compare_dropdown = ttk.Combobox(group, state="readonly", textvariable=self.param_vars["compare_parameter"])
        self.compare_dropdown.configure(width=self.input_width)
        self.compare_dropdown.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        self.compare_dropdown.configure(values=self._compare_fields())
        self.compare_dropdown.bind("<MouseWheel>", lambda e: "break")

        compare_values_entry = ttk.Entry(group, textvariable=self.param_vars["compare_values"], width=self.input_width)
        compare_values_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=4)
        compare_values_entry.bind("<Return>", lambda _e: self._add_compare())

        self.compare_hint = ttk.Label(group, text="", style="Panel.TLabel")
        self.compare_hint.grid(row=2, column=1, sticky="w", padx=6, pady=(0, 4))
        compare_values_entry.bind("<KeyRelease>", self._on_compare_key_release)
        compare_values_entry.bind("<Tab>", self._on_compare_tab_complete)

        self.compare_summary = tk.Text(group, height=3, bg="#2d2d30", fg="#e6e6e6", insertbackground="white", relief="flat")
        self.compare_summary.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 6))
        self.compare_summary.configure(state="disabled")
        self.compare_items: Dict[str, List[str]] = {}

    def _build_group_general(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="General", style="Panel.TLabelframe")
        group.pack(fill="x", padx=6, pady=6)
        group.columnconfigure(0, weight=1)
        group.columnconfigure(1, weight=1)
        self._add_labeled_var(group, "Max steps", tk.IntVar(value=DEFAULT_GENERAL["max_steps"]), 0, 0)
        self._add_labeled_var(group, "Episodes", tk.IntVar(value=DEFAULT_GENERAL["episodes"]), 0, 1)

    def _build_group_specific(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Specific", style="Panel.TLabelframe")
        group.pack(fill="x", padx=6, pady=6)
        group.columnconfigure(0, weight=1)

        top = ttk.Frame(group)
        top.pack(fill="x", padx=6, pady=4)
        ttk.Label(top, text="Policy", style="Panel.TLabel").pack(side="left")
        self.param_vars["Policy"] = tk.StringVar(value="SAC")
        self.policy_dropdown = ttk.Combobox(top, state="readonly", textvariable=self.param_vars["Policy"], values=list(DEFAULT_SPECIFIC.keys()))
        self.policy_dropdown.configure(width=self.input_width)
        self.policy_dropdown.pack(side="right", fill="x", expand=True)
        self.policy_dropdown.bind("<<ComboboxSelected>>", self._on_policy_changed)
        self.policy_dropdown.bind("<MouseWheel>", lambda e: "break")

        self.specific_grid = ttk.Frame(group)
        self.specific_grid.pack(fill="x", padx=6, pady=(0, 6))
        self.specific_grid.columnconfigure(0, weight=1)
        self.specific_grid.columnconfigure(1, weight=1)

        self.specific_entries: Dict[str, tk.Variable] = {}
        self._build_specific_rows("SAC")

    def _build_group_liveplot(self, parent: ttk.Frame) -> None:
        group = ttk.LabelFrame(parent, text="Live Plot", style="Panel.TLabelframe")
        group.pack(fill="x", padx=6, pady=6)
        group.columnconfigure(0, weight=1)
        group.columnconfigure(1, weight=1)
        self._add_labeled_var(group, "Moving average values", tk.IntVar(value=20), 0, 0)
        self._add_labeled_var(group, "Performance mode", tk.BooleanVar(value=False), 0, 1)
        self._add_labeled_var(group, "Plot redraw every (episodes)", tk.IntVar(value=5), 1, 0)
        self._add_labeled_var(group, "Status update every (episodes)", tk.IntVar(value=5), 1, 1)

    def _build_controls_row(self) -> None:
        row = ttk.Frame(self)
        row.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        for i in range(8):
            row.columnconfigure(i, weight=1)

        self.btn_single = ttk.Button(row, text="Run single episode", style="Neutral.TButton", command=self._run_single_episode)
        self.btn_train = ttk.Button(row, text="Train and Run", style="Neutral.TButton", command=self._start_train)
        self.btn_pause = ttk.Button(row, text="Pause", style="Neutral.TButton", command=self._toggle_pause)
        self.btn_reset = ttk.Button(row, text="Reset All", style="Neutral.TButton", command=self._reset_all)
        self.btn_clear = ttk.Button(row, text="Clear Plot", style="Neutral.TButton", command=self._clear_plot)
        self.btn_save_csv = ttk.Button(row, text="Save samplings CSV", style="Neutral.TButton", command=self._save_csv)
        self.btn_save_png = ttk.Button(row, text="Save Plot PNG", style="Neutral.TButton", command=self._save_png)
        self.param_vars["Device"] = tk.StringVar(value="GPU")
        self.device_dropdown = ttk.Combobox(row, state="readonly", textvariable=self.param_vars["Device"], values=["CPU", "GPU"])
        self.device_dropdown.configure(width=self.input_width)
        self.device_dropdown.bind("<MouseWheel>", lambda e: "break")

        controls = [
            self.btn_single,
            self.btn_train,
            self.btn_pause,
            self.btn_reset,
            self.btn_clear,
            self.btn_save_csv,
            self.btn_save_png,
            self.device_dropdown,
        ]
        for idx, control in enumerate(controls):
            control.grid(row=0, column=idx, sticky="ew", padx=4)

    def _build_current_run_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="Current Run", style="Panel.TLabelframe")
        panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=6)
        panel.columnconfigure(1, weight=1)

        ttk.Label(panel, text="Steps", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=6)
        self.steps_progress = ttk.Progressbar(panel, orient="horizontal", mode="determinate")
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=6, pady=4)

        ttk.Label(panel, text="Episodes", style="Panel.TLabel").grid(row=1, column=0, sticky="w", padx=6)
        self.episodes_progress = ttk.Progressbar(panel, orient="horizontal", mode="determinate")
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=6, pady=4)

        self.status_var = tk.StringVar(value="Epsilon: - | LR: - | Best reward: - | Render: idle")
        ttk.Label(panel, textvariable=self.status_var, style="Panel.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 4))

    def _build_plot_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="Live Plot", style="Panel.TLabelframe")
        panel.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(6, 10))
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor("#252526")
        self.ax = self.figure.add_subplot(111)
        self._apply_plot_theme()
        self.figure.subplots_adjust(left=0.06, right=0.78, top=0.96, bottom=0.14)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=panel)
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_canvas.mpl_connect("pick_event", self._on_plot_pick)
        self.plot_canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        self.plot_canvas.mpl_connect("scroll_event", self._on_plot_scroll)

    def _apply_plot_theme(self) -> None:
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes", color="#e6e6e6")
        self.ax.set_ylabel("Reward", color="#e6e6e6")
        self.ax.tick_params(colors="#d0d0d0")
        for spine in self.ax.spines.values():
            spine.set_color("#4a4f55")
        self.ax.grid(True, alpha=0.35, color="#4a4f55")

    def _add_labeled_var(self, parent: ttk.Frame, label: str, variable: tk.Variable, row: int, col: int) -> None:
        container = ttk.Frame(parent)
        container.grid(row=row, column=col, sticky="ew", padx=6, pady=3)
        container.columnconfigure(0, minsize=140)
        container.columnconfigure(1, weight=1, minsize=150)
        ttk.Label(container, text=label, style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 4))

        if label in {"lr_strategy", "LR strategy"}:
            widget = ttk.Combobox(
                container,
                state="readonly",
                textvariable=variable,
                values=["constant", "linear", "exponential"],
                width=self.input_width,
            )
            widget.bind("<MouseWheel>", lambda e: "break")
            widget.grid(row=0, column=1, sticky="ew")
        elif label == "activation":
            widget = ttk.Combobox(
                container,
                state="readonly",
                textvariable=variable,
                values=["ReLU", "Tanh"],
                width=self.input_width,
            )
            widget.bind("<MouseWheel>", lambda e: "break")
            widget.grid(row=0, column=1, sticky="ew")
        elif isinstance(variable, tk.BooleanVar):
            widget = ttk.Checkbutton(container, variable=variable)
            widget.grid(row=0, column=1, sticky="e")
        else:
            widget = ttk.Entry(container, textvariable=variable, width=self.input_width)
            widget.grid(row=0, column=1, sticky="ew")
        self.param_vars[label] = variable
        self.param_widgets[label] = widget

        tooltip_map = {
            "gamma": "Higher gamma values increase focus on long-term rewards.",
            "learning_rate": "Higher learning rate learns faster but can destabilize training.",
            "batch_size": "Larger batch size increases stability but uses more compute.",
            "hidden_layer": "Wider/deeper networks capture more patterns but cost more compute.",
            "lr_strategy": "LR strategy controls how aggressively updates shrink during training.",
            "min_lr": "Minimum LR prevents updates from becoming too small.",
            "lr_decay": "Decay controls how quickly LR drops in exponential mode.",
            "Episodes": "More episodes usually improve policy quality but take longer.",
            "Max steps": "Maximum steps per episode controls rollout horizon and compute.",
        }
        if label in tooltip_map:
            _Tooltip(widget, tooltip_map[label])

    def _build_specific_rows(self, policy_name: str) -> None:
        for child in self.specific_grid.winfo_children():
            child.destroy()
        self.specific_entries.clear()

        values = self.policy_param_cache[policy_name]
        shared_fields = list(self.shared_specific_fields)
        policy_fields = list(self.policy_specific_fields.get(policy_name, []))

        for idx, field in enumerate(shared_fields):
            row = idx // 2
            col = idx % 2
            var = tk.StringVar(value=str(values.get(field, "")))
            self.specific_entries[field] = var
            self._add_labeled_var(self.specific_grid, field, var, row, col)

        shared_rows = (len(shared_fields) + 1) // 2
        separator = ttk.Separator(self.specific_grid, orient="horizontal")
        separator.grid(row=shared_rows, column=0, columnspan=2, sticky="ew", padx=6, pady=6)

        for idx, field in enumerate(policy_fields):
            row = shared_rows + 1 + (idx // 2)
            col = idx % 2
            var = tk.StringVar(value=str(values.get(field, "")))
            self.specific_entries[field] = var
            self._add_labeled_var(self.specific_grid, field, var, row, col)

        self.compare_dropdown.configure(values=self._compare_fields())

    def _on_policy_changed(self, _event=None) -> None:
        current = self.active_policy
        for field, var in self.specific_entries.items():
            self.policy_param_cache[current][field] = var.get()

        selected = self.policy_dropdown.get()
        self.param_vars["Policy"].set(selected)
        self.active_policy = selected
        self._build_specific_rows(selected)

        is_continuous = selected in ("PPO", "SAC", "TD3")
        self.param_vars["continuous"].set(is_continuous)

    def _apply_environment_update(self) -> None:
        self.env_wrapper.update(
            lap_complete_percent=float(self.param_vars["lap_complete_percent"].get()),
            domain_randomize=bool(self.param_vars["domain_randomize"].get()),
            continuous=bool(self.param_vars["continuous"].get()),
        )

    def _compare_fields(self) -> List[str]:
        policy = self.param_vars.get("Policy", tk.StringVar(value="SAC")).get()
        fields = ["Policy", "Max steps", "Episodes"] + self.shared_specific_fields + self.policy_specific_fields.get(policy, [])
        return fields

    def _on_compare_key_release(self, _event=None) -> None:
        key = self.param_vars["compare_parameter"].get()
        text = self.param_vars["compare_values"].get()
        token = text.split(",")[-1].strip()
        categorical = {
            "Policy": ["PPO", "SAC", "TD3", "DDQN", "QR-DQN"],
            "lr_strategy": ["constant", "linear", "exponential"],
            "LR strategy": ["constant", "linear", "exponential"],
            "Activation": ["Tanh", "ReLU"],
        }
        options = categorical.get(key, [])
        suggestion = ""
        if token and options:
            suggestion = next((option for option in options if option.lower().startswith(token.lower())), "")
        self.compare_hint.configure(text=f"Tab -> {suggestion}" if suggestion else "")

    def _on_compare_tab_complete(self, event) -> str:
        hint = self.compare_hint.cget("text")
        if hint.startswith("Tab -> "):
            value = hint.replace("Tab -> ", "")
            current = self.param_vars["compare_values"].get()
            parts = current.split(",")
            if len(parts) > 1:
                prefix = ",".join(parts[:-1]).strip()
                merged = f"{prefix}, {value}" if prefix else value
                self.param_vars["compare_values"].set(merged)
            else:
                self.param_vars["compare_values"].set(value)
            event.widget.icursor(tk.END)
            return "break"
        return ""

    def _on_compare_toggle(self) -> None:
        if bool(self.param_vars["compare_on"].get()):
            self.param_vars["Animation on"].set(False)

    def _on_animation_runtime_toggle(self, *_args) -> None:
        enabled = bool(self.param_vars["Animation on"].get())
        with self.worker_lock:
            trainers = list(self.active_trainers.values())
        for trainer in trainers:
            trainer.set_animation_enabled(enabled)

        if not enabled:
            self.playback_active = False
            self.pending_playback.clear()
            self.current_frames = []
            self.current_frame_idx = 0
            self.current_playback_steps = 0
            self.steps_progress.configure(value=0)

    def _add_compare(self) -> None:
        key = self.param_vars["compare_parameter"].get().strip()
        values = [v.strip() for v in self.param_vars["compare_values"].get().split(",") if v.strip()]
        if not key or not values:
            return
        self.compare_items[key] = values
        self._refresh_compare_summary()

    def _clear_compare(self) -> None:
        self.compare_items.clear()
        self._refresh_compare_summary()

    def _refresh_compare_summary(self) -> None:
        self.compare_summary.configure(state="normal")
        self.compare_summary.delete("1.0", tk.END)
        for key, values in self.compare_items.items():
            self.compare_summary.insert(tk.END, f"{key}: [{', '.join(values)}]\n")
        self.compare_summary.configure(state="disabled")

    def _bind_mousewheel(self):
        self.master.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self.master.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.params_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_params_configure(self, _event=None):
        bbox = self.params_canvas.bbox("all")
        self.params_canvas.configure(scrollregion=bbox)
        if not bbox:
            return
        content_height = bbox[3] - bbox[1]
        view_height = self.params_canvas.winfo_height()
        if content_height > view_height + 2:
            self.params_scroll.grid(row=0, column=1, sticky="ns", padx=(0, 6), pady=6)
        else:
            self.params_scroll.grid_remove()

    def _on_params_canvas_configure(self, event=None):
        if event is not None:
            self.params_canvas.itemconfigure(self.params_window, width=event.width)
        self._on_params_configure()

    def _collect_train_config(self, run_id: str, session_id: str) -> TrainConfig:
        policy = self.param_vars["Policy"].get()
        for field, var in self.specific_entries.items():
            self.policy_param_cache[policy][field] = var.get()

        params: Dict[str, Any] = {}
        for key, value in self.policy_param_cache[policy].items():
            try:
                if "." in str(value) or "e" in str(value).lower():
                    params[key] = float(value)
                else:
                    params[key] = int(value)
            except Exception:
                params[key] = value

        continuous = policy in ("PPO", "SAC", "TD3")
        self.param_vars["continuous"].set(continuous)

        env_cfg = EnvConfig(
            env_id="CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=float(self.param_vars["lap_complete_percent"].get()),
            domain_randomize=bool(self.param_vars["domain_randomize"].get()),
            continuous=continuous,
        )

        return TrainConfig(
            policy_name=policy,
            episodes=int(self.param_vars["Episodes"].get()),
            max_steps=int(self.param_vars["Max steps"].get()),
            params=params,
            env_config=env_cfg,
            animation_on=bool(self.param_vars["Animation on"].get()),
            animation_fps=int(self.param_vars["Animation FPS"].get()),
            update_rate=max(1, int(self.param_vars["Update rate (episodes)"].get())),
            frame_stride=max(1, int(self.param_vars["Frame stride"].get())),
            run_id=run_id,
            session_id=session_id,
            device=self.param_vars["Device"].get(),
            collect_transitions=False,
            cpu_thread_budget=None,
            performance_mode=bool(self.param_vars["Performance mode"].get()),
            num_envs=4,
        )

    def _run_single_episode(self) -> None:
        if self.btn_train.instate(["disabled"]):
            return
        config = self._collect_train_config(run_id="single_episode", session_id=f"session_{time.time_ns()}")
        result = self.trainer.run_episode(
            model=self.trainer.last_model,
            max_steps=config.max_steps,
            deterministic=True,
            capture_frames=True,
            frame_stride=max(1, config.frame_stride),
            collect_transitions=False,
        )
        self._queue_playback(result.get("frames", []))
        self.status_var.set(
            f"Epsilon: - | LR: - | Best reward: {result['reward']:.2f} | Render: {'on' if result.get('frames') else 'off'}"
        )

    def _start_train(self) -> None:
        self._cancel_active_workers()
        session_id = f"session_{time.time_ns()}"
        self.current_session_id = session_id
        self._flush_pending_events()

        compare_on = bool(self.param_vars["compare_on"].get())
        if compare_on and self.compare_items:
            run_configs = self._expand_compare_configs(session_id)
        else:
            run_configs = [self._collect_train_config(run_id=f"run_{int(time.time())}", session_id=session_id)]

        self.batch_compare_active = compare_on and bool(self.param_vars["Batch compare mode"].get()) and len(run_configs) > 1

        max_workers = min(4, len(run_configs))
        total_cores = max(1, int(os.cpu_count() or 1))
        per_worker_threads = max(1, total_cores // max(1, max_workers))
        self._set_training_active(True)
        selected_policy = self.param_vars["Policy"].get()
        render_run_id = run_configs[0].run_id if run_configs else None
        for cfg in run_configs:
            if cfg.policy_name == selected_policy:
                render_run_id = cfg.run_id
                break

        for cfg in run_configs:
            if compare_on:
                cfg.animation_on = cfg.run_id == render_run_id and cfg.animation_on
            cfg.cpu_thread_budget = per_worker_threads if len(run_configs) > 1 else None
            trainer = CarRacingTrainer(CarRacingEnvWrapper(EnvConfig(**DEFAULT_ENV_CONFIG)), event_sink=self._enqueue_event)
            worker = threading.Thread(target=self._worker_train, args=(cfg, trainer), daemon=True)
            with self.worker_lock:
                self.active_trainers[cfg.run_id] = trainer
                self.active_workers[cfg.run_id] = worker
            worker.start()
            if len(run_configs) > max_workers:
                while self._alive_workers_count() >= max_workers:
                    time.sleep(0.05)

    def _worker_train(self, config: TrainConfig, trainer: CarRacingTrainer) -> None:
        try:
            trainer.train(config)
        except Exception as exc:
            self._enqueue_event(
                {
                    "type": "error",
                    "session_id": config.session_id,
                    "run_id": config.run_id,
                    "message": str(exc),
                }
            )
        finally:
            with self.worker_lock:
                self.active_workers.pop(config.run_id, None)
                self.active_trainers.pop(config.run_id, None)

    def _expand_compare_configs(self, session_id: str) -> List[TrainConfig]:
        base = self._collect_train_config(run_id="compare_base", session_id=session_id)
        keys = list(self.compare_items.keys())
        value_lists = [self.compare_items[key] for key in keys]
        configs: List[TrainConfig] = []

        for idx, combo in enumerate(itertools.product(*value_lists)):
            cfg = replace(base)
            cfg.run_id = f"cmp_{idx}_{int(time.time() * 1000)}"

            combo_map = {key: value for key, value in zip(keys, combo)}
            selected_policy = combo_map.get("Policy", base.policy_name)
            cfg.policy_name = selected_policy

            if "Policy" in combo_map:
                cfg.params = dict(DEFAULT_SPECIFIC.get(selected_policy, {}))
            else:
                cfg.params = dict(base.params)

            cfg.max_steps = int(float(combo_map.get("Max steps", cfg.max_steps)))
            cfg.episodes = int(float(combo_map.get("Episodes", cfg.episodes)))

            for key, value in combo_map.items():
                if key in {"Policy", "Max steps", "Episodes"}:
                    continue
                if key in cfg.params:
                    cfg.params[key] = self._parse_compare_value(value)

            cfg.env_config.continuous = selected_policy in ("PPO", "SAC", "TD3")
            configs.append(cfg)
        return configs

    @staticmethod
    def _parse_compare_value(value: str) -> Any:
        text = str(value).strip()
        try:
            if "." in text or "e" in text.lower():
                return float(text)
            return int(text)
        except Exception:
            return text

    def _toggle_pause(self) -> None:
        paused = self.btn_pause.cget("text") == "Run"
        with self.worker_lock:
            trainers = list(self.active_trainers.values())
        if paused:
            for trainer in trainers:
                trainer.set_paused(False)
            self.btn_pause.configure(text="Pause", style="Neutral.TButton")
            self.btn_train.configure(style="Train.TButton")
        else:
            for trainer in trainers:
                trainer.set_paused(True)
            self.btn_pause.configure(text="Run", style="Pause.TButton")
            self.btn_train.configure(style="Neutral.TButton")

    def _reset_all(self) -> None:
        self._cancel_active_workers()
        self.param_vars["Animation on"].set(True)
        self.param_vars["Animation FPS"].set(30)
        self.param_vars["Update rate (episodes)"].set(1)
        self.param_vars["Frame stride"].set(2)
        self.param_vars["Performance mode"].set(False)
        self.param_vars["Plot redraw every (episodes)"].set(5)
        self.param_vars["Status update every (episodes)"].set(5)
        self.param_vars["Batch compare mode"].set(False)
        self.param_vars["lap_complete_percent"].set(0.95)
        self.param_vars["domain_randomize"].set(False)
        self.param_vars["continuous"].set(True)
        self.param_vars["Max steps"].set(DEFAULT_GENERAL["max_steps"])
        self.param_vars["Episodes"].set(DEFAULT_GENERAL["episodes"])
        self.param_vars["Policy"].set("SAC")
        self.policy_dropdown.set("SAC")
        self.param_vars["Device"].set("GPU")
        self.active_policy = "SAC"
        self.policy_param_cache = {policy: dict(values) for policy, values in DEFAULT_SPECIFIC.items()}
        self._build_specific_rows("SAC")
        self._clear_plot()
        self.status_var.set("Epsilon: - | LR: - | Best reward: - | Render: idle")
        self.steps_progress.configure(value=0, maximum=100)
        self.episodes_progress.configure(value=0, maximum=100)

    def _clear_plot(self) -> None:
        self.run_history.clear()
        self.run_visibility.clear()
        self.run_meta.clear()
        self.run_compare_meta.clear()
        self.run_label_cache.clear()
        self._legend_item_to_run.clear()
        self._hovered_legend_item = None
        self.legend = None
        self.legend_scroll_center = 0.5
        self.ax.clear()
        self._apply_plot_theme()
        self.plot_canvas.draw_idle()

    def _save_csv(self) -> None:
        run_label = self.param_vars["Policy"].get().replace("-", "_")
        self.trainer.export_sampled_transitions_csv(output_dir="results_csv", run_label=run_label)

    def _save_png(self) -> None:
        Path("plots").mkdir(parents=True, exist_ok=True)
        policy = self.param_vars["Policy"].get()
        max_steps = self.param_vars["Max steps"].get()
        gamma = self.specific_entries.get("gamma", tk.StringVar(value="-")).get()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"plots/{policy}_steps-{max_steps}_gamma-{gamma}_{stamp}.png"
        self.figure.savefig(filename, dpi=150)

    def _enqueue_event(self, payload: Dict[str, Any]) -> None:
        self.event_queue.put(payload)

    def _pump_events(self) -> None:
        handled = 0
        while handled < self.EVENT_PUMP_MAX_BATCH:
            try:
                payload = self.event_queue.get_nowait()
            except queue.Empty:
                break
            handled += 1
            if payload.get("session_id") != self.current_session_id:
                continue
            self._handle_event(payload)

        if self._alive_workers_count() == 0 and self.btn_train.cget("style") == "Train.TButton":
            self._set_training_active(False)

        self.after(self.EVENT_PUMP_INTERVAL_MS, self._pump_events)

    def _handle_event(self, payload: Dict[str, Any]) -> None:
        event_type = payload.get("type")
        if event_type == "step":
            return
        if event_type == "episode":
            self._on_episode(payload)
        elif event_type == "episode_aux":
            self._on_episode_aux(payload)
        elif event_type == "training_done":
            self._on_training_done(payload)
        elif event_type == "error":
            self.status_var.set(f"Error: {payload.get('message', 'unknown error')}")

    def _on_episode(self, payload: Dict[str, Any]) -> None:
        run_id = payload["run_id"]
        is_new_run = run_id not in self.run_history
        entry = self.run_history.setdefault(run_id, {"episodes": [], "rewards": [], "ma": [], "eval": [], "lines": {}})
        self.run_visibility.setdefault(run_id, True)
        label_changed = False
        if run_id not in self.run_meta:
            self.run_meta[run_id] = dict(payload.get("meta") or {})
            self.run_compare_meta[run_id] = self._build_compare_enrichment(self.run_meta[run_id])
            self.run_label_cache[run_id] = self._format_run_label(self.run_meta[run_id], self.run_compare_meta[run_id])
            label_changed = True
        elif payload.get("meta"):
            previous_label = self.run_label_cache.get(run_id, "")
            self.run_meta[run_id] = dict(payload.get("meta") or {})
            self.run_compare_meta[run_id] = self._build_compare_enrichment(self.run_meta[run_id])
            self.run_label_cache[run_id] = self._format_run_label(self.run_meta[run_id], self.run_compare_meta[run_id])
            label_changed = self.run_label_cache[run_id] != previous_label
        entry["episodes"].append(payload["episode"])
        entry["rewards"].append(payload["reward"])
        entry["ma"].append(payload["moving_average"])
        entry["eval"] = list(payload.get("eval_points", []))

        self._upsert_run_lines(run_id)
        self._update_run_lines_data(run_id)
        if is_new_run or label_changed:
            self._refresh_legend()
        # Force immediate repaint so each episode becomes visible without
        # waiting for Matplotlib idle coalescing.
        self.plot_canvas.draw()

        episode = int(payload["episode"])
        episodes_total = max(1, int(payload["episodes"]))
        if self.batch_compare_active and self._should_redraw_plot(episode=episode, episodes_total=episodes_total):
            self._redraw_plot()
        self.episodes_progress.configure(maximum=max(1, int(payload["episodes"])), value=int(payload["episode"]))
        self.current_playback_steps = int(payload["steps"])
        self.steps_progress.configure(maximum=max(1, self.current_playback_steps), value=0)

        if self._should_update_status(episode=episode, episodes_total=episodes_total):
            epsilon = payload.get("epsilon")
            epsilon_text = "-" if epsilon is None or epsilon != epsilon else f"{epsilon:.4f}"
            lr = payload.get("lr")
            lr_text = "-" if lr is None or lr != lr else f"{lr:.2e}"
            self.status_var.set(
                f"Epsilon: {epsilon_text} | LR: {lr_text} | Best reward: {payload.get('best_reward', 0.0):.2f} | Render: {payload.get('render_state', 'idle')}"
            )

        frames = payload.get("frames") or []
        if frames and bool(self.param_vars["Animation on"].get()):
            self._queue_playback(frames)

    def _on_episode_aux(self, payload: Dict[str, Any]) -> None:
        run_id = payload["run_id"]
        entry = self.run_history.get(run_id)
        if entry is None:
            return

        # Aux updates can arrive after heavy rollout/eval work; apply them
        # without appending duplicate episode points.
        if "eval_points" in payload:
            entry["eval"] = list(payload.get("eval_points", []))
            self._update_run_lines_data(run_id)
            self.plot_canvas.draw_idle()

        episode = int(payload.get("episode", 0))
        episodes_total = max(1, int(payload.get("episodes", max(1, episode))))
        if self.batch_compare_active and episode > 0 and self._should_redraw_plot(episode=episode, episodes_total=episodes_total):
            self._redraw_plot()

        if episode > 0:
            self.current_playback_steps = int(payload.get("steps", self.current_playback_steps))
            self.steps_progress.configure(maximum=max(1, self.current_playback_steps), value=0)
            if self._should_update_status(episode=episode, episodes_total=episodes_total):
                epsilon = payload.get("epsilon")
                epsilon_text = "-" if epsilon is None or epsilon != epsilon else f"{epsilon:.4f}"
                lr = payload.get("lr")
                lr_text = "-" if lr is None or lr != lr else f"{lr:.2e}"
                self.status_var.set(
                    f"Epsilon: {epsilon_text} | LR: {lr_text} | Best reward: {payload.get('best_reward', 0.0):.2f} | Render: {payload.get('render_state', 'idle')}"
                )

        frames = payload.get("frames") or []
        if frames and bool(self.param_vars["Animation on"].get()):
            self._queue_playback(frames)

    def _on_training_done(self, payload: Dict[str, Any]) -> None:
        cancelled = payload.get("cancelled", False)
        if self.batch_compare_active:
            self._redraw_plot()
        if cancelled:
            self.status_var.set("Training cancelled")
        else:
            self.status_var.set(
                f"Epsilon: - | LR: - | Best reward: {payload.get('best_reward', float('nan')):.2f} | Render: idle"
            )

    def _redraw_plot(self) -> None:
        self.ax.clear()
        self._apply_plot_theme()

        colors = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["#4e79a7"])
        color_index = 0
        for run_id, data in self.run_history.items():
            color = colors[color_index % len(colors)]
            color_index += 1
            label_reward = self.run_label_cache.get(run_id)
            if not label_reward:
                meta = self.run_meta.get(run_id, {})
                label_reward = self._format_run_label(meta, self.run_compare_meta.get(run_id, {}))
                self.run_label_cache[run_id] = label_reward
            visible = self.run_visibility.get(run_id, True)

            reward_line, = self.ax.plot(data["episodes"], data["rewards"], color=color, alpha=0.60, label=label_reward)
            ma_line, = self.ax.plot(data["episodes"], data["ma"], color=color, linestyle="--", linewidth=2.0, alpha=1.0)
            if data["eval"]:
                eval_x = [point[0] for point in data["eval"]]
                eval_y = [point[1] for point in data["eval"]]
                eval_line, = self.ax.plot(eval_x, eval_y, color=color, linestyle=":", linewidth=2.0, marker="o", alpha=1.0)
            else:
                eval_line = None

            reward_line.set_visible(visible)
            ma_line.set_visible(visible)
            if eval_line is not None:
                eval_line.set_visible(visible)

            data["lines"] = {"reward": reward_line, "ma": ma_line, "eval": eval_line}
        self._refresh_legend()
        self.plot_canvas.draw_idle()

    def _upsert_run_lines(self, run_id: str) -> None:
        data = self.run_history.get(run_id)
        if data is None:
            return
        lines = data.setdefault("lines", {})
        if lines.get("reward") is not None and lines.get("ma") is not None:
            label_reward = self.run_label_cache.get(run_id)
            if label_reward:
                lines["reward"].set_label(label_reward)
            return

        colors = matplotlib.rcParams["axes.prop_cycle"].by_key().get("color", ["#4e79a7"])
        existing_count = sum(1 for value in self.run_history.values() if value.get("lines", {}).get("reward") is not None)
        color = colors[existing_count % len(colors)]
        label_reward = self.run_label_cache.get(run_id)
        if not label_reward:
            meta = self.run_meta.get(run_id, {})
            label_reward = self._format_run_label(meta, self.run_compare_meta.get(run_id, {}))
            self.run_label_cache[run_id] = label_reward

        reward_line, = self.ax.plot([], [], color=color, alpha=0.60, label=label_reward)
        ma_line, = self.ax.plot([], [], color=color, linestyle="--", linewidth=2.0, alpha=1.0)
        lines["reward"] = reward_line
        lines["ma"] = ma_line
        lines["eval"] = None

    def _update_run_lines_data(self, run_id: str) -> None:
        data = self.run_history.get(run_id)
        if data is None:
            return
        lines = data.get("lines", {})
        reward_line = lines.get("reward")
        ma_line = lines.get("ma")
        if reward_line is None or ma_line is None:
            return

        episodes = data.get("episodes", [])
        rewards = data.get("rewards", [])
        moving_average = data.get("ma", [])
        reward_line.set_data(episodes, rewards)
        ma_line.set_data(episodes, moving_average)

        eval_points = data.get("eval", [])
        eval_line = lines.get("eval")
        if eval_points:
            if eval_line is None:
                color = reward_line.get_color()
                eval_line, = self.ax.plot([], [], color=color, linestyle=":", linewidth=2.0, marker="o", alpha=1.0)
                lines["eval"] = eval_line
            eval_x = [point[0] for point in eval_points]
            eval_y = [point[1] for point in eval_points]
            eval_line.set_data(eval_x, eval_y)
        elif eval_line is not None:
            eval_line.remove()
            lines["eval"] = None

        visible = self.run_visibility.get(run_id, True)
        reward_line.set_visible(visible)
        ma_line.set_visible(visible)
        if lines.get("eval") is not None:
            lines["eval"].set_visible(visible)

        self.ax.relim()
        self.ax.autoscale_view()

    def _refresh_legend(self) -> None:
        run_order: List[str] = []
        reward_handles: List[Any] = []
        reward_labels: List[str] = []
        for run_id, data in self.run_history.items():
            reward_line = data.get("lines", {}).get("reward")
            if reward_line is None:
                continue
            run_order.append(run_id)
            reward_handles.append(reward_line)
            reward_labels.append(reward_line.get_label())

        if self.legend is not None:
            try:
                self.legend.remove()
            except Exception:
                pass
            self.legend = None

        self._legend_item_to_run.clear()
        if not reward_handles:
            return

        proxy_ma = Line2D([0], [0], color="#b0b0b0", linestyle="--", linewidth=2.0)
        proxy_eval = Line2D([0], [0], color="#b0b0b0", linestyle=":", linewidth=2.0, marker="o")
        legend_handles = reward_handles + [proxy_ma, proxy_eval]
        legend_labels = reward_labels + ["moving average", "evaluation rollout"]

        self.legend = self.ax.legend(
            legend_handles,
            legend_labels,
            loc="center left",
            bbox_to_anchor=(1.01, self.legend_scroll_center),
            frameon=False,
            borderaxespad=0.0,
            fontsize=8,
        )
        for text in self.legend.get_texts():
            text.set_color("#e6e6e6")
        self._wire_legend_interactions(run_order)

    def _wire_legend_interactions(self, run_order: List[str]) -> None:
        self._legend_item_to_run.clear()
        if self.legend is None:
            return
        handles = list(getattr(self.legend, "legendHandles", getattr(self.legend, "legend_handles", [])))
        texts = list(self.legend.get_texts())
        for idx, run_id in enumerate(run_order):
            if idx < len(handles):
                handle = handles[idx]
                handle.set_picker(True)
                self._legend_item_to_run[handle] = run_id
            if idx < len(texts):
                text = texts[idx]
                text.set_picker(True)
                text.set_alpha(1.0 if self.run_visibility.get(run_id, True) else 0.35)
                self._legend_item_to_run[text] = run_id

    def _toggle_run_visibility(self, run_id: str) -> None:
        visible = not self.run_visibility.get(run_id, True)
        self.run_visibility[run_id] = visible
        lines = self.run_history.get(run_id, {}).get("lines", {})
        for line in lines.values():
            if line is not None:
                line.set_visible(visible)
        self.plot_canvas.draw_idle()

    def _on_plot_pick(self, event) -> None:
        run_id = self._legend_item_to_run.get(event.artist)
        if run_id is not None:
            self._toggle_run_visibility(run_id)

    def _on_plot_hover(self, event) -> None:
        widget = self.plot_canvas.get_tk_widget()
        hovered = None
        for item in self._legend_item_to_run:
            contains, _ = item.contains(event)
            if contains:
                hovered = item
                break
        if hovered is not None:
            widget.configure(cursor="hand2")
            if self._hovered_legend_item is not None and self._hovered_legend_item in self._legend_item_to_run:
                self._hovered_legend_item.set_alpha(1.0)
            hovered.set_alpha(0.7)
            self._hovered_legend_item = hovered
            self.plot_canvas.draw_idle()
        else:
            widget.configure(cursor="")
            if self._hovered_legend_item is not None:
                self._hovered_legend_item.set_alpha(1.0)
                self._hovered_legend_item = None
                self.plot_canvas.draw_idle()

    def _on_plot_scroll(self, event) -> None:
        if self.legend is None:
            return
        renderer = self.figure.canvas.get_renderer()
        bbox = self.legend.get_window_extent(renderer=renderer)
        if not bbox.contains(event.x, event.y):
            return
        delta = 0.04 if event.step > 0 else -0.04
        self.legend_scroll_center = min(1.0, max(0.0, self.legend_scroll_center + delta))
        self.legend.set_bbox_to_anchor((1.01, self.legend_scroll_center), transform=self.ax.transAxes)
        self.plot_canvas.draw_idle()

    def _build_compare_enrichment(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        params = meta.get("params", {}) if isinstance(meta.get("params", {}), dict) else {}
        base_fields = {
            "Policy",
            "Max steps",
            "Episodes",
            "gamma",
            "learning_rate",
            "lr_strategy",
            "lr_decay",
            "lap_complete_percent",
            "domain_randomize",
            "continuous",
        }
        result: Dict[str, Any] = {}
        for key in self.compare_items:
            if key in base_fields:
                continue
            if key in params:
                result[key] = params[key]
        return result

    def _format_run_label(self, meta: Dict[str, Any], enrichment: Dict[str, Any]) -> str:
        line1 = f"{meta.get('policy', '-')} | steps={meta.get('max_steps', '-')} | gamma={meta.get('gamma', '-')}"
        line2 = (
            f"epsilon={meta.get('epsilon', '-')} | epsilon_decay={meta.get('epsilon_decay', '-')} "
            f"| epsilon_min={meta.get('epsilon_min', '-')}"
        )
        line3 = (
            f"LR={meta.get('learning_rate', '-')} | LR strategy={meta.get('lr_strategy', '-')} "
            f"| LR decay={meta.get('lr_decay', '-')}"
        )
        line4 = (
            f"lap_complete_percent={meta.get('lap_complete_percent', '-')} | "
            f"domain_randomize={meta.get('domain_randomize', '-')} | continuous={meta.get('continuous', '-')}"
        )
        if enrichment:
            extra = " | ".join(f"{key}={value}" for key, value in enrichment.items())
            return f"{line1}\n{line2}\n{line3}\n{line4}\n{extra}"
        return f"{line1}\n{line2}\n{line3}\n{line4}"

    def _queue_playback(self, frames: List[Any]) -> None:
        if not frames:
            return
        if self.playback_active:
            if len(self.pending_playback) == self.pending_playback.maxlen:
                self.pending_playback.popleft()
            self.pending_playback.append(frames)
            return
        self.current_frames = frames
        self.current_frame_idx = 0
        self.playback_active = True
        self._playback_step()

    def _playback_step(self) -> None:
        if not self.playback_active:
            return
        if self.current_frame_idx >= len(self.current_frames):
            self.playback_active = False
            if self.pending_playback:
                next_frames = self.pending_playback.popleft()
                self._queue_playback(next_frames)
            return

        frame = self.current_frames[self.current_frame_idx]
        self.current_frame_idx += 1
        if self.current_frames:
            ratio = self.current_frame_idx / max(1, len(self.current_frames))
            self.steps_progress.configure(value=min(self.current_playback_steps, int(self.current_playback_steps * ratio)))
        self._draw_frame(frame)
        fps = max(1, int(self.param_vars["Animation FPS"].get()))
        self.after(int(1000 / fps), self._playback_step)

    def _draw_frame(self, frame: Any) -> None:
        if Image is None or ImageTk is None:
            return
        image = Image.fromarray(frame)
        canvas_w = max(1, self.render_canvas.winfo_width())
        canvas_h = max(1, self.render_canvas.winfo_height())
        image.thumbnail((canvas_w, canvas_h))
        self.render_photo = ImageTk.PhotoImage(image)
        self.render_canvas.delete("all")
        self.render_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self.render_photo)

    def _draw_last_frame(self) -> None:
        if self.current_frames and self.current_frame_idx > 0:
            idx = min(self.current_frame_idx - 1, len(self.current_frames) - 1)
            self._draw_frame(self.current_frames[idx])

    def _cancel_active_workers(self) -> None:
        with self.worker_lock:
            workers = list(self.active_workers.values())
            trainers = list(self.active_trainers.values())
        for trainer in trainers:
            trainer.set_paused(False)
            trainer.cancel()
        for worker in workers:
            worker.join(timeout=0.2)
        with self.worker_lock:
            self.active_workers.clear()
            self.active_trainers.clear()
        self.trainer.reset_control_flags()

    def _alive_workers_count(self) -> int:
        with self.worker_lock:
            return sum(1 for worker in self.active_workers.values() if worker.is_alive())

    def _flush_pending_events(self) -> None:
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                return

    def _set_training_active(self, active: bool) -> None:
        if active:
            self.btn_train.configure(style="Train.TButton")
            self.btn_single.state(["disabled"])
        else:
            self.batch_compare_active = False
            self.btn_train.configure(style="Neutral.TButton")
            self.btn_pause.configure(text="Pause", style="Neutral.TButton")
            self.btn_single.state(["!disabled"])

    def _get_interval(self, key: str, fallback: int = 1) -> int:
        value = self.param_vars.get(key)
        if value is None:
            return max(1, int(fallback))
        try:
            return max(1, int(value.get()))
        except Exception:
            return max(1, int(fallback))

    def _should_redraw_plot(self, episode: int, episodes_total: int) -> bool:
        if episode >= episodes_total:
            return True
        perf_mode_var = self.param_vars.get("Performance mode")
        if bool(perf_mode_var.get()) if perf_mode_var is not None else False:
            interval = self._get_interval("Plot redraw every (episodes)", fallback=5)
        else:
            interval = 1
        if self.batch_compare_active:
            interval = max(interval, 10)
        return episode % interval == 0

    def _should_update_status(self, episode: int, episodes_total: int) -> bool:
        if episode >= episodes_total:
            return True
        perf_mode_var = self.param_vars.get("Performance mode")
        if bool(perf_mode_var.get()) if perf_mode_var is not None else False:
            interval = self._get_interval("Status update every (episodes)", fallback=5)
            return episode % interval == 0
        return True

    def _on_close(self) -> None:
        self.trainer.set_paused(False)
        self._cancel_active_workers()
        self.master.destroy()
