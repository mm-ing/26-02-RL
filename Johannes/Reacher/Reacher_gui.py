from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
from queue import Empty, Queue
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Reacher_logic import EnvironmentConfig, ReacherTrainer, TrainingConfig

matplotlib.use("TkAgg")

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None

try:
    import torch
except Exception:  # pragma: no cover - optional dependency
    torch = None


PALETTE = {
    "main_bg": "#1e1e1e",
    "panel_bg": "#252526",
    "input_bg": "#2d2d30",
    "text": "#e6e6e6",
    "muted": "#d0d0d0",
    "accent": "#0e639c",
    "btn_neutral": "#3a3d41",
    "btn_neutral_active": "#4a4f55",
    "btn_neutral_pressed": "#2f3338",
    "btn_train": "#0e639c",
    "btn_train_active": "#1177bb",
    "btn_train_pressed": "#0b4f7a",
    "btn_pause": "#a66a00",
    "btn_pause_active": "#bf7a00",
    "btn_pause_pressed": "#8c5900",
}

POLICY_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "PPO": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "Tanh",
        "lr_strategy": "linear",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "n_steps": 2048,
    },
    "SAC": {
        "gamma": 0.99,
        "learning_rate": 3e-4,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "train_freq": 1,
        "gradient_steps": 1,
        "buffer_size": 100000,
        "learning_starts": 1000,
        "tau": 0.005,
    },
    "TD3": {
        "gamma": 0.99,
        "learning_rate": 1e-3,
        "batch_size": 256,
        "hidden_layer": "256,256",
        "activation": "ReLU",
        "lr_strategy": "constant",
        "min_lr": 1e-5,
        "lr_decay": 0.995,
        "train_freq": 1,
        "gradient_steps": 1,
        "buffer_size": 200000,
        "learning_starts": 1000,
        "tau": 0.005,
        "policy_delay": 2,
    },
}


@dataclass
class WorkerHandle:
    trainer: ReacherTrainer
    run_id: str
    session_id: int
    cpu_threads: int = 1


class _SimpleTooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tip: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event) -> None:
        if self.tip is not None:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tip,
            text=self.text,
            bg="#2d2d30",
            fg="#e6e6e6",
            relief="solid",
            borderwidth=1,
            font=("Segoe UI", 9),
            padx=6,
            pady=4,
            wraplength=340,
            justify="left",
        )
        label.pack()

    def _hide(self, _event: tk.Event) -> None:
        if self.tip is not None:
            self.tip.destroy()
            self.tip = None


class ReacherGUI(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master)
        self.master = master
        self.grid(row=0, column=0, sticky="nsew")

        self.event_queue: "Queue[Dict[str, Any]]" = Queue()
        self.executor = ThreadPoolExecutor(max_workers=4)

        self.worker_lock = Lock()
        self.active_workers: Dict[str, WorkerHandle] = {}
        self.current_session_id = 0
        self.is_training = False
        self.is_paused = False

        self.strict_visual_mode = tk.BooleanVar(value=False)

        self.run_plot_data: Dict[str, Dict[str, List[Any]]] = {}
        self.run_plot_artists: Dict[str, Dict[str, Any]] = {}
        self.run_visibility: Dict[str, bool] = {}
        self.run_metadata: Dict[str, Dict[str, Any]] = {}

        self._active_playback = False
        self._pending_playback: Optional[List[Any]] = None
        self._current_playback_frames: List[Any] = []
        self._playback_index = 0
        self._render_photo = None
        self._legend_scroll_offset = 0.0
        self._legend_pick_to_run: Dict[Any, str] = {}
        self._tooltips: List[_SimpleTooltip] = []

        self.policy_cache: Dict[str, Dict[str, Any]] = {k: dict(v) for k, v in POLICY_DEFAULTS.items()}
        self._active_policy = "SAC"

        self._build_style()
        self._build_state()
        self._build_layout()

        self.after(16, self._ui_pump)

    def _build_style(self) -> None:
        self.master.title("Reacher Trainer")
        self.master.configure(bg=PALETTE["main_bg"])
        self.master.rowconfigure(0, weight=1)
        self.master.columnconfigure(0, weight=1)

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            style.theme_use("vista")

        style.configure("Root.TFrame", background=PALETTE["main_bg"])
        style.configure("Panel.TLabelframe", background=PALETTE["panel_bg"], foreground=PALETTE["text"])
        style.configure("Panel.TLabelframe.Label", background=PALETTE["panel_bg"], foreground=PALETTE["text"], font=("Segoe UI", 10, "bold"))
        style.configure("Panel.TFrame", background=PALETTE["panel_bg"])
        style.configure("Panel.TLabel", background=PALETTE["panel_bg"], foreground=PALETTE["text"], font=("Segoe UI", 10))
        style.configure("Panel.TCheckbutton", background=PALETTE["panel_bg"], foreground=PALETTE["text"], font=("Segoe UI", 10))
        style.configure(
            "Neutral.TButton",
            background=PALETTE["btn_neutral"],
            foreground=PALETTE["text"],
            font=("Segoe UI", 10, "bold"),
            borderwidth=1,
            focusthickness=0,
        )
        style.map(
            "Neutral.TButton",
            background=[("active", PALETTE["btn_neutral_active"]), ("pressed", PALETTE["btn_neutral_pressed"])],
        )
        style.configure("Train.TButton", background=PALETTE["btn_train"], foreground=PALETTE["text"], font=("Segoe UI", 10, "bold"))
        style.map("Train.TButton", background=[("active", PALETTE["btn_train_active"]), ("pressed", PALETTE["btn_train_pressed"])])
        style.configure("Pause.TButton", background=PALETTE["btn_pause"], foreground=PALETTE["text"], font=("Segoe UI", 10, "bold"))
        style.map("Pause.TButton", background=[("active", PALETTE["btn_pause_active"]), ("pressed", PALETTE["btn_pause_pressed"])])

        style.configure("Dark.Horizontal.TProgressbar", troughcolor="#343434", background=PALETTE["accent"], bordercolor="#343434")

        # Style ttk.Combobox popup listbox to dark theme on Tk-based platforms.
        self.master.option_add("*TCombobox*Listbox*Background", PALETTE["input_bg"])
        self.master.option_add("*TCombobox*Listbox*Foreground", PALETTE["text"])
        self.master.option_add("*TCombobox*Listbox*selectBackground", PALETTE["accent"])
        self.master.option_add("*TCombobox*Listbox*selectForeground", "white")

    def _build_state(self) -> None:
        self.configure(style="Root.TFrame")

        self.var_animation_on = tk.BooleanVar(value=True)
        self.var_animation_fps = tk.StringVar(value="30")
        self.var_update_rate = tk.StringVar(value="1")
        self.var_frame_stride = tk.StringVar(value="2")

        self.var_compare_on = tk.BooleanVar(value=False)
        self.var_compare_param = tk.StringVar(value="Policy")
        self.var_compare_values = tk.StringVar(value="")
        self.var_compare_hint = tk.StringVar(value="")
        self.compare_params: Dict[str, List[str]] = {}

        self.var_max_steps = tk.StringVar(value="200")
        self.var_episodes = tk.StringVar(value="3000")

        self.var_policy = tk.StringVar(value="SAC")

        self.var_reward_dist_weight = tk.StringVar(value="1.0")
        self.var_reward_control_weight = tk.StringVar(value="0.1")

        self.var_moving_average = tk.StringVar(value="20")
        self.var_eval_rollout_on = tk.BooleanVar(value=False)
        self.var_device = tk.StringVar(value="CPU")

        self.status_var = tk.StringVar(value="Epsilon: n/a | LR: n/a | Best reward: n/a | Render: idle")
        self.steps_var = tk.StringVar(value="Playback frames: 0/0")
        self.episodes_var = tk.StringVar(value="Episodes: 0/0")

        self._specific_vars: Dict[str, tk.Variable] = {}
        self._specific_widgets: Dict[str, tk.Widget] = {}

    def _build_layout(self) -> None:
        self.rowconfigure(0, weight=3)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=4)
        self.columnconfigure(0, weight=2)
        self.columnconfigure(1, weight=1)

        self._build_environment_panel()
        self._build_parameters_panel()
        self._build_controls_row()
        self._build_current_run_panel()
        self._build_live_plot_panel()

    def _build_environment_panel(self) -> None:
        panel = ttk.LabelFrame(self, text="Environment", style="Panel.TLabelframe")
        panel.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(panel, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.render_canvas.bind("<Configure>", lambda _e: self._redraw_current_frame())

    def _build_parameters_panel(self) -> None:
        outer = ttk.LabelFrame(self, text="Parameters", style="Panel.TLabelframe")
        outer.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=PALETTE["panel_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        self.param_frame = ttk.Frame(canvas, style="Panel.TFrame")
        self.param_frame.columnconfigure(0, weight=1)

        self.param_frame.bind(
            "<Configure>",
            lambda _e: canvas.configure(scrollregion=canvas.bbox("all")),
        )

        canvas_window = canvas.create_window((0, 0), window=self.param_frame, anchor="nw")

        def _on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfigure(canvas_window, width=event.width)

        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        canvas.bind("<Enter>", lambda _e: canvas.bind_all("<MouseWheel>", lambda ev: canvas.yview_scroll(int(-ev.delta / 120), "units")))
        canvas.bind("<Leave>", lambda _e: canvas.unbind_all("<MouseWheel>"))

        self._build_environment_group()
        self._build_compare_group()
        self._build_general_group()
        self._build_specific_group()
        self._build_live_plot_group()

    def _build_environment_group(self) -> None:
        frame = ttk.LabelFrame(self.param_frame, text="Environment", style="Panel.TLabelframe")
        frame.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
        frame.columnconfigure(0, weight=0, minsize=92)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0, minsize=92)
        frame.columnconfigure(3, weight=1)

        ttk.Checkbutton(frame, text="Animation on", variable=self.var_animation_on, style="Panel.TCheckbutton").grid(row=0, column=0, columnspan=2, sticky="w", padx=4, pady=4)

        animation_fps = self._add_labeled_entry(frame, 1, "Animation FPS", self.var_animation_fps, pair=0)
        update_rate = self._add_labeled_entry(frame, 1, "Update rate (episodes)", self.var_update_rate, pair=1)
        frame_stride = self._add_labeled_entry(frame, 2, "Frame stride", self.var_frame_stride, pair=0)

        update_btn = ttk.Button(frame, text="Update", style="Neutral.TButton", command=self._update_environment)
        update_btn.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=(6, 4))

        # Environment-specific parameters in two columns under Update.
        ttk.Label(frame, text="reward_dist_weight", style="Panel.TLabel").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        self.entry_reward_dist = tk.Entry(frame, textvariable=self.var_reward_dist_weight)
        self._style_entry(self.entry_reward_dist)
        self.entry_reward_dist.grid(row=4, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, text="reward_control_weight", style="Panel.TLabel").grid(row=4, column=2, sticky="w", padx=4, pady=4)
        self.entry_reward_ctrl = tk.Entry(frame, textvariable=self.var_reward_control_weight)
        self._style_entry(self.entry_reward_ctrl)
        self.entry_reward_ctrl.grid(row=4, column=3, sticky="ew", padx=4, pady=4)

        self._attach_tooltip(animation_fps, "Higher FPS makes playback smoother but increases GUI redraw work.")
        self._attach_tooltip(update_rate, "Lower update rate emits animation more frequently for closer visual tracking.")
        self._attach_tooltip(frame_stride, "Higher stride captures fewer frames, reducing capture cost but making playback less detailed.")
        self._attach_tooltip(self.entry_reward_dist, "Increasing this raises pressure to minimize distance to the target, often speeding convergence to reach behavior.")
        self._attach_tooltip(self.entry_reward_ctrl, "Increasing this penalizes control effort more, usually improving smoothness but slowing aggressive exploration.")

    def _build_compare_group(self) -> None:
        frame = ttk.LabelFrame(self.param_frame, text="Compare", style="Panel.TLabelframe")
        frame.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=1)

        ttk.Checkbutton(frame, text="Compare on", variable=self.var_compare_on, style="Panel.TCheckbutton", command=self._on_compare_toggle).grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Button(frame, text="Clear", style="Neutral.TButton", command=self._clear_compare).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(frame, text="Add", style="Neutral.TButton", command=self._add_compare).grid(row=0, column=2, sticky="ew", padx=4, pady=4)

        self.cmb_compare_param = ttk.Combobox(frame, state="readonly", textvariable=self.var_compare_param, values=self._compare_param_values())
        self._style_combobox(self.cmb_compare_param)
        self.cmb_compare_param.grid(row=1, column=0, sticky="ew", padx=4, pady=4)

        self.entry_compare_values = tk.Entry(frame, textvariable=self.var_compare_values)
        self._style_entry(self.entry_compare_values)
        self.entry_compare_values.grid(row=1, column=1, columnspan=2, sticky="ew", padx=4, pady=4)
        self.entry_compare_values.bind("<Return>", lambda _e: self._add_compare())
        self.entry_compare_values.bind("<Tab>", self._tab_complete_compare)
        self.entry_compare_values.bind("<KeyRelease>", lambda _e: self._refresh_compare_hint())

        hint = ttk.Label(frame, textvariable=self.var_compare_hint, style="Panel.TLabel")
        hint.grid(row=2, column=1, columnspan=2, sticky="w", padx=4, pady=(0, 4))

        self.compare_summary = tk.Text(frame, height=4, bg=PALETTE["input_bg"], fg=PALETTE["text"], insertbackground="white", relief="flat", wrap="word")
        self.compare_summary.grid(row=3, column=0, columnspan=3, sticky="ew", padx=4, pady=4)
        self.compare_summary.configure(state="disabled")

    def _build_general_group(self) -> None:
        frame = ttk.LabelFrame(self.param_frame, text="General", style="Panel.TLabelframe")
        frame.grid(row=2, column=0, sticky="ew", padx=6, pady=6)
        frame.columnconfigure(0, weight=0, minsize=92)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0, minsize=92)
        frame.columnconfigure(3, weight=1)

        max_steps = self._add_labeled_entry(frame, 0, "Max steps", self.var_max_steps, pair=0)
        episodes = self._add_labeled_entry(frame, 0, "Episodes", self.var_episodes, pair=1)
        self._attach_tooltip(max_steps, "Higher max steps allows longer trajectories but increases compute per episode.")
        self._attach_tooltip(episodes, "More episodes improve final performance potential but increase total training time.")

    def _build_specific_group(self) -> None:
        self.specific_group = ttk.LabelFrame(self.param_frame, text="Specific", style="Panel.TLabelframe")
        self.specific_group.grid(row=3, column=0, sticky="ew", padx=6, pady=6)
        self.specific_group.columnconfigure(0, weight=0, minsize=92)
        self.specific_group.columnconfigure(1, weight=1)
        self.specific_group.columnconfigure(2, weight=0, minsize=92)
        self.specific_group.columnconfigure(3, weight=1)

        ttk.Label(self.specific_group, text="Policy", style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        cmb = ttk.Combobox(self.specific_group, state="readonly", textvariable=self.var_policy, values=["PPO", "SAC", "TD3"])
        self._style_combobox(cmb)
        cmb.grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=4)
        cmb.bind("<<ComboboxSelected>>", lambda _e: self._on_policy_change())

        self.shared_keys = ["gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        self._render_specific_fields()

    def _build_live_plot_group(self) -> None:
        frame = ttk.LabelFrame(self.param_frame, text="Live Plot", style="Panel.TLabelframe")
        frame.grid(row=4, column=0, sticky="ew", padx=6, pady=6)
        frame.columnconfigure(0, weight=0, minsize=92)
        frame.columnconfigure(1, weight=1)
        frame.columnconfigure(2, weight=0, minsize=92)
        frame.columnconfigure(3, weight=1)

        self._add_labeled_entry(frame, 0, "Moving average values", self.var_moving_average, pair=0)
        ttk.Checkbutton(
            frame,
            text="Evaluation rollout on",
            variable=self.var_eval_rollout_on,
            style="Panel.TCheckbutton",
        ).grid(row=0, column=2, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Checkbutton(frame, text="Strict visual mode", variable=self.strict_visual_mode, style="Panel.TCheckbutton").grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=4)

    def _build_controls_row(self) -> None:
        frame = ttk.Frame(self, style="Panel.TFrame")
        frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        for col in range(8):
            frame.columnconfigure(col, weight=1)

        self.btn_run_one = ttk.Button(frame, text="Run single episode", style="Neutral.TButton", command=self._run_single_episode)
        self.btn_train = ttk.Button(frame, text="Train and Run", style="Neutral.TButton", command=self._train_and_run)
        self.btn_pause = ttk.Button(frame, text="Pause", style="Neutral.TButton", command=self._toggle_pause)
        self.btn_reset = ttk.Button(frame, text="Reset All", style="Neutral.TButton", command=self._reset_all)
        self.btn_clear_plot = ttk.Button(frame, text="Clear Plot", style="Neutral.TButton", command=self._clear_plot)
        self.btn_csv = ttk.Button(frame, text="Save samplings CSV", style="Neutral.TButton", command=self._save_csv)
        self.btn_png = ttk.Button(frame, text="Save Plot PNG", style="Neutral.TButton", command=self._save_png)

        self.device_combo = ttk.Combobox(frame, state="readonly", values=["CPU", "GPU"], textvariable=self.var_device)
        self._style_combobox(self.device_combo)

        widgets = [
            self.btn_run_one,
            self.btn_train,
            self.btn_pause,
            self.btn_reset,
            self.btn_clear_plot,
            self.btn_csv,
            self.btn_png,
            self.device_combo,
        ]
        for idx, widget in enumerate(widgets):
            widget.grid(row=0, column=idx, sticky="ew", padx=4, pady=4)

    def _build_current_run_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Current Run", style="Panel.TLabelframe")
        frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, textvariable=self.steps_var, style="Panel.TLabel").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.steps_progress = ttk.Progressbar(frame, mode="determinate", style="Dark.Horizontal.TProgressbar")
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, textvariable=self.episodes_var, style="Panel.TLabel").grid(row=1, column=0, sticky="w", padx=4, pady=4)
        self.episodes_progress = ttk.Progressbar(frame, mode="determinate", style="Dark.Horizontal.TProgressbar")
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

        ttk.Label(frame, textvariable=self.status_var, style="Panel.TLabel").grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)

    def _build_live_plot_panel(self) -> None:
        frame = ttk.LabelFrame(self, text="Live Plot", style="Panel.TLabelframe")
        frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        fig = Figure(figsize=(10, 4), dpi=100)
        fig.patch.set_facecolor(PALETTE["panel_bg"])
        self.ax = fig.add_subplot(111)
        fig.subplots_adjust(left=0.04, right=0.78, bottom=0.12, top=0.98)
        self._style_plot_axes()

        self.canvas_plot = FigureCanvasTkAgg(fig, master=frame)
        self.canvas_plot.get_tk_widget().configure(bg=PALETTE["panel_bg"], highlightthickness=0)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas_plot.mpl_connect("pick_event", self._on_legend_pick)
        self.canvas_plot.mpl_connect("motion_notify_event", self._on_plot_hover)
        self.canvas_plot.mpl_connect("scroll_event", self._on_legend_scroll)

        self.legend = None

    def _style_plot_axes(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("Episodes", color=PALETTE["text"])
        self.ax.set_ylabel("Reward", color=PALETTE["text"])
        self.ax.grid(True, alpha=0.35)
        self.ax.set_facecolor(PALETTE["panel_bg"])
        for spine in self.ax.spines.values():
            spine.set_color(PALETTE["muted"])
        self.ax.tick_params(colors=PALETTE["text"])

    def _style_entry(self, widget: tk.Entry) -> None:
        widget.configure(bg=PALETTE["input_bg"], fg=PALETTE["text"], insertbackground="white", relief="flat")

    def _style_combobox(self, widget: ttk.Combobox) -> None:
        style_name = "Dark.TCombobox"
        style = ttk.Style()
        style.configure(style_name, fieldbackground=PALETTE["input_bg"], background=PALETTE["input_bg"], foreground=PALETTE["text"])
        style.map(
            style_name,
            fieldbackground=[("readonly", PALETTE["input_bg"])],
            foreground=[("readonly", PALETTE["text"])],
            selectbackground=[("readonly", PALETTE["accent"])],
            selectforeground=[("readonly", "white")],
        )
        widget.configure(style=style_name)
        widget.bind("<MouseWheel>", lambda e: "break")

        # Fallback for Tk popdown listbox colors (helps on some Windows themes).
        try:
            popdown = widget.tk.eval(f"ttk::combobox::PopdownWindow {widget}")
            listbox_path = f"{popdown}.f.l"
            widget.tk.call(
                listbox_path,
                "configure",
                "-background",
                PALETTE["input_bg"],
                "-foreground",
                PALETTE["text"],
                "-selectbackground",
                PALETTE["accent"],
                "-selectforeground",
                "white",
            )
        except Exception:
            pass

    def _add_labeled_entry(self, parent: tk.Widget, row: int, label: str, var: tk.Variable, pair: int = 0) -> tk.Entry:
        base_col = 0 if pair <= 0 else 2
        ttk.Label(parent, text=label, style="Panel.TLabel").grid(row=row, column=base_col, sticky="w", padx=4, pady=4)
        entry = tk.Entry(parent, textvariable=var, width=9)
        self._style_entry(entry)
        entry.grid(row=row, column=base_col + 1, sticky="ew", padx=4, pady=4)
        return entry

    def _attach_tooltip(self, widget: tk.Widget, text: str) -> None:
        self._tooltips.append(_SimpleTooltip(widget, text))

    def _compare_param_values(self) -> List[str]:
        shared = ["Policy", "max_steps", "episodes", "gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        policy = self.var_policy.get()
        specific = [k for k in POLICY_DEFAULTS.get(policy, {}).keys() if k not in shared]
        return shared + specific

    def _refresh_compare_hint(self) -> None:
        raw = self.var_compare_values.get().strip()
        if not raw:
            self.var_compare_hint.set("")
            return

        prefix = raw.split(",")[-1].strip()
        param = self.var_compare_param.get()
        suggestions = {
            "Policy": ["PPO", "SAC", "TD3"],
            "activation": ["ReLU", "Tanh"],
            "lr_strategy": ["constant", "linear", "exponential"],
        }.get(param, [])

        match = next((s for s in suggestions if s.lower().startswith(prefix.lower())), None)
        self.var_compare_hint.set(f"Tab -> {match}" if match and prefix else "")

    def _tab_complete_compare(self, _event: tk.Event) -> str:
        raw = self.var_compare_values.get()
        parts = [p.strip() for p in raw.split(",")]
        if not parts:
            return "break"

        prefix = parts[-1]
        param = self.var_compare_param.get()
        suggestions = {
            "Policy": ["PPO", "SAC", "TD3"],
            "activation": ["ReLU", "Tanh"],
            "lr_strategy": ["constant", "linear", "exponential"],
        }.get(param, [])

        match = next((s for s in suggestions if s.lower().startswith(prefix.lower())), None)
        if match:
            parts[-1] = match
            value = ", ".join([p for p in parts if p])
            self.var_compare_values.set(value)
            self.entry_compare_values.icursor(len(value))
        self._refresh_compare_hint()
        return "break"

    def _on_compare_toggle(self) -> None:
        if self.var_compare_on.get() and self.var_animation_on.get():
            self.var_animation_on.set(False)

    def _add_compare(self) -> None:
        key = self.var_compare_param.get().strip()
        values = [v.strip() for v in self.var_compare_values.get().split(",") if v.strip()]
        if not key or not values:
            return
        self.compare_params[key] = values
        self._refresh_compare_summary()
        self.var_compare_values.set("")
        self._refresh_compare_hint()

    def _clear_compare(self) -> None:
        self.compare_params.clear()
        self._refresh_compare_summary()

    def _refresh_compare_summary(self) -> None:
        self.compare_summary.configure(state="normal")
        self.compare_summary.delete("1.0", "end")
        for key, values in self.compare_params.items():
            self.compare_summary.insert("end", f"{key}: [{', '.join(values)}]\n")
        self.compare_summary.configure(state="disabled")

    def _on_policy_change(self) -> None:
        self._cache_policy_values(self._active_policy)
        self._active_policy = self.var_policy.get()
        self._render_specific_fields()
        self._refresh_compare_dropdown()

    def _refresh_compare_dropdown(self) -> None:
        values = self._compare_param_values()
        self.cmb_compare_param.configure(values=values)
        if self.var_compare_param.get() not in values:
            self.var_compare_param.set(values[0])

    def _cache_policy_values(self, policy: Optional[str] = None) -> None:
        policy = policy or self.var_policy.get()
        if policy not in self.policy_cache:
            self.policy_cache[policy] = {}
        for key, var in self._specific_vars.items():
            self.policy_cache[policy][key] = var.get()

    def _render_specific_fields(self) -> None:
        for name, widget in list(self._specific_widgets.items()):
            widget.destroy()
            del self._specific_widgets[name]
        self._specific_vars.clear()

        policy = self.var_policy.get()
        values = dict(POLICY_DEFAULTS.get(policy, {}))
        values.update(self.policy_cache.get(policy, {}))

        row = 1
        for idx, key in enumerate(self.shared_keys):
            col_pair = idx % 2
            row = self._create_specific_field(row, key, values.get(key, ""), readonly_options=self._readonly_options(key), pair=col_pair)
            if col_pair == 1:
                row += 1

        if len(self.shared_keys) % 2 == 1:
            row += 1

        sep = ttk.Separator(self.specific_group, orient="horizontal")
        sep.grid(row=row, column=0, columnspan=4, sticky="ew", padx=4, pady=6)
        self._specific_widgets[f"sep_{policy}"] = sep
        row += 1

        specific_keys = sorted(k for k in values.keys() if k not in self.shared_keys)
        for idx, key in enumerate(specific_keys):
            col_pair = idx % 2
            row = self._create_specific_field(row, key, values.get(key, ""), readonly_options=None, pair=col_pair)
            if col_pair == 1:
                row += 1

    def _readonly_options(self, key: str) -> Optional[List[str]]:
        if key == "activation":
            return ["ReLU", "Tanh"]
        if key == "lr_strategy":
            return ["constant", "linear", "exponential"]
        return None

    def _create_specific_field(self, row: int, key: str, value: Any, readonly_options: Optional[List[str]], pair: int = 0) -> int:
        base_col = 0 if pair <= 0 else 2
        ttk.Label(self.specific_group, text=key, style="Panel.TLabel").grid(row=row, column=base_col, sticky="w", padx=4, pady=4)
        if readonly_options:
            var = tk.StringVar(value=str(value))
            widget = ttk.Combobox(self.specific_group, state="readonly", textvariable=var, values=readonly_options)
            self._style_combobox(widget)
            widget.grid(row=row, column=base_col + 1, sticky="ew", padx=4, pady=4)
        else:
            var = tk.StringVar(value=str(value))
            widget = tk.Entry(self.specific_group, textvariable=var)
            self._style_entry(widget)
            widget.grid(row=row, column=base_col + 1, sticky="ew", padx=4, pady=4)

        self._specific_vars[key] = var
        self._specific_widgets[key] = widget

        hints = {
            "gamma": "Higher gamma values emphasize long-term reward and can improve final return at slower feedback speed.",
            "learning_rate": "Larger learning rates speed updates but may destabilize training if too high.",
            "batch_size": "Larger batch sizes reduce update noise but increase per-update compute cost.",
            "hidden_layer": "Wider or deeper networks can model harder policies but require more compute and data.",
            "activation": "Activation affects optimization dynamics; Tanh is smoother while ReLU can train faster on some tasks.",
            "lr_strategy": "Learning-rate schedules trade early speed for later stability.",
            "min_lr": "Minimum learning rate sets how far schedules can decay before updates become tiny.",
            "lr_decay": "Higher decay shrinks learning rate faster in exponential mode, often improving late stability.",
            "train_freq": "Higher train frequency updates more often and increases compute intensity.",
            "gradient_steps": "More gradient steps per rollout improve fit but can bottleneck runtime.",
            "buffer_size": "Larger replay buffers improve diversity but consume more memory.",
            "learning_starts": "Higher warmup delays learning updates to build a broader initial replay buffer.",
            "tau": "Lower tau makes target updates smoother and usually more stable.",
            "policy_delay": "Higher policy delay updates actor less often, which can stabilize TD3 learning.",
            "n_steps": "Larger PPO rollout steps improve gradient estimates but increase update latency.",
        }
        if key in hints:
            self._attach_tooltip(widget, hints[key])
        return row

    def _update_environment(self) -> None:
        # Environment updates apply to future runs.
        self.status_var.set("Environment parameters updated")

    def _flush_event_queue(self) -> None:
        while True:
            try:
                self.event_queue.get_nowait()
            except Empty:
                break

    def _build_env_config(self) -> EnvironmentConfig:
        return EnvironmentConfig(
            env_id="Reacher-v5",
            reward_dist_weight=float(self.var_reward_dist_weight.get()),
            reward_control_weight=float(self.var_reward_control_weight.get()),
            render_enabled=bool(self.var_animation_on.get()),
        )

    def _build_train_config(
        self,
        run_id: str,
        policy_override: Optional[str] = None,
        use_policy_cache: bool = True,
    ) -> TrainingConfig:
        policy = policy_override or self.var_policy.get()
        values = dict(POLICY_DEFAULTS.get(policy, {}))
        if use_policy_cache:
            values.update(self.policy_cache.get(policy, {}))
        if use_policy_cache and policy == self.var_policy.get():
            values.update({k: v.get() for k, v in self._specific_vars.items()})

        return TrainingConfig(
            policy=policy,
            episodes=int(self.var_episodes.get()),
            max_steps=int(self.var_max_steps.get()),
            gamma=float(values.get("gamma", POLICY_DEFAULTS[policy]["gamma"])),
            learning_rate=float(values.get("learning_rate", POLICY_DEFAULTS[policy]["learning_rate"])),
            batch_size=int(values.get("batch_size", POLICY_DEFAULTS[policy]["batch_size"])),
            hidden_layer=str(values.get("hidden_layer", POLICY_DEFAULTS[policy]["hidden_layer"])),
            activation=str(values.get("activation", POLICY_DEFAULTS[policy]["activation"])),
            lr_strategy=str(values.get("lr_strategy", POLICY_DEFAULTS[policy]["lr_strategy"])),
            min_lr=float(values.get("min_lr", POLICY_DEFAULTS[policy]["min_lr"])),
            lr_decay=float(values.get("lr_decay", POLICY_DEFAULTS[policy]["lr_decay"])),
            train_freq=int(values.get("train_freq", 1)),
            gradient_steps=int(values.get("gradient_steps", 1)),
            buffer_size=int(values.get("buffer_size", 100000)),
            learning_starts=int(values.get("learning_starts", 1000)),
            tau=float(values.get("tau", 0.005)),
            policy_delay=int(values.get("policy_delay", 2)),
            update_rate_episodes=int(self.var_update_rate.get()),
            frame_stride=int(self.var_frame_stride.get()),
            animation_fps=int(self.var_animation_fps.get()),
            moving_average_window=int(self.var_moving_average.get()),
            eval_rollout_on=bool(self.var_eval_rollout_on.get()),
            device=self.var_device.get(),
            run_id=run_id,
        )

    def _cancel_all_workers(self) -> None:
        with self.worker_lock:
            handles = list(self.active_workers.values())
        for handle in handles:
            handle.trainer.cancel()

    def _set_pause_for_workers(self, paused: bool) -> None:
        with self.worker_lock:
            handles = list(self.active_workers.values())
        for handle in handles:
            handle.trainer.set_pause(paused)

    def _register_worker(self, handle: WorkerHandle) -> None:
        with self.worker_lock:
            self.active_workers[handle.run_id] = handle

    def _start_worker(self, handle: WorkerHandle) -> None:
        self._register_worker(handle)

        def _run() -> None:
            if torch is not None:
                try:
                    torch.set_num_threads(max(1, int(handle.cpu_threads)))
                except Exception:
                    pass
            handle.trainer.train()

        self.executor.submit(_run)

    def _single_event_sink(self, session_id: int):
        def sink(payload: Dict[str, Any]) -> None:
            payload["session_id"] = session_id
            self.event_queue.put(payload)

        return sink

    def _run_single_episode(self) -> None:
        self._cancel_all_workers()
        self._flush_event_queue()
        self.current_session_id += 1
        session_id = self.current_session_id
        run_id = f"single_{datetime.now().strftime('%H%M%S')}"

        env_config = self._build_env_config()
        train_config = self._build_train_config(run_id=run_id)
        train_config.episodes = 1

        trainer = ReacherTrainer(env_config, train_config, event_sink=self._single_event_sink(session_id))
        self._start_worker(WorkerHandle(trainer=trainer, run_id=run_id, session_id=session_id, cpu_threads=1))

        self.is_training = True
        self.is_paused = False
        self._update_button_highlights()
        self._snapshot_run_metadata(run_id, train_config, env_config)

    def _train_and_run(self) -> None:
        if self.is_paused:
            self._cancel_all_workers()
            self.is_paused = False
        self._flush_event_queue()

        self.current_session_id += 1
        session_id = self.current_session_id
        self.is_training = True

        if self.var_compare_on.get() and self.compare_params:
            self._start_compare_runs(session_id)
        else:
            run_id = f"run_{datetime.now().strftime('%H%M%S')}"
            env_config = self._build_env_config()
            train_config = self._build_train_config(run_id)
            trainer = ReacherTrainer(env_config, train_config, event_sink=self._single_event_sink(session_id))
            self._start_worker(WorkerHandle(trainer=trainer, run_id=run_id, session_id=session_id, cpu_threads=1))
            self._snapshot_run_metadata(run_id, train_config, env_config)

        self._update_button_highlights()

    def _start_compare_runs(self, session_id: int) -> None:
        grid = self._build_compare_grid()
        if not grid:
            return

        worker_count = min(4, max(1, len(grid)))
        cpu_total = max(1, int(os.cpu_count() or 1))
        cpu_threads = max(1, cpu_total // worker_count)

        selected_policy = self.var_policy.get()
        compare_has_policy_axis = "Policy" in self.compare_params
        render_idx = self._select_render_combo_index(grid, selected_policy)
        for idx, combo in enumerate(grid):
            run_id = f"cmp_{idx}_{datetime.now().strftime('%H%M%S')}"
            env_config = self._build_env_config()
            target_policy = combo.get("Policy", selected_policy)
            train_config = self._build_train_config(
                run_id,
                policy_override=target_policy,
                use_policy_cache=not compare_has_policy_axis,
            )
            train_config.policy = target_policy

            allowed = self._allowed_compare_keys(target_policy)

            for key, value in combo.items():
                if key not in allowed:
                    continue
                if not hasattr(train_config, key):
                    continue
                target_type = type(getattr(train_config, key))
                try:
                    cast_val = target_type(value)
                except Exception:
                    cast_val = value
                setattr(train_config, key, cast_val)

            env_config.render_enabled = bool(self.var_animation_on.get()) and idx == render_idx

            trainer = ReacherTrainer(env_config, train_config, event_sink=self._single_event_sink(session_id))
            self._start_worker(
                WorkerHandle(trainer=trainer, run_id=run_id, session_id=session_id, cpu_threads=cpu_threads)
            )
            self._snapshot_run_metadata(run_id, train_config, env_config, combo)

    def _allowed_compare_keys(self, policy: str) -> set[str]:
        shared = {
            "Policy",
            "max_steps",
            "episodes",
            "gamma",
            "learning_rate",
            "batch_size",
            "hidden_layer",
            "activation",
            "lr_strategy",
            "min_lr",
            "lr_decay",
        }
        specific = set(k for k in POLICY_DEFAULTS.get(policy, {}).keys() if k not in shared)
        return shared | specific

    def _select_render_combo_index(self, grid: List[Dict[str, str]], selected_policy: str) -> int:
        for i, combo in enumerate(grid):
            if combo.get("Policy") == selected_policy:
                return i
        return 0

    def _build_compare_grid(self) -> List[Dict[str, str]]:
        if not self.compare_params:
            return []

        keys = list(self.compare_params.keys())
        grid: List[Dict[str, str]] = []

        def _recurse(i: int, current: Dict[str, str]) -> None:
            if i >= len(keys):
                grid.append(dict(current))
                return
            key = keys[i]
            for val in self.compare_params[key]:
                current[key] = val
                _recurse(i + 1, current)

        _recurse(0, {})
        return grid

    def _snapshot_run_metadata(self, run_id: str, train_cfg: TrainingConfig, env_cfg: EnvironmentConfig, compare_values: Optional[Dict[str, str]] = None) -> None:
        self.run_metadata[run_id] = {
            "policy": train_cfg.policy,
            "max_steps": train_cfg.max_steps,
            "gamma": train_cfg.gamma,
            "learning_rate": train_cfg.learning_rate,
            "lr_strategy": train_cfg.lr_strategy,
            "lr_decay": train_cfg.lr_decay,
            "eval_rollout_on": bool(train_cfg.eval_rollout_on),
            "env": {
                "reward_dist_weight": env_cfg.reward_dist_weight,
                "reward_control_weight": env_cfg.reward_control_weight,
            },
            "compare": compare_values or {},
        }

    def _toggle_pause(self) -> None:
        if not self.is_training:
            return
        self.is_paused = not self.is_paused
        self._set_pause_for_workers(self.is_paused)
        self._update_button_highlights()

    def _reset_all(self) -> None:
        self._cancel_all_workers()
        self.is_training = False
        self.is_paused = False
        self._update_button_highlights()

        self.var_animation_on.set(True)
        self.var_animation_fps.set("30")
        self.var_update_rate.set("1")
        self.var_frame_stride.set("2")
        self.var_max_steps.set("200")
        self.var_episodes.set("3000")
        self.var_device.set("CPU")
        self.var_moving_average.set("20")
        self.var_eval_rollout_on.set(False)
        self.var_reward_dist_weight.set("1.0")
        self.var_reward_control_weight.set("0.1")

        self.var_policy.set("SAC")
        self.policy_cache = {k: dict(v) for k, v in POLICY_DEFAULTS.items()}
        self._render_specific_fields()

        self.steps_progress.configure(value=0, maximum=1)
        self.episodes_progress.configure(value=0, maximum=1)
        self.steps_var.set("Playback frames: 0/0")
        self.episodes_var.set("Episodes: 0/0")
        self.status_var.set("Epsilon: n/a | LR: n/a | Best reward: n/a | Render: idle")

        self._active_playback = False
        self._pending_playback = None
        self._current_playback_frames = []
        self._playback_index = 0
        self.render_canvas.delete("all")

    def _clear_plot(self) -> None:
        self.run_plot_data.clear()
        self.run_plot_artists.clear()
        self.run_visibility.clear()
        self._style_plot_axes()
        self.legend = None
        self.canvas_plot.draw_idle()

    def _save_csv(self) -> None:
        with self.worker_lock:
            handles = list(self.active_workers.values())
        saved: List[str] = []
        for handle in handles:
            path = handle.trainer.export_transitions_csv(output_dir="results_csv")
            if path is not None:
                saved.append(str(path))

        if saved:
            messagebox.showinfo("CSV export", "Saved:\n" + "\n".join(saved))
        else:
            messagebox.showinfo("CSV export", "No samples available yet.")

    def _save_png(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("plots") / f"reacher_plot_{stamp}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.canvas_plot.figure.savefig(path, dpi=150)
        messagebox.showinfo("PNG export", f"Saved: {path}")

    def _update_button_highlights(self) -> None:
        if self.is_training and not self.is_paused:
            self.btn_train.configure(style="Train.TButton")
            self.btn_pause.configure(style="Neutral.TButton", text="Pause")
        elif self.is_training and self.is_paused:
            self.btn_train.configure(style="Neutral.TButton")
            self.btn_pause.configure(style="Pause.TButton", text="Run")
        else:
            self.btn_train.configure(style="Neutral.TButton")
            self.btn_pause.configure(style="Neutral.TButton", text="Pause")

    def _ui_pump(self) -> None:
        batch_limit = 1 if self.strict_visual_mode.get() else 100
        handled = 0
        needs_draw = False

        while handled < batch_limit:
            try:
                event = self.event_queue.get_nowait()
            except Empty:
                break

            handled += 1
            if event.get("session_id") != self.current_session_id:
                continue

            etype = event.get("type")
            if etype == "step":
                self._handle_step_event(event)
            elif etype == "episode":
                self._handle_episode_event(event)
                needs_draw = True
            elif etype == "episode_aux":
                self._handle_episode_aux_event(event)
                needs_draw = True
            elif etype == "training_done":
                self._handle_training_done(event)
            elif etype == "error":
                self._handle_error_event(event)

        if needs_draw:
            if self.strict_visual_mode.get():
                self.canvas_plot.draw()
            else:
                self.canvas_plot.draw_idle()

        interval = 16 if self.strict_visual_mode.get() else 50
        self.after(interval, self._ui_pump)

    def _handle_step_event(self, event: Dict[str, Any]) -> None:
        # Steps progress is reserved for replay playback progression.
        _ = event

    def _handle_episode_event(self, event: Dict[str, Any]) -> None:
        run_id = event["run_id"]
        ep = int(event.get("episode", 0))
        total = int(event.get("episodes", 1))
        reward = float(event.get("reward", 0.0))
        moving_avg = float(event.get("moving_average", 0.0))

        self.episodes_progress.configure(maximum=total, value=ep)
        self.episodes_var.set(f"Episodes: {ep}/{total}")
        self.status_var.set(
            f"Epsilon: {event.get('epsilon', 'n/a')} | LR: {event.get('lr', 'n/a')} | "
            f"Best reward: {event.get('best_reward', 'n/a')} | Render: {event.get('render_state', 'idle')}"
        )

        if run_id not in self.run_plot_data:
            self.run_plot_data[run_id] = {"x": [], "reward": [], "ma": [], "eval_x": [], "eval_y": []}
            self.run_visibility.setdefault(run_id, True)

        d = self.run_plot_data[run_id]
        d["x"].append(ep)
        d["reward"].append(reward)
        d["ma"].append(moving_avg)

        self._redraw_plot(run_id)

    def _handle_episode_aux_event(self, event: Dict[str, Any]) -> None:
        run_id = event["run_id"]
        eval_points = event.get("eval_points") or []
        if run_id in self.run_plot_data:
            self.run_plot_data[run_id]["eval_x"] = [p[0] for p in eval_points]
            self.run_plot_data[run_id]["eval_y"] = [p[1] for p in eval_points]

        frames = event.get("frames") or []
        if frames:
            if self._active_playback:
                self._pending_playback = frames
            else:
                self._start_playback(frames)

        self._redraw_plot(run_id)

    def _handle_training_done(self, _event: Dict[str, Any]) -> None:
        with self.worker_lock:
            self.active_workers = {k: v for k, v in self.active_workers.items() if v.session_id == self.current_session_id}
            self.is_training = bool(self.active_workers)
        self.is_paused = False
        self._update_button_highlights()

    def _handle_error_event(self, event: Dict[str, Any]) -> None:
        messagebox.showerror("Training error", event.get("error", "Unknown error"))
        self.is_training = False
        self.is_paused = False
        self._update_button_highlights()

    def _start_playback(self, frames: List[Any]) -> None:
        self._active_playback = True
        self._current_playback_frames = frames
        self._playback_index = 0
        total = max(1, len(frames))
        self.steps_progress.configure(maximum=total, value=0)
        self.steps_var.set(f"Playback frames: 0/{total}")
        self._playback_tick()

    def _playback_tick(self) -> None:
        if not self._active_playback:
            return
        if self._playback_index >= len(self._current_playback_frames):
            self._active_playback = False
            if self._pending_playback is not None:
                pending = self._pending_playback
                self._pending_playback = None
                self._start_playback(pending)
            return

        frame = self._current_playback_frames[self._playback_index]
        self._playback_index += 1
        self._draw_frame(frame)
        total = max(1, len(self._current_playback_frames))
        self.steps_progress.configure(maximum=total, value=self._playback_index)
        self.steps_var.set(f"Playback frames: {self._playback_index}/{total}")

        fps = max(1, int(self.var_animation_fps.get() or 30))
        delay = int(1000 / fps)
        self.after(delay, self._playback_tick)

    def _draw_frame(self, frame: Any) -> None:
        if Image is None or ImageTk is None:
            return
        if frame is None:
            return

        canvas_w = max(1, self.render_canvas.winfo_width())
        canvas_h = max(1, self.render_canvas.winfo_height())
        img = Image.fromarray(frame)
        img.thumbnail((canvas_w, canvas_h))
        self._render_photo = ImageTk.PhotoImage(img)

        self.render_canvas.delete("all")
        self.render_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._render_photo, anchor="center")

    def _redraw_current_frame(self) -> None:
        if self._playback_index > 0 and self._playback_index <= len(self._current_playback_frames):
            self._draw_frame(self._current_playback_frames[self._playback_index - 1])

    def _format_run_label(self, run_id: str) -> str:
        md = self.run_metadata.get(run_id, {})
        env = md.get("env", {})
        label = (
            f"{md.get('policy', 'Policy')} | steps={md.get('max_steps', '?')} | gamma={md.get('gamma', '?')}\n"
            f"epsilon=n/a | epsilon_decay=n/a | epsilon_min=n/a\n"
            f"LR={md.get('learning_rate', '?')} | LR strategy={md.get('lr_strategy', '?')} | LR decay={md.get('lr_decay', '?')}\n"
            f"reward_dist_weight={env.get('reward_dist_weight', '?')} | reward_control_weight={env.get('reward_control_weight', '?')}"
        )

        compare = md.get("compare", {})
        for k, v in compare.items():
            if k.lower() not in label.lower():
                label += f" | {k}={v}"
        return label

    def _redraw_plot(self, run_id: str) -> None:
        d = self.run_plot_data[run_id]
        if run_id not in self.run_plot_artists:
            color = None
            reward_line, = self.ax.plot([], [], alpha=0.3, linewidth=1.2, label=self._format_run_label(run_id), color=color)
            ma_line, = self.ax.plot([], [], linestyle="-", linewidth=2.4, alpha=1.0, label="moving average", color=reward_line.get_color())
            eval_line, = self.ax.plot([], [], linestyle=":", marker="o", linewidth=2.4, alpha=1.0, label="evaluation rollout", color=reward_line.get_color())
            self.run_plot_artists[run_id] = {
                "reward": reward_line,
                "ma": ma_line,
                "eval": eval_line,
            }

        artists = self.run_plot_artists[run_id]
        artists["reward"].set_data(d["x"], d["reward"])
        artists["ma"].set_data(d["x"], d["ma"])
        artists["eval"].set_data(d["eval_x"], d["eval_y"])

        visible = self.run_visibility.get(run_id, True)
        artists["reward"].set_visible(visible)
        artists["ma"].set_visible(visible)
        eval_enabled = bool(self.run_metadata.get(run_id, {}).get("eval_rollout_on", False))
        artists["eval"].set_visible(visible and eval_enabled)

        self.ax.relim()
        self.ax.autoscale_view()
        self._draw_legend()

    def _draw_legend(self) -> None:
        self._legend_pick_to_run.clear()
        handles = []
        labels = []
        for run_id, artists in self.run_plot_artists.items():
            handles.extend([artists["reward"], artists["ma"], artists["eval"]])
            labels.extend([artists["reward"].get_label(), "moving average", "evaluation rollout"])

        if self.legend is not None:
            self.legend.remove()

        self.legend = self.ax.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0 + self._legend_scroll_offset),
            frameon=False,
            labelcolor=PALETTE["text"],
            handlelength=3.0,
        )

        run_ids = list(self.run_plot_artists.keys())
        for idx, (text, handle) in enumerate(zip(self.legend.get_texts(), self.legend.legend_handles)):
            text.set_picker(True)
            handle.set_picker(True)
            if idx % 3 == 0 and (idx // 3) < len(run_ids):
                run_id = run_ids[idx // 3]
                self._legend_pick_to_run[text] = run_id
                self._legend_pick_to_run[handle] = run_id

    def _on_legend_pick(self, event: Any) -> None:
        if self.legend is None:
            return
        run_id = self._legend_pick_to_run.get(event.artist)
        if run_id is None:
            return
        self.run_visibility[run_id] = not self.run_visibility.get(run_id, True)
        for rid, artists in self.run_plot_artists.items():
            visible = self.run_visibility.get(rid, True)
            for artist in artists.values():
                artist.set_visible(visible)
        self._draw_legend()
        self.canvas_plot.draw_idle()

    def _on_plot_hover(self, event: Any) -> None:
        if self.legend is None or event is None:
            return
        over = False
        for text in self.legend.get_texts():
            contains, _ = text.contains(event)
            if contains:
                over = True
                text.set_color(PALETTE["accent"])
            else:
                text.set_color(PALETTE["text"])
        self.canvas_plot.get_tk_widget().configure(cursor="hand2" if over else "")

    def _on_legend_scroll(self, event: Any) -> None:
        if self.legend is None or event is None:
            return
        contains, _ = self.legend.contains(event)
        if not contains:
            return

        label_count = len(self.legend.get_texts())
        visible_rows = 12
        if label_count <= visible_rows:
            return

        max_shift = (label_count - visible_rows) * 0.05
        step = 0.08 if getattr(event, "step", 0) < 0 else -0.08
        self._legend_scroll_offset = max(-max_shift, min(0.0, self._legend_scroll_offset + step))
        self._draw_legend()
        self.canvas_plot.draw_idle()


def build_gui_root() -> tk.Tk:
    root = tk.Tk()
    ReacherGUI(root)
    return root
