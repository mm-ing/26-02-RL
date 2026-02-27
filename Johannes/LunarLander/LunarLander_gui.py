from __future__ import annotations

import os
import threading
import time
import tkinter as tk
from dataclasses import asdict
from datetime import datetime
from tkinter import messagebox, ttk
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from LunarLander_logic import POLICY_DEFAULTS, Trainer

matplotlib.use("TkAgg")

try:
    from PIL import Image, ImageTk
except Exception:
    Image = None
    ImageTk = None


class LunarLanderGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LunarLander RL")
        self.root.geometry("1300x900")

        self.PAD_OUTER = 10
        self.PAD_INNER = 6
        self.PAD_TIGHT = 4
        self.LABEL_COL_WIDTH = 92
        self.PARAM_INPUT_WIDTH = 9

        self._configure_styles()

        self.trainer = Trainer()
        self._pending_lock = threading.Lock()
        self._pending_state: Dict[str, object] = {}
        self._pending_policy_state: Dict[str, Dict[str, object]] = {}
        self._stop_requested = threading.Event()
        self._pause_requested = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._workers: Dict[str, threading.Thread] = {}
        self._policy_trainers: Dict[str, Trainer] = {}
        self._compare_mode_active = False
        self._compare_run_meta: Dict[str, Dict[str, float]] = {}
        self._latest_compare_rewards: Dict[str, List[float]] = {}
        self._compare_finalized_policies: set[str] = set()

        self._latest_rewards_snapshot: Optional[List[float]] = None
        self._latest_episode = 0
        self._latest_step = 0
        self._last_plot_update = 0.0
        self._last_render_update = 0.0
        self._last_rendered_step = -1
        self._last_rendered_episode = -1

        self._plot_runs: List[Dict[str, object]] = []
        self._run_counter = 0
        self._current_x = None
        self._best_x = None

        self._legend_map: Dict[object, object] = {}
        self._preview_single_lines: Optional[tuple] = None
        self._preview_compare_lines: Dict[str, tuple] = {}

        self._build_variables()
        self._build_layout()
        self._apply_policy_defaults(self.policy_var.get())
        self._selected_render_trainer = self.trainer
        self._set_status_text(self.epsilon_max_var.get(), None, None)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(40, self._ui_pump)
        self.root.after(50, self._render_tick)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        available = set(style.theme_names())
        if "clam" in available:
            style.theme_use("clam")
        elif "vista" in available:
            style.theme_use("vista")

        default_font = ("Segoe UI", 10)
        heading_font = ("Segoe UI", 10, "bold")
        button_font = ("Segoe UI", 10, "bold")

        bg_main = "#1e1e1e"
        bg_panel = "#252526"
        bg_input = "#2d2d30"
        fg_text = "#e6e6e6"
        fg_muted = "#d0d0d0"
        accent = "#0e639c"

        self.root.configure(bg=bg_main)
        self.root.option_add("*TCombobox*Listbox*Background", bg_input)
        self.root.option_add("*TCombobox*Listbox*Foreground", fg_text)
        self.root.option_add("*TCombobox*Listbox*selectBackground", accent)
        self.root.option_add("*TCombobox*Listbox*selectForeground", "white")

        style.configure("TFrame", background=bg_main)
        style.configure("TLabelframe", background=bg_panel, foreground=fg_text, padding=(8, 8))
        style.configure("TLabelframe.Label", background=bg_panel, foreground=fg_text, font=heading_font)
        style.configure("TLabel", background=bg_panel, foreground=fg_text, font=default_font)

        style.configure("TButton", background="#3a3d41", foreground=fg_text, font=button_font, padding=(10, 5))
        style.map("TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")], foreground=[("disabled", "#b8b8b8")])

        style.configure("Primary.TButton", background=accent, foreground="white", font=button_font, padding=(10, 5))
        style.map("Primary.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")], foreground=[("disabled", "#ededed")])

        style.configure("Pause.TButton", background="#a66a00", foreground="white", font=button_font, padding=(10, 5))
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")], foreground=[("disabled", "#ededed")])

        style.configure("TEntry", fieldbackground=bg_input, foreground=fg_text, insertcolor=fg_text, padding=(6, 4))
        style.configure("TCombobox", fieldbackground=bg_input, foreground=fg_text, padding=(6, 4))
        style.map("TCombobox", fieldbackground=[("readonly", bg_input)], foreground=[("readonly", fg_text)])
        style.configure("TCheckbutton", background=bg_panel, foreground=fg_text)
        style.map("TCheckbutton", background=[("active", bg_panel), ("!active", bg_panel)], foreground=[("active", fg_text), ("!active", fg_text)])
        style.configure("TProgressbar", troughcolor="#343434", background=accent)
        style.configure("Vertical.TScrollbar", background="#3a3d41", troughcolor="#252526", arrowcolor=fg_muted)

    def _build_variables(self) -> None:
        self.policy_var = tk.StringVar(value="D3QN")

        self.animation_fps_var = tk.IntVar(value=10)
        self.animation_on_var = tk.BooleanVar(value=True)
        self.gravity_var = tk.DoubleVar(value=-10.0)
        self.wind_on_var = tk.BooleanVar(value=False)
        self.wind_power_var = tk.DoubleVar(value=15.0)
        self.turbulence_var = tk.DoubleVar(value=1.5)
        self.compare_on_var = tk.BooleanVar(value=False)
        self.compare_on_var.trace_add("write", self._on_compare_toggle_changed)

        self.max_steps_var = tk.IntVar(value=1000)
        self.episodes_var = tk.IntVar(value=5000)
        self.epsilon_max_var = tk.DoubleVar(value=1.0)
        self.epsilon_decay_var = tk.DoubleVar(value=0.995)
        self.epsilon_min_var = tk.DoubleVar(value=0.05)

        self.gamma_var = tk.DoubleVar(value=0.99)
        self.learning_rate_var = tk.DoubleVar(value=0.001)
        self.replay_size_var = tk.IntVar(value=50000)
        self.batch_size_var = tk.IntVar(value=64)
        self.target_update_var = tk.IntVar(value=100)
        self.replay_warmup_var = tk.IntVar(value=1000)
        self.learning_cadence_var = tk.IntVar(value=2)
        self.activation_var = tk.StringVar(value="ReLU")
        self.hidden_layers_var = tk.StringVar(value="128")

        self.moving_avg_var = tk.IntVar(value=20)

        self.steps_progress_var = tk.DoubleVar(value=0)
        self.episodes_progress_var = tk.DoubleVar(value=0)
        self.status_var = tk.StringVar(value="")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=4)
        self.root.columnconfigure(1, weight=0)
        self.root.rowconfigure(0, weight=5)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=4)

        self.env_frame = ttk.LabelFrame(self.root, text="Environment")
        self.env_frame.grid(row=0, column=0, sticky="nsew", padx=self.PAD_OUTER, pady=self.PAD_OUTER)
        self.env_frame.rowconfigure(0, weight=1)
        self.env_frame.columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(self.env_frame, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew")
        self.render_canvas.bind("<Configure>", lambda _e: self._draw_latest_frame())
        self._tk_img = None
        self._canvas_image_id = None

        self.params_frame = ttk.LabelFrame(self.root, text="Parameters")
        self.params_frame.grid(row=0, column=1, sticky="ns", padx=(0, self.PAD_OUTER), pady=self.PAD_OUTER)
        self.params_frame.rowconfigure(0, weight=1)
        self.params_frame.columnconfigure(0, weight=1)

        self._build_scrollable_params()

        self.controls_frame = ttk.LabelFrame(self.root, text="Controls")
        self.controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        for i in range(7):
            self.controls_frame.columnconfigure(i, weight=1)

        self.btn_single = ttk.Button(self.controls_frame, text="Run single episode", command=self.run_single_episode)
        self.btn_train = ttk.Button(self.controls_frame, text="Train and Run", command=self.train_and_run)
        self.btn_pause = ttk.Button(self.controls_frame, text="Pause", command=self.toggle_pause)
        self.btn_reset = ttk.Button(self.controls_frame, text="Reset All", command=self.reset_all)
        self.btn_clear = ttk.Button(self.controls_frame, text="Clear Plot", command=self.clear_plot)
        self.btn_csv = ttk.Button(self.controls_frame, text="Save samplings CSV", command=self.save_samplings_csv)
        self.btn_png = ttk.Button(self.controls_frame, text="Save Plot PNG", command=self.save_plot_png)

        buttons = [self.btn_single, self.btn_train, self.btn_pause, self.btn_reset, self.btn_clear, self.btn_csv, self.btn_png]
        for idx, button in enumerate(buttons):
            button.grid(row=0, column=idx, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        self._update_control_highlights()

        self.current_frame = ttk.LabelFrame(self.root, text="Current Run")
        self.current_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        self.current_frame.columnconfigure(0, weight=0, minsize=self.LABEL_COL_WIDTH)
        self.current_frame.columnconfigure(1, weight=1)

        ttk.Label(self.current_frame, text="Steps").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        self.steps_bar = ttk.Progressbar(self.current_frame, orient="horizontal", mode="determinate", variable=self.steps_progress_var)
        self.steps_bar.grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)

        ttk.Label(self.current_frame, text="Episodes").grid(row=1, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        self.episodes_bar = ttk.Progressbar(self.current_frame, orient="horizontal", mode="determinate", variable=self.episodes_progress_var)
        self.episodes_bar.grid(row=1, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)

        ttk.Label(self.current_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=self.PAD_TIGHT, pady=2)

        self.plot_frame = ttk.LabelFrame(self.root, text="Live Plot")
        self.plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor("#1e1e1e")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.xaxis.label.set_color("#e2e2e2")
        self.ax.yaxis.label.set_color("#e2e2e2")
        self.ax.tick_params(axis="x", colors="#dddddd")
        self.ax.tick_params(axis="y", colors="#dddddd")
        self.ax.grid(True, alpha=0.3, linewidth=0.9)
        for spine in self.ax.spines.values():
            spine.set_alpha(0.5)
            spine.set_color("#9a9a9a")
        self.figure.subplots_adjust(right=0.75)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_canvas.mpl_connect("pick_event", self._on_legend_pick)

    def _build_scrollable_params(self) -> None:
        self.params_canvas = tk.Canvas(self.params_frame, highlightthickness=0, width=430, bg="#1e1e1e")
        self.params_canvas.grid(row=0, column=0, sticky="nsew")
        self.params_scroll = ttk.Scrollbar(self.params_frame, orient="vertical", command=self.params_canvas.yview)
        self.params_canvas.configure(yscrollcommand=self.params_scroll.set)

        self.params_inner = ttk.Frame(self.params_canvas)
        self.params_window = self.params_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")

        self.params_inner.bind("<Configure>", self._on_params_configure)
        self.params_canvas.bind("<Configure>", self._on_params_canvas_configure)

        self.params_canvas.bind("<Enter>", self._bind_mousewheel)
        self.params_canvas.bind("<Leave>", self._unbind_mousewheel)

        self._build_parameter_groups()

    def _build_parameter_groups(self) -> None:
        env_group = ttk.LabelFrame(self.params_inner, text="Environment")
        env_group.grid(row=0, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        env_group.columnconfigure(1, weight=1)
        env_group.columnconfigure(3, weight=1)

        self._add_pair(env_group, 0, "Animation on", ttk.Checkbutton(env_group, variable=self.animation_on_var), "Animation FPS", ttk.Entry(env_group, textvariable=self.animation_fps_var, width=self.PARAM_INPUT_WIDTH))
        ttk.Button(env_group, text="Update", command=self.update_environment).grid(row=1, column=0, columnspan=4, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        self._add_pair(env_group, 2, "Gravity", ttk.Entry(env_group, textvariable=self.gravity_var, width=self.PARAM_INPUT_WIDTH), "Wind on", ttk.Checkbutton(env_group, variable=self.wind_on_var))
        self._add_pair(env_group, 3, "Wind power", ttk.Entry(env_group, textvariable=self.wind_power_var, width=self.PARAM_INPUT_WIDTH), "Turbulence", ttk.Entry(env_group, textvariable=self.turbulence_var, width=self.PARAM_INPUT_WIDTH))

        compare_group = ttk.LabelFrame(self.params_inner, text="Compare")
        compare_group.grid(row=1, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        compare_group.columnconfigure(1, weight=1)
        compare_group.columnconfigure(3, weight=1)
        self._add_pair(compare_group, 0, "Compare on", ttk.Checkbutton(compare_group, variable=self.compare_on_var), "", ttk.Label(compare_group, text=""))

        general_group = ttk.LabelFrame(self.params_inner, text="General")
        general_group.grid(row=2, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        general_group.columnconfigure(1, weight=1)
        general_group.columnconfigure(3, weight=1)

        ttk.Label(general_group, text="Policy").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        policy_combo = ttk.Combobox(general_group, textvariable=self.policy_var, values=["DuelingDQN", "D3QN", "DDQN+PER"], state="readonly")
        policy_combo.grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        policy_combo.bind("<<ComboboxSelected>>", self._on_policy_changed)

        self._add_pair(general_group, 1, "Max steps", ttk.Entry(general_group, textvariable=self.max_steps_var, width=self.PARAM_INPUT_WIDTH), "Episodes", ttk.Entry(general_group, textvariable=self.episodes_var, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(general_group, 2, "Epsilon max", ttk.Entry(general_group, textvariable=self.epsilon_max_var, width=self.PARAM_INPUT_WIDTH), "Epsilon decay", ttk.Entry(general_group, textvariable=self.epsilon_decay_var, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(general_group, 3, "Epsilon min", ttk.Entry(general_group, textvariable=self.epsilon_min_var, width=self.PARAM_INPUT_WIDTH), "", ttk.Label(general_group, text=""))

        specific_group = ttk.LabelFrame(self.params_inner, text="Specific")
        specific_group.grid(row=3, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        specific_group.columnconfigure(1, weight=1)
        specific_group.columnconfigure(3, weight=1)

        self._add_pair(specific_group, 0, "Gamma", ttk.Entry(specific_group, textvariable=self.gamma_var, width=self.PARAM_INPUT_WIDTH), "Learning rate", ttk.Entry(specific_group, textvariable=self.learning_rate_var, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(specific_group, 1, "Replay size", ttk.Entry(specific_group, textvariable=self.replay_size_var, width=self.PARAM_INPUT_WIDTH), "Batch size", ttk.Entry(specific_group, textvariable=self.batch_size_var, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(specific_group, 2, "Target update", ttk.Entry(specific_group, textvariable=self.target_update_var, width=self.PARAM_INPUT_WIDTH), "Replay warmup", ttk.Entry(specific_group, textvariable=self.replay_warmup_var, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(specific_group, 3, "Learning cadence", ttk.Entry(specific_group, textvariable=self.learning_cadence_var, width=self.PARAM_INPUT_WIDTH), "Activation", ttk.Combobox(specific_group, textvariable=self.activation_var, values=["ReLU", "Tanh", "LeakyReLU", "ELU"], state="readonly", width=self.PARAM_INPUT_WIDTH))
        self._add_pair(specific_group, 4, "Hidden layers", ttk.Entry(specific_group, textvariable=self.hidden_layers_var, width=self.PARAM_INPUT_WIDTH), "", ttk.Label(specific_group, text=""))

        plot_group = ttk.LabelFrame(self.params_inner, text="Live Plot")
        plot_group.grid(row=4, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        plot_group.columnconfigure(1, weight=1)
        ttk.Label(plot_group, text="Moving average values").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        ttk.Entry(plot_group, textvariable=self.moving_avg_var, width=self.PARAM_INPUT_WIDTH).grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)

    def _add_pair(self, parent: ttk.LabelFrame, row: int, l1: str, w1, l2: str, w2) -> None:
        ttk.Label(parent, text=l1).grid(row=row, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        w1.grid(row=row, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)
        ttk.Label(parent, text=l2).grid(row=row, column=2, sticky="w", padx=self.PAD_TIGHT, pady=2)
        w2.grid(row=row, column=3, sticky="ew", padx=self.PAD_TIGHT, pady=2)

    def _on_params_configure(self, _event=None) -> None:
        self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
        self._refresh_scrollbar_visibility()

    def _on_params_canvas_configure(self, event) -> None:
        self.params_canvas.itemconfigure(self.params_window, width=event.width)
        self._refresh_scrollbar_visibility()

    def _refresh_scrollbar_visibility(self) -> None:
        self.root.update_idletasks()
        bbox = self.params_canvas.bbox("all")
        if bbox is None:
            return
        content_h = bbox[3] - bbox[1]
        view_h = self.params_canvas.winfo_height()
        if content_h > view_h + 2:
            self.params_scroll.grid(row=0, column=1, sticky="ns")
            self._scroll_visible = True
        else:
            self.params_scroll.grid_forget()
            self._scroll_visible = False
            self.params_canvas.yview_moveto(0)

    def _bind_mousewheel(self, _event=None) -> None:
        self.params_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self, _event=None) -> None:
        self.params_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event) -> None:
        if not getattr(self, "_scroll_visible", False):
            return
        self.params_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_policy_changed(self, _event=None) -> None:
        self._apply_policy_defaults(self.policy_var.get())
        selected = self.policy_var.get()
        if self._compare_mode_active and selected in self._policy_trainers:
            self._selected_render_trainer = self._policy_trainers[selected]

    def _on_compare_toggle_changed(self, *_args) -> None:
        if self.compare_on_var.get():
            self.animation_on_var.set(False)

    def _apply_policy_defaults(self, policy: str) -> None:
        cfg = POLICY_DEFAULTS[policy]
        self.gamma_var.set(cfg.gamma)
        self.learning_rate_var.set(cfg.learning_rate)
        self.replay_size_var.set(cfg.replay_size)
        self.batch_size_var.set(cfg.batch_size)
        self.target_update_var.set(cfg.target_update)
        self.replay_warmup_var.set(cfg.replay_warmup)
        self.learning_cadence_var.set(cfg.learning_cadence)
        self.activation_var.set(cfg.activation_function)
        self.hidden_layers_var.set(cfg.hidden_layers)

    def _snapshot_ui(self) -> Dict[str, object]:
        return {
            "policy": self.policy_var.get(),
            "max_steps": max(1, int(self.max_steps_var.get())),
            "episodes": max(1, int(self.episodes_var.get())),
            "epsilon_max": float(self.epsilon_max_var.get()),
            "epsilon_decay": float(self.epsilon_decay_var.get()),
            "epsilon_min": float(self.epsilon_min_var.get()),
            "gamma": float(self.gamma_var.get()),
            "learning_rate": float(self.learning_rate_var.get()),
            "replay_size": max(100, int(self.replay_size_var.get())),
            "batch_size": max(1, int(self.batch_size_var.get())),
            "target_update": max(1, int(self.target_update_var.get())),
            "replay_warmup": max(1, int(self.replay_warmup_var.get())),
            "learning_cadence": max(1, int(self.learning_cadence_var.get())),
            "activation_function": self.activation_var.get(),
            "hidden_layers": self.hidden_layers_var.get(),
            "moving_avg": max(1, int(self.moving_avg_var.get())),
            "animation_fps": max(1, int(self.animation_fps_var.get())),
            "animation_on": bool(self.animation_on_var.get()),
            "gravity": float(self.gravity_var.get()),
            "enable_wind": bool(self.wind_on_var.get()),
            "wind_power": float(self.wind_power_var.get()),
            "turbulence_power": float(self.turbulence_var.get()),
            "compare_on": bool(self.compare_on_var.get()),
        }

    def _apply_snapshot_to_trainer(self, snap: Dict[str, object]) -> None:
        self.trainer.set_policy_config(
            snap["policy"],
            gamma=snap["gamma"],
            learning_rate=snap["learning_rate"],
            replay_size=snap["replay_size"],
            batch_size=snap["batch_size"],
            target_update=snap["target_update"],
            replay_warmup=snap["replay_warmup"],
            learning_cadence=snap["learning_cadence"],
            activation_function=snap["activation_function"],
            hidden_layers=snap["hidden_layers"],
        )

    def update_environment(self) -> None:
        if self._worker and self._worker.is_alive():
            messagebox.showwarning("Training running", "Pause/stop training before environment update.")
            return
        snap = self._snapshot_ui()
        self.trainer.rebuild_environment(
            snap["gravity"],
            snap["enable_wind"],
            snap["wind_power"],
            snap["turbulence_power"],
        )
        self._current_x = None
        self._best_x = None
        self._set_status_text(snap["epsilon_max"], None, None)
        self._draw_latest_frame()

    def run_single_episode(self) -> None:
        if self._has_active_workers():
            return
        self._start_worker(single_episode=True)

    def train_and_run(self) -> None:
        if self._has_active_workers():
            if self._pause_requested.is_set():
                self._stop_requested.set()
                self._pause_requested.clear()
                self.btn_pause.configure(text="Pause")
                self._start_after_worker_stops(single_episode=False)
            return
        self._start_worker(single_episode=False)

    def _has_active_workers(self) -> bool:
        if self._worker and self._worker.is_alive():
            return True
        for thread in self._workers.values():
            if thread.is_alive():
                return True
        return False

    def _start_after_worker_stops(self, single_episode: bool) -> None:
        if self._has_active_workers():
            self.root.after(50, lambda: self._start_after_worker_stops(single_episode=single_episode))
            return
        self._start_worker(single_episode=single_episode)

    def _policy_default_dict(self, policy: str) -> Dict[str, object]:
        cfg = POLICY_DEFAULTS[policy]
        data = asdict(cfg)
        return {
            "gamma": data["gamma"],
            "learning_rate": data["learning_rate"],
            "replay_size": data["replay_size"],
            "batch_size": data["batch_size"],
            "target_update": data["target_update"],
            "replay_warmup": data["replay_warmup"],
            "learning_cadence": data["learning_cadence"],
            "activation_function": data["activation_function"],
            "hidden_layers": data["hidden_layers"],
        }

    def _set_policy_defaults_on_trainer(self, trainer: Trainer, policy: str) -> None:
        trainer.set_policy_config(policy, **self._policy_default_dict(policy))

    def _close_aux_policy_trainers(self) -> None:
        for policy, trainer in list(self._policy_trainers.items()):
            if trainer is not self.trainer:
                trainer.close()
        self._policy_trainers.clear()

    def _build_base_label(self, policy: str, eps_max: float, eps_min: float, learning_rate: float) -> str:
        return f"{policy} | eps({eps_max}/{eps_min}) | lr={learning_rate}"

    def _start_worker(self, single_episode: bool) -> None:
        snap = self._snapshot_ui()
        if snap["compare_on"] and not single_episode:
            self._start_compare_workers(snap)
            return

        self._compare_mode_active = False
        self._workers.clear()
        self._policy_trainers.clear()
        self._selected_render_trainer = self.trainer
        self._apply_snapshot_to_trainer(snap)
        if not single_episode:
            self.trainer.reset_policy_agent(str(snap["policy"]))

        self._stop_requested.clear()
        self._pause_requested.clear()
        self.btn_pause.configure(text="Pause")

        self.steps_bar.configure(maximum=snap["max_steps"])
        self.episodes_bar.configure(maximum=1 if single_episode else snap["episodes"])
        self.steps_progress_var.set(0)
        self.episodes_progress_var.set(0)
        self._best_x = None
        self._current_x = None
        self._set_status_text(snap["epsilon_max"], None, None)

        self._worker = threading.Thread(target=self._worker_loop, args=(snap, single_episode), daemon=True)
        self._worker.start()
        self._update_control_highlights()

    def _start_compare_workers(self, snap: Dict[str, object]) -> None:
        policies = ["DuelingDQN", "D3QN", "DDQN+PER"]
        selected_policy = snap["policy"]

        self._compare_mode_active = True
        self._workers.clear()
        self._close_aux_policy_trainers()
        self._pending_policy_state.clear()
        self._latest_compare_rewards = {}
        self._compare_run_meta = {}
        self._compare_finalized_policies.clear()
        self._worker = None

        self._stop_requested.clear()
        self._pause_requested.clear()
        self.btn_pause.configure(text="Pause")

        self.steps_bar.configure(maximum=snap["max_steps"])
        self.episodes_bar.configure(maximum=snap["episodes"])
        self.steps_progress_var.set(0)
        self.episodes_progress_var.set(0)
        self._best_x = None
        self._current_x = None
        self._set_status_text(snap["epsilon_max"], None, None)

        self.trainer.rebuild_environment(
            snap["gravity"],
            snap["enable_wind"],
            snap["wind_power"],
            snap["turbulence_power"],
        )

        for policy in policies:
            if policy == selected_policy:
                trainer = self.trainer
            else:
                trainer = Trainer()
                trainer.rebuild_environment(
                    snap["gravity"],
                    snap["enable_wind"],
                    snap["wind_power"],
                    snap["turbulence_power"],
                )

            self._set_policy_defaults_on_trainer(trainer, policy)
            trainer.reset_policy_agent(policy)
            cfg = trainer.get_policy_config(policy)
            self._compare_run_meta[policy] = {
                "eps_max": float(snap["epsilon_max"]),
                "eps_min": float(snap["epsilon_min"]),
                "learning_rate": float(cfg.learning_rate),
                "moving_avg": float(snap["moving_avg"]),
            }
            self._latest_compare_rewards[policy] = []
            self._policy_trainers[policy] = trainer

            thread = threading.Thread(target=self._compare_worker_loop, args=(policy, trainer, snap), daemon=True)
            self._workers[policy] = thread
            thread.start()

        self._selected_render_trainer = self._policy_trainers.get(selected_policy, self.trainer)
        self._worker = self._workers.get(selected_policy)
        self._update_live_plot()
        self._update_control_highlights()

    def _worker_loop(self, snap: Dict[str, object], single_episode: bool) -> None:
        policy = snap["policy"]
        episodes = 1 if single_episode else snap["episodes"]
        epsilon = float(snap["epsilon_max"])
        rewards: List[float] = []

        for episode in range(1, episodes + 1):
            if self._stop_requested.is_set():
                break
            while self._pause_requested.is_set() and not self._stop_requested.is_set():
                time.sleep(0.05)
            if self._stop_requested.is_set():
                break

            result = self.trainer.run_episode(
                policy,
                epsilon=epsilon,
                max_steps=int(snap["max_steps"]),
                progress_callback=lambda step: self._set_pending(step=step, episode=episode),
            )
            reward = float(result["reward"])
            rewards.append(reward)
            final_state = result.get("final_state")
            current_x = float(final_state[0]) if final_state is not None and len(final_state) > 0 else None
            best_x = float(result.get("best_x", np.nan))

            self._set_pending(
                step=int(result["steps"]),
                episode=episode,
                epsilon=epsilon,
                current_x=current_x,
                best_x=best_x,
                rewards_snapshot=list(rewards),
            )
            epsilon = max(float(snap["epsilon_min"]), epsilon * float(snap["epsilon_decay"]))

        self._set_pending(finalize_run=True, rewards_snapshot=list(rewards), finished=True, epsilon=epsilon)

    def _compare_worker_loop(self, policy: str, trainer: Trainer, snap: Dict[str, object]) -> None:
        episodes = snap["episodes"]
        epsilon = float(snap["epsilon_max"])
        rewards: List[float] = []

        for episode in range(1, episodes + 1):
            if self._stop_requested.is_set():
                break
            while self._pause_requested.is_set() and not self._stop_requested.is_set():
                time.sleep(0.05)
            if self._stop_requested.is_set():
                break

            result = trainer.run_episode(
                policy,
                epsilon=epsilon,
                max_steps=int(snap["max_steps"]),
                progress_callback=lambda step, p=policy: self._set_policy_pending(p, step=step, episode=episode),
            )
            reward = float(result["reward"])
            rewards.append(reward)
            final_state = result.get("final_state")
            current_x = float(final_state[0]) if final_state is not None and len(final_state) > 0 else None
            best_x = float(result.get("best_x", np.nan))

            self._set_policy_pending(
                policy,
                step=int(result["steps"]),
                episode=episode,
                epsilon=epsilon,
                current_x=current_x,
                best_x=best_x,
                rewards_snapshot=list(rewards),
            )
            epsilon = max(float(snap["epsilon_min"]), epsilon * float(snap["epsilon_decay"]))

        self._set_policy_pending(policy, finalize_run=True, rewards_snapshot=list(rewards), finished=True, epsilon=epsilon)

    def _set_pending(self, **kwargs) -> None:
        with self._pending_lock:
            self._pending_state.update(kwargs)

    def _set_policy_pending(self, policy: str, **kwargs) -> None:
        with self._pending_lock:
            if policy not in self._pending_policy_state:
                self._pending_policy_state[policy] = {}
            self._pending_policy_state[policy].update(kwargs)

    def _ui_pump(self) -> None:
        pending = None
        pending_policy = None
        with self._pending_lock:
            if self._pending_state:
                pending = dict(self._pending_state)
                self._pending_state.clear()
            if self._pending_policy_state:
                pending_policy = {k: dict(v) for k, v in self._pending_policy_state.items()}
                self._pending_policy_state.clear()
        if pending:
            self._consume_pending(pending)
        if pending_policy:
            self._consume_policy_pending(pending_policy)
        if self._compare_mode_active and not self._has_active_workers():
            self._compare_mode_active = False
            self.btn_pause.configure(text="Pause")
            self._update_control_highlights()
        self.root.after(40, self._ui_pump)

    def _consume_pending(self, pending: Dict[str, object]) -> None:
        plot_interval = 0.30 if self._compare_mode_active else 0.15
        if "step" in pending:
            self.steps_progress_var.set(float(pending["step"]))
            self._latest_step = int(pending["step"])
        if "episode" in pending:
            self.episodes_progress_var.set(float(pending["episode"]))
            self._latest_episode = int(pending["episode"])
        if "current_x" in pending:
            self._current_x = pending["current_x"]
        if "best_x" in pending:
            if pending["best_x"] is not None:
                self._best_x = pending["best_x"]
        if "epsilon" in pending:
            self._set_status_text(float(pending["epsilon"]), self._current_x, self._best_x)

        if "rewards_snapshot" in pending:
            self._latest_rewards_snapshot = list(pending["rewards_snapshot"])
            now = time.time()
            if now - self._last_plot_update >= plot_interval:
                self._update_live_plot()
                self._last_plot_update = now

        if pending.get("finalize_run"):
            self._finalize_run(self._latest_rewards_snapshot or [])

        if pending.get("finished"):
            self.btn_pause.configure(text="Pause")
            self._update_control_highlights()

    def _consume_policy_pending(self, pending_policy: Dict[str, Dict[str, object]]) -> None:
        selected_policy = self.policy_var.get()
        plot_interval = 0.30 if self._compare_mode_active else 0.15
        for policy, pending in pending_policy.items():
            if "rewards_snapshot" in pending:
                self._latest_compare_rewards[policy] = list(pending["rewards_snapshot"])
            if policy == selected_policy:
                if "step" in pending:
                    self.steps_progress_var.set(float(pending["step"]))
                    self._latest_step = int(pending["step"])
                if "episode" in pending:
                    self.episodes_progress_var.set(float(pending["episode"]))
                    self._latest_episode = int(pending["episode"])
                if "current_x" in pending:
                    self._current_x = pending["current_x"]
                if "best_x" in pending and pending["best_x"] is not None:
                    self._best_x = pending["best_x"]
                if "epsilon" in pending:
                    self._set_status_text(float(pending["epsilon"]), self._current_x, self._best_x)

            if pending.get("finalize_run") and policy not in self._compare_finalized_policies:
                rewards = self._latest_compare_rewards.get(policy, [])
                self._finalize_run_for_policy(policy, rewards)
                self._compare_finalized_policies.add(policy)
                self._latest_compare_rewards.pop(policy, None)

        now = time.time()
        if now - self._last_plot_update >= plot_interval and self._latest_compare_rewards:
            self._update_live_plot()
            self._last_plot_update = now

    def _set_status_text(self, epsilon: float, current_x, best_x) -> None:
        cur = "n/a" if current_x is None else f"{float(current_x):.3f}"
        best = "n/a" if best_x is None else f"{float(best_x):.3f}"
        self.status_var.set(f"Epsilon: {epsilon:.4f} | Current x: {cur} | Best x: {best}")

    def _update_control_highlights(self) -> None:
        running = self._has_active_workers()
        paused = self._pause_requested.is_set()

        if paused:
            self.btn_pause.configure(style="Pause.TButton")
            self.btn_train.configure(style="TButton")
            return

        self.btn_pause.configure(style="TButton")
        if running:
            self.btn_train.configure(style="Primary.TButton")
        else:
            self.btn_train.configure(style="TButton")

    def _moving_average(self, values: List[float], window: int) -> List[float]:
        if not values:
            return []
        arr = np.asarray(values, dtype=np.float32)
        out = []
        for i in range(len(arr)):
            start = max(0, i - window + 1)
            out.append(float(arr[start : i + 1].mean()))
        return out

    def _update_live_plot(self) -> None:
        if self._compare_mode_active:
            preview_active = {
                policy: rewards
                for policy, rewards in self._latest_compare_rewards.items()
                if policy not in self._compare_finalized_policies
            }
            if not preview_active:
                self._clear_compare_preview_lines()
                self.plot_canvas.draw_idle()
                return
            self._update_compare_preview(preview_active)
            return
        if self._latest_rewards_snapshot is None:
            return
        rewards = self._latest_rewards_snapshot
        self._update_single_preview(rewards)

    def _clear_single_preview_lines(self) -> None:
        if self._preview_single_lines is None:
            return
        for line in self._preview_single_lines:
            try:
                line.remove()
            except Exception:
                pass
        self._preview_single_lines = None

    def _clear_compare_preview_lines(self) -> None:
        if not self._preview_compare_lines:
            return
        for pair in self._preview_compare_lines.values():
            for line in pair:
                try:
                    line.remove()
                except Exception:
                    pass
        self._preview_compare_lines.clear()

    def _remove_compare_preview_policy(self, policy: str) -> None:
        pair = self._preview_compare_lines.pop(policy, None)
        if not pair:
            return
        for line in pair:
            try:
                line.remove()
            except Exception:
                pass

    def _autoscale_plot(self) -> None:
        self.ax.relim()
        self.ax.autoscale_view()
        self._refresh_legend()
        self.plot_canvas.draw_idle()

    def _refresh_legend(self) -> None:
        self._legend_map.clear()
        handles = []
        labels = []
        for line in self.ax.lines:
            label = line.get_label()
            if not label or label.startswith("_"):
                continue
            handles.append(line)
            labels.append(label)

        if not handles:
            existing = self.ax.get_legend()
            if existing:
                existing.remove()
            return

        legend = self.ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        if not legend:
            return
        legend.get_frame().set_facecolor("#2d2d30")
        legend.get_frame().set_edgecolor("#4f5358")
        legend.get_frame().set_alpha(0.9)

        for leg_handle, line in zip(legend.legend_handles, handles):
            leg_handle.set_picker(5)
            leg_handle.set_alpha(1.0 if line.get_visible() else 0.25)
            self._legend_map[leg_handle] = line

        for leg_text, line in zip(legend.get_texts(), handles):
            leg_text.set_picker(True)
            leg_text.set_alpha(1.0 if line.get_visible() else 0.25)
            leg_text.set_color("#d4d4d4")
            self._legend_map[leg_text] = line

    def _update_single_preview(self, rewards: List[float]) -> None:
        self._clear_compare_preview_lines()
        x = list(range(1, len(rewards) + 1))
        snap = self._snapshot_ui()
        base = self._build_base_label(snap["policy"], snap["epsilon_max"], snap["epsilon_min"], snap["learning_rate"])
        ma = self._moving_average(rewards, int(snap["moving_avg"]))
        if self._preview_single_lines is None:
            reward_line, = self.ax.plot(x, rewards, alpha=0.4, linewidth=1.0, label=f"{base} | reward")
            ma_line, = self.ax.plot(x, ma, alpha=1.0, linewidth=2.2, color=reward_line.get_color(), label=f"{base} | MA")
            self._preview_single_lines = (reward_line, ma_line)
        else:
            reward_line, ma_line = self._preview_single_lines
            reward_line.set_data(x, rewards)
            ma_line.set_data(x, ma)
            reward_line.set_label(f"{base} | reward")
            ma_line.set_label(f"{base} | MA")
        self._autoscale_plot()

    def _update_compare_preview(self, preview_active: Dict[str, List[float]]) -> None:
        self._clear_single_preview_lines()

        stale = [policy for policy in self._preview_compare_lines if policy not in preview_active]
        for policy in stale:
            self._remove_compare_preview_policy(policy)

        for policy, rewards in preview_active.items():
            x = list(range(1, len(rewards) + 1))
            meta = self._compare_run_meta.get(policy, {})
            eps_max = float(meta.get("eps_max", self.epsilon_max_var.get()))
            eps_min = float(meta.get("eps_min", self.epsilon_min_var.get()))
            lr = float(meta.get("learning_rate", self.learning_rate_var.get()))
            base = self._build_base_label(policy, eps_max, eps_min, lr)
            ma_window = int(meta.get("moving_avg", self.moving_avg_var.get()))
            ma = self._moving_average(rewards, ma_window)

            if policy not in self._preview_compare_lines:
                reward_line, = self.ax.plot(x, rewards, alpha=0.4, linewidth=1.0, label=f"{base} | reward")
                ma_line, = self.ax.plot(x, ma, alpha=1.0, linewidth=2.2, color=reward_line.get_color(), label=f"{base} | MA")
                self._preview_compare_lines[policy] = (reward_line, ma_line)
            else:
                reward_line, ma_line = self._preview_compare_lines[policy]
                reward_line.set_data(x, rewards)
                ma_line.set_data(x, ma)
                reward_line.set_label(f"{base} | reward")
                ma_line.set_label(f"{base} | MA")

        self._autoscale_plot()

    def _finalize_run(self, rewards: List[float]) -> None:
        if not rewards:
            return
        self._clear_single_preview_lines()
        snap = self._snapshot_ui()
        base_label = self._build_base_label(snap["policy"], snap["epsilon_max"], snap["epsilon_min"], snap["learning_rate"])
        self._run_counter += 1
        self._plot_runs.append(
            {
                "id": self._run_counter,
                "base": base_label,
                "policy": snap["policy"],
                "rewards": list(rewards),
                "ma": self._moving_average(list(rewards), int(snap["moving_avg"])),
            }
        )
        self._redraw_plot()

    def _finalize_run_for_policy(self, policy: str, rewards: List[float]) -> None:
        if not rewards:
            return
        self._remove_compare_preview_policy(policy)
        meta = self._compare_run_meta.get(policy, {})
        eps_max = float(meta.get("eps_max", self.epsilon_max_var.get()))
        eps_min = float(meta.get("eps_min", self.epsilon_min_var.get()))
        lr = float(meta.get("learning_rate", self.learning_rate_var.get()))
        ma_window = int(meta.get("moving_avg", self.moving_avg_var.get()))
        base_label = self._build_base_label(policy, eps_max, eps_min, lr)
        self._run_counter += 1
        self._plot_runs.append(
            {
                "id": self._run_counter,
                "base": base_label,
                "policy": policy,
                "rewards": list(rewards),
                "ma": self._moving_average(list(rewards), ma_window),
            }
        )
        self._redraw_plot()

    def _redraw_plot(self, preview_rewards: Optional[List[float]] = None, preview_compare: Optional[Dict[str, List[float]]] = None) -> None:
        self.ax.clear()
        self._preview_single_lines = None
        self._preview_compare_lines.clear()
        self.figure.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.xaxis.label.set_color("#e2e2e2")
        self.ax.yaxis.label.set_color("#e2e2e2")
        self.ax.tick_params(axis="x", colors="#dddddd")
        self.ax.tick_params(axis="y", colors="#dddddd")
        self.ax.grid(True, alpha=0.3, linewidth=0.9)
        for spine in self.ax.spines.values():
            spine.set_alpha(0.5)
            spine.set_color("#9a9a9a")
        self.figure.subplots_adjust(right=0.75)
        self._legend_map.clear()

        lines = []

        for run in self._plot_runs:
            x = list(range(1, len(run["rewards"]) + 1))
            color = None
            reward_line, = self.ax.plot(x, run["rewards"], alpha=0.4, linewidth=1.0, label=f"{run['base']} | reward", picker=5)
            color = reward_line.get_color()
            ma_line, = self.ax.plot(x, run["ma"], alpha=1.0, linewidth=2.2, color=color, label=f"{run['base']} | MA", picker=5)
            lines.extend([reward_line, ma_line])

        if preview_rewards is not None:
            snap = self._snapshot_ui()
            base = self._build_base_label(snap["policy"], snap["epsilon_max"], snap["epsilon_min"], snap["learning_rate"])
            x = list(range(1, len(preview_rewards) + 1))
            reward_line, = self.ax.plot(x, preview_rewards, alpha=0.4, linewidth=1.0, label=f"{base} | reward")
            ma = self._moving_average(preview_rewards, int(snap["moving_avg"]))
            ma_line, = self.ax.plot(x, ma, alpha=1.0, linewidth=2.2, color=reward_line.get_color(), label=f"{base} | MA")
            lines.extend([reward_line, ma_line])

        if preview_compare is not None:
            for policy, rewards in preview_compare.items():
                if not rewards:
                    continue
                meta = self._compare_run_meta.get(policy, {})
                eps_max = float(meta.get("eps_max", self.epsilon_max_var.get()))
                eps_min = float(meta.get("eps_min", self.epsilon_min_var.get()))
                lr = float(meta.get("learning_rate", self.learning_rate_var.get()))
                ma_window = int(meta.get("moving_avg", self.moving_avg_var.get()))
                base = self._build_base_label(policy, eps_max, eps_min, lr)
                x = list(range(1, len(rewards) + 1))
                reward_line, = self.ax.plot(x, rewards, alpha=0.4, linewidth=1.0, label=f"{base} | reward")
                ma = self._moving_average(rewards, ma_window)
                ma_line, = self.ax.plot(x, ma, alpha=1.0, linewidth=2.2, color=reward_line.get_color(), label=f"{base} | MA")
                lines.extend([reward_line, ma_line])

        if lines:
            legend = self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
            if legend:
                legend.get_frame().set_facecolor("#2d2d30")
                legend.get_frame().set_edgecolor("#4f5358")
                legend.get_frame().set_alpha(0.9)
                for leg_handle, line in zip(legend.legend_handles, lines):
                    leg_handle.set_picker(5)
                    leg_handle.set_alpha(1.0)
                    self._legend_map[leg_handle] = line
                    line.set_gid(id(line))

                for leg_text in legend.get_texts():
                    leg_text.set_picker(True)
                    leg_text.set_alpha(1.0)
                    leg_text.set_color("#d4d4d4")
                for leg_text, line in zip(legend.get_texts(), lines):
                    self._legend_map[leg_text] = line
        else:
            existing = self.ax.get_legend()
            if existing:
                existing.remove()

        self.plot_canvas.draw_idle()

    def _on_legend_pick(self, event) -> None:
        artist = event.artist
        line = self._legend_map.get(artist)
        if line is None:
            return
        visible = not line.get_visible()
        line.set_visible(visible)
        artist.set_alpha(1.0 if visible else 0.25)

        legend = self.ax.get_legend()
        if legend:
            handles = legend.legend_handles
            texts = legend.get_texts()
            for idx, candidate in enumerate(handles):
                if candidate is artist and idx < len(texts):
                    texts[idx].set_alpha(1.0 if visible else 0.25)
                    break
            for idx, candidate in enumerate(texts):
                if candidate is artist and idx < len(handles):
                    handles[idx].set_alpha(1.0 if visible else 0.25)
                    break

        self.plot_canvas.draw_idle()

    def _render_tick(self) -> None:
        now = time.time()
        fps = max(1, int(self.animation_fps_var.get()))
        if self.animation_on_var.get() and now - self._last_render_update >= 1.0 / fps:
            should_draw = True
            if self._has_active_workers():
                should_draw = not (
                    self._latest_step == self._last_rendered_step
                    and self._latest_episode == self._last_rendered_episode
                )
            if should_draw:
                self._draw_latest_frame()
                self._last_rendered_step = self._latest_step
                self._last_rendered_episode = self._latest_episode
                self._last_render_update = now
        self.root.after(30, self._render_tick)

    def _draw_latest_frame(self) -> None:
        trainer = self._selected_render_trainer if self._selected_render_trainer is not None else self.trainer
        try:
            frame = trainer.env.render()
        except Exception:
            return
        if frame is None:
            return
        h, w = frame.shape[0], frame.shape[1]
        canvas_w = max(1, self.render_canvas.winfo_width())
        canvas_h = max(1, self.render_canvas.winfo_height())
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        if Image is not None and ImageTk is not None:
            img = Image.fromarray(frame)
            if (new_w, new_h) != (w, h):
                img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
            self._tk_img = ImageTk.PhotoImage(img)
            if self._canvas_image_id is None:
                self._canvas_image_id = self.render_canvas.create_image(canvas_w // 2, canvas_h // 2, image=self._tk_img, anchor="center")
            else:
                self.render_canvas.coords(self._canvas_image_id, canvas_w // 2, canvas_h // 2)
                self.render_canvas.itemconfig(self._canvas_image_id, image=self._tk_img)
        else:
            self.render_canvas.delete("all")
            self._canvas_image_id = None
            self.render_canvas.create_text(canvas_w // 2, canvas_h // 2, text="Pillow not installed for image rendering", fill="white")

    def toggle_pause(self) -> None:
        if not self._has_active_workers():
            return
        if self._pause_requested.is_set():
            self._pause_requested.clear()
            self.btn_pause.configure(text="Pause")
        else:
            self._pause_requested.set()
            self.btn_pause.configure(text="Run")
        self._update_control_highlights()

    def reset_all(self) -> None:
        self._stop_requested.set()
        self._pause_requested.clear()
        self.btn_pause.configure(text="Pause")
        self._worker = None
        self._workers.clear()
        self._close_aux_policy_trainers()
        self._compare_mode_active = False
        self._latest_compare_rewards.clear()
        self._compare_finalized_policies.clear()
        self._compare_run_meta.clear()
        self._pending_policy_state.clear()
        self._selected_render_trainer = self.trainer
        self._clear_single_preview_lines()
        self._clear_compare_preview_lines()
        self._plot_runs.clear()
        self._latest_rewards_snapshot = None
        self._current_x = None
        self._best_x = None
        self.steps_progress_var.set(0)
        self.episodes_progress_var.set(0)
        self._last_rendered_step = -1
        self._last_rendered_episode = -1
        self._set_status_text(self.epsilon_max_var.get(), None, None)
        self._redraw_plot()
        self._update_control_highlights()

    def clear_plot(self) -> None:
        self._plot_runs.clear()
        self._latest_rewards_snapshot = None
        self._latest_compare_rewards.clear()
        self._compare_finalized_policies.clear()
        self._clear_single_preview_lines()
        self._clear_compare_preview_lines()
        self._redraw_plot()

    def save_samplings_csv(self) -> None:
        snap = self._snapshot_ui()
        self._apply_snapshot_to_trainer(snap)
        name = datetime.now().strftime("samplings_%Y%m%d_%H%M%S")
        self.trainer.train(
            policy=snap["policy"],
            num_episodes=1,
            max_steps=snap["max_steps"],
            epsilon=snap["epsilon_max"],
            save_csv=name,
        )
        messagebox.showinfo("CSV saved", f"Saved to results_csv/{name}.csv")

    def save_plot_png(self) -> None:
        snap = self._snapshot_ui()
        path = self.trainer.save_plot_png(
            self.figure,
            snap["policy"],
            snap["epsilon_max"],
            snap["epsilon_min"],
            snap["learning_rate"],
            snap["gamma"],
            snap["episodes"],
            snap["max_steps"],
        )
        messagebox.showinfo("PNG saved", f"Saved: {os.path.basename(path)}")

    def _on_close(self) -> None:
        self._stop_requested.set()
        self._pause_requested.clear()
        self._update_control_highlights()
        self._close_aux_policy_trainers()
        self.trainer.close()
        self.root.destroy()
