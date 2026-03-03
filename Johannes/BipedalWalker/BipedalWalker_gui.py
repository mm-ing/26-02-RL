from __future__ import annotations

import itertools
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib.pyplot as plt
import numpy as np
import torch as th
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

from BipedalWalker_logic import (
    BipedalWalkerConfig,
    BipedalWalkerEnvironment,
    BipedalWalkerTrainer,
    POLICY_DISPLAY_NAMES,
    build_compare_combinations,
    get_policy_default_configs,
    make_run_label,
    png_filename,
)


class BipedalWalkerGUI(ttk.Frame):
    COMPARE_MAX_WORKERS = 4

    def __init__(self, master: tk.Tk):
        super().__init__(master)
        self.master = master
        self.master.title("Bipedal Walker - SB3")

        self.PAD_OUTER = 10
        self.PAD_INNER = 6
        self.PAD_TIGHT = 4
        self.LABEL_COL_WIDTH = 92
        self.PARAM_INPUT_WIDTH = 9

        self.pack(fill="both", expand=True)

        self.root_dir = Path(__file__).resolve().parent
        self.results_dir = self.root_dir / "results_csv"
        self.plots_dir = self.root_dir / "plots"
        self.results_dir.mkdir(exist_ok=True)
        self.plots_dir.mkdir(exist_ok=True)

        self.policy_defaults = get_policy_default_configs()
        self.event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        self.active_trainer: Optional[BipedalWalkerTrainer] = None
        self._active_trainers: List[BipedalWalkerTrainer] = []
        self._active_trainers_lock = threading.Lock()
        self._last_finished_trainer: Optional[BipedalWalkerTrainer] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.run_counter = 0
        self.is_paused = False

        self.latest_frame = None
        self.latest_frame_lock = threading.Lock()
        self._latest_frame_version = 0
        self._rendered_frame_version = -1
        self._rollout_playback_frames: List[Any] = []
        self._rollout_playback_index = 0
        self._rollout_playback_total = 0
        self.render_photo = None
        self._canvas_image_id = None
        self.render_state = "idle"
        self._pending_resize_job = None
        self._last_plot_draw = 0.0
        self._defer_plot_refresh = False
        self.use_gpu = False

        self.run_histories: Dict[str, Dict[str, Any]] = {}
        self.lines_by_run: Dict[str, Dict[str, Any]] = {}
        self.legend_artist_map: Dict[Any, Any] = {}
        self.compare_values: Dict[str, List[Any]] = {}

        self._configure_style()
        self._build_variables()
        self._build_layout()
        self._apply_policy_defaults(self.var_policy.get())
        self._update_compare_summary()
        self._refresh_plot()

        self.after(50, self._pump_events)
        self.after(100, self._render_tick)

    def _configure_style(self):
        style = ttk.Style(self.master)
        available = set(style.theme_names())
        if "clam" in available:
            style.theme_use("clam")
        elif "vista" in available:
            style.theme_use("vista")

        default_font = ("Segoe UI", 10)
        heading_font = ("Segoe UI", 10, "bold")
        button_font = ("Segoe UI", 10, "bold")

        main_bg = "#1e1e1e"
        panel_bg = "#252526"
        input_bg = "#2d2d30"
        fg = "#e6e6e6"
        muted = "#d0d0d0"
        accent = "#0e639c"

        self.master.configure(bg=main_bg)
        self.master.option_add("*TCombobox*Listbox*Background", input_bg)
        self.master.option_add("*TCombobox*Listbox*Foreground", fg)
        self.master.option_add("*TCombobox*Listbox*selectBackground", accent)
        self.master.option_add("*TCombobox*Listbox*selectForeground", "white")

        style.configure("TFrame", background=main_bg)
        style.configure("TLabelframe", background=panel_bg, foreground=fg, padding=(8, 8))
        style.configure("TLabelframe.Label", background=panel_bg, foreground=fg, font=heading_font)
        style.configure("TLabel", background=panel_bg, foreground=fg, font=default_font)

        style.configure("TButton", background="#3a3d41", foreground=fg, font=button_font, padding=(10, 5))
        style.map("TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")], foreground=[("disabled", "#b8b8b8")])

        style.configure("Primary.TButton", background=accent, foreground="white", font=button_font, padding=(10, 5))
        style.map("Primary.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")], foreground=[("disabled", "#ededed")])

        style.configure("Pause.TButton", background="#a66a00", foreground="white", font=button_font, padding=(10, 5))
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")], foreground=[("disabled", "#ededed")])

        style.configure("TCheckbutton", background=panel_bg, foreground=fg)
        style.map("TCheckbutton", background=[("active", panel_bg), ("!active", panel_bg)], foreground=[("active", fg), ("!active", fg)])
        style.configure("TCombobox", fieldbackground=input_bg, background=input_bg, foreground=fg)
        style.map("TCombobox", fieldbackground=[("readonly", input_bg)], foreground=[("readonly", fg)])
        style.configure("TEntry", fieldbackground=input_bg, foreground=fg, insertcolor=fg, padding=(6, 4))
        style.configure("Horizontal.TProgressbar", troughcolor="#343434", background=accent)
        style.configure("Vertical.TScrollbar", background="#3a3d41", troughcolor="#252526", arrowcolor=muted)

        self.colors = {
            "main_bg": main_bg,
            "panel_bg": panel_bg,
            "input_bg": input_bg,
            "fg": fg,
            "muted": muted,
            "accent": accent,
        }

    def _build_variables(self):
        self.var_animation_on = tk.BooleanVar(value=True)
        self.var_animation_fps = tk.IntVar(value=10)
        self.var_update_rate = tk.IntVar(value=5)
        self.var_hardcore = tk.BooleanVar(value=False)

        self.var_compare_on = tk.BooleanVar(value=False)
        self.var_compare_param = tk.StringVar(value="Policy")
        self.var_compare_values = tk.StringVar(value="")

        self.var_max_steps = tk.IntVar(value=1600)
        self.var_episodes = tk.IntVar(value=50)
        self.var_epsilon_max = tk.DoubleVar(value=1.0)
        self.var_epsilon_decay = tk.DoubleVar(value=0.995)
        self.var_epsilon_min = tk.DoubleVar(value=0.05)
        self.var_gamma = tk.DoubleVar(value=0.99)

        self.var_policy = tk.StringVar(value="PPO")
        self.var_hidden_layer = tk.StringVar(value="256,256")
        self.var_activation = tk.StringVar(value="ReLU")
        self.var_lr = tk.StringVar(value="3.0e-04")
        self.var_lr_strategy = tk.StringVar(value="constant")
        self.var_min_lr = tk.StringVar(value="1.0e-05")
        self.var_lr_decay = tk.DoubleVar(value=0.999)
        self.var_replay_size = tk.IntVar(value=300000)
        self.var_batch_size = tk.IntVar(value=64)
        self.var_learning_start = tk.IntVar(value=10000)
        self.var_learning_frequency = tk.IntVar(value=1)
        self.var_target_update = tk.IntVar(value=2)

        self.var_moving_average = tk.IntVar(value=20)
        self.var_show_advanced = tk.BooleanVar(value=False)
        self.var_rollout_short_capture_steps = tk.IntVar(value=120)
        self.var_low_overhead_animation = tk.BooleanVar(value=False)
        self.var_status = tk.StringVar(value="Epsilon: - | LR: - | Best reward: - | Render: idle")

    def _build_layout(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=5)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=0)
        self.rowconfigure(3, weight=4)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="nsew", padx=self.PAD_OUTER, pady=self.PAD_OUTER)
        top.columnconfigure(0, weight=2)
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        self._build_environment_panel(top)
        self._build_parameters_panel(top)
        self._build_controls_row()
        self._build_current_run_panel()
        self._build_live_plot_panel()

    def _build_environment_panel(self, parent):
        frame = ttk.LabelFrame(parent, text="Environment")
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, self.PAD_OUTER))
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(
            frame,
            bg="#111111",
            highlightthickness=0,
            bd=0,
        )
        self.render_canvas.grid(row=0, column=0, sticky="nsew")
        self.render_canvas.bind("<Configure>", self._on_render_resize)

    def _build_parameters_panel(self, parent):
        outer = ttk.LabelFrame(parent, text="Parameters")
        outer.grid(row=0, column=1, sticky="nsew")
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        canvas = tk.Canvas(outer, bg=self.colors["main_bg"], highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.columnconfigure(0, weight=1)

        content.bind(
            "<Configure>",
            lambda _: self._on_param_content_configure(canvas, content, scrollbar),
        )
        canvas.bind("<Configure>", self._on_param_canvas_configure)
        self.param_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        self.param_canvas = canvas
        self.param_scrollbar = scrollbar
        self.param_content = content
        self._scroll_visible = False

        self._bind_mousewheel_for_param_panel()
        self._build_param_groups(content)

    def _bind_mousewheel_for_param_panel(self):
        def bind_wheel(_):
            self.param_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        def unbind_wheel(_):
            self.param_canvas.unbind_all("<MouseWheel>")

        self.param_canvas.bind("<Enter>", bind_wheel)
        self.param_canvas.bind("<Leave>", unbind_wheel)

    def _on_mousewheel(self, event):
        if not self._scroll_visible:
            return
        self.param_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _on_param_content_configure(self, canvas, content, scrollbar):
        canvas.configure(scrollregion=canvas.bbox("all"))
        self._refresh_scrollbar_visibility(canvas, content, scrollbar)

    def _on_param_canvas_configure(self, event):
        self.param_canvas.itemconfigure(self.param_window, width=event.width)
        self._refresh_scrollbar_visibility(self.param_canvas, self.param_content, self.param_scrollbar)

    def _refresh_scrollbar_visibility(self, canvas, content, scrollbar):
        self.master.update_idletasks()
        needs_scroll = content.winfo_reqheight() > canvas.winfo_height() + 2
        if needs_scroll:
            scrollbar.grid(row=0, column=1, sticky="ns")
            self._scroll_visible = True
        else:
            scrollbar.grid_forget()
            self._scroll_visible = False
            canvas.yview_moveto(0)

    def _auto_fit_params_canvas_width(self):
        return

    def _build_param_groups(self, parent):
        env_group = ttk.LabelFrame(parent, text="Environment")
        env_group.grid(row=0, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        env_group.columnconfigure(1, weight=1)
        env_group.columnconfigure(3, weight=1)
        self._add_pair(
            env_group,
            0,
            "Animation on",
            ttk.Checkbutton(env_group, variable=self.var_animation_on, command=self._on_animation_toggle_changed),
            "Animation FPS",
            ttk.Entry(env_group, textvariable=self.var_animation_fps, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_single(
            env_group,
            1,
            "Update rate (episodes)",
            ttk.Entry(env_group, textvariable=self.var_update_rate, width=self.PARAM_INPUT_WIDTH),
        )
        ttk.Button(env_group, text="Update", command=self._on_update_environment).grid(
            row=2, column=0, columnspan=4, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT
        )
        self._add_single(env_group, 3, "hardcore", ttk.Checkbutton(env_group, variable=self.var_hardcore))

        compare_group = ttk.LabelFrame(parent, text="Compare")
        compare_group.grid(row=1, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        compare_group.columnconfigure(1, weight=1)
        compare_group.columnconfigure(3, weight=1)
        self._add_pair(
            compare_group,
            0,
            "Compare on",
            ttk.Checkbutton(compare_group, variable=self.var_compare_on, command=self._on_compare_toggle),
            "",
            ttk.Button(compare_group, text="Clear", command=self._clear_compare),
        )
        compare_combo = ttk.Combobox(
            compare_group,
            textvariable=self.var_compare_param,
            values=["Policy", "hardcore", "gamma", "learning_rate", "batch_size", "max_steps", "episodes"],
            state="readonly",
            width=self.PARAM_INPUT_WIDTH,
        )
        compare_entry = ttk.Entry(compare_group, textvariable=self.var_compare_values)
        self._add_pair(compare_group, 1, "Parameter", compare_combo, "Options", compare_entry)
        compare_entry.bind("<Return>", lambda _: self._sync_current_compare_param())
        compare_entry.bind("<FocusOut>", lambda _: self._sync_current_compare_param())
        compare_combo.bind("<<ComboboxSelected>>", lambda _: self._sync_current_compare_param())

        self.compare_summary = tk.Text(compare_group, height=4, bg=self.colors["input_bg"], fg=self.colors["muted"], relief="flat")
        self.compare_summary.grid(row=2, column=0, columnspan=4, sticky="ew", padx=self.PAD_TIGHT, pady=(0, self.PAD_TIGHT))
        self.compare_summary.configure(state="disabled")

        general_group = ttk.LabelFrame(parent, text="General")
        general_group.grid(row=2, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        general_group.columnconfigure(1, weight=1)
        general_group.columnconfigure(3, weight=1)
        self._add_pair(
            general_group,
            0,
            "Max steps",
            ttk.Entry(general_group, textvariable=self.var_max_steps, width=self.PARAM_INPUT_WIDTH),
            "Episodes",
            ttk.Entry(general_group, textvariable=self.var_episodes, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_pair(
            general_group,
            1,
            "Epsilon max",
            ttk.Entry(general_group, textvariable=self.var_epsilon_max, width=self.PARAM_INPUT_WIDTH),
            "Epsilon decay",
            ttk.Entry(general_group, textvariable=self.var_epsilon_decay, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_pair(
            general_group,
            2,
            "Epsilon min",
            ttk.Entry(general_group, textvariable=self.var_epsilon_min, width=self.PARAM_INPUT_WIDTH),
            "Gamma",
            ttk.Entry(general_group, textvariable=self.var_gamma, width=self.PARAM_INPUT_WIDTH),
        )

        specific_group = ttk.LabelFrame(parent, text="Specific")
        specific_group.grid(row=3, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        specific_group.columnconfigure(1, weight=1)
        specific_group.columnconfigure(3, weight=1)
        ttk.Label(specific_group, text="Policy").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        policy_combo = ttk.Combobox(
            specific_group,
            textvariable=self.var_policy,
            values=POLICY_DISPLAY_NAMES,
            state="readonly",
            width=self.PARAM_INPUT_WIDTH,
        )
        policy_combo.grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        policy_combo.bind("<<ComboboxSelected>>", lambda _: self._apply_policy_defaults(self.var_policy.get()))

        lr_entry = ttk.Entry(specific_group, textvariable=self.var_lr, width=self.PARAM_INPUT_WIDTH)
        min_lr_entry = ttk.Entry(specific_group, textvariable=self.var_min_lr, width=self.PARAM_INPUT_WIDTH)
        self._add_pair(
            specific_group,
            1,
            "Hidden layer",
            ttk.Entry(specific_group, textvariable=self.var_hidden_layer, width=self.PARAM_INPUT_WIDTH),
            "Activation",
            ttk.Combobox(specific_group, textvariable=self.var_activation, values=["ReLU", "Tanh", "ELU", "LeakyReLU"], state="readonly", width=self.PARAM_INPUT_WIDTH),
        )
        self._add_pair(
            specific_group,
            2,
            "LR",
            lr_entry,
            "LR strategy",
            ttk.Combobox(specific_group, textvariable=self.var_lr_strategy, values=["constant", "linear", "exponential"], state="readonly", width=self.PARAM_INPUT_WIDTH),
        )
        self._add_pair(specific_group, 3, "Min LR", min_lr_entry, "LR decay", ttk.Entry(specific_group, textvariable=self.var_lr_decay, width=self.PARAM_INPUT_WIDTH))
        self._add_pair(
            specific_group,
            4,
            "Replay size",
            ttk.Entry(specific_group, textvariable=self.var_replay_size, width=self.PARAM_INPUT_WIDTH),
            "Batch size",
            ttk.Entry(specific_group, textvariable=self.var_batch_size, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_pair(
            specific_group,
            5,
            "Learning start",
            ttk.Entry(specific_group, textvariable=self.var_learning_start, width=self.PARAM_INPUT_WIDTH),
            "Learning frequency",
            ttk.Entry(specific_group, textvariable=self.var_learning_frequency, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_single(specific_group, 6, "Target update", ttk.Entry(specific_group, textvariable=self.var_target_update, width=self.PARAM_INPUT_WIDTH))
        lr_entry.bind("<FocusOut>", lambda _e: self._normalize_lr_inputs())
        min_lr_entry.bind("<FocusOut>", lambda _e: self._normalize_lr_inputs())

        live_group = ttk.LabelFrame(parent, text="Live Plot")
        live_group.grid(row=4, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        live_group.columnconfigure(1, weight=1)
        ttk.Label(live_group, text="Moving average values").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        ttk.Entry(live_group, textvariable=self.var_moving_average, width=self.PARAM_INPUT_WIDTH).grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_TIGHT)
        ttk.Checkbutton(live_group, text="Show Advanced", variable=self.var_show_advanced, command=self._toggle_advanced_params).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=self.PAD_TIGHT, pady=(0, self.PAD_TIGHT)
        )

        advanced_group = ttk.LabelFrame(parent, text="Advanced")
        advanced_group.grid(row=5, column=0, sticky="ew", padx=0, pady=self.PAD_TIGHT)
        advanced_group.columnconfigure(1, weight=1)
        self._add_single(
            advanced_group,
            0,
            "Rollout full-capture steps",
            ttk.Entry(advanced_group, textvariable=self.var_rollout_short_capture_steps, width=self.PARAM_INPUT_WIDTH),
        )
        self._add_single(
            advanced_group,
            1,
            "Low-overhead animation",
            ttk.Checkbutton(advanced_group, variable=self.var_low_overhead_animation),
        )
        self.advanced_group = advanced_group
        self.advanced_group.grid_remove()

    def _toggle_advanced_params(self):
        if self.var_show_advanced.get():
            self.advanced_group.grid()
        else:
            self.advanced_group.grid_remove()
        self._on_param_content_configure(self.param_canvas, self.param_content, self.param_scrollbar)

    def _add_pair(self, parent, row, l1, w1, l2, w2):
        ttk.Label(parent, text=l1).grid(row=row, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        w1.grid(row=row, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)
        ttk.Label(parent, text=l2).grid(row=row, column=2, sticky="w", padx=self.PAD_TIGHT, pady=2)
        w2.grid(row=row, column=3, sticky="ew", padx=self.PAD_TIGHT, pady=2)

    def _add_single(self, parent, row, label, widget):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        widget.grid(row=row, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)

    def _normalize_lr_inputs(self):
        try:
            self.var_lr.set(f"{float(self.var_lr.get()):.1e}")
        except Exception:
            self.var_lr.set("3.0e-04")
        try:
            self.var_min_lr.set(f"{float(self.var_min_lr.get()):.1e}")
        except Exception:
            self.var_min_lr.set("1.0e-05")

    def _grid_dual_field(self, parent, row, left_label, left_var, right_label, right_var, combobox_values=None):
        combobox_values = combobox_values or [None, None]
        ttk.Label(parent, text=left_label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        if combobox_values[0]:
            ttk.Combobox(parent, textvariable=left_var, values=combobox_values[0], state="readonly", width=14).grid(
                row=row, column=1, sticky="ew", padx=4, pady=2
            )
        else:
            ttk.Entry(parent, textvariable=left_var, width=14).grid(row=row, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(parent, text=right_label).grid(row=row, column=2, sticky="w", padx=4, pady=2)
        if combobox_values[1]:
            ttk.Combobox(parent, textvariable=right_var, values=combobox_values[1], state="readonly", width=14).grid(
                row=row, column=3, sticky="ew", padx=4, pady=2
            )
        else:
            ttk.Entry(parent, textvariable=right_var, width=14).grid(row=row, column=3, sticky="ew", padx=4, pady=2)

    def _build_controls_row(self):
        row = ttk.Frame(self)
        row.grid(row=1, column=0, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        for index in range(8):
            row.columnconfigure(index, weight=1, uniform="controls")

        ttk.Button(row, text="Run single episode", command=self._run_single_episode).grid(row=0, column=0, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        self.train_button = ttk.Button(row, text="Train and Run", command=self._train_and_run)
        self.train_button.grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        self.pause_button = ttk.Button(row, text="Pause", command=self._toggle_pause)
        self.pause_button.grid(row=0, column=2, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        ttk.Button(row, text="Reset All", command=self._reset_all).grid(row=0, column=3, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        ttk.Button(row, text="Clear Plot", command=self._clear_plot).grid(row=0, column=4, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        ttk.Button(row, text="Save samplings CSV", command=self._save_samplings_csv).grid(row=0, column=5, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        ttk.Button(row, text="Save Plot PNG", command=self._save_plot_png).grid(row=0, column=6, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)

        device_text = "CPU"
        self.device_button = ttk.Button(row, text=f"Current device: {device_text}", command=self._show_device)
        self.device_button.grid(row=0, column=7, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)
        self._update_control_highlights()

    def _update_control_highlights(self):
        is_running = bool(self.worker_thread and self.worker_thread.is_alive())
        self.train_button.configure(style="Primary.TButton" if is_running else "TButton")
        self.pause_button.configure(style="Pause.TButton" if (is_running and self.is_paused) else "TButton")

    def _build_current_run_panel(self):
        frame = ttk.LabelFrame(self, text="Current Run")
        frame.grid(row=2, column=0, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        frame.columnconfigure(0, weight=0, minsize=self.LABEL_COL_WIDTH)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Steps").grid(row=0, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        self.steps_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)

        ttk.Label(frame, text="Episodes").grid(row=1, column=0, sticky="w", padx=self.PAD_TIGHT, pady=2)
        self.episodes_progress = ttk.Progressbar(frame, mode="determinate", maximum=100)
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=self.PAD_TIGHT, pady=2)

        ttk.Label(frame, textvariable=self.var_status).grid(row=2, column=0, columnspan=2, sticky="w", padx=self.PAD_TIGHT, pady=2)

    def _build_live_plot_panel(self):
        panel = ttk.LabelFrame(self, text="Live Plot")
        panel.grid(row=3, column=0, sticky="nsew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        panel.rowconfigure(0, weight=1)
        panel.columnconfigure(0, weight=1)

        self.figure, self.ax = plt.subplots(figsize=(10, 4), dpi=100)
        self.figure.patch.set_facecolor(self.colors["main_bg"])
        self.figure.subplots_adjust(left=0.07, right=0.78)
        self.ax.set_facecolor(self.colors["panel_bg"])
        self.ax.tick_params(colors=self.colors["fg"])
        self.ax.grid(True, alpha=0.3, linewidth=0.9)
        self.ax.spines["bottom"].set_color(self.colors["fg"])
        self.ax.spines["left"].set_color(self.colors["fg"])
        self.ax.spines["top"].set_alpha(0.5)
        self.ax.spines["right"].set_alpha(0.5)
        self.ax.set_xlabel("Episode", color=self.colors["fg"])
        self.ax.set_ylabel("Reward", color=self.colors["fg"])

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=panel)
        self.plot_canvas.draw_idle()
        self.plot_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.figure.canvas.mpl_connect("pick_event", self._on_legend_pick)

    def _build_config(self) -> BipedalWalkerConfig:
        return BipedalWalkerConfig(
            hardcore=bool(self.var_hardcore.get()),
            animation_on=bool(self.var_animation_on.get()),
            animation_fps=max(1, int(self.var_animation_fps.get())),
            update_rate_episodes=max(1, int(self.var_update_rate.get())),
            max_steps=max(1, int(self.var_max_steps.get())),
            episodes=max(1, int(self.var_episodes.get())),
            epsilon_max=float(self.var_epsilon_max.get()),
            epsilon_decay=float(self.var_epsilon_decay.get()),
            epsilon_min=float(self.var_epsilon_min.get()),
            gamma=float(self.var_gamma.get()),
            policy=self.var_policy.get(),
            hidden_layer=self.var_hidden_layer.get(),
            activation=self.var_activation.get(),
            learning_rate=float(self.var_lr.get()),
            lr_strategy=self.var_lr_strategy.get(),
            min_lr=float(self.var_min_lr.get()),
            lr_decay=float(self.var_lr_decay.get()),
            replay_size=max(1000, int(self.var_replay_size.get())),
            batch_size=max(16, int(self.var_batch_size.get())),
            learning_start=max(0, int(self.var_learning_start.get())),
            learning_frequency=max(1, int(self.var_learning_frequency.get())),
            target_update=max(1, int(self.var_target_update.get())),
            moving_average_values=max(1, int(self.var_moving_average.get())),
            short_episode_full_capture_steps=max(10, int(self.var_rollout_short_capture_steps.get())),
            low_overhead_animation=bool(self.var_low_overhead_animation.get()),
            device="cpu",
        )

    def _on_compare_toggle(self):
        if self.var_compare_on.get() and self.var_animation_on.get():
            self.var_animation_on.set(False)
            self._apply_runtime_environment_settings()

    def _on_animation_toggle_changed(self):
        self._apply_runtime_environment_settings()

    def _on_update_environment(self):
        self._apply_runtime_environment_settings()

    def _apply_runtime_environment_settings(self):
        animation_on = bool(self.var_animation_on.get())

        with self._active_trainers_lock:
            trainers = list(self._active_trainers)

        for trainer in trainers:
            trainer.update_environment(
                self.var_hardcore.get(),
                animation_on=animation_on,
                animation_fps=max(1, int(self.var_animation_fps.get())),
                update_rate_episodes=max(1, int(self.var_update_rate.get())),
            )

        if not animation_on:
            self._rollout_playback_frames = []
            self._rollout_playback_index = 0
            self._rollout_playback_total = 0
            self.steps_progress["value"] = 0
            self.render_state = "off"
            current = self.var_status.get()
            if "| Render:" in current:
                prefix = current.split("| Render:")[0].rstrip()
                self._set_status(f"{prefix} | Render: off")
            else:
                self._set_status("Epsilon: - | LR: - | Best reward: - | Render: off")

    def _parse_compare_values(self, key: str, raw_values: str) -> List[Any]:
        if not raw_values.strip():
            return []
        parts = [value.strip() for value in raw_values.split(",") if value.strip()]
        parsed: List[Any] = []
        for value in parts:
            if key == "Policy":
                if value in POLICY_DISPLAY_NAMES:
                    parsed.append(value)
            elif key == "hardcore":
                parsed.append(value.lower() in {"true", "1", "yes", "on"})
            elif key in {"batch_size", "max_steps", "episodes"}:
                parsed.append(int(value))
            else:
                parsed.append(float(value))
        return parsed

    def _sync_current_compare_param(self):
        key = self.var_compare_param.get()
        values = self._parse_compare_values(key, self.var_compare_values.get())
        if values:
            self.compare_values[key] = values
        elif key in self.compare_values:
            del self.compare_values[key]
        self._update_compare_summary()

    def _clear_compare(self):
        self.compare_values.clear()
        self.var_compare_values.set("")
        self._update_compare_summary()

    def _update_compare_summary(self):
        lines = [f"{key}: {values}" for key, values in self.compare_values.items()]
        self.compare_summary.configure(state="normal")
        self.compare_summary.delete("1.0", tk.END)
        self.compare_summary.insert(tk.END, "\n".join(lines) if lines else "No active compare parameters")
        self.compare_summary.configure(state="disabled")

    def _apply_policy_defaults(self, policy: str):
        defaults = self.policy_defaults.get(policy, {})
        if "hidden_layer" in defaults:
            self.var_hidden_layer.set(defaults["hidden_layer"])
        if "activation" in defaults:
            self.var_activation.set(defaults["activation"])
        if "learning_rate" in defaults:
            self.var_lr.set(f"{defaults['learning_rate']:.1e}")
        if "lr_strategy" in defaults:
            self.var_lr_strategy.set(defaults["lr_strategy"])
        if "min_lr" in defaults:
            self.var_min_lr.set(f"{defaults['min_lr']:.1e}")
        if "lr_decay" in defaults:
            self.var_lr_decay.set(defaults["lr_decay"])
        if "replay_size" in defaults:
            self.var_replay_size.set(defaults["replay_size"])
        if "batch_size" in defaults:
            self.var_batch_size.set(defaults["batch_size"])
        if "learning_start" in defaults:
            self.var_learning_start.set(defaults["learning_start"])
        if "learning_frequency" in defaults:
            self.var_learning_frequency.set(defaults["learning_frequency"])
        if "target_update" in defaults:
            self.var_target_update.set(defaults["target_update"])
        if "gamma" in defaults:
            self.var_gamma.set(defaults["gamma"])

    def _build_compare_configs(self, base_config: BipedalWalkerConfig) -> List[BipedalWalkerConfig]:
        self._sync_current_compare_param()
        if not self.var_compare_on.get():
            return [base_config]

        combinations = build_compare_combinations(self.compare_values)
        configs: List[BipedalWalkerConfig] = []

        for values in combinations:
            if "Policy" in values:
                chosen_policy = values["Policy"]
                policy_defaults = self.policy_defaults.get(chosen_policy, {})
                cfg = replace(base_config, policy=chosen_policy)
                for key, value in policy_defaults.items():
                    setattr(cfg, key, value)
            else:
                cfg = replace(base_config)

            mapped = {}
            for key, value in values.items():
                if key == "Policy":
                    mapped["policy"] = value
                elif key == "learning_rate":
                    mapped["learning_rate"] = value
                else:
                    mapped[key] = value

            for key, value in mapped.items():
                setattr(cfg, key, value)
            configs.append(cfg)

        return configs or [base_config]

    def _clear_pending_events(self):
        while not self.event_queue.empty():
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def _run_single_episode(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A training run is already active.")
            return

        config = self._build_config()
        config = replace(config, episodes=1)
        self._start_worker([config], collect_transitions=False)

    def _train_and_run(self):
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Busy", "A training run is already active.")
            return

        base = self._build_config()
        configs = self._build_compare_configs(base)
        self._start_worker(configs, collect_transitions=True)

    def _start_worker(self, configs: List[BipedalWalkerConfig], collect_transitions: bool):
        self._clear_pending_events()
        self.is_paused = False
        self.pause_button.configure(text="Pause")
        self._update_control_highlights()
        self._set_status("Epsilon: - | LR: - | Best reward: - | Render: idle")
        self.steps_progress["value"] = 0
        self._defer_plot_refresh = False

        base_run_counter = self.run_counter
        self.run_counter += len(configs)
        run_items: List[Tuple[BipedalWalkerConfig, str]] = []
        for idx, config in enumerate(configs, start=1):
            run_id = base_run_counter + idx
            run_label = f"{make_run_label(config)} | run={run_id}.{idx}"
            run_items.append((config, run_label))

        def register_trainer(trainer: BipedalWalkerTrainer):
            with self._active_trainers_lock:
                self._active_trainers.append(trainer)
                self.active_trainer = trainer

        def unregister_trainer(trainer: BipedalWalkerTrainer):
            with self._active_trainers_lock:
                self._active_trainers = [entry for entry in self._active_trainers if entry is not trainer]
                self._last_finished_trainer = trainer
                self.active_trainer = self._active_trainers[-1] if self._active_trainers else None

        def stop_all_active_trainers():
            with self._active_trainers_lock:
                trainers = list(self._active_trainers)
            for trainer in trainers:
                trainer.stop()

        def run_one(config: BipedalWalkerConfig, run_label: str) -> bool:
            trainer = BipedalWalkerTrainer(config, event_callback=self.event_queue.put)
            register_trainer(trainer)
            try:
                if self.is_paused:
                    trainer.set_paused(True)
                trainer.train(collect_transitions=collect_transitions, run_label=run_label)
                return trainer.stop_event.is_set()
            finally:
                unregister_trainer(trainer)

        def worker():
            try:
                if len(run_items) <= 1:
                    for config, run_label in run_items:
                        if run_one(config, run_label):
                            break
                else:
                    max_workers = max(1, min(self.COMPARE_MAX_WORKERS, len(run_items)))
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(run_one, config, run_label) for config, run_label in run_items]
                        for future in as_completed(futures):
                            if future.result():
                                stop_all_active_trainers()
                                break
            except Exception as exc:
                stop_all_active_trainers()
                self.event_queue.put({"type": "error", "message": str(exc), "run_label": "worker"})
            finally:
                self.event_queue.put({"type": "worker_done"})

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
        self._update_control_highlights()

    def _toggle_pause(self):
        with self._active_trainers_lock:
            trainers = list(self._active_trainers)

        if not trainers:
            return
        self.is_paused = not self.is_paused
        for trainer in trainers:
            trainer.set_paused(self.is_paused)
        self.pause_button.configure(text="Run" if self.is_paused else "Pause")
        self._update_control_highlights()

    def _reset_all(self):
        with self._active_trainers_lock:
            trainers = list(self._active_trainers)
        for trainer in trainers:
            trainer.stop()

        self.var_animation_on.set(True)
        self.var_animation_fps.set(10)
        self.var_update_rate.set(5)
        self.var_hardcore.set(False)
        self.var_compare_on.set(False)
        self._clear_compare()

        self.var_max_steps.set(1600)
        self.var_episodes.set(50)
        self.var_epsilon_max.set(1.0)
        self.var_epsilon_decay.set(0.995)
        self.var_epsilon_min.set(0.05)
        self.var_gamma.set(0.99)

        self.var_policy.set("PPO")
        self._apply_policy_defaults("PPO")
        self.var_moving_average.set(20)
        self.var_low_overhead_animation.set(False)

        self.steps_progress["value"] = 0
        self.episodes_progress["value"] = 0
        self._set_status("Epsilon: - | LR: - | Best reward: - | Render: idle")
        self._clear_plot()
        self.is_paused = False
        self.pause_button.configure(text="Pause")
        self._update_control_highlights()

    def _clear_plot(self):
        self.run_histories.clear()
        self.lines_by_run.clear()
        self.ax.clear()
        self.ax.set_facecolor(self.colors["panel_bg"])
        self.ax.tick_params(colors=self.colors["fg"])
        self.ax.grid(True, alpha=0.3, linewidth=0.9)
        self.ax.spines["bottom"].set_color(self.colors["fg"])
        self.ax.spines["left"].set_color(self.colors["fg"])
        self.ax.spines["top"].set_alpha(0.5)
        self.ax.spines["right"].set_alpha(0.5)
        self.ax.set_xlabel("Episode", color=self.colors["fg"])
        self.ax.set_ylabel("Reward", color=self.colors["fg"])
        self.legend_artist_map.clear()
        self._refresh_plot(force=True)

    def _save_samplings_csv(self):
        trainer = self.active_trainer or self._last_finished_trainer
        if not trainer:
            messagebox.showinfo("No data", "No trainer data available yet.")
            return
        path = trainer.save_transitions_csv(self.results_dir, "BipedalWalker_samples")
        if path is None:
            messagebox.showinfo("No data", "No transition samples collected yet.")
        else:
            messagebox.showinfo("Saved", f"CSV saved to:\n{path}")

    def _save_plot_png(self):
        config = self._build_config()
        filename = png_filename(config)
        output = self.plots_dir / filename
        self.figure.savefig(output, dpi=150)
        messagebox.showinfo("Saved", f"Plot saved to:\n{output}")

    def _show_device(self):
        messagebox.showinfo("Device", "Current device: CPU")

    def _pump_events(self):
        for _ in range(200):
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break
            self._handle_event(event)

        self._update_control_highlights()

        self.after(50, self._pump_events)

    def _handle_event(self, event: Dict[str, Any]):
        event_type = event.get("type")
        if event_type == "episode":
            self._handle_episode_event(event)
        elif event_type == "training_done":
            self._handle_training_done(event)
        elif event_type == "paused":
            self.render_state = "off"
        elif event_type == "error":
            messagebox.showerror("Training error", event.get("message", "Unknown error"))
            self._set_status("Epsilon: - | LR: - | Best reward: - | Render: off")
        elif event_type == "worker_done":
            self.active_trainer = None
            with self._active_trainers_lock:
                self._active_trainers.clear()
            self.is_paused = False
            self.pause_button.configure(text="Pause")
            self._update_control_highlights()

    def _handle_episode_event(self, event: Dict[str, Any]):
        run_label = event["run_label"]
        history = self.run_histories.setdefault(
            run_label,
            {
                "reward": [],
                "moving_average": [],
                "eval": [],
                "config": None,
            },
        )
        history["reward"].append(float(event["reward"]))
        history["moving_average"].append(float(event["moving_average"]))
        if event.get("eval_reward") is not None:
            history["eval"].append(float(event["eval_reward"]))

        episode = max(1, int(event["episode"]))
        episodes = max(1, int(event["episodes"]))

        self.episodes_progress["value"] = 100.0 * episode / episodes

        self.render_state = event.get("render_state", "idle")
        self._set_status(
            f"Epsilon: {event['epsilon']:.4f} | LR: {event['learning_rate']:.2e} | "
            f"Best reward: {event['best_reward']:.2f} | Render: {self.render_state}"
        )

        rollout_frames = event.get("rollout_frames") or []
        if rollout_frames:
            self._start_rollout_playback(rollout_frames)
        else:
            self.steps_progress["value"] = 0
            frame = event.get("frame")
            if frame is not None:
                with self.latest_frame_lock:
                    self.latest_frame = frame
                    self._latest_frame_version += 1

        if bool(event.get("plot_refresh", True)):
            if self.var_low_overhead_animation.get() and rollout_frames:
                self._defer_plot_refresh = True
            else:
                self._refresh_plot()

    def _start_rollout_playback(self, frames: List[Any]):
        self._rollout_playback_frames = list(frames)
        self._rollout_playback_index = 0
        self._rollout_playback_total = len(self._rollout_playback_frames)
        self.steps_progress["value"] = 0
        if not self._rollout_playback_frames:
            return
        with self.latest_frame_lock:
            self.latest_frame = self._rollout_playback_frames[0]
            self._latest_frame_version += 1

    def _handle_training_done(self, event: Dict[str, Any]):
        run_label = event["run_label"]
        history = event.get("history", {})
        config = event.get("config", {})

        entry = self.run_histories.setdefault(run_label, {"reward": [], "moving_average": [], "eval": [], "config": None})
        entry["reward"] = history.get("reward", entry["reward"])
        entry["moving_average"] = history.get("moving_average", entry["moving_average"])
        entry["eval"] = history.get("eval", entry["eval"])
        entry["config"] = config
        self._refresh_plot(force=True)

    def _set_status(self, text: str):
        self.var_status.set(text)

    def _refresh_plot(self, force: bool = False):
        now = time.time()
        if not force and now - self._last_plot_draw < 0.2:
            return
        self._last_plot_draw = now

        colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#4c72b0", "#dd8452", "#55a868", "#c44e52"])
        self.ax.clear()
        self.ax.set_facecolor(self.colors["panel_bg"])
        self.ax.tick_params(colors=self.colors["fg"])
        self.ax.spines["bottom"].set_color(self.colors["fg"])
        self.ax.spines["left"].set_color(self.colors["fg"])
        self.ax.set_xlabel("Episode", color=self.colors["fg"])
        self.ax.set_ylabel("Reward", color=self.colors["fg"])

        legend_handles = []
        legend_labels = []
        self.legend_artist_map.clear()

        for index, (run_label, history) in enumerate(self.run_histories.items()):
            color = colors[index % len(colors)]
            rewards = history.get("reward", [])
            moving_avg = history.get("moving_average", [])
            eval_values = history.get("eval", [])
            if not rewards:
                continue

            x = np.arange(1, len(rewards) + 1)
            reward_line, = self.ax.plot(x, rewards, color=color, linewidth=1.4)
            ma_line, = self.ax.plot(x, moving_avg, color=color, linewidth=2.0, alpha=0.7, linestyle="--")

            eval_x = np.linspace(1, len(rewards), num=len(eval_values), dtype=int) if eval_values else []
            eval_line = None
            if len(eval_x) > 0:
                eval_line, = self.ax.plot(eval_x, eval_values, color=color, linewidth=1.8, linestyle=":")

            wrapped = run_label.replace(", ", ",\n", 1)
            legend_handles.append(reward_line)
            legend_labels.append(wrapped)
            legend_handles.append(ma_line)
            legend_labels.append("MA")
            if eval_line is not None:
                legend_handles.append(eval_line)
                legend_labels.append("eval")

        if legend_handles:
            legend = self.ax.legend(
                legend_handles,
                legend_labels,
                loc="center left",
                bbox_to_anchor=(1.01, 0.5),
                frameon=False,
                labelcolor=self.colors["fg"],
                borderaxespad=0.0,
            )
            for text in legend.get_texts():
                text.set_picker(True)
            for handle in legend.legend_handles:
                handle.set_picker(True)

            for artist, target in zip(legend.legend_handles + legend.get_texts(), legend_handles + legend_handles):
                self.legend_artist_map[artist] = target

        self.plot_canvas.draw_idle()

    def _on_legend_pick(self, event):
        artist = event.artist
        target = self.legend_artist_map.get(artist)
        if target is None:
            return
        visible = not target.get_visible()
        target.set_visible(visible)
        self.plot_canvas.draw_idle()

    def _render_tick(self):
        fps = max(1, int(self.var_animation_fps.get()))
        interval = max(20, int(1000 / fps))

        if self._rollout_playback_frames:
            if self._rollout_playback_index < len(self._rollout_playback_frames):
                frame = self._rollout_playback_frames[self._rollout_playback_index]
                with self.latest_frame_lock:
                    self.latest_frame = frame
                    self._latest_frame_version += 1
                self._rollout_playback_index += 1
                if self._rollout_playback_total > 0:
                    self.steps_progress["value"] = min(
                        100.0,
                        100.0 * self._rollout_playback_index / self._rollout_playback_total,
                    )
            else:
                self._rollout_playback_frames = []
                self._rollout_playback_index = 0
                self._rollout_playback_total = 0
                self.steps_progress["value"] = 100.0
                if self._defer_plot_refresh:
                    self._defer_plot_refresh = False
                    self._refresh_plot(force=True)

        self._draw_latest_frame()
        self.after(interval, self._render_tick)

    def _on_render_resize(self, _):
        if self._pending_resize_job is not None:
            self.after_cancel(self._pending_resize_job)
        self._rendered_frame_version = -1
        self._pending_resize_job = self.after(100, self._draw_latest_frame)

    def _draw_latest_frame(self):
        with self.latest_frame_lock:
            frame = self.latest_frame
            frame_version = self._latest_frame_version

        if frame is None or not PIL_AVAILABLE:
            return

        if frame_version == self._rendered_frame_version:
            return

        canvas_width = max(1, self.render_canvas.winfo_width())
        canvas_height = max(1, self.render_canvas.winfo_height())
        if canvas_width <= 2 or canvas_height <= 2:
            return

        image = Image.fromarray(frame)
        src_w, src_h = image.size
        if src_w <= 0 or src_h <= 0:
            return

        scale = min(canvas_width / src_w, canvas_height / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.BILINEAR)
        self.render_photo = ImageTk.PhotoImage(resized)

        if self._canvas_image_id is None:
            self._canvas_image_id = self.render_canvas.create_image(
                canvas_width // 2,
                canvas_height // 2,
                image=self.render_photo,
                anchor="center",
            )
        else:
            self.render_canvas.coords(self._canvas_image_id, canvas_width // 2, canvas_height // 2)
            self.render_canvas.itemconfigure(self._canvas_image_id, image=self.render_photo)

        self._rendered_frame_version = frame_version
