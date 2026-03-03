import itertools
import os
import queue
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox, ttk

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover
    Image = None
    ImageTk = None

from InvDoubPend_logic import (
    ALLOWED_POLICIES,
    InvertedDoublePendulumEnvironment,
    InvertedDoublePendulumTrainer,
    EnvironmentConfig,
    build_run_label,
    timestamp_run_id,
)


class InvDoubPendGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Inverted Double Pendulum - SB3")
        self.root.configure(bg="#1e1e1e")
        self.root.geometry("1500x920")

        self.PAD_OUTER = 10
        self.PAD_INNER = 6
        self.PAD_TIGHT = 4
        self.LABEL_COL_WIDTH = 92
        self.PARAM_INPUT_WIDTH = 9

        self.event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.training_active = False
        self.paused = False
        self.current_workers: List[InvertedDoublePendulumTrainer] = []
        self.main_worker_thread: Optional[threading.Thread] = None
        self.run_artifacts: Dict[str, Dict[str, Any]] = {}
        self.plot_series: Dict[str, Dict[str, Any]] = {}
        self.finalized_runs: List[str] = []
        self.render_run_id: Optional[str] = None
        self.current_training_session_id = 0

        self.latest_frame = None
        self.canvas_photo = None
        self.current_canvas_size = (1, 1)
        self.last_rendered_frame_id = None
        self.playback_frames: List[Any] = []
        self.playback_index = 0

        self._setup_style()
        self._build_layout()
        self._initialize_defaults()
        self._apply_policy_defaults()

        self._start_event_pump()
        self._start_render_tick()

    def _setup_style(self):
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
        style.configure("Panel.TLabelframe", background=bg_panel, foreground=fg_text, padding=(8, 8))
        style.configure("Panel.TLabelframe.Label", background=bg_panel, foreground=fg_text, font=heading_font)
        style.configure("TLabel", background=bg_panel, foreground=fg_text, font=default_font)
        style.configure("Muted.TLabel", background=bg_panel, foreground=fg_muted, font=default_font)

        style.configure("TButton", background="#3a3d41", foreground=fg_text, font=button_font, padding=(10, 5))
        style.map("TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")], foreground=[("disabled", "#b8b8b8")])

        style.configure("Control.TButton", background="#3a3d41", foreground=fg_text, font=button_font, padding=(10, 5))
        style.map("Control.TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")], foreground=[("disabled", "#b8b8b8")])

        style.configure("Accent.TButton", background=accent, foreground="white", font=button_font, padding=(10, 5))
        style.map("Accent.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")], foreground=[("disabled", "#ededed")])

        style.configure("Amber.TButton", background="#a66a00", foreground="white", font=button_font, padding=(10, 5))
        style.map("Amber.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")], foreground=[("disabled", "#ededed")])

        style.configure("TEntry", fieldbackground=bg_input, foreground=fg_text, insertcolor=fg_text, padding=(6, 4))
        style.configure("TCombobox", fieldbackground=bg_input, foreground=fg_text, padding=(6, 4))
        style.map("TCombobox", fieldbackground=[("readonly", bg_input)], foreground=[("readonly", fg_text)])
        style.configure("TCheckbutton", background=bg_panel, foreground=fg_text, font=default_font)
        style.map("TCheckbutton", background=[("active", bg_panel), ("!active", bg_panel)], foreground=[("active", fg_text), ("!active", fg_text)])
        style.configure("TProgressbar", troughcolor="#343434", background=accent)

    def _build_layout(self):
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=50)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_rowconfigure(2, weight=0)
        self.root.grid_rowconfigure(3, weight=36)

        self.env_panel = ttk.LabelFrame(self.root, text="Environment", style="Panel.TLabelframe")
        self.env_panel.grid(row=0, column=0, sticky="nsew", padx=self.PAD_OUTER, pady=self.PAD_OUTER)
        self.env_panel.grid_rowconfigure(0, weight=1)
        self.env_panel.grid_columnconfigure(0, weight=1)

        self.render_canvas = tk.Canvas(self.env_panel, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.render_canvas.bind("<Configure>", self._on_canvas_resize)

        self.params_panel = ttk.LabelFrame(self.root, text="Parameters", style="Panel.TLabelframe")
        self.params_panel.grid(row=0, column=1, sticky="nsew", padx=(0, self.PAD_OUTER), pady=self.PAD_OUTER)
        self.params_panel.grid_rowconfigure(0, weight=1)
        self.params_panel.grid_columnconfigure(0, weight=1)

        self._build_scrollable_parameters()

        self.controls_panel = ttk.LabelFrame(self.root, text="Controls", style="Panel.TLabelframe")
        self.controls_panel.grid(row=1, column=0, columnspan=2, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        for i in range(8):
            self.controls_panel.grid_columnconfigure(i, weight=1)

        self.current_panel = ttk.LabelFrame(self.root, text="Current Run", style="Panel.TLabelframe")
        self.current_panel.grid(row=2, column=0, columnspan=2, sticky="ew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        self.current_panel.grid_columnconfigure(0, weight=0, minsize=self.LABEL_COL_WIDTH)
        self.current_panel.grid_columnconfigure(1, weight=1)

        self.plot_panel = ttk.LabelFrame(self.root, text="Live Plot", style="Panel.TLabelframe")
        self.plot_panel.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=self.PAD_OUTER, pady=(0, self.PAD_OUTER))
        self.plot_panel.grid_rowconfigure(0, weight=1)
        self.plot_panel.grid_columnconfigure(0, weight=1)

        self._build_controls()
        self._build_current_run()
        self._build_plot()

    def _build_scrollable_parameters(self):
        container = ttk.Frame(self.params_panel)
        container.grid(row=0, column=0, sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.param_canvas = tk.Canvas(container, bg="#252526", highlightthickness=0)
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        self.param_scrollbar = ttk.Scrollbar(container, orient="vertical", command=self.param_canvas.yview)
        self.param_scrollbar.grid(row=0, column=1, sticky="ns")
        self.param_canvas.configure(yscrollcommand=self.param_scrollbar.set)

        self.params_content = ttk.Frame(self.param_canvas)
        self.params_window = self.param_canvas.create_window((0, 0), window=self.params_content, anchor="nw")

        self.params_content.bind("<Configure>", self._on_params_content_configure)
        self.param_canvas.bind("<Configure>", self._on_params_canvas_configure)
        self.param_canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.param_canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        self._build_param_groups()

    def _build_param_groups(self):
        self.params_content.grid_columnconfigure(0, weight=1)

        self.group_env = ttk.LabelFrame(self.params_content, text="Environment", style="Panel.TLabelframe")
        self.group_compare = ttk.LabelFrame(self.params_content, text="Compare", style="Panel.TLabelframe")
        self.group_general = ttk.LabelFrame(self.params_content, text="General", style="Panel.TLabelframe")
        self.group_specific = ttk.LabelFrame(self.params_content, text="Specific", style="Panel.TLabelframe")
        self.group_plot = ttk.LabelFrame(self.params_content, text="Live Plot", style="Panel.TLabelframe")

        self.group_env.grid(row=0, column=0, sticky="ew", padx=6, pady=4)
        self.group_compare.grid(row=1, column=0, sticky="ew", padx=6, pady=4)
        self.group_general.grid(row=2, column=0, sticky="ew", padx=6, pady=4)
        self.group_specific.grid(row=3, column=0, sticky="ew", padx=6, pady=4)
        self.group_plot.grid(row=4, column=0, sticky="ew", padx=6, pady=4)

        self.group_env.grid_columnconfigure(1, weight=1)
        self.group_env.grid_columnconfigure(3, weight=1)
        self.group_compare.grid_columnconfigure(0, weight=1, uniform="compare_fields")
        self.group_compare.grid_columnconfigure(1, weight=1, uniform="compare_fields")
        self.group_general.grid_columnconfigure(1, weight=1, uniform="general_inputs")
        self.group_general.grid_columnconfigure(3, weight=1, uniform="general_inputs")
        self.group_specific.grid_columnconfigure(1, weight=1)
        self.group_specific.grid_columnconfigure(3, weight=1)
        self.group_plot.grid_columnconfigure(1, weight=1)

        self._build_environment_group()
        self._build_compare_group()
        self._build_general_group()
        self._build_specific_group()
        self._build_live_plot_group()

    def _build_environment_group(self):
        self.var_animation_on = tk.BooleanVar(value=True)
        self.var_animation_fps = tk.StringVar(value="10")
        self.var_update_rate = tk.StringVar(value="5")
        self.var_healthy_reward = tk.StringVar(value="10")
        self.var_reset_noise_scale = tk.StringVar(value="0.1")

        ttk.Checkbutton(self.group_env, text="Animation on", variable=self.var_animation_on, command=self._on_animation_toggle).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self._entry_row(self.group_env, 0, 1, "Animation FPS", self.var_animation_fps)
        self._entry_row(self.group_env, 1, 0, "Update rate (episodes)", self.var_update_rate)
        ttk.Button(self.group_env, text="Update", style="Control.TButton", command=self._apply_environment_update).grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=(6, 6))
        self._entry_row(self.group_env, 3, 0, "healthy_reward", self.var_healthy_reward)
        self._entry_row(self.group_env, 3, 1, "reset_noise_scale", self.var_reset_noise_scale)


    def _build_compare_group(self):
        self.var_compare_on = tk.BooleanVar(value=False)
        self.var_compare_param = tk.StringVar(value="Policy")
        self.var_compare_values = tk.StringVar(value="")
        self.var_compare_value_hint = tk.StringVar(value="")
        self.compare_entries: Dict[str, List[str]] = {}
        self.compare_param_options = [
            "Policy",
            "Max steps",
            "Episodes",
            "Epsilon max",
            "Epsilon decay",
            "Epsilon min",
            "Gamma",
            "Hidden layer",
            "Activation",
            "LR",
            "LR strategy",
            "Min LR",
            "LR decay",
            "Replay size",
            "Batch size",
            "Learning start",
            "Learning frequency",
            "Target update",
        ]
        self.compare_value_suggestions: Dict[str, List[str]] = {
            "Policy": list(ALLOWED_POLICIES),
            "Activation": ["ReLU", "Tanh", "ELU", "LeakyReLU"],
            "LR strategy": ["constant", "decay"],
        }

        ttk.Checkbutton(self.group_compare, text="Compare on", variable=self.var_compare_on, command=self._on_compare_toggle).grid(row=0, column=0, sticky="w", padx=6, pady=3)

        self.compare_buttons_row = ttk.Frame(self.group_compare)
        self.compare_buttons_row.grid(row=0, column=1, sticky="e", padx=6, pady=3)
        self.compare_buttons_row.grid_columnconfigure(0, weight=1, uniform="compare_btns")
        self.compare_buttons_row.grid_columnconfigure(1, weight=1, uniform="compare_btns")
        ttk.Button(self.compare_buttons_row, text="Clear", style="Control.TButton", command=self._clear_compare_entries).grid(row=0, column=0, sticky="ew", padx=(0, 3), pady=0)
        ttk.Button(self.compare_buttons_row, text="Add", style="Control.TButton", command=self._add_compare_entry).grid(row=0, column=1, sticky="ew", padx=(3, 0), pady=0)

        self.compare_param_combo = ttk.Combobox(
            self.group_compare,
            textvariable=self.var_compare_param,
            values=self.compare_param_options,
            state="readonly",
            width=self.PARAM_INPUT_WIDTH,
        )
        self.compare_param_combo.grid(row=1, column=0, sticky="ew", padx=6, pady=3)
        self.compare_param_combo.bind("<<ComboboxSelected>>", self._on_compare_param_changed)

        self.compare_values_entry = ttk.Entry(
            self.group_compare,
            textvariable=self.var_compare_values,
            width=self.PARAM_INPUT_WIDTH,
        )
        self.compare_values_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        self.compare_values_entry.bind("<Tab>", self._on_compare_values_tab_complete)
        self.compare_values_entry.bind("<KeyRelease>", self._on_compare_values_key_release)
        self.compare_values_entry.bind("<Return>", self._on_compare_values_enter)
        self.compare_values_entry.bind("<KP_Enter>", self._on_compare_values_enter)

        self.compare_hint_label = ttk.Label(self.group_compare, textvariable=self.var_compare_value_hint, style="Muted.TLabel")
        self.compare_hint_label.grid(row=2, column=1, sticky="w", padx=6, pady=(0, 3))

        self.compare_summary = ttk.Label(self.group_compare, text="(none)", style="Muted.TLabel", justify="left")
        self.compare_summary.grid(row=3, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        self._on_compare_param_changed()

    def _on_compare_param_changed(self, _event=None):
        self._update_compare_value_hint()

    def _on_compare_values_key_release(self, _event=None):
        self._update_compare_value_hint()

    def _get_compare_completion_candidate(self) -> Optional[str]:
        param = self.var_compare_param.get().strip()
        suggestions = self.compare_value_suggestions.get(param, [])
        if not suggestions:
            return None

        raw = self.var_compare_values.get()
        token = raw.split(",")[-1].strip()
        if token == "":
            return None

        token_low = token.lower()
        for candidate in suggestions:
            if candidate.lower().startswith(token_low):
                if candidate.lower() == token_low:
                    return None
                return candidate
        return None

    def _update_compare_value_hint(self):
        candidate = self._get_compare_completion_candidate()
        if candidate:
            self.var_compare_value_hint.set(f"Tab → {candidate}")
        else:
            self.var_compare_value_hint.set("")

    def _on_compare_values_tab_complete(self, event):
        candidate = self._get_compare_completion_candidate()
        if candidate is None:
            return None

        raw = self.var_compare_values.get()
        parts = raw.split(",")
        prefix = ",".join(parts[:-1]).strip()
        completed = f"{prefix}, {candidate}" if prefix else candidate
        self.var_compare_values.set(completed)
        event.widget.icursor(len(completed))
        self._update_compare_value_hint()
        return "break"

    def _on_compare_values_enter(self, _event=None):
        self._add_compare_entry()
        return "break"

    def _build_general_group(self):
        self.var_max_steps = tk.StringVar(value="1000")
        self.var_episodes = tk.StringVar(value="100")
        self.var_epsilon_max = tk.StringVar(value="0.0")
        self.var_epsilon_decay = tk.StringVar(value="0.995")
        self.var_epsilon_min = tk.StringVar(value="0.0")
        self.var_gamma = tk.StringVar(value="0.99")

        self._entry_row(self.group_general, 0, 0, "Max steps", self.var_max_steps)
        self._entry_row(self.group_general, 0, 1, "Episodes", self.var_episodes)
        self._entry_row(self.group_general, 1, 0, "Epsilon max", self.var_epsilon_max)
        self._entry_row(self.group_general, 1, 1, "Epsilon decay", self.var_epsilon_decay)
        self._entry_row(self.group_general, 2, 0, "Epsilon min", self.var_epsilon_min)
        self._entry_row(self.group_general, 2, 1, "Gamma", self.var_gamma)

    def _build_specific_group(self):
        self.var_policy = tk.StringVar(value="PPO")
        self.var_hidden_layer = tk.StringVar(value="256")
        self.var_activation = tk.StringVar(value="Tanh")
        self.var_lr = tk.StringVar(value="3e-4")
        self.var_lr_strategy = tk.StringVar(value="constant")
        self.var_min_lr = tk.StringVar(value="1e-5")
        self.var_lr_decay = tk.StringVar(value="0.995")
        self.var_replay_size = tk.StringVar(value="100000")
        self.var_batch_size = tk.StringVar(value="128")
        self.var_learning_start = tk.StringVar(value="0")
        self.var_learning_frequency = tk.StringVar(value="1")
        self.var_target_update = tk.StringVar(value="1")

        ttk.Label(self.group_specific, text="Policy").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.policy_combo = ttk.Combobox(self.group_specific, textvariable=self.var_policy, values=list(ALLOWED_POLICIES), state="readonly", width=self.PARAM_INPUT_WIDTH)
        self.policy_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        self.policy_combo.bind("<<ComboboxSelected>>", lambda _e: self._apply_policy_defaults())

        self._entry_row(self.group_specific, 1, 0, "Hidden layer", self.var_hidden_layer)
        self._combo_row(self.group_specific, 1, 1, "Activation", self.var_activation, ["ReLU", "Tanh", "ELU", "LeakyReLU"])
        self._entry_row(self.group_specific, 2, 0, "LR", self.var_lr)
        self._combo_row(self.group_specific, 2, 1, "LR strategy", self.var_lr_strategy, ["constant", "decay"])
        self._entry_row(self.group_specific, 3, 0, "Min LR", self.var_min_lr)
        self._entry_row(self.group_specific, 3, 1, "LR decay", self.var_lr_decay)
        self._entry_row(self.group_specific, 4, 0, "Replay size", self.var_replay_size)
        self._entry_row(self.group_specific, 4, 1, "Batch size", self.var_batch_size)
        self._entry_row(self.group_specific, 5, 0, "Learning start", self.var_learning_start)
        self._entry_row(self.group_specific, 5, 1, "Learning frequency", self.var_learning_frequency)
        self._entry_row(self.group_specific, 6, 0, "Target update", self.var_target_update)

    def _build_live_plot_group(self):
        self.var_moving_average = tk.StringVar(value="20")
        self.var_show_advanced = tk.BooleanVar(value=False)
        self.var_rollout_full_capture_steps = tk.StringVar(value="120")
        self.var_low_overhead = tk.BooleanVar(value=False)

        self._entry_row(self.group_plot, 0, 0, "Moving average values", self.var_moving_average)
        ttk.Checkbutton(self.group_plot, text="Show Advanced", variable=self.var_show_advanced, command=self._toggle_advanced_plot).grid(row=1, column=0, sticky="w", padx=6, pady=3)

        self.advanced_row_frame = ttk.Frame(self.group_plot)
        self.advanced_row_frame.grid(row=2, column=0, columnspan=2, sticky="ew")
        self.advanced_row_frame.grid_columnconfigure(1, weight=1)
        ttk.Label(self.advanced_row_frame, text="Rollout full-capture steps").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(self.advanced_row_frame, textvariable=self.var_rollout_full_capture_steps, width=self.PARAM_INPUT_WIDTH).grid(row=0, column=1, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(self.advanced_row_frame, text="Low-overhead animation", variable=self.var_low_overhead).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        self._toggle_advanced_plot()

    def _build_controls(self):
        self.var_device = tk.StringVar(value="CPU")
        self.btn_run_single = ttk.Button(self.controls_panel, text="Run single episode", style="Control.TButton", command=self.run_single_episode)
        self.btn_train = ttk.Button(self.controls_panel, text="Train and Run", style="Control.TButton", command=self.train_and_run)
        self.btn_pause = ttk.Button(self.controls_panel, text="Pause", style="Control.TButton", command=self.toggle_pause)
        self.btn_reset = ttk.Button(self.controls_panel, text="Reset All", style="Control.TButton", command=self.reset_all)
        self.btn_clear_plot = ttk.Button(self.controls_panel, text="Clear Plot", style="Control.TButton", command=self.clear_plot)
        self.btn_save_csv = ttk.Button(self.controls_panel, text="Save samplings CSV", style="Control.TButton", command=self.save_samplings_csv)
        self.btn_save_png = ttk.Button(self.controls_panel, text="Save Plot PNG", style="Control.TButton", command=self.save_plot_png)
        self.combo_device = ttk.Combobox(
            self.controls_panel,
            textvariable=self.var_device,
            values=["CPU", "GPU"],
            state="readonly",
            width=self.PARAM_INPUT_WIDTH,
        )
        self.combo_device.bind("<<ComboboxSelected>>", self._on_device_changed)

        buttons = [
            self.btn_run_single,
            self.btn_train,
            self.btn_pause,
            self.btn_reset,
            self.btn_clear_plot,
            self.btn_save_csv,
            self.btn_save_png,
            self.combo_device,
        ]
        for idx, button in enumerate(buttons):
            button.grid(row=0, column=idx, sticky="ew", padx=self.PAD_TIGHT, pady=self.PAD_INNER)

    def _on_device_changed(self, _event=None):
        if self.var_device.get().upper() != "GPU":
            return
        try:
            import torch

            if not torch.cuda.is_available():
                self.var_device.set("CPU")
                messagebox.showwarning("Device Selection", "GPU is not available in the current environment. Falling back to CPU.")
        except Exception:
            self.var_device.set("CPU")
            messagebox.showwarning("Device Selection", "PyTorch CUDA support is unavailable. Falling back to CPU.")

    def _build_current_run(self):
        self.steps_label = ttk.Label(self.current_panel, text="Steps")
        self.episodes_label = ttk.Label(self.current_panel, text="Episodes")
        self.steps_progress = ttk.Progressbar(self.current_panel, orient="horizontal", mode="determinate", maximum=100)
        self.episodes_progress = ttk.Progressbar(self.current_panel, orient="horizontal", mode="determinate", maximum=100)
        self.status_var = tk.StringVar(value="Epsilon: - | LR: - | Best reward: - | Render: idle")
        self.status_label = ttk.Label(self.current_panel, textvariable=self.status_var, style="Muted.TLabel")

        self.steps_label.grid(row=0, column=0, sticky="w", padx=6, pady=(6, 3))
        self.episodes_label.grid(row=1, column=0, sticky="w", padx=6, pady=3)
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=6, pady=(6, 3))
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        self.status_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=6, pady=(3, 6))

    def _build_plot(self):
        self.figure = Figure(figsize=(12, 5), facecolor="#252526")
        self.ax = self.figure.add_subplot(111)
        self._style_plot_axes()
        self.figure.subplots_adjust(left=0.04, right=0.78, top=0.96, bottom=0.12)

        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.plot_panel)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.legend = None
        self.legend_map = {}
        self.legend_entry_targets: List[Any] = []
        self.legend_hover_defaults: Dict[Any, Dict[str, Any]] = {}
        self.legend_hover_artist = None

        self.figure.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_legend_hover)
        self.figure.canvas.mpl_connect("figure_leave_event", self._on_legend_leave)

    def _style_plot_axes(self):
        fg = "#e6e6e6"
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, color=fg, alpha=0.18, linewidth=0.8)
        self.ax.tick_params(colors=fg, which="both")
        self.ax.xaxis.label.set_color(fg)
        self.ax.yaxis.label.set_color(fg)
        self.ax.title.set_color(fg)
        for spine in self.ax.spines.values():
            spine.set_color(fg)
            spine.set_alpha(0.35)

    def _initialize_defaults(self):
        self.trainer_template = InvertedDoublePendulumTrainer()
        self.env_preview = InvertedDoublePendulumEnvironment(
            EnvironmentConfig(
                healthy_reward=float(self.var_healthy_reward.get()),
                reset_noise_scale=float(self.var_reset_noise_scale.get()),
                render_mode=None,
            )
        )

    def _entry_row(self, parent, row, col_group, label, var):
        col = col_group * 2
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=3)
        ttk.Entry(parent, textvariable=var, width=self.PARAM_INPUT_WIDTH).grid(row=row, column=col + 1, sticky="ew", padx=6, pady=3)

    def _combo_row(self, parent, row, col_group, label, var, values):
        col = col_group * 2
        ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=3)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=self.PARAM_INPUT_WIDTH)
        combo.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=3)

    def _bind_mousewheel(self):
        self.param_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

    def _unbind_mousewheel(self):
        self.param_canvas.unbind_all("<MouseWheel>")

    def _on_mousewheel(self, event):
        self.param_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_params_content_configure(self, _event):
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
        bbox = self.param_canvas.bbox("all")
        if bbox:
            needs_scroll = (bbox[3] - bbox[1]) > self.param_canvas.winfo_height() + 2
            if needs_scroll:
                self.param_scrollbar.grid()
            else:
                self.param_scrollbar.grid_remove()

    def _on_params_canvas_configure(self, event):
        self.param_canvas.itemconfigure(self.params_window, width=event.width)

    def _on_canvas_resize(self, event):
        self.current_canvas_size = (max(1, event.width), max(1, event.height))
        self.last_rendered_frame_id = None

    def _toggle_advanced_plot(self):
        if self.var_show_advanced.get():
            self.advanced_row_frame.grid()
        else:
            self.advanced_row_frame.grid_remove()

    def _on_compare_toggle(self):
        if self.var_compare_on.get() and self.var_animation_on.get():
            self.var_animation_on.set(False)
            self._on_animation_toggle()

    def _on_animation_toggle(self):
        if not self.var_animation_on.get():
            self.playback_frames.clear()
            self.playback_index = 0
            self.steps_progress["value"] = 0
            self.status_var.set(self._status_line(render_state="off"))
            for worker in self.current_workers:
                worker.set_runtime_animation(
                    animation_on=False,
                    update_rate=int(self.var_update_rate.get() or 5),
                    rollout_full_capture_steps=int(self.var_rollout_full_capture_steps.get() or 120),
                    low_overhead_animation=self.var_low_overhead.get(),
                )

    def _apply_environment_update(self):
        try:
            healthy_reward = float(self.var_healthy_reward.get())
            reset_noise = float(self.var_reset_noise_scale.get())
            self.env_preview.update(healthy_reward, reset_noise, render_mode=None)
        except Exception as exc:
            messagebox.showerror("Environment Update", f"Invalid environment parameters: {exc}")

    def _add_compare_entry(self):
        key = self.var_compare_param.get().strip()
        values = [v.strip() for v in self.var_compare_values.get().split(",") if v.strip()]
        if not key or not values:
            return
        self.compare_entries[key] = values
        self._refresh_compare_summary()

    def _clear_compare_entries(self):
        self.compare_entries.clear()
        self._refresh_compare_summary()

    def _refresh_compare_summary(self):
        if not self.compare_entries:
            self.compare_summary.configure(text="(none)")
            return
        lines = [f"{k}: [{', '.join(v)}]" for k, v in self.compare_entries.items()]
        self.compare_summary.configure(text="\n".join(lines))

    def _apply_policy_defaults(self):
        policy = self.var_policy.get()
        defaults = self.trainer_template.policy_factory.get_defaults(policy)
        self.var_hidden_layer.set(str(defaults["hidden_layer"]))
        self.var_activation.set(str(defaults["activation"]))
        self.var_lr.set(f"{float(defaults['lr']):.1e}")
        self.var_lr_strategy.set(str(defaults["lr_strategy"]))
        self.var_min_lr.set(f"{float(defaults['min_lr']):.1e}")
        self.var_lr_decay.set(str(defaults["lr_decay"]))
        self.var_replay_size.set(str(defaults["replay_size"]))
        self.var_batch_size.set(str(defaults["batch_size"]))
        self.var_learning_start.set(str(defaults["learning_start"]))
        self.var_learning_frequency.set(str(defaults["learning_frequency"]))
        self.var_target_update.set(str(defaults["target_update"]))

    def _collect_general_params(self) -> Dict[str, Any]:
        return {
            "max_steps": int(self.var_max_steps.get()),
            "episodes": int(self.var_episodes.get()),
            "epsilon_max": float(self.var_epsilon_max.get()),
            "epsilon_decay": float(self.var_epsilon_decay.get()),
            "epsilon_min": float(self.var_epsilon_min.get()),
            "gamma": float(self.var_gamma.get()),
            "moving_average": int(self.var_moving_average.get()),
        }

    def _collect_env_params(self) -> Dict[str, Any]:
        return {
            "healthy_reward": float(self.var_healthy_reward.get()),
            "reset_noise_scale": float(self.var_reset_noise_scale.get()),
        }

    def _collect_specific_params(self) -> Dict[str, Any]:
        return {
            "hidden_layer": int(self.var_hidden_layer.get()),
            "activation": self.var_activation.get(),
            "lr": float(self.var_lr.get()),
            "lr_strategy": self.var_lr_strategy.get(),
            "min_lr": float(self.var_min_lr.get()),
            "lr_decay": float(self.var_lr_decay.get()),
            "replay_size": int(self.var_replay_size.get()),
            "batch_size": int(self.var_batch_size.get()),
            "learning_start": int(self.var_learning_start.get()),
            "learning_frequency": int(self.var_learning_frequency.get()),
            "target_update": int(self.var_target_update.get()),
            "device": self.var_device.get(),
        }

    def _status_line(self, epsilon: Optional[float] = None, lr: Optional[float] = None, best_reward: Optional[float] = None, render_state: str = "idle") -> str:
        eps_text = "-" if epsilon is None else f"{epsilon:.4f}"
        lr_text = "-" if lr is None else f"{lr:.2e}"
        br_text = "-" if best_reward is None else f"{best_reward:.2f}"
        return f"Epsilon: {eps_text} | LR: {lr_text} | Best reward: {br_text} | Render: {render_state}"

    def run_single_episode(self):
        if self.training_active:
            return
        try:
            env = InvertedDoublePendulumEnvironment(
                EnvironmentConfig(
                    healthy_reward=float(self.var_healthy_reward.get()),
                    reset_noise_scale=float(self.var_reset_noise_scale.get()),
                    render_mode="rgb_array",
                )
            )
            trainer = InvertedDoublePendulumTrainer()
            trainer.set_runtime_animation(
                animation_on=self.var_animation_on.get(),
                update_rate=int(self.var_update_rate.get()),
                rollout_full_capture_steps=int(self.var_rollout_full_capture_steps.get()),
                low_overhead_animation=self.var_low_overhead.get(),
            )
            rollout = trainer.run_episode(
                env=env.env,
                max_steps=int(self.var_max_steps.get()),
                epsilon=0.0,
                deterministic=False,
                collect_transitions=False,
                capture_rollout=self.var_animation_on.get(),
            )
            if rollout["frames"]:
                self.playback_frames = rollout["frames"]
                self.playback_index = 0
            self.status_var.set(self._status_line(best_reward=float(rollout["reward"]), render_state="on" if rollout["frames"] else "off"))
            env.close()
        except Exception as exc:
            messagebox.showerror("Run single episode", str(exc))

    def train_and_run(self):
        if self.training_active:
            if not self.paused:
                return
            self._cancel_current_training_for_restart()

        self._flush_event_queue()
        self.current_training_session_id += 1
        session_id = self.current_training_session_id
        self.training_active = True
        self.paused = False
        self.btn_train.configure(style="Accent.TButton")
        self.btn_pause.configure(text="Pause", style="Control.TButton")
        self.episodes_progress["value"] = 0
        self.steps_progress["value"] = 0
        self.current_workers = []

        def _orchestrate():
            try:
                compare_runs = self._build_compare_runs()
                self.render_run_id = None
                if compare_runs and self.var_animation_on.get():
                    selected_policy = self.var_policy.get()
                    for candidate in compare_runs:
                        if candidate.get("policy") == selected_policy:
                            self.render_run_id = candidate["run_id"]
                            break
                    if self.render_run_id is None:
                        self.render_run_id = compare_runs[0]["run_id"]

                max_workers = max(1, min(4, len(compare_runs)))
                self._configure_compare_parallelism(max_workers)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    running = {}
                    pending_runs = list(compare_runs)

                    while pending_runs or running:
                        while pending_runs and len(running) < max_workers:
                            run_cfg = pending_runs.pop(0)
                            trainer = InvertedDoublePendulumTrainer()
                            should_render_for_run = bool(self.var_animation_on.get() and run_cfg.get("run_id") == self.render_run_id)
                            trainer.set_runtime_animation(
                                animation_on=should_render_for_run,
                                update_rate=int(self.var_update_rate.get()),
                                rollout_full_capture_steps=int(self.var_rollout_full_capture_steps.get()),
                                low_overhead_animation=self.var_low_overhead.get(),
                            )
                            self.current_workers.append(trainer)
                            fut = executor.submit(self._run_training_job, trainer, run_cfg, session_id)
                            running[fut] = (trainer, run_cfg)

                        done, _ = wait(running.keys(), timeout=0.2, return_when=FIRST_COMPLETED)
                        for fut in done:
                            running.pop(fut, None)
            finally:
                self.event_queue.put({"type": "all_done", "session_id": session_id})

        self.main_worker_thread = threading.Thread(target=_orchestrate, daemon=True)
        self.main_worker_thread.start()

    def _configure_compare_parallelism(self, max_workers: int):
        if max_workers <= 1:
            return
        try:
            import torch

            cpu_count = os.cpu_count() or 1
            per_worker_threads = max(1, cpu_count // max_workers)
            torch.set_num_threads(per_worker_threads)
        except Exception:
            pass

    def _build_compare_runs(self) -> List[Dict[str, Any]]:
        base = {
            "policy": self.var_policy.get(),
            "env": self._collect_env_params(),
            "general": self._collect_general_params(),
            "specific": self._collect_specific_params(),
        }

        if not self.var_compare_on.get() or not self.compare_entries:
            base["run_id"] = timestamp_run_id("run")
            return [base]

        keys = list(self.compare_entries.keys())
        values = [self.compare_entries[k] for k in keys]
        runs = []

        for combo_idx, combo in enumerate(itertools.product(*values), start=1):
            run_cfg = {
                "policy": base["policy"],
                "env": dict(base["env"]),
                "general": dict(base["general"]),
                "specific": dict(base["specific"]),
            }
            for key, value in zip(keys, combo):
                low = key.lower()
                if low == "policy":
                    run_cfg["policy"] = value
                    policy_defaults = self.trainer_template.policy_factory.get_defaults(value)
                    run_cfg["specific"].update(policy_defaults)
                elif low in ("max steps", "max_steps"):
                    run_cfg["general"]["max_steps"] = int(value)
                elif low == "episodes":
                    run_cfg["general"]["episodes"] = int(value)
                elif low == "epsilon max":
                    run_cfg["general"]["epsilon_max"] = float(value)
                elif low == "epsilon decay":
                    run_cfg["general"]["epsilon_decay"] = float(value)
                elif low == "epsilon min":
                    run_cfg["general"]["epsilon_min"] = float(value)
                elif low == "gamma":
                    run_cfg["general"]["gamma"] = float(value)
                elif low == "hidden layer":
                    run_cfg["specific"]["hidden_layer"] = int(value)
                elif low == "activation":
                    run_cfg["specific"]["activation"] = str(value)
                elif low == "lr":
                    run_cfg["specific"]["lr"] = float(value)
                elif low == "lr strategy":
                    run_cfg["specific"]["lr_strategy"] = str(value)
                elif low == "min lr":
                    run_cfg["specific"]["min_lr"] = float(value)
                elif low == "lr decay":
                    run_cfg["specific"]["lr_decay"] = float(value)
                elif low == "replay size":
                    run_cfg["specific"]["replay_size"] = int(value)
                elif low == "batch size":
                    run_cfg["specific"]["batch_size"] = int(value)
                elif low == "learning start":
                    run_cfg["specific"]["learning_start"] = int(value)
                elif low == "learning frequency":
                    run_cfg["specific"]["learning_frequency"] = int(value)
                elif low == "target update":
                    run_cfg["specific"]["target_update"] = int(value)

            run_cfg["run_id"] = f"{timestamp_run_id('cmp')}_{combo_idx}"
            runs.append(run_cfg)

        return runs

    def _run_training_job(self, trainer: InvertedDoublePendulumTrainer, run_cfg: Dict[str, Any], session_id: int):
        run_id = run_cfg["run_id"]
        policy = run_cfg["policy"]
        env_params = run_cfg["env"]
        general = run_cfg["general"]
        specific = run_cfg["specific"]

        label = build_run_label(policy, env_params, general, specific)
        self.run_artifacts[run_id] = {
            "label": label,
            "policy": policy,
            "trainer": trainer,
            "env": env_params,
            "general": general,
            "specific": specific,
        }

        def _session_event_sink(event: Dict[str, Any]):
            event_with_session = dict(event)
            event_with_session["session_id"] = session_id
            self.event_queue.put(event_with_session)

        trainer.train(
            run_id=run_id,
            policy_name=policy,
            env_params=env_params,
            general_params=general,
            specific_params=specific,
            event_sink=_session_event_sink,
        )

    def _cancel_current_training_for_restart(self):
        for worker in self.current_workers:
            worker.resume()
            worker.stop()
        self.training_active = False
        self.paused = False
        self.btn_train.configure(style="Control.TButton")
        self.btn_pause.configure(text="Pause", style="Control.TButton")

    def toggle_pause(self):
        if not self.training_active:
            return
        self.paused = not self.paused
        if self.paused:
            for worker in self.current_workers:
                worker.pause()
            self.btn_train.configure(style="Control.TButton")
            self.btn_pause.configure(text="Run", style="Amber.TButton")
        else:
            for worker in self.current_workers:
                worker.resume()
            self.btn_train.configure(style="Accent.TButton")
            self.btn_pause.configure(text="Pause", style="Control.TButton")

    def reset_all(self):
        for worker in self.current_workers:
            worker.resume()
            worker.stop()
        self.training_active = False
        self.paused = False
        self.btn_train.configure(style="Control.TButton")
        self.btn_pause.configure(text="Pause", style="Control.TButton")
        self.playback_frames.clear()
        self.playback_index = 0
        self.steps_progress["value"] = 0
        self.episodes_progress["value"] = 0
        self.status_var.set(self._status_line(render_state="idle"))

    def clear_plot(self):
        self.plot_series.clear()
        self.finalized_runs.clear()
        self.ax.clear()
        self._style_plot_axes()
        self.canvas_plot.draw_idle()

    def save_samplings_csv(self):
        paths = []
        for run_id in self.finalized_runs:
            trainer = self.run_artifacts.get(run_id, {}).get("trainer")
            if trainer:
                path = trainer.export_transitions_csv(filename_prefix=f"samplings_{run_id}")
                if path:
                    paths.append(path)
        if not paths:
            messagebox.showinfo("CSV export", "No sampled transitions available for export.")
            return
        messagebox.showinfo("CSV export", f"Saved {len(paths)} CSV file(s) to results_csv/.")

    def save_plot_png(self):
        os.makedirs("plots", exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        policy = self.var_policy.get()
        file_name = (
            f"InvDoubPend_{policy}_hr{self.var_healthy_reward.get()}_rn{self.var_reset_noise_scale.get()}_"
            f"ep{self.var_episodes.get()}_{now}.png"
        )
        path = os.path.join("plots", file_name)
        self.figure.savefig(path, dpi=150)
        messagebox.showinfo("Save Plot", f"Saved: {path}")

    def _start_event_pump(self):
        def _pump():
            self._consume_events()
            self.root.after(100, _pump)

        _pump()

    def _consume_events(self):
        while True:
            try:
                event = self.event_queue.get_nowait()
            except queue.Empty:
                break

            kind = event.get("type")
            event_session_id = event.get("session_id")
            if event_session_id is not None and event_session_id != self.current_training_session_id:
                continue
            if kind == "episode":
                self._handle_episode_event(event)
            elif kind == "training_done":
                self._handle_done_event(event)
            elif kind == "error":
                self._handle_error_event(event)
            elif kind == "all_done":
                self.training_active = False
                self.btn_train.configure(style="Control.TButton")
                self.btn_pause.configure(text="Pause", style="Control.TButton")
                self.paused = False

    def _handle_episode_event(self, event: Dict[str, Any]):
        run_id = event["run_id"]
        episode = int(event["episode"])
        episodes = int(event["episodes"])
        reward = float(event["reward"])
        ma = float(event["moving_average"])
        eval_points = list(event.get("eval_points", []))

        series = self.plot_series.setdefault(run_id, {"reward": [], "ma": [], "eval": []})
        series["reward"].append(reward)
        series["ma"].append(ma)
        series["eval"] = eval_points

        if run_id == self.render_run_id:
            self.episodes_progress["value"] = (episode / max(1, episodes)) * 100.0
            self.status_var.set(
                self._status_line(
                    epsilon=float(event.get("epsilon", 0.0)),
                    lr=float(event.get("lr", 0.0)),
                    best_reward=float(event.get("best_reward", reward)),
                    render_state=str(event.get("render_state", "idle")),
                )
            )
            if self.var_animation_on.get() and event.get("frames"):
                self.playback_frames = event["frames"]
                self.playback_index = 0

        self._redraw_plot()

    def _handle_done_event(self, event: Dict[str, Any]):
        run_id = event.get("run_id")
        if run_id and run_id not in self.finalized_runs:
            self.finalized_runs.append(run_id)

    def _handle_error_event(self, event: Dict[str, Any]):
        messagebox.showerror("Training Error", event.get("message", "Unknown error"))

    def _flush_event_queue(self):
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break

    def _start_render_tick(self):
        def _tick():
            self._play_next_frame()
            self.root.after(max(10, int(1000 / max(1, int(self.var_animation_fps.get() or 10)))), _tick)

        _tick()

    def _play_next_frame(self):
        if not self.var_animation_on.get():
            return
        if not self.playback_frames:
            return

        frame_count = len(self.playback_frames)
        if self.playback_index >= frame_count:
            self.playback_frames.clear()
            self.playback_index = 0
            self.steps_progress["value"] = 0
            return

        frame = self.playback_frames[self.playback_index]
        self.playback_index += 1
        self.steps_progress["value"] = (self.playback_index / max(1, frame_count)) * 100.0
        self._render_frame(frame)

    def _render_preview_frame(self):
        try:
            obs, _ = self.env_preview.env.reset()
            _ = obs
            frame = self.env_preview.env.render()
            if frame is not None:
                self._render_frame(frame)
        except Exception:
            pass

    def _render_frame(self, frame):
        if Image is None or ImageTk is None:
            return

        frame_id = (id(frame), self.current_canvas_size)
        if frame_id == self.last_rendered_frame_id:
            return
        self.last_rendered_frame_id = frame_id

        canvas_w, canvas_h = self.current_canvas_size
        img = Image.fromarray(frame)
        src_w, src_h = img.size
        scale = min(canvas_w / max(1, src_w), canvas_h / max(1, src_h))
        dst_w = max(1, int(src_w * scale))
        dst_h = max(1, int(src_h * scale))
        resized = img.resize((dst_w, dst_h), Image.Resampling.LANCZOS)
        self.canvas_photo = ImageTk.PhotoImage(resized)

        x = (canvas_w - dst_w) // 2
        y = (canvas_h - dst_h) // 2

        if not hasattr(self, "canvas_image_item"):
            self.canvas_image_item = self.render_canvas.create_image(x, y, image=self.canvas_photo, anchor="nw")
        else:
            self.render_canvas.coords(self.canvas_image_item, x, y)
            self.render_canvas.itemconfigure(self.canvas_image_item, image=self.canvas_photo)

    def _build_reward_legend_label(self, meta: Dict[str, Any]) -> str:
        policy = str(meta.get("policy", "?"))
        general = meta.get("general", {})
        specific = meta.get("specific", {})
        env = meta.get("env", {})

        line_1 = f"{policy} | steps={general.get('max_steps', '?')} | gamma={general.get('gamma', '?')}"
        line_2 = (
            f"epsilon={general.get('epsilon_max', '?')} | "
            f"epsilon_decay={general.get('epsilon_decay', '?')} | "
            f"epsilon_min={general.get('epsilon_min', '?')}"
        )
        line_3 = (
            f"LR={specific.get('lr', '?')} | "
            f"LR strategy={specific.get('lr_strategy', '?')} | "
            f"LR decay={specific.get('lr_decay', '?')}"
        )

        env_items = [f"{k}={v}" for k, v in env.items()]
        line_4 = " | ".join(env_items) if env_items else "env=-"
        return "\n".join([line_1, line_2, line_3, line_4])

    def _redraw_plot(self):
        self.ax.clear()
        self._style_plot_axes()

        self.legend_map = {}
        legend_targets: List[Any] = []

        for run_id, series in self.plot_series.items():
            meta = self.run_artifacts.get(run_id, {})
            label = self._build_reward_legend_label(meta)
            x = list(range(1, len(series["reward"]) + 1))
            if not x:
                continue
            reward_line_width = 1.6
            reward_alpha = 0.60
            (reward_line,) = self.ax.plot(
                x,
                series["reward"],
                linewidth=reward_line_width,
                alpha=reward_alpha,
                linestyle="-",
                label=label,
            )

            run_color = reward_line.get_color()
            doubled_line_width = reward_line_width * 2.0
            (ma_line,) = self.ax.plot(
                x,
                series["ma"],
                color=run_color,
                linewidth=doubled_line_width,
                linestyle="--",
                alpha=1.0,
                label="moving average",
            )
            legend_targets.append(reward_line)
            legend_targets.append(ma_line)
            eval_data = series.get("eval", [])
            if eval_data:
                ex = [p[0] for p in eval_data]
                ey = [p[1] for p in eval_data]
                (eval_line,) = self.ax.plot(
                    ex,
                    ey,
                    color=run_color,
                    linestyle=":",
                    marker="o",
                    markersize=3,
                    linewidth=doubled_line_width,
                    alpha=1.0,
                    label="evaluation rollout",
                )
                legend_targets.append(eval_line)
                self.legend_map[eval_line] = eval_line
            self.legend_map[reward_line] = reward_line
            self.legend_map[ma_line] = ma_line

        if self.plot_series:
            self.legend = self.ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), framealpha=0.4)
            if self.legend:
                self.legend.get_frame().set_facecolor("#252526")
                self.legend.get_frame().set_edgecolor("#e6e6e6")
                self.legend.get_frame().set_alpha(0.45)
                handles = self.legend.legend_handles
                texts = self.legend.get_texts()
                self.legend_entry_targets = legend_targets[: len(handles)]
                self.legend_hover_defaults = {}
                self.legend_hover_artist = None

                for legline in handles:
                    legline.set_picker(True)
                    self.legend_hover_defaults[legline] = {
                        "linewidth": legline.get_linewidth(),
                        "alpha": legline.get_alpha(),
                    }
                for text in texts:
                    text.set_color("#e6e6e6")
                    text.set_picker(True)
                    self.legend_hover_defaults[text] = {
                        "color": text.get_color(),
                        "weight": text.get_fontweight(),
                    }
        else:
            self.legend = None
            self.legend_entry_targets = []
            self.legend_hover_defaults = {}
            self.legend_hover_artist = None
            self.canvas_plot.get_tk_widget().configure(cursor="")

        self.canvas_plot.draw_idle()

    def _on_legend_pick(self, event):
        artist = event.artist
        if self.legend is None:
            return

        handles = self.legend.legend_handles
        texts = self.legend.get_texts()

        target = None
        if artist in handles:
            idx = handles.index(artist)
            if idx < len(self.legend_entry_targets):
                target = self.legend_entry_targets[idx]
        elif artist in texts:
            idx = texts.index(artist)
            if idx < len(self.legend_entry_targets):
                target = self.legend_entry_targets[idx]

        if target is not None:
            target.set_visible(not target.get_visible())
            self.canvas_plot.draw_idle()

    def _on_legend_hover(self, event):
        if self.legend is None:
            return

        handles = self.legend.legend_handles
        texts = self.legend.get_texts()
        hovered = None

        for art in [*handles, *texts]:
            try:
                contains, _info = art.contains(event)
            except Exception:
                contains = False
            if contains:
                hovered = art
                break

        self._set_legend_hover(hovered)

    def _on_legend_leave(self, _event):
        self._set_legend_hover(None)

    def _set_legend_hover(self, artist):
        if artist is self.legend_hover_artist:
            return

        for art, defaults in self.legend_hover_defaults.items():
            if hasattr(art, "set_linewidth") and "linewidth" in defaults:
                art.set_linewidth(defaults["linewidth"])
            if hasattr(art, "set_alpha") and "alpha" in defaults:
                art.set_alpha(defaults["alpha"])
            if hasattr(art, "set_color") and "color" in defaults:
                art.set_color(defaults["color"])
            if hasattr(art, "set_fontweight") and "weight" in defaults:
                art.set_fontweight(defaults["weight"])

        if artist is not None:
            if hasattr(artist, "set_linewidth"):
                linewidth = getattr(artist, "get_linewidth", lambda: 1.0)()
                artist.set_linewidth(max(2.0, linewidth * 1.4))
            if hasattr(artist, "set_alpha"):
                artist.set_alpha(1.0)
            if hasattr(artist, "set_color"):
                artist.set_color("#ffffff")
            if hasattr(artist, "set_fontweight"):
                artist.set_fontweight("bold")
            self.canvas_plot.get_tk_widget().configure(cursor="hand2")
        else:
            self.canvas_plot.get_tk_widget().configure(cursor="")

        self.legend_hover_artist = artist
        self.canvas_plot.draw_idle()


def launch_gui():
    root = tk.Tk()
    app = InvDoubPendGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: _safe_close(root, app))
    root.mainloop()


def _safe_close(root: tk.Tk, app: InvDoubPendGUI):
    try:
        for worker in app.current_workers:
            worker.resume()
            worker.stop()
        app.env_preview.close()
    finally:
        root.destroy()
