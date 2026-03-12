from __future__ import annotations

import queue
import threading
import time
import os
import multiprocessing as mp
import logging
import warnings
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Ant_logic import (
    EnvironmentConfig,
    POLICY_DEFAULTS,
    TrainerConfig,
    AntTrainer,
    expand_compare_runs,
)

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional for rendering quality
    Image = None  # type: ignore[assignment]
    ImageTk = None  # type: ignore[assignment]


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self.tip_window: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            return
        x = self.widget.winfo_rootx() + 8
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        lbl = tk.Label(
            tw,
            text=self.text,
            bg="#2d2d30",
            fg="#e6e6e6",
            bd=1,
            relief="solid",
            padx=6,
            pady=2,
            font=("Segoe UI", 9),
        )
        lbl.pack()
        self.tip_window = tw

    def _hide(self, _event: tk.Event) -> None:
        if self.tip_window is not None:
            self.tip_window.destroy()
            self.tip_window = None


class _ProcessTrainerProxy:
    def __init__(self, stop_event: Any, pause_event: Any):
        self._stop_event = stop_event
        self._pause_event = pause_event

    def pause(self) -> None:
        self._pause_event.clear()

    def resume(self) -> None:
        self._pause_event.set()

    def cancel(self) -> None:
        self._pause_event.set()
        self._stop_event.set()


def _ant_worker_main(
    event_queue: Any,
    env_cfg_data: Dict[str, Any],
    tr_cfg_data: Dict[str, Any],
    session_id: str,
    run_id: str,
    stop_event: Any,
    pause_event: Any,
    output_csv_dir: str,
) -> None:
    logging.getLogger("evotorch").setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore",
        message="To copy construct from a tensor",
        category=UserWarning,
    )

    env_cfg = EnvironmentConfig(**env_cfg_data)
    tr_cfg = TrainerConfig(**tr_cfg_data)

    def _sink(event: Dict[str, Any]) -> None:
        event_queue.put(event)

    trainer = AntTrainer(
        env_cfg,
        tr_cfg,
        event_sink=_sink,
        session_id=session_id,
        run_id=run_id,
        stop_event=stop_event,
        pause_event=pause_event,
    )
    trainer.train(collect_transitions=True)
    csv_path = trainer.export_sampled_transitions_csv(Path(output_csv_dir))
    if csv_path is not None:
        event_queue.put(
            {
                "type": "csv_ready",
                "session_id": session_id,
                "run_id": run_id,
                "path": str(csv_path),
            }
        )


class AntGUI:
    LABEL_WIDTH_CHARS = 18

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Ant RL/Evolution Trainer")
        self.root.geometry("1380x900")

        self.project_dir = Path(__file__).resolve().parent
        self.output_csv_dir = self.project_dir / "results_csv"
        self.output_plot_dir = self.project_dir / "plots"

        self._event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._mp_ctx = mp.get_context("spawn")
        self._process_event_queue = self._mp_ctx.Queue()
        self._active_processes: Dict[str, Any] = {}
        self._pending_runs: set[str] = set()
        self._session_id = self._new_session_id()
        self._active_trainers: Dict[str, Any] = {}
        self._trainers_lock = threading.Lock()
        self._training_active = False
        self._paused = False

        self._frame_playback_active = False
        self._frame_pending: Optional[List[np.ndarray]] = None
        self._frame_photo = None

        self._run_history: Dict[str, Dict[str, Any]] = {}
        self._line_map: Dict[str, Dict[str, Any]] = {}
        self._csv_exports: Dict[str, str] = {}
        self._run_label_snapshots: Dict[str, str] = {}
        self._render_run_id: Optional[str] = None
        self._legend = None
        self._legend_anchor_y = 1.0
        self._legend_scroll_enabled = False
        self._legend_scroll_limit = 0.0
        self._mpl_events_bound = False

        self.policy_param_vars: Dict[str, Dict[str, tk.Variable]] = {}
        self.policy_specific_keys: Dict[str, List[str]] = {
            "SAC": ["buffer_size", "tau", "train_freq", "gradient_steps", "learning_starts"],
            "TQC": [
                "buffer_size",
                "tau",
                "train_freq",
                "gradient_steps",
                "learning_starts",
                "top_quantiles_to_drop_per_net",
            ],
            "CMA-ES": [
                "sigma",
                "number_of_agents",
                "iterations_per_episode",
                "cmaes_eval_horizon",
                "cmaes_rollouts",
                "cmaes_hidden_units",
                "cmaes_init_spread",
                "cmaes_action_noise",
                "cmaes_elite_k",
                "cmaes_center_momentum",
                "cmaes_stagnation_patience",
                "cmaes_restart_spread",
            ],
        }

        self._configure_styles()
        self._build_layout()
        self._init_vars()
        self._populate_specific_panel()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(16, self._pump_events)

    def _new_session_id(self) -> str:
        return f"session-{int(time.time() * 1000)}"

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            style.theme_use("vista")

        self.root.configure(bg="#1e1e1e")
        style.configure("TFrame", background="#1e1e1e")
        style.configure("Card.TLabelframe", background="#252526", foreground="#e6e6e6")
        style.configure("Card.TLabelframe.Label", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10, "bold"))
        style.configure("TLabel", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10))
        style.configure("TEntry", fieldbackground="#2d2d30", foreground="#e6e6e6")
        style.configure("TCombobox", fieldbackground="#2d2d30", foreground="#e6e6e6")
        style.configure("Dark.TCombobox", fieldbackground="#2d2d30", background="#2d2d30", foreground="#e6e6e6")
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#2d2d30")],
            foreground=[("readonly", "#e6e6e6")],
            selectbackground=[("readonly", "#0e639c")],
            selectforeground=[("readonly", "#ffffff")],
        )
        style.configure("Neutral.TButton", background="#3a3d41", foreground="#e6e6e6", font=("Segoe UI", 10, "bold"))
        style.map(
            "Neutral.TButton",
            background=[("active", "#4a4f55"), ("pressed", "#2f3338")],
        )
        style.configure("Train.TButton", background="#0e639c", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Train.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.configure("Pause.TButton", background="#a66a00", foreground="white", font=("Segoe UI", 10, "bold"))
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])
        style.configure("TProgressbar", troughcolor="#343434", background="#0e639c", bordercolor="#343434", lightcolor="#0e639c", darkcolor="#0e639c")

        # Enforce dark caret/listbox visuals for better readability in dark mode.
        self.root.option_add("*Entry.insertBackground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox*Background", "#2d2d30")
        self.root.option_add("*TCombobox*Listbox*Foreground", "#e6e6e6")
        self.root.option_add("*TCombobox*Listbox*selectBackground", "#0e639c")
        self.root.option_add("*TCombobox*Listbox*selectForeground", "#ffffff")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=2)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=3)
        self.root.rowconfigure(1, weight=0)
        self.root.rowconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=2)

        self.env_group = ttk.LabelFrame(self.root, text="Environment", style="Card.TLabelframe")
        self.env_group.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.params_group = ttk.LabelFrame(self.root, text="Parameters", style="Card.TLabelframe")
        self.params_group.grid(row=0, column=1, sticky="nsew", padx=(0, 10), pady=10)

        self.controls_group = ttk.Frame(self.root)
        self.controls_group.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self.current_group = ttk.LabelFrame(self.root, text="Current Run", style="Card.TLabelframe")
        self.current_group.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        self.plot_group = ttk.LabelFrame(self.root, text="Live Plot", style="Card.TLabelframe")
        self.plot_group.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 10))

        self._build_environment_panel()
        self._build_parameters_panel()
        self._build_controls_row()
        self._build_current_panel()
        self._build_plot_panel()

    def _build_environment_panel(self) -> None:
        self.env_group.rowconfigure(0, weight=1)
        self.env_group.columnconfigure(0, weight=1)
        self.render_canvas = tk.Canvas(self.env_group, bg="#111111", highlightthickness=0)
        self.render_canvas.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.render_canvas.bind("<Configure>", lambda _e: self._render_placeholder())
        self._render_placeholder()

    def _build_parameters_panel(self) -> None:
        self.params_group.rowconfigure(0, weight=1)
        self.params_group.columnconfigure(0, weight=1)

        outer = ttk.Frame(self.params_group)
        outer.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)

        self.param_canvas = tk.Canvas(outer, bg="#252526", highlightthickness=0)
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        self.param_scroll = ttk.Scrollbar(outer, orient="vertical", command=self.param_canvas.yview)
        self.param_scroll.grid(row=0, column=1, sticky="ns")
        self.param_canvas.configure(yscrollcommand=self.param_scroll.set)

        self.param_inner = ttk.Frame(self.param_canvas)
        self.param_inner.columnconfigure(0, weight=1)
        self.param_window = self.param_canvas.create_window((0, 0), window=self.param_inner, anchor="nw")
        self.param_inner.bind("<Configure>", lambda _e: self._update_param_scroll())
        self.param_canvas.bind("<Configure>", self._on_param_canvas_configure)
        self.param_inner.bind("<Enter>", self._bind_param_wheel)
        self.param_inner.bind("<Leave>", self._unbind_param_wheel)

        self.env_params_group = ttk.LabelFrame(self.param_inner, text="Environment", style="Card.TLabelframe")
        self.compare_params_group = ttk.LabelFrame(self.param_inner, text="Compare", style="Card.TLabelframe")
        self.general_params_group = ttk.LabelFrame(self.param_inner, text="General", style="Card.TLabelframe")
        self.specific_params_group = ttk.LabelFrame(self.param_inner, text="Specific", style="Card.TLabelframe")
        self.live_plot_group = ttk.LabelFrame(self.param_inner, text="Live Plot", style="Card.TLabelframe")

        groups = [
            self.env_params_group,
            self.compare_params_group,
            self.general_params_group,
            self.specific_params_group,
            self.live_plot_group,
        ]
        for idx, group in enumerate(groups):
            group.grid(row=idx, column=0, sticky="nsew", padx=4, pady=4)
            group.columnconfigure(1, weight=1)

    def _build_controls_row(self) -> None:
        for i in range(8):
            self.controls_group.columnconfigure(i, weight=1)

        self.btn_run_episode = ttk.Button(self.controls_group, text="Run single episode", style="Neutral.TButton", command=self.run_single_episode)
        self.btn_train = ttk.Button(self.controls_group, text="Train and Run", style="Neutral.TButton", command=self.train_and_run)
        self.btn_pause = ttk.Button(self.controls_group, text="Pause", style="Neutral.TButton", command=self.toggle_pause)
        self.btn_reset = ttk.Button(self.controls_group, text="Reset All", style="Neutral.TButton", command=self.reset_all)
        self.btn_clear_plot = ttk.Button(self.controls_group, text="Clear Plot", style="Neutral.TButton", command=self.clear_plot)
        self.btn_csv = ttk.Button(self.controls_group, text="Save samplings CSV", style="Neutral.TButton", command=self.save_csv)
        self.btn_png = ttk.Button(self.controls_group, text="Save Plot PNG", style="Neutral.TButton", command=self.save_plot_png)

        self.device_var = tk.StringVar(value="CPU")
        self.device_combo = ttk.Combobox(self.controls_group, textvariable=self.device_var, values=["CPU", "GPU"], state="readonly")
        self._style_combobox(self.device_combo)

        widgets = [
            self.btn_run_episode,
            self.btn_train,
            self.btn_pause,
            self.btn_reset,
            self.btn_clear_plot,
            self.btn_csv,
            self.btn_png,
            self.device_combo,
        ]
        for idx, w in enumerate(widgets):
            w.grid(row=0, column=idx, sticky="ew", padx=4, pady=4)

    def _build_current_panel(self) -> None:
        self.current_group.columnconfigure(1, weight=1)

        ttk.Label(self.current_group, text="Steps").grid(row=0, column=0, sticky="w", padx=8, pady=(6, 2))
        self.steps_progress = ttk.Progressbar(self.current_group, mode="determinate", maximum=100)
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=8, pady=(6, 2))

        ttk.Label(self.current_group, text="Episodes").grid(row=1, column=0, sticky="w", padx=8, pady=(2, 2))
        self.episodes_progress = ttk.Progressbar(self.current_group, mode="determinate", maximum=100)
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=8, pady=(2, 2))

        self.status_var = tk.StringVar(value="Epsilon: 0.0000 | LR: 0.0000e+00 | Best reward: 0.00 | Render: idle")
        ttk.Label(self.current_group, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(2, 8))

    def _build_plot_panel(self) -> None:
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self._apply_plot_theme()
        self.figure.subplots_adjust(left=0.06, right=0.78, bottom=0.12, top=0.96)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_group)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._bind_mpl_events_once()

    def _init_vars(self) -> None:
        # Environment group
        self.animation_on_var = tk.BooleanVar(value=True)
        self.animation_fps_var = tk.StringVar(value="30")
        self.update_rate_var = tk.StringVar(value="1")
        self.frame_stride_var = tk.StringVar(value="2")
        self.forward_reward_weight_var = tk.StringVar(value="1.0")
        self.ctrl_cost_weight_var = tk.StringVar(value="0.5")
        self.contact_cost_weight_var = tk.StringVar(value="5e-4")

        # Compare group
        self.compare_on_var = tk.BooleanVar(value=False)
        self.compare_param_var = tk.StringVar(value="Policy")
        self.compare_values_var = tk.StringVar(value="")
        self.compare_preview_var = tk.StringVar(value="")
        self.compare_entries: Dict[str, List[str]] = {}

        # General group
        self.max_steps_var = tk.StringVar(value="1000")
        self.episodes_var = tk.StringVar(value="3000")

        # Specific group
        self.policy_var = tk.StringVar(value="SAC")

        # Live plot group
        self.ma_values_var = tk.StringVar(value="20")
        self.eval_rollout_var = tk.BooleanVar(value=False)
        self.strict_update_var = tk.BooleanVar(value=False)

        self._build_env_group_fields()
        self._build_compare_group_fields()
        self._build_general_group_fields()
        self._build_specific_group_fields()
        self._build_live_plot_group_fields()

    def _build_env_group_fields(self) -> None:
        self.env_params_group.columnconfigure(0, weight=0, minsize=92)
        self.env_params_group.columnconfigure(1, weight=1)
        self.env_params_group.columnconfigure(2, weight=0, minsize=92)
        self.env_params_group.columnconfigure(3, weight=1)

        ttk.Checkbutton(self.env_params_group, text="Animation on", variable=self.animation_on_var).grid(row=0, column=0, columnspan=2, sticky="w", padx=6, pady=3)
        self._add_labeled_entry(self.env_params_group, row=1, label="Animation FPS", var=self.animation_fps_var, pair=0)
        self._add_labeled_entry(self.env_params_group, row=1, label="Update rate (episodes)", var=self.update_rate_var, pair=1)
        self._add_labeled_entry(self.env_params_group, row=2, label="Frame stride", var=self.frame_stride_var, pair=0)

        update_row = 3
        ttk.Button(self.env_params_group, text="Update", style="Neutral.TButton", command=self._render_placeholder).grid(
            row=update_row, column=0, columnspan=4, sticky="ew", padx=6, pady=6
        )
        self._add_labeled_entry(self.env_params_group, row=4, label="forward_reward_weight", var=self.forward_reward_weight_var, pair=0)
        self._add_labeled_entry(self.env_params_group, row=4, label="ctrl_cost_weight", var=self.ctrl_cost_weight_var, pair=1)
        self._add_labeled_entry(self.env_params_group, row=5, label="contact_cost_weight", var=self.contact_cost_weight_var, pair=0)

    def _build_compare_group_fields(self) -> None:
        top = ttk.Frame(self.compare_params_group)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(4, 2))
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)

        ttk.Checkbutton(top, text="Compare on", variable=self.compare_on_var, command=self._on_compare_toggle).grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Clear", style="Neutral.TButton", command=self._clear_compare_entries).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Add", style="Neutral.TButton", command=self._add_compare_entry).grid(row=0, column=2, padx=4)

        self.compare_param_combo = ttk.Combobox(
            self.compare_params_group,
            textvariable=self.compare_param_var,
            values=self._compare_param_choices(),
            state="readonly",
        )
        self._style_combobox(self.compare_param_combo)
        self.compare_param_combo.grid(row=1, column=0, sticky="ew", padx=6, pady=3)

        self.compare_values_entry = ttk.Entry(self.compare_params_group, textvariable=self.compare_values_var)
        self.compare_values_entry.grid(row=1, column=1, sticky="ew", padx=6, pady=3)
        self.compare_values_entry.bind("<Return>", lambda _e: self._add_compare_entry())
        self.compare_values_entry.bind("<Tab>", self._compare_tab_complete)
        self.compare_values_entry.bind("<KeyRelease>", lambda _e: self._update_compare_preview())

        ttk.Label(self.compare_params_group, textvariable=self.compare_preview_var).grid(row=2, column=1, sticky="w", padx=6, pady=(0, 4))
        self.compare_summary_label = ttk.Label(self.compare_params_group, text="")
        self.compare_summary_label.grid(row=3, column=0, columnspan=2, sticky="w", padx=6, pady=(0, 6))

    def _build_general_group_fields(self) -> None:
        self.general_params_group.columnconfigure(0, weight=0, minsize=92)
        self.general_params_group.columnconfigure(1, weight=1)
        self.general_params_group.columnconfigure(2, weight=0, minsize=92)
        self.general_params_group.columnconfigure(3, weight=1)
        self._add_labeled_entry(self.general_params_group, row=0, label="Max steps", var=self.max_steps_var, pair=0)
        self._add_labeled_entry(self.general_params_group, row=0, label="Episodes", var=self.episodes_var, pair=1)

    def _build_specific_group_fields(self) -> None:
        self.specific_params_group.columnconfigure(0, weight=0, minsize=92)
        self.specific_params_group.columnconfigure(1, weight=1)
        self.specific_params_group.columnconfigure(2, weight=0, minsize=92)
        self.specific_params_group.columnconfigure(3, weight=1)

        ttk.Label(self.specific_params_group, text="Policy", width=self.LABEL_WIDTH_CHARS, anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.policy_combo = ttk.Combobox(self.specific_params_group, textvariable=self.policy_var, values=["SAC", "TQC", "CMA-ES"], state="readonly")
        self._style_combobox(self.policy_combo)
        self.policy_combo.grid(row=0, column=1, columnspan=3, sticky="ew", padx=6, pady=3)
        self.policy_combo.bind("<<ComboboxSelected>>", lambda _e: self._populate_specific_panel())

        self.specific_shared_frame = ttk.Frame(self.specific_params_group)
        self.specific_shared_frame.grid(row=1, column=0, columnspan=4, sticky="ew", padx=6, pady=3)
        self.specific_dynamic_sep = ttk.Separator(self.specific_params_group, orient="horizontal")
        self.specific_dynamic_sep.grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=3)
        self.specific_dynamic_frame = ttk.Frame(self.specific_params_group)
        self.specific_dynamic_frame.grid(row=3, column=0, columnspan=4, sticky="ew", padx=6, pady=3)

        for frame in (self.specific_shared_frame, self.specific_dynamic_frame):
            frame.columnconfigure(0, weight=0, minsize=92)
            frame.columnconfigure(1, weight=1)
            frame.columnconfigure(2, weight=0, minsize=92)
            frame.columnconfigure(3, weight=1)

    def _build_live_plot_group_fields(self) -> None:
        self.live_plot_group.columnconfigure(0, weight=0, minsize=92)
        self.live_plot_group.columnconfigure(1, weight=1)
        self.live_plot_group.columnconfigure(2, weight=0, minsize=92)
        self.live_plot_group.columnconfigure(3, weight=1)

        self._add_labeled_entry(self.live_plot_group, row=0, label="Moving average values", var=self.ma_values_var, pair=0)
        ttk.Checkbutton(self.live_plot_group, text="Evaluation rollout on", variable=self.eval_rollout_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=3)
        ttk.Checkbutton(self.live_plot_group, text="Strict visual update", variable=self.strict_update_var).grid(row=1, column=0, columnspan=2, sticky="w", padx=6, pady=3)

    def _add_labeled_entry(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable, pair: int = 0) -> ttk.Entry:
        base_col = 0 if pair <= 0 else 2
        ttk.Label(parent, text=label, width=self.LABEL_WIDTH_CHARS, anchor="w").grid(row=row, column=base_col, sticky="w", padx=6, pady=3)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=base_col + 1, sticky="ew", padx=6, pady=3)
        return entry

    def _bind_mpl_events_once(self) -> None:
        if self._mpl_events_bound:
            return
        self._mpl_events_bound = True
        self.figure.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.figure.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

    def _populate_specific_panel(self) -> None:
        for child in self.specific_shared_frame.winfo_children():
            child.destroy()
        for child in self.specific_dynamic_frame.winfo_children():
            child.destroy()

        policy = self.policy_var.get()
        for p_name, defaults in POLICY_DEFAULTS.items():
            if p_name not in self.policy_param_vars:
                self.policy_param_vars[p_name] = {}
            for key, value in defaults.items():
                if key not in self.policy_param_vars[p_name]:
                    self.policy_param_vars[p_name][key] = tk.StringVar(value=str(value))

        shared_keys = ["gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        tooltips = {
            "gamma": "Higher gamma values emphasize long-term return and can improve long-horizon behavior.",
            "learning_rate": "Higher values speed up updates but can destabilize training.",
            "batch_size": "Larger batches reduce gradient noise but increase compute and memory use.",
            "hidden_layer": "Wider/deeper networks improve capacity but are slower to train.",
            "activation": "Activation controls nonlinearity and can affect stability and final policy quality.",
            "lr_strategy": "Schedule shape controls how aggressively learning slows during training.",
            "min_lr": "Lower floor allows slower late-stage refinement while preventing full stall.",
            "lr_decay": "In exponential mode this controls how quickly the LR shrinks.",
        }

        for idx, key in enumerate(shared_keys):
            row = idx // 2
            base_col = 0 if idx % 2 == 0 else 2
            ttk.Label(self.specific_shared_frame, text=key, width=self.LABEL_WIDTH_CHARS, anchor="w").grid(row=row, column=base_col, sticky="w", padx=4, pady=2)
            var = self.policy_param_vars[policy][key]
            if key == "activation":
                widget = ttk.Combobox(self.specific_shared_frame, textvariable=var, values=["ReLU", "Tanh"], state="readonly")
                self._style_combobox(widget)
            elif key == "lr_strategy":
                widget = ttk.Combobox(self.specific_shared_frame, textvariable=var, values=["constant", "linear", "exponential"], state="readonly")
                self._style_combobox(widget)
            else:
                widget = ttk.Entry(self.specific_shared_frame, textvariable=var)
            widget.grid(row=row, column=base_col + 1, sticky="ew", padx=4, pady=2)
            ToolTip(widget, tooltips[key])

        for idx, key in enumerate(self.policy_specific_keys.get(policy, [])):
            row = idx // 2
            base_col = 0 if idx % 2 == 0 else 2
            ttk.Label(self.specific_dynamic_frame, text=key, width=self.LABEL_WIDTH_CHARS, anchor="w").grid(row=row, column=base_col, sticky="w", padx=4, pady=2)
            ttk.Entry(self.specific_dynamic_frame, textvariable=self.policy_param_vars[policy][key]).grid(row=row, column=base_col + 1, sticky="ew", padx=4, pady=2)

        self.compare_param_combo.configure(values=self._compare_param_choices())

    def _compare_param_choices(self) -> List[str]:
        current_policy = self.policy_var.get() or "SAC"
        general = ["Policy", "max_steps", "episodes", "gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        specific = self.policy_specific_keys.get(current_policy, [])
        return general + specific

    def _on_param_canvas_configure(self, event: tk.Event) -> None:
        self.param_canvas.itemconfigure(self.param_window, width=event.width)
        self._update_param_scroll()

    def _update_param_scroll(self) -> None:
        self.param_canvas.configure(scrollregion=self.param_canvas.bbox("all"))
        canvas_h = max(1, self.param_canvas.winfo_height())
        inner_h = max(1, self.param_inner.winfo_reqheight())
        if inner_h > canvas_h:
            self.param_scroll.grid(row=0, column=1, sticky="ns")
        else:
            self.param_scroll.grid_remove()

    def _bind_param_wheel(self, _event: tk.Event) -> None:
        self.param_canvas.bind_all("<MouseWheel>", self._on_param_mousewheel)

    def _unbind_param_wheel(self, _event: tk.Event) -> None:
        self.param_canvas.unbind_all("<MouseWheel>")

    def _on_param_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        steps = -1 if delta > 0 else 1
        self.param_canvas.yview_scroll(steps, "units")

    def _disable_combobox_mousewheel(self, widget: ttk.Combobox) -> None:
        widget.bind("<MouseWheel>", lambda _e: "break")

    def _style_combobox(self, widget: ttk.Combobox) -> None:
        widget.configure(style="Dark.TCombobox")
        self._disable_combobox_mousewheel(widget)
        try:
            popdown = widget.tk.eval(f"ttk::combobox::PopdownWindow {widget}")
            listbox_path = f"{popdown}.f.l"
            widget.tk.call(
                listbox_path,
                "configure",
                "-background",
                "#2d2d30",
                "-foreground",
                "#e6e6e6",
                "-selectbackground",
                "#0e639c",
                "-selectforeground",
                "#ffffff",
            )
        except Exception:
            pass

    def _on_compare_toggle(self) -> None:
        if self.compare_on_var.get():
            self.animation_on_var.set(False)

    def _parse_values(self, text: str) -> List[str]:
        return [v.strip() for v in text.split(",") if v.strip()]

    def _add_compare_entry(self) -> None:
        key = self.compare_param_var.get().strip()
        vals = self._parse_values(self.compare_values_var.get())
        if not key or not vals:
            return
        self.compare_entries[key] = vals
        self._refresh_compare_summary()

    def _clear_compare_entries(self) -> None:
        self.compare_entries.clear()
        self._refresh_compare_summary()

    def _refresh_compare_summary(self) -> None:
        lines = [f"{k}: [{', '.join(v)}]" for k, v in self.compare_entries.items()]
        self.compare_summary_label.configure(text="\n".join(lines))

    def _update_compare_preview(self) -> None:
        key = self.compare_param_var.get().strip()
        text = self.compare_values_var.get().strip()
        if "," in text:
            text = text.split(",")[-1].strip()
        suggestions = []
        if key == "Policy":
            suggestions = ["SAC", "TQC", "CMA-ES"]
        elif key == "activation":
            suggestions = ["ReLU", "Tanh"]
        elif key == "lr_strategy":
            suggestions = ["constant", "linear", "exponential"]
        match = next((s for s in suggestions if s.lower().startswith(text.lower()) and text), "")
        self.compare_preview_var.set(f"Tab -> {match}" if match else "")

    def _compare_tab_complete(self, event: tk.Event) -> str:
        self._update_compare_preview()
        preview = self.compare_preview_var.get()
        if preview.startswith("Tab -> "):
            suggestion = preview.replace("Tab -> ", "", 1)
            raw = self.compare_values_var.get()
            if "," in raw:
                parts = [p for p in raw.split(",")]
                parts[-1] = suggestion
                new_text = ",".join(parts)
            else:
                new_text = suggestion
            self.compare_values_var.set(new_text)
            self.compare_values_entry.icursor(len(new_text))
            self.compare_preview_var.set("")
        return "break"

    def _render_placeholder(self) -> None:
        self.render_canvas.delete("all")
        w = max(10, self.render_canvas.winfo_width())
        h = max(10, self.render_canvas.winfo_height())
        self.render_canvas.create_text(w // 2, h // 2, text="Ant-v5 rgb_array preview", fill="#d0d0d0", font=("Segoe UI", 12, "bold"))

    def _collect_specific_params(self, policy: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, var in self.policy_param_vars[policy].items():
            raw = var.get().strip()
            if key in {"activation", "lr_strategy", "hidden_layer"}:
                out[key] = raw
                continue
            try:
                if any(ch in raw for ch in [".", "e", "E"]):
                    out[key] = float(raw)
                else:
                    out[key] = int(raw)
            except ValueError:
                out[key] = raw
        if policy == "CMA-ES":
            # Keep legacy key in sync so existing compare/config paths remain compatible.
            if "number_of_agents" in out:
                out["popsize"] = out["number_of_agents"]
        return out

    def _to_int(self, value: Any, field: str) -> int:
        try:
            return int(float(str(value).strip()))
        except Exception as exc:
            raise ValueError(f"Invalid integer for '{field}': {value}") from exc

    def _to_float(self, value: Any, field: str) -> float:
        try:
            return float(str(value).strip())
        except Exception as exc:
            raise ValueError(f"Invalid number for '{field}': {value}") from exc

    def _build_configs(self, override: Optional[Dict[str, Any]] = None) -> Tuple[EnvironmentConfig, TrainerConfig]:
        policy = str((override or {}).get("Policy", self.policy_var.get()))

        env_cfg = EnvironmentConfig(
            env_id="Ant-v5",
            max_steps=self._to_int((override or {}).get("max_steps", self.max_steps_var.get()), "max_steps"),
            render_mode="rgb_array",
            forward_reward_weight=self._to_float(self.forward_reward_weight_var.get(), "forward_reward_weight"),
            ctrl_cost_weight=self._to_float(self.ctrl_cost_weight_var.get(), "ctrl_cost_weight"),
            contact_cost_weight=self._to_float(self.contact_cost_weight_var.get(), "contact_cost_weight"),
        )

        policy_compared = bool(self.compare_on_var.get() and "Policy" in self.compare_entries and override and "Policy" in override)
        if policy_compared:
            params = dict(POLICY_DEFAULTS.get(policy, {}))
        else:
            params = self._collect_specific_params(policy)
        if override:
            for key, value in override.items():
                if key in params:
                    params[key] = value

        tr_cfg = TrainerConfig(
            policy=policy,
            episodes=self._to_int((override or {}).get("episodes", self.episodes_var.get()), "episodes"),
            device=self.device_var.get(),
            update_rate=self._to_int(self.update_rate_var.get(), "update_rate"),
            frame_stride=self._to_int(self.frame_stride_var.get(), "frame_stride"),
            moving_average_window=self._to_int(self.ma_values_var.get(), "moving_average_window"),
            evaluation_rollout_on=bool(self.eval_rollout_var.get()),
            specific_params=params,
        )
        return env_cfg, tr_cfg

    def run_single_episode(self) -> None:
        if self._training_active:
            return
        try:
            self._start_training(single_episode=True)
        except Exception as exc:
            self._training_active = False
            self._paused = False
            self._set_button_states()
            self.status_var.set(f"Epsilon: 0.0000 | LR: 0.0000e+00 | Best reward: 0.00 | Render: off | ERROR: {exc}")
            messagebox.showerror("Start Training Error", str(exc))

    def train_and_run(self) -> None:
        if self._training_active and self._paused:
            self._cancel_active_trainers()
            # Start fresh immediately; old worker events are discarded via session filter.
            self._training_active = False
            self._paused = False
            self._set_button_states()
        if self._training_active:
            return
        try:
            self._start_training(single_episode=False)
        except Exception as exc:
            self._training_active = False
            self._paused = False
            self._set_button_states()
            self.status_var.set(f"Epsilon: 0.0000 | LR: 0.0000e+00 | Best reward: 0.00 | Render: off | ERROR: {exc}")
            messagebox.showerror("Start Training Error", str(exc))

    def _start_training(self, single_episode: bool) -> None:
        self._flush_event_queue()
        self._session_id = self._new_session_id()
        self._training_active = True
        self._paused = False
        self._render_run_id = None
        self._run_label_snapshots.clear()
        self._csv_exports.clear()
        self._set_button_states()

        compare_runs = self._build_compare_runs()
        if single_episode:
            compare_runs = compare_runs[:1]
            compare_runs[0]["episodes"] = 1

        max_workers = min(4, max(1, len(compare_runs)))
        available_cores = max(1, os.cpu_count() or 1)
        cpu_threads_per_worker = max(1, available_cores // max_workers)
        render_index = self._select_render_run_index(compare_runs)
        worker_render_allowed = self._is_worker_render_safe()
        if bool(self.animation_on_var.get()) and not worker_render_allowed:
            self.status_var.set(
                "Epsilon: 0.0000 | LR: 0.0000e+00 | Best reward: 0.00 | Render: skipped"
            )

        prepared: List[Tuple[EnvironmentConfig, TrainerConfig, str]] = []
        for idx, override in enumerate(compare_runs):
            env_cfg, tr_cfg = self._build_configs(override=override)
            run_id = f"run-{idx}-{int(time.time() * 1000)}"
            tr_cfg.specific_params["cpu_threads"] = cpu_threads_per_worker
            tr_cfg.render_enabled = bool(self.animation_on_var.get()) and worker_render_allowed and (render_index == idx)

            prepared.append((env_cfg, tr_cfg, run_id))
            self._run_label_snapshots[run_id] = self._build_run_label_snapshot(run_id, env_cfg, tr_cfg, override)

        if prepared:
            self._render_run_id = prepared[render_index][2]

        with self._trainers_lock:
            self._active_trainers.clear()
            self._active_processes.clear()
            self._pending_runs.clear()

            for env_cfg, tr_cfg, run_id in prepared:
                stop_event = self._mp_ctx.Event()
                pause_event = self._mp_ctx.Event()
                pause_event.set()

                proc = self._mp_ctx.Process(
                    target=_ant_worker_main,
                    args=(
                        self._process_event_queue,
                        asdict(env_cfg),
                        asdict(tr_cfg),
                        self._session_id,
                        run_id,
                        stop_event,
                        pause_event,
                        str(self.output_csv_dir),
                    ),
                    daemon=True,
                )
                proc.start()

                self._active_processes[run_id] = proc
                self._active_trainers[run_id] = _ProcessTrainerProxy(stop_event, pause_event)
                self._pending_runs.add(run_id)

    def _build_compare_runs(self) -> List[Dict[str, Any]]:
        base = {
            "Policy": self.policy_var.get(),
            "max_steps": int(float(self.max_steps_var.get())),
            "episodes": int(float(self.episodes_var.get())),
        }
        if not self.compare_on_var.get() or not self.compare_entries:
            return [base]

        converted: Dict[str, List[Any]] = {}
        for key, values in self.compare_entries.items():
            conv: List[Any] = []
            for v in values:
                if key in {"Policy", "activation", "lr_strategy", "hidden_layer"}:
                    conv.append(v)
                else:
                    try:
                        if any(ch in v for ch in [".", "e", "E"]):
                            conv.append(float(v))
                        else:
                            conv.append(int(v))
                    except ValueError:
                        conv.append(v)
            converted[key] = conv
        return expand_compare_runs(base, converted)

    def _is_worker_render_safe(self) -> bool:
        # Kill-switch for systems that still show native render instability:
        # set ANT_DISABLE_WORKER_RENDER=1 to force rendering off in workers.
        return os.environ.get("ANT_DISABLE_WORKER_RENDER", "0") != "1"

    def _bridge_event(self, event: Dict[str, Any]) -> None:
        event.setdefault("session_id", self._session_id)
        self._event_queue.put(event)

    def _drain_process_events(self, max_events: int) -> int:
        moved = 0
        while moved < max_events:
            try:
                event = self._process_event_queue.get_nowait()
            except queue.Empty:
                break
            self._event_queue.put(event)
            moved += 1
        return moved

    def _reap_dead_processes(self) -> None:
        dead: List[Tuple[str, int]] = []
        with self._trainers_lock:
            for run_id, proc in list(self._active_processes.items()):
                if proc.is_alive():
                    continue
                exitcode = int(proc.exitcode or 0)
                dead.append((run_id, exitcode))
                del self._active_processes[run_id]

        for run_id, exitcode in dead:
            if run_id in self._pending_runs and exitcode != 0:
                self._event_queue.put(
                    {
                        "type": "error",
                        "run_id": run_id,
                        "session_id": self._session_id,
                        "message": f"worker process exited unexpectedly (code {exitcode})",
                    }
                )
            if run_id in self._pending_runs:
                self._mark_run_complete(run_id)

    def _mark_run_complete(self, run_id: str) -> None:
        with self._trainers_lock:
            self._pending_runs.discard(run_id)
            self._active_trainers.pop(run_id, None)
            proc = self._active_processes.pop(run_id, None)
        if proc is not None and proc.is_alive():
            proc.join(timeout=0.05)
        if not self._pending_runs:
            self._event_queue.put(
                {
                    "type": "training_done",
                    "session_id": self._session_id,
                    "run_id": "aggregate",
                    "status": "completed",
                }
            )

    def _pump_events(self) -> None:
        strict = bool(self.strict_update_var.get())
        max_events = 1 if strict else 25
        self._drain_process_events(max_events)
        self._reap_dead_processes()
        processed = 0
        while processed < max_events:
            try:
                event = self._event_queue.get_nowait()
            except queue.Empty:
                break
            if event.get("session_id") != self._session_id:
                continue
            self._handle_event(event)
            processed += 1
        self.root.after(16 if strict else 50, self._pump_events)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "step":
            steps = int(event.get("steps", 0))
            max_steps = int(float(self.max_steps_var.get()))
            self.steps_progress["value"] = max(0, min(100, 100.0 * (steps / max(1, max_steps))))
            return

        if etype == "episode":
            run_id = str(event.get("run_id", "run"))
            episode = int(event.get("episode", 0))
            episodes = int(event.get("episodes", 1))
            reward = float(event.get("reward", 0.0))
            moving_average = float(event.get("moving_average", reward))

            hist = self._run_history.setdefault(
                run_id,
                {
                    "rewards": [],
                    "ma": [],
                    "eval": [],
                    "visible": True,
                    "label": self._run_label_snapshots.get(run_id, self._legend_label(run_id)),
                },
            )
            hist["rewards"].append(reward)
            hist["ma"].append(moving_average)

            self.episodes_progress["value"] = max(0, min(100, 100.0 * (episode / max(1, episodes))))

            lr = float(event.get("lr", 0.0))
            best = float(event.get("best_reward", reward))
            render_state = str(event.get("render_state", "idle"))
            self.status_var.set(f"Epsilon: 0.0000 | LR: {lr:.4e} | Best reward: {best:.2f} | Render: {render_state}")
            self._refresh_plot()
            return

        if etype == "episode_aux":
            run_id = str(event.get("run_id", "run"))
            hist = self._run_history.setdefault(
                run_id,
                {
                    "rewards": [],
                    "ma": [],
                    "eval": [],
                    "visible": True,
                    "label": self._run_label_snapshots.get(run_id, self._legend_label(run_id)),
                },
            )
            hist["eval"] = event.get("eval_points", [])

            frames = event.get("frames", [])
            if self.animation_on_var.get() and frames and (self._render_run_id is None or run_id == self._render_run_id):
                self._enqueue_frames(frames)
            self._refresh_plot()
            return

        if etype == "training_done":
            run_id = str(event.get("run_id", ""))
            if run_id and run_id != "aggregate":
                self._mark_run_complete(run_id)
                return
            if event.get("run_id") == "aggregate":
                self._training_active = False
                self._paused = False
                self._set_button_states()
            return

        if etype == "csv_ready":
            run_id = str(event.get("run_id", ""))
            path = str(event.get("path", "")).strip()
            if run_id and path:
                self._csv_exports[run_id] = path
            return

        if etype == "error":
            self.status_var.set(f"Epsilon: 0.0000 | LR: 0.0000e+00 | Best reward: 0.00 | Render: off | ERROR: {event.get('message', 'unknown')}")
            run_id = str(event.get("run_id", ""))
            if run_id:
                self._mark_run_complete(run_id)
            if not self._pending_runs:
                self._training_active = False
                self._paused = False
                self._set_button_states()

    def _legend_label(self, run_id: str) -> str:
        policy = self.policy_var.get()
        gamma = self.policy_param_vars[policy]["gamma"].get()
        lr = self.policy_param_vars[policy]["learning_rate"].get()
        lr_strategy = self.policy_param_vars[policy]["lr_strategy"].get()
        lr_decay = self.policy_param_vars[policy]["lr_decay"].get()
        return (
            f"{policy} | steps={self.max_steps_var.get()} | gamma={gamma}\n"
            f"epsilon=0 | epsilon_decay=0 | epsilon_min=0\n"
            f"LR={lr} | LR strategy={lr_strategy} | LR decay={lr_decay}\n"
            f"forward_reward_weight={self.forward_reward_weight_var.get()} | ctrl_cost_weight={self.ctrl_cost_weight_var.get()} | contact_cost_weight={self.contact_cost_weight_var.get()} | id={run_id}"
        )

    def _build_run_label_snapshot(
        self,
        run_id: str,
        env_cfg: EnvironmentConfig,
        tr_cfg: TrainerConfig,
        override: Dict[str, Any],
    ) -> str:
        policy = tr_cfg.policy
        gamma = tr_cfg.specific_params.get("gamma", "?")
        lr = tr_cfg.specific_params.get("learning_rate", "?")
        lr_strategy = tr_cfg.specific_params.get("lr_strategy", "?")
        lr_decay = tr_cfg.specific_params.get("lr_decay", "?")

        shown_keys = {
            "Policy",
            "policy",
            "max_steps",
            "episodes",
            "gamma",
            "learning_rate",
            "lr_strategy",
            "lr_decay",
            "forward_reward_weight",
            "ctrl_cost_weight",
            "contact_cost_weight",
        }
        extra = [f"{k}={v}" for k, v in override.items() if k not in shown_keys]
        extra_line = f" | compare: {' | '.join(extra)}" if extra else ""

        return (
            f"{policy} | steps={env_cfg.max_steps} | gamma={gamma}\n"
            f"epsilon=0 | epsilon_decay=0 | epsilon_min=0\n"
            f"LR={lr} | LR strategy={lr_strategy} | LR decay={lr_decay}\n"
            f"forward_reward_weight={env_cfg.forward_reward_weight} | ctrl_cost_weight={env_cfg.ctrl_cost_weight} | contact_cost_weight={env_cfg.contact_cost_weight}{extra_line} | id={run_id}"
        )

    def _select_render_run_index(self, runs: List[Dict[str, Any]]) -> int:
        if not runs:
            return 0
        if not self.compare_on_var.get() or len(runs) == 1:
            return 0
        preferred_policy = self.policy_var.get()
        for idx, run in enumerate(runs):
            if str(run.get("Policy", "")) == preferred_policy:
                return idx
        return 0

    def _flush_event_queue(self) -> None:
        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    def _refresh_plot(self) -> None:
        self.ax.clear()
        self._apply_plot_theme()

        colors = ["#4db6ac", "#ffb74d", "#64b5f6", "#ef5350", "#81c784", "#ba68c8"]
        self._line_map.clear()

        for idx, (run_id, hist) in enumerate(self._run_history.items()):
            if not hist.get("visible", True):
                continue
            x = np.arange(1, len(hist["rewards"]) + 1)
            color = colors[idx % len(colors)]
            reward_line, = self.ax.plot(x, hist["rewards"], color=color, alpha=0.30, lw=1.2, label=hist["label"])
            ma_line, = self.ax.plot(x, hist["ma"], color=color, alpha=1.0, lw=2.0, linestyle="-", label="moving average")
            if self.eval_rollout_var.get() and hist["eval"]:
                ex = [p[0] for p in hist["eval"]]
                ey = [p[1] for p in hist["eval"]]
                eval_line, = self.ax.plot(ex, ey, color=color, alpha=1.0, lw=2.0, linestyle=":", marker="o", label="evaluation rollout")
            else:
                eval_line = None
            self._line_map[run_id] = {"reward": reward_line, "ma": ma_line, "eval": eval_line}

        handles, labels = self.ax.get_legend_handles_labels()
        legend = None
        if handles:
            legend = self.ax.legend(loc="upper left", bbox_to_anchor=(1.01, self._legend_anchor_y), frameon=False)
            if legend is not None:
                for txt in legend.get_texts():
                    txt.set_color("#e6e6e6")
                    txt.set_picker(True)
                for handle in legend.legend_handles:
                    handle.set_picker(True)
        self._legend = legend

        if self.strict_update_var.get():
            self.canvas.draw()
        else:
            self.canvas.draw_idle()

        self._update_legend_scroll_bounds()

    def _apply_plot_theme(self) -> None:
        self.figure.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes", color="#e6e6e6")
        self.ax.set_ylabel("Reward", color="#e6e6e6")
        self.ax.tick_params(colors="#d0d0d0")
        for spine in self.ax.spines.values():
            spine.set_color("#606060")
        self.ax.grid(True, color="#4a4a4a", alpha=0.35)

    def _on_legend_pick(self, event: Any) -> None:
        label = ""
        if hasattr(event.artist, "get_label"):
            label = str(event.artist.get_label())
        if (not label or label.startswith("_")) and hasattr(event.artist, "get_text"):
            label = str(event.artist.get_text())

        for run_id, hist in self._run_history.items():
            if hist.get("label") == label:
                hist["visible"] = not hist.get("visible", True)
                break
        self._refresh_plot()

    def _on_plot_motion(self, event: Any) -> None:
        widget = self.canvas.get_tk_widget()
        if self._legend_contains(event):
            widget.configure(cursor="hand2")
            self._set_legend_hover_state(True)
        else:
            widget.configure(cursor="")
            self._set_legend_hover_state(False)

    def _set_legend_hover_state(self, hovering: bool) -> None:
        if self._legend is None:
            return
        hover_color = "#ffffff" if hovering else "#e6e6e6"
        hover_alpha = 1.0 if hovering else 0.9
        for txt in self._legend.get_texts():
            txt.set_color(hover_color)
            txt.set_alpha(hover_alpha)
        self.canvas.draw_idle()

    def _on_plot_scroll(self, event: Any) -> None:
        if not self._legend_scroll_enabled or not self._legend_contains(event):
            return
        step = 0.06
        direction = 1 if (getattr(event, "button", "up") == "up") else -1
        min_anchor = 1.0 - self._legend_scroll_limit
        max_anchor = 1.0
        self._legend_anchor_y = max(min_anchor, min(max_anchor, self._legend_anchor_y + (direction * step)))
        self._refresh_plot()

    def _legend_contains(self, event: Any) -> bool:
        if self._legend is None or event is None:
            return False
        if getattr(event, "x", None) is None or getattr(event, "y", None) is None:
            return False
        renderer = self.figure.canvas.get_renderer()
        if renderer is None:
            return False
        bbox = self._legend.get_window_extent(renderer)
        return bool(bbox.contains(event.x, event.y))

    def _update_legend_scroll_bounds(self) -> None:
        self._legend_scroll_enabled = False
        self._legend_scroll_limit = 0.0
        if self._legend is None:
            return
        renderer = self.figure.canvas.get_renderer()
        if renderer is None:
            return
        legend_h = float(self._legend.get_window_extent(renderer).height)
        axes_h = float(self.ax.get_window_extent(renderer).height)
        if axes_h <= 0:
            return
        overflow_ratio = max(0.0, (legend_h - axes_h) / axes_h)
        self._legend_scroll_enabled = overflow_ratio > 0.0
        self._legend_scroll_limit = min(1.2, overflow_ratio)

    def _enqueue_frames(self, frames: List[np.ndarray]) -> None:
        if not frames:
            return
        if self._frame_playback_active:
            self._frame_pending = frames
            return
        self._frame_playback_active = True
        self._play_frames(frames, 0)

    def _play_frames(self, frames: List[np.ndarray], idx: int) -> None:
        if not self.animation_on_var.get():
            self._frame_playback_active = False
            self._frame_pending = None
            self.steps_progress["value"] = 0
            return
        if idx >= len(frames):
            self._frame_playback_active = False
            if self._frame_pending is not None:
                pending = self._frame_pending
                self._frame_pending = None
                self._enqueue_frames(pending)
            return

        frame = frames[idx]
        self._draw_frame(frame)
        self.steps_progress["value"] = 100.0 * ((idx + 1) / max(1, len(frames)))

        fps = max(1, int(float(self.animation_fps_var.get() or "30")))
        delay = int(1000 / fps)
        self.root.after(delay, lambda: self._play_frames(frames, idx + 1))

    def _draw_frame(self, frame: np.ndarray) -> None:
        if Image is None or ImageTk is None:
            return
        h, w = frame.shape[:2]
        cw = max(1, self.render_canvas.winfo_width())
        ch = max(1, self.render_canvas.winfo_height())
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        img = Image.fromarray(frame)
        img = img.resize((nw, nh), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(img)

        self.render_canvas.delete("all")
        self.render_canvas.create_image(cw // 2, ch // 2, image=photo)
        self._frame_photo = photo

    def _set_button_states(self) -> None:
        self.btn_train.configure(style="Train.TButton" if self._training_active and not self._paused else "Neutral.TButton")
        self.btn_pause.configure(style="Pause.TButton" if self._paused else "Neutral.TButton")
        self.btn_pause.configure(text="Run" if self._paused else "Pause")

    def toggle_pause(self) -> None:
        with self._trainers_lock:
            trainers = list(self._active_trainers.values())
        if not trainers:
            return
        if self._paused:
            for t in trainers:
                t.resume()
            self._paused = False
        else:
            for t in trainers:
                t.pause()
            self._paused = True
        self._set_button_states()

    def _cancel_active_trainers(self) -> None:
        with self._trainers_lock:
            trainers = list(self._active_trainers.values())
            processes = list(self._active_processes.values())
        for t in trainers:
            # Ensure paused loops are unblocked before stop so shutdown can complete cleanly.
            t.resume()
            t.cancel()
        for proc in processes:
            if proc.is_alive():
                proc.join(timeout=0.25)
                if proc.is_alive():
                    proc.terminate()

        with self._trainers_lock:
            self._active_trainers.clear()
            self._active_processes.clear()
            self._pending_runs.clear()

    def reset_all(self) -> None:
        self._cancel_active_trainers()
        self._training_active = False
        self._paused = False
        self._set_button_states()

        self.policy_var.set("SAC")
        self.max_steps_var.set("1000")
        self.episodes_var.set("3000")
        self.device_var.set("CPU")
        self.animation_on_var.set(True)
        self.animation_fps_var.set("30")
        self.update_rate_var.set("1")
        self.frame_stride_var.set("2")
        self.ma_values_var.set("20")
        self.eval_rollout_var.set(False)
        self.strict_update_var.set(False)

        for policy, defaults in POLICY_DEFAULTS.items():
            for key, value in defaults.items():
                self.policy_param_vars.setdefault(policy, {})
                self.policy_param_vars[policy].setdefault(key, tk.StringVar())
                self.policy_param_vars[policy][key].set(str(value))

        self._populate_specific_panel()
        self.clear_plot()
        self._render_placeholder()

    def clear_plot(self) -> None:
        self._run_history.clear()
        self._line_map.clear()
        self._csv_exports.clear()
        self._run_label_snapshots.clear()
        self._render_run_id = None
        self._refresh_plot()
        self.steps_progress["value"] = 0
        self.episodes_progress["value"] = 0

    def save_csv(self) -> None:
        if not self._csv_exports:
            messagebox.showinfo("CSV export", "No CSV files available yet.")
            return
        lines = [self._csv_exports[k] for k in sorted(self._csv_exports.keys())]
        messagebox.showinfo("CSV export", "Saved:\n" + "\n".join(lines))

    def save_plot_png(self) -> None:
        self.output_plot_dir.mkdir(parents=True, exist_ok=True)
        policy = self.policy_var.get()
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"ant_plot_{policy}_{self.max_steps_var.get()}_{ts}.png"
        path = self.output_plot_dir / fname
        self.figure.savefig(path, dpi=120)

    def on_close(self) -> None:
        self._cancel_active_trainers()
        try:
            self._process_event_queue.close()
        except Exception:
            pass
        self.root.after(10, self.root.destroy)
