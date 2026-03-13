from __future__ import annotations

import os
import multiprocessing as mp
import queue
import threading
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None

from Humanoid_logic import (
    EnvironmentConfig,
    LearningRateConfig,
    NetworkConfig,
    POLICY_DEFAULTS,
    TrainerConfig,
    build_default_trainer,
    expand_compare_runs,
    run_trainer_subprocess,
)


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


class HumanoidGUI:
    LABEL_WIDTH_CHARS = 0
    LABEL_CELL_BG = "#252526"
    LABEL_CELL_FG = "#e6e6e6"
    POLICIES = ["PPO", "SAC", "TD3"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Humanoid RL Workbench")
        try:
            # Windows-friendly start in maximized windowed mode.
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1420x930")

        self.project_dir = Path(__file__).resolve().parent
        self.output_csv_dir = self.project_dir / "results_csv"
        self.output_plot_dir = self.project_dir / "plots"

        self._event_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._mp_ctx = mp.get_context("spawn")
        self._mp_event_queue = self._mp_ctx.Queue()
        self._last_error_message = ""
        self._session_id = self._new_session_id()
        self._active_workers: Dict[str, Dict[str, Any]] = {}
        self._pending_runs: set[str] = set()
        self._trainers_lock = threading.Lock()
        self._training_active = False
        self._paused = False

        self._run_history: Dict[str, Dict[str, Any]] = {}
        self._line_map: Dict[str, Dict[str, Any]] = {}
        self._run_label_snapshots: Dict[str, str] = {}
        self._run_compare_meta: Dict[str, Dict[str, Any]] = {}
        self._csv_exports: Dict[str, str] = {}
        self._render_run_id: Optional[str] = None
        self._legend = None
        self._legend_anchor_y = 1.0
        self._legend_scroll_enabled = False
        self._legend_scroll_limit = 0.0

        self._frame_playback_active = False
        self._frame_pending: Optional[List[np.ndarray]] = None
        self._frame_photo = None
        self._last_valid_frame: Optional[np.ndarray] = None

        self.policy_param_vars: Dict[str, Dict[str, tk.StringVar]] = {}
        self.policy_specific_keys: Dict[str, List[str]] = {
            "PPO": ["n_steps", "ent_coef"],
            "SAC": ["buffer_size", "learning_starts", "tau", "train_freq", "gradient_steps"],
            "TD3": [
                "buffer_size",
                "learning_starts",
                "tau",
                "policy_delay",
                "target_policy_noise",
                "target_noise_clip",
                "action_noise_sigma",
            ],
        }

        self._configure_styles()
        self._init_vars()
        self._build_layout()
        self._populate_specific_panel()
        self._refresh_compare_summary()

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
        style.configure("TEntry", fieldbackground="#2d2d30", foreground="#e6e6e6", insertcolor="#ffffff")
        style.configure("TCheckbutton", background="#252526", foreground="#e6e6e6", font=("Segoe UI", 10))
        style.map("TCheckbutton", background=[("active", "#252526"), ("selected", "#252526")], foreground=[("disabled", "#8a8a8a")])
        style.configure("Dark.TCombobox", fieldbackground="#2d2d30", background="#2d2d30", foreground="#e6e6e6")
        style.map(
            "Dark.TCombobox",
            fieldbackground=[("readonly", "#2d2d30")],
            foreground=[("readonly", "#e6e6e6")],
            selectbackground=[("readonly", "#0e639c")],
            selectforeground=[("readonly", "#ffffff")],
        )
        style.configure("Neutral.TButton", background="#3a3d41", foreground="#e6e6e6", font=("Segoe UI", 10, "bold"))
        style.configure("Train.TButton", background="#0e639c", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.configure("Pause.TButton", background="#a66a00", foreground="#ffffff", font=("Segoe UI", 10, "bold"))
        style.map("Neutral.TButton", background=[("active", "#4a4f55"), ("pressed", "#2f3338")])
        style.map("Train.TButton", background=[("active", "#1177bb"), ("pressed", "#0b4f7a")])
        style.map("Pause.TButton", background=[("active", "#bf7a00"), ("pressed", "#8c5900")])
        style.configure("Horizontal.TProgressbar", troughcolor="#343434", background="#0e639c")
        self.root.option_add("*insertBackground", "#ffffff")
        self.root.option_add("*TCombobox*Listbox.background", "#2d2d30")
        self.root.option_add("*TCombobox*Listbox.foreground", "#e6e6e6")
        self.root.option_add("*TCombobox*Listbox.selectBackground", "#0e639c")
        self.root.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")

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

    def _init_vars(self) -> None:
        self.animation_on_var = tk.BooleanVar(value=True)
        self.animation_fps_var = tk.StringVar(value="30")
        self.update_rate_var = tk.StringVar(value="1")
        self.frame_stride_var = tk.StringVar(value="2")

        self.compare_on_var = tk.BooleanVar(value=False)
        self.compare_param_var = tk.StringVar(value="Policy")
        self.compare_values_var = tk.StringVar(value="")
        self.compare_preview_var = tk.StringVar(value="")
        self.compare_entries: Dict[str, List[str]] = {}

        self.max_steps_var = tk.StringVar(value="1000")
        self.episodes_var = tk.StringVar(value="4000")

        self.policy_var = tk.StringVar(value="SAC")
        self.ma_values_var = tk.StringVar(value="20")
        self.eval_rollout_var = tk.BooleanVar(value=False)
        self.device_var = tk.StringVar(value="CPU")

        self.env_vars: Dict[str, tk.Variable] = {
            "forward_reward_weight": tk.StringVar(value="1.25"),
            "ctrl_cost_weight": tk.StringVar(value="0.1"),
            "contact_cost_weight": tk.StringVar(value="5e-7"),
            "healthy_reward": tk.StringVar(value="5.0"),
            "terminate_when_unhealthy": tk.BooleanVar(value=True),
        }

        for policy, defaults in POLICY_DEFAULTS.items():
            self.policy_param_vars[policy] = {}
            for key, value in defaults.items():
                if isinstance(value, tuple):
                    text = ",".join(str(v) for v in value)
                else:
                    text = str(value)
                self.policy_param_vars[policy][key] = tk.StringVar(value=text)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=2)
        outer.columnconfigure(1, weight=1)
        # Keep Environment panel taller than Live Plot with an explicit 1.5 ratio.
        outer.rowconfigure(0, weight=3, minsize=540)
        outer.rowconfigure(1, weight=0)
        outer.rowconfigure(2, weight=0)
        outer.rowconfigure(3, weight=2, minsize=360)

        env_group = ttk.LabelFrame(outer, text="Environment", style="Card.TLabelframe", padding=8)
        env_group.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        self.render_canvas = tk.Canvas(env_group, bg="#111111", highlightthickness=0, height=540)
        self.render_canvas.pack(fill=tk.BOTH, expand=True)

        param_group = ttk.LabelFrame(outer, text="Parameters", style="Card.TLabelframe", padding=0)
        param_group.grid(row=0, column=1, sticky="nsew", pady=(0, 6))
        param_group.rowconfigure(0, weight=1)
        param_group.columnconfigure(0, weight=1)
        self.param_canvas = tk.Canvas(param_group, bg="#252526", highlightthickness=0)
        self.param_canvas.grid(row=0, column=0, sticky="nsew")
        self.param_scroll = ttk.Scrollbar(param_group, orient="vertical", command=self.param_canvas.yview)
        self.param_canvas.configure(yscrollcommand=self.param_scroll.set)
        self.param_window = self.param_canvas.create_window((0, 0), anchor="nw")
        self.param_inner = ttk.Frame(self.param_canvas)
        self.param_canvas.itemconfigure(self.param_window, window=self.param_inner)
        self.param_inner.bind("<Configure>", self._on_param_inner_configure)
        self.param_canvas.bind("<Configure>", self._on_param_canvas_configure)
        # Use a global wheel binding with panel hit-testing for reliable scrolling.
        self.root.bind_all("<MouseWheel>", self._on_global_mousewheel, add="+")

        self._build_param_groups()

        controls = ttk.LabelFrame(outer, text="Controls", style="Card.TLabelframe", padding=8)
        controls.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        for col in range(8):
            controls.columnconfigure(col, weight=1)

        self.btn_run_episode = ttk.Button(controls, text="Run single episode", style="Neutral.TButton", command=self.run_single_episode)
        self.btn_train = ttk.Button(controls, text="Train and Run", style="Neutral.TButton", command=self.train_and_run)
        self.btn_pause = ttk.Button(controls, text="Pause", style="Neutral.TButton", command=self.toggle_pause)
        self.btn_reset = ttk.Button(controls, text="Reset All", style="Neutral.TButton", command=self.reset_all)
        self.btn_clear = ttk.Button(controls, text="Clear Plot", style="Neutral.TButton", command=self.clear_plot)
        self.btn_csv = ttk.Button(controls, text="Save samplings CSV", style="Neutral.TButton", command=self.save_samplings_csv)
        self.btn_png = ttk.Button(controls, text="Save Plot PNG", style="Neutral.TButton", command=self.save_plot_png)
        self.device_combo = ttk.Combobox(controls, textvariable=self.device_var, values=["CPU", "GPU"], state="readonly")
        self._style_combobox(self.device_combo)

        self.btn_run_episode.grid(row=0, column=0, padx=3, sticky="ew")
        self.btn_train.grid(row=0, column=1, padx=3, sticky="ew")
        self.btn_pause.grid(row=0, column=2, padx=3, sticky="ew")
        self.btn_reset.grid(row=0, column=3, padx=3, sticky="ew")
        self.btn_clear.grid(row=0, column=4, padx=3, sticky="ew")
        self.btn_csv.grid(row=0, column=5, padx=3, sticky="ew")
        self.btn_png.grid(row=0, column=6, padx=3, sticky="ew")
        self.device_combo.grid(row=0, column=7, padx=3, sticky="ew")

        current = ttk.LabelFrame(outer, text="Current Run", style="Card.TLabelframe", padding=8)
        current.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        current.columnconfigure(1, weight=1)
        ttk.Label(current, text="Steps").grid(row=0, column=0, sticky="w", padx=4)
        self.steps_progress = ttk.Progressbar(current, orient="horizontal", mode="determinate", maximum=100)
        self.steps_progress.grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(current, text="Episodes").grid(row=1, column=0, sticky="w", padx=4)
        self.episodes_progress = ttk.Progressbar(current, orient="horizontal", mode="determinate", maximum=100)
        self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=4)
        self.status_var = tk.StringVar(value="LR: - | Best reward: - | Render: idle")
        ttk.Label(current, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 0))

        plot_group = ttk.LabelFrame(outer, text="Live Plot", style="Card.TLabelframe", padding=8)
        plot_group.grid(row=3, column=0, columnspan=2, sticky="nsew")
        plot_group.rowconfigure(0, weight=1)
        plot_group.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10.5, 3.6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_group)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas.get_tk_widget().configure(height=360)
        self._apply_plot_theme()
        self._bind_mpl_events_once()

    def _build_param_groups(self) -> None:
        self.env_params_group = ttk.LabelFrame(self.param_inner, text="Environment", style="Card.TLabelframe", padding=6)
        self.compare_params_group = ttk.LabelFrame(self.param_inner, text="Compare", style="Card.TLabelframe", padding=6)
        self.general_params_group = ttk.LabelFrame(self.param_inner, text="General", style="Card.TLabelframe", padding=6)
        self.specific_params_group = ttk.LabelFrame(self.param_inner, text="Specific", style="Card.TLabelframe", padding=6)
        self.live_plot_group = ttk.LabelFrame(self.param_inner, text="Live Plot", style="Card.TLabelframe", padding=6)

        self.env_params_group.pack(fill="x", padx=6, pady=4)
        self.compare_params_group.pack(fill="x", padx=6, pady=4)
        self.general_params_group.pack(fill="x", padx=6, pady=4)
        self.specific_params_group.pack(fill="x", padx=6, pady=4)
        self.live_plot_group.pack(fill="x", padx=6, pady=4)

        self._build_environment_group_fields()
        self._build_compare_group_fields()
        self._build_general_group_fields()
        self._build_specific_group_fields()
        self._build_live_plot_group_fields()

    def _build_environment_group_fields(self) -> None:
        self._configure_pair_columns(self.env_params_group)

        # Keep animation/render pacing controls above the environment update button.
        ttk.Checkbutton(self.env_params_group, text="Animation On", variable=self.animation_on_var).grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self._add_labeled_entry(self.env_params_group, 0, "Animation Fps", self.animation_fps_var, pair=1)
        self._add_labeled_entry(self.env_params_group, 1, "Update Rate (Episodes)", self.update_rate_var, pair=0)
        self._add_labeled_entry(self.env_params_group, 1, "Frame Stride", self.frame_stride_var, pair=1)
        ttk.Button(self.env_params_group, text="Update", style="Neutral.TButton", command=self._update_environment).grid(row=2, column=0, columnspan=4, sticky="ew", padx=6, pady=4)

        env_fields: List[Tuple[str, str]] = [
            ("Forward Reward Weight", "forward_reward_weight"),
            ("Healthy Reward", "healthy_reward"),
            ("Ctrl Cost Weight", "ctrl_cost_weight"),
            ("Contact Cost Weight", "contact_cost_weight"),
        ]
        for idx, (label, key) in enumerate(env_fields):
            row = 3 + (idx // 2)
            pair = idx % 2
            self._add_labeled_entry(self.env_params_group, row, label, self.env_vars[key], pair=pair)

        ttk.Checkbutton(self.env_params_group, text="Terminate When Unhealthy", variable=self.env_vars["terminate_when_unhealthy"]).grid(
            row=5,
            column=0,
            columnspan=4,
            sticky="w",
            padx=6,
            pady=3,
        )

    def _build_compare_group_fields(self) -> None:
        self.compare_params_group.columnconfigure(0, weight=1)
        self.compare_params_group.columnconfigure(1, weight=1)
        top = ttk.Frame(self.compare_params_group)
        top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=2)
        top.columnconfigure(0, weight=1)
        top.columnconfigure(1, weight=0)
        top.columnconfigure(2, weight=0)

        ttk.Checkbutton(top, text="Compare On", variable=self.compare_on_var, command=self._on_compare_toggle).grid(row=0, column=0, sticky="w")
        ttk.Button(top, text="Clear", style="Neutral.TButton", command=self._clear_compare_entries).grid(row=0, column=1, padx=4)
        ttk.Button(top, text="Add", style="Neutral.TButton", command=self._add_compare_entry).grid(row=0, column=2, padx=4)

        self.compare_param_combo = ttk.Combobox(self.compare_params_group, textvariable=self.compare_param_var, values=self._compare_param_choices(), state="readonly")
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
        self._configure_pair_columns(self.general_params_group)
        self._add_labeled_entry(self.general_params_group, row=0, label="Max Steps", var=self.max_steps_var, pair=0)
        self._add_labeled_entry(self.general_params_group, row=0, label="Episodes", var=self.episodes_var, pair=1)

    def _build_specific_group_fields(self) -> None:
        self._configure_pair_columns(self.specific_params_group)

        ttk.Label(self.specific_params_group, text="Policy", width=self.LABEL_WIDTH_CHARS, anchor="w").grid(row=0, column=0, sticky="w", padx=6, pady=3)
        self.policy_combo = ttk.Combobox(self.specific_params_group, textvariable=self.policy_var, values=self.POLICIES, state="readonly")
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
            self._configure_pair_columns(frame)

    def _build_live_plot_group_fields(self) -> None:
        self._configure_pair_columns(self.live_plot_group)
        self._add_labeled_entry(self.live_plot_group, row=0, label="Moving average values", var=self.ma_values_var, pair=0)
        ttk.Checkbutton(self.live_plot_group, text="Evaluation Rollout On", variable=self.eval_rollout_var).grid(row=0, column=2, columnspan=2, sticky="w", padx=6, pady=3)

    def _configure_pair_columns(self, parent: ttk.Widget) -> None:
        # Enforce 50:50 label/input width split for each pair.
        parent.columnconfigure(0, weight=1, uniform="pair")
        parent.columnconfigure(1, weight=1, uniform="pair")
        parent.columnconfigure(2, weight=1, uniform="pair")
        parent.columnconfigure(3, weight=1, uniform="pair")

    def _add_cell_label(self, parent: ttk.Widget, row: int, col: int, text: str, padx: Tuple[int, int] = (0, 0), pady: int = 0) -> None:
        # Dedicated label cell frame ensures background fills the full label column cell.
        label_cell = tk.Frame(parent, bg=self.LABEL_CELL_BG, bd=0, highlightthickness=0)
        label_cell.grid(row=row, column=col, sticky="nsew", padx=padx, pady=pady)
        tk.Label(
            label_cell,
            text=text,
            anchor="w",
            bg=self.LABEL_CELL_BG,
            fg=self.LABEL_CELL_FG,
            font=("Segoe UI", 10),
            padx=6,
            pady=3,
        ).pack(fill="both", expand=True)

    @staticmethod
    def _pretty_label(text: str) -> str:
        words = [w for w in str(text).replace("_", " ").split(" ") if w]
        if not words:
            return ""
        return " ".join(word[:1].upper() + word[1:] for word in words)

    def _add_labeled_entry(self, parent: ttk.Widget, row: int, label: str, var: tk.Variable, pair: int = 0) -> ttk.Entry:
        base_col = 0 if pair <= 0 else 2
        self._add_cell_label(parent, row=row, col=base_col, text=label, padx=(0, 0), pady=0)
        entry = ttk.Entry(parent, textvariable=var)
        entry.grid(row=row, column=base_col + 1, sticky="ew", padx=(4, 6), pady=3)
        return entry

    def _on_param_inner_configure(self, _event: tk.Event) -> None:
        self._update_param_scroll()

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

    def _is_widget_in_params_panel(self, widget: Optional[tk.Widget]) -> bool:
        current = widget
        while current is not None:
            if current is self.param_canvas or current is self.param_inner:
                return True
            current = getattr(current, "master", None)
        return False

    def _on_global_mousewheel(self, event: tk.Event) -> None:
        widget_under_cursor = self.root.winfo_containing(event.x_root, event.y_root)
        if not self._is_widget_in_params_panel(widget_under_cursor):
            return
        self._on_param_mousewheel(event)

    def _on_param_mousewheel(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0))
        if delta == 0:
            return
        self.param_canvas.yview_scroll(-1 if delta > 0 else 1, "units")

    def _populate_specific_panel(self) -> None:
        for child in self.specific_shared_frame.winfo_children():
            child.destroy()
        for child in self.specific_dynamic_frame.winfo_children():
            child.destroy()

        policy = self.policy_var.get()
        shared_keys = ["gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        tooltips = {
            "gamma": "Higher gamma values emphasize long-term return and can improve long-horizon behavior.",
            "learning_rate": "Higher values speed up updates but can destabilize training.",
            "batch_size": "Larger batches reduce gradient noise but increase compute and memory use.",
            "hidden_layer": "Wider or deeper networks increase capacity but cost more compute.",
            "activation": "Activation changes nonlinearity and can affect stability and final score.",
            "lr_strategy": "Schedule shape controls how quickly learning slows down.",
            "min_lr": "Lower floors allow slower late-stage refinement.",
            "lr_decay": "In exponential mode this controls shrink speed.",
        }

        for idx, key in enumerate(shared_keys):
            row = idx // 2
            base_col = 0 if idx % 2 == 0 else 2
            self._add_cell_label(self.specific_shared_frame, row=row, col=base_col, text=self._pretty_label(key), padx=(0, 0), pady=0)
            var = self.policy_param_vars[policy][key]
            if key == "activation":
                widget = ttk.Combobox(self.specific_shared_frame, textvariable=var, values=["ReLU", "Tanh"], state="readonly")
                self._style_combobox(widget)
            elif key == "lr_strategy":
                widget = ttk.Combobox(self.specific_shared_frame, textvariable=var, values=["constant", "linear", "exponential"], state="readonly")
                self._style_combobox(widget)
            else:
                widget = ttk.Entry(self.specific_shared_frame, textvariable=var)
            widget.grid(row=row, column=base_col + 1, sticky="ew", padx=(4, 6), pady=3)
            ToolTip(widget, tooltips[key])

        for idx, key in enumerate(self.policy_specific_keys.get(policy, [])):
            row = idx // 2
            base_col = 0 if idx % 2 == 0 else 2
            self._add_cell_label(self.specific_dynamic_frame, row=row, col=base_col, text=self._pretty_label(key), padx=(0, 0), pady=0)
            ttk.Entry(self.specific_dynamic_frame, textvariable=self.policy_param_vars[policy][key]).grid(row=row, column=base_col + 1, sticky="ew", padx=(4, 6), pady=3)

        self.compare_param_combo.configure(values=self._compare_param_choices())

    def _compare_param_choices(self) -> List[str]:
        current_policy = self.policy_var.get() or "SAC"
        general = ["Policy", "max_steps", "episodes", "gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]
        specific = self.policy_specific_keys.get(current_policy, [])
        return general + specific

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
        suggestions: List[str] = []
        if key == "Policy":
            suggestions = self.POLICIES
        elif key == "activation":
            suggestions = ["ReLU", "Tanh"]
        elif key == "lr_strategy":
            suggestions = ["constant", "linear", "exponential"]
        match = next((s for s in suggestions if s.lower().startswith(text.lower()) and text), "")
        self.compare_preview_var.set(f"Tab -> {match}" if match else "")

    def _compare_tab_complete(self, _event: tk.Event) -> str:
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

    @staticmethod
    def _parse_float(value: str) -> float:
        text = value.strip().lower()
        if text in {"-inf", "-infinity"}:
            return float("-inf")
        if text in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        return float(value)

    @staticmethod
    def _parse_train_freq(raw: str) -> Any:
        text = str(raw).strip()
        if not text:
            return 1
        compact = text.strip("()[]").replace(" ", "")
        if "," in compact:
            parts = [p.strip("\"'").lower() for p in compact.split(",") if p]
            if len(parts) == 2:
                return (int(parts[0]), parts[1])
        return int(compact)

    def _build_env_config(self) -> EnvironmentConfig:
        defaults = EnvironmentConfig()
        return EnvironmentConfig(
            forward_reward_weight=float(str(self.env_vars["forward_reward_weight"].get())),
            ctrl_cost_weight=float(str(self.env_vars["ctrl_cost_weight"].get())),
            contact_cost_weight=float(str(self.env_vars["contact_cost_weight"].get())),
            contact_cost_range_low=defaults.contact_cost_range_low,
            contact_cost_range_high=defaults.contact_cost_range_high,
            healthy_reward=float(str(self.env_vars["healthy_reward"].get())),
            terminate_when_unhealthy=bool(self.env_vars["terminate_when_unhealthy"].get()),
            healthy_z_range_low=defaults.healthy_z_range_low,
            healthy_z_range_high=defaults.healthy_z_range_high,
            render_mode="rgb_array",
        )

    def _update_environment(self) -> None:
        try:
            cfg = self._build_env_config()
            with self._trainers_lock:
                trainers = [
                    entry.get("trainer")
                    for entry in self._active_workers.values()
                    if entry.get("type") == "thread" and entry.get("trainer") is not None
                ]
            for trainer in trainers:
                trainer.update_environment(cfg)
            self.status_var.set("LR: - | Best reward: - | Render: idle")
        except ValueError:
            messagebox.showerror("Invalid input", "Environment parameters must be numeric.")

    def _collect_specific_params(self, policy: str) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for key, var in self.policy_param_vars[policy].items():
            raw = str(var.get()).strip()
            if key in {"activation", "lr_strategy", "hidden_layer"}:
                out[key] = raw
                continue
            if key == "train_freq":
                try:
                    out[key] = self._parse_train_freq(raw)
                except (TypeError, ValueError):
                    out[key] = raw
                continue
            try:
                if any(ch in raw for ch in [".", "e", "E"]):
                    out[key] = float(raw)
                else:
                    out[key] = int(raw)
            except ValueError:
                out[key] = raw
        return out

    def _build_trainer_config(self, override: Optional[Dict[str, Any]] = None) -> Tuple[EnvironmentConfig, TrainerConfig]:
        override = override or {}
        policy = str(override.get("Policy", self.policy_var.get()))

        params = self._collect_specific_params(policy)
        for key, value in override.items():
            if key in {"Policy", "max_steps", "episodes"}:
                continue
            if key in params:
                params[key] = value

        lr_cfg = LearningRateConfig(
            learning_rate=float(params.get("learning_rate", 3e-4)),
            lr_strategy=str(params.get("lr_strategy", "constant")),
            min_lr=float(params.get("min_lr", 1e-5)),
            lr_decay=float(params.get("lr_decay", 0.999)),
        )
        network_cfg = NetworkConfig(
            hidden_layer=str(params.get("hidden_layer", "256,256")),
            activation=str(params.get("activation", "ReLU")).lower(),
        )

        policy_params = dict(params)
        for key in ["hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay", "learning_rate"]:
            policy_params.pop(key, None)

        env_cfg = self._build_env_config()

        max_steps = int(override.get("max_steps", int(float(self.max_steps_var.get()))))
        episodes = int(override.get("episodes", int(float(self.episodes_var.get()))))
        eval_every = 10 if self.eval_rollout_var.get() else max(episodes + 1, 999999)

        cfg = TrainerConfig(
            policy_name=policy,
            episodes=episodes,
            max_steps=max_steps,
            update_rate=int(float(self.update_rate_var.get())),
            frame_stride=int(float(self.frame_stride_var.get())),
            deterministic_eval_every=eval_every,
            deterministic_eval_episodes=1,
            collect_transitions=True,
            export_csv=True,
            split_aux_events=bool(self.compare_on_var.get()),
            device=self.device_var.get(),
            session_id=self._session_id,
            run_id="",
            env=env_cfg,
            network=network_cfg,
            lr=lr_cfg,
            policy_params=policy_params,
        )
        return env_cfg, cfg

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
                    continue
                try:
                    if any(ch in v for ch in [".", "e", "E"]):
                        conv.append(float(v))
                    else:
                        conv.append(int(v))
                except ValueError:
                    conv.append(v)
            converted[key] = conv
        return expand_compare_runs(base, converted)

    def _select_render_run_index(self, runs: List[Dict[str, Any]]) -> int:
        selected_policy = self.policy_var.get()
        for idx, run in enumerate(runs):
            if str(run.get("Policy", "")) == selected_policy:
                return idx
        return 0

    def run_single_episode(self) -> None:
        self._start_training(single_episode=True)

    def train_and_run(self) -> None:
        if self._training_active and self._paused:
            self._cancel_active_trainers()
            self._training_active = False
            self._paused = False
            self._set_button_states()
        if self._training_active:
            return
        self._start_training(single_episode=False)

    def _start_training(self, single_episode: bool) -> None:
        self._flush_event_queue()
        self._session_id = self._new_session_id()
        self._mp_event_queue = self._mp_ctx.Queue()
        self._last_error_message = ""
        self._training_active = True
        self._paused = False
        self._render_run_id = None
        self._run_label_snapshots.clear()
        self._run_compare_meta.clear()
        self._set_button_states()

        runs = self._build_compare_runs()
        if single_episode:
            runs = runs[:1]
            runs[0]["episodes"] = 1

        if self.compare_on_var.get():
            self.animation_on_var.set(False)

        render_idx = self._select_render_run_index(runs)
        max_workers = min(4, max(1, len(runs)))
        sem = threading.Semaphore(max_workers)

        self.steps_progress["value"] = 0
        self.episodes_progress["value"] = 0

        prepared: List[Tuple[str, Dict[str, Any], EnvironmentConfig, TrainerConfig]] = []
        for idx, override in enumerate(runs):
            env_cfg, tr_cfg = self._build_trainer_config(override)
            run_id = f"run-{idx}-{int(time.time() * 1000)}"
            # Keep only the selected run render-capable. The trainer's live
            # animation callback decides whether rendering is actually on/off.
            should_render = idx == render_idx
            if not should_render:
                env_cfg = replace(env_cfg, render_mode=None)
                tr_cfg = replace(tr_cfg, env=env_cfg)
            tr_cfg = replace(tr_cfg, session_id=self._session_id, run_id=run_id)
            prepared.append((run_id, override, env_cfg, tr_cfg))

        if prepared:
            self._render_run_id = prepared[render_idx][0]

        with self._trainers_lock:
            self._active_workers.clear()
            self._pending_runs.clear()
            for run_id, override, env_cfg, tr_cfg in prepared:
                self._run_compare_meta[run_id] = dict(override)
                self._run_label_snapshots[run_id] = self._build_run_label_snapshot(run_id, env_cfg, tr_cfg, override)
                use_process = self._should_use_process_worker(tr_cfg)
                if use_process:
                    pause_event = self._mp_ctx.Event()
                    pause_event.set()
                    cancel_event = self._mp_ctx.Event()
                    proc = self._mp_ctx.Process(
                        target=run_trainer_subprocess,
                        args=(env_cfg, tr_cfg, self._mp_event_queue, pause_event, cancel_event),
                        daemon=True,
                    )
                    self._active_workers[run_id] = {
                        "type": "process",
                        "process": proc,
                        "pause_event": pause_event,
                        "cancel_event": cancel_event,
                    }
                else:
                    trainer = build_default_trainer(
                        event_sink=self._make_worker_sink(run_id),
                        render_enabled_fn=self.animation_on_var.get,
                    )
                    trainer.update_environment(env_cfg)
                    self._active_workers[run_id] = {"type": "thread", "trainer": trainer, "thread": None}
                self._pending_runs.add(run_id)

        for run_id, _override, _env_cfg, tr_cfg in prepared:
            with self._trainers_lock:
                entry = self._active_workers.get(run_id)
            if entry is None:
                continue
            if entry.get("type") == "process":
                proc = entry.get("process")
                if proc is not None:
                    proc.start()
            else:
                thread = threading.Thread(target=self._worker_entry, args=(run_id, tr_cfg, sem), daemon=True)
                with self._trainers_lock:
                    if run_id in self._active_workers:
                        self._active_workers[run_id]["thread"] = thread
                thread.start()

    def _should_use_process_worker(self, cfg: TrainerConfig) -> bool:
        return str(cfg.device).upper() == "GPU"

    def _worker_entry(self, run_id: str, cfg: TrainerConfig, sem: threading.Semaphore) -> None:
        with sem:
            trainer = None
            with self._trainers_lock:
                if run_id in self._active_workers:
                    trainer = self._active_workers[run_id]["trainer"]
            if trainer is None:
                return
            try:
                result = trainer.train(cfg)
                self._event_queue.put(result)
            except Exception as exc:  # pragma: no cover - runtime safety
                self._event_queue.put(
                    {
                        "type": "error",
                        "session_id": cfg.session_id,
                        "run_id": run_id,
                        "message": str(exc),
                    }
                )

    def _make_worker_sink(self, run_id: str):
        def _sink(event: Dict[str, Any]) -> None:
            payload = dict(event)
            payload["session_id"] = self._session_id
            payload["run_id"] = run_id
            self._event_queue.put(payload)

        return _sink

    def _flush_event_queue(self) -> None:
        while True:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break

    def _set_button_states(self) -> None:
        self.btn_train.configure(style="Train.TButton" if self._training_active and not self._paused else "Neutral.TButton")
        self.btn_pause.configure(style="Pause.TButton" if self._paused else "Neutral.TButton")
        self.btn_pause.configure(text="Run" if self._paused else "Pause")

    def toggle_pause(self) -> None:
        with self._trainers_lock:
            trainers = [
                entry.get("trainer")
                for entry in self._active_workers.values()
                if entry.get("type") == "thread" and entry.get("trainer") is not None
            ]
            process_pause_events = [
                entry.get("pause_event")
                for entry in self._active_workers.values()
                if entry.get("type") == "process" and entry.get("pause_event") is not None
            ]

        if not trainers and not process_pause_events:
            return

        if self._paused:
            for t in trainers:
                t.resume()
            for evt in process_pause_events:
                evt.set()
            self._paused = False
        else:
            for t in trainers:
                t.pause()
            for evt in process_pause_events:
                evt.clear()
            self._paused = True
        self._set_button_states()

    def _cancel_active_trainers(self) -> None:
        with self._trainers_lock:
            workers = list(self._active_workers.values())
        for entry in workers:
            if entry.get("type") == "process":
                cancel_evt = entry.get("cancel_event")
                pause_evt = entry.get("pause_event")
                if cancel_evt is not None:
                    cancel_evt.set()
                if pause_evt is not None:
                    pause_evt.set()
                proc = entry.get("process")
                if proc is not None and proc.is_alive():
                    proc.join(timeout=1.0)
                    if proc.is_alive():
                        proc.terminate()
                        proc.join(timeout=1.0)
                continue
            trainer = entry.get("trainer")
            if trainer is not None:
                trainer.resume()
                trainer.cancel()

    def reset_all(self) -> None:
        self._cancel_active_trainers()
        self._training_active = False
        self._paused = False
        self._set_button_states()
        self.policy_var.set("SAC")
        self.max_steps_var.set("1000")
        self.episodes_var.set("4000")
        self.device_var.set("CPU")
        self.compare_on_var.set(False)
        self._clear_compare_entries()
        self.clear_plot()

    def clear_plot(self) -> None:
        self._run_history.clear()
        self._line_map.clear()
        self._refresh_plot()

    def save_samplings_csv(self) -> None:
        if not self._csv_exports:
            self.status_var.set("LR: - | Best reward: - | Render: off")
            return
        latest = sorted(self._csv_exports.items(), key=lambda item: item[0])[-1][1]
        self.status_var.set(f"LR: - | Best reward: - | Render: off | CSV: {latest}")

    def save_plot_png(self) -> None:
        self.output_plot_dir.mkdir(exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = self.output_plot_dir / f"Humanoid_plot_{now}.png"
        self.figure.savefig(str(output), dpi=120)
        self.status_var.set(f"LR: - | Best reward: - | Render: off")

    def _pump_events(self) -> None:
        self._drain_mp_events()
        max_events = 25
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

        self.root.after(50, self._pump_events)

    def _drain_mp_events(self) -> None:
        while True:
            try:
                event = self._mp_event_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
            if isinstance(event, dict):
                self._event_queue.put(event)

    def _handle_event(self, event: Dict[str, Any]) -> None:
        etype = event.get("type")
        if etype == "episode":
            self._handle_episode_event(event)
            return
        if etype == "episode_aux":
            self._handle_episode_aux_event(event)
            return
        if etype == "training_done":
            self._handle_training_done(event)
            return
        if etype == "error":
            self._handle_error(event)

    def _handle_episode_event(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id", "run"))
        episode = int(payload.get("episode", 0))
        episodes = int(payload.get("episodes", 1))
        reward = float(payload.get("reward", 0.0))
        moving_average = float(payload.get("moving_average", reward))
        steps = int(payload.get("steps", 0))

        hist = self._run_history.setdefault(
            run_id,
            {
                "rewards": [],
                "ma": [],
                "eval": [],
                "visible": True,
                "label": self._run_label_snapshots.get(run_id, run_id),
            },
        )
        hist["rewards"].append(reward)
        hist["ma"].append(moving_average)
        if "eval_points" in payload:
            hist["eval"] = list(payload.get("eval_points", hist["eval"]))

        if run_id == self._render_run_id:
            best_reward = float(max(hist["rewards"])) if hist["rewards"] else 0.0
            lr = payload.get("lr", "-")
            render_state = "on" if self.animation_on_var.get() else "off"
            self.status_var.set(f"LR: {lr} | Best reward: {best_reward:.2f} | Render: {render_state}")
            self.episodes_progress["value"] = max(0, min(100, 100.0 * (episode / max(1, episodes))))
            if not self._frame_playback_active:
                self.steps_progress["value"] = max(0, min(100, 100.0 * (steps / max(1, int(float(self.max_steps_var.get()))))))

            frames = payload.get("frames")
            if isinstance(frames, list) and frames:
                self._enqueue_frames(frames)
            else:
                render_state = payload.get("render_state")
                if isinstance(render_state, np.ndarray) and not self._frame_playback_active:
                    self._draw_frame(render_state)

        self._refresh_plot()

    def _handle_episode_aux_event(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id", "run"))
        hist = self._run_history.get(run_id)
        if hist is None:
            return
        hist["eval"] = list(payload.get("eval_points", hist["eval"]))

        if run_id == self._render_run_id:
            frames = payload.get("frames")
            if isinstance(frames, list) and frames:
                self._enqueue_frames(frames)
            else:
                frame = payload.get("frame")
                if isinstance(frame, np.ndarray) and not self._frame_playback_active:
                    self._draw_frame(frame)

        self._refresh_plot()

    def _handle_training_done(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id", ""))
        csv_path = payload.get("csv_path")
        if isinstance(csv_path, str) and csv_path:
            self._csv_exports[run_id] = csv_path
        with self._trainers_lock:
            self._pending_runs.discard(run_id)
            entry = self._active_workers.pop(run_id, None)
            if entry and entry.get("type") == "process":
                proc = entry.get("process")
                if proc is not None:
                    proc.join(timeout=0.2)

        if not self._pending_runs:
            self._training_active = False
            self._paused = False
            self._set_button_states()
            self.status_var.set("LR: - | Best reward: - | Render: idle")

    def _handle_error(self, payload: Dict[str, Any]) -> None:
        run_id = str(payload.get("run_id", ""))
        with self._trainers_lock:
            self._pending_runs.discard(run_id)
            entry = self._active_workers.pop(run_id, None)
            if entry and entry.get("type") == "process":
                proc = entry.get("process")
                if proc is not None and proc.is_alive():
                    proc.terminate()
                    proc.join(timeout=0.2)
        self._training_active = False
        self._paused = False
        self._set_button_states()
        message = str(payload.get("message", "Unknown worker error"))
        self._last_error_message = message
        self.status_var.set(f"LR: - | Best reward: - | Render: off")
        messagebox.showerror("Training error", message)

    def _apply_plot_theme(self) -> None:
        self.figure.patch.set_facecolor("#1e1e1e")
        self.ax.set_facecolor("#252526")
        self.ax.set_xlabel("Episodes", color="#e6e6e6")
        self.ax.set_ylabel("Reward", color="#e6e6e6")
        self.ax.tick_params(colors="#d0d0d0")
        for spine in self.ax.spines.values():
            spine.set_color("#606060")
        self.ax.grid(True, color="#4a4a4a", alpha=0.35)

    def _build_run_label_snapshot(
        self,
        run_id: str,
        env_cfg: EnvironmentConfig,
        tr_cfg: TrainerConfig,
        override: Dict[str, Any],
    ) -> str:
        line1 = f"{tr_cfg.policy_name} | steps={tr_cfg.max_steps} | gamma={tr_cfg.policy_params.get('gamma', '-')}"
        line2 = f"LR={tr_cfg.lr.learning_rate} | LR strategy={tr_cfg.lr.lr_strategy} | LR decay={tr_cfg.lr.lr_decay}"
        line3 = (
            f"batch_size={tr_cfg.policy_params.get('batch_size', '-')} | "
            f"hidden_layer={tr_cfg.network.hidden_layer} | "
            f"buffer_size={tr_cfg.policy_params.get('buffer_size', '-')}"
        )
        line4 = (
            f"learning_starts={tr_cfg.policy_params.get('learning_starts', '-')} | "
            f"train_freq={tr_cfg.policy_params.get('train_freq', '-')} | "
            f"gradient_steps={tr_cfg.policy_params.get('gradient_steps', '-')}"
        )
        line5 = f"action_noise_sigma={tr_cfg.policy_params.get('action_noise_sigma', '-')}"

        shown_keys = {
            "Policy",
            "max_steps",
            "episodes",
            "gamma",
            "learning_rate",
            "lr_strategy",
            "lr_decay",
            "batch_size",
            "hidden_layer",
            "buffer_size",
            "learning_starts",
            "train_freq",
            "gradient_steps",
            "action_noise_sigma",
        }
        extras: List[str] = []
        for key, value in override.items():
            if key in shown_keys:
                continue
            extras.append(f"{key}={value}")
        if extras:
            line4 = line4 + " | " + " | ".join(extras)
        return f"{line1}\n{line2}\n{line3}\n{line4}\n{line5}"

    def _refresh_plot(self) -> None:
        self.ax.clear()
        self._apply_plot_theme()
        self._line_map.clear()

        prop_cycle = matplotlib.rcParams.get("axes.prop_cycle")
        colors = list(prop_cycle.by_key().get("color", ["#4fc3f7", "#ffb74d", "#81c784", "#e57373"]))

        for idx, (run_id, hist) in enumerate(self._run_history.items()):
            if not hist.get("visible", True):
                continue
            rewards = list(hist.get("rewards", []))
            if not rewards:
                continue
            color = colors[idx % len(colors)]
            x = list(range(1, len(rewards) + 1))
            reward_line, = self.ax.plot(x, rewards, color=color, alpha=0.30, lw=1.0, label=hist["label"])
            ma_line, = self.ax.plot(x, list(hist.get("ma", [])), color=color, alpha=1.0, lw=2.0, linestyle="-", label="moving average")
            eval_line = None
            if self.eval_rollout_var.get() and hist.get("eval"):
                ex = [p[0] for p in hist["eval"]]
                ey = [p[1] for p in hist["eval"]]
                eval_line, = self.ax.plot(ex, ey, color=color, alpha=1.0, lw=2.0, linestyle=":", marker="o", label="evaluation rollout")
            self._line_map[run_id] = {"reward": reward_line, "ma": ma_line, "eval": eval_line}

        handles, _labels = self.ax.get_legend_handles_labels()
        self._legend = None
        if handles:
            self._legend = self.ax.legend(loc="upper left", bbox_to_anchor=(1.01, self._legend_anchor_y), frameon=False)
            if self._legend is not None:
                for txt in self._legend.get_texts():
                    txt.set_color("#e6e6e6")
                    txt.set_picker(True)
                for handle in self._legend.legend_handles:
                    handle.set_picker(True)

        self.figure.subplots_adjust(left=0.04, right=0.78)
        self.canvas.draw_idle()
        self._update_legend_scroll_bounds()

    def _bind_mpl_events_once(self) -> None:
        self.figure.canvas.mpl_connect("pick_event", self._on_legend_pick)
        self.figure.canvas.mpl_connect("motion_notify_event", self._on_plot_motion)
        self.figure.canvas.mpl_connect("scroll_event", self._on_plot_scroll)

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
        direction = 1 if (getattr(event, "button", "up") == "up") else -1
        step = 0.06
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
        try:
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

            self._draw_frame(frames[idx])
            self.steps_progress["value"] = 100.0 * ((idx + 1) / max(1, len(frames)))
            try:
                fps = max(1, int(float(self.animation_fps_var.get() or "30")))
            except ValueError:
                fps = 30
            self.root.after(int(1000 / fps), lambda: self._play_frames(frames, idx + 1))
        except Exception:
            # Fail-safe: never leave playback latched in active state after callback errors.
            self._frame_playback_active = False
            if self._frame_pending is not None:
                pending = self._frame_pending
                self._frame_pending = None
                self._enqueue_frames(pending)

    def _draw_frame(self, frame: np.ndarray) -> None:
        if Image is None or ImageTk is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        arr = np.array(frame, copy=False)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, None], 3, axis=2)
        if arr.ndim != 3:
            return
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]

        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32, copy=False)
            max_v = float(np.nanmax(arr)) if arr.size else 0.0
            if max_v <= 1.0:
                arr = arr * 255.0
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

        # Only treat fully zeroed frames as invalid; low-brightness frames can be valid.
        if int(np.max(arr)) == 0 and self._last_valid_frame is not None:
            arr = self._last_valid_frame
        else:
            self._last_valid_frame = np.array(arr, copy=True)

        h, w = arr.shape[:2]
        cw = max(1, self.render_canvas.winfo_width())
        ch = max(1, self.render_canvas.winfo_height())
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))

        try:
            img = Image.fromarray(arr)
            img = img.resize((nw, nh), Image.Resampling.BILINEAR)
            photo = ImageTk.PhotoImage(img)

            self.render_canvas.delete("all")
            self.render_canvas.create_image(cw // 2, ch // 2, image=photo)
            self._frame_photo = photo
        except Exception:
            # Ignore single-frame rendering failures and allow playback loop to continue.
            return

    def on_close(self) -> None:
        self._cancel_active_trainers()
        self.root.destroy()
