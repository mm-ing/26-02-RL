"""
CliffWalking RL Workbench – UI Layer
=====================================
Tkinter GUI: config panel, environment visualisation, plot, progress bar,
training-status window.  Communicates with logic layer exclusively via EventBus.
"""

import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk

from CliffWalking_logic import (
    AlgorithmConfig,
    ACTIVATION_MAP,
    CheckpointManager,
    Event,
    EventBus,
    EventType,
    EpisodeResult,
    JobStatus,
    TrainingJob,
    TrainingManager,
)

# ---------------------------------------------------------------------------
# Colour / style constants
# ---------------------------------------------------------------------------

BG = "#0f111a"
BG_PANEL = "#161926"
BG_ENTRY = "#1c2033"
FG = "#e6e6e6"
FG_DIM = "#b5b5b5"
ACCENT = "#4cc9f0"
GRID_CLR = "#2a2f3a"
BORDER = "#2a2f3a"
SELECT_BG = "#2e3a5a"
BUTTON_BG = "#1e2740"
BUTTON_FG = "#e6e6e6"
HOVER_BG = "#283350"

PLOT_COLORS = [
    "#4cc9f0",
    "#f72585",
    "#7209b7",
    "#4361ee",
    "#90be6d",
    "#f77f00",
    "#f94144",
    "#43aa8b",
]

FONT_FAMILY = "Segoe UI"
FONT = (FONT_FAMILY, 10)
FONT_SMALL = (FONT_FAMILY, 9)
FONT_BOLD = (FONT_FAMILY, 10, "bold")
FONT_HEADER = (FONT_FAMILY, 11, "bold")

# Tuneable parameters for the UI forms
TUNEABLE_PARAMS = [
    "learning_rate",
    "gamma",
    "tau",
    "batch_size",
    "buffer_size",
    "learning_starts",
    "train_freq",
    "gradient_steps",
    "target_update_interval",
    "exploration_fraction",
    "exploration_initial_eps",
    "exploration_final_eps",
    "max_grad_norm",
]


# ---------------------------------------------------------------------------
# Style helper
# ---------------------------------------------------------------------------

def apply_theme(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=FG, font=FONT,
                     fieldbackground=BG_ENTRY, borderwidth=0)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("TLabelframe", background=BG, foreground=ACCENT, font=FONT_BOLD)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=FONT_BOLD)
    style.configure("TEntry", fieldbackground=BG_ENTRY, foreground=FG, insertcolor=FG)
    style.configure("TButton", background=BUTTON_BG, foreground=BUTTON_FG, font=FONT,
                     padding=(8, 4))
    style.map("TButton",
              background=[("active", HOVER_BG), ("pressed", ACCENT)],
              foreground=[("active", FG)])
    style.configure("Compact.TButton", padding=(4, 2), font=FONT_SMALL)
    style.map("Compact.TButton",
              background=[("active", HOVER_BG), ("pressed", ACCENT)],
              foreground=[("active", FG)])
    style.configure("TCheckbutton", background=BG, foreground=FG, font=FONT)
    style.map("TCheckbutton", background=[("active", BG)])
    style.configure("TCombobox", fieldbackground=BG_ENTRY, foreground=FG, font=FONT,
                     selectbackground=SELECT_BG, selectforeground=FG)
    style.map("TCombobox", fieldbackground=[("readonly", BG_ENTRY)])
    style.configure("TProgressbar", troughcolor=BG_PANEL, background=ACCENT)
    style.configure("TPanedwindow", background=BORDER)
    style.configure("Sash", sashthickness=5, gripcount=0)

    # Treeview
    style.configure("Treeview", background=BG, foreground=FG, fieldbackground=BG,
                     font=FONT_SMALL, rowheight=24)
    style.configure("Treeview.Heading", background=GRID_CLR, foreground=FG,
                     font=FONT_BOLD)
    style.map("Treeview",
              background=[("selected", SELECT_BG)],
              foreground=[("selected", FG)])

    root.configure(bg=BG)
    root.option_add("*TCombobox*Listbox.background", BG_ENTRY)
    root.option_add("*TCombobox*Listbox.foreground", FG)
    root.option_add("*TCombobox*Listbox.selectBackground", SELECT_BG)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Config Panel (left side)                                              ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class ConfigPanel(ttk.Frame):
    """Environment + Episode configuration panel."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._vars: Dict[str, tk.Variable] = {}
        self._build()

    # -- helpers ----------------------------------------------------------

    def _add_row2(self, parent, lbl1, key1, lbl2, key2, row, default1="", default2=""):
        """Two labelled entries on one row."""
        ttk.Label(parent, text=lbl1, font=FONT_SMALL).grid(row=row, column=0, sticky="w", padx=(4, 2), pady=1)
        v1 = tk.StringVar(value=str(default1))
        e1 = ttk.Entry(parent, textvariable=v1, width=10)
        e1.grid(row=row, column=1, sticky="ew", padx=2, pady=1)
        self._vars[key1] = v1

        ttk.Label(parent, text=lbl2, font=FONT_SMALL).grid(row=row, column=2, sticky="w", padx=(8, 2), pady=1)
        v2 = tk.StringVar(value=str(default2))
        e2 = ttk.Entry(parent, textvariable=v2, width=10)
        e2.grid(row=row, column=3, sticky="ew", padx=2, pady=1)
        self._vars[key2] = v2

    def _add_row1(self, parent, label, key, row, default="", colspan=3):
        ttk.Label(parent, text=label, font=FONT_SMALL).grid(row=row, column=0, sticky="w", padx=(4, 2), pady=1)
        v = tk.StringVar(value=str(default))
        e = ttk.Entry(parent, textvariable=v, width=22)
        e.grid(row=row, column=1, columnspan=colspan, sticky="ew", padx=2, pady=1)
        self._vars[key] = v

    # -- build ------------------------------------------------------------

    def _build(self):
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self._inner = ttk.Frame(canvas)

        self._inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self._inner, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        inner = self._inner
        inner.columnconfigure(1, weight=1)
        inner.columnconfigure(3, weight=1)

        # ---- Environment Config ----
        env_lf = ttk.LabelFrame(inner, text=" Environment Configuration ")
        env_lf.grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        env_lf.columnconfigure(1, weight=1)
        env_lf.columnconfigure(3, weight=1)

        self._vis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_lf, text="Visualisation enabled", variable=self._vis_var
                         ).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=1)

        ttk.Label(env_lf, text="Frame interval (ms)", font=FONT_SMALL).grid(
            row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self._frame_interval_var = tk.StringVar(value="10")
        ttk.Entry(env_lf, textvariable=self._frame_interval_var, width=8).grid(
            row=1, column=1, sticky="w", padx=2, pady=1)

        ttk.Label(env_lf, text="Env ID", font=FONT_SMALL).grid(
            row=2, column=0, sticky="w", padx=(4, 2), pady=1)
        self._env_id_var = tk.StringVar(value="CliffWalking-v1")
        ttk.Entry(env_lf, textvariable=self._env_id_var, width=22).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=2, pady=1)

        # ---- Episode / Algorithm Config ----
        ep_lf = ttk.LabelFrame(inner, text=" Episode Configuration ")
        ep_lf.grid(row=1, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        ep_lf.columnconfigure(1, weight=1)
        ep_lf.columnconfigure(3, weight=1)

        # Algorithm
        ttk.Label(ep_lf, text="Algorithm", font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        self._algo_var = tk.StringVar(value="VDQN")
        cb = ttk.Combobox(ep_lf, textvariable=self._algo_var, values=["VDQN", "DDQN"],
                          state="readonly", width=10)
        cb.grid(row=0, column=1, sticky="w", padx=2, pady=1)

        r = 1
        defaults = AlgorithmConfig()
        self._add_row2(ep_lf, "Episodes", "episodes", "Max Steps", "max_steps", r,
                       defaults.episodes, defaults.max_steps); r += 1
        self._add_row2(ep_lf, "LR", "learning_rate", "Gamma", "gamma", r,
                       defaults.learning_rate, defaults.gamma); r += 1
        self._add_row2(ep_lf, "Eps init", "exploration_initial_eps",
                       "Eps final", "exploration_final_eps", r,
                       defaults.exploration_initial_eps, defaults.exploration_final_eps); r += 1
        self._add_row1(ep_lf, "Expl. frac", "exploration_fraction", r,
                       defaults.exploration_fraction); r += 1
        self._add_row2(ep_lf, "Buffer", "buffer_size", "Batch", "batch_size", r,
                       defaults.buffer_size, defaults.batch_size); r += 1
        self._add_row2(ep_lf, "Learn starts", "learning_starts",
                       "Train freq", "train_freq", r,
                       defaults.learning_starts, defaults.train_freq); r += 1
        self._add_row2(ep_lf, "Grad steps", "gradient_steps",
                       "Target upd", "target_update_interval", r,
                       defaults.gradient_steps, defaults.target_update_interval); r += 1
        self._add_row2(ep_lf, "Tau", "tau", "Max grad norm", "max_grad_norm", r,
                       defaults.tau, defaults.max_grad_norm); r += 1
        self._add_row1(ep_lf, "Hidden layers", "hidden_layers", r,
                       ",".join(str(x) for x in defaults.hidden_layers)); r += 1

        ttk.Label(ep_lf, text="Activation", font=FONT_SMALL).grid(
            row=r, column=0, sticky="w", padx=(4, 2), pady=1)
        self._act_var = tk.StringVar(value=defaults.activation)
        ttk.Combobox(ep_lf, textvariable=self._act_var,
                     values=list(ACTIVATION_MAP.keys()),
                     state="readonly", width=12).grid(row=r, column=1, sticky="w", padx=2, pady=1)
        r += 1

        self._add_row1(ep_lf, "Mov.Avg win", "moving_avg_window", r,
                       defaults.moving_avg_window); r += 1

        # ---- Compare / Tuning ----
        mode_lf = ttk.LabelFrame(inner, text=" Mode ")
        mode_lf.grid(row=2, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        mode_lf.columnconfigure(1, weight=1)
        mode_lf.columnconfigure(3, weight=1)

        self._compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_lf, text="Compare Methods", variable=self._compare_var,
                         command=self._on_mode_change).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=1)

        self._tuning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_lf, text="Parameter Tuning", variable=self._tuning_var,
                         command=self._on_mode_change).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=4, pady=1)

        self._tune_frame = ttk.Frame(mode_lf)
        self._tune_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        self._tune_frame.columnconfigure(1, weight=1)
        self._tune_frame.columnconfigure(3, weight=1)

        ttk.Label(self._tune_frame, text="Parameter", font=FONT_SMALL).grid(
            row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        self._tune_param_var = tk.StringVar(value=TUNEABLE_PARAMS[0])
        ttk.Combobox(self._tune_frame, textvariable=self._tune_param_var,
                     values=TUNEABLE_PARAMS, state="readonly", width=18).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=2, pady=1)

        self._tune_min_var = tk.StringVar(value="0.0001")
        self._tune_max_var = tk.StringVar(value="0.01")
        self._tune_step_var = tk.StringVar(value="0.002")
        ttk.Label(self._tune_frame, text="Min", font=FONT_SMALL).grid(row=1, column=0, sticky="w", padx=(4,2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_min_var, width=8).grid(row=1, column=1, sticky="ew", padx=2, pady=1)
        ttk.Label(self._tune_frame, text="Max", font=FONT_SMALL).grid(row=1, column=2, sticky="w", padx=(8,2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_max_var, width=8).grid(row=1, column=3, sticky="ew", padx=2, pady=1)
        ttk.Label(self._tune_frame, text="Step", font=FONT_SMALL).grid(row=2, column=0, sticky="w", padx=(4,2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_step_var, width=8).grid(row=2, column=1, sticky="ew", padx=2, pady=1)

        self._on_mode_change()

    def _on_mode_change(self):
        if self._tuning_var.get():
            self._tune_frame.grid()
        else:
            self._tune_frame.grid_remove()

    # -- read config -------------------------------------------------------

    def get_config(self) -> AlgorithmConfig:
        def _f(key, default=0.0):
            try:
                return float(self._vars[key].get())
            except (ValueError, KeyError):
                return float(default)

        def _i(key, default=0):
            try:
                return int(float(self._vars[key].get()))
            except (ValueError, KeyError):
                return int(default)

        hl_str = self._vars.get("hidden_layers", tk.StringVar(value="128,128")).get()
        try:
            hidden = [int(x.strip()) for x in hl_str.split(",") if x.strip()]
        except ValueError:
            hidden = [128, 128]

        return AlgorithmConfig(
            algorithm=self._algo_var.get(),
            learning_rate=_f("learning_rate", 1e-3),
            buffer_size=_i("buffer_size", 50000),
            learning_starts=_i("learning_starts", 500),
            batch_size=_i("batch_size", 64),
            tau=_f("tau", 1.0),
            gamma=_f("gamma", 0.99),
            train_freq=_i("train_freq", 4),
            gradient_steps=_i("gradient_steps", 1),
            target_update_interval=_i("target_update_interval", 500),
            exploration_fraction=_f("exploration_fraction", 0.3),
            exploration_initial_eps=_f("exploration_initial_eps", 1.0),
            exploration_final_eps=_f("exploration_final_eps", 0.05),
            hidden_layers=hidden,
            activation=self._act_var.get(),
            max_grad_norm=_f("max_grad_norm", 10.0),
            episodes=_i("episodes", 300),
            max_steps=_i("max_steps", 200),
            moving_avg_window=_i("moving_avg_window", 20),
        )

    @property
    def visualization_enabled(self) -> bool:
        return self._vis_var.get()

    @property
    def frame_interval_ms(self) -> int:
        try:
            return max(1, int(self._frame_interval_var.get()))
        except ValueError:
            return 10

    @property
    def compare_mode(self) -> bool:
        return self._compare_var.get()

    @property
    def tuning_mode(self) -> bool:
        return self._tuning_var.get()

    @property
    def tune_params(self):
        return (
            self._tune_param_var.get(),
            float(self._tune_min_var.get() or 0),
            float(self._tune_max_var.get() or 0),
            float(self._tune_step_var.get() or 1),
        )


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Visualisation Panel (right side – renders gym frames)                 ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class VisualizationPanel(ttk.Frame):
    """Displays the latest rendered frame from the environment."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._photo: Optional[ImageTk.PhotoImage] = None
        self._canvas.bind("<Configure>", self._on_resize)
        self._frame: Optional[np.ndarray] = None

    def _on_resize(self, _event=None):
        if self._frame is not None:
            self._render(self._frame)

    def update_frame(self, frame: Optional[np.ndarray]):
        if frame is None:
            return
        self._frame = frame
        self._render(frame)

    def _render(self, frame: np.ndarray):
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img = Image.fromarray(frame)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        img = img.resize((new_w, new_h), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Plot Panel (bottom – matplotlib)                                      ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class PlotPanel(ttk.Frame):
    """Matplotlib plot: raw returns + moving average per job."""

    def __init__(self, parent, **kw):
        super().__init__(parent, **kw)
        self._fig, self._ax = plt.subplots(figsize=(6, 2.5), dpi=100)
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(BG)
        self._ax.tick_params(colors=FG_DIM)
        self._ax.set_xlabel("Episode", color=FG_DIM, fontsize=9)
        self._ax.set_ylabel("Return", color=FG_DIM, fontsize=9)
        self._ax.grid(color=GRID_CLR, linestyle="--", alpha=0.5)
        for spine in self._ax.spines.values():
            spine.set_color(GRID_CLR)
        self._fig.tight_layout(pad=1.0)

        self._canvas_widget = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas_widget.get_tk_widget().pack(fill="both", expand=True)
        self._last_redraw = 0.0

    def redraw(self, jobs: List[TrainingJob], force: bool = False):
        now = time.time()
        if not force and (now - self._last_redraw) < 0.08:
            return  # throttle to ~12 Hz
        self._last_redraw = now

        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlabel("Episode", color=FG_DIM, fontsize=9)
        ax.set_ylabel("Return", color=FG_DIM, fontsize=9)
        ax.tick_params(colors=FG_DIM)
        ax.grid(color=GRID_CLR, linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)

        legend_entries = 0
        for idx, job in enumerate(jobs):
            if not job.visible or not job.episode_returns:
                continue
            color = PLOT_COLORS[idx % len(PLOT_COLORS)]
            x = list(range(1, len(job.episode_returns) + 1))
            # Raw
            ax.plot(x, job.episode_returns, color=color, alpha=0.35, linewidth=1.0)
            # Moving avg
            w = job.config.moving_avg_window
            if len(job.episode_returns) >= 2:
                ma = self._moving_avg(job.episode_returns, w)
                ax.plot(x, ma, color=color, alpha=1.0, linewidth=2.5, label=job.name)
            else:
                ax.plot([], [], color=color, alpha=1.0, linewidth=2.5, label=job.name)
            legend_entries += 1

        if legend_entries:
            loc = "lower left" if any(len(j.episode_returns) > 4 for j in jobs if j.visible) else "upper right"
            leg = ax.legend(loc=loc, facecolor=BG, edgecolor=GRID_CLR, labelcolor=FG,
                            fontsize=8, framealpha=0.9)

        try:
            self._fig.tight_layout(pad=1.0)
            self._canvas_widget.draw_idle()
        except Exception:
            pass

    @staticmethod
    def _moving_avg(data: List[float], window: int) -> List[float]:
        out = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            out.append(float(np.mean(data[start:i + 1])))
        return out

    def save_plot(self, path: str):
        self._fig.savefig(path, facecolor=BG, dpi=150)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Training Status Window                                                ║
# ╚═════════════════════════════════════════════════════════════════════════╝

import time  # already imported, repeated for clarity in section

class StatusWindow:
    """Toplevel window with Treeview table for TrainingJobs."""

    COLUMNS = ("algorithm", "episode", "return", "movavg", "epsilon", "loss",
               "duration", "steps", "visible")

    def __init__(self, parent: tk.Tk, manager: TrainingManager, event_bus: EventBus):
        self._parent = parent
        self._manager = manager
        self._event_bus = event_bus

        self._win: Optional[tk.Toplevel] = None
        self._tree: Optional[ttk.Treeview] = None
        self._last_update: Dict[str, float] = {}  # rate-limit per job

    def show(self):
        if self._win is not None and self._win.winfo_exists():
            self._win.lift()
            return
        self._win = tk.Toplevel(self._parent)
        self._win.title("Training Status")
        self._win.geometry("900x400")
        self._win.configure(bg=BG)

        # Treeview
        cols = self.COLUMNS
        self._tree = ttk.Treeview(self._win, columns=cols, show="headings", selectmode="browse")
        for c in cols:
            self._tree.heading(c, text=c.capitalize(), command=lambda _c=c: self._sort(_c))
            self._tree.column(c, width=90, anchor="center")
        self._tree.column("algorithm", width=160, anchor="w", stretch=True)
        self._tree.pack(fill="both", expand=True, padx=4, pady=4)

        # Double-click → toggle visibility
        self._tree.bind("<Double-1>", self._on_double_click)
        # Context menu
        self._tree.bind("<Button-3>", self._on_right_click)
        # Keyboard
        self._tree.bind("<Return>", lambda e: self._toggle_selected())
        self._tree.bind("<space>", lambda e: self._pause_resume_selected())

        # Buttons
        btn_frame = ttk.Frame(self._win)
        btn_frame.pack(fill="x", padx=4, pady=(0, 4))
        for text, cmd in [
            ("Toggle Visible", self._toggle_selected),
            ("Train", self._train_selected),
            ("Run", self._run_selected),
            ("Stop", self._stop_selected),
            ("Remove", self._remove_selected),
        ]:
            ttk.Button(btn_frame, text=text, command=cmd).pack(side="left", padx=2)

        self._refresh_all()

    # -- Treeview helpers -------------------------------------------------

    def _job_id_for_item(self, item: str) -> Optional[str]:
        return item if item in self._manager.jobs else None

    def _selected_job_id(self) -> Optional[str]:
        sel = self._tree.selection() if self._tree else ()
        return sel[0] if sel else None

    def _refresh_all(self):
        if self._tree is None:
            return
        existing = set(self._tree.get_children())
        current = {j.job_id for j in self._manager.job_list()}
        # Remove stale
        for iid in existing - current:
            self._tree.delete(iid)
        # Insert / update
        for job in self._manager.job_list():
            vals = self._job_values(job)
            if job.job_id in existing:
                self._tree.item(job.job_id, values=vals)
            else:
                self._tree.insert("", "end", iid=job.job_id, values=vals)

    def _job_values(self, job: TrainingJob):
        ep_done = job.total_episodes_done
        ep_total = job.config.episodes
        ret = f"{job.episode_returns[-1]:.1f}" if job.episode_returns else "-"
        ma = f"{job.moving_avg:.1f}" if job.episode_returns else "-"
        eps = f"{job.episode_epsilons[-1]:.4f}" if job.episode_epsilons else "-"
        loss = f"{job.episode_losses[-1]:.4f}" if job.episode_losses else "-"
        dur = f"{job.episode_durations[-1]:.3f}s" if job.episode_durations else "-"
        steps = str(job.episode_lengths[-1]) if job.episode_lengths else "-"
        vis = "Yes" if job.visible else "No"
        return (job.name, f"{ep_done}/{ep_total}", ret, ma, eps, loss, dur, steps, vis)

    def update_job(self, job_id: str):
        """Update single row (rate-limited to ~20 Hz)."""
        now = time.time()
        if now - self._last_update.get(job_id, 0) < 0.05:
            return
        self._last_update[job_id] = now
        if self._tree is None or self._win is None or not self._win.winfo_exists():
            return
        job = self._manager.get_job(job_id)
        if job is None:
            return
        vals = self._job_values(job)
        if job.job_id in set(self._tree.get_children()):
            self._tree.item(job.job_id, values=vals)
        else:
            self._tree.insert("", "end", iid=job.job_id, values=vals)

    # -- actions ----------------------------------------------------------

    def _toggle_selected(self):
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job:
                job.visible = not job.visible
                self.update_job(jid)

    def _train_selected(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.start_job(jid)

    def _run_selected(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.run_job(jid)

    def _stop_selected(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.cancel_job(jid)

    def _remove_selected(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.remove_job(jid)
            if self._tree and jid in set(self._tree.get_children()):
                self._tree.delete(jid)

    def _pause_resume_selected(self):
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job:
                if job.status == JobStatus.RUNNING:
                    self._manager.pause_job(jid)
                elif job.status == JobStatus.PAUSED:
                    self._manager.resume_job(jid)
                self.update_job(jid)

    # -- double-click / context menu --------------------------------------

    def _on_double_click(self, _event):
        self._toggle_selected()

    def _on_right_click(self, event):
        item = self._tree.identify_row(event.y) if self._tree else None
        if item:
            self._tree.selection_set(item)
            menu = tk.Menu(self._win, tearoff=0, bg=BG_PANEL, fg=FG,
                           activebackground=SELECT_BG, activeforeground=FG)
            menu.add_command(label="Toggle Visibility", command=self._toggle_selected)
            menu.add_command(label="Train", command=self._train_selected)
            menu.add_command(label="Run", command=self._run_selected)
            menu.add_command(label="Stop", command=self._stop_selected)
            menu.add_command(label="Remove", command=self._remove_selected)
            menu.tk_popup(event.x_root, event.y_root)

    def _sort(self, col):
        if self._tree is None:
            return
        items = list(self._tree.get_children())
        idx = self.COLUMNS.index(col)
        items.sort(key=lambda i: self._tree.item(i, "values")[idx])
        for i, iid in enumerate(items):
            self._tree.move(iid, "", i)


# ╔═════════════════════════════════════════════════════════════════════════╗
# ║  Main Workbench UI                                                     ║
# ╚═════════════════════════════════════════════════════════════════════════╝

class WorkbenchUI:
    """Top-level application window."""

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("CliffWalking RL Workbench")
        root.geometry("1280x860")
        root.minsize(800, 600)
        apply_theme(root)

        # Core logic
        self.event_bus = EventBus()
        self.manager = TrainingManager(self.event_bus)
        self.event_bus.subscribe(self._on_event)

        # Status window (created lazily)
        self._status_win = StatusWindow(root, self.manager, self.event_bus)

        # Resize debounce
        self._resize_after_id: Optional[str] = None
        self._current_btn_style = "TButton"

        # --- Layout: vertical PanedWindow ---
        self._vpw = ttk.PanedWindow(root, orient="vertical")
        self._vpw.pack(fill="both", expand=True)

        # Top pane (horizontal split)
        top_frame = ttk.Frame(self._vpw)
        self._vpw.add(top_frame, weight=2)

        self._hpw = ttk.PanedWindow(top_frame, orient="horizontal")
        self._hpw.pack(fill="both", expand=True)

        # Config panel (left)
        self.config_panel = ConfigPanel(self._hpw)
        self._hpw.add(self.config_panel, weight=1)

        # Visualisation panel (right)
        self.vis_panel = VisualizationPanel(self._hpw)
        self._hpw.add(self.vis_panel, weight=2)

        # Bottom pane
        bottom_frame = ttk.Frame(self._vpw)
        self._vpw.add(bottom_frame, weight=1)

        # Progress bar
        self._progress_var = tk.DoubleVar(value=0)
        self._pbar = ttk.Progressbar(bottom_frame, variable=self._progress_var,
                                      maximum=100, mode="determinate")
        self._pbar.pack(fill="x", padx=6, pady=(4, 2))

        # Buttons
        btn_frame = ttk.Frame(bottom_frame)
        btn_frame.pack(fill="x", padx=6, pady=2)

        self._buttons: Dict[str, ttk.Button] = {}
        for text, cmd in [
            ("Add Job", self._on_add_job),
            ("Train", self._on_train),
            ("Status", self._on_status),
            ("Save Plot", self._on_save_plot),
            ("Cancel", self._on_cancel),
            ("Save", self._on_save),
            ("Load", self._on_load),
        ]:
            b = ttk.Button(btn_frame, text=text, command=cmd)
            b.pack(side="left", padx=2, expand=True, fill="x")
            self._buttons[text] = b

        # Plot
        self.plot_panel = PlotPanel(bottom_frame)
        self.plot_panel.pack(fill="both", expand=True, padx=6, pady=(2, 4))

        # --- periodic UI refresh ---
        self._poll_interval = 10  # ms
        self._schedule_poll()

        # Resize handler
        root.bind("<Configure>", self._on_configure)

    # -- periodic poll -----------------------------------------------------

    def _schedule_poll(self):
        self.event_bus.process_events()
        self._update_vis()
        self._update_progress()
        self.root.after(self._poll_interval, self._schedule_poll)

    def _update_vis(self):
        """Show latest frame from selected / first running job."""
        if not self.config_panel.visualization_enabled:
            return
        sel_id = self._status_win._selected_job_id() if (
            self._status_win._win and self._status_win._win.winfo_exists()
        ) else None
        job = None
        if sel_id:
            job = self.manager.get_job(sel_id)
            if job and not job.is_alive():
                job = None
        if job is None:
            for j in self.manager.job_list():
                if j.is_alive():
                    job = j
                    break
        if job:
            frame = job.get_latest_frame()
            self.vis_panel.update_frame(frame)

    def _update_progress(self):
        jobs = self.manager.job_list()
        if not jobs:
            self._progress_var.set(0)
            return
        total = sum(j.config.episodes for j in jobs)
        done = sum(j.total_episodes_done for j in jobs)
        if total > 0:
            self._progress_var.set(done / total * 100)

    # -- event handler -----------------------------------------------------

    def _on_event(self, event: Event):
        """Called from EventBus.process_events (UI thread)."""
        if event.type == EventType.EPISODE_COMPLETED:
            jid = event.data["job_id"]
            result: EpisodeResult = event.data["result"]
            job = self.manager.get_job(jid)
            if job:
                job.record_episode(result)
                # Sync vis flag
                job.visualization_enabled = self.config_panel.visualization_enabled
                job.render_interval = self.config_panel.frame_interval_ms / 1000.0
            self._status_win.update_job(jid)
            # Redraw plot (throttled inside)
            visible_jobs = [j for j in self.manager.job_list()]
            self.plot_panel.redraw(visible_jobs)

        elif event.type == EventType.TRAINING_DONE:
            jid = event.data["job_id"]
            self._status_win.update_job(jid)
            self.plot_panel.redraw(self.manager.job_list(), force=True)

        elif event.type == EventType.JOB_CREATED:
            self._status_win._refresh_all()

        elif event.type == EventType.JOB_STATE_CHANGED:
            jid = event.data.get("job_id")
            if jid:
                self._status_win.update_job(jid)

        elif event.type == EventType.ERROR:
            msg = event.data.get("error", "Unknown error")
            messagebox.showerror("Error", msg)

    # -- button commands ---------------------------------------------------

    def _on_add_job(self):
        cfg = self.config_panel.get_config()
        if self.config_panel.compare_mode:
            self.manager.add_compare_jobs(cfg)
        elif self.config_panel.tuning_mode:
            param, mn, mx, st = self.config_panel.tune_params
            self.manager.add_tuning_jobs(cfg, param, mn, mx, st)
        else:
            self.manager.add_job(cfg)
        self._status_win._refresh_all()

    def _on_train(self):
        # If no pending jobs exist, auto-create one from current config
        pending = [j for j in self.manager.job_list()
                   if j.status in (JobStatus.PENDING, JobStatus.COMPLETED,
                                   JobStatus.CANCELLED)]
        if not pending:
            self._on_add_job()
        self.manager.start_all_pending()

    def _on_status(self):
        self._status_win.show()

    def _on_save_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png",
                                             filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self.plot_panel.save_plot(path)

    def _on_cancel(self):
        self.manager.cancel_all()

    def _on_save(self):
        d = filedialog.askdirectory(title="Save jobs to…")
        if d:
            CheckpointManager.save_all(self.manager.job_list(), d)

    def _on_load(self):
        d = filedialog.askdirectory(title="Load jobs from…")
        if d:
            loaded = CheckpointManager.load_all(d)
            for job in loaded:
                self.manager.jobs[job.job_id] = job
                self.event_bus.publish(Event(EventType.JOB_CREATED, {"job_id": job.job_id}))
            self.plot_panel.redraw(self.manager.job_list(), force=True)

    # -- resize debounce ---------------------------------------------------

    def _on_configure(self, event):
        if event.widget != self.root:
            return
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self._apply_resize)

    def _apply_resize(self):
        self._resize_after_id = None
        w = self.root.winfo_width()
        # Button style hysteresis
        target = "TButton" if w >= 1100 else "Compact.TButton"
        if target != self._current_btn_style:
            self._current_btn_style = target
            for b in self._buttons.values():
                b.configure(style=target)
        # Throttled plot redraw
        self.plot_panel.redraw(self.manager.job_list(), force=True)
