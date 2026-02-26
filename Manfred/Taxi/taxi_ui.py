"""Taxi RL Workbench – UI Layer."""

from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import matplotlib
import numpy as np
from PIL import Image, ImageTk

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from taxi_logic import (
    ACTIVATION_MAP,
    AlgorithmConfig,
    CheckpointManager,
    Event,
    EventBus,
    EventType,
    EpisodeResult,
    JobStatus,
    TrainingJob,
    TrainingManager,
)


BG = "#0f111a"
BG_PANEL = "#161926"
BG_ENTRY = "#1c2033"
FG = "#e6e6e6"
FG_DIM = "#b5b5b5"
ACCENT = "#4cc9f0"
GRID_CLR = "#2a2f3a"
SELECT_BG = "#2e3a5a"
BUTTON_BG = "#1e2740"
HOVER_BG = "#283350"

PLOT_COLORS = ["#4cc9f0", "#f72585", "#7209b7", "#4361ee", "#90be6d", "#f77f00", "#f94144"]
FONT = ("Segoe UI", 10)
FONT_SMALL = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 10, "bold")

ALGORITHMS = ["VDQN", "DDQN", "Dueling DQN", "Prioritized DQN"]
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
    "prioritized_alpha",
    "prioritized_beta",
]


def apply_theme(root: tk.Tk):
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background=BG, foreground=FG, font=FONT, fieldbackground=BG_ENTRY, borderwidth=0)
    style.configure("TFrame", background=BG)
    style.configure("TLabel", background=BG, foreground=FG, font=FONT)
    style.configure("TLabelframe", background=BG, foreground=ACCENT, font=FONT_BOLD)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=FONT_BOLD)
    style.configure("TEntry", fieldbackground=BG_ENTRY, foreground=FG, insertcolor=FG)
    style.configure("TButton", background=BUTTON_BG, foreground=FG, padding=(8, 4))
    style.map("TButton", background=[("active", HOVER_BG), ("pressed", ACCENT)])
    style.configure("Compact.TButton", padding=(4, 2), font=FONT_SMALL)
    style.map("Compact.TButton", background=[("active", HOVER_BG), ("pressed", ACCENT)])
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TCombobox", fieldbackground=BG_ENTRY, foreground=FG, selectbackground=SELECT_BG)
    style.configure("TProgressbar", troughcolor=BG_PANEL, background=ACCENT)
    style.configure("Treeview", background=BG, foreground=FG, fieldbackground=BG, rowheight=24)
    style.configure("Treeview.Heading", background=GRID_CLR, foreground=FG, font=FONT_BOLD)
    style.map("Treeview", background=[("selected", SELECT_BG)], foreground=[("selected", FG)])
    root.configure(bg=BG)


class ConfigPanel(ttk.Frame):
    def __init__(self, parent, on_apply_env=None, **kwargs):
        super().__init__(parent, **kwargs)
        self._vars: Dict[str, tk.Variable] = {}
        self._on_apply_env = on_apply_env
        self._build()

    def _add_row2(self, parent, l1, k1, l2, k2, row, d1="", d2=""):
        ttk.Label(parent, text=l1, font=FONT_SMALL).grid(row=row, column=0, sticky="w", padx=(4, 2), pady=1)
        v1 = tk.StringVar(value=str(d1))
        ttk.Entry(parent, textvariable=v1, width=10).grid(row=row, column=1, sticky="ew", padx=2, pady=1)
        self._vars[k1] = v1
        ttk.Label(parent, text=l2, font=FONT_SMALL).grid(row=row, column=2, sticky="w", padx=(8, 2), pady=1)
        v2 = tk.StringVar(value=str(d2))
        ttk.Entry(parent, textvariable=v2, width=10).grid(row=row, column=3, sticky="ew", padx=2, pady=1)
        self._vars[k2] = v2

    def _add_row1(self, parent, label, key, row, default="", colspan=3):
        ttk.Label(parent, text=label, font=FONT_SMALL).grid(row=row, column=0, sticky="w", padx=(4, 2), pady=1)
        v = tk.StringVar(value=str(default))
        ttk.Entry(parent, textvariable=v).grid(row=row, column=1, columnspan=colspan, sticky="ew", padx=2, pady=1)
        self._vars[key] = v

    def _build(self):
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        scroll = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=scroll.set)
        canvas.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"))

        env_lf = ttk.LabelFrame(inner, text=" Environment Configuration ")
        env_lf.grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        for col in (1, 3):
            env_lf.columnconfigure(col, weight=1)

        self._vis_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_lf, text="Visualisierung aktivieren", variable=self._vis_var).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=4, pady=1
        )

        ttk.Label(env_lf, text="Env", font=FONT_SMALL).grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self._env_var = tk.StringVar(value="Taxi-v3")
        ttk.Entry(env_lf, textvariable=self._env_var).grid(row=1, column=1, columnspan=3, sticky="ew", padx=2, pady=1)

        self._raining_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(env_lf, text="is_raining", variable=self._raining_var).grid(
            row=2, column=0, sticky="w", padx=(4, 2), pady=1
        )

        self._fickle_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(env_lf, text="fickle_passenger", variable=self._fickle_var).grid(
            row=2, column=2, sticky="w", padx=(8, 2), pady=1
        )

        ttk.Label(env_lf, text="Frame-Intervall (ms)", font=FONT_SMALL).grid(row=3, column=2, sticky="w", padx=(8, 2), pady=1)
        self._frame_var = tk.StringVar(value="10")
        ttk.Entry(env_lf, textvariable=self._frame_var, width=8).grid(row=3, column=3, sticky="w", padx=2, pady=1)

        ttk.Label(env_lf, text="Notiz", font=FONT_SMALL).grid(row=4, column=0, sticky="w", padx=(4, 2), pady=1)
        self._note_var = tk.StringVar(value="")
        ttk.Entry(env_lf, textvariable=self._note_var).grid(row=4, column=1, columnspan=3, sticky="ew", padx=2, pady=1)

        ttk.Button(env_lf, text="Apply and reset", command=self._apply_env).grid(row=5, column=0, columnspan=4, sticky="ew", padx=4, pady=(3, 2))

        ep_lf = ttk.LabelFrame(inner, text=" Episode Configuration ")
        ep_lf.grid(row=1, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        for col in (1, 3):
            ep_lf.columnconfigure(col, weight=1)

        ttk.Label(ep_lf, text="Algorithm", font=FONT_SMALL).grid(row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        self._algo_var = tk.StringVar(value=ALGORITHMS[0])
        algo_cb = ttk.Combobox(ep_lf, textvariable=self._algo_var, values=ALGORITHMS, state="readonly", width=18)
        algo_cb.grid(row=0, column=1, columnspan=3, sticky="ew", padx=2, pady=1)
        algo_cb.bind("<<ComboboxSelected>>", lambda _e: self._on_algo_change())
        self._algo_var.trace_add("write", lambda *_args: self._on_algo_change())

        defaults = AlgorithmConfig()
        r = 1
        self._add_row2(ep_lf, "Episodes", "episodes", "Max-Steps", "max_steps", r, defaults.episodes, defaults.max_steps); r += 1
        self._add_row2(ep_lf, "Alpha/LR", "learning_rate", "Gamma", "gamma", r, defaults.learning_rate, defaults.gamma); r += 1
        self._add_row2(ep_lf, "Eps min", "exploration_final_eps", "Eps max", "exploration_initial_eps", r,
                       defaults.exploration_final_eps, defaults.exploration_initial_eps); r += 1
        self._add_row1(ep_lf, "Expl. frac", "exploration_fraction", r, defaults.exploration_fraction); r += 1
        self._add_row2(ep_lf, "Buffer", "buffer_size", "Batch", "batch_size", r, defaults.buffer_size, defaults.batch_size); r += 1
        self._add_row2(ep_lf, "Learn starts", "learning_starts", "Train freq", "train_freq", r,
                       defaults.learning_starts, defaults.train_freq); r += 1
        self._add_row2(ep_lf, "Grad steps", "gradient_steps", "Target upd", "target_update_interval", r,
                       defaults.gradient_steps, defaults.target_update_interval); r += 1
        self._add_row2(ep_lf, "Tau", "tau", "Max grad", "max_grad_norm", r, defaults.tau, defaults.max_grad_norm); r += 1
        self._add_row1(ep_lf, "Hidden-Layer (csv)", "hidden_layers", r, "128,128"); r += 1

        ttk.Label(ep_lf, text="Activation", font=FONT_SMALL).grid(row=r, column=0, sticky="w", padx=(4, 2), pady=1)
        self._act_var = tk.StringVar(value=defaults.network.activation)
        ttk.Combobox(ep_lf, textvariable=self._act_var, values=list(ACTIVATION_MAP.keys()), state="readonly").grid(
            row=r, column=1, sticky="w", padx=2, pady=1
        )
        r += 1

        self._prio_frame = ttk.Frame(ep_lf)
        self._prio_frame.grid(row=r, column=0, columnspan=4, sticky="ew", padx=0, pady=0)
        self._prio_frame.columnconfigure(1, weight=1)
        self._prio_frame.columnconfigure(3, weight=1)
        self._add_row2(self._prio_frame, "Prio alpha", "prioritized_alpha", "Prio beta", "prioritized_beta", 0,
                       defaults.prioritized_alpha, defaults.prioritized_beta)
        r += 1

        self._add_row1(ep_lf, "Mov.Avg win", "moving_avg_window", r, defaults.moving_avg_window)

        mode_lf = ttk.LabelFrame(inner, text=" Mode ")
        mode_lf.grid(row=2, column=0, columnspan=4, sticky="ew", padx=4, pady=(4, 2))
        for col in (1, 3):
            mode_lf.columnconfigure(col, weight=1)

        self._compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_lf, text="Compare Methods", variable=self._compare_var, command=self._on_compare_toggle).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=1
        )
        self._tuning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(mode_lf, text="Parameter Tuning", variable=self._tuning_var, command=self._on_tuning_toggle).grid(
            row=1, column=0, columnspan=4, sticky="w", padx=4, pady=1
        )

        self._tune_frame = ttk.Frame(mode_lf)
        self._tune_frame.grid(row=2, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        self._tune_frame.columnconfigure(1, weight=1)
        self._tune_frame.columnconfigure(3, weight=1)

        ttk.Label(self._tune_frame, text="Parameter", font=FONT_SMALL).grid(row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        self._tune_param_var = tk.StringVar(value=TUNEABLE_PARAMS[0])
        ttk.Combobox(self._tune_frame, textvariable=self._tune_param_var, values=TUNEABLE_PARAMS, state="readonly").grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=2, pady=1
        )
        self._tune_min_var = tk.StringVar(value="0.0001")
        self._tune_max_var = tk.StringVar(value="0.01")
        self._tune_step_var = tk.StringVar(value="0.002")
        ttk.Label(self._tune_frame, text="Min", font=FONT_SMALL).grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_min_var, width=8).grid(row=1, column=1, sticky="ew", padx=2, pady=1)
        ttk.Label(self._tune_frame, text="Max", font=FONT_SMALL).grid(row=1, column=2, sticky="w", padx=(8, 2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_max_var, width=8).grid(row=1, column=3, sticky="ew", padx=2, pady=1)
        ttk.Label(self._tune_frame, text="Step", font=FONT_SMALL).grid(row=2, column=0, sticky="w", padx=(4, 2), pady=1)
        ttk.Entry(self._tune_frame, textvariable=self._tune_step_var, width=8).grid(row=2, column=1, sticky="ew", padx=2, pady=1)

        self._on_mode_change()
        self._on_algo_change()

    def _apply_env(self):
        if self._on_apply_env:
            self._on_apply_env()

    def _on_mode_change(self):
        if self._tuning_var.get():
            self._tune_frame.grid()
        else:
            self._tune_frame.grid_remove()

    def _on_compare_toggle(self):
        if self._compare_var.get():
            self._tuning_var.set(False)
        self._on_mode_change()

    def _on_tuning_toggle(self):
        if self._tuning_var.get():
            self._compare_var.set(False)
        self._on_mode_change()

    def _on_algo_change(self):
        algo = (self._algo_var.get() or "").strip()
        if algo not in ALGORITHMS:
            self._algo_var.set(ALGORITHMS[0])
            algo = ALGORITHMS[0]
        if algo == "Prioritized DQN":
            self._prio_frame.grid()
        else:
            self._prio_frame.grid_remove()

    def get_config(self) -> AlgorithmConfig:
        def _f(key, default=0.0):
            try:
                return float(self._vars[key].get())
            except Exception:
                return float(default)

        def _i(key, default=0):
            try:
                return int(float(self._vars[key].get()))
            except Exception:
                return int(default)

        try:
            hidden_layers = [int(x.strip()) for x in self._vars["hidden_layers"].get().split(",") if x.strip()]
        except Exception:
            hidden_layers = [128, 128]

        cfg = AlgorithmConfig(
            algorithm=self._algo_var.get(),
            learning_rate=_f("learning_rate", 5e-4),
            buffer_size=_i("buffer_size", 5000),
            learning_starts=_i("learning_starts", 64),
            batch_size=_i("batch_size", 64),
            tau=_f("tau", 1.0),
            gamma=_f("gamma", 0.99),
            train_freq=_i("train_freq", 1),
            gradient_steps=_i("gradient_steps", 1),
            target_update_interval=_i("target_update_interval", 200),
            exploration_fraction=_f("exploration_fraction", 0.8),
            exploration_initial_eps=_f("exploration_initial_eps", 1.0),
            exploration_final_eps=_f("exploration_final_eps", 0.02),
            max_grad_norm=_f("max_grad_norm", 10.0),
            prioritized_alpha=_f("prioritized_alpha", 0.6),
            prioritized_beta=_f("prioritized_beta", 0.4),
            episodes=_i("episodes", 1000),
            max_steps=_i("max_steps", 80),
            moving_avg_window=_i("moving_avg_window", 50),
            env_name=self._env_var.get() or "Taxi-v3",
            is_raining=self._raining_var.get(),
            fickle_passenger=self._fickle_var.get(),
        )
        cfg.network.hidden_layers = hidden_layers
        cfg.network.activation = self._act_var.get()
        return cfg

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

    @property
    def visualization_enabled(self) -> bool:
        return self._vis_var.get()

    @property
    def frame_interval_ms(self) -> int:
        try:
            return max(1, int(self._frame_var.get()))
        except Exception:
            return 10

class VisualizationPanel(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._frame: Optional[np.ndarray] = None
        self._photo = None
        self._canvas.bind("<Configure>", lambda _e: self._render(self._frame) if self._frame is not None else None)

    def update_frame(self, frame: Optional[np.ndarray]):
        if frame is None:
            return
        self._frame = frame
        self._render(frame)

    def _render(self, frame: np.ndarray):
        cw, ch = self._canvas.winfo_width(), self._canvas.winfo_height()
        if cw < 10 or ch < 10:
            return
        img = Image.fromarray(frame)
        iw, ih = img.size
        scale = min(cw / iw, ch / ih)
        nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
        img = img.resize((nw, nh), Image.NEAREST)
        self._photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2, image=self._photo, anchor="center")


class PlotPanel(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._fig, self._ax = plt.subplots(figsize=(7, 3), dpi=100)
        self._fig.patch.set_facecolor(BG)
        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True)
        self._last_redraw = 0.0

    def redraw(self, jobs: List[TrainingJob], force: bool = False):
        now = matplotlib.dates.date2num(matplotlib.dates.datetime.datetime.now())
        if not force and (now - self._last_redraw) < (1 / (24 * 3600 * 12)):
            return
        self._last_redraw = now
        ax = self._ax
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlabel("Episodes", color=FG_DIM)
        ax.set_ylabel("Return", color=FG_DIM)
        ax.tick_params(colors=FG_DIM)
        ax.grid(color=GRID_CLR, linestyle="--", alpha=0.5)
        for spine in ax.spines.values():
            spine.set_color(GRID_CLR)

        visible = [j for j in jobs if j.visible and j.episode_returns]
        for i, job in enumerate(visible):
            color = PLOT_COLORS[i % len(PLOT_COLORS)]
            x = list(range(1, len(job.episode_returns) + 1))
            ax.plot(x, job.episode_returns, color=color, alpha=0.35, linewidth=1.0)
            w = max(1, job.config.moving_avg_window)
            ma = []
            for idx in range(len(job.episode_returns)):
                start = max(0, idx - w + 1)
                ma.append(float(np.mean(job.episode_returns[start:idx + 1])))
            ax.plot(x, ma, color=color, alpha=1.0, linewidth=2.5, label=job.name)

        if visible:
            loc = "lower left" if any(len(j.episode_returns) > 4 for j in visible) else "upper right"
            ax.legend(loc=loc, facecolor=BG, edgecolor=GRID_CLR, labelcolor=FG)

        self._fig.tight_layout(pad=1.0)
        self._canvas.draw_idle()

    def save_plot(self, path: str):
        self._fig.savefig(path, dpi=160, facecolor=BG)


class StatusWindow:
    COLUMNS = ("algorithm", "episode", "return", "movingavg", "epsilon", "loss", "duration", "steps", "visible")

    def __init__(self, parent: tk.Tk, manager: TrainingManager):
        self.parent = parent
        self.manager = manager
        self.win: Optional[tk.Toplevel] = None
        self.tree: Optional[ttk.Treeview] = None
        self._last_row_update: Dict[str, float] = {}

    def show(self):
        if self.win and self.win.winfo_exists():
            self.win.lift()
            return
        self.win = tk.Toplevel(self.parent)
        self.win.title("Training Status")
        self.win.geometry("980x420")
        self.win.configure(bg=BG)
        self.tree = ttk.Treeview(self.win, columns=self.COLUMNS, show="headings", selectmode="browse")
        self.tree.pack(fill="both", expand=True, padx=4, pady=4)

        for col in self.COLUMNS:
            self.tree.heading(col, text=col.capitalize(), command=lambda c=col: self._sort(c))
            self.tree.column(col, width=100, anchor="center")
        self.tree.column("algorithm", width=180, stretch=True, anchor="w")

        self.tree.bind("<Double-1>", lambda _e: self.toggle_selected_visibility())
        self.tree.bind("<Return>", lambda _e: self.toggle_selected_visibility())
        self.tree.bind("<space>", lambda _e: self.pause_resume_selected())
        self.tree.bind("<Button-3>", self._show_context)

        action_frame = ttk.Frame(self.win)
        action_frame.pack(fill="x", padx=4, pady=(0, 4))
        for text, cmd in [
            ("Toggle Visibility", self.toggle_selected_visibility),
            ("Pause", self.pause_selected),
            ("Resume", self.resume_selected),
            ("Cancel", self.cancel_selected),
            ("Restart", self.restart_selected),
            ("Run", self.run_selected),
            ("Remove", self.remove_selected),
        ]:
            ttk.Button(action_frame, text=text, command=cmd).pack(side="left", padx=2)

        self.refresh_all()

    def selected_job_id(self) -> Optional[str]:
        if self.tree is None:
            return None
        sel = self.tree.selection()
        return sel[0] if sel else None

    def refresh_all(self):
        if self.tree is None:
            return
        existing = set(self.tree.get_children())
        current = {j.job_id for j in self.manager.job_list()}
        for stale in existing - current:
            self.tree.delete(stale)
        for job in self.manager.job_list():
            values = self._values(job)
            if job.job_id in existing:
                self.tree.item(job.job_id, values=values)
            else:
                self.tree.insert("", "end", iid=job.job_id, values=values)

    def update_job(self, job_id: str):
        if self.tree is None:
            return
        now = tk._get_default_root().tk.call("clock", "milliseconds") if tk._get_default_root() else 0
        if now - self._last_row_update.get(job_id, 0) < 50:
            return
        self._last_row_update[job_id] = now
        job = self.manager.get_job(job_id)
        if not job:
            return
        values = self._values(job)
        if job_id in set(self.tree.get_children()):
            self.tree.item(job_id, values=values)
        else:
            self.tree.insert("", "end", iid=job_id, values=values)

    def _values(self, job: TrainingJob):
        ep = f"{job.total_episodes_done}/{job.config.episodes}"
        ret = f"{job.episode_returns[-1]:.2f}" if job.episode_returns else "-"
        ma = f"{job.moving_avg:.2f}" if job.episode_returns else "-"
        eps = f"{job.episode_epsilons[-1]:.4f}" if job.episode_epsilons else "-"
        loss = f"{job.episode_losses[-1]:.2e}" if job.episode_losses else "-"
        dur = f"{job.episode_durations[-1]:.3f}s" if job.episode_durations else "-"
        steps = str(job.episode_lengths[-1]) if job.episode_lengths else "-"
        return (job.name, ep, ret, ma, eps, loss, dur, steps, "Yes" if job.visible else "No")

    def toggle_selected_visibility(self):
        jid = self.selected_job_id()
        if not jid:
            return
        job = self.manager.get_job(jid)
        if job:
            job.visible = not job.visible
            self.update_job(jid)

    def pause_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.pause(jid)

    def resume_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.resume(jid)

    def cancel_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.cancel(jid)

    def restart_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.restart(jid)

    def pause_resume_selected(self):
        jid = self.selected_job_id()
        if not jid:
            return
        job = self.manager.get_job(jid)
        if not job:
            return
        if job.status == JobStatus.RUNNING:
            self.manager.pause(jid)
        elif job.status == JobStatus.PAUSED:
            self.manager.resume(jid)

    def run_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.run_job(jid)

    def remove_selected(self):
        jid = self.selected_job_id()
        if jid:
            self.manager.remove(jid)
            if self.tree and jid in set(self.tree.get_children()):
                self.tree.delete(jid)

    def _show_context(self, event):
        if self.tree is None:
            return
        item = self.tree.identify_row(event.y)
        if not item:
            return
        self.tree.selection_set(item)
        menu = tk.Menu(self.win, tearoff=0, bg=BG_PANEL, fg=FG, activebackground=SELECT_BG, activeforeground=FG)
        menu.add_command(label="Toggle Visibility", command=self.toggle_selected_visibility)
        menu.add_command(label="Pause", command=self.pause_selected)
        menu.add_command(label="Resume", command=self.resume_selected)
        menu.add_command(label="Cancel", command=self.cancel_selected)
        menu.add_command(label="Restart", command=self.restart_selected)
        menu.tk_popup(event.x_root, event.y_root)

    def _sort(self, col):
        if self.tree is None:
            return
        idx = self.COLUMNS.index(col)
        items = list(self.tree.get_children())
        items.sort(key=lambda i: self.tree.item(i, "values")[idx])
        for pos, iid in enumerate(items):
            self.tree.move(iid, "", pos)


class WorkbenchUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Taxi RL Workbench")
        root.geometry("1280x860")
        root.minsize(900, 620)
        apply_theme(root)

        self.event_bus = EventBus()
        self.manager = TrainingManager(self.event_bus)
        self.event_bus.subscribe(self._on_event)

        self.status_window = StatusWindow(root, self.manager)

        self._resize_after_id = None
        self._current_button_style = "TButton"
        self._plot_after_id = None

        vpw = ttk.PanedWindow(root, orient="vertical")
        vpw.pack(fill="both", expand=True)
        top = ttk.Frame(vpw)
        bottom = ttk.Frame(vpw)
        vpw.add(top, weight=2)
        vpw.add(bottom, weight=1)

        hpw = ttk.PanedWindow(top, orient="horizontal")
        hpw.pack(fill="both", expand=True)
        self.config_panel = ConfigPanel(hpw, on_apply_env=self._on_apply_env)
        self.visualization_panel = VisualizationPanel(hpw)
        hpw.add(self.config_panel, weight=1)
        hpw.add(self.visualization_panel, weight=2)

        self.progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(bottom, variable=self.progress_var, maximum=100, mode="determinate").pack(fill="x", padx=6, pady=(4, 2))

        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(fill="x", padx=6, pady=2)
        self._buttons: Dict[str, ttk.Button] = {}
        for label, cmd in [
            ("Add Job", self._on_add_job),
            ("Train", self._on_train),
            ("Training Status", self._on_status),
            ("Save plot", self._on_save_plot),
            ("Cancel Training", self._on_cancel),
            ("Save", self._on_save),
            ("Load", self._on_load),
        ]:
            b = ttk.Button(btn_frame, text=label, command=cmd)
            b.pack(side="left", padx=2, fill="x", expand=True)
            self._buttons[label] = b

        self.plot_panel = PlotPanel(bottom)
        self.plot_panel.pack(fill="both", expand=True, padx=6, pady=(2, 4))

        self._poll_interval = 10
        self._poll()
        root.bind("<Configure>", self._on_resize)

    def _poll(self):
        self.event_bus.process_events()
        self._update_progress()
        self._update_visualisation()
        self.root.after(self._poll_interval, self._poll)

    def _on_event(self, event: Event):
        if event.type == EventType.EPISODE_COMPLETED:
            jid = event.data["job_id"]
            result: EpisodeResult = event.data["result"]
            job = self.manager.get_job(jid)
            if job:
                job.record_episode(result)
                job.visualization_enabled = self.config_panel.visualization_enabled
                job.render_interval = self.config_panel.frame_interval_ms / 1000.0
            self.status_window.update_job(jid)
            self.plot_panel.redraw(self.manager.job_list())
        elif event.type in (EventType.TRAINING_DONE, EventType.JOB_STATE_CHANGED):
            jid = event.data.get("job_id")
            if jid:
                self.status_window.update_job(jid)
            self.plot_panel.redraw(self.manager.job_list(), force=True)
        elif event.type == EventType.JOB_CREATED:
            self.status_window.refresh_all()
        elif event.type == EventType.ERROR:
            messagebox.showerror("Error", event.data.get("error", "Unknown error"))

    def _update_progress(self):
        jobs = self.manager.job_list()
        if not jobs:
            self.progress_var.set(0)
            return
        total = sum(j.config.episodes for j in jobs)
        done = sum(j.total_episodes_done for j in jobs)
        self.progress_var.set(0 if total == 0 else (done / total * 100.0))

    def _update_visualisation(self):
        if not self.config_panel.visualization_enabled:
            return
        selected_job = None
        sid = self.status_window.selected_job_id() if (self.status_window.win and self.status_window.win.winfo_exists()) else None
        if sid:
            selected_job = self.manager.get_job(sid)
            if selected_job and not selected_job.is_alive():
                selected_job = None
        if selected_job is None:
            for job in self.manager.job_list():
                if job.is_alive():
                    selected_job = job
                    break
        if selected_job:
            self.visualization_panel.update_frame(selected_job.get_latest_frame())

    def _on_add_job(self):
        cfg = self.config_panel.get_config()
        if self.config_panel.compare_mode:
            self.manager.add_compare_jobs(cfg)
        elif self.config_panel.tuning_mode:
            p, mn, mx, st = self.config_panel.tune_params
            self.manager.add_tuning_jobs(cfg, p, mn, mx, st)
        else:
            self.manager.add_job(cfg)
        self.status_window.refresh_all()

    def _on_train(self):
        selected_cfg = self.config_panel.get_config()
        if not self.config_panel.compare_mode and not self.config_panel.tuning_mode:
            selectable_states = (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.CANCELLED)
            has_selected_method = any(
                (j.status in selectable_states) and (j.config.algorithm == selected_cfg.algorithm)
                for j in self.manager.job_list()
            )
            if not has_selected_method:
                self.manager.add_job(selected_cfg)

        pending = [j for j in self.manager.job_list() if j.status in (JobStatus.PENDING, JobStatus.COMPLETED, JobStatus.CANCELLED)]
        if not pending:
            self._on_add_job()
        self.manager.start_all_pending()

    def _on_status(self):
        self.status_window.show()

    def _on_save_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.plot_panel.save_plot(path)

    def _on_cancel(self):
        self.manager.cancel_all()

    def _on_save(self):
        d = filedialog.askdirectory(title="Save training jobs")
        if d:
            CheckpointManager.save_all(self.manager.job_list(), d)

    def _on_load(self):
        d = filedialog.askdirectory(title="Load training jobs")
        if not d:
            return
        loaded = CheckpointManager.load_all(d)
        for job in loaded:
            self.manager.jobs[job.job_id] = job
            self.event_bus.publish(Event(EventType.JOB_CREATED, {"job_id": job.job_id}))
        self.plot_panel.redraw(self.manager.job_list(), force=True)

    def _on_apply_env(self):
        for job in self.manager.job_list():
            if not job.is_alive():
                job.cleanup()
                job.model = None
                job.status = JobStatus.PENDING
        self.plot_panel.redraw(self.manager.job_list(), force=True)

    def _on_resize(self, event):
        if event.widget != self.root:
            return
        if self._resize_after_id:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self._apply_resize)

    def _apply_resize(self):
        self._resize_after_id = None
        width = self.root.winfo_width()
        target = "TButton" if width >= 1100 else "Compact.TButton"
        if target != self._current_button_style:
            self._current_button_style = target
            for button in self._buttons.values():
                button.configure(style=target)
        if self._plot_after_id:
            self.root.after_cancel(self._plot_after_id)
        self._plot_after_id = self.root.after(60, lambda: self.plot_panel.redraw(self.manager.job_list(), force=True))
