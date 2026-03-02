"""
bipedal_walker_ui.py
Tkinter-based Workbench UI for BipedalWalker RL.
"""

from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from PIL import Image, ImageTk

from bipedal_walker_logic import (
    A2CConfig,
    AlgorithmType,
    CheckpointManager,
    EnvironmentConfig,
    EpisodeConfig,
    Event,
    EventBus,
    EventType,
    JobConfig,
    JobStatus,
    LRSchedule,
    NetworkConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    TrainingJob,
    TrainingManager,
    TuningConfig,
    get_bus,
)

# ---------------------------------------------------------------------------
# Theme / Style Constants
# ---------------------------------------------------------------------------

BG        = "#0f111a"
BG2       = "#161929"
BG3       = "#1e2233"
ACCENT    = "#4cc9f0"
ACCENT2   = "#7b2fff"
FG        = "#e6e6e6"
FG2       = "#b5b5b5"
GRID_COL  = "#2a2f3a"
SEL_BG    = "#2a3a5a"
ERR_COL   = "#f0544c"
FONT      = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_BIG  = ("Segoe UI", 11, "bold")
FONT_MONO = ("Consolas", 9)

ALGO_COLORS = {
    "PPO": "#4cc9f0",
    "A2C": "#f7b731",
    "SAC": "#a29bfe",
    "TD3": "#55efc4",
}

PALETTE = ["#4cc9f0", "#f7b731", "#a29bfe", "#55efc4", "#fd79a8", "#e17055", "#00b894", "#6c5ce7"]


def setup_ttk_style(root: tk.Tk) -> None:
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    # General
    style.configure(".", background=BG, foreground=FG, font=FONT,
                    fieldbackground=BG2, troughcolor=BG3, insertcolor=FG,
                    selectbackground=SEL_BG, selectforeground=FG,
                    bordercolor=GRID_COL, lightcolor=BG2, darkcolor=BG)

    for w in ("TFrame", "TLabelframe"):
        style.configure(w, background=BG, bordercolor=GRID_COL)

    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=FONT_BOLD)

    style.configure("TLabel", background=BG, foreground=FG2, font=FONT)
    style.configure("Header.TLabel", background=BG, foreground=ACCENT, font=FONT_BOLD)

    style.configure("TEntry", fieldbackground=BG2, foreground=FG, insertcolor=FG,
                    bordercolor=GRID_COL, selectbackground=SEL_BG)

    style.configure("TCombobox", fieldbackground=BG2, foreground=FG,
                    background=BG2, arrowcolor=ACCENT)
    style.map("TCombobox", fieldbackground=[("readonly", BG2)],
              selectbackground=[("readonly", BG2)], selectforeground=[("readonly", FG)])

    style.configure("TCheckbutton", background=BG, foreground=FG2)
    style.map("TCheckbutton", background=[("active", BG)])

    style.configure("TNotebook", background=BG, bordercolor=GRID_COL, tabmargins=[2, 2, 0, 0])
    style.configure("TNotebook.Tab", background=BG3, foreground=FG2, font=FONT,
                    padding=[8, 3], bordercolor=GRID_COL)
    style.map("TNotebook.Tab",
              background=[("selected", BG), ("active", GRID_COL)],
              foreground=[("selected", ACCENT)])

    style.configure("TButton", background=BG3, foreground=FG, font=FONT,
                    bordercolor=GRID_COL, relief="flat", padding=[8, 4])
    style.map("TButton",
              background=[("active", GRID_COL), ("pressed", SEL_BG)],
              foreground=[("active", ACCENT)])

    style.configure("Accent.TButton", background=ACCENT2, foreground=FG, font=FONT_BOLD,
                    padding=[8, 4])
    style.map("Accent.TButton", background=[("active", "#5a1fc0")])

    style.configure("TScrollbar", background=BG3, troughcolor=BG, arrowcolor=FG2,
                    bordercolor=GRID_COL)

    style.configure("TProgressbar", background=ACCENT, troughcolor=BG3,
                    bordercolor=GRID_COL, thickness=6)

    style.configure("Treeview", background=BG2, foreground=FG, fieldbackground=BG2,
                    rowheight=22, font=FONT, bordercolor=GRID_COL)
    style.configure("Treeview.Heading", background=BG3, foreground=ACCENT,
                    font=FONT_BOLD, bordercolor=GRID_COL, relief="flat")
    style.map("Treeview", background=[("selected", SEL_BG)], foreground=[("selected", FG)])

    style.configure("TSeparator", background=GRID_COL)

    style.configure("Compact.TButton", padding=[4, 2], font=("Segoe UI", 8))
    style.configure("Compact.Accent.TButton", background=ACCENT2, foreground=FG,
                    padding=[4, 2], font=("Segoe UI", 8, "bold"))


# ---------------------------------------------------------------------------
# Helper Widgets
# ---------------------------------------------------------------------------

def labeled_entry(parent, label: str, default: str, row: int, col: int = 0,
                  width: int = 8) -> tk.StringVar:
    """Create label+entry pair, returns StringVar."""
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=1)
    var = tk.StringVar(value=default)
    ttk.Entry(parent, textvariable=var, width=width).grid(
        row=row, column=col + 1, sticky="ew", padx=(0, 8), pady=1)
    return var


def labeled_combo(parent, label: str, values: List[str], default: str,
                  row: int, col: int = 0, width: int = 12) -> tk.StringVar:
    ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=(0, 4), pady=1)
    var = tk.StringVar(value=default)
    cb = ttk.Combobox(parent, textvariable=var, values=values, width=width, state="readonly")
    cb.grid(row=row, column=col + 1, sticky="ew", padx=(0, 8), pady=1)
    return var


def titled_frame(parent, title: str) -> ttk.LabelFrame:
    f = ttk.LabelFrame(parent, text=title, style="TLabelframe")
    return f


# ---------------------------------------------------------------------------
# Scrollable Frame
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        vsb    = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)

        self.inner.bind("<Configure>",
                        lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _on_mousewheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)


# ---------------------------------------------------------------------------
# Config Panel
# ---------------------------------------------------------------------------

class ConfigPanel(ttk.Frame):
    """Left panel: all configuration forms."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        sf = ScrollableFrame(self)
        sf.pack(fill="both", expand=True)
        self._inner = sf.inner

        self._build_env_frame()
        self._build_episode_frame()
        self._build_tuning_frame()
        self._build_algo_tabs()
        self._build_apply_button()

    # ------------------------------------------------------------------
    # Environment Config
    # ------------------------------------------------------------------

    def _build_env_frame(self):
        f = titled_frame(self._inner, "Environment Configuration")
        f.pack(fill="x", padx=4, pady=(4, 2))
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        ttk.Label(f, text="Environment").grid(row=0, column=0, sticky="w", padx=(4, 4), pady=2)
        self._env_name = tk.StringVar(value="BipedalWalker-v3")
        ttk.Entry(f, textvariable=self._env_name, width=18).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=2)

        self._hardcore_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Hardcore", variable=self._hardcore_var).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=4, pady=1)

        self._render_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Enable Visualization", variable=self._render_var).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=4, pady=1)

        r = 2
        ttk.Label(f, text="Render Interval (ms)").grid(row=r, column=0, sticky="w", padx=4, pady=1)
        self._render_ms = tk.StringVar(value="10")
        ttk.Entry(f, textvariable=self._render_ms, width=6).grid(
            row=r, column=1, sticky="ew", padx=(0, 8), pady=1)

    # ------------------------------------------------------------------
    # Episode Config
    # ------------------------------------------------------------------

    def _build_episode_frame(self):
        f = titled_frame(self._inner, "Episode Configuration")
        f.pack(fill="x", padx=4, pady=2)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Compare Methods (run all algorithms)",
                        variable=self._compare_var).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        # Row 1: Episodes + Max-Steps
        ttk.Label(f, text="Episodes").grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self._n_episodes = tk.StringVar(value="3000")
        ttk.Entry(f, textvariable=self._n_episodes, width=8).grid(
            row=1, column=1, sticky="ew", padx=(0, 6), pady=1)
        ttk.Label(f, text="Max Steps").grid(row=1, column=2, sticky="w", padx=(0, 2), pady=1)
        self._max_steps = tk.StringVar(value="1600")
        ttk.Entry(f, textvariable=self._max_steps, width=8).grid(
            row=1, column=3, sticky="ew", padx=(0, 4), pady=1)

        # Row 2: Alpha + Gamma
        ttk.Label(f, text="Alpha").grid(row=2, column=0, sticky="w", padx=(4, 2), pady=1)
        self._alpha = tk.StringVar(value="3e-4")
        ttk.Entry(f, textvariable=self._alpha, width=8).grid(
            row=2, column=1, sticky="ew", padx=(0, 6), pady=1)
        ttk.Label(f, text="Gamma").grid(row=2, column=2, sticky="w", padx=(0, 2), pady=1)
        self._gamma = tk.StringVar(value="0.99")
        ttk.Entry(f, textvariable=self._gamma, width=8).grid(
            row=2, column=3, sticky="ew", padx=(0, 4), pady=1)

        # Row 3: LR Schedule
        ttk.Label(f, text="LR Schedule").grid(row=3, column=0, sticky="w", padx=(4, 2), pady=1)
        self._lr_schedule = tk.StringVar(value=LRSchedule.CONSTANT.value)
        ttk.Combobox(f, textvariable=self._lr_schedule,
                     values=[s.value for s in LRSchedule], width=10, state="readonly").grid(
            row=3, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=1)

    # ------------------------------------------------------------------
    # Tuning Config
    # ------------------------------------------------------------------

    def _build_tuning_frame(self):
        f = titled_frame(self._inner, "Parameter Tuning")
        f.pack(fill="x", padx=4, pady=2)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._tuning_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Enable Tuning", variable=self._tuning_var).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        ttk.Label(f, text="Parameter").grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        self._tune_param = tk.StringVar(value="alpha")
        ttk.Combobox(f, textvariable=self._tune_param,
                     values=["alpha", "gamma", "buffer_size", "batch_size",
                             "clip_range", "ent_coef", "tau", "gae_lambda"],
                     width=10, state="readonly").grid(
            row=1, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=1)

        # Min + Max
        ttk.Label(f, text="Min").grid(row=2, column=0, sticky="w", padx=(4, 2), pady=1)
        self._tune_min = tk.StringVar(value="1e-4")
        ttk.Entry(f, textvariable=self._tune_min, width=8).grid(
            row=2, column=1, sticky="ew", padx=(0, 6), pady=1)
        ttk.Label(f, text="Max").grid(row=2, column=2, sticky="w", padx=(0, 2), pady=1)
        self._tune_max = tk.StringVar(value="1e-3")
        ttk.Entry(f, textvariable=self._tune_max, width=8).grid(
            row=2, column=3, sticky="ew", padx=(0, 4), pady=1)

        # Step
        ttk.Label(f, text="Step").grid(row=3, column=0, sticky="w", padx=(4, 2), pady=1)
        self._tune_step = tk.StringVar(value="2e-4")
        ttk.Entry(f, textvariable=self._tune_step, width=8).grid(
            row=3, column=1, sticky="ew", padx=(0, 6), pady=1)

        # Hidden Layer Configs: 3 discrete architecture variants to sweep
        ttk.Label(f, text="Layer Configs (one per field, e.g. 256  /  256,256  /  256,256,128)",
                  font=("Segoe UI", 8)).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=(4, 1))
        self._hl_cfg1 = tk.StringVar(value="256")
        self._hl_cfg2 = tk.StringVar(value="256,256")
        self._hl_cfg3 = tk.StringVar(value="256,256,128")
        for i, (var, label) in enumerate([
            (self._hl_cfg1, "Cfg 1"), (self._hl_cfg2, "Cfg 2"), (self._hl_cfg3, "Cfg 3")
        ]):
            ttk.Label(f, text=label).grid(row=5 + i, column=0, sticky="w", padx=(4, 2), pady=1)
            ttk.Entry(f, textvariable=var, width=18).grid(
                row=5 + i, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=1)

    # ------------------------------------------------------------------
    # Algorithm Tabs
    # ------------------------------------------------------------------

    def _build_algo_tabs(self):
        f = titled_frame(self._inner, "Algorithm Configuration")
        f.pack(fill="x", padx=4, pady=2)

        self._algo_select = tk.StringVar(value=AlgorithmType.PPO.value)
        sel_frame = ttk.Frame(f)
        sel_frame.pack(fill="x", padx=4, pady=2)
        ttk.Label(sel_frame, text="Algorithm:").pack(side="left", padx=(0, 6))
        for algo in AlgorithmType:
            ttk.Radiobutton(
                sel_frame, text=algo.value, variable=self._algo_select,
                value=algo.value, command=self._on_algo_change
            ).pack(side="left", padx=4)

        self._tab_notebook = ttk.Notebook(f)
        self._tab_notebook.pack(fill="both", expand=True, padx=4, pady=4)

        self._ppo_frame  = self._build_ppo_tab()
        self._a2c_frame  = self._build_a2c_tab()
        self._sac_frame  = self._build_sac_tab()
        self._td3_frame  = self._build_td3_tab()

        for frame, name in [(self._ppo_frame, "PPO"), (self._a2c_frame, "A2C"),
                            (self._sac_frame, "SAC"), (self._td3_frame, "TD3")]:
            self._tab_notebook.add(frame, text=name)

    def _on_algo_change(self):
        algo = self._algo_select.get()
        idx = {"PPO": 0, "A2C": 1, "SAC": 2, "TD3": 3}
        self._tab_notebook.select(idx.get(algo, 0))

    def _net_subframe(self, parent, row: int, default_layers: str = "256,256",
                      default_activation: str = "relu") -> Tuple:
        sf = ttk.LabelFrame(parent, text="Neural Network")
        sf.grid(row=row, column=0, columnspan=4, sticky="ew", padx=2, pady=2)
        sf.columnconfigure(1, weight=1)
        sf.columnconfigure(3, weight=1)

        ttk.Label(sf, text="Hidden Layers").grid(row=0, column=0, sticky="w", padx=(4, 2), pady=1)
        layers_var = tk.StringVar(value=default_layers)
        ttk.Entry(sf, textvariable=layers_var, width=14).grid(
            row=0, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=1)

        ttk.Label(sf, text="Activation").grid(row=1, column=0, sticky="w", padx=(4, 2), pady=1)
        act_var = tk.StringVar(value=default_activation)
        ttk.Combobox(sf, textvariable=act_var,
                     values=["relu", "tanh", "elu", "leaky"], width=8, state="readonly").grid(
            row=1, column=1, sticky="ew", padx=(0, 8), pady=1)

        return layers_var, act_var

    def _build_ppo_tab(self) -> ttk.Frame:
        f = ttk.Frame(self._tab_notebook)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._ppo_n_steps      = labeled_entry(f, "N Steps",      "2048",   0)
        self._ppo_batch_size   = labeled_entry(f, "Batch Size",   "64",     0, col=2)
        self._ppo_n_epochs     = labeled_entry(f, "N Epochs",     "10",     1)
        self._ppo_clip_range   = labeled_entry(f, "Clip Range",   "0.2",    1, col=2)
        self._ppo_ent_coef     = labeled_entry(f, "Ent Coef",     "0.0",    2)
        self._ppo_vf_coef      = labeled_entry(f, "VF Coef",      "0.5",    2, col=2)
        self._ppo_gae_lambda   = labeled_entry(f, "GAE Lambda",   "0.95",   3)
        self._ppo_max_grad_norm= labeled_entry(f, "Max Grad Norm","0.5",    3, col=2)
        self._ppo_norm_adv_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(f, text="Normalize Advantages",
                        variable=self._ppo_norm_adv_var).grid(row=4, column=0, columnspan=4, sticky="w", padx=4, pady=1)

        self._ppo_layers, self._ppo_act = self._net_subframe(f, row=5, default_layers="256,256", default_activation="relu")
        return f

    def _build_a2c_tab(self) -> ttk.Frame:
        f = ttk.Frame(self._tab_notebook)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._a2c_n_steps     = labeled_entry(f, "N Steps",       "5",    0)
        self._a2c_ent_coef    = labeled_entry(f, "Ent Coef",      "0.0",  0, col=2)
        self._a2c_vf_coef     = labeled_entry(f, "VF Coef",       "0.5",  1)
        self._a2c_gae_lambda  = labeled_entry(f, "GAE Lambda",    "1.0",  1, col=2)
        self._a2c_max_grad    = labeled_entry(f, "Max Grad Norm", "0.5",  2)
        self._a2c_rms_eps     = labeled_entry(f, "RMSProp Eps",   "1e-5", 2, col=2)
        self._a2c_use_rms_var = tk.BooleanVar(value=True)
        self._a2c_norm_adv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Use RMSProp",
                        variable=self._a2c_use_rms_var).grid(
            row=3, column=0, columnspan=2, sticky="w", padx=4, pady=1)
        ttk.Checkbutton(f, text="Normalize Advantages",
                        variable=self._a2c_norm_adv_var).grid(
            row=3, column=2, columnspan=2, sticky="w", padx=4, pady=1)

        self._a2c_layers, self._a2c_act = self._net_subframe(f, row=4, default_layers="64,64", default_activation="relu")
        return f

    def _build_sac_tab(self) -> ttk.Frame:
        f = ttk.Frame(self._tab_notebook)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._sac_buffer_size     = labeled_entry(f, "Buffer Size",    "1000000", 0)
        self._sac_batch_size      = labeled_entry(f, "Batch Size",     "256",     0, col=2)
        self._sac_learning_starts = labeled_entry(f, "Learning Starts","100",     1)
        self._sac_train_freq      = labeled_entry(f, "Train Freq",     "1",       1, col=2)
        self._sac_gradient_steps  = labeled_entry(f, "Gradient Steps", "1",       2)
        self._sac_tau             = labeled_entry(f, "Tau",            "0.005",   2, col=2)
        self._sac_ent_coef        = labeled_entry(f, "Ent Coef",       "auto",    3)
        self._sac_target_update   = labeled_entry(f, "Target Update",  "1",       3, col=2)
        self._sac_use_sde_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(f, text="Use gSDE",
                        variable=self._sac_use_sde_var).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=1)

        self._sac_layers, self._sac_act = self._net_subframe(f, row=5, default_layers="256,256", default_activation="relu")
        return f

    def _build_td3_tab(self) -> ttk.Frame:
        f = ttk.Frame(self._tab_notebook)
        f.columnconfigure(1, weight=1)
        f.columnconfigure(3, weight=1)

        self._td3_buffer_size     = labeled_entry(f, "Buffer Size",     "1000000", 0)
        self._td3_batch_size      = labeled_entry(f, "Batch Size",      "256",     0, col=2)
        self._td3_learning_starts = labeled_entry(f, "Learning Starts", "100",     1)
        self._td3_train_freq      = labeled_entry(f, "Train Freq",      "1",       1, col=2)
        self._td3_gradient_steps  = labeled_entry(f, "Gradient Steps",  "1",       2)
        self._td3_tau             = labeled_entry(f, "Tau",             "0.005",   2, col=2)
        self._td3_policy_delay    = labeled_entry(f, "Policy Delay",    "2",       3)
        self._td3_noise_std       = labeled_entry(f, "Action Noise Std","0.1",     3, col=2)
        self._td3_target_noise    = labeled_entry(f, "Target Noise",    "0.2",     4)
        self._td3_noise_clip      = labeled_entry(f, "Noise Clip",      "0.5",     4, col=2)

        self._td3_layers, self._td3_act = self._net_subframe(f, row=5, default_layers="256,256", default_activation="relu")
        return f

    # ------------------------------------------------------------------
    # Apply Button
    # ------------------------------------------------------------------

    def _build_apply_button(self):
        f = ttk.Frame(self._inner)
        f.pack(fill="x", padx=4, pady=4)
        self._apply_btn = ttk.Button(f, text="Apply and Reset", style="Accent.TButton",
                                     command=self._on_apply)
        self._apply_btn.pack(fill="x")

    def _on_apply(self):
        # Notify listeners
        if self._on_apply_cb:
            self._on_apply_cb()

    _on_apply_cb: Optional[Callable] = None

    def set_apply_callback(self, cb: Callable) -> None:
        self._on_apply_cb = cb

    # ------------------------------------------------------------------
    # Read Config
    # ------------------------------------------------------------------

    def _parse_layers(self, var: tk.StringVar) -> List[int]:
        try:
            parts = [p.strip() for p in var.get().split(",") if p.strip()]
            return [int(p) for p in parts if p.isdigit() or p.lstrip("-").isdigit()]
        except Exception:
            return [256, 256]

    def get_job_config(self) -> JobConfig:
        """Build a JobConfig from current form values."""
        import uuid

        def flt(s: str, default: float) -> float:
            try:
                return float(s)
            except Exception:
                return default

        def integer(s: str, default: int) -> int:
            try:
                return int(float(s))
            except Exception:
                return default

        env_cfg = EnvironmentConfig(
            env_name=self._env_name.get().strip(),
            hardcore=self._hardcore_var.get(),
            render_interval_ms=integer(self._render_ms.get(), 10),
        )
        ep_cfg = EpisodeConfig(
            n_episodes=integer(self._n_episodes.get(), 3000),
            max_steps=integer(self._max_steps.get(), 1600),
            alpha=flt(self._alpha.get(), 3e-4),
            gamma=flt(self._gamma.get(), 0.99),
            lr_schedule=self._lr_schedule.get(),
        )
        tuning_cfg = TuningConfig(
            enabled=self._tuning_var.get(),
            parameter=self._tune_param.get(),
            min_value=flt(self._tune_min.get(), 1e-4),
            max_value=flt(self._tune_max.get(), 1e-3),
            step=flt(self._tune_step.get(), 2e-4),
        )
        ppo_cfg = PPOConfig(
            n_steps=integer(self._ppo_n_steps.get(), 2048),
            batch_size=integer(self._ppo_batch_size.get(), 64),
            n_epochs=integer(self._ppo_n_epochs.get(), 10),
            clip_range=flt(self._ppo_clip_range.get(), 0.2),
            ent_coef=flt(self._ppo_ent_coef.get(), 0.0),
            vf_coef=flt(self._ppo_vf_coef.get(), 0.5),
            gae_lambda=flt(self._ppo_gae_lambda.get(), 0.95),
            max_grad_norm=flt(self._ppo_max_grad_norm.get(), 0.5),
            normalize_advantage=self._ppo_norm_adv_var.get(),
            network=NetworkConfig(self._parse_layers(self._ppo_layers),
                                  self._ppo_act.get()),
        )
        a2c_cfg = A2CConfig(
            n_steps=integer(self._a2c_n_steps.get(), 5),
            ent_coef=flt(self._a2c_ent_coef.get(), 0.0),
            vf_coef=flt(self._a2c_vf_coef.get(), 0.5),
            gae_lambda=flt(self._a2c_gae_lambda.get(), 1.0),
            max_grad_norm=flt(self._a2c_max_grad.get(), 0.5),
            rms_prop_eps=flt(self._a2c_rms_eps.get(), 1e-5),
            use_rms_prop=self._a2c_use_rms_var.get(),
            normalize_advantage=self._a2c_norm_adv_var.get(),
            network=NetworkConfig(self._parse_layers(self._a2c_layers),
                                  self._a2c_act.get()),
        )
        sac_cfg = SACConfig(
            buffer_size=integer(self._sac_buffer_size.get(), 1_000_000),
            batch_size=integer(self._sac_batch_size.get(), 256),
            learning_starts=integer(self._sac_learning_starts.get(), 100),
            train_freq=integer(self._sac_train_freq.get(), 1),
            gradient_steps=integer(self._sac_gradient_steps.get(), 1),
            tau=flt(self._sac_tau.get(), 0.005),
            ent_coef=self._sac_ent_coef.get() if self._sac_ent_coef.get() == "auto"
                     else flt(self._sac_ent_coef.get(), 0.5),
            target_update_interval=integer(self._sac_target_update.get(), 1),
            use_sde=self._sac_use_sde_var.get(),
            network=NetworkConfig(self._parse_layers(self._sac_layers),
                                  self._sac_act.get()),
        )
        td3_cfg = TD3Config(
            buffer_size=integer(self._td3_buffer_size.get(), 1_000_000),
            batch_size=integer(self._td3_batch_size.get(), 256),
            learning_starts=integer(self._td3_learning_starts.get(), 100),
            train_freq=integer(self._td3_train_freq.get(), 1),
            gradient_steps=integer(self._td3_gradient_steps.get(), 1),
            tau=flt(self._td3_tau.get(), 0.005),
            policy_delay=integer(self._td3_policy_delay.get(), 2),
            target_policy_noise=flt(self._td3_target_noise.get(), 0.2),
            target_noise_clip=flt(self._td3_noise_clip.get(), 0.5),
            action_noise_std=flt(self._td3_noise_std.get(), 0.1),
            network=NetworkConfig(self._parse_layers(self._td3_layers),
                                  self._td3_act.get()),
        )

        algo = self._algo_select.get()
        jid  = str(__import__("uuid").uuid4())[:8]
        return JobConfig(
            job_id=jid,
            name=f"{algo}-{jid}",
            algorithm=algo,
            env_cfg=env_cfg,
            ep_cfg=ep_cfg,
            ppo_cfg=ppo_cfg,
            a2c_cfg=a2c_cfg,
            sac_cfg=sac_cfg,
            td3_cfg=td3_cfg,
            tuning_cfg=tuning_cfg,
        )

    @property
    def compare_mode(self) -> bool:
        return self._compare_var.get()

    @property
    def tuning_mode(self) -> bool:
        return self._tuning_var.get()

    @property
    def tuning_layer_configs(self) -> List[List[int]]:
        """Return all non-empty hidden layer config variants from the tuning panel."""
        result: List[List[int]] = []
        for var in (self._hl_cfg1, self._hl_cfg2, self._hl_cfg3):
            raw = var.get().strip()
            if not raw:
                continue
            layers = self._parse_layers(var)
            if layers:
                result.append(layers)
        return result

    @property
    def visualization_enabled(self) -> bool:
        return self._render_var.get()

    @property
    def render_interval_ms(self) -> int:
        try:
            return int(self._render_ms.get())
        except Exception:
            return 10


# ---------------------------------------------------------------------------
# Visualization Panel
# ---------------------------------------------------------------------------

class VisualizationPanel(ttk.Frame):
    """Right panel: live env rendering."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(style="TFrame")
        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._current_img_tk = None
        self._no_signal_drawn = False
        self._draw_no_signal()

    def _draw_no_signal(self):
        self._canvas.delete("all")
        w = self._canvas.winfo_width() or 400
        h = self._canvas.winfo_height() or 300
        self._canvas.create_text(
            w // 2, h // 2,
            text="No Visualization",
            fill=GRID_COL, font=("Segoe UI", 14),
        )

    def update_frame(self, frame: np.ndarray) -> None:
        """Display an RGB numpy array frame, maintaining aspect ratio."""
        if frame is None:
            return

        try:
            cw = self._canvas.winfo_width()
            ch = self._canvas.winfo_height()
            if cw < 2 or ch < 2:
                return

            fh, fw = frame.shape[:2]
            scale = min(cw / fw, ch / fh)
            nw = int(fw * scale)
            nh = int(fh * scale)
            x0 = (cw - nw) // 2
            y0 = (ch - nh) // 2

            img = Image.fromarray(frame)
            img = img.resize((nw, nh), Image.NEAREST)
            img_tk = ImageTk.PhotoImage(img)
            self._canvas.delete("all")
            self._canvas.create_rectangle(0, 0, cw, ch, fill=BG, outline="")
            self._canvas.create_image(x0, y0, anchor="nw", image=img_tk)
            self._current_img_tk = img_tk  # prevent GC
        except Exception as exc:
            pass


# ---------------------------------------------------------------------------
# Bottom Panel (Progress + Buttons + Plot)
# ---------------------------------------------------------------------------

class PlotPanel(ttk.Frame):
    """Bottom area: progress bar, control buttons, matplotlib plot."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self._jobs_data: Dict[str, Dict] = {}  # job_id -> {returns, moving_avg, color, label}
        self._visible: Dict[str, bool] = {}

        self._build_progress()
        self._build_buttons()
        self._build_plot()

        self._compact = False
        self._resize_id = None

        self.bind("<Configure>", self._on_resize_debounce)

    def _build_progress(self):
        pf = ttk.Frame(self)
        pf.pack(fill="x", padx=4, pady=(4, 0))
        ttk.Label(pf, text="Progress:").pack(side="left", padx=(0, 4))
        self._progress_var = tk.DoubleVar()
        self._progress_bar = ttk.Progressbar(
            pf, variable=self._progress_var,
            maximum=100, mode="determinate"
        )
        self._progress_bar.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._progress_label = ttk.Label(pf, text="0 / 0", width=14)
        self._progress_label.pack(side="left")
        # Status label – gives immediate feedback while model is building
        self._status_label = ttk.Label(pf, text="Idle", foreground=ACCENT, width=28)
        self._status_label.pack(side="right", padx=(8, 4))

    def set_status(self, text: str) -> None:
        """Update training status text shown next to the progress bar."""
        self._status_label.configure(text=text)

    def _build_buttons(self):
        self._btn_frame = ttk.Frame(self)
        self._btn_frame.pack(fill="x", padx=4, pady=2)

        btn_defs = [
            ("Add Job",           "add_job",        "TButton"),
            ("Train",             "train",           "Accent.TButton"),
            ("Training Status",   "status",          "TButton"),
            ("Save Image",        "save_image",      "TButton"),
            ("Save Content",      "save_content",    "TButton"),
            ("Load Content",      "load_content",    "TButton"),
            ("Cancel Training",   "cancel_training", "TButton"),
            ("Reset Training",    "reset_training",  "TButton"),
            ("Save Jobs",         "save_jobs",       "TButton"),
            ("Load Jobs",         "load_jobs",       "TButton"),
        ]

        self._buttons: Dict[str, ttk.Button] = {}
        for label, key, style in btn_defs:
            btn = ttk.Button(self._btn_frame, text=label, style=style,
                             command=lambda k=key: self._on_button(k))
            btn.pack(side="left", padx=2, pady=1)
            self._buttons[key] = btn

    def _on_button(self, key: str):
        if self._button_cb:
            self._button_cb(key)

    _button_cb: Optional[Callable[[str], None]] = None

    def set_button_callback(self, cb: Callable[[str], None]) -> None:
        self._button_cb = cb

    def _build_plot(self):
        self._fig = plt.Figure(figsize=(8, 3), facecolor=BG)
        self._ax  = self._fig.add_subplot(111)
        self._setup_axes()

        self._canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)

        # Legend container (outside plot right side)
        self._legend_frame = ttk.Frame(self)
        # will be managed by the workbench

    def _setup_axes(self):
        ax = self._ax
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG2)
        ax.set_xlabel("Episode", color=FG2, fontsize=9)
        ax.set_ylabel("Return",  color=FG2, fontsize=9)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COL)
        ax.grid(True, color=GRID_COL, linestyle="--", alpha=0.5)
        self._fig.tight_layout(pad=1.5)

    def update_job_data(self, job_id: str, returns: List[float],
                        moving_avg: List[float], label: str, color: str) -> None:
        self._jobs_data[job_id] = {
            "returns":    list(returns),
            "moving_avg": list(moving_avg),
            "label":      label,
            "color":      color,
        }
        if job_id not in self._visible:
            self._visible[job_id] = True

    def toggle_visibility(self, job_id: str) -> None:
        self._visible[job_id] = not self._visible.get(job_id, True)
        self.redraw()

    def set_visibility(self, job_id: str, visible: bool) -> None:
        self._visible[job_id] = visible

    def remove_job(self, job_id: str) -> None:
        self._jobs_data.pop(job_id, None)
        self._visible.pop(job_id, None)
        self.redraw()

    def redraw(self) -> None:
        self._ax.clear()
        self._setup_axes()

        for i, (jid, data) in enumerate(self._jobs_data.items()):
            if not self._visible.get(jid, True):
                continue
            color = data["color"]
            returns    = data["returns"]
            moving_avg = data["moving_avg"]
            label      = data["label"]
            x = range(1, len(returns) + 1)
            xm = range(1, len(moving_avg) + 1)
            self._ax.plot(x, returns, color=color, alpha=0.35, linewidth=1.0,
                          label=f"{label} (raw)")
            self._ax.plot(xm, moving_avg, color=color, alpha=1.0, linewidth=2.5,
                          label=f"{label} (avg)")

        # Legend outside axes – shrink the axes right margin to fit it
        handles = self._ax.get_lines()
        if handles:
            n_visible = sum(1 for jid in self._jobs_data
                            if self._visible.get(jid, True))
            # Each job = 2 legend lines; longer labels need more space
            right = max(0.45, 0.90 - 0.04 * n_visible)
            self._fig.subplots_adjust(left=0.07, right=right,
                                      bottom=0.13, top=0.93)
            self._ax.legend(
                loc="upper left", bbox_to_anchor=(1.01, 1),
                facecolor=BG, edgecolor=GRID_COL,
                labelcolor=FG, fontsize=8,
            )
        else:
            self._fig.subplots_adjust(left=0.07, right=0.97,
                                      bottom=0.13, top=0.93)

        self._canvas.draw_idle()

    def update_progress(self, episode: int, total: int) -> None:
        pct = (episode / total * 100) if total > 0 else 0
        self._progress_var.set(pct)
        self._progress_label.configure(text=f"{episode} / {total}")

    def reset(self) -> None:
        self._jobs_data.clear()
        self._visible.clear()
        self._progress_var.set(0)
        self._progress_label.configure(text="0 / 0")
        self._status_label.configure(text="Idle")
        self.redraw()

    def save_image(self, path: str) -> None:
        self._fig.savefig(path, facecolor=BG, dpi=150, bbox_inches="tight")

    def get_plot_data(self) -> Dict[str, Any]:
        return copy.deepcopy(self._jobs_data)

    def load_plot_data(self, data: Dict[str, Any]) -> None:
        self._jobs_data = data
        for jid in data:
            self._visible[jid] = True
        self.redraw()

    def _on_resize_debounce(self, _event=None):
        if self._resize_id:
            self.after_cancel(self._resize_id)
        self._resize_id = self.after(100, self._apply_resize)

    _COMPACT_THRESHOLD = 1000

    def _apply_resize(self):
        width = self.winfo_width()
        compact = width < self._COMPACT_THRESHOLD

        if compact != self._compact:
            self._compact = compact
            style = "Compact.TButton" if compact else "TButton"
            accent_style = "Compact.Accent.TButton" if compact else "Accent.TButton"
            for key, btn in self._buttons.items():
                if key == "train":
                    btn.configure(style=accent_style)
                else:
                    btn.configure(style=style)


# ---------------------------------------------------------------------------
# Training Status Window
# ---------------------------------------------------------------------------

class TrainingStatusWindow(tk.Toplevel):
    """Modal-like window showing training job status."""

    COLUMNS = ("algorithm", "episode", "return", "moving_avg", "loss", "duration", "steps", "visible")
    COL_WIDTHS = {
        "algorithm": 120, "episode": 90, "return": 80,
        "moving_avg": 80, "loss": 80, "duration": 80, "steps": 60, "visible": 60,
    }
    COL_HEADS = {
        "algorithm": "Algorithm", "episode": "Episode",
        "return": "Return", "moving_avg": "Avg Return",
        "loss": "Loss", "duration": "Duration(s)", "steps": "Steps", "visible": "Visible",
    }

    def __init__(self, parent, manager: TrainingManager, plot_panel: PlotPanel,
                 on_select_cb: Callable[[Optional[str]], None], **kwargs):
        super().__init__(parent, **kwargs)
        self.title("Training Status")
        self.configure(bg=BG)
        self.geometry("900x400")
        self.minsize(600, 300)

        self._manager = manager
        self._plot    = plot_panel
        self._on_select_cb = on_select_cb
        self._last_update: Dict[str, float] = {}  # rate limiting: job_id -> timestamp

        self._build_ui()
        self._populate_existing()

    def _build_ui(self):
        # Table
        tree_frame = ttk.Frame(self)
        tree_frame.pack(fill="both", expand=True, padx=4, pady=4)

        vsb = ttk.Scrollbar(tree_frame, orient="vertical")
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal")

        self._tree = ttk.Treeview(
            tree_frame,
            columns=self.COLUMNS,
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
        )
        vsb.configure(command=self._tree.yview)
        hsb.configure(command=self._tree.xview)

        for col in self.COLUMNS:
            self._tree.heading(col, text=self.COL_HEADS[col],
                               command=lambda c=col: self._sort(c))
            stretch = col == "algorithm"
            self._tree.column(col, width=self.COL_WIDTHS[col],
                              stretch=stretch, anchor="center")

        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)

        # Bind events
        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.bind("<Button-3>", self._show_context_menu)
        self._tree.bind("<<TreeviewSelect>>", self._on_selection)
        self._tree.bind("<Return>", lambda e: self._cmd_toggle_vis())
        self._tree.bind("<space>", lambda e: self._cmd_pause_resume())

        # Buttons
        btn_f = ttk.Frame(self)
        btn_f.pack(fill="x", padx=4, pady=4)

        for label, cmd in [
            ("Toggle Visibility", self._cmd_toggle_vis),
            ("Train / Resume",    self._cmd_resume),
            ("Run",               self._cmd_run),
            ("Pause",             self._cmd_pause),
            ("Stop",              self._cmd_stop),
            ("Remove",            self._cmd_remove),
        ]:
            ttk.Button(btn_f, text=label, command=cmd).pack(side="left", padx=3)

        # Context menu
        self._context_menu = tk.Menu(self, tearoff=0, bg=BG2, fg=FG,
                                     activebackground=SEL_BG, activeforeground=ACCENT)
        for label, cmd in [
            ("Toggle Visibility", self._cmd_toggle_vis),
            ("Train / Resume",    self._cmd_resume),
            ("Run",               self._cmd_run),
            ("Pause",             self._cmd_pause),
            ("Stop",              self._cmd_stop),
            ("Remove",            self._cmd_remove),
        ]:
            self._context_menu.add_command(label=label, command=cmd)

    def _populate_existing(self):
        for job in self._manager.jobs.values():
            self._add_row(job)

    def _add_row(self, job: TrainingJob) -> None:
        ep   = len(job.returns)
        tot  = job.config.ep_cfg.n_episodes
        ret  = job.returns[-1] if job.returns else 0.0
        ma   = job.moving_avg[-1] if job.moving_avg else 0.0
        vis  = "✓" if job.config.visible else "✗"

        self._tree.insert("", "end", iid=job.job_id, values=(
            f"{job.config.algorithm} ({job.name})",
            f"{ep}/{tot}",
            f"{ret:.2f}",
            f"{ma:.2f}",
            "—",
            "—",
            "—",
            vis,
        ))

    def update_job_row(self, event_data: Dict) -> None:
        """Update a tree row with latest episode data. Rate-limited."""
        job_id = event_data.get("job_id", "")
        now = time.time()
        if now - self._last_update.get(job_id, 0) < 0.05:  # max 20 Hz
            return
        self._last_update[job_id] = now

        if not self._tree.exists(job_id):
            job = self._manager.get_job(job_id)
            if job:
                self._add_row(job)
            return

        ep    = event_data.get("episode", 0)
        tot   = event_data.get("total_episodes", 0)
        ret   = event_data.get("return", 0.0)
        ma    = event_data.get("moving_avg", 0.0)
        loss  = event_data.get("loss")
        dur   = event_data.get("duration", 0.0)
        steps = event_data.get("steps", 0)
        job   = self._manager.get_job(job_id)
        vis   = "✓" if (job and job.config.visible) else "✗"

        loss_str = f"{loss:.4f}" if loss is not None else "—"

        try:
            self._tree.item(job_id, values=(
                self._tree.item(job_id, "values")[0],
                f"{ep}/{tot}",
                f"{ret:.2f}",
                f"{ma:.2f}",
                loss_str,
                f"{dur:.2f}",
                str(steps),
                vis,
            ))
        except Exception:
            pass

    def add_job_row(self, job: TrainingJob) -> None:
        if not self._tree.exists(job.job_id):
            self._add_row(job)

    def remove_job_row(self, job_id: str) -> None:
        if self._tree.exists(job_id):
            self._tree.delete(job_id)

    def _selected_job_id(self) -> Optional[str]:
        sel = self._tree.selection()
        return sel[0] if sel else None

    def _on_selection(self, _e=None):
        jid = self._selected_job_id()
        self._on_select_cb(jid)

    def _on_double_click(self, _e=None):
        self._cmd_toggle_vis()

    def _cmd_toggle_vis(self):
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job:
                job.config.visible = not job.config.visible
                self._plot.set_visibility(jid, job.config.visible)
                self._plot.redraw()
                # update row
                vals = list(self._tree.item(jid, "values"))
                vals[7] = "✓" if job.config.visible else "✗"
                self._tree.item(jid, values=vals)

    def _cmd_resume(self):
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job and job.status in (JobStatus.PAUSED, JobStatus.DONE,
                                       JobStatus.PENDING, JobStatus.CANCELLED):
                self._manager.start_job(jid)
            elif job and job.status == JobStatus.RUNNING:
                pass

    def _cmd_run(self):
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job and job.model is not None:
                job.run_validation()

    def _cmd_pause(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.pause(jid)

    def _cmd_stop(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.cancel(jid)

    def _cmd_remove(self):
        jid = self._selected_job_id()
        if jid:
            self._manager.remove(jid)

    def _show_context_menu(self, event):
        row = self._tree.identify_row(event.y)
        if row:
            self._tree.selection_set(row)
        try:
            self._context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._context_menu.grab_release()

    def _sort(self, col: str):
        data = [(self._tree.set(iid, col), iid) for iid in self._tree.get_children("")]
        try:
            data.sort(key=lambda x: float(x[0].replace("—", "0").split("/")[0]))
        except Exception:
            data.sort()
        for i, (_, iid) in enumerate(data):
            self._tree.move(iid, "", i)


# ---------------------------------------------------------------------------
# Main Workbench Window
# ---------------------------------------------------------------------------

class WorkbenchUI:
    """Main application window."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("BipedalWalker RL Workbench")
        self.root.geometry("1400x900")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG)

        setup_ttk_style(self.root)

        self._bus = get_bus()
        self._manager = TrainingManager(self._bus)

        self._status_window: Optional[TrainingStatusWindow] = None
        self._selected_job_id: Optional[str] = None
        self._color_counter = 0
        self._color_map: Dict[str, str] = {}  # job_id -> color

        self._frame_update_id = None
        self._plot_update_id  = None

        self._plot_dirty: Dict[str, bool] = {}  # job_id -> needs redraw
        self._plot_update_pending = False

        self._build_layout()
        self._subscribe_events()
        self._start_timers()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_layout(self):
        # Vertical PanedWindow: top (2/3) + bottom (1/3)
        self._v_pane = ttk.PanedWindow(self.root, orient="vertical")
        self._v_pane.pack(fill="both", expand=True)

        top_frame = ttk.Frame(self._v_pane)
        bot_frame = ttk.Frame(self._v_pane)

        self._v_pane.add(top_frame, weight=2)
        self._v_pane.add(bot_frame, weight=1)

        # Horizontal PanedWindow inside top: left (1/3) config + right (2/3) viz
        self._h_pane = ttk.PanedWindow(top_frame, orient="horizontal")
        self._h_pane.pack(fill="both", expand=True)

        left_frame  = ttk.Frame(self._h_pane)
        right_frame = ttk.Frame(self._h_pane)

        self._h_pane.add(left_frame, weight=1)
        self._h_pane.add(right_frame, weight=2)

        # Config Panel
        self._config_panel = ConfigPanel(left_frame)
        self._config_panel.pack(fill="both", expand=True)
        self._config_panel.set_apply_callback(self._on_apply)

        # Visualization Panel
        self._viz_panel = VisualizationPanel(right_frame)
        self._viz_panel.pack(fill="both", expand=True)

        # Bottom: Plot Panel
        self._plot_panel = PlotPanel(bot_frame)
        self._plot_panel.pack(fill="both", expand=True)
        self._plot_panel.set_button_callback(self._on_button)

    # ------------------------------------------------------------------
    # Event Subscriptions
    # ------------------------------------------------------------------

    def _subscribe_events(self):
        sub = self._bus.subscribe
        sub(EventType.JOB_CREATED,       self._evt_job_created)
        sub(EventType.JOB_STARTED,       self._evt_job_started)
        sub(EventType.JOB_DONE,          self._evt_job_done)
        sub(EventType.JOB_FAILED,        self._evt_job_failed)
        sub(EventType.JOB_CANCELLED,     self._evt_job_status_change)
        sub(EventType.JOB_PAUSED,        self._evt_job_status_change)
        sub(EventType.JOB_RESUMED,       self._evt_job_status_change)
        sub(EventType.JOB_REMOVED,       self._evt_job_removed)
        sub(EventType.EPISODE_COMPLETED, self._evt_episode)
        sub(EventType.TRAINING_DONE,     self._evt_training_done)
        sub(EventType.ERROR,             self._evt_error)

    def _evt_job_created(self, e: Event):
        job = self._manager.get_job(e.job_id)
        if job and self._status_window and self._status_window.winfo_exists():
            self._status_window.add_job_row(job)
        color = self._next_color(e.job_id)
        self._plot_panel.update_job_data(e.job_id, [], [], e.job_id, color)

    def _evt_job_started(self, e: Event):
        self._update_title(e.job_id, "RUNNING")
        running = sum(1 for j in self._manager.jobs.values() if j.status == JobStatus.RUNNING)
        self._plot_panel.set_status(f"Training — {running} job(s) running")
        # Auto-open the status window so the user sees RUNNING state immediately
        if not (self._status_window and self._status_window.winfo_exists()):
            self._do_show_status()

    def _evt_job_done(self, e: Event):
        self._update_title(e.job_id, "DONE")
        running = sum(1 for j in self._manager.jobs.values() if j.status == JobStatus.RUNNING)
        if running == 0:
            self._plot_panel.set_status("Done")
        else:
            self._plot_panel.set_status(f"Training — {running} job(s) running")

    def _evt_job_failed(self, e: Event):
        self._update_title(e.job_id, "FAILED")
        self._plot_panel.set_status("Failed")
        import traceback as _tb
        detail = e.data if isinstance(e.data, str) else str(e.data)
        messagebox.showerror("Training Error",
                             f"Job {e.job_id} failed:\n{detail}")

    def _evt_job_status_change(self, e: Event):
        pass

    def _evt_job_removed(self, e: Event):
        if self._status_window and self._status_window.winfo_exists():
            self._status_window.remove_job_row(e.job_id)
        self._plot_panel.remove_job(e.job_id)
        if self._selected_job_id == e.job_id:
            self._selected_job_id = None

    def _evt_episode(self, e: Event):
        job = self._manager.get_job(e.job_id)
        if not job:
            return

        data = e.data or {}
        color = self._get_color(e.job_id)

        # Update plot data
        self._plot_panel.update_job_data(
            e.job_id, job.returns, job.moving_avg,
            job.name, color,
        )
        self._plot_dirty[e.job_id] = True

        # Update progress (use this job if it's selected or active)
        if e.job_id == self._selected_job_id or self._selected_job_id is None:
            ep  = data.get("episode", 0)
            tot = data.get("total_episodes", 0)
            self._plot_panel.update_progress(ep, tot)

        # Update status window
        if self._status_window and self._status_window.winfo_exists():
            update_data = {**data, "job_id": e.job_id}
            self._status_window.update_job_row(update_data)

    def _evt_training_done(self, e: Event):
        self._plot_panel.redraw()

    def _evt_error(self, e: Event):
        messagebox.showerror("Error", str(e.data))

    def _update_title(self, job_id: str, status: str):
        pass

    # ------------------------------------------------------------------
    # Timers
    # ------------------------------------------------------------------

    def _start_timers(self):
        self._event_drain_loop()
        self._render_loop()
        self._plot_refresh_loop()

    def _event_drain_loop(self):
        """Drain EventBus every 10ms."""
        self._bus.drain()
        self.root.after(10, self._event_drain_loop)

    def _render_loop(self):
        """Update visualization every N ms.
        Reads the pre-computed frame from the render background thread —
        no PyTorch calls happen here on the UI thread.
        """
        if self._config_panel.visualization_enabled:
            active_job = self._get_active_render_job()
            if active_job:
                frame = active_job.render_mgr.latest_frame
                if frame is not None:
                    self._viz_panel.update_frame(frame)

        interval = self._config_panel.render_interval_ms
        self.root.after(max(10, interval), self._render_loop)

    def _plot_refresh_loop(self):
        """Redraw plot periodically if dirty."""
        if self._plot_dirty:
            self._plot_panel.redraw()
            self._plot_dirty.clear()
        self.root.after(200, self._plot_refresh_loop)

    def _get_active_render_job(self) -> Optional[TrainingJob]:
        """Return job to visualize based on selection or first running."""
        if self._selected_job_id:
            job = self._manager.get_job(self._selected_job_id)
            if job and job.status == JobStatus.RUNNING:
                return job

        # Fall back to first running job
        return self._manager.get_active_job()

    # ------------------------------------------------------------------
    # Button Callbacks
    # ------------------------------------------------------------------

    def _on_button(self, key: str):
        actions = {
            "add_job":        self._do_add_job,
            "train":          self._do_train,
            "status":         self._do_show_status,
            "save_image":     self._do_save_image,
            "save_content":   self._do_save_content,
            "load_content":   self._do_load_content,
            "cancel_training":self._do_cancel_training,
            "reset_training": self._do_reset_training,
            "save_jobs":      self._do_save_jobs,
            "load_jobs":      self._do_load_jobs,
        }
        fn = actions.get(key)
        if fn:
            try:
                fn()
            except Exception as exc:
                import traceback
                traceback.print_exc()
                messagebox.showerror("Error", f"{key}: {exc}")

    def _on_apply(self):
        """Apply & Reset: reset all training."""
        self._do_reset_training()

    def _do_add_job(self):
        """Add one or more jobs based on config settings."""
        import uuid as _uuid
        base_cfg = self._config_panel.get_job_config()

        if self._config_panel.tuning_mode:
            # Parameter sweep (min/max/step)
            param_configs = self._manager.build_tuning_jobs(base_cfg, base_cfg.tuning_cfg)
            for cfg in param_configs:
                job = self._manager.add_job(cfg)
                self._assign_color(job.job_id)

            # Hidden layer architecture sweep
            layer_variants = self._config_panel.tuning_layer_configs
            for layers in layer_variants:
                cfg = copy.deepcopy(base_cfg)
                cfg.job_id = str(_uuid.uuid4())[:8]
                layers_str = ",".join(str(n) for n in layers)
                cfg.name = f"{base_cfg.algorithm} layers={layers_str}"
                # Apply the architecture to all algorithm configs
                for algo_cfg in (cfg.ppo_cfg, cfg.a2c_cfg, cfg.sac_cfg, cfg.td3_cfg):
                    algo_cfg.network.hidden_layers = list(layers)
                job = self._manager.add_job(cfg)
                self._assign_color(job.job_id)

        elif self._config_panel.compare_mode:
            for algo in AlgorithmType:
                cfg = copy.deepcopy(base_cfg)
                import uuid as _uuid
                cfg.job_id = str(_uuid.uuid4())[:8]
                cfg.algorithm = algo.value
                cfg.name = f"{algo.value}-{cfg.job_id}"
                cfg.render_enabled = False  # animation off: 4 render threads compete for GIL
                job = self._manager.add_job(cfg)
                self._assign_color(job.job_id)
        else:
            job = self._manager.add_job(base_cfg)
            self._assign_color(job.job_id)

        if self._status_window and not self._status_window.winfo_exists():
            self._status_window = None

    def _do_train(self):
        """Start all pending/stopped jobs.
        If no jobs exist at all, auto-adds one from the current config first,
        so a single 'Train' click is enough to get started.
        """
        has_any = bool(self._manager.jobs)
        if not has_any:
            self._do_add_job()
        self._plot_panel.set_status("Building model…")
        self._manager.start_all_pending()

    def _do_show_status(self):
        if self._status_window and self._status_window.winfo_exists():
            self._status_window.lift()
            return

        self._status_window = TrainingStatusWindow(
            self.root,
            self._manager,
            self._plot_panel,
            on_select_cb=self._on_status_select,
        )

    def _on_status_select(self, job_id: Optional[str]):
        self._selected_job_id = job_id

    def _do_save_image(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")],
            initialfile="training_plot.png",
        )
        if path:
            try:
                self._plot_panel.save_image(path)
                messagebox.showinfo("Saved", f"Image saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _do_save_content(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="training_data.json",
        )
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(self._plot_panel.get_plot_data(), f, indent=2)
                messagebox.showinfo("Saved", f"Data saved to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _do_load_content(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if path:
            try:
                with open(path) as f:
                    data = json.load(f)
                self._plot_panel.load_plot_data(data)
            except Exception as e:
                messagebox.showerror("Error", str(e))

    def _do_cancel_training(self):
        self._manager.cancel_all()
        self._plot_panel.set_status("Cancelled")

    def _do_reset_training(self):
        self._manager.cancel_all()
        for jid in list(self._manager.jobs.keys()):
            self._manager.remove(jid)
        self._plot_panel.reset()
        self._color_map.clear()
        self._color_counter = 0
        if self._status_window and self._status_window.winfo_exists():
            self._status_window.destroy()
            self._status_window = None

    def _do_save_jobs(self):
        dir_path = filedialog.askdirectory(title="Select Save Directory")
        if not dir_path:
            return
        try:
            for job in self._manager.jobs.values():
                CheckpointManager.save(job, dir_path)
            messagebox.showinfo("Saved", f"Jobs saved to:\n{dir_path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))

    def _do_load_jobs(self):
        dir_path = filedialog.askdirectory(title="Select Load Directory")
        if not dir_path:
            return
        try:
            # Each subdirectory is a job
            loaded = 0
            for sub in os.listdir(dir_path):
                sub_path = os.path.join(dir_path, sub)
                if os.path.isdir(sub_path):
                    job = CheckpointManager.load(sub_path, self._bus)
                    if job:
                        with self._manager._lock:
                            self._manager.jobs[job.job_id] = job
                        self._assign_color(job.job_id)
                        color = self._get_color(job.job_id)
                        self._plot_panel.update_job_data(
                            job.job_id, job.returns, job.moving_avg,
                            job.name, color,
                        )
                        self._bus.publish(Event(EventType.JOB_CREATED, job.job_id))
                        loaded += 1
            self._plot_panel.redraw()
            messagebox.showinfo("Loaded", f"Loaded {loaded} job(s) from:\n{dir_path}")
        except Exception as e:
            messagebox.showerror("Load Error", str(e))
            import traceback; traceback.print_exc()

    # ------------------------------------------------------------------
    # Color Management
    # ------------------------------------------------------------------

    def _assign_color(self, job_id: str) -> str:
        if job_id not in self._color_map:
            algo = None
            job = self._manager.get_job(job_id)
            if job:
                algo = job.config.algorithm
            if algo and algo in ALGO_COLORS:
                # Check if already taken
                used = set(self._color_map.values())
                color = ALGO_COLORS[algo]
                if color in used:
                    color = PALETTE[self._color_counter % len(PALETTE)]
                    self._color_counter += 1
            else:
                color = PALETTE[self._color_counter % len(PALETTE)]
                self._color_counter += 1
            self._color_map[job_id] = color
        return self._color_map[job_id]

    def _get_color(self, job_id: str) -> str:
        if job_id not in self._color_map:
            return self._assign_color(job_id)
        return self._color_map[job_id]

    def _next_color(self, job_id: str) -> str:
        return self._assign_color(job_id)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _on_close(self):
        self._manager.cancel_all()
        # Close all render envs
        for job in self._manager.jobs.values():
            job.render_mgr.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
