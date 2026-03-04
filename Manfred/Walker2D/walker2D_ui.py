"""
UI layer for the Walker2D RL Workbench.
Dark-themed Tkinter + ttk application.
"""
from __future__ import annotations

import json
import os
import queue
from copy import deepcopy
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
import tkinter.ttk as ttk

import numpy as np

from walker2D_logic import (
    ALGO_CONFIGS,
    CheckpointManager,
    EnvConfig,
    EpisodeConfig,
    EventBus,
    EventType,
    JobStatus,
    NetworkConfig,
    PPOConfig,
    SACConfig,
    TD3Config,
    TrainingManager,
    expand_tuning_values,
)

# ─────────────────────────────────────────────────────────────────────────────
# Palette
# ─────────────────────────────────────────────────────────────────────────────
BG      = "#0f111a"
BG2     = "#161824"
BG3     = "#1e2130"
ACCENT  = "#4cc9f0"
ACCENT2 = "#7b5ea7"
FG      = "#e6e6e6"
FG2     = "#b5b5b5"
BORDER  = "#2a2f3a"
SEL     = "#2a3550"
BTN_BG  = "#232638"
BTN_HOV = "#2e3352"

PLOT_COLORS = [
    "#4cc9f0", "#f72585", "#7b5ea7",
    "#06d6a0", "#ffd166", "#ef4444",
]

FONT      = ("Segoe UI", 10)
FONT_BOLD = ("Segoe UI", 10, "bold")
FONT_H    = ("Segoe UI", 11, "bold")
FONT_SM   = ("Segoe UI", 9)


# ─────────────────────────────────────────────────────────────────────────────
# Theme
# ─────────────────────────────────────────────────────────────────────────────

def apply_theme(root: tk.Tk) -> None:
    s = ttk.Style(root)
    s.theme_use("clam")
    s.configure(".", background=BG, foreground=FG,
                 fieldbackground=BG2, troughcolor=BG2,
                 selectbackground=SEL, selectforeground=FG, font=FONT)
    for w in ("TFrame", "TLabelframe"):
        s.configure(w, background=BG)
    s.configure("TLabel",    background=BG,  foreground=FG,  font=FONT)
    s.configure("TEntry",    fieldbackground=BG2, foreground=FG,
                insertcolor=FG, bordercolor=BORDER, lightcolor=BG2, darkcolor=BG2)
    s.configure("TCombobox", fieldbackground=BG2, background=BG2,
                foreground=FG, selectbackground=SEL, arrowcolor=FG2)
    s.configure("TCheckbutton", background=BG, foreground=FG)
    s.configure("TNotebook",     background=BG2, bordercolor=BORDER)
    s.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                padding=[8, 3], font=FONT)
    s.map("TNotebook.Tab",
          background=[("selected", BG2)],
          foreground=[("selected", ACCENT)])
    s.configure("TScrollbar", background=BG3, troughcolor=BG2,
                arrowcolor=FG2, bordercolor=BORDER)
    s.configure("TButton", background=BTN_BG, foreground=FG,
                bordercolor=BORDER, padding=[10, 5], font=FONT)
    s.map("TButton",
          background=[("active", BTN_HOV), ("disabled", BG3)],
          foreground=[("disabled", FG2)])
    s.configure("Accent.TButton", background=ACCENT2, foreground=FG,
                bordercolor=ACCENT2, padding=[10, 5], font=FONT_BOLD)
    s.map("Accent.TButton", background=[("active", "#9b6ec7")])
    s.configure("Compact.TButton", background=BTN_BG, foreground=FG,
                bordercolor=BORDER, padding=[5, 3], font=FONT_SM)
    s.map("Compact.TButton", background=[("active", BTN_HOV)])
    s.configure("Treeview", background=BG2, foreground=FG,
                fieldbackground=BG2, bordercolor=BORDER, rowheight=24, font=FONT)
    s.configure("Treeview.Heading", background=BG3, foreground=FG2,
                bordercolor=BORDER, font=FONT_BOLD)
    s.map("Treeview",
          background=[("selected", SEL)],
          foreground=[("selected", FG)])
    s.configure("TProgressbar", troughcolor=BG2, background=ACCENT,
                bordercolor=BORDER)
    s.configure("TSeparator", background=BORDER)
    s.configure("TLabelframe",       background=BG, bordercolor=BORDER, labelmargins=4)
    s.configure("TLabelframe.Label", background=BG, foreground=ACCENT, font=FONT_BOLD)
    root.configure(background=BG)
    root.option_add("*tearOff", False)


def _lf(parent, text: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=text, style="TLabelframe")


def _row(parent, label: str, var: tk.Variable,
         r: int, c: int = 0, w: int = 9) -> ttk.Entry:
    ttk.Label(parent, text=label).grid(row=r, column=c,     sticky="w", padx=(4, 2), pady=3)
    e = ttk.Entry(parent, textvariable=var, width=w)
    e.grid(row=r, column=c + 1, sticky="ew", padx=(0, 6), pady=3)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# ConfigPanel
# ─────────────────────────────────────────────────────────────────────────────

class ConfigPanel(ttk.Frame):

    ALGO_NAMES = ["PPO", "SAC", "TD3"]

    def __init__(self, parent: tk.Widget, manager: TrainingManager,
                 on_apply: callable) -> None:
        super().__init__(parent)
        self._manager  = manager
        self._on_apply = on_apply
        self._tune_items: List[Tuple[str, str, str]] = []
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)

        cvs = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        sb  = ttk.Scrollbar(self, orient="vertical", command=cvs.yview)
        cvs.configure(yscrollcommand=sb.set)
        cvs.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)

        inner = ttk.Frame(cvs)
        inner.columnconfigure(0, weight=1)
        cvs.create_window((0, 0), window=inner, anchor="nw", tags="inner")

        def _resize(e):
            cvs.configure(scrollregion=cvs.bbox("all"))
            cvs.itemconfig("inner", width=e.width)

        cvs.bind("<Configure>", _resize)
        inner.bind("<Configure>", lambda e: cvs.configure(
            scrollregion=cvs.bbox("all")))
        cvs.bind_all("<MouseWheel>",
                     lambda e: cvs.yview_scroll(int(-1 * e.delta / 120), "units"))

        r = 0

        # ── Environment Configuration ─────────────────────────────────
        fe = _lf(inner, "Environment Configuration")
        fe.grid(row=r, column=0, sticky="ew", padx=4, pady=(6, 2))
        fe.columnconfigure(1, weight=1); fe.columnconfigure(3, weight=1)
        r += 1

        self._fwd_reward_w = tk.DoubleVar(value=1.0)
        self._ctrl_cost_w  = tk.DoubleVar(value=0.001)
        self._healthy_rew  = tk.DoubleVar(value=1.0)
        self._term_unhlt   = tk.BooleanVar(value=True)
        self._hz_min       = tk.DoubleVar(value=0.8)
        self._hz_max       = tk.DoubleVar(value=2.0)
        self._ha_min       = tk.DoubleVar(value=-1.0)
        self._ha_max       = tk.DoubleVar(value=1.0)
        self._noise_scale  = tk.DoubleVar(value=0.005)
        self._excl_pos     = tk.BooleanVar(value=True)
        self._render_ms    = tk.IntVar(value=10)
        self._visualize    = tk.BooleanVar(value=True)

        _row(fe, "Fwd Reward W",   self._fwd_reward_w, 0, 0, 7)
        _row(fe, "Ctrl Cost W",    self._ctrl_cost_w,  0, 2, 7)
        _row(fe, "Healthy Reward", self._healthy_rew,  1, 0, 7)
        ttk.Checkbutton(fe, text="Term. Unhealthy",
                        variable=self._term_unhlt).grid(
            row=1, column=2, columnspan=2, sticky="w", padx=4)
        _row(fe, "Healthy Z min",  self._hz_min,  2, 0, 7)
        _row(fe, "Healthy Z max",  self._hz_max,  2, 2, 7)
        _row(fe, "Angle min",      self._ha_min,  3, 0, 7)
        _row(fe, "Angle max",      self._ha_max,  3, 2, 7)
        _row(fe, "Noise Scale",    self._noise_scale, 4, 0, 7)
        ttk.Checkbutton(fe, text="Excl. Cur. Pos.",
                        variable=self._excl_pos).grid(
            row=4, column=2, columnspan=2, sticky="w", padx=4)
        _row(fe, "Render ms", self._render_ms, 5, 0, 7)
        ttk.Checkbutton(fe, text="Visualize",
                        variable=self._visualize).grid(
            row=5, column=2, columnspan=2, sticky="w", padx=4)

        # ── Episode Configuration ─────────────────────────────────────
        fp = _lf(inner, "Episode Configuration")
        fp.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        fp.columnconfigure(1, weight=1); fp.columnconfigure(3, weight=1)
        r += 1

        self._n_episodes = tk.IntVar(value=3000)
        self._max_steps  = tk.IntVar(value=1000)
        self._alpha      = tk.DoubleVar(value=3e-4)
        self._gamma      = tk.DoubleVar(value=0.99)
        self._compare    = tk.BooleanVar(value=False)

        _row(fp, "Episodes",  self._n_episodes, 0, 0, 7)
        _row(fp, "Max Steps", self._max_steps,  0, 2, 7)
        _row(fp, "Alpha",     self._alpha,      1, 0, 7)
        _row(fp, "Gamma",     self._gamma,      1, 2, 7)
        ttk.Checkbutton(fp, text="Compare Methods",
                        variable=self._compare).grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        # ── Parameter Tuning ─────────────────────────────────────────
        ft = _lf(inner, "Parameter Tuning")
        ft.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        ft.columnconfigure(1, weight=1); ft.columnconfigure(3, weight=1)
        r += 1

        self._tune_enabled   = tk.BooleanVar(value=False)
        self._tune_algo_var  = tk.StringVar(value="PPO")
        self._tune_param_var = tk.StringVar(value="alpha")
        self._tune_values    = tk.StringVar(value="1e-4;3e-4;1e-3")

        ttk.Checkbutton(ft, text="Enable Parameter Tuning",
                        variable=self._tune_enabled).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)
        ttk.Label(ft, text="Method").grid(row=1, column=0, sticky="w", padx=(4, 2))
        ttk.Combobox(ft, textvariable=self._tune_algo_var,
                     values=self.ALGO_NAMES, state="readonly",
                     width=8).grid(row=1, column=1, sticky="ew", padx=(0, 6), pady=2)
        ttk.Label(ft, text="Parameter").grid(row=1, column=2, sticky="w", padx=(4, 2))
        ttk.Combobox(ft, textvariable=self._tune_param_var,
                     values=["alpha", "gamma", "buffer_size",
                             "batch_size", "hidden_layers"],
                     state="readonly", width=12).grid(
            row=1, column=3, sticky="ew", padx=(0, 4), pady=2)
        ttk.Label(ft, text="Values (;)").grid(row=2, column=0, sticky="w", padx=(4, 2))
        ttk.Entry(ft, textvariable=self._tune_values).grid(
            row=2, column=1, columnspan=3, sticky="ew", padx=(0, 4), pady=2)

        tf = ttk.Frame(ft)
        tf.grid(row=3, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        tf.columnconfigure(0, weight=1)
        self._tune_tree = ttk.Treeview(
            tf, columns=("method", "param", "values"),
            show="headings", height=3, selectmode="browse")
        for col, txt, w in [("method", "Method", 80),
                              ("param",  "Param",  90),
                              ("values", "Values", 140)]:
            self._tune_tree.heading(col, text=txt)
            self._tune_tree.column(col, width=w, stretch=(col == "values"))
        self._tune_tree.grid(row=0, column=0, sticky="ew")

        bf = ttk.Frame(ft)
        bf.grid(row=4, column=0, columnspan=4, sticky="ew", padx=4, pady=2)
        ttk.Button(bf, text="Add to List",     command=self._tune_add).pack(side="left", padx=2)
        ttk.Button(bf, text="Remove Selected", command=self._tune_remove).pack(side="left", padx=2)
        ttk.Button(bf, text="Clear List",      command=self._tune_clear).pack(side="left", padx=2)

        # ── RL Methods tabs ───────────────────────────────────────────
        fm = _lf(inner, "RL Methods")
        fm.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        fm.columnconfigure(0, weight=1)
        r += 1

        self._nb = ttk.Notebook(fm)
        self._nb.pack(fill="both", expand=True, padx=2, pady=2)
        self._nb.add(self._build_ppo_tab(), text=" PPO ")
        self._nb.add(self._build_sac_tab(), text=" SAC ")
        self._nb.add(self._build_td3_tab(), text=" TD3 ")

        # ── Apply & Reset ─────────────────────────────────────────────
        ttk.Button(inner, text="Apply & Reset",
                   style="Accent.TButton",
                   command=self._apply).grid(
            row=r, column=0, sticky="ew", padx=4, pady=6)

    # ── Method tabs ───────────────────────────────────────────────────

    def _build_ppo_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self._nb)
        tab.columnconfigure(1, weight=1); tab.columnconfigure(3, weight=1)
        self._ppo_n_steps    = tk.IntVar(value=2048)
        self._ppo_batch_size = tk.IntVar(value=64)
        self._ppo_n_epochs   = tk.IntVar(value=10)
        self._ppo_clip_range = tk.DoubleVar(value=0.2)
        self._ppo_ent_coef   = tk.DoubleVar(value=0.0)
        self._ppo_vf_coef    = tk.DoubleVar(value=0.5)
        self._ppo_grad_norm  = tk.DoubleVar(value=0.5)
        self._ppo_gae_lambda = tk.DoubleVar(value=0.95)
        self._ppo_hidden     = tk.StringVar(value="256,256")
        self._ppo_act        = tk.StringVar(value="relu")
        _row(tab, "N Steps",    self._ppo_n_steps,    0, 0, 7)
        _row(tab, "Batch Size", self._ppo_batch_size, 0, 2, 7)
        _row(tab, "N Epochs",   self._ppo_n_epochs,   1, 0, 7)
        _row(tab, "Clip Range", self._ppo_clip_range, 1, 2, 7)
        _row(tab, "Ent Coef",   self._ppo_ent_coef,   2, 0, 7)
        _row(tab, "VF Coef",    self._ppo_vf_coef,    2, 2, 7)
        _row(tab, "Grad Norm",  self._ppo_grad_norm,  3, 0, 7)
        _row(tab, "GAE λ",      self._ppo_gae_lambda, 3, 2, 7)
        _row(tab, "Hidden",     self._ppo_hidden,     4, 0, 12)
        ttk.Label(tab, text="Activation").grid(row=4, column=2, sticky="w", padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._ppo_act,
                     values=["relu", "tanh", "elu"], state="readonly", width=7).grid(
            row=4, column=3, sticky="ew", padx=(0, 4), pady=2)
        return tab

    def _build_sac_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self._nb)
        tab.columnconfigure(1, weight=1); tab.columnconfigure(3, weight=1)
        self._sac_buf     = tk.IntVar(value=300_000)
        self._sac_batch   = tk.IntVar(value=256)
        self._sac_starts  = tk.IntVar(value=10_000)
        self._sac_trainf  = tk.IntVar(value=1)
        self._sac_grads   = tk.IntVar(value=1)
        self._sac_tau     = tk.DoubleVar(value=0.005)
        self._sac_ent     = tk.StringVar(value="auto")
        self._sac_tgt_upd = tk.IntVar(value=1)
        self._sac_hidden  = tk.StringVar(value="256,256")
        self._sac_act     = tk.StringVar(value="relu")
        _row(tab, "Buffer Size",   self._sac_buf,     0, 0, 10)
        _row(tab, "Batch Size",    self._sac_batch,   0, 2, 7)
        _row(tab, "Learn Starts",  self._sac_starts,  1, 0, 10)
        _row(tab, "Train Freq",    self._sac_trainf,  1, 2, 7)
        _row(tab, "Grad Steps",    self._sac_grads,   2, 0, 10)
        _row(tab, "Tau",           self._sac_tau,     2, 2, 7)
        _row(tab, "Ent Coef",      self._sac_ent,     3, 0, 10)
        _row(tab, "Target Upd",    self._sac_tgt_upd, 3, 2, 7)
        _row(tab, "Hidden",        self._sac_hidden,  4, 0, 12)
        ttk.Label(tab, text="Activation").grid(row=4, column=2, sticky="w", padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._sac_act,
                     values=["relu", "tanh", "elu"], state="readonly", width=7).grid(
            row=4, column=3, sticky="ew", padx=(0, 4), pady=2)
        return tab

    def _build_td3_tab(self) -> ttk.Frame:
        tab = ttk.Frame(self._nb)
        tab.columnconfigure(1, weight=1); tab.columnconfigure(3, weight=1)
        self._td3_buf      = tk.IntVar(value=300_000)
        self._td3_batch    = tk.IntVar(value=256)
        self._td3_starts   = tk.IntVar(value=10_000)
        self._td3_trainf   = tk.IntVar(value=1)
        self._td3_grads    = tk.IntVar(value=1)
        self._td3_tau      = tk.DoubleVar(value=0.005)
        self._td3_pol_del  = tk.IntVar(value=2)
        self._td3_tgt_noi  = tk.DoubleVar(value=0.2)
        self._td3_noi_clip = tk.DoubleVar(value=0.5)
        self._td3_act_noi  = tk.DoubleVar(value=0.1)
        self._td3_hidden   = tk.StringVar(value="256,256")
        self._td3_act      = tk.StringVar(value="relu")
        _row(tab, "Buffer Size",    self._td3_buf,      0, 0, 10)
        _row(tab, "Batch Size",     self._td3_batch,    0, 2, 7)
        _row(tab, "Learn Starts",   self._td3_starts,   1, 0, 10)
        _row(tab, "Train Freq",     self._td3_trainf,   1, 2, 7)
        _row(tab, "Grad Steps",     self._td3_grads,    2, 0, 10)
        _row(tab, "Tau",            self._td3_tau,      2, 2, 7)
        _row(tab, "Policy Delay",   self._td3_pol_del,  3, 0, 10)
        _row(tab, "Target Noise",   self._td3_tgt_noi,  3, 2, 7)
        _row(tab, "Noise Clip",     self._td3_noi_clip, 4, 0, 10)
        _row(tab, "Action Noise σ", self._td3_act_noi,  4, 2, 7)
        _row(tab, "Hidden",         self._td3_hidden,   5, 0, 12)
        ttk.Label(tab, text="Activation").grid(row=5, column=2, sticky="w", padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._td3_act,
                     values=["relu", "tanh", "elu"], state="readonly", width=7).grid(
            row=5, column=3, sticky="ew", padx=(0, 4), pady=2)
        return tab

    # ── Tune helpers ──────────────────────────────────────────────────

    def _tune_add(self) -> None:
        algo   = self._tune_algo_var.get()
        param  = self._tune_param_var.get()
        values = self._tune_values.get().strip()
        if not param or not values:
            return
        iid = self._tune_tree.insert("", "end", values=(algo, param, values))
        self._tune_items.append((param, values, iid))

    def _tune_remove(self) -> None:
        sel = self._tune_tree.selection()
        if sel:
            self._tune_tree.delete(sel[0])
            self._tune_items = [i for i in self._tune_items if i[2] != sel[0]]

    def _tune_clear(self) -> None:
        for iid in self._tune_tree.get_children():
            self._tune_tree.delete(iid)
        self._tune_items.clear()

    # ── Read helpers ──────────────────────────────────────────────────

    def _parse_hidden(self, var: tk.StringVar) -> List[int]:
        try:
            return [int(x.strip()) for x in var.get().split(",") if x.strip()]
        except ValueError:
            return [256, 256]

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            forward_reward_weight                    = float(self._fwd_reward_w.get()),
            ctrl_cost_weight                         = float(self._ctrl_cost_w.get()),
            healthy_reward                           = float(self._healthy_rew.get()),
            terminate_when_unhealthy                 = bool(self._term_unhlt.get()),
            healthy_z_min                            = float(self._hz_min.get()),
            healthy_z_max                            = float(self._hz_max.get()),
            healthy_angle_min                        = float(self._ha_min.get()),
            healthy_angle_max                        = float(self._ha_max.get()),
            reset_noise_scale                        = float(self._noise_scale.get()),
            exclude_current_positions_from_observation = bool(self._excl_pos.get()),
            render_interval_ms                       = int(self._render_ms.get()),
            visualize                                = bool(self._visualize.get()),
        )

    def get_ep_config(self) -> EpisodeConfig:
        return EpisodeConfig(
            n_episodes      = int(self._n_episodes.get()),
            max_steps       = int(self._max_steps.get()),
            alpha           = float(self._alpha.get()),
            gamma           = float(self._gamma.get()),
            compare_methods = bool(self._compare.get()),
        )

    def get_ppo_config(self) -> PPOConfig:
        return PPOConfig(
            n_steps       = int(self._ppo_n_steps.get()),
            batch_size    = int(self._ppo_batch_size.get()),
            n_epochs      = int(self._ppo_n_epochs.get()),
            clip_range    = float(self._ppo_clip_range.get()),
            ent_coef      = float(self._ppo_ent_coef.get()),
            vf_coef       = float(self._ppo_vf_coef.get()),
            max_grad_norm = float(self._ppo_grad_norm.get()),
            gae_lambda    = float(self._ppo_gae_lambda.get()),
            network       = NetworkConfig(
                hidden_layers=self._parse_hidden(self._ppo_hidden),
                activation=self._ppo_act.get()),
        )

    def get_sac_config(self) -> SACConfig:
        return SACConfig(
            buffer_size           = int(self._sac_buf.get()),
            batch_size            = int(self._sac_batch.get()),
            learning_starts       = int(self._sac_starts.get()),
            train_freq            = int(self._sac_trainf.get()),
            gradient_steps        = int(self._sac_grads.get()),
            tau                   = float(self._sac_tau.get()),
            ent_coef              = self._sac_ent.get(),
            target_update_interval= int(self._sac_tgt_upd.get()),
            network               = NetworkConfig(
                hidden_layers=self._parse_hidden(self._sac_hidden),
                activation=self._sac_act.get()),
        )

    def get_td3_config(self) -> TD3Config:
        return TD3Config(
            buffer_size        = int(self._td3_buf.get()),
            batch_size         = int(self._td3_batch.get()),
            learning_starts    = int(self._td3_starts.get()),
            train_freq         = int(self._td3_trainf.get()),
            gradient_steps     = int(self._td3_grads.get()),
            tau                = float(self._td3_tau.get()),
            policy_delay       = int(self._td3_pol_del.get()),
            target_noise       = float(self._td3_tgt_noi.get()),
            noise_clip         = float(self._td3_noi_clip.get()),
            action_noise_sigma = float(self._td3_act_noi.get()),
            network            = NetworkConfig(
                hidden_layers=self._parse_hidden(self._td3_hidden),
                activation=self._td3_act.get()),
        )

    def get_visualize(self) -> bool:
        return bool(self._visualize.get())

    def _apply(self) -> None:
        if callable(self._on_apply):
            self._on_apply()


# ─────────────────────────────────────────────────────────────────────────────
# VisualizationPanel
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationPanel(ttk.Frame):

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._img_ref     = None
        self._debounce_id = None
        self._cw = self._ch = 1

        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_resize)

        self._lbl = ttk.Label(self, text="No visualization",
                              foreground=FG2, background=BG, font=FONT_H)
        self._lbl.place(relx=0.5, rely=0.5, anchor="center")

    def _on_resize(self, e: tk.Event) -> None:
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._cw = max(e.width, 1)
        self._ch = max(e.height, 1)
        self._debounce_id = self.after(100, self._redraw)

    def _redraw(self) -> None:
        if self._img_ref:
            self._canvas.delete("all")
            self._canvas.create_image(
                self._cw // 2, self._ch // 2,
                image=self._img_ref, anchor="center")

    def update_frame(self, frame) -> None:
        from PIL import Image, ImageTk
        self._lbl.place_forget()
        h, w = frame.shape[:2]
        scale = min(self._cw / w, self._ch / h)
        nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
        img = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self._img_ref = tk_img
        self._canvas.delete("all")
        self._canvas.create_image(
            self._cw // 2, self._ch // 2, image=tk_img, anchor="center")

    def clear(self) -> None:
        self._img_ref = None
        self._canvas.delete("all")
        self._lbl.place(relx=0.5, rely=0.5, anchor="center")


# ─────────────────────────────────────────────────────────────────────────────
# PlotPanel
# ─────────────────────────────────────────────────────────────────────────────

class PlotPanel(ttk.Frame):

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._job_data: Dict[str, dict] = {}
        self._debounce_id = None
        self._resize_id   = None
        self._resizing    = False
        self._build()

    def _build(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        import matplotlib.pyplot as plt

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._fig, self._ax = plt.subplots(figsize=(8, 3))
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(BG)
        self._ax.tick_params(colors=FG2)
        self._ax.xaxis.label.set_color(FG2)
        self._ax.yaxis.label.set_color(FG2)
        self._ax.set_xlabel("Episode")
        self._ax.set_ylabel("Return")
        for sp in self._ax.spines.values():
            sp.set_edgecolor(BORDER)
        self._ax.grid(color=BORDER, linestyle="--", alpha=0.5)
        self._fig.tight_layout(pad=0.5)

        self._mpl = FigureCanvasTkAgg(self._fig, master=self)
        self._mpl.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, _: tk.Event) -> None:
        if self._resize_id:
            self.after_cancel(self._resize_id)
        self._resizing  = True
        self._resize_id = self.after(80, self._finish_resize)

    def _finish_resize(self) -> None:
        self._resizing = False
        self._redraw()

    def update_job(self, job_id: str, label: str, returns: List[float],
                   moving_avg: List[float], visible: bool, ci: int) -> None:
        self._job_data[job_id] = dict(label=label, returns=returns,
                                      moving_avg=moving_avg,
                                      visible=visible, ci=ci)
        if not self._resizing:
            if self._debounce_id:
                self.after_cancel(self._debounce_id)
            self._debounce_id = self.after(50, self._redraw)

    def remove_job(self, job_id: str) -> None:
        self._job_data.pop(job_id, None)
        self._redraw()

    def toggle_visibility(self, job_id: str, visible: bool) -> None:
        if job_id in self._job_data:
            self._job_data[job_id]["visible"] = visible
            self._redraw()

    def _redraw(self) -> None:
        ax = self._ax
        ax.clear()
        ax.set_facecolor(BG)
        ax.tick_params(colors=FG2)
        ax.xaxis.label.set_color(FG2)
        ax.yaxis.label.set_color(FG2)
        ax.set_xlabel("Episode"); ax.set_ylabel("Return")
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.grid(color=BORDER, linestyle="--", alpha=0.5)

        for d in self._job_data.values():
            if not d["visible"]:
                continue
            c  = PLOT_COLORS[d["ci"] % len(PLOT_COLORS)]
            ep = list(range(1, len(d["returns"]) + 1))
            if d["returns"]:
                ax.plot(ep, d["returns"], color=c, alpha=0.35,
                        linewidth=1.0, label=f"{d['label']} Raw")
            if d["moving_avg"]:
                ax.plot(list(range(1, len(d["moving_avg"]) + 1)),
                        d["moving_avg"], color=c, alpha=1.0,
                        linewidth=2.5, label=f"{d['label']} Avg")

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="upper left",
                      bbox_to_anchor=(1.0, 1.0), facecolor=BG,
                      edgecolor=BORDER, labelcolor="#e6e6e6", fontsize=8)
        try:
            self._fig.tight_layout(rect=[0, 0, 0.80, 1.0], pad=0.5)
        except Exception:
            pass
        try:
            self._mpl.draw_idle()
        except Exception:
            pass

    def save_image(self, path: str) -> None:
        self._fig.savefig(path, facecolor=BG, dpi=150)

    def get_data(self) -> dict:
        return {jid: dict(label=d["label"], returns=d["returns"],
                          moving_avg=d["moving_avg"])
                for jid, d in self._job_data.items()}

    def load_data(self, data: dict) -> None:
        idx = len(self._job_data)
        for jid, d in data.items():
            self._job_data[jid] = dict(
                label=d.get("label", jid),
                returns=d.get("returns", []),
                moving_avg=d.get("moving_avg", []),
                visible=True, ci=idx)
            idx += 1
        self._redraw()


# ─────────────────────────────────────────────────────────────────────────────
# Training Status Window
# ─────────────────────────────────────────────────────────────────────────────

class TrainingStatusWindow(tk.Toplevel):

    COLS = ("label", "episode", "return", "moving_avg",
            "loss", "duration", "steps", "status", "visible")

    def __init__(self, parent: tk.Widget, manager: TrainingManager,
                 on_select_job: callable) -> None:
        super().__init__(parent)
        self.title("Training Status")
        self.configure(bg=BG)
        self.geometry("960x420")
        self.protocol("WM_DELETE_WINDOW", self.withdraw)
        self._manager   = manager
        self._on_select = on_select_job
        self._sort_col  = None
        self._sort_rev  = False
        self._build()

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        frm = ttk.Frame(self)
        frm.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        frm.columnconfigure(0, weight=1)
        frm.rowconfigure(0, weight=1)

        self._tv = ttk.Treeview(frm, columns=self.COLS,
                                show="headings", selectmode="browse")
        specs = {
            "label":      ("Algorithm",  140, True),
            "episode":    ("Episode",     80, False),
            "return":     ("Return",      80, False),
            "moving_avg": ("MovingAvg",   80, False),
            "loss":       ("Loss",        80, False),
            "duration":   ("Duration(s)", 80, False),
            "steps":      ("Steps",       70, False),
            "status":     ("Status",      80, False),
            "visible":    ("Visible",     60, False),
        }
        for col, (txt, w, st) in specs.items():
            self._tv.heading(col, text=txt,
                             command=lambda c=col: self._sort(c))
            self._tv.column(col, width=w, stretch=st)

        sby = ttk.Scrollbar(frm, orient="vertical",   command=self._tv.yview)
        sbx = ttk.Scrollbar(frm, orient="horizontal", command=self._tv.xview)
        self._tv.configure(yscrollcommand=sby.set, xscrollcommand=sbx.set)
        self._tv.grid(row=0, column=0, sticky="nsew")
        sby.grid(row=0, column=1, sticky="ns")
        sbx.grid(row=1, column=0, sticky="ew")

        self._tv.bind("<<TreeviewSelect>>", self._on_sel)
        self._tv.bind("<Double-1>",         lambda _: self._toggle())
        self._tv.bind("<Button-3>",         self._on_rclick)
        self._tv.bind("<Return>",           lambda _: self._toggle())
        self._tv.bind("<space>",            lambda _: self._pause_resume())

        self._ctx = tk.Menu(self, tearoff=False, bg=BG2, fg=FG,
                            activebackground=SEL, activeforeground=FG)
        for lbl, cmd in [
            ("Toggle Visibility", self._toggle),
            ("Train",  self._train),
            ("Run",    self._run),
            ("Pause",  self._pause_resume),
            ("Stop",   self._stop),
            ("Remove", self._remove),
        ]:
            self._ctx.add_command(label=lbl, command=cmd)

        bf = ttk.Frame(self)
        bf.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        for lbl, cmd in [
            ("Toggle Visible", self._toggle),
            ("Train",          self._train),
            ("Run",            self._run),
            ("Pause/Resume",   self._pause_resume),
            ("Stop",           self._stop),
            ("Remove",         self._remove),
        ]:
            ttk.Button(bf, text=lbl, command=cmd, width=12).pack(side="left", padx=2)

    def _on_sel(self, _: tk.Event) -> None:
        jid = self._sel_job_id()
        if jid and callable(self._on_select):
            self._on_select(jid)

    def _on_rclick(self, e: tk.Event) -> None:
        row = self._tv.identify_row(e.y)
        if row:
            self._tv.selection_set(row)
        self._ctx.post(e.x_root, e.y_root)

    def _sel_job_id(self) -> Optional[str]:
        sel = self._tv.selection()
        if not sel:
            return None
        tags = self._tv.item(sel[0], "tags")
        return tags[0] if tags else None

    def _toggle(self) -> None:
        jid = self._sel_job_id()
        if jid:
            self._manager.toggle_visibility(jid)
            self._update_row(jid)

    def _train(self) -> None:
        jid = self._sel_job_id()
        if jid:
            self._manager.start_job(jid)

    def _run(self) -> None:
        jid = self._sel_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job and job.model is not None:
                self._manager.run_inference(jid)

    def _pause_resume(self) -> None:
        jid = self._sel_job_id()
        if not jid:
            return
        job = self._manager.get_job(jid)
        if job:
            if job.status == JobStatus.RUNNING:
                self._manager.pause(jid)
            elif job.status == JobStatus.PAUSED:
                self._manager.resume(jid)

    def _stop(self) -> None:
        jid = self._sel_job_id()
        if jid:
            self._manager.cancel(jid)

    def _remove(self) -> None:
        jid = self._sel_job_id()
        if jid and messagebox.askyesno("Remove Job",
                                        "Remove this training job?",
                                        parent=self):
            self._manager.remove(jid)
            self.refresh_all()

    def _sort(self, col: str) -> None:
        self._sort_rev = (not self._sort_rev) if self._sort_col == col else False
        self._sort_col = col
        data = [(self._tv.set(iid, col), iid) for iid in self._tv.get_children()]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=self._sort_rev)
        except ValueError:
            data.sort(key=lambda t: t[0], reverse=self._sort_rev)
        for idx, (_, iid) in enumerate(data):
            self._tv.move(iid, "", idx)

    def upsert_job(self, job_id: str, **kw) -> None:
        vals = (
            kw.get("label", job_id[:8]),
            f"{kw.get('episode', 0)}/{kw.get('n_ep', '?')}",
            f"{kw.get('ret', 0.0):.1f}",
            f"{kw.get('mavg', 0.0):.1f}",
            f"{kw.get('loss', '')}" if kw.get("loss") is not None else "",
            f"{kw.get('dur', 0.0):.2f}",
            str(kw.get("steps", 0)),
            kw.get("status", ""),
            "✓" if kw.get("visible", True) else "✗",
        )
        iids = self._tv.tag_has(job_id)
        if iids:
            self._tv.item(iids[0], values=vals)
        else:
            self._tv.insert("", "end", values=vals, tags=(job_id,))

    def _update_row(self, job_id: str) -> None:
        job = self._manager.get_job(job_id)
        if not job:
            return
        iids = self._tv.tag_has(job_id)
        if iids:
            old = list(self._tv.item(iids[0], "values"))
            if len(old) >= 9:
                old[8] = "✓" if job.visible else "✗"
                self._tv.item(iids[0], values=tuple(old))

    def refresh_all(self) -> None:
        for iid in self._tv.get_children():
            self._tv.delete(iid)
        for job in self._manager.jobs:
            win  = min(50, len(job.returns))
            mavg = float(np.mean(job.returns[-win:])) if job.returns else 0.0
            self.upsert_job(
                job.job_id, label=job.label,
                episode=job.current_episode,
                n_ep=job.ep_config.n_episodes,
                ret=job.returns[-1] if job.returns else 0.0,
                mavg=mavg,
                status=job.status.value,
                visible=job.visible,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main WorkbenchApp
# ─────────────────────────────────────────────────────────────────────────────

class WorkbenchApp(tk.Tk):

    ALGO_NAMES = ConfigPanel.ALGO_NAMES

    def __init__(self) -> None:
        super().__init__()
        self.title("RL Workbench – Walker2d-v5")
        self.geometry("1400x850")
        self.minsize(900, 600)
        apply_theme(self)

        self._bus     = EventBus()
        self._manager = TrainingManager(self._bus)
        self._bus.subscribe(self._on_event)

        self._job_colors: Dict[str, int] = {}
        self._active_viz_job: Optional[str] = None
        self._compact_buttons = False
        self._resize_debounce = None

        self._build_layout()
        self._start_polling()

    # ── Layout ────────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        outer = ttk.PanedWindow(self, orient="vertical")
        outer.grid(row=0, column=0, sticky="nsew")

        top_pw = ttk.PanedWindow(outer, orient="horizontal")

        ch = ttk.Frame(top_pw)
        ch.columnconfigure(0, weight=1); ch.rowconfigure(0, weight=1)
        self._cfg = ConfigPanel(ch, self._manager, on_apply=self._on_apply)
        self._cfg.grid(row=0, column=0, sticky="nsew")

        vh = ttk.Frame(top_pw)
        vh.columnconfigure(0, weight=1); vh.rowconfigure(0, weight=1)
        self._viz = VisualizationPanel(vh)
        self._viz.grid(row=0, column=0, sticky="nsew")

        top_pw.add(ch, weight=1)
        top_pw.add(vh, weight=2)

        bottom = ttk.Frame(outer)
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_lbl = ttk.Label(bottom, text="No active job",
                                       foreground=FG2, font=FONT_SM)
        self._progress_lbl.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 0))
        ttk.Progressbar(bottom, variable=self._progress_var,
                        maximum=100.0).grid(row=1, column=0, sticky="ew",
                                            padx=4, pady=2)

        self._btn_frame = ttk.Frame(bottom)
        self._btn_frame.grid(row=2, column=0, sticky="ew", padx=4, pady=2)
        self._build_buttons()

        self._plot = PlotPanel(bottom)
        self._plot.grid(row=3, column=0, sticky="nsew", padx=4, pady=4)
        bottom.rowconfigure(3, weight=1)

        outer.add(top_pw, weight=2)
        outer.add(bottom, weight=1)

        self.bind("<Configure>", self._on_win_resize)

    def _build_buttons(self) -> None:
        for w in self._btn_frame.winfo_children():
            w.destroy()
        s = "Compact.TButton" if self._compact_buttons else "TButton"
        for txt, cmd in [
            ("Add Job",         self._add_job),
            ("Train",           self._train),
            ("Training Status", self._open_status),
            ("Save Image",      self._save_image),
            ("Save Content",    self._save_content),
            ("Load Content",    self._load_content),
            ("Save Job",        self._save_job),
            ("Load Job",        self._load_job),
            ("Cancel All",      self._cancel_all),
            ("Reset Training",  self._reset_training),
        ]:
            ttk.Button(self._btn_frame, text=txt, command=cmd,
                       style=s).pack(side="left", padx=2, pady=2)

    def _on_win_resize(self, e: tk.Event) -> None:
        if e.widget is not self:
            return
        if self._resize_debounce:
            self.after_cancel(self._resize_debounce)
        self._resize_debounce = self.after(
            100, lambda w=e.width: self._check_compact(w))

    def _check_compact(self, width: int) -> None:
        new = width < 1100
        if new != self._compact_buttons:
            self._compact_buttons = new
            self._build_buttons()

    # ── Polling ───────────────────────────────────────────────────────

    def _start_polling(self) -> None:
        self._bus.drain()
        self._poll_frames()
        self.after(10, self._start_polling)

    def _poll_frames(self) -> None:
        if not self._active_viz_job:
            for job in self._manager.jobs:
                if job.status == JobStatus.RUNNING and self._cfg.get_visualize():
                    self._active_viz_job = job.job_id
                    break
            return
        job = self._manager.get_job(self._active_viz_job)
        if not job or job.status not in (JobStatus.RUNNING, JobStatus.PAUSED):
            self._active_viz_job = None
            return
        try:
            frame = job.frame_queue.get_nowait()
            self._viz.update_frame(frame)
        except queue.Empty:
            pass

    # ── Event handling ────────────────────────────────────────────────

    def _on_event(self, ev) -> None:
        t = ev.type

        if t == EventType.JOB_CREATED:
            idx = len(self._job_colors)
            self._job_colors[ev.job_id] = idx
            job = self._manager.get_job(ev.job_id)
            if job:
                self._plot.update_job(ev.job_id, job.label, [], [], True, idx)

        elif t == EventType.JOB_STARTED:
            job = self._manager.get_job(ev.job_id)
            self._progress_var.set(0)
            self._progress_lbl.config(
                text=f"{job.label if job else ev.job_id[:8]}  Building model…")

        elif t == EventType.EPISODE_COMPLETED:
            d     = ev.data
            ret   = d.get("return", 0.0)
            mavg  = d.get("moving_avg", 0.0)
            ep    = d.get("episode", 0)
            n_ep  = d.get("n_episodes", 1)
            dur   = d.get("duration", 0.0)
            steps = d.get("steps", 0)
            loss  = d.get("loss")
            rets  = d.get("returns", [])

            job = self._manager.get_job(ev.job_id)
            if job:
                job.returns         = rets
                job.current_episode = ep
                win = 50
                while len(job.moving_avg) < len(rets):
                    i = len(job.moving_avg)
                    job.moving_avg.append(
                        float(np.mean(rets[max(0, i - win + 1):i + 1])))
                if len(job.moving_avg) > len(rets):
                    job.moving_avg = job.moving_avg[:len(rets)]

            pct = (ep / n_ep * 100) if n_ep else 0
            self._progress_var.set(pct)
            lbl = job.label if job else ev.job_id[:8]
            self._progress_lbl.config(
                text=f"{lbl}  Ep {ep}/{n_ep}  Ret {ret:.1f}  Avg {mavg:.1f}")

            if job and job.visible:
                self._plot.update_job(
                    ev.job_id, job.label, job.returns, job.moving_avg,
                    True, self._job_colors.get(ev.job_id, 0))

            if hasattr(self, "_status_win") and self._status_win.winfo_exists():
                self._status_win.upsert_job(
                    ev.job_id, label=job.label if job else "",
                    episode=ep, n_ep=n_ep, ret=ret, mavg=mavg,
                    loss=loss, dur=dur, steps=steps,
                    status=job.status.value if job else "",
                    visible=job.visible if job else True)

        elif t in (EventType.JOB_COMPLETED, EventType.TRAINING_DONE):
            job = self._manager.get_job(ev.job_id)
            if job:
                self._plot.update_job(
                    ev.job_id, job.label, job.returns, job.moving_avg,
                    job.visible, self._job_colors.get(ev.job_id, 0))
            self._progress_var.set(100)
            if self._active_viz_job == ev.job_id:
                self._active_viz_job = None

        elif t == EventType.JOB_REMOVED:
            self._plot.remove_job(ev.job_id)
            self._job_colors.pop(ev.job_id, None)

        elif t == EventType.JOB_ERROR:
            job   = self._manager.get_job(ev.job_id)
            label = job.label if job else ev.job_id[:8]
            err   = ev.data.get("error", "Unknown error")
            tb    = ev.data.get("traceback", "")
            self._progress_lbl.config(text=f"{label}  ✗ Error: {err}")
            messagebox.showerror("Training Error",
                                 f"{err}\n\n{tb}".strip() if tb else err,
                                 parent=self)

    # ── Button actions ────────────────────────────────────────────────

    def _get_algo_cfg(self, name: str) -> Any:
        if name == "PPO":  return self._cfg.get_ppo_config()
        if name == "SAC":  return self._cfg.get_sac_config()
        return self._cfg.get_td3_config()

    def _add_job(self) -> None:
        try:
            ep  = self._cfg.get_ep_config()
            env = self._cfg.get_env_config()

            if self._cfg._tune_enabled.get() and self._cfg._tune_items:
                self._add_tuning_jobs(ep, env)
                return

            tab_idx   = self._cfg._nb.index("current")
            algo_name = self.ALGO_NAMES[tab_idx] if tab_idx < 3 else "PPO"

            if ep.compare_methods:
                for name in self.ALGO_NAMES:
                    self._manager.create_job(name, self._get_algo_cfg(name), env, ep)
            else:
                self._manager.create_job(
                    algo_name, self._get_algo_cfg(algo_name), env, ep)

            if hasattr(self, "_status_win") and self._status_win.winfo_exists():
                self._status_win.refresh_all()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Add Job Error", str(exc), parent=self)

    def _add_tuning_jobs(self, ep: EpisodeConfig, env: EnvConfig) -> None:
        tab_idx   = self._cfg._nb.index("current")
        algo_name = self.ALGO_NAMES[tab_idx] if tab_idx < 3 else "PPO"
        for param, val_str, _ in self._cfg._tune_items:
            for v in expand_tuning_values(val_str):
                ep2 = deepcopy(ep)
                if param == "alpha":  ep2.alpha = float(v)
                elif param == "gamma": ep2.gamma = float(v)
                ac = self._get_algo_cfg(algo_name)
                if param == "buffer_size" and hasattr(ac, "buffer_size"):
                    ac.buffer_size = int(v)
                elif param == "batch_size" and hasattr(ac, "batch_size"):
                    ac.batch_size = int(v)
                elif param == "hidden_layers":
                    ac.network.hidden_layers = v if isinstance(v, list) else [int(v)]
                self._manager.create_job(algo_name, ac, env, ep2,
                                         label=f"{algo_name} {param}={v}")
        if hasattr(self, "_status_win") and self._status_win.winfo_exists():
            self._status_win.refresh_all()

    def _train(self) -> None:
        ready = [j for j in self._manager.jobs
                 if j.status in (JobStatus.PENDING, JobStatus.COMPLETED,
                                 JobStatus.CANCELLED)]
        if not ready:
            self._add_job()
        for job in self._manager.jobs:
            if job.status in (JobStatus.PENDING, JobStatus.COMPLETED,
                              JobStatus.CANCELLED):
                self._manager.start_job(job.job_id)

    def _on_apply(self) -> None:
        self._cancel_all()
        self._reset_training()

    def _cancel_all(self) -> None:
        for job in self._manager.jobs:
            if job.status in (JobStatus.RUNNING, JobStatus.PAUSED):
                self._manager.cancel(job.job_id)

    def _reset_training(self) -> None:
        for job in list(self._manager.jobs):
            self._manager.remove(job.job_id)
        self._plot._job_data.clear()
        self._plot._redraw()
        self._job_colors.clear()
        self._progress_var.set(0)
        self._progress_lbl.config(text="No active job")

    def _open_status(self) -> None:
        if not hasattr(self, "_status_win") or \
                not self._status_win.winfo_exists():
            self._status_win = TrainingStatusWindow(
                self, self._manager, on_select_job=self._set_viz_job)
        else:
            self._status_win.deiconify()
        self._status_win.refresh_all()

    def _set_viz_job(self, job_id: str) -> None:
        self._active_viz_job = job_id

    def _save_image(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("All", "*.*")])
        if path:
            self._plot.save_image(path)

    def _save_content(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self, defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            with open(path, "w") as fh:
                json.dump(self._plot.get_data(), fh, indent=2)

    def _load_content(self) -> None:
        path = filedialog.askopenfilename(
            parent=self, filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            with open(path) as fh:
                data = json.load(fh)
            self._plot.load_data(data)

    def _save_job(self) -> None:
        base = filedialog.askdirectory(parent=self, title="Save job to…")
        if not base:
            return
        saved = [CheckpointManager.save(job, base)
                 for job in self._manager.jobs]
        if saved:
            messagebox.showinfo("Saved",
                                f"Saved {len(saved)} job(s).", parent=self)

    def _load_job(self) -> None:
        base = filedialog.askdirectory(parent=self, title="Load job from…")
        if not base:
            return
        try:
            job = CheckpointManager.load(base, self._manager)
            idx = len(self._job_colors)
            self._job_colors[job.job_id] = idx
            j = self._manager.get_job(job.job_id)
            if j:
                self._plot.update_job(job.job_id, j.label,
                                      j.returns, j.moving_avg, True, idx)
            if hasattr(self, "_status_win") and self._status_win.winfo_exists():
                self._status_win.refresh_all()
        except Exception as exc:
            messagebox.showerror("Load Error", str(exc), parent=self)
