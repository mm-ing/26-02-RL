"""
UI layer for the Ant RL Workbench.
Tkinter + ttk dark-themed application.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from copy import deepcopy
from dataclasses import asdict
from tkinter import filedialog, messagebox
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
import tkinter.ttk as ttk

import numpy as np

from ant_logic import (
    ALGO_CONFIGS,
    CheckpointManager,
    CMAESConfig,
    EnvConfig,
    EpisodeConfig,
    EventBus,
    EventType,
    JobStatus,
    NetworkConfig,
    SACConfig,
    TQCConfig,
    TrainingManager,
    expand_tuning_values,
)

# ─────────────────────────────────────────────────────────────────────────────
# Colour palette
# ─────────────────────────────────────────────────────────────────────────────

BG       = "#0f111a"
BG2      = "#161824"
BG3      = "#1e2130"
ACCENT   = "#4cc9f0"
ACCENT2  = "#7b5ea7"
FG       = "#e6e6e6"
FG2      = "#b5b5b5"
BORDER   = "#2a2f3a"
SEL      = "#2a3550"
BTN_BG   = "#232638"
BTN_HOV  = "#2e3352"
RED      = "#f07070"
GREEN    = "#70f0a0"

PLOT_COLORS = [
    ("#4cc9f0", "#4cc9f0"),
    ("#f72585", "#f72585"),
    ("#7b5ea7", "#7b5ea7"),
    ("#06d6a0", "#06d6a0"),
    ("#ffd166", "#ffd166"),
    ("#ef233c", "#ef233c"),
]

FONT      = ("Segoe UI", 9)
FONT_BOLD = ("Segoe UI", 9, "bold")
FONT_H    = ("Segoe UI", 10, "bold")
FONT_SM   = ("Segoe UI", 8)


# ─────────────────────────────────────────────────────────────────────────────
# Theme helpers
# ─────────────────────────────────────────────────────────────────────────────

def apply_theme(root: tk.Tk) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    style.configure(".", background=BG, foreground=FG,
                    fieldbackground=BG2, troughcolor=BG2,
                    selectbackground=SEL, selectforeground=FG,
                    font=FONT)
    style.configure("TFrame",    background=BG)
    style.configure("TLabel",    background=BG,  foreground=FG,  font=FONT)
    style.configure("TEntry",    fieldbackground=BG2, foreground=FG,
                    insertcolor=FG, bordercolor=BORDER, lightcolor=BG2,
                    darkcolor=BG2)
    style.configure("TCombobox", fieldbackground=BG2, background=BG2,
                    foreground=FG, selectbackground=SEL, arrowcolor=FG2)
    style.configure("TCheckbutton", background=BG, foreground=FG)
    style.configure("TNotebook",  background=BG2, bordercolor=BORDER)
    style.configure("TNotebook.Tab", background=BG3, foreground=FG2,
                    padding=[8, 3], font=FONT)
    style.map("TNotebook.Tab",
              background=[("selected", BG2)],
              foreground=[("selected", ACCENT)])
    style.configure("TScrollbar", background=BG3, troughcolor=BG2,
                    arrowcolor=FG2, bordercolor=BORDER)

    style.configure("TButton", background=BTN_BG, foreground=FG,
                    bordercolor=BORDER, padding=[8, 4], font=FONT)
    style.map("TButton",
              background=[("active", BTN_HOV), ("disabled", BG3)],
              foreground=[("disabled", FG2)])

    style.configure("Accent.TButton", background=ACCENT2, foreground=FG,
                    bordercolor=ACCENT2, padding=[8, 4], font=FONT_BOLD)
    style.map("Accent.TButton",
              background=[("active", "#9b6ec7")])

    style.configure("Compact.TButton", background=BTN_BG, foreground=FG,
                    bordercolor=BORDER, padding=[4, 2], font=FONT_SM)
    style.map("Compact.TButton",
              background=[("active", BTN_HOV)])

    style.configure("Treeview", background=BG2, foreground=FG,
                    fieldbackground=BG2, bordercolor=BORDER,
                    rowheight=22, font=FONT)
    style.configure("Treeview.Heading", background=BG3, foreground=FG2,
                    bordercolor=BORDER, font=FONT_BOLD)
    style.map("Treeview",
              background=[("selected", SEL)],
              foreground=[("selected", FG)])

    style.configure("TProgressbar", troughcolor=BG2, background=ACCENT,
                    bordercolor=BORDER)
    style.configure("TSeparator", background=BORDER)
    style.configure("TLabelframe", background=BG, bordercolor=BORDER,
                    labelmargins=4)
    style.configure("TLabelframe.Label", background=BG, foreground=ACCENT,
                    font=FONT_BOLD)

    root.configure(background=BG)
    root.option_add("*tearOff", False)


def lf(parent, text: str) -> ttk.LabelFrame:
    return ttk.LabelFrame(parent, text=text, style="TLabelframe")


def row(parent, label: str, var: tk.Variable,
        row_idx: int, col_start: int = 0, width: int = 8) -> ttk.Entry:
    ttk.Label(parent, text=label).grid(row=row_idx, column=col_start,
                                       sticky="w", padx=(4, 2), pady=2)
    e = ttk.Entry(parent, textvariable=var, width=width)
    e.grid(row=row_idx, column=col_start + 1, sticky="ew",
           padx=(0, 6), pady=2)
    return e


# ─────────────────────────────────────────────────────────────────────────────
# Config Panel
# ─────────────────────────────────────────────────────────────────────────────

class ConfigPanel(ttk.Frame):

    ALGO_NAMES = ["TQC", "SAC", "CMA-ES"]

    def __init__(self, parent: tk.Widget, manager: TrainingManager,
                 on_apply: callable) -> None:
        super().__init__(parent)
        self._manager  = manager
        self._on_apply = on_apply
        self._build()

    # ── build ──────────────────────────────────────────────────────────

    def _build(self) -> None:
        self.columnconfigure(0, weight=1)

        canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        sb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        sb.grid(row=0, column=1, sticky="ns")
        self.rowconfigure(0, weight=1)

        inner = ttk.Frame(canvas)
        inner.columnconfigure(0, weight=1)
        canvas.create_window((0, 0), window=inner, anchor="nw", tags="inner")

        def _resize(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig("inner", width=event.width)

        canvas.bind("<Configure>", _resize)
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        def _scroll(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _scroll)

        r = 0

        # ── Environment Configuration ──────────────────────────────────
        frm_env = lf(inner, "Environment Configuration")
        frm_env.grid(row=r, column=0, sticky="ew", padx=4, pady=(6, 2))
        frm_env.columnconfigure(1, weight=1)
        frm_env.columnconfigure(3, weight=1)
        r += 1

        self._fwd_reward_w    = tk.DoubleVar(value=1.0)
        self._ctrl_cost_w     = tk.DoubleVar(value=0.5)
        self._contact_cost_w  = tk.DoubleVar(value=5e-4)
        self._healthy_reward  = tk.DoubleVar(value=1.0)
        self._healthy_z_min   = tk.DoubleVar(value=0.2)
        self._healthy_z_max   = tk.DoubleVar(value=1.0)
        self._contact_f_min   = tk.DoubleVar(value=-1.0)
        self._contact_f_max   = tk.DoubleVar(value=1.0)
        self._reset_noise     = tk.DoubleVar(value=0.1)
        self._excl_cur_pos    = tk.BooleanVar(value=True)
        self._incl_cfrc       = tk.BooleanVar(value=True)
        self._terminate_unh   = tk.BooleanVar(value=True)
        self._render_interval = tk.IntVar(value=10)
        self._visualize       = tk.BooleanVar(value=True)

        row(frm_env, "Fwd Reward W",  self._fwd_reward_w,   0, 0, 7)
        row(frm_env, "Ctrl Cost W",   self._ctrl_cost_w,    0, 2, 7)
        row(frm_env, "Contact Cost W",self._contact_cost_w, 1, 0, 7)
        row(frm_env, "Healthy Reward",self._healthy_reward, 1, 2, 7)
        row(frm_env, "Healthy Z Min", self._healthy_z_min,  2, 0, 7)
        row(frm_env, "Healthy Z Max", self._healthy_z_max,  2, 2, 7)
        row(frm_env, "Cont Force Min",self._contact_f_min,  3, 0, 7)
        row(frm_env, "Cont Force Max",self._contact_f_max,  3, 2, 7)
        row(frm_env, "Noise Scale",   self._reset_noise,    4, 0, 7)
        row(frm_env, "Render ms",     self._render_interval,4, 2, 7)
        ttk.Checkbutton(frm_env, text="Excl. Cur. Pos.",
                        variable=self._excl_cur_pos).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(frm_env, text="Incl. cfrc_ext",
                        variable=self._incl_cfrc).grid(
            row=5, column=2, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(frm_env, text="Terminate Unhealthy",
                        variable=self._terminate_unh).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=4)
        ttk.Checkbutton(frm_env, text="Visualize",
                        variable=self._visualize).grid(
            row=6, column=2, columnspan=2, sticky="w", padx=4)

        # ── Episode Configuration ──────────────────────────────────────
        frm_ep = lf(inner, "Episode Configuration")
        frm_ep.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        frm_ep.columnconfigure(1, weight=1)
        frm_ep.columnconfigure(3, weight=1)
        r += 1

        self._n_episodes  = tk.IntVar(value=3000)
        self._max_steps   = tk.IntVar(value=1000)
        self._alpha       = tk.DoubleVar(value=3e-4)
        self._gamma       = tk.DoubleVar(value=0.99)
        self._compare     = tk.BooleanVar(value=False)

        row(frm_ep, "Episodes",  self._n_episodes, 0, 0, 7)
        row(frm_ep, "Max Steps", self._max_steps,  0, 2, 7)
        row(frm_ep, "Alpha",     self._alpha,      1, 0, 7)
        row(frm_ep, "Gamma",     self._gamma,      1, 2, 7)
        ttk.Checkbutton(frm_ep, text="Compare Methods",
                        variable=self._compare).grid(
            row=2, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        # ── Parameter Tuning ───────────────────────────────────────────
        frm_tune = lf(inner, "Parameter Tuning")
        frm_tune.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        frm_tune.columnconfigure(1, weight=1)
        frm_tune.columnconfigure(3, weight=1)
        r += 1

        self._tune_enabled   = tk.BooleanVar(value=False)
        self._tune_algo_var  = tk.StringVar(value="TQC")
        self._tune_param_var = tk.StringVar(value="alpha")
        self._tune_values    = tk.StringVar(value="1e-4;3e-4;1e-3")

        ttk.Checkbutton(frm_tune, text="Enable Parameter Tuning",
                        variable=self._tune_enabled).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        ttk.Label(frm_tune, text="Method").grid(
            row=1, column=0, sticky="w", padx=(4, 2))
        ttk.Combobox(frm_tune, textvariable=self._tune_algo_var,
                     values=self.ALGO_NAMES, state="readonly",
                     width=8).grid(row=1, column=1, sticky="ew",
                                   padx=(0, 6), pady=2)

        ttk.Label(frm_tune, text="Parameter").grid(
            row=1, column=2, sticky="w", padx=(4, 2))
        self._tune_param_cb = ttk.Combobox(
            frm_tune, textvariable=self._tune_param_var,
            values=["alpha", "gamma", "buffer_size", "batch_size", "hidden_layers"],
            state="readonly", width=12)
        self._tune_param_cb.grid(row=1, column=3, sticky="ew",
                                 padx=(0, 4), pady=2)

        ttk.Label(frm_tune, text="Values (;)").grid(
            row=2, column=0, sticky="w", padx=(4, 2))
        ttk.Entry(frm_tune, textvariable=self._tune_values).grid(
            row=2, column=1, columnspan=3, sticky="ew",
            padx=(0, 4), pady=2)

        self._tune_items: List[Tuple[str, str, str]] = []
        tune_list_frm = ttk.Frame(frm_tune)
        tune_list_frm.grid(row=3, column=0, columnspan=4,
                           sticky="ew", padx=4, pady=2)
        tune_list_frm.columnconfigure(0, weight=1)

        self._tune_tree = ttk.Treeview(
            tune_list_frm,
            columns=("param", "values"),
            show="headings",
            height=3,
            selectmode="browse",
        )
        self._tune_tree.heading("param",  text="Parameter")
        self._tune_tree.heading("values", text="Values")
        self._tune_tree.column("param",   width=110, stretch=False)
        self._tune_tree.column("values",  width=150, stretch=True)
        self._tune_tree.grid(row=0, column=0, sticky="ew")

        btn_frm = ttk.Frame(frm_tune)
        btn_frm.grid(row=4, column=0, columnspan=4,
                     sticky="ew", padx=4, pady=2)
        ttk.Button(btn_frm, text="Add to List",
                   command=self._tune_add).pack(side="left", padx=2)
        ttk.Button(btn_frm, text="Remove Selected",
                   command=self._tune_remove).pack(side="left", padx=2)

        # ── Methods Tab Control ────────────────────────────────────────
        frm_methods = lf(inner, "RL Methods")
        frm_methods.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
        frm_methods.columnconfigure(0, weight=1)
        r += 1

        self._nb = ttk.Notebook(frm_methods)
        self._nb.pack(fill="both", expand=True, padx=2, pady=2)

        self._tqc_tab   = self._build_tqc_tab(self._nb)
        self._sac_tab   = self._build_sac_tab(self._nb)
        self._cmaes_tab = self._build_cmaes_tab(self._nb)
        self._nb.add(self._tqc_tab,   text=" TQC ")
        self._nb.add(self._sac_tab,   text=" SAC ")
        self._nb.add(self._cmaes_tab, text=" CMA-ES ")

        # ── Apply & Reset ──────────────────────────────────────────────
        ttk.Button(inner, text="Apply & Reset",
                   style="Accent.TButton",
                   command=self._apply).grid(
            row=r, column=0, sticky="ew", padx=4, pady=6)

    # ── TQC tab ───────────────────────────────────────────────────────

    def _build_tqc_tab(self, nb: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(nb)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)

        self._tqc_buf_size  = tk.IntVar(value=300_000)
        self._tqc_batch     = tk.IntVar(value=256)
        self._tqc_starts    = tk.IntVar(value=10_000)
        self._tqc_train_f   = tk.IntVar(value=1)
        self._tqc_grad_s    = tk.IntVar(value=1)
        self._tqc_tau       = tk.DoubleVar(value=0.005)
        self._tqc_ent       = tk.StringVar(value="auto")
        self._tqc_tgt_upd   = tk.IntVar(value=1)
        self._tqc_top_quant = tk.IntVar(value=2)
        self._tqc_n_quant   = tk.IntVar(value=25)
        self._tqc_n_critics = tk.IntVar(value=2)
        self._tqc_hidden    = tk.StringVar(value="256,256")
        self._tqc_activation= tk.StringVar(value="relu")

        row(tab, "Buffer Size",     self._tqc_buf_size,  0, 0, 10)
        row(tab, "Batch Size",      self._tqc_batch,     0, 2, 7)
        row(tab, "Learn Starts",    self._tqc_starts,    1, 0, 10)
        row(tab, "Train Freq",      self._tqc_train_f,   1, 2, 7)
        row(tab, "Grad Steps",      self._tqc_grad_s,    2, 0, 10)
        row(tab, "Tau",             self._tqc_tau,       2, 2, 7)
        row(tab, "Ent Coef",        self._tqc_ent,       3, 0, 10)
        row(tab, "Target Upd Int",  self._tqc_tgt_upd,   3, 2, 7)
        row(tab, "Top Quant Drop",  self._tqc_top_quant, 4, 0, 10)
        row(tab, "N Quantiles",     self._tqc_n_quant,   4, 2, 7)
        row(tab, "N Critics",       self._tqc_n_critics, 5, 0, 10)
        row(tab, "Hidden Layers",   self._tqc_hidden,    5, 2, 10)
        ttk.Label(tab, text="Activation").grid(row=6, column=0, sticky="w",
                                               padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._tqc_activation,
                     values=["relu", "tanh", "elu"],
                     state="readonly", width=7).grid(row=6, column=1,
                                                     sticky="ew",
                                                     padx=(0, 4), pady=2)
        return tab

    # ── SAC tab ───────────────────────────────────────────────────────

    def _build_sac_tab(self, nb: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(nb)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)

        self._sac_buf_size  = tk.IntVar(value=300_000)
        self._sac_batch     = tk.IntVar(value=256)
        self._sac_starts    = tk.IntVar(value=10_000)
        self._sac_train_f   = tk.IntVar(value=1)
        self._sac_grad_s    = tk.IntVar(value=1)
        self._sac_tau       = tk.DoubleVar(value=0.005)
        self._sac_ent       = tk.StringVar(value="auto")
        self._sac_tgt_upd   = tk.IntVar(value=1)
        self._sac_hidden    = tk.StringVar(value="256,256")
        self._sac_activation= tk.StringVar(value="relu")

        row(tab, "Buffer Size",    self._sac_buf_size, 0, 0, 10)
        row(tab, "Batch Size",     self._sac_batch,    0, 2, 7)
        row(tab, "Learn Starts",   self._sac_starts,   1, 0, 10)
        row(tab, "Train Freq",     self._sac_train_f,  1, 2, 7)
        row(tab, "Grad Steps",     self._sac_grad_s,   2, 0, 10)
        row(tab, "Tau",            self._sac_tau,      2, 2, 7)
        row(tab, "Ent Coef",       self._sac_ent,      3, 0, 10)
        row(tab, "Target Upd Int", self._sac_tgt_upd,  3, 2, 7)
        row(tab, "Hidden Layers",  self._sac_hidden,   4, 0, 12)
        ttk.Label(tab, text="Activation").grid(row=4, column=2, sticky="w",
                                               padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._sac_activation,
                     values=["relu", "tanh", "elu"],
                     state="readonly", width=7).grid(row=4, column=3,
                                                     sticky="ew",
                                                     padx=(0, 4), pady=2)
        return tab

    # ── CMA-ES tab ───────────────────────────────────────────────────

    def _build_cmaes_tab(self, nb: ttk.Notebook) -> ttk.Frame:
        tab = ttk.Frame(nb)
        tab.columnconfigure(1, weight=1)
        tab.columnconfigure(3, weight=1)

        self._cmaes_popsize    = tk.IntVar(value=16)
        self._cmaes_stdev      = tk.DoubleVar(value=0.5)
        self._cmaes_eval_eps   = tk.IntVar(value=3)
        self._cmaes_num_actors = tk.IntVar(value=2)
        self._cmaes_max_mem_gb = tk.DoubleVar(value=2.0)
        self._cmaes_hidden     = tk.StringVar(value="64,64")
        self._cmaes_activation = tk.StringVar(value="tanh")

        row(tab, "Pop Size",       self._cmaes_popsize,    0, 0, 7)
        row(tab, "Stdev Init",     self._cmaes_stdev,      0, 2, 7)
        row(tab, "Eval Episodes",  self._cmaes_eval_eps,   1, 0, 7)
        row(tab, "Num Actors",     self._cmaes_num_actors, 1, 2, 7)
        row(tab, "Max Mem GB",     self._cmaes_max_mem_gb, 2, 0, 7)
        row(tab, "Hidden Layers",  self._cmaes_hidden,     2, 2, 10)
        ttk.Label(tab, text="Activation").grid(row=3, column=0, sticky="w",
                                               padx=(4, 2))
        ttk.Combobox(tab, textvariable=self._cmaes_activation,
                     values=["tanh", "relu", "elu"],
                     state="readonly", width=7).grid(row=3, column=1,
                                                     sticky="ew",
                                                     padx=(0, 4), pady=2)
        ttk.Label(tab, text="Note: Episodes = Generations  |  Num Actors > 1 requires Ray  |  Pop Size auto-clamped to Max Mem GB",
                  foreground=FG2, font=FONT_SM).grid(
            row=4, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        return tab

    # ── Tune helpers ──────────────────────────────────────────────────

    def _tune_add(self) -> None:
        param  = self._tune_param_var.get()
        values = self._tune_values.get().strip()
        if not param or not values:
            return
        iid = self._tune_tree.insert("", "end", values=(param, values))
        self._tune_items.append((param, values, iid))

    def _tune_remove(self) -> None:
        sel = self._tune_tree.selection()
        if sel:
            self._tune_tree.delete(sel[0])
            self._tune_items = [
                item for item in self._tune_items if item[2] != sel[0]
            ]

    # ── Config readers ─────────────────────────────────────────────────

    def get_env_config(self) -> EnvConfig:
        return EnvConfig(
            forward_reward_weight                    = float(self._fwd_reward_w.get()),
            ctrl_cost_weight                         = float(self._ctrl_cost_w.get()),
            contact_cost_weight                      = float(self._contact_cost_w.get()),
            healthy_reward                           = float(self._healthy_reward.get()),
            terminate_when_unhealthy                 = bool(self._terminate_unh.get()),
            healthy_z_min                            = float(self._healthy_z_min.get()),
            healthy_z_max                            = float(self._healthy_z_max.get()),
            contact_force_min                        = float(self._contact_f_min.get()),
            contact_force_max                        = float(self._contact_f_max.get()),
            reset_noise_scale                        = float(self._reset_noise.get()),
            exclude_current_positions_from_observation = bool(self._excl_cur_pos.get()),
            include_cfrc_ext_in_observation          = bool(self._incl_cfrc.get()),
            render_interval_ms                       = int(self._render_interval.get()),
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

    def _parse_hidden(self, var: tk.StringVar) -> List[int]:
        raw = var.get().strip()
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return [256, 256]

    def get_tqc_config(self) -> TQCConfig:
        return TQCConfig(
            buffer_size            = int(self._tqc_buf_size.get()),
            batch_size             = int(self._tqc_batch.get()),
            learning_starts        = int(self._tqc_starts.get()),
            train_freq             = int(self._tqc_train_f.get()),
            gradient_steps         = int(self._tqc_grad_s.get()),
            tau                    = float(self._tqc_tau.get()),
            ent_coef               = self._tqc_ent.get(),
            target_update_interval = int(self._tqc_tgt_upd.get()),
            top_quantiles_to_drop  = int(self._tqc_top_quant.get()),
            n_quantiles            = int(self._tqc_n_quant.get()),
            n_critics              = int(self._tqc_n_critics.get()),
            network = NetworkConfig(
                hidden_layers = self._parse_hidden(self._tqc_hidden),
                activation    = self._tqc_activation.get(),
            ),
        )

    def get_sac_config(self) -> SACConfig:
        return SACConfig(
            buffer_size            = int(self._sac_buf_size.get()),
            batch_size             = int(self._sac_batch.get()),
            learning_starts        = int(self._sac_starts.get()),
            train_freq             = int(self._sac_train_f.get()),
            gradient_steps         = int(self._sac_grad_s.get()),
            tau                    = float(self._sac_tau.get()),
            ent_coef               = self._sac_ent.get(),
            target_update_interval = int(self._sac_tgt_upd.get()),
            network = NetworkConfig(
                hidden_layers = self._parse_hidden(self._sac_hidden),
                activation    = self._sac_activation.get(),
            ),
        )

    def get_cmaes_config(self) -> CMAESConfig:
        return CMAESConfig(
            popsize         = int(self._cmaes_popsize.get()),
            stdev_init      = float(self._cmaes_stdev.get()),
            n_eval_episodes = int(self._cmaes_eval_eps.get()),
            num_actors      = max(1, int(self._cmaes_num_actors.get())),
            max_memory_gb   = max(0.1, float(self._cmaes_max_mem_gb.get())),
            network = NetworkConfig(
                hidden_layers = self._parse_hidden(self._cmaes_hidden),
                activation    = self._cmaes_activation.get(),
            ),
        )

    def get_visualize(self) -> bool:
        return bool(self._visualize.get())

    def _apply(self) -> None:
        if callable(self._on_apply):
            self._on_apply()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Panel
# ─────────────────────────────────────────────────────────────────────────────

class VisualizationPanel(ttk.Frame):

    def __init__(self, parent: tk.Widget, interval_var: tk.IntVar) -> None:
        super().__init__(parent)
        self._interval_var = interval_var
        self._img_ref      = None
        self._debounce_id  = None
        self._canvas_w     = 1
        self._canvas_h     = 1

        self._canvas = tk.Canvas(self, bg=BG, highlightthickness=0, bd=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_resize)

        self._label = ttk.Label(self, text="No visualization",
                                foreground=FG2, background=BG,
                                font=FONT_H)
        self._label.place(relx=0.5, rely=0.5, anchor="center")

    def _on_resize(self, event: tk.Event) -> None:
        if self._debounce_id:
            self.after_cancel(self._debounce_id)
        self._canvas_w = max(event.width,  1)
        self._canvas_h = max(event.height, 1)
        self._debounce_id = self.after(100, self._redraw)

    def _redraw(self) -> None:
        if self._img_ref:
            self._canvas.delete("all")
            self._canvas.create_image(
                self._canvas_w // 2,
                self._canvas_h // 2,
                image=self._img_ref,
                anchor="center",
            )

    def update_frame(self, frame: np.ndarray) -> None:
        from PIL import Image, ImageTk
        self._label.place_forget()
        h_orig, w_orig = frame.shape[:2]
        cw, ch = self._canvas_w, self._canvas_h
        scale = min(cw / w_orig, ch / h_orig)
        nw = max(int(w_orig * scale), 1)
        nh = max(int(h_orig * scale), 1)
        img    = Image.fromarray(frame).resize((nw, nh), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        self._img_ref = tk_img
        self._canvas.delete("all")
        self._canvas.create_image(cw // 2, ch // 2,
                                  image=tk_img, anchor="center")

    def clear(self) -> None:
        self._img_ref = None
        self._canvas.delete("all")
        self._label.place(relx=0.5, rely=0.5, anchor="center")


# ─────────────────────────────────────────────────────────────────────────────
# Plot Panel
# ─────────────────────────────────────────────────────────────────────────────

class PlotPanel(ttk.Frame):

    def __init__(self, parent: tk.Widget) -> None:
        super().__init__(parent)
        self._job_data: Dict[str, dict] = {}
        self._debounce_id         = None
        self._resize_throttle_id  = None
        self._resizing            = False
        self._build()

    def _build(self) -> None:
        import matplotlib
        matplotlib.use("TkAgg")
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        import matplotlib.pyplot as plt
        self._fig, self._ax = plt.subplots(figsize=(8, 3))
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(BG)
        self._ax.tick_params(colors=FG2)
        self._ax.xaxis.label.set_color(FG2)
        self._ax.yaxis.label.set_color(FG2)
        self._ax.set_xlabel("Episode / Generation")
        self._ax.set_ylabel("Return")
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._ax.grid(color="#2a2f3a", linestyle="--", alpha=0.5)
        self._fig.tight_layout(pad=0.5)

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=self)
        self._mpl_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.bind("<Configure>", self._on_resize)

    def _on_resize(self, event: tk.Event) -> None:
        if self._resize_throttle_id:
            self.after_cancel(self._resize_throttle_id)
        self._resizing = True
        self._resize_throttle_id = self.after(80, self._finish_resize)

    def _finish_resize(self) -> None:
        self._resizing = False
        self._redraw_now()

    def update_job(self, job_id: str, label: str,
                   returns: List[float], moving_avg: List[float],
                   visible: bool, color_idx: int) -> None:
        self._job_data[job_id] = {
            "label":      label,
            "returns":    returns,
            "moving_avg": moving_avg,
            "visible":    visible,
            "color_idx":  color_idx,
        }
        if not self._resizing:
            if self._debounce_id:
                self.after_cancel(self._debounce_id)
            self._debounce_id = self.after(50, self._redraw_now)

    def remove_job(self, job_id: str) -> None:
        self._job_data.pop(job_id, None)
        self._redraw_now()

    def toggle_visibility(self, job_id: str, visible: bool) -> None:
        if job_id in self._job_data:
            self._job_data[job_id]["visible"] = visible
            self._redraw_now()

    def _redraw_now(self) -> None:
        self._ax.clear()
        self._ax.set_facecolor(BG)
        self._ax.tick_params(colors=FG2)
        self._ax.xaxis.label.set_color(FG2)
        self._ax.yaxis.label.set_color(FG2)
        self._ax.set_xlabel("Episode / Generation")
        self._ax.set_ylabel("Return")
        for spine in self._ax.spines.values():
            spine.set_edgecolor(BORDER)
        self._ax.grid(color="#2a2f3a", linestyle="--", alpha=0.5)

        for jdata in self._job_data.values():
            if not jdata["visible"]:
                continue
            ci    = jdata["color_idx"] % len(PLOT_COLORS)
            raw_c = PLOT_COLORS[ci][0]
            avg_c = PLOT_COLORS[ci][1]
            ret   = jdata["returns"]
            mavg  = jdata["moving_avg"]
            ep    = list(range(1, len(ret) + 1))
            if ret:
                self._ax.plot(ep, ret,
                              color=raw_c, alpha=0.35, linewidth=1.0,
                              label=f"{jdata['label']} Raw")
            if mavg:
                ep_m = list(range(1, len(mavg) + 1))
                self._ax.plot(ep_m, mavg,
                              color=avg_c, alpha=1.0, linewidth=2.5,
                              label=f"{jdata['label']} Avg")

        handles, labels = self._ax.get_legend_handles_labels()
        if handles:
            self._ax.legend(
                handles, labels,
                loc="upper left",
                bbox_to_anchor=(1.0, 1.0),
                facecolor=BG,
                edgecolor=BORDER,
                labelcolor="#e6e6e6",
                fontsize=8,
            )

        try:
            self._fig.tight_layout(rect=[0, 0, 0.80, 1.0], pad=0.5)
        except Exception:
            pass
        try:
            self._mpl_canvas.draw_idle()
        except Exception:
            pass

    def save_image(self, path: str) -> None:
        self._fig.savefig(path, facecolor=BG, dpi=150)

    def get_data(self) -> Dict[str, list]:
        return {
            jid: {"label": d["label"],
                  "returns": d["returns"],
                  "moving_avg": d["moving_avg"]}
            for jid, d in self._job_data.items()
        }

    def load_data(self, data: dict) -> None:
        idx = len(self._job_data)
        for jid, d in data.items():
            self._job_data[jid] = {
                "label":      d.get("label", jid),
                "returns":    d.get("returns", []),
                "moving_avg": d.get("moving_avg", []),
                "visible":    True,
                "color_idx":  idx,
            }
            idx += 1
        self._redraw_now()


# ─────────────────────────────────────────────────────────────────────────────
# Training Status Window
# ─────────────────────────────────────────────────────────────────────────────

class TrainingStatusWindow(tk.Toplevel):

    COLS = ("label", "episode", "return", "moving_avg",
            "loss", "duration", "steps", "status", "visible")

    def __init__(self, parent: tk.Widget, manager: TrainingManager,
                 on_select_job: callable) -> None:
        super().__init__(parent)
        self.title("Training Status – Ant-v5")
        self.configure(bg=BG)
        self.geometry("980x420")
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

        self._tv = ttk.Treeview(
            frm,
            columns=self.COLS,
            show="headings",
            selectmode="browse",
        )
        headings = {
            "label":      ("Algorithm",   140, True),
            "episode":    ("Episode",      80, False),
            "return":     ("Return",       80, False),
            "moving_avg": ("MovingAvg",    80, False),
            "loss":       ("Loss",         80, False),
            "duration":   ("Duration(s)",  80, False),
            "steps":      ("Steps",        70, False),
            "status":     ("Status",       80, False),
            "visible":    ("Visible",      60, False),
        }
        for col, (txt, w, stretch) in headings.items():
            self._tv.heading(col, text=txt, command=lambda c=col: self._sort(c))
            self._tv.column(col, width=w, stretch=stretch)

        sb_y = ttk.Scrollbar(frm, orient="vertical",   command=self._tv.yview)
        sb_x = ttk.Scrollbar(frm, orient="horizontal", command=self._tv.xview)
        self._tv.configure(yscrollcommand=sb_y.set, xscrollcommand=sb_x.set)
        self._tv.grid(row=0, column=0, sticky="nsew")
        sb_y.grid(row=0, column=1, sticky="ns")
        sb_x.grid(row=1, column=0, sticky="ew")

        self._tv.bind("<<TreeviewSelect>>", self._on_tv_select)
        self._tv.bind("<Double-1>",         self._on_double_click)
        self._tv.bind("<Button-3>",         self._on_right_click)
        self._tv.bind("<Return>",           lambda e: self._toggle())
        self._tv.bind("<space>",            lambda e: self._pause_resume())

        self._ctx = tk.Menu(self, tearoff=False, bg=BG2, fg=FG,
                            activebackground=SEL, activeforeground=FG)
        self._ctx.add_command(label="Toggle Visibility", command=self._toggle)
        self._ctx.add_command(label="Train",             command=self._train)
        self._ctx.add_command(label="Run",               command=self._run)
        self._ctx.add_command(label="Pause/Resume",      command=self._pause_resume)
        self._ctx.add_command(label="Stop",              command=self._stop)
        self._ctx.add_command(label="Remove",            command=self._remove)

        btn_frm = ttk.Frame(self)
        btn_frm.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        for txt, cmd in [
            ("Toggle Visible", self._toggle),
            ("Train",          self._train),
            ("Run",            self._run),
            ("Pause/Resume",   self._pause_resume),
            ("Stop",           self._stop),
            ("Remove",         self._remove),
        ]:
            ttk.Button(btn_frm, text=txt, command=cmd, width=12).pack(
                side="left", padx=2)

    # ── Treeview events ───────────────────────────────────────────────

    def _on_tv_select(self, _e: tk.Event) -> None:
        jid = self._selected_job_id()
        if jid and callable(self._on_select):
            self._on_select(jid)

    def _on_double_click(self, _e: tk.Event) -> None:
        self._toggle()

    def _on_right_click(self, event: tk.Event) -> None:
        row = self._tv.identify_row(event.y)
        if row:
            self._tv.selection_set(row)
        self._ctx.post(event.x_root, event.y_root)

    # ── Actions ───────────────────────────────────────────────────────

    def _selected_job_id(self) -> Optional[str]:
        sel = self._tv.selection()
        if not sel:
            return None
        tags = self._tv.item(sel[0], "tags")
        return tags[0] if tags else None

    def _toggle(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._manager.toggle_visibility(jid)
            self._update_row(jid)

    def _train(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._manager.start_job(jid)

    def _run(self) -> None:
        jid = self._selected_job_id()
        if jid:
            job = self._manager.get_job(jid)
            if job and job.model is not None:
                self._manager.run_inference(jid)

    def _pause_resume(self) -> None:
        jid = self._selected_job_id()
        if not jid:
            return
        job = self._manager.get_job(jid)
        if job:
            if job.status == JobStatus.RUNNING:
                self._manager.pause(jid)
            elif job.status == JobStatus.PAUSED:
                self._manager.resume(jid)

    def _stop(self) -> None:
        jid = self._selected_job_id()
        if jid:
            self._manager.cancel(jid)

    def _remove(self) -> None:
        jid = self._selected_job_id()
        if jid:
            if messagebox.askyesno("Remove Job", "Remove this training job?",
                                   parent=self):
                self._manager.remove(jid)
                self.refresh_all()

    def _sort(self, col: str) -> None:
        if self._sort_col == col:
            self._sort_rev = not self._sort_rev
        else:
            self._sort_col = col
            self._sort_rev = False
        data = [(self._tv.set(iid, col), iid) for iid in self._tv.get_children()]
        try:
            data.sort(key=lambda t: float(t[0]), reverse=self._sort_rev)
        except ValueError:
            data.sort(key=lambda t: t[0], reverse=self._sort_rev)
        for idx, (_, iid) in enumerate(data):
            self._tv.move(iid, "", idx)

    # ── Public update API ─────────────────────────────────────────────

    def upsert_job(self, job_id: str, **kw) -> None:
        ep_str  = f"{kw.get('episode', 0)}/{kw.get('n_ep', '?')}"
        ret_s   = f"{kw.get('ret', 0.0):.1f}"
        mavg_s  = f"{kw.get('mavg', 0.0):.1f}"
        loss_s  = f"{kw.get('loss', '')}" if kw.get("loss") is not None else ""
        dur_s   = f"{kw.get('dur', 0.0):.2f}"
        steps_s = str(kw.get("steps", 0))
        stat_s  = kw.get("status", "")
        vis_s   = "✓" if kw.get("visible", True) else "✗"
        label_s = kw.get("label", job_id[:8])

        vals = (label_s, ep_str, ret_s, mavg_s, loss_s,
                dur_s, steps_s, stat_s, vis_s)

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
                job.job_id,
                label=job.label,
                episode=job.current_episode,
                n_ep=job.ep_config.n_episodes,
                ret=job.returns[-1] if job.returns else 0.0,
                mavg=mavg,
                status=job.status.value,
                visible=job.visible,
            )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

class WorkbenchApp(tk.Tk):

    ALGO_NAMES = ConfigPanel.ALGO_NAMES

    def __init__(self) -> None:
        super().__init__()
        self.title("RL Workbench – Ant-v5")
        self.geometry("1400x880")
        self.minsize(900, 620)
        apply_theme(self)

        self._bus     = EventBus()
        self._manager = TrainingManager(self._bus)
        self._ckpt    = CheckpointManager()

        self._bus.subscribe(self._on_event)

        self._job_colors:  Dict[str, int]  = {}
        self._job_returns: Dict[str, list] = {}
        self._job_mavg:    Dict[str, list] = {}

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

        config_holder = ttk.Frame(top_pw)
        config_holder.columnconfigure(0, weight=1)
        config_holder.rowconfigure(0, weight=1)
        self._cfg = ConfigPanel(config_holder, self._manager,
                                on_apply=self._on_apply)
        self._cfg.grid(row=0, column=0, sticky="nsew")

        viz_holder = ttk.Frame(top_pw)
        viz_holder.columnconfigure(0, weight=1)
        viz_holder.rowconfigure(0, weight=1)
        self._viz = VisualizationPanel(viz_holder, self._cfg._render_interval)
        self._viz.grid(row=0, column=0, sticky="nsew")

        top_pw.add(config_holder, weight=1)
        top_pw.add(viz_holder,    weight=2)

        bottom = ttk.Frame(outer)
        bottom.columnconfigure(0, weight=1)

        self._progress_var = tk.DoubleVar(value=0.0)
        self._progress_lbl = ttk.Label(bottom, text="No active job",
                                       foreground=FG2, font=FONT_SM)
        self._progress_lbl.grid(row=0, column=0, sticky="ew",
                                padx=4, pady=(4, 0))
        ttk.Progressbar(bottom, variable=self._progress_var,
                        maximum=100.0).grid(
            row=1, column=0, sticky="ew", padx=4, pady=2)

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
        btns = [
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
        ]
        for txt, cmd in btns:
            ttk.Button(self._btn_frame, text=txt, command=cmd,
                       style=s).pack(side="left", padx=2, pady=2)

    def _on_win_resize(self, event: tk.Event) -> None:
        if event.widget is not self:
            return
        if self._resize_debounce:
            self.after_cancel(self._resize_debounce)
        self._resize_debounce = self.after(
            100, lambda w=event.width: self._check_compact(w))

    def _check_compact(self, width: int) -> None:
        new_compact = width < 1100
        if new_compact != self._compact_buttons:
            self._compact_buttons = new_compact
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
                self._plot.update_job(ev.job_id, job.label,
                                      [], [], True, idx)

        elif t == EventType.JOB_STARTED:
            job = self._manager.get_job(ev.job_id)
            label = job.label if job else ev.job_id[:8]
            self._progress_var.set(0)
            self._progress_lbl.config(text=f"{label}  Building model…")

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
                if len(job.moving_avg) < len(rets):
                    for i in range(len(job.moving_avg), len(rets)):
                        window = rets[max(0, i - win + 1):i + 1]
                        job.moving_avg.append(float(np.mean(window)))
                elif len(job.moving_avg) > len(rets):
                    job.moving_avg = job.moving_avg[:len(rets)]

            pct = (ep / n_ep * 100) if n_ep else 0
            self._progress_var.set(pct)
            label = job.label if job else ev.job_id[:8]
            self._progress_lbl.config(
                text=f"{label}  Ep {ep}/{n_ep}  "
                     f"Return {ret:.1f}  Avg {mavg:.1f}")

            if job and job.visible:
                idx = self._job_colors.get(ev.job_id, 0)
                self._plot.update_job(
                    ev.job_id, job.label,
                    job.returns, job.moving_avg,
                    True, idx)

            if hasattr(self, "_status_win") and self._status_win.winfo_exists():
                self._status_win.upsert_job(
                    ev.job_id,
                    label=job.label if job else "",
                    episode=ep,
                    n_ep=n_ep,
                    ret=ret,
                    mavg=mavg,
                    loss=loss,
                    dur=dur,
                    steps=steps,
                    status=(job.status.value if job else ""),
                    visible=(job.visible if job else True),
                )

        elif t in (EventType.JOB_COMPLETED, EventType.TRAINING_DONE):
            job = self._manager.get_job(ev.job_id)
            if job:
                idx = self._job_colors.get(ev.job_id, 0)
                self._plot.update_job(
                    ev.job_id, job.label,
                    job.returns, job.moving_avg,
                    job.visible, idx)
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
            detail = f"{err}\n\n{tb}".strip() if tb else err
            messagebox.showerror("Training Error", detail, parent=self)

    # ── Button actions ────────────────────────────────────────────────

    def _add_job(self) -> None:
        try:
            ep  = self._cfg.get_ep_config()
            env = self._cfg.get_env_config()

            if self._cfg._tune_enabled.get() and self._cfg._tune_items:
                self._add_tuning_jobs(ep, env)
                return

            tab_idx   = self._cfg._nb.index("current")
            algo_name = self.ALGO_NAMES[tab_idx] if tab_idx < len(self.ALGO_NAMES) else "TQC"
            algo_cfg  = self._get_algo_cfg(algo_name)

            if ep.compare_methods:
                for name in self.ALGO_NAMES:
                    ac = self._get_algo_cfg(name)
                    self._manager.create_job(name, ac, env, ep)
            else:
                self._manager.create_job(algo_name, algo_cfg, env, ep)

            if hasattr(self, "_status_win") and self._status_win.winfo_exists():
                self._status_win.refresh_all()
        except Exception as exc:
            import traceback
            traceback.print_exc()
            messagebox.showerror("Add Job Error", str(exc), parent=self)

    def _get_algo_cfg(self, name: str) -> Any:
        if name == "TQC":
            return self._cfg.get_tqc_config()
        elif name == "SAC":
            return self._cfg.get_sac_config()
        else:
            return self._cfg.get_cmaes_config()

    def _add_tuning_jobs(self, ep: EpisodeConfig, env: EnvConfig) -> None:
        tab_idx   = self._cfg._nb.index("current")
        algo_name = self.ALGO_NAMES[tab_idx] if tab_idx < len(self.ALGO_NAMES) else "TQC"

        for param, val_str, _ in self._cfg._tune_items:
            values = expand_tuning_values(val_str)
            for v in values:
                ep2 = deepcopy(ep)
                if param == "alpha":
                    ep2.alpha = float(v)
                elif param == "gamma":
                    ep2.gamma = float(v)
                ac = self._get_algo_cfg(algo_name)
                if param == "buffer_size" and hasattr(ac, "buffer_size"):
                    ac.buffer_size = int(v)
                elif param == "batch_size" and hasattr(ac, "batch_size"):
                    ac.batch_size = int(v)
                elif param == "hidden_layers":
                    ac.network.hidden_layers = v if isinstance(v, list) else [int(v)]
                label = f"{algo_name} {param}={v}"
                self._manager.create_job(algo_name, ac, env, ep2, label=label)

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
        self._plot._redraw_now()
        self._job_colors.clear()
        self._progress_var.set(0)
        self._progress_lbl.config(text="No active job")

    def _open_status(self) -> None:
        if not hasattr(self, "_status_win") or \
                not self._status_win.winfo_exists():
            self._status_win = TrainingStatusWindow(
                self, self._manager,
                on_select_job=self._set_viz_job,
            )
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
            parent=self,
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if path:
            with open(path) as fh:
                data = json.load(fh)
            self._plot.load_data(data)

    def _save_job(self) -> None:
        base = filedialog.askdirectory(parent=self, title="Save job to…")
        if not base:
            return
        saved = []
        for job in self._manager.jobs:
            p = CheckpointManager.save(job, base)
            saved.append(p)
        if saved:
            messagebox.showinfo("Saved", f"Saved {len(saved)} job(s).", parent=self)

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
