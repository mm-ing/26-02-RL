from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Dict, Optional

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from mountain_car_logic import (
    AlgorithmConfig,
    EpisodeConfig,
    EventBus,
    JobConfig,
    NetworkConfig,
    TrainingManager,
    TuneConfig,
)


class TrainingStatusWindow(tk.Toplevel):
    COLUMNS = ("algorithm", "episode", "return", "moving", "epsilon", "loss", "duration", "steps", "visible")

    def __init__(self, master: tk.Misc, manager: TrainingManager):
        super().__init__(master)
        self.title("Training Status")
        self.geometry("1080x440")
        self.manager = manager
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        style = ttk.Style(self)
        style.configure("Status.Treeview", background="#0f111a", fieldbackground="#0f111a", foreground="#e6e6e6")
        style.configure("Status.Treeview.Heading", background="#2a2f3a", foreground="#e6e6e6")

        self.tree = ttk.Treeview(self, columns=self.COLUMNS, show="headings", style="Status.Treeview")
        self.tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        widths = {
            "algorithm": 230,
            "episode": 90,
            "return": 80,
            "moving": 95,
            "epsilon": 70,
            "loss": 80,
            "duration": 80,
            "steps": 70,
            "visible": 70,
        }
        labels = {
            "algorithm": "Algorithm",
            "episode": "Episode",
            "return": "Return",
            "moving": "MovingAvg",
            "epsilon": "Epsilon",
            "loss": "Loss",
            "duration": "Duration",
            "steps": "Steps",
            "visible": "Visible",
        }

        for col in self.COLUMNS:
            self.tree.heading(col, text=labels[col], command=lambda c=col: self.sort_by(c, False))
            self.tree.column(col, width=widths[col], stretch=(col == "algorithm"))

        self.tree.bind("<Double-1>", self._on_double_click)
        self.tree.bind("<Return>", self._on_double_click)
        self.tree.bind("<space>", self._on_space)

        button_bar = ttk.Frame(self)
        button_bar.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(5):
            button_bar.columnconfigure(i, weight=1)

        ttk.Button(button_bar, text="Toggle Visibility", command=self.toggle_visibility).grid(row=0, column=0, sticky="ew", padx=2)
        ttk.Button(button_bar, text="Pause", command=self.pause_selected).grid(row=0, column=1, sticky="ew", padx=2)
        ttk.Button(button_bar, text="Resume", command=self.resume_selected).grid(row=0, column=2, sticky="ew", padx=2)
        ttk.Button(button_bar, text="Cancel", command=self.cancel_selected).grid(row=0, column=3, sticky="ew", padx=2)
        ttk.Button(button_bar, text="Restart", command=self.restart_selected).grid(row=0, column=4, sticky="ew", padx=2)

    def selected_job_id(self) -> Optional[str]:
        selection = self.tree.selection()
        if not selection:
            return None
        return selection[0]

    def upsert_row(self, payload: Dict):
        job_id = payload["job_id"]
        values = (
            payload.get("algorithm", "-"),
            f"{payload.get('episode', 0)}/{payload.get('episodes_total', 0)}",
            f"{payload.get('return', 0.0):.2f}",
            f"{payload.get('moving_avg', 0.0):.2f}",
            f"{payload.get('epsilon', 0.0):.3f}",
            f"{payload.get('loss', float('nan')):.3f}",
            f"{payload.get('duration', 0.0):.2f}s",
            str(payload.get("steps", 0)),
            "Yes" if payload.get("visible", True) else "No",
        )
        if self.tree.exists(job_id):
            self.tree.item(job_id, values=values)
        else:
            self.tree.insert("", "end", iid=job_id, values=values)

    def remove_row(self, job_id: str):
        if self.tree.exists(job_id):
            self.tree.delete(job_id)

    def toggle_visibility(self):
        job_id = self.selected_job_id()
        if not job_id:
            return
        self.manager.toggle_visibility(job_id)

    def pause_selected(self):
        job_id = self.selected_job_id()
        if job_id:
            self.manager.pause(job_id)

    def resume_selected(self):
        job_id = self.selected_job_id()
        if job_id:
            self.manager.resume(job_id)

    def cancel_selected(self):
        job_id = self.selected_job_id()
        if job_id:
            self.manager.cancel(job_id)

    def restart_selected(self):
        job_id = self.selected_job_id()
        if job_id:
            self.manager.start_job(job_id)

    def _on_double_click(self, _event=None):
        self.toggle_visibility()

    def _on_space(self, _event=None):
        job_id = self.selected_job_id()
        if not job_id:
            return
        job = self.manager.jobs.get(job_id)
        if not job:
            return
        if job.status == "paused":
            self.manager.resume(job_id)
        else:
            self.manager.pause(job_id)

    def sort_by(self, col: str, descending: bool):
        items = [(self.tree.set(item, col), item) for item in self.tree.get_children("")]
        try:
            items.sort(key=lambda x: float(str(x[0]).replace("s", "").split("/")[0]), reverse=descending)
        except ValueError:
            items.sort(reverse=descending)
        for index, (_, item) in enumerate(items):
            self.tree.move(item, "", index)
        self.tree.heading(col, command=lambda: self.sort_by(col, not descending))


class WorkbenchUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("RL Workbench - MountainCar")
        self.root.geometry("1400x900")
        self.event_bus = EventBus()
        self.manager = TrainingManager(self.event_bus)
        self.status_window: Optional[TrainingStatusWindow] = None
        self._selected_status_job_id: Optional[str] = None
        self._resize_after_id: Optional[str] = None
        self._current_compact = False
        self._last_plot_draw = 0.0
        self._plot_redraw_ms = 80
        self.method_vars: Dict[str, Dict[str, tk.StringVar]] = {}

        self._build_styles()
        self._build_layout()
        self._build_plot()
        self._schedule_polling()

    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f111a")
        style.configure("TLabelframe", background="#0f111a", foreground="#e6e6e6")
        style.configure("TLabelframe.Label", background="#0f111a", foreground="#e6e6e6")
        style.configure("TLabel", background="#0f111a", foreground="#e6e6e6", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10), padding=(10, 4))
        style.configure("Compact.TButton", font=("Segoe UI", 9), padding=(5, 2))
        style.configure("TEntry", fieldbackground="#1b1f2a", foreground="#e6e6e6")
        self.root.configure(bg="#0f111a")

    def _build_layout(self):
        self.main_paned = ttk.PanedWindow(self.root, orient=tk.VERTICAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True)

        self.top_paned = ttk.PanedWindow(self.main_paned, orient=tk.HORIZONTAL)
        self.main_paned.add(self.top_paned, weight=2)

        self.bottom_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(self.bottom_frame, weight=1)

        self.left_frame = ttk.Frame(self.top_paned)
        self.right_frame = ttk.Frame(self.top_paned)
        self.top_paned.add(self.left_frame, weight=1)
        self.top_paned.add(self.right_frame, weight=2)

        self._build_forms(self.left_frame)
        self._build_visualization(self.right_frame)
        self._build_bottom(self.bottom_frame)

        self.root.bind("<Configure>", self._on_resize)

    def _build_forms(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)

        env_box = ttk.LabelFrame(parent, text="Environment Configuration")
        env_box.grid(row=0, column=0, sticky="ew", padx=8, pady=6)
        env_box.columnconfigure(1, weight=1)

        ttk.Label(env_box, text="Environment").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.env_var = tk.StringVar(value="MountainCar-v0")
        ttk.Entry(env_box, textvariable=self.env_var).grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(env_box, text="goal_velocity").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.goal_velocity_var = tk.StringVar(value="0.1")
        ttk.Entry(env_box, textvariable=self.goal_velocity_var, width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(env_box, text="x_init").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.x_init_var = tk.StringVar(value="3.141592653589793")
        ttk.Entry(env_box, textvariable=self.x_init_var, width=10).grid(row=2, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(env_box, text="y_init").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.y_init_var = tk.StringVar(value="1")
        ttk.Entry(env_box, textvariable=self.y_init_var, width=10).grid(row=3, column=1, sticky="ew", padx=4, pady=2)

        self.visualize_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(env_box, text="Visualize", variable=self.visualize_var).grid(row=4, column=0, sticky="w", padx=4, pady=2)

        ttk.Label(env_box, text="Frame interval (ms)").grid(row=4, column=1, sticky="w", padx=4, pady=2)
        self.frame_interval_var = tk.StringVar(value="10")
        ttk.Entry(env_box, textvariable=self.frame_interval_var, width=8).grid(row=4, column=1, sticky="e", padx=4, pady=2)

        episode_box = ttk.LabelFrame(parent, text="Episode Configuration")
        episode_box.grid(row=1, column=0, sticky="ew", padx=8, pady=6)
        for i in range(4):
            episode_box.columnconfigure(i, weight=1)

        self.compare_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(episode_box, text="Compare Methods", variable=self.compare_var).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        ttk.Label(episode_box, text="Episodes").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.episodes_var = tk.StringVar(value="200")
        ttk.Entry(episode_box, textvariable=self.episodes_var, width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(episode_box, text="Max-Steps").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        self.max_steps_var = tk.StringVar(value="200")
        ttk.Entry(episode_box, textvariable=self.max_steps_var, width=10).grid(row=1, column=3, sticky="ew", padx=4, pady=2)

        ttk.Label(episode_box, text="Alpha").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.alpha_var = tk.StringVar(value="0.0005")
        ttk.Entry(episode_box, textvariable=self.alpha_var, width=10).grid(row=2, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(episode_box, text="Gamma").grid(row=2, column=2, sticky="w", padx=4, pady=2)
        self.gamma_var = tk.StringVar(value="0.99")
        ttk.Entry(episode_box, textvariable=self.gamma_var, width=10).grid(row=2, column=3, sticky="ew", padx=4, pady=2)

        tune_box = ttk.LabelFrame(parent, text="Parameter Tuning")
        tune_box.grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        for i in range(4):
            tune_box.columnconfigure(i, weight=1)

        self.tune_enabled_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tune_box, text="Enable Tuning", variable=self.tune_enabled_var).grid(row=0, column=0, columnspan=4, sticky="w", padx=4, pady=2)

        ttk.Label(tune_box, text="Parameter").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.tune_param_var = tk.StringVar(value="learning_rate")
        param_box = ttk.Combobox(
            tune_box,
            textvariable=self.tune_param_var,
            values=["learning_rate", "gamma", "buffer_size", "batch_size", "target_update_interval"],
            state="readonly",
        )
        param_box.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(tune_box, text="Min").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        self.tune_min_var = tk.StringVar(value="0.0002")
        ttk.Entry(tune_box, textvariable=self.tune_min_var, width=8).grid(row=1, column=3, sticky="ew", padx=4, pady=2)

        ttk.Label(tune_box, text="Max").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.tune_max_var = tk.StringVar(value="0.001")
        ttk.Entry(tune_box, textvariable=self.tune_max_var, width=8).grid(row=2, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(tune_box, text="Step").grid(row=2, column=2, sticky="w", padx=4, pady=2)
        self.tune_step_var = tk.StringVar(value="0.0002")
        ttk.Entry(tune_box, textvariable=self.tune_step_var, width=8).grid(row=2, column=3, sticky="ew", padx=4, pady=2)

        method_box = ttk.LabelFrame(parent, text="Methods")
        method_box.grid(row=3, column=0, sticky="ew", padx=8, pady=6)
        method_box.columnconfigure(0, weight=1)
        method_box.rowconfigure(0, weight=1)

        self.method_notebook = ttk.Notebook(method_box)
        self.method_notebook.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self._build_method_tabs()

        ttk.Button(parent, text="Apply and Reset", command=self._on_apply_reset).grid(row=4, column=0, sticky="ew", padx=8, pady=8)

    def _build_visualization(self, parent: ttk.Frame):
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.visual_label = ttk.Label(parent)
        self.visual_label.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self._current_photo = None

    def _build_bottom(self, parent: ttk.Frame):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.grid(row=0, column=0, sticky="ew", padx=8, pady=(6, 4))

        btn_row = ttk.Frame(parent)
        btn_row.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        for i in range(8):
            btn_row.columnconfigure(i, weight=1)

        self.buttons = {}
        defs = [
            ("Add Job", self._on_add_job),
            ("Train", self._on_train),
            ("Training Status", self._open_status_window),
            ("Save plot", self._on_save_plot),
            ("Cancel Training", self._on_cancel_training),
            ("Reset Training", self._on_reset_training),
            ("Save", self._on_save_jobs),
            ("Load", self._on_load_jobs),
        ]
        for idx, (label, command) in enumerate(defs):
            b = ttk.Button(btn_row, text=label, command=command)
            b.grid(row=0, column=idx, sticky="ew", padx=2)
            self.buttons[label] = b

        self.plot_container = ttk.Frame(parent)
        self.plot_container.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.plot_container.rowconfigure(0, weight=1)
        self.plot_container.columnconfigure(0, weight=1)

    def _build_plot(self):
        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.figure.patch.set_facecolor("#0f111a")
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor("#0f111a")
        self.ax.tick_params(colors="#b5b5b5")
        self.ax.xaxis.label.set_color("#b5b5b5")
        self.ax.yaxis.label.set_color("#b5b5b5")
        self.ax.grid(True, color="#2a2f3a", linestyle="--", alpha=0.5)
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Return")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _on_apply_reset(self):
        self._on_reset_training()

    def _build_method_tabs(self):
        defaults = {
            "D3QN": {
                "learning_rate": "0.0005",
                "gamma": "0.99",
                "buffer_size": "100000",
                "batch_size": "128",
                "learning_starts": "2000",
                "train_freq": "4",
                "gradient_steps": "1",
                "target_update_interval": "300",
                "exploration_fraction": "0.3",
                "exploration_initial_eps": "1.0",
                "exploration_final_eps": "0.05",
                "hidden_layers": "256,256,128",
                "activation": "relu",
            },
            "Double DQN + Prioritized Experience Replay": {
                "learning_rate": "0.0005",
                "gamma": "0.99",
                "buffer_size": "150000",
                "batch_size": "128",
                "learning_starts": "2000",
                "train_freq": "4",
                "gradient_steps": "1",
                "target_update_interval": "500",
                "exploration_fraction": "0.3",
                "exploration_initial_eps": "1.0",
                "exploration_final_eps": "0.05",
                "hidden_layers": "256,256",
                "activation": "relu",
            },
            "Dueling DQN": {
                "learning_rate": "0.0005",
                "gamma": "0.99",
                "buffer_size": "100000",
                "batch_size": "128",
                "learning_starts": "2000",
                "train_freq": "4",
                "gradient_steps": "1",
                "target_update_interval": "500",
                "exploration_fraction": "0.3",
                "exploration_initial_eps": "1.0",
                "exploration_final_eps": "0.05",
                "hidden_layers": "256,256,128",
                "activation": "relu",
            },
        }

        for method, cfg in defaults.items():
            tab = ttk.Frame(self.method_notebook)
            self.method_notebook.add(tab, text=method)
            for col in range(4):
                tab.columnconfigure(col, weight=1)

            vars_for_method = {
                "learning_rate": tk.StringVar(value=cfg["learning_rate"]),
                "gamma": tk.StringVar(value=cfg["gamma"]),
                "buffer_size": tk.StringVar(value=cfg["buffer_size"]),
                "batch_size": tk.StringVar(value=cfg["batch_size"]),
                "learning_starts": tk.StringVar(value=cfg["learning_starts"]),
                "train_freq": tk.StringVar(value=cfg["train_freq"]),
                "gradient_steps": tk.StringVar(value=cfg["gradient_steps"]),
                "target_update_interval": tk.StringVar(value=cfg["target_update_interval"]),
                "exploration_fraction": tk.StringVar(value=cfg["exploration_fraction"]),
                "exploration_initial_eps": tk.StringVar(value=cfg["exploration_initial_eps"]),
                "exploration_final_eps": tk.StringVar(value=cfg["exploration_final_eps"]),
                "hidden_layers": tk.StringVar(value=cfg["hidden_layers"]),
                "activation": tk.StringVar(value=cfg["activation"]),
            }
            self.method_vars[method] = vars_for_method

            ttk.Label(tab, text="Alpha").grid(row=0, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["learning_rate"], width=10).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Gamma").grid(row=0, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["gamma"], width=10).grid(row=0, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Buffer Size").grid(row=1, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["buffer_size"], width=10).grid(row=1, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Batch Size").grid(row=1, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["batch_size"], width=10).grid(row=1, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Learning Starts").grid(row=2, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["learning_starts"], width=10).grid(row=2, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Train Freq").grid(row=2, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["train_freq"], width=10).grid(row=2, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Gradient Steps").grid(row=3, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["gradient_steps"], width=10).grid(row=3, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Target Update").grid(row=3, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["target_update_interval"], width=10).grid(row=3, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Exploration Fraction").grid(row=4, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["exploration_fraction"], width=10).grid(row=4, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Initial Epsilon").grid(row=4, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["exploration_initial_eps"], width=10).grid(row=4, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Final Epsilon").grid(row=5, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["exploration_final_eps"], width=10).grid(row=5, column=1, sticky="ew", padx=4, pady=2)
            ttk.Label(tab, text="Hidden Layers").grid(row=5, column=2, sticky="w", padx=4, pady=2)
            ttk.Entry(tab, textvariable=vars_for_method["hidden_layers"]).grid(row=5, column=3, sticky="ew", padx=4, pady=2)

            ttk.Label(tab, text="Activation").grid(row=6, column=0, sticky="w", padx=4, pady=2)
            ttk.Combobox(tab, textvariable=vars_for_method["activation"], values=["relu", "tanh", "elu", "gelu"], state="readonly").grid(
                row=6,
                column=1,
                sticky="ew",
                padx=4,
                pady=2,
            )

    def _parse_layers(self, layers_text: str):
        return [int(part.strip()) for part in layers_text.split(",") if part.strip()]

    def _selected_method_name(self) -> str:
        tab_id = self.method_notebook.select()
        return self.method_notebook.tab(tab_id, "text")

    def _build_algo_cfg(self, method_name: str) -> AlgorithmConfig:
        vars_for_method = self.method_vars[method_name]
        net = NetworkConfig(
            hidden_layers=self._parse_layers(vars_for_method["hidden_layers"].get()),
            activation=vars_for_method["activation"].get(),
        )
        return AlgorithmConfig(
            algorithm=method_name,
            learning_rate=float(vars_for_method["learning_rate"].get()),
            gamma=float(vars_for_method["gamma"].get()),
            buffer_size=int(float(vars_for_method["buffer_size"].get())),
            batch_size=int(float(vars_for_method["batch_size"].get())),
            learning_starts=int(float(vars_for_method["learning_starts"].get())),
            train_freq=int(float(vars_for_method["train_freq"].get())),
            gradient_steps=int(float(vars_for_method["gradient_steps"].get())),
            target_update_interval=int(float(vars_for_method["target_update_interval"].get())),
            exploration_fraction=float(vars_for_method["exploration_fraction"].get()),
            exploration_initial_eps=float(vars_for_method["exploration_initial_eps"].get()),
            exploration_final_eps=float(vars_for_method["exploration_final_eps"].get()),
            net=net,
        )

    def _build_episode_cfg(self) -> EpisodeConfig:
        return EpisodeConfig(episodes=int(self.episodes_var.get()), max_steps=int(self.max_steps_var.get()))

    def _build_tune_cfg(self) -> TuneConfig:
        return TuneConfig(
            enabled=self.tune_enabled_var.get(),
            parameter=self.tune_param_var.get(),
            min_value=float(self.tune_min_var.get()),
            max_value=float(self.tune_max_var.get()),
            step=float(self.tune_step_var.get()),
        )

    def _on_add_job(self):
        try:
            episode_cfg = self._build_episode_cfg()
            tune_cfg = self._build_tune_cfg()
            env_kwargs = {
                "goal_velocity": float(self.goal_velocity_var.get()),
                "x_init": float(self.x_init_var.get()),
                "y_init": float(self.y_init_var.get()),
            }

            if tune_cfg.enabled:
                selected_method = self._selected_method_name()
                created = self.manager.create_jobs_for_compare_or_tuning(
                    env_id=self.env_var.get(),
                    episode_cfg=episode_cfg,
                    base_algorithm=self._build_algo_cfg(selected_method),
                    compare_methods=False,
                    tuning=tune_cfg,
                    env_kwargs=env_kwargs,
                )
            elif self.compare_var.get():
                created = []
                for method_name in self.method_vars.keys():
                    config = JobConfig(
                        name=method_name,
                        env_id=self.env_var.get(),
                        env_kwargs=env_kwargs,
                        algorithm=self._build_algo_cfg(method_name),
                        episodes=episode_cfg,
                        visible=True,
                    )
                    created.append(self.manager.add_job(config))
            else:
                selected_method = self._selected_method_name()
                config = JobConfig(
                    name=selected_method,
                    env_id=self.env_var.get(),
                    env_kwargs=env_kwargs,
                    algorithm=self._build_algo_cfg(selected_method),
                    episodes=episode_cfg,
                    visible=True,
                )
                created = [self.manager.add_job(config)]
        except Exception as exc:
            messagebox.showerror("Invalid config", str(exc))
            return

        for job_id in created:
            self.event_bus.publish(
                "EpisodeCompleted",
                {
                    "job_id": job_id,
                    "episode": 0,
                    "episodes_total": int(self.episodes_var.get()),
                    "return": 0.0,
                    "moving_avg": 0.0,
                    "epsilon": 1.0,
                    "loss": np.nan,
                    "duration": 0.0,
                    "steps": 0,
                    "algorithm": self.manager.jobs[job_id].config.algorithm.algorithm,
                    "visible": True,
                },
            )

    def _on_train(self):
        if not self.manager.jobs:
            self._on_add_job()
        self.manager.start_all_pending()

    def _on_cancel_training(self):
        self.manager.cancel_all()

    def _on_reset_training(self):
        self.manager.reset_training()
        self.ax.clear()
        self._style_axes()
        self.canvas.draw_idle()

    def _on_save_plot(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG", "*.png")])
        if path:
            self.figure.savefig(path)

    def _on_save_jobs(self):
        path = filedialog.askdirectory()
        if not path:
            return
        self.manager.save_all(Path(path))

    def _on_load_jobs(self):
        path = filedialog.askdirectory()
        if not path:
            return
        loaded = self.manager.load_all(Path(path))
        for job_id in loaded:
            job = self.manager.jobs[job_id]
            self.event_bus.publish(
                "EpisodeCompleted",
                {
                    "job_id": job_id,
                    "episode": job.current_episode,
                    "episodes_total": job.config.episodes.episodes,
                    "return": job.returns[-1] if job.returns else 0.0,
                    "moving_avg": job.moving_avg[-1] if job.moving_avg else 0.0,
                    "epsilon": job.epsilon,
                    "loss": job.loss,
                    "duration": job.last_duration,
                    "steps": job.last_steps,
                    "algorithm": job.config.algorithm.algorithm,
                    "visible": job.visible,
                },
            )

    def _open_status_window(self):
        if self.status_window and self.status_window.winfo_exists():
            self.status_window.focus_set()
            return
        self.status_window = TrainingStatusWindow(self.root, self.manager)
        self.status_window.tree.bind("<<TreeviewSelect>>", self._on_status_select)

    def _on_status_select(self, _event=None):
        if not self.status_window:
            return
        self._selected_status_job_id = self.status_window.selected_job_id()

    def _style_axes(self):
        self.ax.set_facecolor("#0f111a")
        self.ax.tick_params(colors="#b5b5b5")
        self.ax.xaxis.label.set_color("#b5b5b5")
        self.ax.yaxis.label.set_color("#b5b5b5")
        self.ax.grid(True, color="#2a2f3a", linestyle="--", alpha=0.5)
        self.ax.set_xlabel("Episodes")
        self.ax.set_ylabel("Return")

    def _plot_jobs(self):
        self.ax.clear()
        self._style_axes()
        palette = ["#4cc9f0", "#f72585", "#b8de6f", "#ffd166", "#c77dff", "#ff9f1c"]
        legend_loc = "upper right"

        index = 0
        for job in self.manager.jobs.values():
            if not job.visible or not job.returns:
                continue
            x = np.arange(1, len(job.returns) + 1)
            color = palette[index % len(palette)]
            self.ax.plot(x, job.returns, color=color, alpha=0.35, linewidth=1.0, label=f"{job.config.name} Raw")
            self.ax.plot(x, job.moving_avg, color=color, alpha=1.0, linewidth=2.5, label=f"{job.config.name} Avg")
            if len(job.returns) > 4:
                legend_loc = "lower left"
            index += 1

        if index:
            legend = self.ax.legend(loc=legend_loc)
            legend.get_frame().set_facecolor("#0f111a")
            legend.get_frame().set_edgecolor("#2a2f3a")
            for text in legend.get_texts():
                text.set_color("#e6e6e6")
        self.canvas.draw_idle()

    def _update_progress(self):
        all_eps = 0
        done = 0
        for job in self.manager.jobs.values():
            all_eps += max(1, job.config.episodes.episodes)
            done += min(job.current_episode, job.config.episodes.episodes)
        percent = (done / all_eps * 100.0) if all_eps else 0.0
        self.progress["value"] = percent

    def _render_frame(self):
        if not self.visualize_var.get():
            return
        frame = self.manager.get_latest_frame(self._selected_status_job_id)
        if frame is None:
            return

        try:
            from PIL import Image, ImageTk
        except ImportError:
            return

        frame_h, frame_w = frame.shape[:2]
        box_w = max(1, self.visual_label.winfo_width())
        box_h = max(1, self.visual_label.winfo_height())

        scale = min(box_w / frame_w, box_h / frame_h)
        target_w = max(1, int(frame_w * scale))
        target_h = max(1, int(frame_h * scale))

        image = Image.fromarray(frame).resize((target_w, target_h), Image.Resampling.NEAREST)
        photo = ImageTk.PhotoImage(image)
        self._current_photo = photo
        self.visual_label.configure(image=photo)

    def _schedule_polling(self):
        self._poll_events()
        interval = 10
        try:
            interval = max(1, int(self.frame_interval_var.get()))
        except Exception:
            pass
        self.root.after(interval, self._schedule_polling)

    def _poll_events(self):
        events = self.event_bus.poll(200)
        dirty_plot = False

        for event in events:
            etype = event["type"]
            payload = event["payload"]
            if etype == "EpisodeCompleted":
                if self.status_window and self.status_window.winfo_exists():
                    self.status_window.upsert_row(payload)
                dirty_plot = True
            elif etype == "JobRemoved":
                if self.status_window and self.status_window.winfo_exists():
                    self.status_window.remove_row(payload["job_id"])
                dirty_plot = True
            elif etype == "Error":
                messagebox.showerror(
                    "Training Error",
                    f"Job {payload.get('job_name', payload.get('job_id', '?'))} ({payload.get('algorithm', '-')}) failed:\n{payload.get('message', 'Unknown error')}",
                )

        self._update_progress()
        now = self.root.tk.call("clock", "milliseconds")
        if dirty_plot and (now - self._last_plot_draw >= self._plot_redraw_ms):
            self._plot_jobs()
            self._last_plot_draw = now
        self._render_frame()

    def _on_resize(self, _event=None):
        if self._resize_after_id is not None:
            self.root.after_cancel(self._resize_after_id)
        self._resize_after_id = self.root.after(100, self._apply_responsive_styles)

    def _apply_responsive_styles(self):
        width = self.root.winfo_width()
        use_compact = self._current_compact

        if self._current_compact and width > 1140:
            use_compact = False
        elif not self._current_compact and width < 1060:
            use_compact = True

        if use_compact != self._current_compact:
            self._current_compact = use_compact
            style = "Compact.TButton" if use_compact else "TButton"
            for button in self.buttons.values():
                button.configure(style=style)
