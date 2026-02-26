from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
import numpy as np

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import messagebox, ttk

try:
	from PIL import Image, ImageTk
except Exception:
	Image = None
	ImageTk = None

from MountainCar_logic import Trainer, make_default_trainer


class MountainCarGUI:
	def __init__(self, root: tk.Tk) -> None:
		self.root = root
		self.root.title("MountainCar RL")
		self.root.geometry("1300x850")

		self.base_dir = Path(__file__).resolve().parent
		self.results_dir = self.base_dir / "results_csv"
		self.plots_dir = self.base_dir / "plots"
		self.results_dir.mkdir(parents=True, exist_ok=True)
		self.plots_dir.mkdir(parents=True, exist_ok=True)

		self.trainer: Trainer = make_default_trainer(output_dir=self.base_dir)
		self.current_policy = "Dueling DQN"
		self._train_thread: Optional[threading.Thread] = None
		self._pause_event = threading.Event()
		self._pause_event.set()
		self._stop_requested = False
		self._lock = threading.Lock()
		self._pending: Dict[str, object] = {
			"current_episode": 0,
			"current_step": 0,
			"steps_effective_max": 1,
			"epsilon": 0.0,
			"animation_fps": 10,
			"reward_snapshot": [],
			"frame": None,
			"finalize_run": False,
			"policy_label": "",
			"episodes_total": 1,
			"steps_total": 1,
		}
		self._last_plot_update = 0.0
		self._last_plot_rewards_len = 0
		self._last_frame_np: Optional[np.ndarray] = None
		self._last_env_draw_ts = 0.0
		self._single_episode_last_draw_ts = 0.0
		self._is_closing = False
		self._line_registry: Dict[object, object] = {}
		self._run_entries: List[Dict[str, object]] = []
		self._active_training_run_id: Optional[int] = None
		self._plot_color_cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
		self._plot_idx = 0
		self._best_position_current_run: Optional[float] = None

		self._build_vars()
		self._build_layout()
		self._draw_env_frame(force=True)
		self._schedule_ui_pump()

	def _build_vars(self) -> None:
		self.animation_fps_var = tk.StringVar(value="10")
		self.goal_velocity_var = tk.StringVar(value="0.0")
		self.x_var = tk.StringVar(value="3.14159")
		self.y_var = tk.StringVar(value="1.0")

		self.policy_var = tk.StringVar(value="Dueling DQN")
		self.max_steps_var = tk.StringVar(value="200")
		self.episodes_var = tk.StringVar(value="500")
		self.epsilon_max_var = tk.StringVar(value="1.0")
		self.epsilon_decay_var = tk.StringVar(value="0.997")
		self.epsilon_min_var = tk.StringVar(value="0.02")

		self.gamma_var = tk.StringVar(value="0.99")
		self.lr_var = tk.StringVar(value="0.0005")
		self.replay_size_var = tk.StringVar(value="100000")
		self.batch_size_var = tk.StringVar(value="128")
		self.target_update_var = tk.StringVar(value="250")
		self.replay_warmup_var = tk.StringVar(value="1000")
		self.train_every_var = tk.StringVar(value="2")
		self.activation_function_var = tk.StringVar(value="ReLU")
		self._activation_function_options = ("ReLU", "Tanh", "LeakyReLU", "ELU")
		self.hidden_layer_var = tk.StringVar(value="256,256")

		self.per_alpha_var = tk.StringVar(value="0.6")
		self.per_beta_start_var = tk.StringVar(value="0.4")
		self.per_beta_frames_var = tk.StringVar(value="50000")
		self.per_eps_var = tk.StringVar(value="0.00001")

		self.moving_avg_var = tk.StringVar(value="20")

		self._policy_ui_defaults: Dict[str, Dict[str, str]] = {
			"Dueling DQN": {
				"epsilon_max": "1.0",
				"epsilon_decay": "0.995",
				"epsilon_min": "0.02",
				"gamma": "0.99",
				"learning_rate": "0.001",
				"replay_size": "50000",
				"batch_size": "64",
				"target_update": "100",
				"replay_warmup": "1000",
				"train_every": "2",
				"activation_function": "ReLU",
				"hidden_layer_size": "128,128",
			},
			"D3QN": {
				"epsilon_max": "1.0",
				"epsilon_decay": "0.996",
				"epsilon_min": "0.02",
				"gamma": "0.99",
				"learning_rate": "0.0008",
				"replay_size": "50000",
				"batch_size": "64",
				"target_update": "100",
				"replay_warmup": "1000",
				"train_every": "2",
				"activation_function": "ReLU",
				"hidden_layer_size": "128,128",
			},
			"DDQN+PER": {
				"epsilon_max": "1.0",
				"epsilon_decay": "0.997",
				"epsilon_min": "0.02",
				"gamma": "0.99",
				"learning_rate": "0.0005",
				"replay_size": "75000",
				"batch_size": "64",
				"target_update": "100",
				"replay_warmup": "1000",
				"train_every": "2",
				"activation_function": "ReLU",
				"hidden_layer_size": "128,128",
				"alpha": "0.6",
				"beta_start": "0.4",
				"beta_frames": "80000",
				"eps_prio": "0.00001",
			},
		}

		self.status_var = tk.StringVar(value="Epsilon: 0.00 | Current x: n/a | Best x: n/a")

	def _build_layout(self) -> None:
		self.root.grid_columnconfigure(0, weight=1)
		self.root.grid_columnconfigure(1, weight=0)
		self.root.grid_rowconfigure(0, weight=3)
		self.root.grid_rowconfigure(1, weight=0)
		self.root.grid_rowconfigure(2, weight=0)
		self.root.grid_rowconfigure(3, weight=2)

		self.environment_frame = ttk.LabelFrame(self.root, text="Environment")
		self.environment_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
		self.environment_frame.grid_rowconfigure(0, weight=1)
		self.environment_frame.grid_columnconfigure(0, weight=1)

		self.env_canvas = tk.Canvas(self.environment_frame, highlightthickness=0)
		self.env_canvas.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
		self._env_tk_img = None
		self._env_canvas_image_id = None
		self.env_canvas.bind("<Configure>", lambda _e: self._draw_env_frame())
		self.environment_frame.bind("<Configure>", lambda _e: self._draw_env_frame())

		self.parameters_outer = ttk.LabelFrame(self.root, text="Parameters")
		self.parameters_outer.grid(row=0, column=1, sticky="ns", padx=8, pady=8)
		self.parameters_outer.grid_rowconfigure(0, weight=1)
		self.parameters_outer.grid_columnconfigure(0, weight=1)

		self.params_canvas = tk.Canvas(self.parameters_outer, width=430, highlightthickness=0)
		self.params_scrollbar = ttk.Scrollbar(self.parameters_outer, orient="vertical", command=self.params_canvas.yview)
		self.params_inner = ttk.Frame(self.params_canvas)
		self.params_inner.bind(
			"<Configure>",
			self._on_params_content_configure,
		)
		self._params_window_id = self.params_canvas.create_window((0, 0), window=self.params_inner, anchor="nw")
		self.params_canvas.configure(yscrollcommand=self.params_scrollbar.set)
		self.params_canvas.grid(row=0, column=0, sticky="nsew")
		self.params_canvas.bind("<Configure>", self._on_params_canvas_configure)
		self.params_canvas.bind("<Enter>", self._bind_mousewheel_for_params)
		self.params_canvas.bind("<Leave>", self._unbind_mousewheel_for_params)
		self.parameters_outer.bind("<Configure>", self._on_params_outer_configure)

		self._build_parameter_groups()
		self._update_parameter_scrollbar_visibility()

		self.controls_frame = ttk.LabelFrame(self.root, text="Controls")
		self.controls_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
		for col in range(7):
			self.controls_frame.grid_columnconfigure(col, weight=1)

		self.run_episode_btn = ttk.Button(self.controls_frame, text="Run single episode", command=self.run_single_episode)
		self.train_btn = ttk.Button(self.controls_frame, text="Train and Run", command=self.train_and_run)
		self.pause_btn = ttk.Button(self.controls_frame, text="Pause", command=self.toggle_pause)
		self.reset_btn = ttk.Button(self.controls_frame, text="Reset All", command=self.reset_all)
		self.clear_btn = ttk.Button(self.controls_frame, text="Clear Plot", command=self.clear_plot)
		self.save_csv_btn = ttk.Button(self.controls_frame, text="Save samplings CSV", command=self.save_samplings_csv)
		self.save_plot_btn = ttk.Button(self.controls_frame, text="Save Plot PNG", command=self.save_plot_png)

		buttons = [
			self.run_episode_btn,
			self.train_btn,
			self.pause_btn,
			self.reset_btn,
			self.clear_btn,
			self.save_csv_btn,
			self.save_plot_btn,
		]
		for i, button in enumerate(buttons):
			button.grid(row=0, column=i, padx=4, pady=4, sticky="ew")

		self.current_run_frame = ttk.LabelFrame(self.root, text="Current Run")
		self.current_run_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=(0, 8))
		self.current_run_frame.grid_columnconfigure(0, weight=0, minsize=80)
		self.current_run_frame.grid_columnconfigure(1, weight=1)

		ttk.Label(self.current_run_frame, text="Steps").grid(row=0, column=0, sticky="w", padx=4, pady=4)
		self.steps_progress = ttk.Progressbar(self.current_run_frame, orient="horizontal", mode="determinate")
		self.steps_progress.grid(row=0, column=1, sticky="ew", padx=4, pady=4)

		ttk.Label(self.current_run_frame, text="Episodes").grid(row=1, column=0, sticky="w", padx=4, pady=4)
		self.episodes_progress = ttk.Progressbar(self.current_run_frame, orient="horizontal", mode="determinate")
		self.episodes_progress.grid(row=1, column=1, sticky="ew", padx=4, pady=4)

		ttk.Label(self.current_run_frame, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=4)

		self.plot_frame = ttk.LabelFrame(self.root, text="Live Plot")
		self.plot_frame.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=8, pady=(0, 8))
		self.plot_frame.grid_rowconfigure(0, weight=1)
		self.plot_frame.grid_columnconfigure(0, weight=1)

		self.figure = Figure(figsize=(8, 4), dpi=100)
		self.ax = self.figure.add_subplot(111)
		self.ax.set_xlabel("Episode")
		self.ax.set_ylabel("Reward")
		self.figure.subplots_adjust(right=0.76)

		self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
		self.canvas_plot_widget = self.canvas_plot.get_tk_widget()
		self.canvas_plot_widget.grid(row=0, column=0, sticky="nsew")

		self.figure.canvas.mpl_connect("pick_event", self._on_pick_legend)

		self.root.protocol("WM_DELETE_WINDOW", self._on_close)

	def _build_parameter_groups(self) -> None:
		env_group = ttk.LabelFrame(self.params_inner, text="Environment")
		env_group.grid(row=0, column=0, sticky="ew", padx=6, pady=6)
		for c in range(4):
			env_group.grid_columnconfigure(c, weight=1)

		ttk.Button(env_group, text="Update", command=self.update_environment).grid(row=0, column=0, columnspan=4, sticky="ew", padx=4, pady=3)
		self._add_four_col_row(env_group, 1, "animation FPS", self.animation_fps_var, "goal velocity", self.goal_velocity_var)
		self._add_four_col_row(env_group, 2, "x", self.x_var, "y", self.y_var)

		general_group = ttk.LabelFrame(self.params_inner, text="General")
		general_group.grid(row=1, column=0, sticky="ew", padx=6, pady=6)
		for c in range(4):
			general_group.grid_columnconfigure(c, weight=1)

		ttk.Label(general_group, text="policy").grid(row=0, column=0, sticky="w", padx=4, pady=3)
		policy_menu = ttk.OptionMenu(general_group, self.policy_var, self.policy_var.get(), "Dueling DQN", "D3QN", "DDQN+PER", command=self._on_policy_change)
		policy_menu.grid(row=0, column=1, columnspan=3, sticky="ew", padx=4, pady=3)

		self._add_four_col_row(general_group, 1, "max steps", self.max_steps_var, "episodes", self.episodes_var)
		self._add_four_col_row(general_group, 2, "epsilon max", self.epsilon_max_var, "epsilon decay", self.epsilon_decay_var)
		self._add_four_col_row(general_group, 3, "epsilon min", self.epsilon_min_var, "", tk.StringVar(value=""), second_enabled=False)

		self.specific_group = ttk.LabelFrame(self.params_inner, text="Specific")
		self.specific_group.grid(row=2, column=0, sticky="ew", padx=6, pady=6)
		for c in range(4):
			self.specific_group.grid_columnconfigure(c, weight=1)
		self._rebuild_specific_inputs()

		live_plot_group = ttk.LabelFrame(self.params_inner, text="Live Plot")
		live_plot_group.grid(row=3, column=0, sticky="ew", padx=6, pady=6)
		for c in range(4):
			live_plot_group.grid_columnconfigure(c, weight=1)
		self._add_four_col_row(live_plot_group, 0, "moving average values", self.moving_avg_var, "", tk.StringVar(value=""), second_enabled=False)

	def _add_four_col_row(
		self,
		parent: ttk.LabelFrame,
		row: int,
		l1: str,
		v1: tk.StringVar,
		l2: str,
		v2: tk.StringVar,
		second_enabled: bool = True,
	) -> None:
		ttk.Label(parent, text=l1).grid(row=row, column=0, sticky="w", padx=4, pady=3)
		ttk.Entry(parent, textvariable=v1, width=9).grid(row=row, column=1, sticky="ew", padx=4, pady=3)
		ttk.Label(parent, text=l2).grid(row=row, column=2, sticky="w", padx=4, pady=3)
		state = "normal" if second_enabled else "disabled"
		ttk.Entry(parent, textvariable=v2, width=9, state=state).grid(row=row, column=3, sticky="ew", padx=4, pady=3)

	def _on_params_content_configure(self, _event=None) -> None:
		self.params_canvas.configure(scrollregion=self.params_canvas.bbox("all"))
		self._update_parameter_scrollbar_visibility()

	def _on_params_canvas_configure(self, event) -> None:
		self.params_canvas.itemconfigure(self._params_window_id, width=event.width)
		self._update_parameter_scrollbar_visibility()

	def _on_params_outer_configure(self, _event=None) -> None:
		self._update_parameter_scrollbar_visibility()

	def _update_parameter_scrollbar_visibility(self) -> None:
		self.root.after_idle(self._update_parameter_scrollbar_visibility_now)

	def _update_parameter_scrollbar_visibility_now(self) -> None:
		content_height = self.params_inner.winfo_reqheight()
		canvas_height = self.params_canvas.winfo_height()
		if canvas_height <= 1:
			return

		needs_scrollbar = content_height > canvas_height + 1
		if needs_scrollbar:
			if not self.params_scrollbar.winfo_ismapped():
				self.params_scrollbar.grid(row=0, column=1, sticky="ns")
		else:
			if self.params_scrollbar.winfo_ismapped():
				self.params_scrollbar.grid_remove()
			self.params_canvas.yview_moveto(0)

	def _bind_mousewheel_for_params(self, _event=None) -> None:
		self.root.bind_all("<MouseWheel>", self._on_params_mousewheel)

	def _unbind_mousewheel_for_params(self, _event=None) -> None:
		self.root.unbind_all("<MouseWheel>")

	def _on_params_mousewheel(self, event) -> None:
		if not self.params_scrollbar.winfo_ismapped():
			return
		delta = event.delta
		if delta == 0:
			return
		units = -1 if delta > 0 else 1
		self.params_canvas.yview_scroll(units, "units")

	def _clear_frame_children(self, frame: ttk.LabelFrame) -> None:
		for child in frame.winfo_children():
			child.destroy()

	def _rebuild_specific_inputs(self) -> None:
		self._clear_frame_children(self.specific_group)
		self._add_four_col_row(self.specific_group, 0, "gamma", self.gamma_var, "learning rate", self.lr_var)
		self._add_four_col_row(self.specific_group, 1, "replay size", self.replay_size_var, "batch size", self.batch_size_var)
		self._add_four_col_row(self.specific_group, 2, "target update", self.target_update_var, "hidden layer size", self.hidden_layer_var)
		self._add_four_col_row(self.specific_group, 3, "replay warmup", self.replay_warmup_var, "learning cadence", self.train_every_var)
		ttk.Label(self.specific_group, text="activation function").grid(row=4, column=0, sticky="w", padx=4, pady=3)
		activation_menu = ttk.OptionMenu(
			self.specific_group,
			self.activation_function_var,
			self.activation_function_var.get(),
			*self._activation_function_options,
		)
		activation_menu.grid(row=4, column=1, columnspan=3, sticky="ew", padx=4, pady=3)

		if self.policy_var.get() == "DDQN+PER":
			self._add_four_col_row(self.specific_group, 5, "per alpha", self.per_alpha_var, "beta start", self.per_beta_start_var)
			self._add_four_col_row(self.specific_group, 6, "beta frames", self.per_beta_frames_var, "per epsilon", self.per_eps_var)
		self._update_parameter_scrollbar_visibility()

	def _on_policy_change(self, *_args) -> None:
		self._apply_policy_defaults(self.policy_var.get())
		self._rebuild_specific_inputs()

	def _apply_policy_defaults(self, policy: str) -> None:
		defaults = self._policy_ui_defaults.get(policy)
		if not defaults:
			return
		self.epsilon_max_var.set(defaults.get("epsilon_max", self.epsilon_max_var.get()))
		self.epsilon_decay_var.set(defaults.get("epsilon_decay", self.epsilon_decay_var.get()))
		self.epsilon_min_var.set(defaults.get("epsilon_min", self.epsilon_min_var.get()))
		self.gamma_var.set(defaults.get("gamma", self.gamma_var.get()))
		self.lr_var.set(defaults.get("learning_rate", self.lr_var.get()))
		self.replay_size_var.set(defaults.get("replay_size", self.replay_size_var.get()))
		self.batch_size_var.set(defaults.get("batch_size", self.batch_size_var.get()))
		self.target_update_var.set(defaults.get("target_update", self.target_update_var.get()))
		self.replay_warmup_var.set(defaults.get("replay_warmup", self.replay_warmup_var.get()))
		self.train_every_var.set(defaults.get("train_every", self.train_every_var.get()))
		self.activation_function_var.set(defaults.get("activation_function", self.activation_function_var.get()))
		self.hidden_layer_var.set(defaults.get("hidden_layer_size", self.hidden_layer_var.get()))
		self.per_alpha_var.set(defaults.get("alpha", self.per_alpha_var.get()))
		self.per_beta_start_var.set(defaults.get("beta_start", self.per_beta_start_var.get()))
		self.per_beta_frames_var.set(defaults.get("beta_frames", self.per_beta_frames_var.get()))
		self.per_eps_var.set(defaults.get("eps_prio", self.per_eps_var.get()))

	def _snapshot_training_params(self) -> Dict[str, object]:
		hidden_layers = self._parse_hidden_layers_input(self.hidden_layer_var.get())
		return {
			"policy": self.policy_var.get(),
			"max_steps": int(float(self.max_steps_var.get())),
			"episodes": int(float(self.episodes_var.get())),
			"epsilon_max": float(self.epsilon_max_var.get()),
			"epsilon_decay": float(self.epsilon_decay_var.get()),
			"epsilon_min": float(self.epsilon_min_var.get()),
			"moving_avg": max(1, int(float(self.moving_avg_var.get()))),
			"policy_params": {
				"gamma": float(self.gamma_var.get()),
				"learning_rate": float(self.lr_var.get()),
				"replay_size": int(float(self.replay_size_var.get())),
				"batch_size": int(float(self.batch_size_var.get())),
				"target_update": int(float(self.target_update_var.get())),
				"replay_warmup": int(float(self.replay_warmup_var.get())),
				"train_every": int(float(self.train_every_var.get())),
				"activation_function": str(self.activation_function_var.get()),
				"hidden_layer_size": hidden_layers,
				"alpha": float(self.per_alpha_var.get()),
				"beta_start": float(self.per_beta_start_var.get()),
				"beta_frames": int(float(self.per_beta_frames_var.get())),
				"eps_prio": float(self.per_eps_var.get()),
			},
			"animation_fps": max(1, int(float(self.animation_fps_var.get()))),
		}

	def _parse_hidden_layers_input(self, value: str) -> int | List[int]:
		parts = [p.strip() for p in str(value).split(",") if p.strip()]
		if not parts:
			raise ValueError("hidden layer size must be a positive integer or comma-separated integers")
		layers = [max(1, int(float(p))) for p in parts]
		if len(layers) == 1:
			return layers[0]
		return layers

	def update_environment(self) -> None:
		if self._train_thread and self._train_thread.is_alive():
			messagebox.showwarning("Training active", "Pause/stop training before updating the environment.")
			return
		try:
			goal_velocity = float(self.goal_velocity_var.get())
			x_init = float(self.x_var.get())
			y_init = float(self.y_var.get())
			self.trainer.environment.close()
			self.trainer = make_default_trainer(goal_velocity=goal_velocity, x_init=x_init, y_init=y_init, output_dir=self.base_dir)
			self._draw_env_frame(force=True)
		except Exception as exc:
			messagebox.showerror("Invalid environment settings", str(exc))

	def run_single_episode(self) -> None:
		if self._train_thread and self._train_thread.is_alive():
			messagebox.showwarning("Training active", "Stop the running training first.")
			return
		try:
			self._best_position_current_run = None
			snapshot = self._snapshot_training_params()
			policy = snapshot["policy"]
			epsilon = float(snapshot["epsilon_min"])
			max_steps = int(snapshot["max_steps"])
			self._single_episode_last_draw_ts = 0.0
			policy_params = dict(snapshot["policy_params"])
			self.trainer.get_or_create_agent(policy, overrides=policy_params)

			result = self.trainer.run_episode(policy=policy, epsilon=epsilon, max_steps=max_steps, progress_callback=self._single_episode_progress)
			rewards = [float(result["total_reward"])]
			label = self._build_run_label(snapshot, policy)
			self.steps_progress.configure(maximum=max(1, int(result["steps"])), value=max(1, int(result["steps"])))
			self._append_run_plot(label, rewards, int(snapshot["moving_avg"]))
			self._draw_env_frame(force=True)
		except Exception as exc:
			messagebox.showerror("Run failed", str(exc))

	def _single_episode_progress(self, step: int) -> None:
		fps = max(1, int(float(self.animation_fps_var.get())))
		interval = 1.0 / fps
		now = time.time()
		if now - self._single_episode_last_draw_ts >= interval:
			frame = self.trainer.environment.render_frame()
			if isinstance(frame, np.ndarray) and frame.size:
				self._draw_env_frame(frame=frame)
			self._single_episode_last_draw_ts = now
		remaining = interval - (time.time() - now)
		if remaining > 0:
			time.sleep(remaining)
		self.steps_progress.configure(maximum=max(1, int(float(self.max_steps_var.get()))), value=step)
		self.root.update_idletasks()

	def train_and_run(self) -> None:
		if self._train_thread and self._train_thread.is_alive():
			return
		try:
			snapshot = self._snapshot_training_params()
		except Exception as exc:
			messagebox.showerror("Invalid parameters", str(exc))
			return

		self._best_position_current_run = None
		self._stop_requested = False
		self._pause_event.set()
		self.pause_btn.configure(text="Pause")
		self._last_env_draw_ts = 0.0

		with self._lock:
			run_id = time.time_ns()
			self._pending["current_episode"] = 0
			self._pending["current_step"] = 0
			self._pending["steps_effective_max"] = int(snapshot["max_steps"])
			self._pending["epsilon"] = float(snapshot["epsilon_max"])
			self._pending["animation_fps"] = int(snapshot["animation_fps"])
			self._pending["episodes_total"] = int(snapshot["episodes"])
			self._pending["steps_total"] = int(snapshot["max_steps"])
			self._pending["reward_snapshot"] = []
			self._pending["finalize_run"] = False
			self._pending["policy_label"] = self._build_run_label(snapshot, str(snapshot["policy"]))
			self._pending["run_id"] = run_id

		self._train_thread = threading.Thread(target=self._training_worker, args=(snapshot,), daemon=True)
		self._train_thread.start()

	def _training_worker(self, snapshot: Dict[str, object]) -> None:
		policy = str(snapshot["policy"])
		episodes = int(snapshot["episodes"])
		max_steps = int(snapshot["max_steps"])
		animation_fps = max(1, int(snapshot.get("animation_fps", 10)))
		frame_interval = 1.0 / animation_fps
		epsilon = float(snapshot["epsilon_max"])
		epsilon_decay = float(snapshot["epsilon_decay"])
		epsilon_min = float(snapshot["epsilon_min"])
		policy_params = dict(snapshot["policy_params"])
		last_frame_capture_ts = 0.0

		self.trainer.get_or_create_agent(policy, overrides=policy_params)
		rewards: List[float] = []

		for episode_idx in range(episodes):
			if self._stop_requested:
				break
			self._pause_event.wait()
			with self._lock:
				self._pending["current_step"] = 0
				self._pending["steps_effective_max"] = max_steps

			def step_callback(step: int) -> None:
				with self._lock:
					self._pending["current_step"] = step
				if self._stop_requested:
					return

			result = self.trainer.run_episode(policy=policy, epsilon=epsilon, max_steps=max_steps, progress_callback=step_callback)
			rewards.append(float(result["total_reward"]))

			with self._lock:
				self._pending["current_episode"] = episode_idx + 1
				self._pending["current_step"] = int(result["steps"])
				self._pending["steps_effective_max"] = max(1, int(result["steps"]))
				self._pending["epsilon"] = float(epsilon)
				self._pending["reward_snapshot"] = rewards.copy()

			epsilon = max(epsilon_min, epsilon * epsilon_decay)

		with self._lock:
			self._pending["finalize_run"] = True

	def toggle_pause(self) -> None:
		if not (self._train_thread and self._train_thread.is_alive()):
			return
		if self._pause_event.is_set():
			self._pause_event.clear()
			self.pause_btn.configure(text="Run")
		else:
			self._pause_event.set()
			self.pause_btn.configure(text="Pause")

	def reset_all(self) -> None:
		self._stop_requested = True
		self._pause_event.set()
		self.pause_btn.configure(text="Pause")
		self._best_position_current_run = None
		self.status_var.set("Epsilon: 0.00 | Current x: n/a | Best x: n/a")
		self.clear_plot()

		with self._lock:
			self._pending["current_episode"] = 0
			self._pending["current_step"] = 0
			self._pending["steps_effective_max"] = 1
			self._pending["epsilon"] = 0.0
			self._pending["reward_snapshot"] = []
			self._pending["finalize_run"] = False

		try:
			self.update_environment()
		except Exception:
			pass

	def clear_plot(self) -> None:
		self._run_entries.clear()
		self._line_registry.clear()
		self._active_training_run_id = None
		self._last_plot_rewards_len = 0
		self.ax.clear()
		self.ax.set_xlabel("Episode")
		self.ax.set_ylabel("Reward")
		self.canvas_plot.draw_idle()

	def save_samplings_csv(self) -> None:
		try:
			snapshot = self._snapshot_training_params()
			policy = str(snapshot["policy"])
			self.trainer.get_or_create_agent(policy, overrides=dict(snapshot["policy_params"]))
			base = f"samplings_{policy.replace('+', 'plus').replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
			self.trainer.train(
				policy=policy,
				num_episodes=max(1, int(snapshot["episodes"])),
				max_steps=max(1, int(snapshot["max_steps"])),
				epsilon=float(snapshot["epsilon_min"]),
				save_csv=base,
			)
			messagebox.showinfo("CSV saved", f"Saved CSV in results_csv: {base}.csv")
		except Exception as exc:
			messagebox.showerror("CSV save failed", str(exc))

	def save_plot_png(self) -> None:
		try:
			snapshot = self._snapshot_training_params()
			name = (
				f"{snapshot['policy']}_eps{snapshot['epsilon_min']}-{snapshot['epsilon_max']}_"
				f"lr{snapshot['policy_params']['learning_rate']}_g{snapshot['policy_params']['gamma']}_"
				f"ep{snapshot['episodes']}_ms{snapshot['max_steps']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
			)
			safe_name = str(name).replace(" ", "_").replace("+", "plus")
			path = self.plots_dir / f"{safe_name}.png"
			self.figure.savefig(path, dpi=150, bbox_inches="tight")
			messagebox.showinfo("Plot saved", f"Saved plot PNG: {path.name}")
		except Exception as exc:
			messagebox.showerror("Plot save failed", str(exc))

	def _build_run_label(self, snapshot: Dict[str, object], policy: str) -> str:
		p = snapshot["policy_params"]
		return f"{policy} | eps({snapshot['epsilon_max']}/{snapshot['epsilon_min']}) | lr={p['learning_rate']}"

	def _moving_average(self, values: List[float], window: int) -> np.ndarray:
		if not values:
			return np.array([])
		arr = np.asarray(values, dtype=np.float32)
		if len(arr) < window:
			return np.array([arr[: i + 1].mean() for i in range(len(arr))], dtype=np.float32)
		kernel = np.ones(window, dtype=np.float32) / window
		head = np.array([arr[: i + 1].mean() for i in range(window - 1)], dtype=np.float32)
		tail = np.convolve(arr, kernel, mode="valid")
		return np.concatenate([head, tail])

	def _append_run_plot(self, label: str, rewards: List[float], moving_avg_window: int) -> None:
		x = np.arange(1, len(rewards) + 1)
		color = self._plot_color_cycle[self._plot_idx % len(self._plot_color_cycle)]
		self._plot_idx += 1

		reward_line, = self.ax.plot(x, rewards, color=color, alpha=0.35, linewidth=1.2, label=f"{label} | reward")
		ma = self._moving_average(rewards, moving_avg_window)
		ma_line, = self.ax.plot(x, ma, color=color, alpha=0.95, linewidth=2.4, label=f"{label} | MA")

		self._run_entries.append({"reward": reward_line, "moving": ma_line})
		self._rebuild_legend()
		self.canvas_plot.draw_idle()

	def _rebuild_legend(self) -> None:
		legend = self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
		if legend is None:
			return
		handles = legend.legend_handles
		texts = legend.get_texts()
		lines = list(self.ax.get_lines())
		self._line_registry.clear()

		for idx, leg_handle in enumerate(handles):
			if idx >= len(lines):
				break
			line = lines[idx]
			leg_handle.set_picker(True)
			leg_handle.set_pickradius(6)
			texts[idx].set_picker(True)
			self._line_registry[leg_handle] = line
			self._line_registry[texts[idx]] = line

	def _on_pick_legend(self, event) -> None:
		artist = event.artist
		line = self._line_registry.get(artist)
		if line is None:
			return
		visible = not line.get_visible()
		line.set_visible(visible)
		artist.set_alpha(1.0 if visible else 0.25)
		self.canvas_plot.draw_idle()

	def _draw_env_frame(self, force: bool = False, frame: Optional[np.ndarray] = None) -> None:
		if Image is None or ImageTk is None:
			if force:
				self.env_canvas.delete("all")
				self._env_canvas_image_id = None
				self.env_canvas.create_text(
					max(1, self.env_canvas.winfo_width() // 2),
					max(1, self.env_canvas.winfo_height() // 2),
					text="Install pillow for environment image rendering.",
				)
			return

		if frame is not None:
			self._last_frame_np = np.asarray(frame)
		else:
			fresh_frame = self.trainer.environment.render_frame()
			if fresh_frame is not None:
				frame = np.asarray(fresh_frame)
				self._last_frame_np = frame
			elif self._last_frame_np is not None:
				frame = self._last_frame_np

		if frame is None:
			if force:
				self.env_canvas.delete("all")
				self._env_canvas_image_id = None
				self.env_canvas.create_text(
					max(1, self.env_canvas.winfo_width() // 2),
					max(1, self.env_canvas.winfo_height() // 2),
					text="No render frame available",
				)
			return

		width = max(1, self.env_canvas.winfo_width())
		height = max(1, self.env_canvas.winfo_height())
		if width <= 1 or height <= 1:
			self.root.after(20, self._draw_env_frame)
			return

		frame_np = np.asarray(frame)
		src_h, src_w = frame_np.shape[:2]
		scale = min(width / src_w, height / src_h)
		target_w = max(1, int(src_w * scale))
		target_h = max(1, int(src_h * scale))
		if target_w != src_w or target_h != src_h:
			x_idx = np.linspace(0, src_w - 1, target_w).astype(np.int32)
			y_idx = np.linspace(0, src_h - 1, target_h).astype(np.int32)
			frame_np = frame_np[np.ix_(y_idx, x_idx)]
		image = Image.fromarray(frame_np)
		tk_img = ImageTk.PhotoImage(image=image)
		self._env_tk_img = tk_img
		if self._env_canvas_image_id is None:
			self._env_canvas_image_id = self.env_canvas.create_image(width // 2, height // 2, image=tk_img, anchor="center")
		else:
			self.env_canvas.coords(self._env_canvas_image_id, width // 2, height // 2)
			self.env_canvas.itemconfigure(self._env_canvas_image_id, image=tk_img)

	def _schedule_ui_pump(self) -> None:
		if self._is_closing:
			return
		self._ui_pump()
		self.root.after(33, self._schedule_ui_pump)

	def _get_environment_position(self) -> Optional[float]:
		env = self.trainer.environment
		lock = getattr(env, "_env_lock", None)
		if lock is not None:
			with lock:
				state = env.state
		else:
			state = env.state
		if state is None:
			return None
		return float(state[0])

	def _update_status_line(self, epsilon: float) -> None:
		current_x = self._get_environment_position()
		if current_x is not None:
			if self._best_position_current_run is None:
				self._best_position_current_run = current_x
			else:
				self._best_position_current_run = max(self._best_position_current_run, current_x)

		current_text = f"{current_x:.3f}" if current_x is not None else "n/a"
		best_text = f"{self._best_position_current_run:.3f}" if self._best_position_current_run is not None else "n/a"
		self.status_var.set(f"Epsilon: {epsilon:.2f} | Current x: {current_text} | Best x: {best_text}")

	def _ui_pump(self) -> None:
		with self._lock:
			pending = dict(self._pending)

		self.steps_progress.configure(maximum=max(1, int(pending.get("steps_effective_max", pending.get("steps_total", 1)))), value=int(pending.get("current_step", 0)))
		self.episodes_progress.configure(maximum=max(1, int(pending.get("episodes_total", 1))), value=int(pending.get("current_episode", 0)))
		self._update_status_line(float(pending.get("epsilon", 0.0)))

		now = time.time()
		rewards = list(pending.get("reward_snapshot", []))
		label = str(pending.get("policy_label", ""))
		run_id = pending.get("run_id")
		animation_fps = max(1, int(pending.get("animation_fps", 10)))
		frame_interval = 1.0 / animation_fps
		rewards_len = len(rewards)
		if rewards and rewards_len != self._last_plot_rewards_len and now - self._last_plot_update >= 0.15:
			self._last_plot_update = now
			self._last_plot_rewards_len = rewards_len
			self._render_live_training_plot(label, rewards, run_id)

		frame = pending.get("frame")
		if isinstance(frame, np.ndarray) and frame.size and (now - self._last_env_draw_ts >= frame_interval):
			self._draw_env_frame(frame=frame)
			self._last_env_draw_ts = now
		elif now - self._last_env_draw_ts >= frame_interval:
			self._draw_env_frame()
			self._last_env_draw_ts = now

		if pending.get("finalize_run"):
			with self._lock:
				self._pending["finalize_run"] = False

	def _render_live_training_plot(self, label: str, rewards: List[float], run_id: Optional[int]) -> None:
		moving_avg_window = max(1, int(float(self.moving_avg_var.get())))
		is_new_training_run = (run_id is not None and run_id != self._active_training_run_id)
		if not self._run_entries or is_new_training_run:
			color = self._plot_color_cycle[self._plot_idx % len(self._plot_color_cycle)]
			self._plot_idx += 1
			x = np.arange(1, len(rewards) + 1)
			reward_line, = self.ax.plot(x, rewards, color=color, alpha=0.35, linewidth=1.2, label=f"{label} | reward")
			ma = self._moving_average(rewards, moving_avg_window)
			ma_line, = self.ax.plot(x, ma, color=color, alpha=0.95, linewidth=2.4, label=f"{label} | MA")
			self._run_entries.append({"reward": reward_line, "moving": ma_line, "label": label, "run_id": run_id})
			self._active_training_run_id = run_id
		else:
			current = self._run_entries[-1]
			reward_line = current["reward"]
			moving_line = current["moving"]
			x = np.arange(1, len(rewards) + 1)
			reward_line.set_data(x, rewards)
			moving_line.set_data(x, self._moving_average(rewards, moving_avg_window))
			self.ax.relim()
			self.ax.autoscale_view()

		self._rebuild_legend()
		self.canvas_plot.draw_idle()

	def _on_close(self) -> None:
		self._is_closing = True
		self._stop_requested = True
		self._pause_event.set()
		self._unbind_mousewheel_for_params()
		try:
			self.trainer.environment.close()
		except Exception:
			pass
		self.root.destroy()


def launch_gui() -> None:
	root = tk.Tk()
	MountainCarGUI(root)
	root.mainloop()
