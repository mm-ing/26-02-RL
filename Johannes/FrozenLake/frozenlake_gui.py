from __future__ import annotations

import threading
import time
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, Tk, messagebox, ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from .frozenlake_logic import (
        DQN,
        DoubleDQN,
        DuelingDQN,
        FrozenLakeEnv,
        PrioDQN,
        Trainer,
        ensure_output_dirs,
    )
except Exception:
    from frozenlake_logic import (  # type: ignore
        DQN,
        DoubleDQN,
        DuelingDQN,
        FrozenLakeEnv,
        PrioDQN,
        Trainer,
        ensure_output_dirs,
    )


class FrozenLakeGUI:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("FrozenLake RL - DQN variants")

        self.base_dir = Path(__file__).resolve().parent
        self.results_dir, self.plots_dir = ensure_output_dirs(self.base_dir)

        self.env = FrozenLakeEnv(is_slippery=True, map_name="4x4", success_rate=1.0 / 3.0, render_mode="rgb_array")
        self.trainer = Trainer(self.env, base_dir=self.base_dir)

        self.current_episode = 0
        self.current_step = 0
        self._stop_requested = False
        self._training_thread: threading.Thread | None = None
        self._last_plot_update = 0.0

        self.plot_runs: list[dict] = []
        self._legend_pick_map: dict[str, int] = {}
        self._frame_photo: ImageTk.PhotoImage | None = None

        self._init_vars()
        self._build_layout()
        self._set_current_counters(episode=0, step=0, training=False)
        self._update_env_frame()

    def _init_vars(self) -> None:
        self.max_steps_var = IntVar(value=100)
        self.episodes_var = IntVar(value=300)
        self.policy_var = StringVar(value="DQN")
        self.live_plot_var = BooleanVar(value=True)
        self.reduced_speed_var = BooleanVar(value=True)

        self.lr_var = DoubleVar(value=1e-3)
        self.gamma_var = DoubleVar(value=0.99)
        self.eps_start_var = DoubleVar(value=1.0)
        self.eps_end_var = DoubleVar(value=0.05)
        self.eps_decay_var = DoubleVar(value=0.995)

        self.slippery_var = BooleanVar(value=True)
        self.slippery_rate_var = DoubleVar(value=1.0 / 3.0)
        self.map_size_var = StringVar(value="4x4")

        self.ma_n_values_var = IntVar(value=20)

        self.replay_size_var = IntVar(value=20_000)
        self.batch_size_var = IntVar(value=128)
        self.hidden_size_var = IntVar(value=128)
        self.target_update_var = IntVar(value=100)
        self.warmup_steps_var = IntVar(value=250)
        self.prio_alpha_var = DoubleVar(value=0.6)
        self.prio_beta_var = DoubleVar(value=0.4)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(0, weight=1)
        self.root.rowconfigure(3, weight=1)

        self.env_frame = ttk.LabelFrame(self.root, text="Environment")
        self.env_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        self.env_frame.columnconfigure(1, weight=1)

        env_controls = ttk.Frame(self.env_frame)
        env_controls.grid(row=0, column=0, sticky="nw", padx=6, pady=4)

        ttk.Checkbutton(env_controls, text="slippery", variable=self.slippery_var, command=self._on_env_config_changed).grid(
            row=0, column=0, sticky="w", padx=2, pady=2
        )
        ttk.Label(env_controls, text="slippery rate").grid(row=1, column=0, sticky="w", padx=2, pady=2)
        ttk.Entry(env_controls, textvariable=self.slippery_rate_var, width=10).grid(row=1, column=1, sticky="w", padx=2, pady=2)
        ttk.Label(env_controls, text="map size").grid(row=2, column=0, sticky="w", padx=2, pady=2)
        ttk.Combobox(env_controls, textvariable=self.map_size_var, values=["4x4", "8x8"], state="readonly", width=8).grid(
            row=2, column=1, sticky="w", padx=2, pady=2
        )
        ttk.Button(env_controls, text="Apply env", command=self._on_env_config_changed).grid(row=3, column=0, columnspan=2, sticky="ew", padx=2, pady=4)

        self.env_image_label = ttk.Label(self.env_frame)
        self.env_image_label.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)

        self.controls_frame = ttk.LabelFrame(self.root, text="Controls")
        self.controls_frame.grid(row=1, column=0, sticky="new", padx=6, pady=6)
        for col in range(2):
            self.controls_frame.columnconfigure(col, weight=1)

        ttk.Button(self.controls_frame, text="Reset All", command=self.reset_all).grid(row=0, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(self.controls_frame, text="Clear Plot", command=self.clear_plot).grid(row=0, column=1, sticky="ew", padx=3, pady=3)
        ttk.Button(self.controls_frame, text="Run single episode", command=self.run_single_episode).grid(
            row=1, column=0, sticky="ew", padx=3, pady=3
        )
        ttk.Button(self.controls_frame, text="Save samplings CSV", command=self.save_samplings_csv).grid(
            row=1, column=1, sticky="ew", padx=3, pady=3
        )
        ttk.Button(self.controls_frame, text="Train and Run", command=self.train_and_run).grid(row=2, column=0, sticky="ew", padx=3, pady=3)
        ttk.Button(self.controls_frame, text="Save Plot PNG", command=self.save_plot_png).grid(row=2, column=1, sticky="ew", padx=3, pady=3)

        self.current_state_frame = ttk.LabelFrame(self.root, text="Current State")
        self.current_state_frame.grid(row=2, column=0, sticky="new", padx=6, pady=6)
        self.current_state_label = ttk.Label(self.current_state_frame, text="")
        self.current_state_label.grid(row=0, column=0, sticky="w", padx=6, pady=6)

        self.dnn_frame = ttk.LabelFrame(self.root, text="DNN Parameters")
        self.dnn_frame.grid(row=1, column=1, rowspan=2, sticky="ns", padx=6, pady=6)
        self._build_dnn_param_inputs(self.dnn_frame)

        self.training_frame = ttk.LabelFrame(self.root, text="Training Parameters")
        self.training_frame.grid(row=1, column=2, rowspan=2, sticky="ns", padx=6, pady=6)
        self._build_training_param_inputs(self.training_frame)

        self.plot_frame = ttk.LabelFrame(self.root, text="Live Plot")
        self.plot_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)

        self.figure = Figure(figsize=(9, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.grid(row=0, column=0, sticky="nsew")
        self.canvas.mpl_connect("pick_event", self._on_legend_pick)

    def _build_training_param_inputs(self, parent: ttk.LabelFrame) -> None:
        entries = [
            ("Max steps", self.max_steps_var),
            ("Episodes", self.episodes_var),
            ("Learning rate", self.lr_var),
            ("Gamma", self.gamma_var),
            ("Epsilon start", self.eps_start_var),
            ("Epsilon end", self.eps_end_var),
            ("Epsilon decay", self.eps_decay_var),
            ("MA N values", self.ma_n_values_var),
        ]
        for row, (label, var) in enumerate(entries):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", padx=5, pady=2)

        row = len(entries)
        ttk.Label(parent, text="Policy").grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Combobox(
            parent,
            textvariable=self.policy_var,
            values=["DQN", "DoubleDQN", "DuelingDQN", "PrioDQN"],
            state="readonly",
            width=10,
        ).grid(row=row, column=1, sticky="ew", padx=5, pady=2)
        row += 1

        ttk.Checkbutton(parent, text="Live plot", variable=self.live_plot_var).grid(row=row, column=0, sticky="w", padx=5, pady=2)
        ttk.Checkbutton(parent, text="reduced speed", variable=self.reduced_speed_var).grid(row=row, column=1, sticky="w", padx=5, pady=2)

    def _build_dnn_param_inputs(self, parent: ttk.LabelFrame) -> None:
        entries = [
            ("Replay buffer", self.replay_size_var),
            ("Batch size", self.batch_size_var),
            ("Hidden size", self.hidden_size_var),
            ("Target update", self.target_update_var),
            ("Warmup steps", self.warmup_steps_var),
            ("Prio alpha", self.prio_alpha_var),
            ("Prio beta", self.prio_beta_var),
        ]
        for row, (label, var) in enumerate(entries):
            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=5, pady=2)
            ttk.Entry(parent, textvariable=var, width=12).grid(row=row, column=1, sticky="ew", padx=5, pady=2)

    def _set_current_counters(self, episode: int, step: int, training: bool = False) -> None:
        status = "Training" if training else "Idle"
        status_padded = status.rjust(len("Training"))
        self.current_episode = int(episode)
        self.current_step = int(step)
        text = f"{status_padded}: step: {self.current_step:4d}  episode: {self.current_episode:4d}"
        self.current_state_label.configure(text=text)

    def _on_env_config_changed(self) -> None:
        self.env.update_config(
            is_slippery=self.slippery_var.get(),
            map_name=self.map_size_var.get(),
            success_rate=self.slippery_rate_var.get(),
        )
        self.trainer = Trainer(self.env, base_dir=self.base_dir)
        self._update_env_frame()

    def _policy_from_inputs(self):
        kwargs = {
            "n_states": self.env.n_states,
            "n_actions": self.env.n_actions,
            "learning_rate": self.lr_var.get(),
            "gamma": self.gamma_var.get(),
            "epsilon_start": self.eps_start_var.get(),
            "epsilon_end": self.eps_end_var.get(),
            "epsilon_decay": self.eps_decay_var.get(),
            "replay_buffer_size": self.replay_size_var.get(),
            "batch_size": self.batch_size_var.get(),
            "target_update_frequency": self.target_update_var.get(),
            "hidden_size": self.hidden_size_var.get(),
            "warmup_steps": self.warmup_steps_var.get(),
        }
        policy_name = self.policy_var.get()
        if policy_name == "DoubleDQN":
            return DoubleDQN(**kwargs)
        if policy_name == "DuelingDQN":
            return DuelingDQN(**kwargs)
        if policy_name == "PrioDQN":
            return PrioDQN(
                **kwargs,
                prio_alpha=self.prio_alpha_var.get(),
                prio_beta=self.prio_beta_var.get(),
            )
        return DQN(**kwargs)

    def _update_env_frame(self) -> None:
        frame = self.env.render_frame()
        if frame is None:
            return
        arr = np.asarray(frame, dtype=np.uint8)
        if PIL_AVAILABLE:
            image = Image.fromarray(arr).resize((700, 260))
            self._frame_photo = ImageTk.PhotoImage(image=image)
            self.env_image_label.configure(image=self._frame_photo)
        else:
            self.env_image_label.configure(text="Pillow not available for preview")

    def _moving_average(self, rewards: list[float], window: int) -> np.ndarray:
        values = np.asarray(rewards, dtype=float)
        if values.size == 0:
            return np.array([])

        effective_window = max(1, min(int(window), values.size))
        if effective_window == 1:
            return values.copy()

        cumsum = np.cumsum(np.insert(values, 0, 0.0))
        moving_valid = (cumsum[effective_window:] - cumsum[:-effective_window]) / effective_window
        prefix = np.array([values[:i].mean() for i in range(1, effective_window)], dtype=float)
        return np.concatenate([prefix, moving_valid])

    def _redraw_all_plots(self) -> None:
        self.ax.clear()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        handles = []
        labels = []
        self._legend_pick_map = {}

        ma_window = max(1, int(self.ma_n_values_var.get()))

        for idx, run in enumerate(self.plot_runs):
            if not run.get("visible", True):
                continue

            rewards = run["rewards"]
            x_axis = np.arange(1, len(rewards) + 1)
            base_width = 1.25
            ma_width = base_width * 2.0

            (line_rewards,) = self.ax.plot(
                x_axis,
                rewards,
                linewidth=base_width,
                alpha=0.75,
                label=run["label"],
                zorder=2,
            )
            ma = self._moving_average(rewards, window=ma_window)
            (line_ma,) = self.ax.plot(
                x_axis,
                ma,
                linewidth=ma_width,
                linestyle="--",
                alpha=0.95,
                color=line_rewards.get_color(),
                label=f"{run['label']} (MA)",
                zorder=3,
            )
            handles.extend([line_rewards, line_ma])
            labels.extend([run["label"], f"{run['label']} (MA)"])
            self._legend_pick_map[line_rewards.get_label()] = idx
            self._legend_pick_map[line_ma.get_label()] = idx

        if handles:
            legend = self.ax.legend(handles=handles, labels=labels, loc="best")
            for line in legend.get_lines():
                line.set_picker(True)
        self.canvas.draw_idle()

    def _on_legend_pick(self, event) -> None:
        label = event.artist.get_label()
        idx = self._legend_pick_map.get(label)
        if idx is None:
            return
        self.plot_runs[idx]["visible"] = not self.plot_runs[idx].get("visible", True)
        self._redraw_all_plots()

    def _throttled_plot_update(self, current_rewards: list[float], run_label: str) -> None:
        now = time.time()
        if now - self._last_plot_update < 0.15:
            return
        self._last_plot_update = now

        if self.plot_runs and self.plot_runs[-1].get("is_current", False):
            self.plot_runs[-1]["rewards"] = current_rewards
        else:
            self.plot_runs.append({"label": run_label, "rewards": current_rewards, "visible": True, "is_current": True})
        self._redraw_all_plots()

    def run_single_episode(self) -> None:
        if self._training_thread and self._training_thread.is_alive():
            return

        self._on_env_config_changed()
        policy = self._policy_from_inputs()
        self._stop_requested = False
        self._set_current_counters(episode=1, step=0, training=True)

        def progress(step: int) -> None:
            self.root.after(0, lambda s=step: self._set_current_counters(1, s, training=True))

        def transition_cb(_transition, _step):
            self.root.after(0, self._update_env_frame)

        result = self.trainer.run_episode(
            policy=policy,
            epsilon=self.eps_start_var.get(),
            max_steps=self.max_steps_var.get(),
            progress_callback=progress,
            transition_callback=transition_cb,
        )
        self.plot_runs.append({"label": f"Single-{self.policy_var.get()}", "rewards": [result["total_reward"]], "visible": True})
        self._redraw_all_plots()
        self._set_current_counters(episode=1, step=result["steps"], training=False)

    def train_and_run(self) -> None:
        if self._training_thread and self._training_thread.is_alive():
            return

        self._on_env_config_changed()
        policy = self._policy_from_inputs()
        self._stop_requested = False
        episodes = max(1, self.episodes_var.get())
        max_steps = max(1, self.max_steps_var.get())
        eps_start = self.eps_start_var.get()
        policy_label = self.policy_var.get()
        live_plot_enabled = bool(self.live_plot_var.get())
        reduced_speed_enabled = bool(self.reduced_speed_var.get())
        policy.epsilon = float(eps_start)

        def worker() -> None:
            rewards: list[float] = []
            run_label = f"{policy_label}-{int(time.time())}"

            for episode in range(1, episodes + 1):
                if self._stop_requested:
                    break

                self.root.after(0, lambda ep=episode: self._set_current_counters(ep, 0, training=True))

                def progress(step: int, ep=episode):
                    self.root.after(0, lambda: self._set_current_counters(ep, step, training=True))

                def transition_cb(_transition, _step):
                    self.root.after(0, self._update_env_frame)

                result = self.trainer.run_episode(
                    policy=policy,
                    epsilon=policy.epsilon,
                    max_steps=max_steps,
                    progress_callback=progress,
                    transition_callback=transition_cb,
                )

                rewards.append(result["total_reward"])

                if live_plot_enabled:
                    self.root.after(0, lambda data=rewards.copy(), label=run_label: self._throttled_plot_update(data, label))

                if reduced_speed_enabled:
                    time.sleep(0.033)

            self.root.after(0, self._finish_training)

        self._training_thread = threading.Thread(target=worker, daemon=True)
        self._training_thread.start()

    def _finish_training(self) -> None:
        if self.plot_runs and self.plot_runs[-1].get("is_current"):
            self.plot_runs[-1]["is_current"] = False
        self._set_current_counters(self.current_episode, self.current_step, training=False)

    def clear_plot(self) -> None:
        self.plot_runs.clear()
        self._legend_pick_map.clear()
        self._redraw_all_plots()

    def reset_all(self) -> None:
        self._stop_requested = True
        self.clear_plot()
        self._on_env_config_changed()
        self._set_current_counters(episode=0, step=0, training=False)

    def save_samplings_csv(self) -> None:
        self._on_env_config_changed()
        policy = self._policy_from_inputs()
        ts = int(time.time())
        base = f"frozenlake_samplings_{self.policy_var.get()}_{ts}"
        _, csv_path = self.trainer.train(
            policy=policy,
            num_episodes=max(1, self.episodes_var.get()),
            max_steps=max(1, self.max_steps_var.get()),
            epsilon=self.eps_start_var.get(),
            save_csv=base,
        )
        if csv_path is not None:
            messagebox.showinfo("CSV saved", f"Saved samplings:\n{csv_path}")

    def save_plot_png(self) -> None:
        ts = int(time.time())
        filename = (
            f"policy-{self.policy_var.get()}"
            f"_epsMin-{self.eps_end_var.get():.3f}"
            f"_epsMax-{self.eps_start_var.get():.3f}"
            f"_alpha-{self.lr_var.get():.4f}"
            f"_gamma-{self.gamma_var.get():.3f}"
            f"_episodes-{self.episodes_var.get()}"
            f"_maxSteps-{self.max_steps_var.get()}"
            f"_{ts}.png"
        )
        path = self.plots_dir / filename
        self.figure.savefig(path, dpi=150, bbox_inches="tight")
        messagebox.showinfo("Plot saved", f"Saved plot:\n{path}")

    def on_close(self) -> None:
        self._stop_requested = True
        self.env.close()
        self.root.destroy()
