from __future__ import annotations
"""Tkinter GUI for Taxi RL training, visualization, and artifact export."""

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from Taxi_logic import TaxiEnvironment, Trainer, build_agent

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class TaxiGUI:
    """Main GUI controller handling layout, interaction, and threaded training flow."""
    def __init__(self, root: tk.Tk, environment: TaxiEnvironment, trainer: Trainer) -> None:
        self.root = root
        self.environment = environment
        self.trainer = trainer

        self.root.title("Taxi RL")
        self.root.geometry("1380x920")

        self.current_episode = 0
        self.current_step = 0
        self._stop_requested = False
        self._train_thread: Optional[threading.Thread] = None
        self._last_plot_update = 0.0
        self._current_render_image = None
        self._is_shutting_down = False

        self.agent_instance = None
        self.plot_runs: List[Dict] = []
        self._active_run: Optional[Dict] = None

        self._build_variables()
        self._build_layout()
        self._render_environment()
        self._set_current_counters(0, 0, training=False)

    def _safe_after(self, callback) -> None:
        """Schedule callback on Tk main thread if the window is still alive."""
        if self._is_shutting_down:
            return
        try:
            if int(self.root.winfo_exists()) != 1:
                return
            # Marshal background thread updates onto Tk's UI thread.
            self.root.after(0, callback)
        except (tk.TclError, RuntimeError):
            return

    def _safe_showinfo(self, title: str, message: str) -> None:
        """Show message dialog only when GUI is active, avoiding shutdown races."""
        if self._is_shutting_down:
            return
        try:
            if int(self.root.winfo_exists()) != 1:
                return
            messagebox.showinfo(title, message)
        except (tk.TclError, RuntimeError):
            return

    def _build_variables(self) -> None:
        """Initialize all Tk variables for environment, training, and model parameters."""
        self.is_raining_var = tk.BooleanVar(value=False)
        self.fickle_passenger_var = tk.BooleanVar(value=False)

        self.policy_var = tk.StringVar(value="DQN")
        self.episodes_var = tk.IntVar(value=200)
        self.max_steps_var = tk.IntVar(value=200)
        self.live_plot_var = tk.BooleanVar(value=True)
        self.reduced_speed_var = tk.BooleanVar(value=True)
        self.ma_n_var = tk.IntVar(value=20)
        self.render_every_n_var = tk.IntVar(value=10)

        self.lr_var = tk.DoubleVar(value=1e-3)
        self.gamma_var = tk.DoubleVar(value=0.99)
        self.batch_size_var = tk.IntVar(value=64)
        self.buffer_size_var = tk.IntVar(value=50_000)
        self.target_update_var = tk.IntVar(value=250)
        self.hidden_size_var = tk.IntVar(value=128)
        self.eps_start_var = tk.DoubleVar(value=0.30)
        self.eps_min_var = tk.DoubleVar(value=0.05)
        self.eps_decay_var = tk.DoubleVar(value=0.995)
        self.prio_alpha_var = tk.DoubleVar(value=0.60)

    def _build_layout(self) -> None:
        """Create panel grid layout for environment, controls, parameters, and plot."""
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        main.columnconfigure(0, weight=1)
        main.columnconfigure(1, weight=0)
        main.columnconfigure(2, weight=0)
        main.rowconfigure(3, weight=1)

        self.environment_frame = ttk.LabelFrame(main, text="Environment", padding=8)
        self.environment_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", pady=(0, 8))
        self.environment_frame.columnconfigure(0, weight=1)

        self.controls_frame = ttk.LabelFrame(main, text="Controls", padding=8)
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(0, 8))

        self.current_state_frame = ttk.LabelFrame(main, text="Current State", padding=8)
        self.current_state_frame.grid(row=2, column=0, sticky="ew", padx=(0, 8), pady=(0, 8))

        self.dnn_params_frame = ttk.LabelFrame(main, text="DNN Parameters", padding=8)
        self.dnn_params_frame.grid(row=1, column=1, rowspan=2, sticky="ns", padx=(0, 8), pady=(0, 8))

        self.train_params_frame = ttk.LabelFrame(main, text="Training Parameters", padding=8)
        self.train_params_frame.grid(row=1, column=2, rowspan=2, sticky="ns", pady=(0, 8))

        self.live_plot_frame = ttk.LabelFrame(main, text="Live Plot", padding=8)
        self.live_plot_frame.grid(row=3, column=0, columnspan=3, sticky="nsew")

        self._build_environment_panel()
        self._build_controls_panel()
        self._build_current_state_panel()
        self._build_dnn_params_panel()
        self._build_training_params_panel()
        self._build_plot_panel()

    def _build_environment_panel(self) -> None:
        toggle_bar = ttk.Frame(self.environment_frame)
        toggle_bar.grid(row=0, column=0, sticky="w", pady=(0, 6))

        ttk.Checkbutton(
            toggle_bar,
            text="raining",
            variable=self.is_raining_var,
            command=self._apply_env_modes,
        ).grid(row=0, column=0, padx=(0, 12))

        ttk.Checkbutton(
            toggle_bar,
            text="running man",
            variable=self.fickle_passenger_var,
            command=self._apply_env_modes,
        ).grid(row=0, column=1)

        self.env_canvas_label = ttk.Label(self.environment_frame, anchor="center")
        self.env_canvas_label.grid(row=1, column=0, sticky="nsew")

    def _build_controls_panel(self) -> None:
        for c in range(2):
            self.controls_frame.columnconfigure(c, weight=1)

        buttons = [
            ("Reset All", self._on_reset_all),
            ("Clear Plot", self._on_clear_plot),
            ("Run single episode", self._on_run_single_episode),
            ("Save samplings CSV", self._on_save_samplings_csv),
            ("Train and Run", self._on_train_and_run),
            ("Save Plot PNG", self._on_save_plot_png),
        ]

        for idx, (label, callback) in enumerate(buttons):
            r = idx // 2
            c = idx % 2
            ttk.Button(self.controls_frame, text=label, command=callback).grid(
                row=r, column=c, sticky="ew", padx=4, pady=4
            )

    def _build_current_state_panel(self) -> None:
        self.current_status_var = tk.StringVar(value="")
        ttk.Label(self.current_state_frame, textvariable=self.current_status_var).grid(row=0, column=0, sticky="w")

    def _build_training_params_panel(self) -> None:
        fields = [
            ("episodes", self.episodes_var),
            ("max_steps", self.max_steps_var),
            ("MA N values", self.ma_n_var),
            ("Render every N steps", self.render_every_n_var),
        ]

        row = 0
        ttk.Label(self.train_params_frame, text="policy").grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
        policy_cb = ttk.Combobox(
            self.train_params_frame,
            textvariable=self.policy_var,
            values=["DQN", "DoubleDQN", "DuelingDQN", "PrioDQN"],
            state="readonly",
            width=14,
        )
        policy_cb.grid(row=row, column=1, sticky="w", pady=3)
        row += 1

        for label, var in fields:
            ttk.Label(self.train_params_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
            ttk.Entry(self.train_params_frame, textvariable=var, width=16).grid(row=row, column=1, sticky="w", pady=3)
            row += 1

        ttk.Checkbutton(self.train_params_frame, text="Live plot", variable=self.live_plot_var).grid(
            row=row, column=0, sticky="w", pady=3
        )
        ttk.Checkbutton(self.train_params_frame, text="reduced speed", variable=self.reduced_speed_var).grid(
            row=row, column=1, sticky="w", pady=3
        )

    def _build_dnn_params_panel(self) -> None:
        params = [
            ("learning rate", self.lr_var),
            ("gamma", self.gamma_var),
            ("batch size", self.batch_size_var),
            ("buffer size", self.buffer_size_var),
            ("target update", self.target_update_var),
            ("hidden size", self.hidden_size_var),
            ("eps start", self.eps_start_var),
            ("eps min", self.eps_min_var),
            ("eps decay", self.eps_decay_var),
            ("prio alpha", self.prio_alpha_var),
        ]

        for row, (label, var) in enumerate(params):
            ttk.Label(self.dnn_params_frame, text=label).grid(row=row, column=0, sticky="w", padx=(0, 6), pady=3)
            ttk.Entry(self.dnn_params_frame, textvariable=var, width=12).grid(row=row, column=1, sticky="w", pady=3)

    def _build_plot_panel(self) -> None:
        self.live_plot_frame.columnconfigure(0, weight=1)
        self.live_plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.live_plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.legend_toggle_frame = ttk.Frame(self.live_plot_frame)
        self.legend_toggle_frame.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _apply_env_modes(self) -> None:
        self.environment.set_modes(
            is_raining=self.is_raining_var.get(),
            fickle_passenger=self.fickle_passenger_var.get(),
        )

    def _set_current_counters(self, episode: int, step: int, training: bool = True) -> None:
        """Update status label with padded Training/Idle state, step, and episode counters."""
        self.current_episode = episode
        self.current_step = step
        status = "Training" if training else "Idle"
        padded_status = f"{status:>8}"
        self.current_status_var.set(f"{padded_status}: step:{step:5d}  episode:{episode:5d}")

    def _render_environment(self) -> None:
        """Render current Taxi frame into the Environment panel."""
        frame = self.environment.render_rgb()
        if frame is None:
            self.env_canvas_label.configure(text="Environment frame not available.")
            return

        if not PIL_AVAILABLE:
            self.env_canvas_label.configure(text="Pillow is required to render Taxi frames.")
            return

        image = Image.fromarray(frame)
        image = image.resize((640, 360), Image.Resampling.NEAREST)
        tk_image = ImageTk.PhotoImage(image)
        self._current_render_image = tk_image
        self.env_canvas_label.configure(image=tk_image, text="")

    def _rolling_average(self, values: List[float], window: int) -> np.ndarray:
        if not values:
            return np.array([])
        w = max(1, int(window))
        arr = np.asarray(values, dtype=np.float32)
        if len(arr) < w:
            return np.full_like(arr, arr.mean())
        kernel = np.ones(w, dtype=np.float32) / float(w)
        ma = np.convolve(arr, kernel, mode="valid")
        prefix = np.full((w - 1,), ma[0], dtype=np.float32)
        return np.concatenate([prefix, ma])

    def _new_run_name(self, policy_name: str) -> str:
        timestamp = datetime.now().strftime("%H:%M:%S")
        return f"{policy_name} [{timestamp}]"

    def _update_plot(self, current_rewards: Optional[List[float]] = None, run_name: Optional[str] = None) -> None:
        """Refresh reward + moving-average lines with throttled redraw cadence."""
        now = time.time()
        # Throttle redraw to keep UI responsive during fast training loops.
        if now - self._last_plot_update < 0.15:
            return
        self._last_plot_update = now

        if current_rewards is not None and self._active_run is not None:
            self._active_run["rewards"] = current_rewards
            run_name = run_name or self._active_run["name"]

        self.ax.clear()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        ma_window = max(1, int(self.ma_n_var.get()))

        for run in self.plot_runs:
            rewards = run["rewards"]
            if not rewards:
                continue

            x = np.arange(1, len(rewards) + 1)
            reward_line, = self.ax.plot(x, rewards, linewidth=1.5, label=f"{run['name']} reward")
            ma = self._rolling_average(rewards, ma_window)
            ma_line, = self.ax.plot(x, ma, linewidth=3.0, label=f"{run['name']} ma({ma_window})")
            run["reward_line"] = reward_line
            run["ma_line"] = ma_line

            visible = bool(run["visible_var"].get())
            reward_line.set_visible(visible)
            ma_line.set_visible(visible)

        self.ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        self.figure.tight_layout(rect=(0, 0, 0.82, 1))
        self.canvas.draw_idle()

    def _refresh_plot_toggles(self) -> None:
        for child in self.legend_toggle_frame.winfo_children():
            child.destroy()

        for idx, run in enumerate(self.plot_runs):
            cb = ttk.Checkbutton(
                self.legend_toggle_frame,
                text=run["name"],
                variable=run["visible_var"],
                command=self._on_toggle_visibility,
            )
            cb.grid(row=0, column=idx, padx=(0, 8), sticky="w")

    def _on_toggle_visibility(self) -> None:
        for run in self.plot_runs:
            visible = bool(run["visible_var"].get())
            reward_line = run.get("reward_line")
            ma_line = run.get("ma_line")
            if reward_line is not None:
                reward_line.set_visible(visible)
            if ma_line is not None:
                ma_line.set_visible(visible)
        self.canvas.draw_idle()

    def _collect_agent_params(self) -> Dict:
        return {
            "state_size": self.environment.n_states,
            "action_size": self.environment.n_actions,
            "learning_rate": float(self.lr_var.get()),
            "gamma": float(self.gamma_var.get()),
            "batch_size": int(self.batch_size_var.get()),
            "buffer_size": int(self.buffer_size_var.get()),
            "target_update_freq": int(self.target_update_var.get()),
            "hidden_size": int(self.hidden_size_var.get()),
        }

    def _create_agent(self) -> None:
        """Instantiate selected policy using current DNN parameter values."""
        policy = self.policy_var.get()
        params = self._collect_agent_params()
        if policy == "PrioDQN":
            params["priority_alpha"] = float(self.prio_alpha_var.get())

        self.agent_instance = build_agent(policy, **params)

    def _start_new_run(self) -> None:
        run_name = self._new_run_name(self.policy_var.get())
        visible_var = tk.BooleanVar(value=True)
        run = {
            "name": run_name,
            "rewards": [],
            "visible_var": visible_var,
            "reward_line": None,
            "ma_line": None,
        }
        self.plot_runs.append(run)
        self._active_run = run
        self._refresh_plot_toggles()

    def _on_run_single_episode(self) -> None:
        """Run one episode on a worker thread and animate progress in the GUI."""
        if self._train_thread and self._train_thread.is_alive():
            return

        self._stop_requested = False
        self._create_agent()
        self._apply_env_modes()

        self._start_new_run()
        self._set_current_counters(1, 0, training=True)

        epsilon = float(self.eps_start_var.get())
        max_steps = int(self.max_steps_var.get())

        def worker() -> None:
            def progress(step: int) -> None:
                self._safe_after(lambda s=step: self._set_current_counters(1, s, training=True))
                self._safe_after(self._render_environment)

            result = self.trainer.run_episode(
                policy=self.agent_instance,
                epsilon=epsilon,
                max_steps=max_steps,
                progress_callback=progress,
            )
            rewards = [float(result["total_reward"])]
            self._active_run["rewards"] = rewards
            self._safe_after(lambda: self._update_plot(current_rewards=rewards))
            self._safe_after(lambda: self._set_current_counters(1, int(result["steps"]), training=False))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _on_train_and_run(self) -> None:
        """Run multi-episode training in background thread with live progress updates."""
        if self._train_thread and self._train_thread.is_alive():
            return

        self._stop_requested = False
        self._create_agent()
        self._apply_env_modes()

        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        eps = float(self.eps_start_var.get())
        eps_min = float(self.eps_min_var.get())
        eps_decay = float(self.eps_decay_var.get())

        self._start_new_run()

        def worker() -> None:
            rewards: List[float] = []
            epsilon = eps

            for ep in range(1, episodes + 1):
                if self._stop_requested:
                    break

                self._safe_after(lambda e=ep: self._set_current_counters(e, 0, training=True))

                def progress(step: int, e: int = ep) -> None:
                    self._safe_after(lambda ee=e, ss=step: self._set_current_counters(ee, ss, training=True))
                    render_every = max(1, int(self.render_every_n_var.get()))
                    # Refresh the environment view every N steps (user configurable).
                    if step % render_every == 0:
                        self._safe_after(self._render_environment)

                result = self.trainer.run_episode(
                    policy=self.agent_instance,
                    epsilon=epsilon,
                    max_steps=max_steps,
                    progress_callback=progress,
                )
                rewards.append(float(result["total_reward"]))

                if self.live_plot_var.get():
                    rewards_copy = rewards.copy()
                    self._safe_after(lambda rc=rewards_copy: self._update_plot(current_rewards=rc))

                self._safe_after(self._render_environment)
                epsilon = max(eps_min, epsilon * eps_decay)

                if self.reduced_speed_var.get():
                    time.sleep(0.033)

            self._safe_after(lambda: self._set_current_counters(self.current_episode, self.current_step, training=False))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _on_save_samplings_csv(self) -> None:
        """Generate transition samplings CSV via trainer in a background thread."""
        if self._train_thread and self._train_thread.is_alive():
            return

        self._create_agent()
        self._apply_env_modes()

        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        epsilon = float(self.eps_start_var.get())
        policy_name = self.policy_var.get()

        def worker() -> None:
            base = f"samplings_{policy_name}"
            rewards = self.trainer.train(
                policy=self.agent_instance,
                num_episodes=episodes,
                max_steps=max_steps,
                epsilon=epsilon,
                save_csv=base,
            )
            self._safe_after(lambda: self._safe_showinfo("CSV saved", f"Saved {len(rewards)} episodes to results_csv."))

        self._train_thread = threading.Thread(target=worker, daemon=True)
        self._train_thread.start()

    def _on_save_plot_png(self) -> None:
        """Save current embedded matplotlib figure into the plots directory."""
        out_dir = Path("plots")
        out_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        filename = (
            f"plot_{self.policy_var.get()}"
            f"_epsmin-{self.eps_min_var.get()}"
            f"_epsmax-{self.eps_start_var.get()}"
            f"_alpha-{self.lr_var.get()}"
            f"_gamma-{self.gamma_var.get()}"
            f"_episodes-{self.episodes_var.get()}"
            f"_maxsteps-{self.max_steps_var.get()}"
            f"_{timestamp}.png"
        )
        path = out_dir / filename.replace(" ", "")
        self.figure.savefig(path)
        messagebox.showinfo("Plot saved", f"Saved plot to {path}")

    def _on_clear_plot(self) -> None:
        self.plot_runs.clear()
        self._active_run = None
        self._refresh_plot_toggles()
        self.ax.clear()
        self.ax.set_title("Episode Rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas.draw_idle()

    def _on_reset_all(self) -> None:
        self._stop_requested = True
        self._on_clear_plot()
        self.agent_instance = None
        self.environment.reset()
        self._render_environment()
        self._set_current_counters(0, 0, training=False)

    def shutdown(self) -> None:
        """Request stop, join worker thread briefly, and close environment resources."""
        self._is_shutting_down = True
        self._stop_requested = True
        if self._train_thread and self._train_thread.is_alive():
            self._train_thread.join(timeout=1.5)
        self.environment.close()
