from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import tkinter as tk
from tkinter import ttk

import numpy as np
from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from cliffwalking_logic import CliffWalkingEnv, DDQNetwork, DQNetwork, Trainer


class CliffWalkingGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CliffWalking RL")
        self.geometry("1250x880")

        self.base_dir = Path(__file__).resolve().parent
        self.plots_dir = self.base_dir / "plots"
        self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.env = CliffWalkingEnv(slippery=False)
        self.trainer = Trainer(self.env, output_dir=self.base_dir)

        self.current_policy_obj = None
        self.training_thread: Optional[threading.Thread] = None
        self._stop_requested = False
        self._last_plot_update = 0.0
        self.plot_runs: List[Dict] = []
        self._legend_map: Dict = {}
        self._env_image_tk = None

        self.current_step = 0
        self.current_episode = 0

        self._build_vars()
        self._build_layout()
        self.env.reset()
        self._refresh_environment_view()
        self._set_current_counters(0, 0, training=False)
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

    def _build_vars(self) -> None:
        self.max_steps_var = tk.IntVar(value=200)
        self.episodes_var = tk.IntVar(value=300)
        self.policy_var = tk.StringVar(value="Vanilla DQN")
        self.epsilon_start_var = tk.DoubleVar(value=1.0)
        self.epsilon_end_var = tk.DoubleVar(value=0.05)
        self.epsilon_decay_var = tk.DoubleVar(value=0.995)
        self.learning_rate_var = tk.DoubleVar(value=1e-3)
        self.gamma_var = tk.DoubleVar(value=0.99)
        self.live_plot_var = tk.BooleanVar(value=True)
        self.reduced_speed_var = tk.BooleanVar(value=True)

        self.replay_buffer_var = tk.IntVar(value=5000)
        self.batch_size_var = tk.IntVar(value=64)
        self.activation_var = tk.StringVar(value="relu")
        self.neurons_var = tk.IntVar(value=64)
        self.target_update_var = tk.IntVar(value=200)
        self.slippery_var = tk.BooleanVar(value=False)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)
        self.grid_columnconfigure(2, weight=0)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        self.grid_rowconfigure(3, weight=1)

        self.env_frame = ttk.LabelFrame(self, text="Environment")
        self.env_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        self.env_canvas = tk.Canvas(self.env_frame, width=1000, height=330, bg="black", highlightthickness=0)
        self.env_canvas.pack(fill="both", expand=True, padx=6, pady=6)
        env_controls = ttk.Frame(self.env_frame)
        env_controls.pack(fill="x", padx=6, pady=(0, 6))
        ttk.Checkbutton(env_controls, text="Slippery cliff", variable=self.slippery_var, command=self._on_slippery_toggle).pack(side="left")

        self.controls_frame = ttk.LabelFrame(self, text="Controls")
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=4)
        self.controls_frame.grid_columnconfigure(0, weight=1)
        self.controls_frame.grid_columnconfigure(1, weight=1)

        ttk.Button(self.controls_frame, text="Run single episode", command=self._on_run_episode).grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(self.controls_frame, text="Reset All", command=self._on_reset_all).grid(row=0, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(self.controls_frame, text="Train and Run", command=self._on_train).grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(self.controls_frame, text="Save samplings CSV", command=self._on_save_samplings_csv).grid(row=1, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(self.controls_frame, text="Save Plot PNG", command=self._on_save_plot_png).grid(row=2, column=0, sticky="ew", padx=4, pady=4)
        ttk.Button(self.controls_frame, text="Clear plots", command=self._on_clear_plots).grid(row=2, column=1, sticky="ew", padx=4, pady=4)

        self.state_frame = ttk.LabelFrame(self, text="Current State")
        self.state_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        self.state_label = ttk.Label(self.state_frame, text="")
        self.state_label.pack(fill="x", padx=6, pady=8)

        self.dnn_frame = ttk.LabelFrame(self, text="DNN Parameters")
        self.dnn_frame.grid(row=1, column=1, rowspan=2, sticky="ns", padx=4, pady=4)
        ttk.Label(self.dnn_frame, text="Replay buffer").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.dnn_frame, width=10, textvariable=self.replay_buffer_var).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(self.dnn_frame, text="Batch size").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.dnn_frame, width=10, textvariable=self.batch_size_var).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(self.dnn_frame, text="Activation").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(self.dnn_frame, width=9, state="readonly", values=["relu", "tanh", "sigmoid", "elu"], textvariable=self.activation_var).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(self.dnn_frame, text="Neurons").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.dnn_frame, width=10, textvariable=self.neurons_var).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(self.dnn_frame, text="Target sync").grid(row=4, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.dnn_frame, width=10, textvariable=self.target_update_var).grid(row=4, column=1, sticky="w", padx=6)

        self.params_frame = ttk.LabelFrame(self, text="Training Parameters")
        self.params_frame.grid(row=1, column=2, rowspan=2, sticky="ns", padx=8, pady=4)
        ttk.Label(self.params_frame, text="Max steps").grid(row=0, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.max_steps_var).grid(row=0, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Episodes").grid(row=1, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.episodes_var).grid(row=1, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Policy").grid(row=2, column=0, sticky="w", padx=6, pady=5)
        ttk.Combobox(self.params_frame, width=12, state="readonly", values=["Vanilla DQN", "Double DQN"], textvariable=self.policy_var).grid(row=2, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Eps start").grid(row=3, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.epsilon_start_var).grid(row=3, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Eps end").grid(row=4, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.epsilon_end_var).grid(row=4, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Eps decay").grid(row=5, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.epsilon_decay_var).grid(row=5, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Learn rate").grid(row=6, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.learning_rate_var).grid(row=6, column=1, sticky="w", padx=6)
        ttk.Label(self.params_frame, text="Gamma").grid(row=7, column=0, sticky="w", padx=6, pady=5)
        ttk.Entry(self.params_frame, width=10, textvariable=self.gamma_var).grid(row=7, column=1, sticky="w", padx=6)

        live_row = ttk.Frame(self.params_frame)
        live_row.grid(row=8, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        ttk.Checkbutton(live_row, text="Live plot", variable=self.live_plot_var).pack(side="left", padx=(0, 8))
        ttk.Checkbutton(live_row, text="reduced speed", variable=self.reduced_speed_var).pack(side="left")

        self.plot_frame = ttk.LabelFrame(self, text="Live Plot")
        self.plot_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        self.plot_frame.rowconfigure(0, weight=1)
        self.plot_frame.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10, 3), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas_plot = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas_plot.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.canvas_plot.mpl_connect("pick_event", self._on_legend_pick)

    def _on_window_close(self) -> None:
        self._stop_requested = True
        self.env.close()
        self.destroy()

    def _on_slippery_toggle(self) -> None:
        self.env.set_slippery(self.slippery_var.get())

    def _format_state_text(self, training: bool, step: int, episode: int) -> str:
        state = "Training" if training else "Idle"
        return f"{state:>8}: step:{step:>4d}  episode:{episode:>4d}"

    def _set_current_counters(self, episode: int, step: int, training: bool) -> None:
        self.current_episode = int(episode)
        self.current_step = int(step)
        self.state_label.configure(text=self._format_state_text(training, step=self.current_step, episode=self.current_episode))

    def _build_policy(self):
        common_kwargs = {
            "learning_rate": float(self.learning_rate_var.get()),
            "gamma": float(self.gamma_var.get()),
            "epsilon": float(self.epsilon_start_var.get()),
            "neurons": int(self.neurons_var.get()),
            "activation": self.activation_var.get(),
            "batch_size": int(self.batch_size_var.get()),
            "replay_buffer_size": int(self.replay_buffer_var.get()),
            "target_update_interval": int(self.target_update_var.get()),
        }
        if self.policy_var.get() == "Double DQN":
            return DDQNetwork(**common_kwargs)
        return DQNetwork(**common_kwargs)

    def _refresh_environment_view(self) -> None:
        frame = self.env.render_frame()
        image = Image.fromarray(frame)
        image = image.resize((1000, 300), resample=Image.NEAREST)
        self._env_image_tk = ImageTk.PhotoImage(image)
        self.env_canvas.delete("all")
        self.env_canvas.create_image(0, 0, anchor="nw", image=self._env_image_tk)

    def _moving_average(self, values: List[float], window: int = 10) -> List[float]:
        if not values:
            return []
        out: List[float] = []
        for index in range(len(values)):
            start = max(0, index - window + 1)
            segment = values[start : index + 1]
            out.append(sum(segment) / len(segment))
        return out

    def _update_plot(self, current_rewards: Optional[List[float]] = None) -> None:
        now = time.time()
        if now - self._last_plot_update < 0.15:
            return
        self._last_plot_update = now

        self.ax.clear()
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        handles = []
        labels = []
        for run in self.plot_runs:
            if not run.get("visible", True):
                continue
            rewards = run["rewards"]
            line, = self.ax.plot(rewards, linewidth=1.5)
            moving_avg, = self.ax.plot(self._moving_average(rewards, window=10), linewidth=3.0)
            handles.extend([line, moving_avg])
            labels.extend([run["label"], f"{run['label']} MA"])

        if current_rewards:
            line, = self.ax.plot(current_rewards, linewidth=1.5)
            moving_avg, = self.ax.plot(self._moving_average(current_rewards, window=10), linewidth=3.0)
            handles.extend([line, moving_avg])
            labels.extend(["Current", "Current MA"])

        self._legend_map = {}
        if handles:
            legend = self.ax.legend(handles, labels, loc="best")
            for leg_line, label in zip(legend.get_lines(), labels):
                leg_line.set_picker(True)
                self._legend_map[leg_line] = label

        self.figure.tight_layout()
        self.canvas_plot.draw_idle()

    def _redraw_all_plots(self) -> None:
        self._last_plot_update = 0.0
        self._update_plot()

    def _on_legend_pick(self, event) -> None:
        label = self._legend_map.get(event.artist)
        if not label:
            return
        base_label = label.replace(" MA", "")
        for run in self.plot_runs:
            if run["label"] == base_label:
                run["visible"] = not run.get("visible", True)
                break
        self._redraw_all_plots()

    def _epsilon_for_episode(self, episode: int) -> float:
        eps_start = float(self.epsilon_start_var.get())
        eps_end = float(self.epsilon_end_var.get())
        eps_decay = float(self.epsilon_decay_var.get())
        return max(eps_end, eps_start * (eps_decay ** max(0, episode - 1)))

    def _animate_transitions(self, transitions, on_done=None) -> None:
        self.env.reset()
        self._refresh_environment_view()

        def step_fn(index: int) -> None:
            if index >= len(transitions):
                if on_done is not None:
                    on_done()
                return
            row = int(round(float(transitions[index].next_state[0]) * (self.env.rows - 1)))
            col = int(round(float(transitions[index].next_state[1]) * (self.env.cols - 1)))
            self.env.current_state = self.env.encode_state(row, col)
            self._refresh_environment_view()
            self.after(45, lambda: step_fn(index + 1))

        step_fn(0)

    def _on_run_episode(self) -> None:
        self._stop_requested = False
        self.env.set_slippery(self.slippery_var.get())
        policy = self._build_policy()
        self._set_current_counters(1, 0, training=False)

        total_reward, transitions = self.trainer.run_episode(
            policy=policy,
            epsilon=float(self.epsilon_start_var.get()),
            max_steps=int(self.max_steps_var.get()),
            progress_callback=None,
        )

        self._animate_transitions(
            transitions,
            on_done=lambda: self._set_current_counters(1, len(transitions), training=False),
        )

        label = f"Episode {datetime.now().strftime('%H:%M:%S')}"
        self.plot_runs.append({"label": label, "rewards": [total_reward], "visible": True})
        self._redraw_all_plots()

    def _on_train(self) -> None:
        if self.training_thread and self.training_thread.is_alive():
            return
        self._stop_requested = False
        self.env.set_slippery(self.slippery_var.get())
        self.current_policy_obj = self._build_policy()
        self.training_thread = threading.Thread(target=self._train_worker, daemon=True)
        self.training_thread.start()

    def _train_worker(self) -> None:
        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        rewards: List[float] = []
        reduced = bool(self.reduced_speed_var.get())

        for episode in range(1, episodes + 1):
            if self._stop_requested:
                break

            def progress(step_no: int, ep=episode) -> None:
                self.after(0, lambda: self._set_current_counters(ep, step_no, training=True))
                self.after(0, self._refresh_environment_view)

            reward, _ = self.trainer.run_episode(
                policy=self.current_policy_obj,
                epsilon=self._epsilon_for_episode(episode),
                max_steps=max_steps,
                progress_callback=progress,
            )
            rewards.append(reward)

            if self.live_plot_var.get():
                self.after(0, lambda rewards_copy=rewards.copy(): self._update_plot(current_rewards=rewards_copy))

            if reduced:
                time.sleep(0.033)

        if rewards:
            label = f"Run {datetime.now().strftime('%H:%M:%S')}"
            self.plot_runs.append({"label": label, "rewards": rewards, "visible": True})

        self.after(0, lambda: self._set_current_counters(self.current_episode, self.current_step, training=False))
        self.after(0, self._redraw_all_plots)

    def _on_reset_all(self) -> None:
        self._stop_requested = True
        self.env.reset()
        self._refresh_environment_view()
        self.plot_runs.clear()
        self._legend_map.clear()
        self._redraw_all_plots()
        self.current_policy_obj = None
        self._set_current_counters(0, 0, training=False)

    def _on_save_samplings_csv(self) -> None:
        self.env.set_slippery(self.slippery_var.get())
        policy = self._build_policy()
        base = f"cliff_samplings_{self.policy_var.get().lower().replace(' ', '_')}"
        self.trainer.train(
            policy=policy,
            num_episodes=int(self.episodes_var.get()),
            max_steps=int(self.max_steps_var.get()),
            epsilon=float(self.epsilon_start_var.get()),
            save_csv=base,
        )

    def _on_save_plot_png(self) -> None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"plot_{self.policy_var.get().lower().replace(' ', '_')}_"
            f"epsmin{self.epsilon_end_var.get():.2f}_epsmax{self.epsilon_start_var.get():.2f}_"
            f"alpha{self.learning_rate_var.get():.4f}_gamma{self.gamma_var.get():.2f}_"
            f"episodes{self.episodes_var.get()}_maxsteps{self.max_steps_var.get()}_{stamp}.png"
        )
        self.figure.savefig(self.plots_dir / filename)

    def _on_clear_plots(self) -> None:
        self.plot_runs.clear()
        self._legend_map.clear()
        self._redraw_all_plots()


def run_gui() -> None:
    app = CliffWalkingGUI()
    app.mainloop()
