"""Tkinter GUI for the Manual + Agent-driven Multi-Armed Bandit demo.

This module provides `BanditGUI` and supports fresh agent creation per run
using an `agent_factory`.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from typing import Callable

try:
    import matplotlib

    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    matplotlib = None

from bandit_logic import Agent, EpsilonGreedyPolicy, ThompsonSamplingPolicy


class BanditGUI:
    def __init__(self, master, agent_factory):
        if not callable(agent_factory):
            raise TypeError("BanditGUI expects agent_factory (callable), not Agent instance.")
        self.master = master
        self.agent_factory = agent_factory
        self.agent = self.agent_factory()
        self.env = self.agent.env

        # initial fresh agent
        self.agent = self.agent_factory()
        self.env = self.agent.env

        self.master.title("Bandit Demo — Manual + Agent")
        self.master.geometry("900x620")

        self._running = False
        self._run_after_id = None

        # plot series store: list of dicts {label, x, y, color}
        self.series = []
        self.run_counter = 0
        self.color_cycle = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]

        self._build_ui()
        self._update_display(plot=False)

    def _build_ui(self):
        top_frame = ttk.Frame(self.master, padding=8)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # Manual bandit buttons
        manual_frame = ttk.LabelFrame(top_frame, text="Manual Bandit Pulls", padding=8)
        manual_frame.pack(side=tk.LEFT, padx=8)

        for a in range(self.env.n_actions()):
            btn = ttk.Button(manual_frame, text=f"Bandit {a+1}", command=lambda i=a: self._manual_pull(i))
            btn.grid(row=0, column=a, padx=6)

        # Agent controls
        control_frame = ttk.LabelFrame(top_frame, text="Agent Controls", padding=8)
        control_frame.pack(side=tk.LEFT, padx=8, fill=tk.X)

        # Policy selector
        self.policy_var = tk.StringVar(value=self.agent.policy.name())
        ttk.Radiobutton(
            control_frame,
            text="Epsilon-Greedy",
            variable=self.policy_var,
            value="EpsilonGreedyPolicy",
            command=self._on_policy_change,
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Radiobutton(
            control_frame,
            text="Thompson Sampling",
            variable=self.policy_var,
            value="ThompsonSamplingPolicy",
            command=self._on_policy_change,
        ).grid(row=0, column=1, sticky=tk.W)

        # Epsilon and decay
        ttk.Label(control_frame, text="epsilon:").grid(row=1, column=0, sticky=tk.E)
        self.epsilon_var = tk.DoubleVar(value=0.9)
        ttk.Entry(control_frame, textvariable=self.epsilon_var, width=8).grid(row=1, column=1, sticky=tk.W)

        ttk.Label(control_frame, text="decay:").grid(row=1, column=2, sticky=tk.E)
        self.decay_var = tk.DoubleVar(value=0.001)
        ttk.Entry(control_frame, textvariable=self.decay_var, width=8).grid(row=1, column=3, sticky=tk.W)

        # Memory size
        ttk.Label(control_frame, text="memory (0=full):").grid(row=2, column=0, sticky=tk.E)
        self.memory_var = tk.IntVar(value=100)
        ttk.Entry(control_frame, textvariable=self.memory_var, width=8).grid(row=2, column=1, sticky=tk.W)

        # Single step and run N steps
        ttk.Button(control_frame, text="Single Step", command=self._single_step).grid(row=3, column=0, pady=6)

        ttk.Label(control_frame, text="Run N steps:").grid(row=3, column=1, sticky=tk.E)
        self.nsteps_var = tk.IntVar(value=50)
        ttk.Entry(control_frame, textvariable=self.nsteps_var, width=8).grid(row=3, column=2, sticky=tk.W)
        self.run_button = ttk.Button(control_frame, text="Run", command=self._run_n_steps)
        self.run_button.grid(row=3, column=3, padx=6)

        ttk.Button(control_frame, text="Reset", command=self._reset_all).grid(row=4, column=0, pady=(6, 0))
        ttk.Button(control_frame, text="Save Plot", command=self._save_plot).grid(row=4, column=1, pady=(6, 0), padx=(6, 0), sticky=tk.W)

        # Status and stats
        stats_frame = ttk.LabelFrame(self.master, text="Current State", padding=8)
        stats_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 0))

        self.pull_vars = [tk.StringVar(value="Pulls: 0") for _ in range(self.env.n_actions())]
        self.win_vars = [tk.StringVar(value="Wins: 0") for _ in range(self.env.n_actions())]
        self.rate_vars = [tk.StringVar(value="Rate: 0.00") for _ in range(self.env.n_actions())]

        for i in range(self.env.n_actions()):
            ttk.Label(stats_frame, text=f"Bandit {i+1}").grid(row=0, column=i * 3)
            ttk.Label(stats_frame, textvariable=self.pull_vars[i]).grid(row=1, column=i * 3)
            ttk.Label(stats_frame, textvariable=self.win_vars[i]).grid(row=1, column=i * 3 + 1, padx=(4, 0))
            ttk.Label(stats_frame, textvariable=self.rate_vars[i]).grid(row=1, column=i * 3 + 2, padx=(4, 0))

        self.epsilon_display = tk.StringVar(value=f"epsilon: {getattr(self.agent.policy, 'epsilon', 'n/a')}")
        ttk.Label(stats_frame, textvariable=self.epsilon_display).grid(row=2, column=0, sticky=tk.W, pady=(6, 0))

        self.cum_reward_var = tk.StringVar(value="Cumulative reward: 0")
        ttk.Label(stats_frame, textvariable=self.cum_reward_var).grid(row=2, column=1, sticky=tk.W, pady=(6, 0))

        self.policy_display = tk.StringVar(value=self.agent.policy.name())
        ttk.Label(stats_frame, textvariable=self.policy_display).grid(row=2, column=2, sticky=tk.W, pady=(6, 0))

        # Plot area
        plot_frame = ttk.Frame(self.master, padding=8)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Cumulative Reward over Steps")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Cumulative Reward")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def _selected_policy(self):
        name = self.policy_var.get()
        if name == "EpsilonGreedyPolicy":
            eps = float(self.epsilon_var.get())
            decay = float(self.decay_var.get())
            return EpsilonGreedyPolicy(epsilon=eps, decay=decay)
        return ThompsonSamplingPolicy()

    def _fresh_agent_from_ui(self):
        """Create a brand-new agent and apply current UI-selected policy/params."""
        agent = self.agent_factory()
        agent.memory_size = int(self.memory_var.get())
        agent.set_policy(self._selected_policy())
        return agent

    def _on_policy_change(self):
        # Policy change starts a fresh independent run state (plot kept)
        self.agent = self._fresh_agent_from_ui()
        self.env = self.agent.env
        self._update_display(plot=False)

    def _start_new_series(self, label: str):
        color = self.color_cycle[len(self.series) % len(self.color_cycle)]
        self.series.append({"label": label, "x": [], "y": [], "color": color})
        self._redraw_plot()

    def _policy_label_from_ui(self) -> str:
        mem = int(self.memory_var.get())
        if self.policy_var.get() == "EpsilonGreedyPolicy":
            eps = float(self.epsilon_var.get())
            decay = float(self.decay_var.get())
            return f"EpsilonGreedy | eps={eps:g}, decay={decay:g}, mem={mem}"
        return f"ThompsonSampling | mem={mem}"

    def _next_run_label(self) -> str:
        self.run_counter += 1
        return f"Run {self.run_counter} | {self._policy_label_from_ui()}"

    def _append_series_point(self):
        if not self.series:
            return
        s = self.series[-1]
        s["x"].append(len(s["x"]) + 1)
        s["y"].append(self.agent.cumulative_reward)

    def _redraw_plot(self):
        self.ax.clear()
        self.ax.set_title("Cumulative Reward over Steps")
        self.ax.set_xlabel("Steps")
        self.ax.set_ylabel("Cumulative Reward")

        seen = set()
        has_legend_item = False

        for s in self.series:
            if not s["x"]:
                continue

            # Keep legend labels unique without manual handles.
            label = s["label"] if s["label"] not in seen else "_nolegend_"
            seen.add(s["label"])
            if label != "_nolegend_":
                has_legend_item = True

            self.ax.plot(s["x"], s["y"], color=s["color"], label=label)

        if has_legend_item:
            self.ax.legend()

        self.canvas.draw_idle()

    def _update_display(self, plot: bool = True):
        for i in range(self.env.n_actions()):
            self.pull_vars[i].set(f"Pulls: {self.agent.pulls[i]}")
            self.win_vars[i].set(f"Wins: {self.agent.wins[i]}")
            rate = (self.agent.wins[i] / self.agent.pulls[i]) if self.agent.pulls[i] > 0 else 0.0
            self.rate_vars[i].set(f"Rate: {rate:.2f}")

        eps = getattr(self.agent.policy, "epsilon", None)
        self.epsilon_display.set(f"epsilon: {eps:.4f}" if eps is not None else "epsilon: n/a")
        self.cum_reward_var.set(f"Cumulative reward: {self.agent.cumulative_reward}")
        self.policy_display.set(self.agent.policy.name())

        if plot:
            self._append_series_point()
            self._redraw_plot()

    def _manual_pull(self, action: int):
        self.agent.step(action=action)
        self._update_display()

    def _single_step(self):
        self._sync_params()
        self.agent.step()
        self._update_display()

    def _run_n_steps(self):
        try:
            n = int(self.nsteps_var.get())
        except Exception:
            messagebox.showerror("Invalid N", "Please enter a valid integer for N steps.")
            return

        if n <= 0:
            return

        # Capture label from current UI params at click time.
        run_label = self._next_run_label()

        # Start every run from a fresh independent state
        self.agent = self._fresh_agent_from_ui()
        self.env = self.agent.env
        self._start_new_series(run_label)
        self._update_display()

        self._running = True
        self.run_button.config(state=tk.DISABLED)
        self._run_loop(remaining=n)

    def _run_loop(self, remaining: int):
        if remaining <= 0:
            self._running = False
            self.run_button.config(state=tk.NORMAL)
            return

        self.agent.step()
        self._update_display()

        self._run_after_id = self.master.after(50, lambda: self._run_loop(remaining - 1))

    def _sync_params(self):
        try:
            self.agent.memory_size = int(self.memory_var.get())
        except Exception:
            pass

        if isinstance(self.agent.policy, EpsilonGreedyPolicy):
            try:
                self.agent.policy.epsilon = float(self.epsilon_var.get())
                self.agent.policy.decay = float(self.decay_var.get())
            except Exception:
                pass

    def _reset_all(self):
        if self._run_after_id:
            self.master.after_cancel(self._run_after_id)
            self._run_after_id = None

        self._running = False
        self.run_button.config(state=tk.NORMAL)

        # Fully fresh state on reset + clear plot/legend/history
        self.agent = self._fresh_agent_from_ui()
        self.env = self.agent.env
        self.series.clear()
        self.run_counter = 0
        self._redraw_plot()
        self._update_display(plot=False)

    def _save_plot(self):
        file_path = filedialog.asksaveasfilename(
            title="Save Plot As",
            defaultextension=".png",
            filetypes=[
                ("PNG Image", "*.png"),
                ("JPEG Image", "*.jpg *.jpeg"),
                ("All Files", "*.*"),
            ],
        )
        if not file_path:
            return

        try:
            # Ensure the saved output matches the current GUI-rendered figure.
            self.canvas.draw()
            self.fig.savefig(file_path, dpi=self.fig.dpi)
        except Exception as e:
            messagebox.showerror("Save Failed", f"Could not save plot:\n{e}")