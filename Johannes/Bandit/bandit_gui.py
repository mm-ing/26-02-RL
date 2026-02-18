from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import ttk
from typing import Dict, List

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from bandit_logic import Agent, Environment


class GUI:
    RUN_COLORS = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    def __init__(self, root: tk.Tk, environment: Environment, agent: Agent) -> None:
        self.root = root
        self.environment = environment
        self.agent = agent
        self.run_results: List[Dict[str, object]] = []
        self.run_counter = 0

        self.root.title("Reinforcement Learning - Multi Armed Bandit")
        self.root.geometry("1100x700")

        self.bandit_rows: List[Dict[str, tk.Label]] = []
        self.policy_var = tk.StringVar(value=self.agent.active_policy_name)
        self.probability_mode_var = tk.StringVar(value=self.environment.probability_mode)
        self.run_loops_var = tk.IntVar(value=self.agent.default_loops)
        epsilon_policy = self.agent.policies.get("Epsilon-Greedy")
        default_epsilon = getattr(epsilon_policy, "epsilon", 0.9)
        default_decay = getattr(epsilon_policy, "epsilon_decay", 0.01)
        self.epsilon_var = tk.DoubleVar(value=float(default_epsilon))
        self.epsilon_decay_var = tk.DoubleVar(value=float(default_decay))
        self.agent_memory_var = tk.IntVar(value=int(self.agent.memory))
        self.start_1_var = tk.IntVar(value=int(self.environment.start_amounts[0]))
        self.start_2_var = tk.IntVar(value=int(self.environment.start_amounts[1]))
        self.start_3_var = tk.IntVar(value=int(self.environment.start_amounts[2]))

        self._build_layout()
        self.refresh_all()

    def _build_layout(self) -> None:
        title = tk.Label(self.root, text="Glücksspiel kann süchtig machen!", font=("Arial", 14, "bold"), fg="red")
        title.pack(pady=(8, 4))

        container = tk.Frame(self.root)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(container)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(container)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_bandit_controls(left)
        self._build_agent_controls(left)
        self._build_plot(right)

        self.status_label = tk.Label(self.root, text="Bereit", anchor="w")
        self.status_label.pack(fill=tk.X, padx=10, pady=(0, 8))

    def _build_bandit_controls(self, parent: tk.Frame) -> None:
        section = tk.LabelFrame(parent, text="Manuelle Spielzüge", padx=8, pady=8)
        section.pack(fill=tk.X, pady=(0, 8))

        for index, _ in enumerate(self.environment.bandits):
            row = tk.Frame(section)
            row.pack(fill=tk.X, pady=4)

            button = tk.Button(row, text=f"Bandit {index + 1} (1 Coin)", width=18, command=lambda idx=index: self.on_manual_pull(idx))
            button.grid(row=0, column=0, rowspan=4, padx=(0, 8))

            pulls = tk.Label(row, text="Versuche: 0", width=20, anchor="w")
            pulls.grid(row=0, column=1, sticky="w")

            success = tk.Label(row, text="Success: 0", width=20, anchor="w")
            success.grid(row=1, column=1, sticky="w")

            success_rate = tk.Label(row, text="Success Rate: 0.00", width=20, anchor="w")
            success_rate.grid(row=2, column=1, sticky="w")

            reward = tk.Label(row, text="Cumulative Reward: 0", width=24, anchor="w")
            reward.grid(row=3, column=1, sticky="w")

            self.bandit_rows.append({"pulls": pulls, "success": success, "success_rate": success_rate, "reward": reward})

    def _build_agent_controls(self, parent: tk.Frame) -> None:
        section = tk.LabelFrame(parent, text="Agent Steuerung", padx=8, pady=8)
        section.pack(fill=tk.X)

        ttk.Label(section, text="Policy:").grid(row=0, column=0, sticky="w")
        policy_box = ttk.Combobox(
            section,
            textvariable=self.policy_var,
            values=list(self.agent.policies.keys()),
            state="readonly",
            width=20,
        )
        policy_box.grid(row=0, column=1, sticky="w", padx=(6, 0))
        policy_box.bind("<<ComboboxSelected>>", self.on_policy_change)

        ttk.Label(section, text="Probability Mode:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        mode_box = ttk.Combobox(
            section,
            textvariable=self.probability_mode_var,
            values=["constant", "variable"],
            state="readonly",
            width=20,
        )
        mode_box.grid(row=1, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        mode_box.bind("<<ComboboxSelected>>", self.on_probability_mode_change)

        ttk.Label(section, text="Agent Loops (n):").grid(row=2, column=0, sticky="w", pady=(8, 0))
        loops_entry = ttk.Entry(section, textvariable=self.run_loops_var, width=10)
        loops_entry.grid(row=2, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Epsilon:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.epsilon_var, width=10).grid(row=3, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(section, text="(0.0 bis 1.0)").grid(row=3, column=2, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Epsilon-Decay:").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.epsilon_decay_var, width=10).grid(row=4, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(section, text="(>= 0.0)").grid(row=4, column=2, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Agent Memory:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.agent_memory_var, width=10).grid(row=5, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(section, text="(0 = unbegrenzt)").grid(row=5, column=2, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Start Bandit 1:").grid(row=6, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.start_1_var, width=10).grid(row=6, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Start Bandit 2:").grid(row=7, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.start_2_var, width=10).grid(row=7, column=1, sticky="w", padx=(6, 0), pady=(8, 0))

        ttk.Label(section, text="Start Bandit 3:").grid(row=8, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(section, textvariable=self.start_3_var, width=10).grid(row=8, column=1, sticky="w", padx=(6, 0), pady=(8, 0))
        ttk.Label(section, text="(>= 0)").grid(row=8, column=2, sticky="w", padx=(6, 0), pady=(8, 0))

        tk.Button(section, text="Agent single step", command=self.on_agent_single_step, width=18).grid(
            row=9, column=0, pady=(10, 0), sticky="w"
        )
        tk.Button(section, text="Agent run n loops", command=self.on_agent_run_loops, width=18).grid(
            row=9, column=1, padx=(6, 0), pady=(10, 0), sticky="w"
        )
        tk.Button(section, text="Reset", command=self.on_reset, width=18).grid(
            row=10, column=0, pady=(8, 0), sticky="w"
        )
        tk.Button(section, text="Clear plot history", command=self.on_clear_plot_history, width=18).grid(
            row=10, column=1, padx=(6, 0), pady=(8, 0), sticky="w"
        )
        tk.Button(section, text="Save plot", command=self.on_save_plot, width=18).grid(
            row=11, column=0, pady=(8, 0), sticky="w"
        )

        self.final_stats_label = tk.Label(section, text="Beste Maschine: -", justify="left", anchor="w")
        self.final_stats_label.grid(row=12, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_plot(self, parent: tk.Frame) -> None:
        frame = tk.LabelFrame(parent, text="Cumulative Reward (Live)", padx=6, pady=6)
        frame.pack(fill=tk.BOTH, expand=True)

        self.figure = Figure(figsize=(7, 5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.figure.subplots_adjust(top=0.80)
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Cumulative Reward")
        self.ax.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        self._refresh_plot()

    def on_policy_change(self, _event=None) -> None:
        selected = self.policy_var.get()
        self.agent.set_policy(selected)
        self.status_label.config(text=f"Policy aktiv: {selected}")

    def on_probability_mode_change(self, _event=None) -> None:
        selected_mode = self.probability_mode_var.get()
        self.environment.set_probability_mode(selected_mode)
        self.status_label.config(text=f"Probability Mode aktiv: {selected_mode}")

    def on_manual_pull(self, bandit_index: int) -> None:
        reward = self.environment.step(bandit_index)
        self.refresh_bandit_labels()
        self.status_label.config(text=f"Manueller Spielzug Bandit {bandit_index + 1}: Reward={reward}")
        self._update_final_stats()

    def _apply_agent_inputs(self) -> bool:
        try:
            epsilon = float(self.epsilon_var.get())
            epsilon_decay = float(self.epsilon_decay_var.get())
            memory = int(self.agent_memory_var.get())
            probability_mode = str(self.probability_mode_var.get())
            start_1 = int(self.start_1_var.get())
            start_2 = int(self.start_2_var.get())
            start_3 = int(self.start_3_var.get())
        except (tk.TclError, ValueError):
            self.status_label.config(
                text="Ungültige Eingabe: Epsilon/Decay als Zahl, Memory/Startwerte als Ganzzahlen eingeben."
            )
            return False

        if probability_mode not in {"constant", "variable"}:
            self.status_label.config(text="Ungültiger Probability Mode. Erlaubt: constant oder variable.")
            return False

        epsilon = min(max(epsilon, 0.0), 1.0)
        epsilon_decay = max(0.0, epsilon_decay)
        memory = max(0, memory)
        start_1 = max(0, start_1)
        start_2 = max(0, start_2)
        start_3 = max(0, start_3)

        self.epsilon_var.set(epsilon)
        self.epsilon_decay_var.set(epsilon_decay)
        self.agent_memory_var.set(memory)
        self.probability_mode_var.set(probability_mode)
        self.start_1_var.set(start_1)
        self.start_2_var.set(start_2)
        self.start_3_var.set(start_3)

        self.agent.configure(memory=memory, epsilon=epsilon, epsilon_decay=epsilon_decay)
        self.environment.start_amounts = (start_1, start_2, start_3)
        self.environment.set_probability_mode(probability_mode)
        return True

    def _build_run_label(self, loops: int) -> str:
        policy_name = self.policy_var.get()
        epsilon = float(self.epsilon_var.get())
        epsilon_decay = float(self.epsilon_decay_var.get())
        memory = int(self.agent_memory_var.get())
        probability_mode = self.probability_mode_var.get()
        return (
            f"Run {self.run_counter} | {policy_name} | "
            f"mode={probability_mode} | eps={epsilon:.2f}, dec={epsilon_decay:.3f}, mem={memory}, n={loops}"
        )

    @staticmethod
    def _format_filename_float(value: float) -> str:
        return f"{value:.3f}".replace(".", "p")

    def _build_plot_filename(self) -> str:
        run_count = len(self.run_results)

        if run_count == 0:
            mode = self.probability_mode_var.get()
            epsilon_min = epsilon_max = float(self.epsilon_var.get())
            decay_min = decay_max = float(self.epsilon_decay_var.get())
        else:
            modes = {str(run.get("probability_mode", "")) for run in self.run_results}
            mode = modes.pop() if len(modes) == 1 else "mixed"

            epsilons = [float(run.get("epsilon", 0.0)) for run in self.run_results]
            decays = [float(run.get("epsilon_decay", 0.0)) for run in self.run_results]
            epsilon_min = min(epsilons)
            epsilon_max = max(epsilons)
            decay_min = min(decays)
            decay_max = max(decays)

        return (
            f"bandit_plot_runs{run_count}"
            f"_mode-{mode}"
            f"_epsmin-{self._format_filename_float(epsilon_min)}"
            f"_epsmax-{self._format_filename_float(epsilon_max)}"
            f"_decmin-{self._format_filename_float(decay_min)}"
            f"_decmax-{self._format_filename_float(decay_max)}"
            ".png"
        )

    def _create_new_agent_for_run(self, loops: int) -> None:
        policy_name = self.policy_var.get()
        epsilon = float(self.epsilon_var.get())
        epsilon_decay = float(self.epsilon_decay_var.get())
        memory = int(self.agent_memory_var.get())
        probability_mode = self.probability_mode_var.get()

        self.environment = Environment(start_amounts=self.environment.start_amounts, probability_mode=probability_mode)
        self.agent = Agent(
            environment=self.environment,
            loops=loops,
            memory=memory,
            epsilon_start=epsilon,
            epsilon_decay=epsilon_decay,
        )
        self.agent.set_policy(policy_name)

    def on_agent_single_step(self) -> None:
        if not self._apply_agent_inputs():
            return
        action, reward = self.agent.step()
        self.refresh_all()
        self.status_label.config(
            text=f"Agent Step ({self.agent.active_policy_name}): Bandit {action + 1}, Reward={reward}"
        )

    def on_agent_run_loops(self) -> None:
        if not self._apply_agent_inputs():
            return
        loops = max(0, int(self.run_loops_var.get()))

        self.run_counter += 1
        self._create_new_agent_for_run(loops=loops)
        epsilon = float(self.epsilon_var.get())
        epsilon_decay = float(self.epsilon_decay_var.get())
        probability_mode = self.probability_mode_var.get()
        run_series: Dict[str, object] = {
            "label": self._build_run_label(loops=loops),
            "color": self.RUN_COLORS[(self.run_counter - 1) % len(self.RUN_COLORS)],
            "probability_mode": probability_mode,
            "epsilon": epsilon,
            "epsilon_decay": epsilon_decay,
            "x": [],
            "y": [],
        }
        self.run_results.append(run_series)

        for _ in range(loops):
            self.agent.step()
            x_values = run_series["x"]
            y_values = run_series["y"]
            if isinstance(x_values, list) and isinstance(y_values, list):
                x_values.append(self.agent.total_steps)
                y_values.append(self.agent.total_reward)
            self.refresh_all(redraw_plot=True)
            self.root.update_idletasks()
            self.root.update()

        best = self.agent.get_best_bandit()
        self.status_label.config(
            text=(
                f"Run abgeschlossen ({loops} Loops, Policy: {self.agent.active_policy_name}). "
                f"Beste Maschine: {best['bandit']} mit Reward={best['cumulative_reward']}"
            )
        )

    def on_reset(self) -> None:
        if not self._apply_agent_inputs():
            return
        self.agent.reset()
        self.refresh_all()
        self.status_label.config(text="Reset durchgeführt (Run-Historie im Plot bleibt erhalten)")

    def on_clear_plot_history(self) -> None:
        self.run_results.clear()
        self.run_counter = 0
        self.refresh_all(redraw_plot=True)
        self.status_label.config(text="Plot-Historie gelöscht")

    def on_save_plot(self) -> None:
        plots_dir = Path(__file__).resolve().parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        base_name = self._build_plot_filename()
        file_path = plots_dir / base_name
        if file_path.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = plots_dir / f"{file_path.stem}_{timestamp}.png"
        self.figure.savefig(file_path, format="png", dpi=150, bbox_inches="tight")
        self.status_label.config(text=f"Plot gespeichert: {file_path.name}")

    def refresh_bandit_labels(self) -> None:
        states = self.environment.get_state()
        for index, state in enumerate(states):
            self.bandit_rows[index]["pulls"].config(text=f"Versuche: {state['pulls']}")
            self.bandit_rows[index]["success"].config(text=f"Success: {state['success']}")
            self.bandit_rows[index]["success_rate"].config(text=f"Success Rate: {state['success_rate']:.2f}")
            self.bandit_rows[index]["reward"].config(text=f"Cumulative Reward: {state['cumulative_reward']}")

    def _update_final_stats(self) -> None:
        best = self.agent.get_best_bandit()
        self.final_stats_label.config(
            text=f"Beste Maschine: {best['bandit']} | Pulls={best['pulls']} | Reward={best['cumulative_reward']}"
        )

    def _refresh_plot(self) -> None:
        self.ax.clear()
        self.ax.set_xlabel("Step")
        self.ax.set_ylabel("Cumulative Reward")
        self.ax.grid(True, alpha=0.3)

        for run in self.run_results:
            x_values = run.get("x", [])
            y_values = run.get("y", [])
            if not x_values or not y_values:
                continue
            self.ax.plot(
                x_values,
                y_values,
                color=str(run.get("color", "black")),
                label=str(run.get("label", "Run")),
                linewidth=2,
            )

        if self.run_results:
            self.ax.legend(
                loc="lower center",
                bbox_to_anchor=(0.5, 1.02),
                fontsize=8,
                ncol=1,
                frameon=True,
            )
        self.canvas.draw_idle()

    def refresh_all(self, redraw_plot: bool = True) -> None:
        self.refresh_bandit_labels()
        self._update_final_stats()
        if redraw_plot:
            self._refresh_plot()
