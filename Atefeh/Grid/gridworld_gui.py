from __future__ import annotations

import ast
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from .gridworld_logic import GridMap, GridWorldLab, State
except ImportError:
    from gridworld_logic import GridMap, GridWorldLab, State


class GridWorldGUI(tk.Tk):
    """Tkinter UI with step-wise animation for single-step, single-episode, and training runs."""

    CELL_SIZE = 70

    def __init__(self, lab: GridWorldLab) -> None:
        super().__init__()
        self.title("Gridworld Labyrinth RL")
        self.lab = lab

        self.agent_position: State = self.lab.current_state()
        self.reward_history: List[int] = []
        self.episode_open = False

        self.dragging: Optional[str] = None
        self.is_training = False
        self.stop_training_requested = False

        self.training_target_episodes = 0
        self.training_completed_episodes = 0

        self._build_layout()
        self._draw_grid()
        self._refresh_status("Ready")

    def _build_layout(self) -> None:
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        left = ttk.Frame(self, padding=8)
        left.grid(row=0, column=0, sticky="nsew")
        left.rowconfigure(0, weight=1)
        left.columnconfigure(0, weight=1)

        right = ttk.Frame(self, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(left, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        controls = ttk.LabelFrame(left, text="Grid Configuration", padding=8)
        controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.rows_var = tk.IntVar(value=self.lab.grid_map.rows)
        self.cols_var = tk.IntVar(value=self.lab.grid_map.cols)
        self.blocked_var = tk.StringVar(value=self._blocked_to_string())
        self.start_var = tk.StringVar(value=str(self.lab.grid_map.start))
        self.target_var = tk.StringVar(value=str(self.lab.grid_map.target))

        ttk.Label(controls, text="Rows (N)").grid(row=0, column=0, sticky="w")
        ttk.Entry(controls, textvariable=self.rows_var, width=10).grid(row=0, column=1, sticky="w", padx=(6, 10))

        ttk.Label(controls, text="Cols (M)").grid(row=0, column=2, sticky="w")
        ttk.Entry(controls, textvariable=self.cols_var, width=10).grid(row=0, column=3, sticky="w", padx=(6, 0))

        ttk.Label(controls, text="Blocked").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.blocked_var, width=36).grid(
            row=1, column=1, columnspan=3, sticky="ew", pady=(6, 0)
        )

        ttk.Label(controls, text="Start").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.start_var, width=14).grid(row=2, column=1, sticky="w", pady=(6, 0))

        ttk.Label(controls, text="Target").grid(row=2, column=2, sticky="w", pady=(6, 0))
        ttk.Entry(controls, textvariable=self.target_var, width=14).grid(row=2, column=3, sticky="w", pady=(6, 0))

        self.apply_grid_button = ttk.Button(controls, text="Apply Grid", command=self._apply_grid_and_learning)
        self.apply_grid_button.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(8, 0))

        plot_frame = ttk.LabelFrame(right, text="Episode Reward", padding=8)
        plot_frame.grid(row=0, column=0, sticky="ew")

        figure = Figure(figsize=(4.8, 2.8), dpi=100)
        self.plot_axis = figure.add_subplot(111)
        self.plot_axis.set_xlabel("Episode")
        self.plot_axis.set_ylabel("Reward")
        (self.reward_line,) = self.plot_axis.plot([], [], color="#1f77b4")
        self.plot_canvas = FigureCanvasTkAgg(figure, master=plot_frame)
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)

        params = ttk.LabelFrame(right, text="Learning Parameters", padding=8)
        params.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        self.gamma_var = tk.DoubleVar(value=0.99)
        self.alpha_var = tk.DoubleVar(value=0.2)
        self.max_steps_var = tk.IntVar(value=100)
        self.episodes_var = tk.IntVar(value=100)
        self.eps_max_var = tk.DoubleVar(value=1.0)
        self.eps_min_var = tk.DoubleVar(value=0.05)
        self.eps_decay_var = tk.DoubleVar(value=0.995)

        self.visualize_training_var = tk.BooleanVar(value=True)
        self.visualize_every_var = tk.IntVar(value=1)
        self.delay_ms_var = tk.IntVar(value=90)

        self.policy_var = tk.StringVar(value=self.lab.POLICY_Q_LEARNING)

        fields = [
            ("Gamma (γ)", self.gamma_var),
            ("Alpha (α)", self.alpha_var),
            ("Max steps", self.max_steps_var),
            ("Episodes", self.episodes_var),
            ("Epsilon max", self.eps_max_var),
            ("Epsilon min", self.eps_min_var),
            ("Epsilon decay", self.eps_decay_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(params, text=label).grid(row=row, column=0, sticky="w")
            ttk.Entry(params, textvariable=variable, width=12).grid(row=row, column=1, sticky="w", padx=(6, 0))

        ttk.Label(params, text="Policy").grid(row=7, column=0, sticky="w", pady=(6, 0))
        self.policy_combo = ttk.Combobox(
            params,
            textvariable=self.policy_var,
            values=[self.lab.POLICY_MONTE_CARLO, self.lab.POLICY_Q_LEARNING],
            state="readonly",
            width=15,
        )
        self.policy_combo.grid(row=7, column=1, sticky="w", padx=(6, 0), pady=(6, 0))

        ttk.Checkbutton(
            params,
            text="Visualize training",
            variable=self.visualize_training_var,
        ).grid(row=8, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(params, text="Visualize every k-th").grid(row=9, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.visualize_every_var, width=12).grid(row=9, column=1, sticky="w", padx=(6, 0))

        ttk.Label(params, text="Step delay (ms)").grid(row=10, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.delay_ms_var, width=12).grid(row=10, column=1, sticky="w", padx=(6, 0))

        actions = ttk.LabelFrame(right, text="Actions", padding=8)
        actions.grid(row=2, column=0, sticky="ew", pady=(8, 0))

        self.run_step_button = ttk.Button(actions, text="Run single step", command=self._run_single_step)
        self.run_step_button.grid(row=0, column=0, sticky="ew")

        self.run_episode_button = ttk.Button(actions, text="Run single episode", command=self._run_single_episode)
        self.run_episode_button.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.train_button = ttk.Button(actions, text="Train and run", command=self._train_and_run)
        self.train_button.grid(row=2, column=0, sticky="ew", pady=(4, 0))

        self.stop_button = ttk.Button(actions, text="Stop training", command=self._request_stop_training, state="disabled")
        self.stop_button.grid(row=3, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(actions, text="Show value table", command=self._show_value_table).grid(
            row=4, column=0, sticky="ew", pady=(4, 0)
        )
        ttk.Button(actions, text="Show Q-table", command=self._show_q_table).grid(row=5, column=0, sticky="ew", pady=(4, 0))

        ttk.Button(actions, text="Save samplings into CSV", command=self._save_sampling_csv).grid(
            row=6, column=0, sticky="ew", pady=(4, 0)
        )

        self.reset_button = ttk.Button(actions, text="Rest", command=self._reset_session)
        self.reset_button.grid(row=7, column=0, sticky="ew", pady=(4, 0))

        self.status_var = tk.StringVar(value="")
        ttk.Label(right, textvariable=self.status_var, wraplength=340, justify="left").grid(
            row=3, column=0, sticky="ew", pady=(8, 0)
        )

        for row in range(4):
            right.rowconfigure(row, weight=0)
        right.rowconfigure(0, weight=1)

    def _blocked_to_string(self) -> str:
        return "; ".join(str(cell) for cell in sorted(self.lab.grid_map.blocked))

    def _parse_state(self, raw: str) -> State:
        value = ast.literal_eval(raw.strip())
        if not (isinstance(value, tuple) and len(value) == 2):
            raise ValueError("Expected tuple like (x, y).")
        x, y = value
        if not (isinstance(x, int) and isinstance(y, int)):
            raise ValueError("State tuple must contain integers.")
        return x, y

    def _parse_blocked(self, raw: str) -> List[State]:
        if not raw.strip():
            return []
        cells: List[State] = []
        for item in raw.split(";"):
            parsed = self._parse_state(item)
            cells.append(parsed)
        return cells

    def _apply_grid_and_learning(self) -> bool:
        try:
            rows = int(self.rows_var.get())
            cols = int(self.cols_var.get())
            blocked = self._parse_blocked(self.blocked_var.get())
            start = self._parse_state(self.start_var.get())
            target = self._parse_state(self.target_var.get())

            self.lab.apply_grid_configuration(rows=rows, cols=cols, blocked=blocked, start=start, target=target)
            self.lab.set_policy(self.policy_var.get())
            self.lab.configure_learning(
                gamma=float(self.gamma_var.get()),
                alpha=float(self.alpha_var.get()),
                max_steps_per_episode=int(self.max_steps_var.get()),
                epsilon_max=float(self.eps_max_var.get()),
                epsilon_min=float(self.eps_min_var.get()),
                epsilon_decay=float(self.eps_decay_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Invalid configuration", str(exc))
            return False

        self.agent_position = self.lab.current_state()
        self._draw_grid()
        self._refresh_status("Configuration applied successfully")
        return True

    def _draw_grid(self) -> None:
        grid_map: GridMap = self.lab.grid_map
        width = max(1, grid_map.cols * self.CELL_SIZE)
        height = max(1, grid_map.rows * self.CELL_SIZE)
        self.canvas.configure(width=width, height=height)
        self.canvas.delete("all")

        for x in range(grid_map.cols):
            for y in range(grid_map.rows):
                x0, y0 = x * self.CELL_SIZE, y * self.CELL_SIZE
                x1, y1 = x0 + self.CELL_SIZE, y0 + self.CELL_SIZE
                fill = "#d0d0d0" if (x, y) in grid_map.blocked else "white"
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#666666")

        target_x, target_y = grid_map.target
        tx0, ty0 = target_x * self.CELL_SIZE + 9, target_y * self.CELL_SIZE + 9
        tx1, ty1 = (target_x + 1) * self.CELL_SIZE - 9, (target_y + 1) * self.CELL_SIZE - 9
        self._rounded_rect(tx0, ty0, tx1, ty1, radius=12, fill="#4caf50", outline="#2e7d32")

        ax, ay = self.agent_position
        x0, y0 = ax * self.CELL_SIZE + 13, ay * self.CELL_SIZE + 13
        x1, y1 = (ax + 1) * self.CELL_SIZE - 13, (ay + 1) * self.CELL_SIZE - 13
        self.canvas.create_oval(x0, y0, x1, y1, fill="#1976d2", outline="#0d47a1", width=2)

    def _rounded_rect(self, x0: int, y0: int, x1: int, y1: int, radius: int, **kwargs: object) -> None:
        points = [
            x0 + radius, y0,
            x1 - radius, y0,
            x1, y0,
            x1, y0 + radius,
            x1, y1 - radius,
            x1, y1,
            x1 - radius, y1,
            x0 + radius, y1,
            x0, y1,
            x0, y1 - radius,
            x0, y0 + radius,
            x0, y0,
        ]
        self.canvas.create_polygon(points, smooth=True, **kwargs)

    def _pixel_to_cell(self, x: int, y: int) -> State:
        col = max(0, min(self.lab.grid_map.cols - 1, x // self.CELL_SIZE))
        row = max(0, min(self.lab.grid_map.rows - 1, y // self.CELL_SIZE))
        return int(col), int(row)

    def _on_canvas_press(self, event: tk.Event) -> None:
        cell = self._pixel_to_cell(event.x, event.y)
        if cell == self.lab.grid_map.start:
            self.dragging = "start"
        elif cell == self.lab.grid_map.target:
            self.dragging = "target"
        else:
            self.dragging = None

    def _on_canvas_release(self, event: tk.Event) -> None:
        if self.is_training:
            return
        cell = self._pixel_to_cell(event.x, event.y)
        try:
            if self.dragging == "start":
                self.lab.grid_map.set_start(cell)
                self.start_var.set(str(cell))
            elif self.dragging == "target":
                self.lab.grid_map.set_target(cell)
                self.target_var.set(str(cell))
            else:
                self.lab.grid_map.toggle_blocked(cell)
                self.blocked_var.set(self._blocked_to_string())
        except ValueError as exc:
            messagebox.showerror("Invalid edit", str(exc))

        self.dragging = None
        self.lab.environment.reset()
        self.agent_position = self.lab.current_state()
        self._draw_grid()

    def _start_episode_if_needed(self) -> None:
        if not self.episode_open:
            self.agent_position = self.lab.reset_episode()
            self.episode_open = True
            self._draw_grid()

    def _advance_one_step(self) -> Tuple[State, int, State, int, bool]:
        transition = self.lab.step()
        self.agent_position = transition[2]
        self._draw_grid()
        action_names = {0: "up", 1: "down", 2: "left", 3: "right"}
        self._refresh_status(
            f"Ep {self.lab.current_episode_index} | Step {self.lab.current_step_index} | "
            f"State {transition[0]} | Action {action_names[transition[1]]} | "
            f"Reward {transition[3]}"
        )
        return transition

    def _close_episode(self) -> None:
        self.lab.finish_episode()
        self.reward_history.append(self.lab.episode_reward)
        self._update_reward_plot()
        self.lab.current_episode_index += 1
        self.episode_open = False

    def _episode_should_stop(self, done: bool) -> bool:
        return done or self.lab.current_step_index >= self.lab.max_steps_per_episode

    def _run_single_step(self) -> None:
        if self.is_training:
            return
        if not self._apply_grid_and_learning():
            return
        self._start_episode_if_needed()
        _prev, _action, _next, _reward, done = self._advance_one_step()
        if self._episode_should_stop(done):
            self._close_episode()

    def _run_single_episode(self) -> None:
        if self.is_training:
            return
        if not self._apply_grid_and_learning():
            return

        self._set_controls_busy(True)
        self._start_episode_if_needed()

        # Tkinter animations use after() so the main loop keeps handling paint/events responsively.
        self.after(0, self._single_episode_tick)

    def _single_episode_tick(self) -> None:
        _prev, _action, _next, _reward, done = self._advance_one_step()
        if self._episode_should_stop(done):
            self._close_episode()
            self._set_controls_busy(False)
            return
        delay = max(1, int(self.delay_ms_var.get()))
        self.after(delay, self._single_episode_tick)

    def _train_and_run(self) -> None:
        if self.is_training:
            return
        if not self._apply_grid_and_learning():
            return

        episodes = int(self.episodes_var.get())
        if episodes <= 0:
            messagebox.showerror("Invalid episodes", "Number of episodes must be positive.")
            return

        self.is_training = True
        self.stop_training_requested = False
        self.training_target_episodes = episodes
        self.training_completed_episodes = 0
        self._set_controls_busy(True)
        self.stop_button.configure(state="normal")

        self.after(0, self._training_start_episode)

    def _training_start_episode(self) -> None:
        if self.training_completed_episodes >= self.training_target_episodes:
            self._finish_training("Training completed")
            return

        self._start_episode_if_needed()
        self.after(0, self._training_step_tick)

    def _training_step_tick(self) -> None:
        _prev, _action, _next, _reward, done = self._advance_one_step()

        visualize_enabled = bool(self.visualize_training_var.get())
        every_k = max(1, int(self.visualize_every_var.get()))
        should_visualize_episode = visualize_enabled and (self.training_completed_episodes % every_k == 0)

        if self._episode_should_stop(done):
            self._close_episode()
            self.training_completed_episodes += 1
            self._refresh_status(
                f"Training episode {self.training_completed_episodes}/{self.training_target_episodes} done | "
                f"Reward {self.reward_history[-1]}"
            )

            if self.stop_training_requested:
                self._finish_training("Training stopped by user")
                return

            self.after(1, self._training_start_episode)
            return

        if self.stop_training_requested:
            self._close_episode()
            self.training_completed_episodes += 1
            self._finish_training("Training stopped by user")
            return

        delay = max(1, int(self.delay_ms_var.get())) if should_visualize_episode else 1
        self.after(delay, self._training_step_tick)

    def _request_stop_training(self) -> None:
        if self.is_training:
            self.stop_training_requested = True
            self._refresh_status("Stop requested. Finishing current step/episode gracefully...")

    def _finish_training(self, status: str) -> None:
        self.is_training = False
        self.stop_training_requested = False
        self.stop_button.configure(state="disabled")
        self._set_controls_busy(False)
        self._refresh_status(status)

    def _set_controls_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.run_step_button.configure(state=state)
        self.run_episode_button.configure(state=state)
        self.train_button.configure(state=state)
        self.apply_grid_button.configure(state=state)
        self.policy_combo.configure(state="disabled" if busy else "readonly")

    def _update_reward_plot(self) -> None:
        x_data = list(range(1, len(self.reward_history) + 1))
        self.reward_line.set_data(x_data, self.reward_history)
        self.plot_axis.relim()
        self.plot_axis.autoscale_view()
        self.plot_canvas.draw_idle()

    def _refresh_status(self, text: str) -> None:
        self.status_var.set(text)

    def _show_value_table(self) -> None:
        table = self.lab.get_value_table()
        if not table:
            messagebox.showinfo("Value table", "No Monte Carlo values learned yet.")
            return

        top = tk.Toplevel(self)
        top.title("Monte Carlo Value Table")
        text = tk.Text(top, width=42, height=18)
        text.pack(fill="both", expand=True)
        for state, value in sorted(table.items()):
            text.insert("end", f"V{state} = {value:.4f}\n")

    def _show_q_table(self) -> None:
        table = self.lab.get_q_table()
        if not table:
            messagebox.showinfo("Q-table", "No Q-values learned yet.")
            return

        top = tk.Toplevel(self)
        top.title("Q-Table")
        text = tk.Text(top, width=62, height=20)
        text.pack(fill="both", expand=True)
        for (state, action), value in sorted(table.items()):
            text.insert("end", f"Q({state}, {action}) = {value:.4f}\n")

    def _save_sampling_csv(self) -> None:
        try:
            output_dir = Path(__file__).resolve().parent / "plots"
            csv_path = self.lab.export_samplings_csv(output_dir=output_dir, episodes=max(1, int(self.episodes_var.get())))
        except Exception as exc:
            messagebox.showerror("CSV export failed", str(exc))
            return
        messagebox.showinfo("CSV exported", f"Saved: {csv_path}")

    def _reset_session(self) -> None:
        if self.is_training:
            return
        self.reward_history.clear()
        self._update_reward_plot()
        self.lab.current_episode_index = 0
        self.episode_open = False
        self.agent_position = self.lab.environment.reset()
        self._draw_grid()
        self._refresh_status("Session reset")


def launch_gui(lab: Optional[GridWorldLab] = None) -> None:
    app = GridWorldGUI(lab if lab is not None else GridWorldLab())
    app.mainloop()
