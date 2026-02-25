import threading
import time
from datetime import datetime
from pathlib import Path
from tkinter import BooleanVar, DoubleVar, IntVar, StringVar, TclError, Tk, ttk

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from CartPole_logic import (
    AgentConfig,
    CartPoleEnvironment,
    D3QN,
    DoubleDQN,
    DuelingDQN,
    Trainer,
    moving_average,
    set_global_seed,
)

try:
    from PIL import Image, ImageTk

    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False


class CartPoleGUI:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("CartPole RL - DQN Variants")
        self.base_dir = Path(__file__).resolve().parent
        self.plots_dir = self.base_dir / "plots"
        self.csv_dir = self.base_dir / "results_csv"
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        set_global_seed(42)
        self.env = CartPoleEnvironment(sutton_barto_reward=False, seed=42)
        self.trainer = Trainer(self.env, output_dir=self.base_dir)
        self.agent_instance = None
        self._stop_requested = False
        self._is_training = False
        self._last_plot_update = 0.0
        self._legend_item_map = {}
        self._legend = None
        self._plot_runs = []
        self._last_photo = None
        self._plot_generation = 0
        self._pending_lock = threading.Lock()
        self._pending_counter = None
        self._pending_frame_refresh = False
        self._pending_plot = None
        self._pending_finalize_run = None
        self._pending_training_done = False

        self.current_episode = 0
        self.current_step = 0

        self._init_variables()
        self._build_layout()
        self._refresh_environment_frame()
        self._update_state_label(training=False)
        self.root.after(33, self._ui_pump)

    def _init_variables(self) -> None:
        self.policy_var = StringVar(value="DoubleDQN")
        self.max_steps_var = IntVar(value=500)
        self.episodes_var = IntVar(value=150)
        self.moving_avg_var = IntVar(value=20)
        self.animation_refresh_var = IntVar(value=10)
        self.live_plot_var = BooleanVar(value=True)
        self.reduced_speed_var = BooleanVar(value=True)
        self.sutton_barto_var = BooleanVar(value=False)

        self.eps_min_var = DoubleVar(value=0.05)
        self.eps_max_var = DoubleVar(value=1.0)
        self.eps_decay_var = DoubleVar(value=0.995)

        self.lr_var = DoubleVar(value=0.001)
        self.gamma_var = DoubleVar(value=0.99)
        self.batch_size_var = IntVar(value=64)
        self.replay_size_var = IntVar(value=50000)
        self.target_update_var = IntVar(value=200)
        self.hidden_size_var = StringVar(value="128")

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=0)
        self.root.columnconfigure(2, weight=0)
        self.root.rowconfigure(3, weight=1)

        self.env_frame = ttk.LabelFrame(self.root, text="Environment")
        self.env_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        self.env_frame.columnconfigure(0, weight=1)

        self.image_label = ttk.Label(self.env_frame, text="Environment frame unavailable")
        self.image_label.grid(row=0, column=0, padx=6, pady=6)

        sutton_toggle = ttk.Checkbutton(
            self.env_frame,
            text="sutton_barto_reward",
            variable=self.sutton_barto_var,
            command=self._on_toggle_sutton_barto,
        )
        sutton_toggle.grid(row=1, column=0, sticky="w", padx=6, pady=(0, 6))

        self.controls_frame = ttk.LabelFrame(self.root, text="Controls")
        self.controls_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
        for c in range(2):
            self.controls_frame.columnconfigure(c, weight=1)

        ttk.Button(self.controls_frame, text="Reset All", command=self._reset_all).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self.controls_frame, text="Clear Plot", command=self._clear_plot).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self.controls_frame, text="Run single episode", command=self._run_single_episode).grid(
            row=1, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self.controls_frame, text="Save samplings CSV", command=self._save_samplings_csv).grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self.controls_frame, text="Train and Run", command=self._train_and_run).grid(
            row=2, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(self.controls_frame, text="Save Plot PNG", command=self._save_plot_png).grid(
            row=2, column=1, sticky="ew", padx=4, pady=4
        )

        self.state_frame = ttk.LabelFrame(self.root, text="Current State")
        self.state_frame.grid(row=2, column=0, sticky="ew", padx=6, pady=6)
        self.state_label = ttk.Label(self.state_frame, text="")
        self.state_label.grid(row=0, column=0, sticky="w", padx=6, pady=6)

        self.dnn_frame = ttk.LabelFrame(self.root, text="DNN Parameters")
        self.dnn_frame.grid(row=1, column=1, rowspan=2, sticky="ns", padx=6, pady=6)
        self._build_dnn_inputs(self.dnn_frame)

        self.training_frame = ttk.LabelFrame(self.root, text="Training Parameters")
        self.training_frame.grid(row=1, column=2, rowspan=2, sticky="ns", padx=6, pady=6)
        self._build_training_inputs(self.training_frame)

        self.plot_frame = ttk.LabelFrame(self.root, text="Live Plot")
        self.plot_frame.grid(row=3, column=0, columnspan=3, sticky="nsew", padx=6, pady=6)
        self.plot_frame.columnconfigure(0, weight=1)
        self.plot_frame.rowconfigure(0, weight=1)

        self.figure, self.ax = plt.subplots(figsize=(9, 4))
        self.ax.set_title("Episode rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.canvas.mpl_connect("pick_event", self._on_legend_pick)

    def _build_training_inputs(self, parent: ttk.LabelFrame) -> None:
        row = 0
        self._add_labeled_entry(parent, "max steps", self.max_steps_var, row)
        row += 1
        self._add_labeled_entry(parent, "episodes", self.episodes_var, row)
        row += 1

        ttk.Label(parent, text="policy").grid(row=row, column=0, sticky="w", padx=6, pady=3)
        policy_combo = ttk.Combobox(
            parent,
            textvariable=self.policy_var,
            values=["DoubleDQN", "DuelingDQN", "D3QN"],
            state="readonly",
            width=14,
        )
        policy_combo.grid(row=row, column=1, sticky="w", padx=6, pady=3)
        row += 1

        self._add_labeled_entry(parent, "moving average", self.moving_avg_var, row)
        row += 1
        self._add_labeled_entry(parent, "anim refresh", self.animation_refresh_var, row)
        row += 1
        self._add_labeled_entry(parent, "eps min", self.eps_min_var, row)
        row += 1
        self._add_labeled_entry(parent, "eps max", self.eps_max_var, row)
        row += 1
        self._add_labeled_entry(parent, "eps decay", self.eps_decay_var, row)
        row += 1

        checks = ttk.Frame(parent)
        checks.grid(row=row, column=0, columnspan=2, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(checks, text="Live plot", variable=self.live_plot_var).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(checks, text="reduced speed", variable=self.reduced_speed_var).grid(
            row=0, column=1, sticky="w", padx=(8, 0)
        )

    def _build_dnn_inputs(self, parent: ttk.LabelFrame) -> None:
        row = 0
        self._add_labeled_entry(parent, "learning rate", self.lr_var, row)
        row += 1
        self._add_labeled_entry(parent, "gamma", self.gamma_var, row)
        row += 1
        self._add_labeled_entry(parent, "batch size", self.batch_size_var, row)
        row += 1
        self._add_labeled_entry(parent, "replay size", self.replay_size_var, row)
        row += 1
        self._add_labeled_entry(parent, "target update", self.target_update_var, row)
        row += 1
        self._add_labeled_entry(parent, "hidden size(s)", self.hidden_size_var, row)

    def _add_labeled_entry(self, parent: ttk.Frame, label: str, variable, row: int) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6, pady=3)
        ttk.Entry(parent, textvariable=variable, width=14).grid(row=row, column=1, sticky="w", padx=6, pady=3)

    def _build_agent(self):
        hidden_layers_text = str(self.hidden_size_var.get()).strip()
        hidden_layers = tuple(
            max(1, int(part.strip()))
            for part in hidden_layers_text.split(",")
            if part.strip()
        )
        if not hidden_layers:
            hidden_layers = (128,)

        config = AgentConfig(
            gamma=float(self.gamma_var.get()),
            learning_rate=float(self.lr_var.get()),
            batch_size=int(self.batch_size_var.get()),
            replay_size=int(self.replay_size_var.get()),
            min_replay_size=min(1000, max(128, int(self.batch_size_var.get()))),
            target_update_every=int(self.target_update_var.get()),
            hidden_layers=hidden_layers,
        )

        state_size = self.env.state_size
        action_size = self.env.action_size
        chosen = self.policy_var.get()
        if chosen == "DoubleDQN":
            return DoubleDQN(state_size, action_size, config=config)
        if chosen == "DuelingDQN":
            return DuelingDQN(state_size, action_size, config=config)
        return D3QN(state_size, action_size, config=config)

    def _on_toggle_sutton_barto(self) -> None:
        self.env.close()
        self.env = CartPoleEnvironment(
            sutton_barto_reward=bool(self.sutton_barto_var.get()),
            seed=42,
        )
        self.trainer = Trainer(self.env, output_dir=self.base_dir)
        self.agent_instance = None
        self._refresh_environment_frame()

    def _format_state_text(self, training: bool, step: int, episode: int) -> str:
        prefix = "Training:" if training else "Idle:"
        return f"{prefix:>9} step:{step:>5}  episode:{episode:>5}"

    def _update_state_label(self, training: bool) -> None:
        self.state_label.config(
            text=self._format_state_text(training=training, step=self.current_step, episode=self.current_episode)
        )

    def _set_current_counters(self, episode: int, step: int, training: bool = True) -> None:
        self.current_episode = episode
        self.current_step = step
        self._update_state_label(training=training)

    def _safe_after(self, callback) -> None:
        try:
            self.root.after(0, callback)
        except (RuntimeError, TclError):
            pass

    def _ui_pump(self) -> None:
        try:
            with self._pending_lock:
                pending_counter = self._pending_counter
                self._pending_counter = None

                pending_frame_refresh = self._pending_frame_refresh
                self._pending_frame_refresh = False

                pending_plot = self._pending_plot
                self._pending_plot = None

                pending_finalize_run = self._pending_finalize_run
                self._pending_finalize_run = None

                pending_training_done = self._pending_training_done
                self._pending_training_done = False

            if pending_counter is not None:
                ep, st, training = pending_counter
                self._set_current_counters(ep, st, training)

            if pending_frame_refresh:
                self._refresh_environment_frame()

            if pending_plot is not None:
                rewards, generation = pending_plot
                self._update_live_plot(rewards, generation)

            if pending_finalize_run is not None:
                rewards, run_name, generation = pending_finalize_run
                self._register_run(rewards, run_name, color=None, force_draw=True, generation=generation)

            if pending_training_done:
                self._mark_training_finished()
        finally:
            try:
                self.root.after(33, self._ui_pump)
            except (RuntimeError, TclError):
                pass

    def _refresh_environment_frame(self) -> None:
        frame = self.env.render_frame()
        if frame is None:
            return
        if not PIL_AVAILABLE:
            self.image_label.config(text="Pillow not available; install pillow to show frames")
            return

        image = Image.fromarray(frame)
        image = image.resize((480, 320))
        photo = ImageTk.PhotoImage(image)
        self._last_photo = photo
        self.image_label.config(image=photo, text="")

    def _run_single_episode(self) -> None:
        if self._is_training:
            return
        self._stop_requested = False
        self._is_training = True
        if self.agent_instance is None:
            self.agent_instance = self._build_agent()

        self.current_episode = 1
        self.current_step = 0
        self._update_state_label(training=True)

        eps = float(self.eps_min_var.get())
        animation_refresh_every = max(1, int(self.animation_refresh_var.get()))

        def on_step(step: int) -> None:
            self.current_step = step
            self._update_state_label(training=True)
            if step == 1 or step % animation_refresh_every == 0:
                self._refresh_environment_frame()
                self.root.update_idletasks()
                time.sleep(0.02)

        reward, _ = self.trainer.run_episode(
            self.agent_instance,
            epsilon=eps,
            max_steps=int(self.max_steps_var.get()),
            progress_callback=on_step,
        )

        self._register_run(
            rewards=[reward],
            run_name=self._run_label(),
            color=None,
            force_draw=True,
        )
        self._is_training = False
        self._update_state_label(training=False)

    def _train_and_run(self) -> None:
        if self._is_training:
            return

        self._stop_requested = False
        self._is_training = True
        self.agent_instance = self._build_agent()

        episodes = int(self.episodes_var.get())
        max_steps = int(self.max_steps_var.get())
        eps_min = float(self.eps_min_var.get())
        eps_max = float(self.eps_max_var.get())
        eps_decay = float(self.eps_decay_var.get())
        live_plot_enabled = bool(self.live_plot_var.get())
        reduced_speed_enabled = bool(self.reduced_speed_var.get())
        animation_refresh_every = max(1, int(self.animation_refresh_var.get()))
        run_generation = self._plot_generation
        run_label = self._run_label()

        if eps_decay <= 0.0:
            eps_decay = 1.0

        def worker() -> None:
            rewards = []
            for ep in range(1, episodes + 1):
                if self._stop_requested:
                    break

                epsilon = max(eps_min, eps_max * (eps_decay ** (ep - 1)))

                with self._pending_lock:
                    self._pending_counter = (ep, 0, True)

                def step_cb(step: int, episode=ep) -> None:
                    with self._pending_lock:
                        self._pending_counter = (episode, step, True)
                    if step == 1 or step % animation_refresh_every == 0:
                        with self._pending_lock:
                            self._pending_frame_refresh = True

                total_reward, _ = self.trainer.run_episode(
                    self.agent_instance,
                    epsilon=epsilon,
                    max_steps=max_steps,
                    progress_callback=step_cb,
                )
                rewards.append(total_reward)

                if live_plot_enabled:
                    with self._pending_lock:
                        self._pending_plot = (rewards.copy(), run_generation)

                with self._pending_lock:
                    self._pending_frame_refresh = True

                if reduced_speed_enabled:
                    time.sleep(0.033)

            with self._pending_lock:
                self._pending_finalize_run = (rewards.copy(), run_label, run_generation)
                self._pending_training_done = True

        threading.Thread(target=worker, daemon=True).start()

    def _mark_training_finished(self) -> None:
        self._is_training = False
        self._update_state_label(training=False)

    def _update_live_plot(self, rewards, generation: int):
        if generation != self._plot_generation:
            return
        now = time.time()
        if now - self._last_plot_update < 0.15:
            return
        self._last_plot_update = now

        if not hasattr(self, "_live_reward_line"):
            color = f"C{len(self._plot_runs) % 10}"
            (self._live_reward_line,) = self.ax.plot([], [], color=color, linewidth=1.0, alpha=0.45)
            (self._live_ma_line,) = self.ax.plot([], [], color=color, linewidth=2.2, alpha=1.0)

        x = list(range(1, len(rewards) + 1))
        ma_values = moving_average(rewards, max(1, int(self.moving_avg_var.get())))
        self._live_reward_line.set_data(x, rewards)
        self._live_ma_line.set_data(x, ma_values)
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _register_run(self, rewards, run_name: str, color=None, force_draw: bool = False, generation: int = None) -> None:
        if generation is not None and generation != self._plot_generation:
            return
        if not rewards:
            self._clear_live_lines()
            if force_draw:
                self.canvas.draw_idle()
            return

        if color is None:
            color = f"C{len(self._plot_runs) % 10}"

        x = list(range(1, len(rewards) + 1))
        ma_values = moving_average(rewards, max(1, int(self.moving_avg_var.get())))
        reward_label = f"{run_name} | reward"
        ma_label = f"{run_name} | ma"

        reward_line, = self.ax.plot(x, rewards, color=color, linewidth=1.0, alpha=0.45, label=reward_label)
        ma_line, = self.ax.plot(x, ma_values, color=color, linewidth=2.2, alpha=1.0, label=ma_label)
        self._plot_runs.append({"reward": reward_line, "ma": ma_line})

        self._clear_live_lines()
        self._rebuild_interactive_legend()
        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw_idle()

    def _clear_live_lines(self) -> None:
        if hasattr(self, "_live_reward_line"):
            try:
                self._live_reward_line.remove()
            except (ValueError, NotImplementedError):
                pass
            del self._live_reward_line
        if hasattr(self, "_live_ma_line"):
            try:
                self._live_ma_line.remove()
            except (ValueError, NotImplementedError):
                pass
            del self._live_ma_line

    def _rebuild_interactive_legend(self) -> None:
        self._legend_item_map.clear()
        self._legend = self.ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0)
        if self._legend is None:
            return

        legend_lines = self._legend.get_lines()
        legend_texts = self._legend.get_texts()
        plot_lines = self.ax.get_lines()

        for idx, plot_line in enumerate(plot_lines):
            if idx < len(legend_lines):
                proxy = legend_lines[idx]
                proxy.set_picker(True)
                self._legend_item_map[proxy] = plot_line
                self._set_legend_proxy_style(proxy, plot_line.get_visible())

            if idx < len(legend_texts):
                txt = legend_texts[idx]
                txt.set_picker(True)
                self._legend_item_map[txt] = plot_line
                txt.set_alpha(1.0 if plot_line.get_visible() else 0.35)

    def _set_legend_proxy_style(self, proxy_line, visible: bool) -> None:
        proxy_line.set_alpha(1.0 if visible else 0.25)

    def _on_legend_pick(self, event) -> None:
        artist = event.artist
        if artist not in self._legend_item_map:
            return

        plot_line = self._legend_item_map[artist]
        new_visible = not plot_line.get_visible()
        plot_line.set_visible(new_visible)

        if self._legend is not None:
            for proxy in self._legend.get_lines():
                linked = self._legend_item_map.get(proxy)
                if linked is plot_line:
                    self._set_legend_proxy_style(proxy, new_visible)
            for txt in self._legend.get_texts():
                linked = self._legend_item_map.get(txt)
                if linked is plot_line:
                    txt.set_alpha(1.0 if new_visible else 0.35)

        self.canvas.draw_idle()

    def _clear_plot(self) -> None:
        self._plot_generation += 1
        self._plot_runs.clear()
        self._legend_item_map.clear()
        self._legend = None
        self._clear_live_lines()
        self.ax.clear()
        self.ax.set_title("Episode rewards")
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.canvas.draw_idle()

    def _reset_all(self) -> None:
        self._stop_requested = True
        self._is_training = False
        self._clear_plot()
        self.agent_instance = None
        self.current_episode = 0
        self.current_step = 0
        self.env.reset()
        self._refresh_environment_frame()
        self._update_state_label(training=False)

    def _save_samplings_csv(self) -> None:
        if self.agent_instance is None:
            self.agent_instance = self._build_agent()
        policy_name = self.policy_var.get()
        base = f"samplings_{policy_name}"
        self.trainer.train(
            policy=self.agent_instance,
            num_episodes=max(1, int(self.episodes_var.get())),
            max_steps=max(1, int(self.max_steps_var.get())),
            epsilon=float(self.eps_min_var.get()),
            save_csv=base,
        )

    def _run_label(self) -> str:
        return (
            f"{self.policy_var.get()}"
            f"-eps{self.eps_min_var.get():.2f}-{self.eps_max_var.get():.2f}"
            f"-dec{self.eps_decay_var.get():.3f}"
            f"-lr{self.lr_var.get():.4f}"
            f"-g{self.gamma_var.get():.2f}"
        )

    def _save_plot_png(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"{self.policy_var.get()}"
            f"_eps{self.eps_min_var.get():.2f}-{self.eps_max_var.get():.2f}"
            f"_epsdecay{self.eps_decay_var.get():.4f}"
            f"_alpha{self.lr_var.get():.4f}"
            f"_gamma{self.gamma_var.get():.2f}"
            f"_episodes{self.episodes_var.get()}"
            f"_max_steps{self.max_steps_var.get()}"
            f"_{timestamp}.png"
        )
        out = self.plots_dir / filename
        self.figure.savefig(out, bbox_inches="tight")
