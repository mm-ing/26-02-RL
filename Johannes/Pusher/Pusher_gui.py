from __future__ import annotations

import os
import queue
import threading
import uuid
from dataclasses import replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("TkAgg")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
import tkinter as tk
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk
except Exception:  # pragma: no cover - optional dependency
    Image = None
    ImageTk = None

from Pusher_logic import (
    EnvironmentConfig,
    LearningRateConfig,
    NetworkConfig,
    SB3PolicyFactory,
    TrainerConfig,
    build_default_trainer,
)


class PusherGUI:
    POLICIES = ["PPO", "SAC", "TD3", "DDPG"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pusher RL Workbench")
        self.root.geometry("1320x860")

        self.trainer = build_default_trainer(event_sink=self._on_worker_event)
        self.worker_thread: Optional[threading.Thread] = None

        self._session_id = ""
        self._run_id = ""
        self._queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._queue_lock = threading.Lock()

        self._history_reward: List[float] = []
        self._history_mavg: List[float] = []
        self._history_eval: List[Tuple[int, float]] = []

        self._anim_active = False
        self._anim_pending: Optional[List[np.ndarray]] = None
        self._anim_frames: List[np.ndarray] = []
        self._anim_idx = 0
        self._render_photo = None

        self.policy_param_snapshots: Dict[str, Dict[str, str]] = {}

        self._build_layout()
        self._init_policy_snapshots()
        self.root.after(100, self._pump_worker_events)

    def _build_layout(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(outer)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(outer)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_environment_group(left)
        self._build_runtime_group(left)
        self._build_specific_group(left)
        self._build_control_group(left)

        self._build_plot_group(right)
        self._build_render_group(right)

    def _build_environment_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Environment", padding=10)
        frame.pack(fill=tk.X, pady=6)

        ttk.Label(frame, text="Env ID").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Label(frame, text="Pusher-v5").grid(row=0, column=1, sticky="w", padx=4, pady=2)

        self.reward_near_var = tk.StringVar(value="0.5")
        self.reward_dist_var = tk.StringVar(value="1.0")
        self.reward_ctrl_var = tk.StringVar(value="0.1")

        ttk.Button(frame, text="Update Environment", command=self._update_environment).grid(
            row=1,
            column=0,
            columnspan=2,
            sticky="ew",
            padx=4,
            pady=6,
        )

        ttk.Label(frame, text="reward_near_weight").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.reward_near_var, width=14).grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(frame, text="reward_dist_weight").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.reward_dist_var, width=14).grid(row=3, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(frame, text="reward_control_weight").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(frame, textvariable=self.reward_ctrl_var, width=14).grid(row=4, column=1, sticky="w", padx=4, pady=2)

    def _build_runtime_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Runtime", padding=10)
        frame.pack(fill=tk.X, pady=6)

        self.policy_var = tk.StringVar(value="SAC")
        self.policy_var.trace_add("write", self._on_policy_change)

        self.episodes_var = tk.StringVar(value="3000")
        self.max_steps_var = tk.StringVar(value="200")
        self.update_rate_var = tk.StringVar(value="5")
        self.frame_stride_var = tk.StringVar(value="2")
        self.device_var = tk.StringVar(value="CPU")
        self.seed_var = tk.StringVar(value="42")
        self.export_csv_var = tk.BooleanVar(value=False)
        self.collect_transitions_var = tk.BooleanVar(value=False)

        fields = [
            ("Policy", ttk.Combobox(frame, textvariable=self.policy_var, values=self.POLICIES, state="readonly", width=12)),
            ("Episodes", ttk.Entry(frame, textvariable=self.episodes_var, width=14)),
            ("Max steps", ttk.Entry(frame, textvariable=self.max_steps_var, width=14)),
            ("Update rate (episodes)", ttk.Entry(frame, textvariable=self.update_rate_var, width=14)),
            ("Frame stride", ttk.Entry(frame, textvariable=self.frame_stride_var, width=14)),
            ("Device", ttk.Combobox(frame, textvariable=self.device_var, values=["CPU", "GPU"], state="readonly", width=12)),
            ("Seed", ttk.Entry(frame, textvariable=self.seed_var, width=14)),
        ]

        for idx, (label, widget) in enumerate(fields):
            ttk.Label(frame, text=label).grid(row=idx, column=0, sticky="w", padx=4, pady=2)
            widget.grid(row=idx, column=1, sticky="w", padx=4, pady=2)

        ttk.Checkbutton(frame, text="Collect transitions", variable=self.collect_transitions_var).grid(
            row=len(fields),
            column=0,
            columnspan=2,
            sticky="w",
            padx=4,
            pady=2,
        )
        ttk.Checkbutton(frame, text="Export transition CSV", variable=self.export_csv_var).grid(
            row=len(fields) + 1,
            column=0,
            columnspan=2,
            sticky="w",
            padx=4,
            pady=2,
        )

    def _build_specific_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Specific", padding=10)
        frame.pack(fill=tk.BOTH, expand=False, pady=6)

        self.shared_specific_vars = {
            "gamma": tk.StringVar(value="0.99"),
            "learning_rate": tk.StringVar(value="0.0003"),
            "batch_size": tk.StringVar(value="256"),
            "hidden_layer": tk.StringVar(value="256"),
            "activation": tk.StringVar(value="relu"),
            "lr_strategy": tk.StringVar(value="constant"),
            "min_lr": tk.StringVar(value="1e-05"),
            "lr_decay": tk.StringVar(value="0.999"),
        }

        row_idx = 0
        for key in ["gamma", "learning_rate", "batch_size", "hidden_layer", "activation", "lr_strategy", "min_lr", "lr_decay"]:
            ttk.Label(frame, text=key).grid(row=row_idx, column=0, sticky="w", padx=4, pady=2)
            ttk.Entry(frame, textvariable=self.shared_specific_vars[key], width=16).grid(
                row=row_idx,
                column=1,
                sticky="w",
                padx=4,
                pady=2,
            )
            row_idx += 1

        ttk.Separator(frame).grid(row=row_idx, column=0, columnspan=2, sticky="ew", pady=6)
        row_idx += 1

        self.policy_specific_container = ttk.Frame(frame)
        self.policy_specific_container.grid(row=row_idx, column=0, columnspan=2, sticky="ew")

        self.policy_specific_widgets: Dict[str, Dict[str, tk.StringVar]] = {
            "PPO": {
                "n_steps": tk.StringVar(value="1024"),
                "ent_coef": tk.StringVar(value="0.0"),
            },
            "SAC": {
                "buffer_size": tk.StringVar(value="200000"),
                "learning_starts": tk.StringVar(value="5000"),
                "tau": tk.StringVar(value="0.005"),
            },
            "TD3": {
                "buffer_size": tk.StringVar(value="200000"),
                "learning_starts": tk.StringVar(value="5000"),
                "policy_delay": tk.StringVar(value="2"),
            },
            "DDPG": {
                "buffer_size": tk.StringVar(value="200000"),
                "learning_starts": tk.StringVar(value="5000"),
                "tau": tk.StringVar(value="0.005"),
            },
        }

        self.policy_specific_frames: Dict[str, ttk.Frame] = {}
        for policy in self.POLICIES:
            p_frame = ttk.Frame(self.policy_specific_container)
            p_frame.grid(row=0, column=0, sticky="ew")
            self.policy_specific_frames[policy] = p_frame
            line = 0
            for key, var in self.policy_specific_widgets[policy].items():
                ttk.Label(p_frame, text=key).grid(row=line, column=0, sticky="w", padx=4, pady=2)
                ttk.Entry(p_frame, textvariable=var, width=16).grid(row=line, column=1, sticky="w", padx=4, pady=2)
                line += 1

        self._show_policy_specific("SAC")

    def _build_control_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Controls", padding=10)
        frame.pack(fill=tk.X, pady=6)

        ttk.Button(frame, text="Train and Run", command=self._start_training).grid(row=0, column=0, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Pause", command=self._pause).grid(row=0, column=1, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Resume", command=self._resume).grid(row=0, column=2, sticky="ew", padx=4, pady=2)
        ttk.Button(frame, text="Cancel", command=self._cancel).grid(row=0, column=3, sticky="ew", padx=4, pady=2)

        self.status_var = tk.StringVar(value="Idle")
        ttk.Label(frame, textvariable=self.status_var).grid(row=1, column=0, columnspan=4, sticky="w", padx=4, pady=2)

    def _build_plot_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Training Curve", padding=8)
        frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.figure = Figure(figsize=(8.5, 4.5), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")
        self.ax.grid(True, alpha=0.3)

        self.plot_canvas = FigureCanvasTkAgg(self.figure, master=frame)
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_render_group(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Animation Preview", padding=8)
        frame.pack(fill=tk.BOTH, expand=True, pady=6)

        self.render_canvas = tk.Canvas(frame, width=640, height=280, bg="#111111", highlightthickness=0)
        self.render_canvas.pack(fill=tk.BOTH, expand=True)

    def _init_policy_snapshots(self) -> None:
        for policy in self.POLICIES:
            self.policy_param_snapshots[policy] = self._read_current_specific_vars(policy)
        self._apply_policy_defaults("SAC")

    def _on_policy_change(self, *_: Any) -> None:
        current = self.policy_var.get()
        for policy in self.POLICIES:
            if policy != current:
                self.policy_param_snapshots[policy] = self._read_current_specific_vars(policy)
        self._show_policy_specific(current)
        snapshot = self.policy_param_snapshots.get(current)
        if snapshot:
            self._write_specific_vars(current, snapshot)
        else:
            self._apply_policy_defaults(current)

    def _read_current_specific_vars(self, policy: str) -> Dict[str, str]:
        values = {k: v.get() for k, v in self.shared_specific_vars.items()}
        values.update({k: v.get() for k, v in self.policy_specific_widgets[policy].items()})
        return values

    def _write_specific_vars(self, policy: str, values: Dict[str, str]) -> None:
        for key, var in self.shared_specific_vars.items():
            if key in values:
                var.set(str(values[key]))
        for key, var in self.policy_specific_widgets[policy].items():
            if key in values:
                var.set(str(values[key]))

    def _apply_policy_defaults(self, policy: str) -> None:
        defaults = SB3PolicyFactory.policy_defaults(policy)
        self.shared_specific_vars["learning_rate"].set(str(defaults.get("learning_rate", 0.0003)))
        self.shared_specific_vars["gamma"].set(str(defaults.get("gamma", 0.99)))
        self.shared_specific_vars["batch_size"].set(str(defaults.get("batch_size", 256)))

        for key, var in self.policy_specific_widgets[policy].items():
            if key in defaults:
                var.set(str(defaults[key]))

    def _show_policy_specific(self, policy: str) -> None:
        for name, frame in self.policy_specific_frames.items():
            if name == policy:
                frame.tkraise()

    def _update_environment(self) -> None:
        try:
            cfg = EnvironmentConfig(
                reward_near_weight=float(self.reward_near_var.get()),
                reward_dist_weight=float(self.reward_dist_var.get()),
                reward_control_weight=float(self.reward_ctrl_var.get()),
                render_mode="rgb_array",
            )
            self.trainer.update_environment(cfg)
            self.status_var.set("Environment updated")
        except ValueError:
            messagebox.showerror("Invalid input", "Environment parameters must be numeric.")

    def _build_trainer_config(self) -> TrainerConfig:
        policy = self.policy_var.get()

        lr_cfg = LearningRateConfig(
            learning_rate=float(self.shared_specific_vars["learning_rate"].get()),
            lr_strategy=self.shared_specific_vars["lr_strategy"].get(),
            min_lr=float(self.shared_specific_vars["min_lr"].get()),
            lr_decay=float(self.shared_specific_vars["lr_decay"].get()),
        )
        network_cfg = NetworkConfig(
            hidden_layer=self.shared_specific_vars["hidden_layer"].get(),
            activation=self.shared_specific_vars["activation"].get(),
        )

        policy_params: Dict[str, Any] = {
            "gamma": float(self.shared_specific_vars["gamma"].get()),
            "batch_size": int(self.shared_specific_vars["batch_size"].get()),
        }

        for key, value in self.policy_specific_widgets[policy].items():
            raw = value.get()
            if raw.strip() == "":
                continue
            try:
                if "." in raw or "e" in raw.lower():
                    policy_params[key] = float(raw)
                else:
                    policy_params[key] = int(raw)
            except ValueError:
                policy_params[key] = raw

        env_cfg = EnvironmentConfig(
            reward_near_weight=float(self.reward_near_var.get()),
            reward_dist_weight=float(self.reward_dist_var.get()),
            reward_control_weight=float(self.reward_ctrl_var.get()),
            render_mode="rgb_array",
        )

        seed_value = self.seed_var.get().strip()
        seed = int(seed_value) if seed_value else None

        return TrainerConfig(
            policy_name=policy,
            episodes=int(self.episodes_var.get()),
            max_steps=int(self.max_steps_var.get()),
            update_rate=int(self.update_rate_var.get()),
            frame_stride=int(self.frame_stride_var.get()),
            seed=seed,
            collect_transitions=bool(self.collect_transitions_var.get()),
            export_csv=bool(self.export_csv_var.get()),
            device=self.device_var.get(),
            session_id=self._session_id,
            run_id=self._run_id,
            env=env_cfg,
            network=network_cfg,
            lr=lr_cfg,
            policy_params=policy_params,
        )

    def _start_training(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo("Training active", "A training run is already in progress.")
            return

        try:
            self._session_id = str(uuid.uuid4())
            self._run_id = datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
            cfg = self._build_trainer_config()
            cfg = replace(cfg, session_id=self._session_id, run_id=self._run_id)
        except ValueError:
            messagebox.showerror("Invalid input", "Please check runtime and specific parameters.")
            return

        self._history_reward.clear()
        self._history_mavg.clear()
        self._history_eval.clear()
        self._anim_pending = None
        self._anim_frames = []
        self._anim_idx = 0

        self.status_var.set(f"Training: {cfg.policy_name} on {cfg.device}")

        self.worker_thread = threading.Thread(target=self._train_worker, args=(cfg,), daemon=True)
        self.worker_thread.start()

    def _train_worker(self, config: TrainerConfig) -> None:
        self.trainer.update_environment(config.env)
        self.trainer.train(config)

    def _pause(self) -> None:
        self.trainer.pause()
        self.status_var.set("Paused")

    def _resume(self) -> None:
        self.trainer.resume()
        self.status_var.set("Running")

    def _cancel(self) -> None:
        self.trainer.cancel()
        self.status_var.set("Cancel requested")

    def _on_worker_event(self, payload: Dict[str, Any]) -> None:
        with self._queue_lock:
            self._queue.put(payload)

    def _pump_worker_events(self) -> None:
        events: List[Dict[str, Any]] = []
        with self._queue_lock:
            while not self._queue.empty():
                events.append(self._queue.get())

        for payload in events:
            if payload.get("session_id") != self._session_id:
                continue
            event_type = payload.get("type")
            if event_type == "episode":
                self._handle_episode_event(payload)
            elif event_type == "training_done":
                self._handle_training_done(payload)
            elif event_type == "error":
                self._handle_error(payload)

        self.root.after(100, self._pump_worker_events)

    def _handle_episode_event(self, payload: Dict[str, Any]) -> None:
        episode = int(payload.get("episode", 0))
        episodes = int(payload.get("episodes", 0))
        reward = float(payload.get("reward", 0.0))
        moving_average = float(payload.get("moving_average", reward))

        self._history_reward.append(reward)
        self._history_mavg.append(moving_average)
        self._history_eval = list(payload.get("eval_points", self._history_eval))

        self.status_var.set(
            f"Episode {episode}/{episodes} | reward={reward:.2f} | moving_avg={moving_average:.2f}"
        )

        self._refresh_plot()

        frames = payload.get("frames")
        if isinstance(frames, list) and frames:
            if self._anim_active:
                self._anim_pending = frames
            else:
                self._start_animation(frames)

        render_state = payload.get("render_state")
        if isinstance(render_state, np.ndarray) and not self._anim_active:
            self._draw_frame(render_state)

    def _handle_training_done(self, payload: Dict[str, Any]) -> None:
        if payload.get("stopped"):
            self.status_var.set("Training stopped")
        else:
            self.status_var.set("Training complete")

        csv_path = payload.get("csv_path")
        if csv_path:
            self.status_var.set(f"Training complete | CSV: {csv_path}")

        self._export_plot_snapshot()

    def _handle_error(self, payload: Dict[str, Any]) -> None:
        message = payload.get("message", "Unknown worker error")
        self.status_var.set("Error")
        messagebox.showerror("Training error", str(message))

    def _refresh_plot(self) -> None:
        self.ax.clear()
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel("Episode")
        self.ax.set_ylabel("Reward")

        if self._history_reward:
            x = list(range(1, len(self._history_reward) + 1))
            self.ax.plot(x, self._history_reward, label="Reward", alpha=0.7)
            self.ax.plot(x, self._history_mavg, label="Moving avg(20)", linewidth=2.0)

        if self._history_eval:
            eval_x = [item[0] for item in self._history_eval]
            eval_y = [item[1] for item in self._history_eval]
            self.ax.scatter(eval_x, eval_y, label="Deterministic eval", s=24)

        self.ax.legend(loc="best")
        self.figure.tight_layout()
        self.plot_canvas.draw_idle()

    def _export_plot_snapshot(self) -> None:
        if not self._history_reward:
            return
        os.makedirs("plots", exist_ok=True)
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        label = self.policy_var.get()
        filename = f"Pusher_{label}_{now}.png"
        output_path = os.path.join("plots", filename)
        self.figure.savefig(output_path, dpi=120)

    def _start_animation(self, frames: List[np.ndarray]) -> None:
        if not frames:
            return
        self._anim_active = True
        self._anim_frames = frames
        self._anim_idx = 0
        self._step_animation()

    def _step_animation(self) -> None:
        if not self._anim_active:
            return

        if self._anim_idx >= len(self._anim_frames):
            self._anim_active = False
            if self._anim_pending is not None:
                pending = self._anim_pending
                self._anim_pending = None
                self._start_animation(pending)
            return

        frame = self._anim_frames[self._anim_idx]
        self._draw_frame(frame)
        self._anim_idx += 1
        self.root.after(33, self._step_animation)

    def _draw_frame(self, frame: np.ndarray) -> None:
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            return

        canvas_w = max(1, self.render_canvas.winfo_width())
        canvas_h = max(1, self.render_canvas.winfo_height())

        if Image is not None and ImageTk is not None:
            img = Image.fromarray(frame)
            img = img.resize((canvas_w, canvas_h), Image.BILINEAR)
            self._render_photo = ImageTk.PhotoImage(img)
            self.render_canvas.delete("all")
            self.render_canvas.create_image(0, 0, image=self._render_photo, anchor="nw")
        else:
            self.render_canvas.delete("all")
            self.render_canvas.create_text(
                8,
                8,
                text="Pillow not installed: frame preview unavailable.",
                fill="#e0e0e0",
                anchor="nw",
            )
