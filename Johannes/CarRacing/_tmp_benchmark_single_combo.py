import argparse
import json
import statistics
import sys
import time
import traceback
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from CarRacing_gui import CarRacingGUI
from CarRacing_logic import (
    DEFAULT_SPECIFIC,
    CarRacingEnvWrapper,
    CarRacingTrainer,
    EnvConfig,
    TrainConfig,
)

POLICY = "SAC"
EPISODES = 1
MAX_STEPS = 1000
NUM_ENVS = 4


def as_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes", "y"}:
        return True
    if text in {"off", "false", "0", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean token: {value}")


def build_params(with_gradient_update: bool) -> dict:
    params = dict(DEFAULT_SPECIFIC[POLICY])
    params["buffer_size"] = 2_000
    params["learning_starts"] = 1 if with_gradient_update else 10_000_000
    params["train_freq"] = 1
    params["gradient_steps"] = 1
    return params


def wait_for_gui_completion(root: tk.Tk, gui: CarRacingGUI, timeout_s: float = 300.0) -> float:
    t0 = time.perf_counter()
    idle_cycles = 0
    while True:
        root.update()
        alive = gui._alive_workers_count() > 0
        playback = bool(gui.playback_active)
        pending = bool(gui.pending_playback)
        training_active = gui.btn_single.instate(["disabled"])

        if not training_active and not alive and not playback and not pending:
            idle_cycles += 1
        else:
            idle_cycles = 0

        if idle_cycles >= 10:
            break
        if time.perf_counter() - t0 > timeout_s:
            raise TimeoutError(f"GUI run did not complete within {timeout_s:.0f}s")
        time.sleep(0.01)
    return time.perf_counter() - t0


def configure_gui(gui: CarRacingGUI, animation_on: bool, with_gradient_update: bool) -> None:
    gui._set_training_active(False)
    gui.pending_playback.clear()
    gui.playback_active = False

    gui.param_vars["Policy"].set(POLICY)
    gui.policy_dropdown.set(POLICY)
    gui._on_policy_changed()
    gui.param_vars["Episodes"].set(EPISODES)
    gui.param_vars["Max steps"].set(MAX_STEPS)
    gui.param_vars["Animation on"].set(bool(animation_on))
    gui.param_vars["Update rate (episodes)"].set(1)
    gui.param_vars["Frame stride"].set(2)
    gui.param_vars["Device"].set("GPU")
    gui.param_vars["compare_on"].set(False)
    gui.param_vars["Batch compare mode"].set(False)

    if with_gradient_update:
        gui.specific_entries["learning_starts"].set("1")
    else:
        gui.specific_entries["learning_starts"].set("10000000")
    gui.specific_entries["train_freq"].set("1")
    gui.specific_entries["gradient_steps"].set("1")


def run_one_gui(root: tk.Tk, gui: CarRacingGUI, animation_on: bool, with_gradient_update: bool) -> float:
    configure_gui(gui, animation_on=animation_on, with_gradient_update=with_gradient_update)
    gui._start_train()
    return wait_for_gui_completion(root, gui)


def run_one_headless(animation_on: bool, with_gradient_update: bool) -> float:
    trainer = CarRacingTrainer(CarRacingEnvWrapper())
    cfg = TrainConfig(
        policy_name=POLICY,
        episodes=EPISODES,
        max_steps=MAX_STEPS,
        params=build_params(with_gradient_update=with_gradient_update),
        env_config=EnvConfig(
            env_id="CarRacing-v3",
            render_mode="rgb_array",
            lap_complete_percent=0.95,
            domain_randomize=False,
            continuous=True,
        ),
        animation_on=bool(animation_on),
        animation_fps=30,
        update_rate=1,
        frame_stride=2,
        run_id=f"headless_{time.time_ns()}",
        session_id=f"session_{time.time_ns()}",
        device="GPU",
        collect_transitions=False,
        performance_mode=False,
        num_envs=NUM_ENVS,
    )

    t0 = time.perf_counter()
    trainer.train(cfg)
    return time.perf_counter() - t0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gui", required=True, choices=["on", "off"])
    parser.add_argument("--anim", required=True, choices=["on", "off"])
    parser.add_argument("--grad", required=True, choices=["on", "off"])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    gui_on = as_bool(args.gui)
    animation_on = as_bool(args.anim)
    with_gradient_update = as_bool(args.grad)
    warmup_runs = max(0, int(args.warmup))
    measured_runs = max(1, int(args.runs))

    combo_label = f"gui-{args.gui}_anim-{args.anim}_grad-{args.grad}"
    out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / f"benchmark_{combo_label}_warmup{warmup_runs}_runs{measured_runs}.json"

    root = None
    gui = None
    try:
        warmup_values = []
        measured_values = []

        if gui_on:
            root = tk.Tk()
            gui = CarRacingGUI(root)

        for i in range(1, warmup_runs + 1):
            print(f"[{combo_label}][warmup] run {i}/{warmup_runs} start", flush=True)
            if gui_on:
                value = run_one_gui(root=root, gui=gui, animation_on=animation_on, with_gradient_update=with_gradient_update)
            else:
                value = run_one_headless(animation_on=animation_on, with_gradient_update=with_gradient_update)
            warmup_values.append(value)
            print(f"[{combo_label}][warmup] run {i}/{warmup_runs} done: {value:.3f}s", flush=True)

        for i in range(1, measured_runs + 1):
            print(f"[{combo_label}][measured] run {i}/{measured_runs} start", flush=True)
            if gui_on:
                value = run_one_gui(root=root, gui=gui, animation_on=animation_on, with_gradient_update=with_gradient_update)
            else:
                value = run_one_headless(animation_on=animation_on, with_gradient_update=with_gradient_update)
            measured_values.append(value)
            print(f"[{combo_label}][measured] run {i}/{measured_runs} done: {value:.3f}s", flush=True)

        mean_s = statistics.mean(measured_values)
        std_s = statistics.stdev(measured_values) if len(measured_values) > 1 else 0.0

        payload = {
            "scenario": "single_combo_gui_anim_grad_warmup_plus_measured",
            "combo": {
                "gui_on": gui_on,
                "animation_on": animation_on,
                "with_gradient_update": with_gradient_update,
            },
            "policy": POLICY,
            "episodes": EPISODES,
            "max_steps": MAX_STEPS,
            "device": "GPU",
            "num_envs": NUM_ENVS,
            "learning_starts": 1 if with_gradient_update else 10_000_000,
            "train_freq": 1,
            "gradient_steps": 1,
            "warmup_runs_discarded": warmup_runs,
            "warmup_timings_seconds": [round(x, 3) for x in warmup_values],
            "measured_runs": measured_runs,
            "measured_timings_seconds": [round(x, 3) for x in measured_values],
            "measured_mean_seconds": round(mean_s, 3),
            "measured_std_seconds": round(std_s, 3),
        }

        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        print(f"saved_result={result_file}", flush=True)
        return 0
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 1
    finally:
        if gui is not None:
            try:
                gui._on_close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
