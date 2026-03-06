import json
import statistics
import sys
import time
import traceback
import tkinter as tk
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from CarRacing_gui import CarRacingGUI

WARMUP_RUNS = 1
MEASURED_RUNS = 5


def wait_for_run_completion(root: tk.Tk, gui: CarRacingGUI) -> float:
    t0 = time.perf_counter()
    idle_cycles = 0
    while True:
        root.update()
        alive = gui._alive_workers_count() > 0
        playback = bool(gui.playback_active)
        pending = bool(gui.pending_playback)
        if not alive and not playback and not pending:
            idle_cycles += 1
        else:
            idle_cycles = 0
        if idle_cycles >= 10:
            break
        time.sleep(0.01)
    return time.perf_counter() - t0


def configure_gui(gui: CarRacingGUI, animation_on: bool) -> None:
    gui.param_vars["Policy"].set("SAC")
    gui.policy_dropdown.set("SAC")
    gui._on_policy_changed()
    gui.param_vars["Episodes"].set(1)
    gui.param_vars["Max steps"].set(1000)
    gui.param_vars["Animation on"].set(bool(animation_on))
    gui.param_vars["Update rate (episodes)"].set(1)
    gui.param_vars["Frame stride"].set(2)
    gui.param_vars["Device"].set("GPU")
    gui.param_vars["compare_on"].set(False)
    gui.param_vars["Batch compare mode"].set(False)

    # Ensure gradient updates happen in the single episode.
    gui.specific_entries["learning_starts"].set("1")
    gui.specific_entries["train_freq"].set("1")
    gui.specific_entries["gradient_steps"].set("1")


def run_one(root: tk.Tk, gui: CarRacingGUI) -> float:
    gui._start_train()
    return wait_for_run_completion(root, gui)


def run_mode(animation_on: bool) -> dict:
    root = None
    gui = None
    try:
        root = tk.Tk()
        gui = CarRacingGUI(root)
        configure_gui(gui, animation_on=animation_on)

        warmup_values = []
        measured_values = []

        mode_label = "anim-on" if animation_on else "anim-off"

        for i in range(1, WARMUP_RUNS + 1):
            print(f"[{mode_label}][warmup] run {i}/{WARMUP_RUNS} start", flush=True)
            value = run_one(root, gui)
            warmup_values.append(value)
            print(f"[{mode_label}][warmup] run {i}/{WARMUP_RUNS} done: {value:.3f}s", flush=True)

        for i in range(1, MEASURED_RUNS + 1):
            print(f"[{mode_label}][measured] run {i}/{MEASURED_RUNS} start", flush=True)
            value = run_one(root, gui)
            measured_values.append(value)
            print(f"[{mode_label}][measured] run {i}/{MEASURED_RUNS} done: {value:.3f}s", flush=True)

        mean_s = statistics.mean(measured_values)
        std_s = statistics.stdev(measured_values) if len(measured_values) > 1 else 0.0

        return {
            "policy": "SAC",
            "episodes": 1,
            "max_steps": 1000,
            "animation_on": bool(animation_on),
            "learning_starts": 1,
            "train_freq": 1,
            "gradient_steps": 1,
            "device": "GPU",
            "gui_startups": 1,
            "warmup_runs_discarded": WARMUP_RUNS,
            "warmup_timings_seconds": [round(x, 3) for x in warmup_values],
            "measured_runs": MEASURED_RUNS,
            "measured_timings_seconds": [round(x, 3) for x in measured_values],
            "measured_mean_seconds": round(mean_s, 3),
            "measured_std_seconds": round(std_s, 3),
        }
    finally:
        if gui is not None:
            try:
                gui._on_close()
            except Exception:
                pass


def main() -> int:
    out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "single_gui_session_warmup_plus5_anim_on_off_result.json"

    try:
        print("[anim-on] sequence start", flush=True)
        anim_on = run_mode(animation_on=True)
        print("[anim-on] sequence done", flush=True)

        print("[anim-off] sequence start", flush=True)
        anim_off = run_mode(animation_on=False)
        print("[anim-off] sequence done", flush=True)

        result = {
            "scenario": "single_gui_session_full_gui_1ep_1000steps_gradient_warmup_plus_5_anim_on_and_off",
            "anim_on": anim_on,
            "anim_off": anim_off,
        }

        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        print(f"saved_result={result_file}", flush=True)
        return 0
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
