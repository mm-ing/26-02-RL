import cProfile
import io
import json
import pstats
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


def configure_gui(gui: CarRacingGUI) -> None:
    gui.param_vars["Policy"].set("SAC")
    gui.policy_dropdown.set("SAC")
    gui._on_policy_changed()
    gui.param_vars["Episodes"].set(1)
    gui.param_vars["Max steps"].set(1000)
    gui.param_vars["Animation on"].set(False)
    gui.param_vars["Update rate (episodes)"].set(1)
    gui.param_vars["Frame stride"].set(2)
    gui.param_vars["Device"].set("GPU")
    gui.param_vars["compare_on"].set(False)
    gui.param_vars["Batch compare mode"].set(False)

    # Ensure updates happen in the single episode.
    gui.specific_entries["learning_starts"].set("1")
    gui.specific_entries["train_freq"].set("1")
    gui.specific_entries["gradient_steps"].set("1")


def run_one(root: tk.Tk, gui: CarRacingGUI) -> float:
    gui._start_train()
    return wait_for_run_completion(root, gui)


def profile_one_run(root: tk.Tk, gui: CarRacingGUI) -> dict:
    profiler = cProfile.Profile()
    profiler.enable()
    profiled_seconds = run_one(root, gui)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).strip_dirs().sort_stats("cumtime")
    stats.print_stats(40)
    top_text = stream.getvalue()

    stream2 = io.StringIO()
    stats_callers = pstats.Stats(profiler, stream=stream2).strip_dirs().sort_stats("cumtime")
    stats_callers.print_callers(20)
    callers_text = stream2.getvalue()

    return {
        "profiled_run_seconds": round(profiled_seconds, 3),
        "top_cumulative_text": top_text,
        "callers_text": callers_text,
    }


def main() -> int:
    out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "single_gui_session_anim_off_warmup_plus5_result.json"
    profile_file = out_dir / "single_gui_session_anim_off_profile_top.txt"

    root = None
    gui = None

    try:
        root = tk.Tk()
        gui = CarRacingGUI(root)
        configure_gui(gui)

        warmup_values = []
        measured_values = []

        for i in range(1, WARMUP_RUNS + 1):
            print(f"[anim-off][warmup] run {i}/{WARMUP_RUNS} start", flush=True)
            value = run_one(root, gui)
            warmup_values.append(value)
            print(f"[anim-off][warmup] run {i}/{WARMUP_RUNS} done: {value:.3f}s", flush=True)

        for i in range(1, MEASURED_RUNS + 1):
            print(f"[anim-off][measured] run {i}/{MEASURED_RUNS} start", flush=True)
            value = run_one(root, gui)
            measured_values.append(value)
            print(f"[anim-off][measured] run {i}/{MEASURED_RUNS} done: {value:.3f}s", flush=True)

        print("[anim-off][profile] start", flush=True)
        profile_payload = profile_one_run(root, gui)
        print("[anim-off][profile] done", flush=True)

        mean_s = statistics.mean(measured_values)
        std_s = statistics.stdev(measured_values) if len(measured_values) > 1 else 0.0

        result = {
            "scenario": "single_gui_session_full_gui_1ep_1000steps_gradient_warmup_plus_5_anim_off",
            "policy": "SAC",
            "episodes": 1,
            "max_steps": 1000,
            "animation_on": False,
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
            "profiled_run_seconds": profile_payload["profiled_run_seconds"],
        }

        result_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
        profile_file.write_text(
            "=== Top Cumulative Functions ===\n"
            + profile_payload["top_cumulative_text"]
            + "\n=== Callers (Top) ===\n"
            + profile_payload["callers_text"],
            encoding="utf-8",
        )

        print(json.dumps(result, indent=2), flush=True)
        print(f"saved_result={result_file}", flush=True)
        print(f"saved_profile={profile_file}", flush=True)
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
