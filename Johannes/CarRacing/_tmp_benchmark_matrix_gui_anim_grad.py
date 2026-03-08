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

WARMUP_RUNS = 1
MEASURED_RUNS = 5
POLICY = "SAC"
EPISODES = 1
MAX_STEPS = 1000
NUM_ENVS = 4


def build_params(with_gradient_update: bool):
    params = dict(DEFAULT_SPECIFIC[POLICY])
    # Keep memory bounded for vec-env matrix runs.
    params["buffer_size"] = 2_000
    params["learning_starts"] = 1 if with_gradient_update else 10_000_000
    params["train_freq"] = 1
    params["gradient_steps"] = 1
    return params


def wait_for_gui_completion(root: tk.Tk, gui: CarRacingGUI) -> float:
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
        if time.perf_counter() - t0 > 300.0:
            raise TimeoutError("GUI run did not complete within 300s")
        time.sleep(0.01)
    return time.perf_counter() - t0


def configure_gui(gui: CarRacingGUI, animation_on: bool, with_gradient_update: bool):
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
    gui._set_training_active(False)
    gui.pending_playback.clear()
    gui.playback_active = False
    for _ in range(5):
        root.update()
        time.sleep(0.005)

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


def run_combo(
    gui_on: bool,
    animation_on: bool,
    with_gradient_update: bool,
    gui_ctx: tuple[tk.Tk, CarRacingGUI] | None = None,
) -> dict:
    warmup_values = []
    measured_values = []

    combo_label = (
        f"gui-{'on' if gui_on else 'off'}"
        f"_anim-{'on' if animation_on else 'off'}"
        f"_grad-{'on' if with_gradient_update else 'off'}"
    )

    root = None
    gui = None
    if gui_on:
        if gui_ctx is None:
            raise RuntimeError("GUI context required for gui_on run")
        root, gui = gui_ctx

    try:
        for i in range(1, WARMUP_RUNS + 1):
            print(f"[{combo_label}][warmup] run {i}/{WARMUP_RUNS} start", flush=True)
            if gui_on:
                value = run_one_gui(
                    root=root,
                    gui=gui,
                    animation_on=animation_on,
                    with_gradient_update=with_gradient_update,
                )
            else:
                value = run_one_headless(animation_on=animation_on, with_gradient_update=with_gradient_update)
            warmup_values.append(value)
            print(f"[{combo_label}][warmup] run {i}/{WARMUP_RUNS} done: {value:.3f}s", flush=True)

        for i in range(1, MEASURED_RUNS + 1):
            print(f"[{combo_label}][measured] run {i}/{MEASURED_RUNS} start", flush=True)
            if gui_on:
                value = run_one_gui(
                    root=root,
                    gui=gui,
                    animation_on=animation_on,
                    with_gradient_update=with_gradient_update,
                )
            else:
                value = run_one_headless(animation_on=animation_on, with_gradient_update=with_gradient_update)
            measured_values.append(value)
            print(f"[{combo_label}][measured] run {i}/{MEASURED_RUNS} done: {value:.3f}s", flush=True)
    finally:
        pass

    mean_s = statistics.mean(measured_values)
    std_s = statistics.stdev(measured_values) if len(measured_values) > 1 else 0.0

    return {
        "gui_on": bool(gui_on),
        "animation_on": bool(animation_on),
        "with_gradient_update": bool(with_gradient_update),
        "policy": POLICY,
        "episodes": EPISODES,
        "max_steps": MAX_STEPS,
        "device": "GPU",
        "num_envs": NUM_ENVS,
        "learning_starts": 1 if with_gradient_update else 10_000_000,
        "train_freq": 1,
        "gradient_steps": 1,
        "warmup_runs_discarded": WARMUP_RUNS,
        "warmup_timings_seconds": [round(x, 3) for x in warmup_values],
        "measured_runs": MEASURED_RUNS,
        "measured_timings_seconds": [round(x, 3) for x in measured_values],
        "measured_mean_seconds": round(mean_s, 3),
        "measured_std_seconds": round(std_s, 3),
    }


def main() -> int:
    out_dir = Path("results_csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    result_file = out_dir / "benchmark_matrix_gui_anim_grad_warmup_plus5.json"

    combos = [
        (True, True, True),
        (True, True, False),
        (True, False, True),
        (True, False, False),
        (False, True, True),
        (False, True, False),
        (False, False, True),
        (False, False, False),
    ]

    shared_root = None
    shared_gui = None

    try:
        results = []
        shared_root = tk.Tk()
        shared_gui = CarRacingGUI(shared_root)
        for gui_on, animation_on, with_gradient_update in combos:
            combo_name = (
                f"gui={'on' if gui_on else 'off'},"
                f"anim={'on' if animation_on else 'off'},"
                f"grad={'on' if with_gradient_update else 'off'}"
            )
            print(f"[combo] start {combo_name}", flush=True)
            result = run_combo(
                gui_on=gui_on,
                animation_on=animation_on,
                with_gradient_update=with_gradient_update,
                gui_ctx=(shared_root, shared_gui) if gui_on else None,
            )
            results.append(result)
            print(
                f"[combo] done {combo_name} -> mean={result['measured_mean_seconds']:.3f}s std={result['measured_std_seconds']:.3f}s",
                flush=True,
            )

        payload = {
            "scenario": "matrix_gui_anim_grad_update_warmup_plus_5",
            "results": results,
        }
        result_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(json.dumps(payload, indent=2), flush=True)
        print(f"saved_result={result_file}", flush=True)
        return 0
    except Exception:
        print(traceback.format_exc(), flush=True)
        return 1
    finally:
        if shared_gui is not None:
            try:
                shared_gui._on_close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
