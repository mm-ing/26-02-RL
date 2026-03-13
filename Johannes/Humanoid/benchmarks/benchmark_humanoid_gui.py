from __future__ import annotations

import argparse
import csv
import os
import platform
import statistics
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Apply startup guards before importing GUI and ML modules.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Suppress repetitive SB3 advisory warning for PPO+MLP on GPU during matrix sweeps.
warnings.filterwarnings(
    "ignore",
    message=r"You are trying to run PPO on the GPU, but it is primarily intended to run on the CPU.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"stable_baselines3\.common\.on_policy_algorithm",
)

if platform.system().lower().startswith("win"):
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        os.environ.pop("MUJOCO_GL", None)
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import tkinter as tk
from tkinter import messagebox

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Humanoid_gui import HumanoidGUI


POLICIES = ("PPO", "SAC", "TD3")
HIDDEN_LAYERS = ("256", "256,256")
DEVICES = ("CPU", "GPU")


@dataclass
class RunResult:
    policy: str
    hidden_layer: str
    device: str
    repeat: int
    elapsed_s: float
    completed_episodes: int
    last_reward: float
    status: str
    error: str


def _debug_log(log_path: Path | None, message: str) -> None:
    if log_path is None:
        return
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {message}\n"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _set_policy_defaults_for_gradient_update(app: HumanoidGUI, policy: str, max_steps: int) -> None:
    """Configure policy knobs so a single episode still performs optimization."""
    params = app.policy_param_vars[policy]

    if policy == "PPO":
        params["batch_size"].set("16")
        params["n_steps"].set(str(max(16, max_steps)))
    elif policy in {"SAC", "TD3"}:
        params["batch_size"].set("8")
        params["learning_starts"].set("0")
        if "train_freq" in params:
            params["train_freq"].set("1")
        if "gradient_steps" in params:
            params["gradient_steps"].set("1")


def _configure_single_run(app: HumanoidGUI, policy: str, hidden_layer: str, device: str, max_steps: int) -> None:
    app.clear_plot()
    app.compare_on_var.set(False)
    app.animation_on_var.set(True)
    app.animation_fps_var.set("30")
    app.update_rate_var.set("1")
    app.frame_stride_var.set("1")

    app.episodes_var.set("1")
    app.max_steps_var.set(str(max_steps))
    app.device_var.set(device)
    app.policy_var.set(policy)

    _set_policy_defaults_for_gradient_update(app, policy, max_steps=max_steps)
    app.policy_param_vars[policy]["hidden_layer"].set(hidden_layer)


def _wait_until_idle(app: HumanoidGUI, root: tk.Tk, timeout_s: float) -> Tuple[bool, str]:
    started = time.perf_counter()
    while True:
        root.update_idletasks()
        root.update()

        is_idle = (
            (not app._training_active)
            and (len(app._pending_runs) == 0)
            and app._event_queue.empty()
            and (not app._frame_playback_active)
        )
        if is_idle:
            return True, ""

        if (time.perf_counter() - started) > timeout_s:
            return False, f"timeout after {timeout_s:.1f}s"

        time.sleep(0.01)


def _extract_last_run_stats(app: HumanoidGUI) -> Tuple[int, float]:
    if not app._run_history:
        return 0, 0.0
    last_run_id = next(reversed(app._run_history))
    hist = app._run_history[last_run_id]
    rewards = list(hist.get("rewards", []))
    return len(rewards), (float(rewards[-1]) if rewards else 0.0)


def run_benchmark(
    repeats: int,
    max_steps: int,
    timeout_s: float,
    output_csv: Path,
    output_md: Path,
    verbose: bool,
    combinations: List[Tuple[str, str, str]] | None = None,
    warmup: bool = True,
    debug_log_path: Path | None = None,
) -> List[RunResult]:
    root = tk.Tk()
    app = HumanoidGUI(root)

    # Keep benchmark automation non-blocking if runtime errors occur.
    messagebox.showerror = lambda *_args, **_kwargs: None

    results: List[RunResult] = []
    gpu_available = bool(torch.cuda.is_available())
    fatal_error = ""

    run_combinations = combinations or [
        (policy, hidden_layer, device)
        for policy in POLICIES
        for hidden_layer in HIDDEN_LAYERS
        for device in DEVICES
    ]
    _debug_log(debug_log_path, f"run_benchmark start: combos={len(run_combinations)} repeats={repeats}")

    if warmup and run_combinations:
        # Keep warmup deterministic and CPU-only so resume mode cannot stall on a GPU combo.
        warm_policy, warm_hidden, warm_device = "PPO", "256", "CPU"
        print(
            f"Warmup run (not measured): policy={warm_policy}, hidden={warm_hidden}, device={warm_device}",
            flush=True,
        )
        _debug_log(
            debug_log_path,
            f"warmup start: policy={warm_policy} hidden={warm_hidden} device={warm_device}",
        )
        _configure_single_run(app, warm_policy, warm_hidden, warm_device, max_steps=max_steps)
        try:
            app.train_and_run()
            _wait_until_idle(app, root, timeout_s=timeout_s)
        except Exception:
            # Continue even if warmup fails; measured runs still proceed.
            pass
        _debug_log(debug_log_path, "warmup end")
        # Clear warmup traces from plots/history before measured runs.
        app.clear_plot()
        app._run_history.clear()

    try:
        for policy, hidden_layer, device in run_combinations:
            _debug_log(debug_log_path, f"combo start: policy={policy} hidden={hidden_layer} device={device}")
            combo_start_len = len(results)
            combo_failed = False

            for repeat in range(1, repeats + 1):
                _debug_log(
                    debug_log_path,
                    f"repeat start: policy={policy} hidden={hidden_layer} device={device} repeat={repeat}",
                )
                if device == "GPU" and not gpu_available:
                    results.append(
                        RunResult(
                            policy=policy,
                            hidden_layer=hidden_layer,
                            device=device,
                            repeat=repeat,
                            elapsed_s=0.0,
                            completed_episodes=0,
                            last_reward=0.0,
                            status="skipped",
                            error="GPU not available on this machine",
                        )
                    )
                    continue

                try:
                    _configure_single_run(app, policy, hidden_layer, device, max_steps=max_steps)

                    t0 = time.perf_counter()
                    try:
                        app.train_and_run()
                        ok, err = _wait_until_idle(app, root, timeout_s=timeout_s)
                    except Exception as exc:
                        ok, err = False, f"runtime exception: {exc}"
                    elapsed = time.perf_counter() - t0

                    completed_episodes, last_reward = _extract_last_run_stats(app)
                    last_gui_error = str(getattr(app, "_last_error_message", "") or "").strip()
                    status = "ok" if ok and completed_episodes >= 1 else "error"
                    error_message = ""
                    if status != "ok":
                        error_message = err or last_gui_error or "no episode completed"

                    results.append(
                        RunResult(
                            policy=policy,
                            hidden_layer=hidden_layer,
                            device=device,
                            repeat=repeat,
                            elapsed_s=elapsed,
                            completed_episodes=completed_episodes,
                            last_reward=last_reward,
                            status=status,
                            error=error_message,
                        )
                    )
                    if verbose or status != "ok":
                        print(
                            f"[{policy:>3}] hidden={hidden_layer:<7} device={device:<3} "
                            f"repeat={repeat} status={status} elapsed={elapsed:.2f}s",
                            flush=True,
                        )
                    _debug_log(
                        debug_log_path,
                        f"repeat end: policy={policy} hidden={hidden_layer} device={device} repeat={repeat} "
                        f"status={status} elapsed={elapsed:.2f}s",
                    )
                except Exception as exc:
                    combo_failed = True
                    results.append(
                        RunResult(
                            policy=policy,
                            hidden_layer=hidden_layer,
                            device=device,
                            repeat=repeat,
                            elapsed_s=0.0,
                            completed_episodes=0,
                            last_reward=0.0,
                            status="error",
                            error=f"combo exception: {exc}",
                        )
                    )
                    if verbose:
                        print(
                            f"[{policy:>3}] hidden={hidden_layer:<7} device={device:<3} "
                            f"repeat={repeat} status=error elapsed=0.00s",
                            flush=True,
                        )
                    _debug_log(
                        debug_log_path,
                        f"repeat exception: policy={policy} hidden={hidden_layer} device={device} "
                        f"repeat={repeat} error={exc}",
                    )
                    break

            # Persist intermediate artifacts after every combination.
            combo_results = results[combo_start_len:]
            combo_ok = sum(1 for r in combo_results if r.status == "ok")
            print(
                f"COMBINATION COMPLETE: policy={policy}, hidden={hidden_layer}, "
                f"device={device}, ok={combo_ok}/{len(combo_results)}",
                flush=True,
            )
            _debug_log(
                debug_log_path,
                f"combo complete: policy={policy} hidden={hidden_layer} device={device} "
                f"ok={combo_ok}/{len(combo_results)}",
            )
            _write_csv(results, output_csv)
            _write_markdown(
                results,
                output_md,
                repeats=repeats,
                max_steps=max_steps,
                gpu_available=gpu_available,
                devices=tuple(sorted({device for _, _, device in run_combinations})),
            )

            if combo_failed:
                # Continue with next combination instead of aborting the whole benchmark.
                continue
    except Exception as exc:
        fatal_error = str(exc)
    finally:
        if fatal_error:
            print(f"FATAL benchmark error: {fatal_error}", flush=True)
            traceback.print_exc()
            _debug_log(debug_log_path, f"fatal error: {fatal_error}")
        _write_csv(results, output_csv)
        _write_markdown(
            results,
            output_md,
            repeats=repeats,
            max_steps=max_steps,
            gpu_available=gpu_available,
            devices=tuple(sorted({r.device for r in results})) if results else DEVICES,
        )
        _debug_log(debug_log_path, f"run_benchmark end: results={len(results)}")
        app.on_close()

    return results


def _write_csv(results: List[RunResult], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "policy",
                "hidden_layer",
                "device",
                "repeat",
                "elapsed_s",
                "completed_episodes",
                "last_reward",
                "status",
                "error",
            ]
        )
        for r in results:
            writer.writerow(
                [
                    r.policy,
                    r.hidden_layer,
                    r.device,
                    r.repeat,
                    f"{r.elapsed_s:.6f}",
                    r.completed_episodes,
                    f"{r.last_reward:.6f}",
                    r.status,
                    r.error,
                ]
            )


def _read_csv_results(path: Path) -> List[RunResult]:
    if not path.exists():
        return []

    loaded: List[RunResult] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                loaded.append(
                    RunResult(
                        policy=row.get("policy", "").strip(),
                        hidden_layer=row.get("hidden_layer", "").strip(),
                        device=row.get("device", "").strip(),
                        repeat=int(row.get("repeat", "0")),
                        elapsed_s=float(row.get("elapsed_s", "0") or 0.0),
                        completed_episodes=int(row.get("completed_episodes", "0") or 0),
                        last_reward=float(row.get("last_reward", "0") or 0.0),
                        status=row.get("status", "error").strip(),
                        error=row.get("error", "").strip(),
                    )
                )
            except Exception:
                continue
    return loaded


def _group_ok_elapsed(results: List[RunResult]) -> Dict[Tuple[str, str, str], List[float]]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for r in results:
        if r.status != "ok":
            continue
        key = (r.policy, r.hidden_layer, r.device)
        grouped.setdefault(key, []).append(r.elapsed_s)
    return grouped


def _fmt_mean_std(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "-", "-"
    if len(values) == 1:
        return f"{values[0]:.3f}", "0.000"
    return f"{statistics.mean(values):.3f}", f"{statistics.stdev(values):.3f}"


def _completed_combinations(results: List[RunResult], repeats: int) -> set[Tuple[str, str, str]]:
    grouped = _group_ok_elapsed(results)
    done: set[Tuple[str, str, str]] = set()
    for key, vals in grouped.items():
        if len(vals) >= repeats:
            done.add(key)
    return done


def _bottleneck_notes(results: List[RunResult]) -> List[str]:
    notes: List[str] = []
    grouped = _group_ok_elapsed(results)

    def avg_for(policy: str, hidden: str, device: str) -> float | None:
        vals = grouped.get((policy, hidden, device), [])
        if not vals:
            return None
        return float(statistics.mean(vals))

    # Hidden layer depth effect.
    depth_deltas: List[float] = []
    for policy in POLICIES:
        for device in DEVICES:
            a = avg_for(policy, "256", device)
            b = avg_for(policy, "256,256", device)
            if a is not None and b is not None and a > 0:
                depth_deltas.append((b - a) / a)
    if depth_deltas:
        mean_depth_pct = 100.0 * statistics.mean(depth_deltas)
        notes.append(
            f"Changing hidden layers from `256` to `256,256` changed elapsed time by about `{mean_depth_pct:.1f}%` on average."
        )

    # Device effect.
    device_deltas: List[float] = []
    for policy in POLICIES:
        for hidden in HIDDEN_LAYERS:
            cpu = avg_for(policy, hidden, "CPU")
            gpu = avg_for(policy, hidden, "GPU")
            if cpu is not None and gpu is not None and cpu > 0:
                device_deltas.append((gpu - cpu) / cpu)
    if device_deltas:
        mean_device_pct = 100.0 * statistics.mean(device_deltas)
        if mean_device_pct > 5.0:
            notes.append(
                f"GPU runs were slower by about `{mean_device_pct:.1f}%` on average, which is consistent with small per-step workloads being dominated by GUI rendering, Python overhead, and CPU-GPU transfer latency."
            )
        elif mean_device_pct < -5.0:
            notes.append(
                f"GPU runs were faster by about `{abs(mean_device_pct):.1f}%` on average, indicating model update compute started to dominate the GUI/rendering overhead."
            )
        else:
            notes.append(
                "CPU and GPU times were close, suggesting rendering/animation and environment stepping dominate total runtime in this short single-episode benchmark."
            )

    # Policy effect ranking.
    policy_means: List[Tuple[str, float]] = []
    for policy in POLICIES:
        vals = [r.elapsed_s for r in results if r.status == "ok" and r.policy == policy]
        if vals:
            policy_means.append((policy, float(statistics.mean(vals))))
    if len(policy_means) >= 2:
        policy_means.sort(key=lambda item: item[1])
        fastest = policy_means[0]
        slowest = policy_means[-1]
        gap_pct = 100.0 * ((slowest[1] - fastest[1]) / fastest[1]) if fastest[1] > 0 else 0.0
        notes.append(
            f"Across all tested settings, `{fastest[0]}` was fastest and `{slowest[0]}` slowest, with a mean-gap of about `{gap_pct:.1f}%` in elapsed time."
        )

    if not notes:
        notes.append("Insufficient successful runs to estimate bottlenecks.")

    notes.append(
        "Interpretation focus: this setup includes full Tkinter + Matplotlib updates and frame playback, so measured time combines RL compute, environment simulation, and GUI animation costs."
    )
    return notes


def _hidden_impact_rows(
    grouped: Dict[Tuple[str, str, str], List[float]],
    devices: Tuple[str, ...],
) -> List[Tuple[str, str, str, str, str]]:
    rows: List[Tuple[str, str, str, str, str]] = []
    for policy in POLICIES:
        for device in devices:
            a_vals = grouped.get((policy, "256", device), [])
            b_vals = grouped.get((policy, "256,256", device), [])
            if not a_vals or not b_vals:
                rows.append((policy, device, "-", "-", "-"))
                continue

            a_mean = float(statistics.mean(a_vals))
            b_mean = float(statistics.mean(b_vals))
            if a_mean <= 0:
                rows.append((policy, device, f"{a_mean:.3f}", f"{b_mean:.3f}", "-"))
                continue

            pct = 100.0 * ((b_mean - a_mean) / a_mean)
            rows.append((policy, device, f"{a_mean:.3f}", f"{b_mean:.3f}", f"{pct:+.1f}%"))
    return rows


def _write_markdown(
    results: List[RunResult],
    output_md: Path,
    repeats: int,
    max_steps: int,
    gpu_available: bool,
    devices: Tuple[str, ...] = DEVICES,
) -> None:
    grouped = _group_ok_elapsed(results)
    total_runs = len(results)
    ok_runs = sum(1 for r in results if r.status == "ok")
    skipped_runs = sum(1 for r in results if r.status == "skipped")
    error_runs = sum(1 for r in results if r.status == "error")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Humanoid Benchmark")
    lines.append("")
    lines.append(f"Generated: `{timestamp}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Environment: `Humanoid-v5`")
    lines.append("- Execution path: full `HumanoidGUI` training flow with animation enabled")
    lines.append(f"- Episodes per run: `1`")
    lines.append(f"- Max steps per episode: `{max_steps}`")
    lines.append("- Animation: `on`, `fps=30`, `frame_stride=1`, `update_rate=1`")
    lines.append(f"- Repeats per combination: `{repeats}`")
    lines.append(f"- GPU available: `{gpu_available}`")
    lines.append("")
    lines.append("## Run Summary")
    lines.append("")
    lines.append(f"- Total runs scheduled: `{total_runs}`")
    lines.append(f"- Successful runs: `{ok_runs}`")
    lines.append(f"- Skipped runs: `{skipped_runs}`")
    lines.append(f"- Error runs: `{error_runs}`")
    lines.append("")
    lines.append("## Results (Mean and Std)")
    lines.append("")
    lines.append("| Policy | Hidden Layers | Device | Repeats (ok/target) | Mean elapsed [s] | Std [s] |")
    lines.append("|---|---|---|---:|---:|---:|")

    for policy in POLICIES:
        for hidden in HIDDEN_LAYERS:
            for device in devices:
                vals = grouped.get((policy, hidden, device), [])
                mean_s, std_s = _fmt_mean_std(vals)
                lines.append(
                    f"| {policy} | `{hidden}` | {device} | {len(vals)}/{repeats} | {mean_s} | {std_s} |"
                )

    lines.append("")
    lines.append("## Hidden Layer Impact")
    lines.append("")
    lines.append("| Policy | Device | Mean 256 [s] | Mean 256,256 [s] | Impact (256,256 vs 256) |")
    lines.append("|---|---|---:|---:|---:|")
    for policy, device, a_mean, b_mean, pct in _hidden_impact_rows(grouped, devices=devices):
        lines.append(f"| {policy} | {device} | {a_mean} | {b_mean} | {pct} |")

    lines.append("")
    lines.append("## Bottleneck Analysis")
    lines.append("")
    for note in _bottleneck_notes(results):
        lines.append(f"- {note}")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines.append("")
        lines.append("## Failures and Skips")
        lines.append("")
        lines.append("| Policy | Hidden Layers | Device | Repeat | Status | Reason |")
        lines.append("|---|---|---|---:|---|---|")
        for r in failed:
            reason = r.error.replace("|", "/")
            lines.append(
                f"| {r.policy} | `{r.hidden_layer}` | {r.device} | {r.repeat} | {r.status} | {reason} |"
            )

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Humanoid GUI training matrix")
    parser.add_argument("--repeats", type=int, default=5, help="Repeats per matrix combination")
    parser.add_argument("--max-steps", type=int, default=64, help="Max episode steps per run")
    parser.add_argument("--timeout", type=float, default=300.0, help="Timeout per run in seconds")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="benchmarks/benchmark_results.csv",
        help="Output CSV path (relative to this script directory)",
    )
    parser.add_argument(
        "--output-md",
        type=str,
        default="benchmarks/benchmark.md",
        help="Output markdown report path (relative to this script directory)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print per-run benchmark progress")
    parser.add_argument("--policy", type=str, choices=POLICIES, help="Run only one policy")
    parser.add_argument(
        "--hidden-layer",
        type=str,
        choices=HIDDEN_LAYERS,
        help="Run only one hidden-layer setting",
    )
    parser.add_argument("--device", type=str, choices=DEVICES, help="Run only one device")
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run matrix in CPU-only mode (all policies, all hidden layers, CPU device).",
    )
    parser.add_argument(
        "--append-existing",
        action="store_true",
        help="Append new run results to existing benchmark_results.csv before writing reports",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="Disable the unmeasured warmup run in the same GUI instance",
    )
    parser.add_argument(
        "--debug-log",
        type=str,
        default="benchmarks/benchmark_crash.log",
        help="Debug log path (relative to this script directory)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    output_csv = (script_dir / args.output_csv).resolve()
    output_md = (script_dir / args.output_md).resolve()
    debug_log_path = (script_dir / args.debug_log).resolve() if args.debug_log else None
    if debug_log_path is not None and debug_log_path.exists():
        debug_log_path.unlink()

    if any([args.policy, args.hidden_layer, args.device]) and not all([args.policy, args.hidden_layer, args.device]):
        raise ValueError("If one of --policy/--hidden-layer/--device is set, all three must be set.")

    combinations: List[Tuple[str, str, str]] | None = None
    if all([args.policy, args.hidden_layer, args.device]):
        combinations = [(args.policy, args.hidden_layer, args.device)]

    if args.cpu_only and combinations is None:
        combinations = [(policy, hidden, "CPU") for policy in POLICIES for hidden in HIDDEN_LAYERS]

    existing_results: List[RunResult] = _read_csv_results(output_csv) if args.append_existing else []

    # Resume mode: in full-matrix mode with --append-existing, skip already complete combinations.
    if combinations is None and args.append_existing and existing_results:
        all_combinations = [
            (policy, hidden_layer, device)
            for policy in POLICIES
            for hidden_layer in HIDDEN_LAYERS
            for device in DEVICES
        ]
        completed = _completed_combinations(existing_results, repeats=max(1, args.repeats))
        combinations = [combo for combo in all_combinations if combo not in completed]

    print("Starting Humanoid GUI benchmark matrix...")
    print(f"repeats={args.repeats}, max_steps={args.max_steps}, timeout={args.timeout}s")
    if combinations is not None:
        if len(combinations) == 1:
            p, h, d = combinations[0]
            print(f"single combination: policy={p}, hidden={h}, device={d}")
        else:
            print(f"resume mode: running {len(combinations)} remaining combinations")

    effective_warmup = (not args.no_warmup) and (not args.append_existing)

    new_results = run_benchmark(
        repeats=max(1, args.repeats),
        max_steps=max(1, args.max_steps),
        timeout_s=max(30.0, args.timeout),
        output_csv=output_csv,
        output_md=output_md,
        verbose=bool(args.verbose),
        combinations=combinations,
        warmup=effective_warmup,
        debug_log_path=debug_log_path,
    )

    results = existing_results + new_results
    _write_csv(results, output_csv)
    _write_markdown(
        results,
        output_md,
        repeats=max(1, args.repeats),
        max_steps=max(1, args.max_steps),
        gpu_available=bool(torch.cuda.is_available()),
        devices=(("CPU",) if args.cpu_only else DEVICES),
    )

    ok_runs = sum(1 for r in results if r.status == "ok")
    print(f"Benchmark complete: {ok_runs}/{len(results)} successful runs")
    print(f"CSV written to: {output_csv}")
    print(f"Report written to: {output_md}")


if __name__ == "__main__":
    main()
