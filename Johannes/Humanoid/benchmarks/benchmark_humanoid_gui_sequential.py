from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from Humanoid_gui import HumanoidGUI

POLICIES: Tuple[str, ...] = ("TD3", "SAC", "PPO")


def _is_fully_idle(app: HumanoidGUI) -> bool:
    with app._trainers_lock:
        workers_alive = len(app._active_workers) > 0
    return (
        (not app._training_active)
        and (len(app._pending_runs) == 0)
        and (not workers_alive)
        and app._event_queue.empty()
        and (not app._frame_playback_active)
    )


def _save_named_plot(app: HumanoidGUI, label: str, run_index: int) -> Path:
    app.output_plot_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if (ch.isalnum() or ch in {"-", "_"}) else "_" for ch in label)
    out_path = app.output_plot_dir / f"Humanoid_sequential_{run_index:02d}_{safe_label}_{ts}.png"
    app.figure.savefig(str(out_path), dpi=120)
    return out_path


def _parse_td3_sigma_list(raw: str) -> List[float]:
    values: List[float] = []
    for chunk in str(raw).split(","):
        text = chunk.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("--td3-sigmas must contain at least one numeric value.")
    return values


def run_sequential_defaults(
    timeout_s: Optional[float],
    episodes: int | None,
    max_steps: int | None,
    td3_sigmas: Optional[List[float]] = None,
) -> None:
    root = tk.Tk()
    app = HumanoidGUI(root)

    # Prevent modal popups from blocking automation; errors are reported via status checks.
    messagebox.showerror = lambda *_args, **_kwargs: None

    app.reset_all()
    app.compare_on_var.set(False)
    # Default UI behavior: animation starts OFF.
    app.animation_on_var.set(False)

    if episodes is not None:
        app.episodes_var.set(str(max(1, episodes)))
    if max_steps is not None:
        app.max_steps_var.set(str(max(1, max_steps)))

    print("Starting sequential GUI benchmark (animation off, cumulative plot)...", flush=True)
    print(f"episodes={app.episodes_var.get()}, max_steps={app.max_steps_var.get()}", flush=True)

    run_plan: List[Dict[str, Any]] = []
    if td3_sigmas:
        for sigma in td3_sigmas:
            run_plan.append(
                {
                    "policy": "TD3",
                    "label": f"TD3_sigma_{sigma}",
                    "overrides": {"action_noise_sigma": str(sigma)},
                }
            )
    else:
        for policy in POLICIES:
            run_plan.append({"policy": policy, "label": policy, "overrides": {}})

    saved_paths: list[Path] = []
    run_idx = 0
    run_started_at = 0.0
    stable_idle_ticks = 0
    failure: Optional[str] = None

    def _finish(fail_message: Optional[str] = None) -> None:
        nonlocal failure
        failure = fail_message
        try:
            app.on_close()
        finally:
            try:
                root.quit()
            except Exception:
                pass

    def _start_next() -> None:
        nonlocal run_idx, run_started_at, stable_idle_ticks

        if run_idx >= len(run_plan):
            print("Sequential GUI benchmark completed.", flush=True)
            print("Saved plots:", flush=True)
            for p in saved_paths:
                print(f" - {p}", flush=True)
            print("finished", flush=True)
            _finish()
            return

        if not _is_fully_idle(app):
            root.after(120, _start_next)
            return

        run_cfg = run_plan[run_idx]
        policy = str(run_cfg["policy"])
        app.policy_var.set(policy)
        app._populate_specific_panel()

        for key, value in dict(run_cfg.get("overrides", {})).items():
            if key in app.policy_param_vars[policy]:
                app.policy_param_vars[policy][key].set(str(value))

        print(f"Run {run_idx + 1}/{len(run_plan)}: policy={policy} | label={run_cfg['label']}", flush=True)

        app.train_and_run()
        run_started_at = time.perf_counter()
        stable_idle_ticks = 0
        root.after(120, _monitor_current)

    def _monitor_current() -> None:
        nonlocal run_idx, stable_idle_ticks

        run_cfg = run_plan[run_idx]
        policy = str(run_cfg["policy"])

        if timeout_s is not None and (time.perf_counter() - run_started_at) > timeout_s:
            _finish(f"{policy} did not finish cleanly: timeout after {timeout_s:.1f}s")
            return

        if _is_fully_idle(app):
            stable_idle_ticks += 1
            if stable_idle_ticks >= 5:
                last_error = str(getattr(app, "_last_error_message", "") or "").strip()
                if last_error:
                    _finish(f"{policy} failed: {last_error}")
                    return

                out_path = _save_named_plot(app, str(run_cfg["label"]), run_idx + 1)
                saved_paths.append(out_path)
                print(f"Saved cumulative plot: {out_path}", flush=True)

                run_idx += 1
                stable_idle_ticks = 0
                root.after(120, _start_next)
                return
        else:
            stable_idle_ticks = 0

        root.after(120, _monitor_current)

    root.after(120, _start_next)
    root.mainloop()

    if failure:
        raise RuntimeError(failure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sequential GUI benchmark: run PPO, SAC, TD3 one-after-another in one GUI session",
    )
    parser.add_argument("--timeout", type=float, default=7200.0, help="Timeout per policy run in seconds")
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Optional episode override (default: use GUI default)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional max-step override (default: use GUI default)",
    )
    parser.add_argument(
        "--td3-sigmas",
        type=str,
        default=None,
        help="Optional TD3-only run mode. Comma-separated action_noise_sigma values (e.g. 0.03,0.05,0.1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    timeout_s: Optional[float] = None if args.timeout <= 0 else max(30.0, args.timeout)
    td3_sigmas = _parse_td3_sigma_list(args.td3_sigmas) if args.td3_sigmas else None
    run_sequential_defaults(
        timeout_s=timeout_s,
        episodes=args.episodes,
        max_steps=args.max_steps,
        td3_sigmas=td3_sigmas,
    )


if __name__ == "__main__":
    main()
