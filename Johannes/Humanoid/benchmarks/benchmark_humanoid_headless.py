from __future__ import annotations

import argparse
import csv
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from Humanoid_logic import (
    EnvironmentConfig,
    HumanoidEnvWrapper,
    HumanoidTrainer,
    LearningRateConfig,
    NetworkConfig,
    POLICY_DEFAULTS,
    TrainerConfig,
)

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
    episodes_completed: int
    status: str
    error: str


class HeadlessHumanoidTrainer(HumanoidTrainer):
    """Trainer variant that never calls env.render(), keeping runs truly headless."""

    def _capture_frame(self, env, episode_frames, stride, step_idx):  # type: ignore[override]
        return


def _policy_params_for_benchmark(policy: str, max_steps: int) -> Dict[str, object]:
    # Use the same benchmark profile as previous timing runs for direct
    # comparability across reports.
    params = dict(POLICY_DEFAULTS[policy])
    if policy == "PPO":
        params["batch_size"] = 64
        params["n_steps"] = max(64, max_steps)
    elif policy in {"SAC", "TD3"}:
        params["learning_starts"] = 0
        params["train_freq"] = 1
        params["gradient_steps"] = 1
        params["batch_size"] = 64
    return params


def _train_once(policy: str, hidden_layer: str, device: str, episodes: int, max_steps: int) -> Tuple[float, int, str, str]:
    env_cfg = EnvironmentConfig(render_mode=None)
    wrapper = HumanoidEnvWrapper(env_cfg)
    trainer = HeadlessHumanoidTrainer(wrapper, event_sink=None)

    cfg = TrainerConfig(
        policy_name=policy,
        episodes=episodes,
        max_steps=max_steps,
        update_rate=1,
        frame_stride=1,
        deterministic_eval_every=max(episodes + 1, 999999),
        deterministic_eval_episodes=1,
        seed=42,
        collect_transitions=False,
        export_csv=False,
        split_aux_events=False,
        device=device,
        env=env_cfg,
        network=NetworkConfig(hidden_layer=hidden_layer, activation="relu"),
        lr=LearningRateConfig(lr_strategy="constant", learning_rate=3e-4, min_lr=1e-5, lr_decay=0.999),
        policy_params=_policy_params_for_benchmark(policy, max_steps=max_steps),
        session_id=f"headless-{int(time.time() * 1000)}",
        run_id=f"{policy}-{hidden_layer}-{device}",
    )

    t0 = time.perf_counter()
    try:
        payload = trainer.train(cfg)
        elapsed = time.perf_counter() - t0
        completed = int(payload.get("episodes_completed", 0) or 0)
        status = "ok" if completed >= episodes else "error"
        error = "" if status == "ok" else "incomplete episodes"
        return elapsed, completed, status, error
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, 0, "error", str(exc)


def _write_csv(results: List[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "hidden_layer", "device", "repeat", "elapsed_s", "episodes_completed", "status", "error"])
        for r in results:
            w.writerow([r.policy, r.hidden_layer, r.device, r.repeat, f"{r.elapsed_s:.6f}", r.episodes_completed, r.status, r.error])


def _group_ok(results: List[RunResult]) -> Dict[Tuple[str, str, str], List[float]]:
    out: Dict[Tuple[str, str, str], List[float]] = {}
    for r in results:
        if r.status != "ok":
            continue
        key = (r.policy, r.hidden_layer, r.device)
        out.setdefault(key, []).append(r.elapsed_s)
    return out


def _fmt(values: List[float]) -> Tuple[str, str]:
    if not values:
        return "-", "-"
    if len(values) == 1:
        return f"{values[0]:.3f}", "0.000"
    return f"{statistics.mean(values):.3f}", f"{statistics.stdev(values):.3f}"


def _write_md(results: List[RunResult], path: Path, repeats: int, episodes: int, max_steps: int) -> None:
    grouped = _group_ok(results)
    devices = sorted({r.device for r in results}) if results else list(DEVICES)
    lines: List[str] = []
    lines.append("# Humanoid Headless Benchmark")
    lines.append("")
    lines.append(f"Generated: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Mode: `headless` (no GUI, no animation render)")
    lines.append(f"- Episodes per run: `{episodes}`")
    lines.append(f"- Max steps per episode: `{max_steps}`")
    lines.append(f"- Repeats per combination: `{repeats}`")
    lines.append("")
    lines.append("## Results")
    lines.append("")
    lines.append("| Policy | Hidden Layers | Device | Repeats (ok/target) | Mean elapsed [s] | Std [s] |")
    lines.append("|---|---|---|---:|---:|---:|")
    for policy in POLICIES:
        for hidden in HIDDEN_LAYERS:
            for device in devices:
                vals = grouped.get((policy, hidden, device), [])
                mean_s, std_s = _fmt(vals)
                lines.append(f"| {policy} | `{hidden}` | {device} | {len(vals)}/{repeats} | {mean_s} | {std_s} |")

    lines.append("")
    lines.append("## Hidden Layer Impact")
    lines.append("")
    lines.append("| Policy | Device | Mean 256 [s] | Mean 256,256 [s] | Impact (256,256 vs 256) |")
    lines.append("|---|---|---:|---:|---:|")
    for policy in POLICIES:
        for device in devices:
            a = grouped.get((policy, "256", device), [])
            b = grouped.get((policy, "256,256", device), [])
            if not a or not b:
                lines.append(f"| {policy} | {device} | - | - | - |")
                continue
            am = float(statistics.mean(a))
            bm = float(statistics.mean(b))
            delta = (100.0 * (bm - am) / am) if am > 0 else 0.0
            lines.append(f"| {policy} | {device} | {am:.3f} | {bm:.3f} | {delta:+.1f}% |")

    failed = [r for r in results if r.status != "ok"]
    if failed:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        lines.append("| Policy | Hidden Layers | Device | Repeat | Status | Reason |")
        lines.append("|---|---|---|---:|---|---|")
        for r in failed:
            lines.append(f"| {r.policy} | `{r.hidden_layer}` | {r.device} | {r.repeat} | {r.status} | {r.error.replace('|', '/')} |")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Headless Humanoid benchmark for hidden-layer impact")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--max-steps", type=int, default=256)
    p.add_argument("--output-csv", type=str, default="benchmarks/benchmark_headless_results.csv")
    p.add_argument("--output-md", type=str, default="benchmarks/benchmark_headless.md")
    p.add_argument("--cpu-only", action="store_true")
    p.add_argument("--gpu-only", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    out_csv = (script_dir / args.output_csv).resolve()
    out_md = (script_dir / args.output_md).resolve()

    if args.cpu_only and args.gpu_only:
        raise ValueError("Use only one of --cpu-only or --gpu-only.")

    if args.cpu_only:
        run_devices: Tuple[str, ...] = ("CPU",)
    elif args.gpu_only:
        run_devices = ("GPU",)
    else:
        run_devices = DEVICES

    gpu_available = bool(torch.cuda.is_available())

    print("Starting headless benchmark...")
    print(f"repeats={args.repeats}, episodes={args.episodes}, max_steps={args.max_steps}")
    print(f"devices={','.join(run_devices)} | gpu_available={gpu_available}")

    results: List[RunResult] = []
    for device in run_devices:
        for policy in POLICIES:
            for hidden in HIDDEN_LAYERS:
                for rep in range(1, max(1, args.repeats) + 1):
                    if device == "GPU" and not gpu_available:
                        elapsed, completed, status, error = 0.0, 0, "skipped", "GPU not available"
                    else:
                        elapsed, completed, status, error = _train_once(
                            policy=policy,
                            hidden_layer=hidden,
                            device=device,
                            episodes=max(1, args.episodes),
                            max_steps=max(1, args.max_steps),
                        )
                    results.append(
                        RunResult(
                            policy=policy,
                            hidden_layer=hidden,
                            device=device,
                            repeat=rep,
                            elapsed_s=elapsed,
                            episodes_completed=completed,
                            status=status,
                            error=error,
                        )
                    )
                    if args.verbose or status != "ok":
                        print(
                            f"[{policy}] device={device:<3} hidden={hidden:<7} repeat={rep} status={status} elapsed={elapsed:.2f}s",
                            flush=True,
                        )
                _write_csv(results, out_csv)
                _write_md(results, out_md, repeats=max(1, args.repeats), episodes=max(1, args.episodes), max_steps=max(1, args.max_steps))

    ok = sum(1 for r in results if r.status == "ok")
    print(f"Benchmark complete: {ok}/{len(results)} successful runs")
    print(f"CSV written to: {out_csv}")
    print(f"Report written to: {out_md}")


if __name__ == "__main__":
    main()
