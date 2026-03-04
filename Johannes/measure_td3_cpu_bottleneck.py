from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent
WALKER_DIR = ROOT / "Walker2D"
CHEETAH_DIR = ROOT / "HalfCheetah"

sys.path.insert(0, str(WALKER_DIR))
sys.path.insert(0, str(CHEETAH_DIR))

import Walker2D_logic as w2d  # noqa: E402
import HalfCheetah_logic as hc  # noqa: E402


def _safe_mean(values: List[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def run_walker2d(episodes: int = 12, animation_on: bool = False) -> List[Dict[str, Any]]:
    state = {
        "current": {"learn": 0.0, "rollout": 0.0, "eval": 0.0},
        "rows": [],
    }

    shared = dict(w2d.POLICY_SHARED_DEFAULTS["TD3"])
    specific = dict(w2d.POLICY_DEFAULTS["TD3"])

    trainer = w2d.Walker2DTrainer(
        env_config=w2d.Walker2DEnvConfig(env_id="Walker2d-v5", render_mode=None),
        train_config=w2d.TrainConfig(
            policy_name="TD3",
            episodes=episodes,
            max_steps=1000,
            deterministic_eval_every=10,
            deterministic_eval_max_steps=300,
            animation_on=animation_on,
            low_overhead_animation=False,
            collect_transitions=False,
            device="CPU",
            shared_params=shared,
            specific_params=specific,
            run_id=f"walker_td3_{'anim_on' if animation_on else 'anim_off'}",
        ),
    )

    orig_build_model = trainer._build_model
    orig_run_episode = trainer.run_episode
    orig_evaluate = trainer.evaluate_policy
    orig_emit = trainer._emit

    def wrapped_build_model():
        model = orig_build_model()
        orig_learn = model.learn

        def timed_learn(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig_learn(*args, **kwargs)
            state["current"]["learn"] += time.perf_counter() - t0
            return out

        model.learn = timed_learn
        return model

    def wrapped_run_episode(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_run_episode(*args, **kwargs)
        dt = time.perf_counter() - t0
        is_eval_like = bool(kwargs.get("deterministic", False)) and not bool(kwargs.get("render", False))
        if not is_eval_like:
            state["current"]["rollout"] += dt
        return out

    def wrapped_evaluate(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_evaluate(*args, **kwargs)
        state["current"]["eval"] += time.perf_counter() - t0
        return out

    def wrapped_emit(payload):
        if payload.get("type") == "episode":
            row = {
                "project": "Walker2D",
                "animation_on": int(animation_on),
                "episode": int(payload.get("episode", 0)),
                "steps": int(payload.get("steps", 0)),
                "learn_seconds": state["current"]["learn"],
                "rollout_seconds": state["current"]["rollout"],
                "eval_seconds": state["current"]["eval"],
                "episode_total_seconds": state["current"]["learn"] + state["current"]["rollout"] + state["current"]["eval"],
            }
            state["rows"].append(row)
            state["current"] = {"learn": 0.0, "rollout": 0.0, "eval": 0.0}
        return orig_emit(payload)

    trainer._build_model = wrapped_build_model
    trainer.run_episode = wrapped_run_episode
    trainer.evaluate_policy = wrapped_evaluate
    trainer._emit = wrapped_emit

    trainer.train()
    return state["rows"]


def run_halfcheetah(episodes: int = 12, animation_on: bool = False) -> List[Dict[str, Any]]:
    state = {
        "current": {"learn": 0.0, "rollout": 0.0, "eval": 0.0},
        "rows": [],
    }

    trainer = hc.HalfCheetahTrainer(base_dir=str(CHEETAH_DIR))

    specific = dict(hc.POLICY_DEFAULTS["TD3"])

    orig_create_model = hc.SB3PolicyAgent.create_model
    orig_run_episode = trainer.run_episode
    orig_evaluate = trainer.evaluate_policy
    orig_emit = trainer._emit

    def wrapped_create_model(self, *args, **kwargs):
        model = orig_create_model(self, *args, **kwargs)
        orig_learn = model.learn

        def timed_learn(*l_args, **l_kwargs):
            t0 = time.perf_counter()
            out = orig_learn(*l_args, **l_kwargs)
            state["current"]["learn"] += time.perf_counter() - t0
            return out

        model.learn = timed_learn
        return model

    def wrapped_run_episode(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_run_episode(*args, **kwargs)
        dt = time.perf_counter() - t0
        if not bool(kwargs.get("deterministic", True)):
            state["current"]["rollout"] += dt
        return out

    def wrapped_evaluate(*args, **kwargs):
        t0 = time.perf_counter()
        out = orig_evaluate(*args, **kwargs)
        state["current"]["eval"] += time.perf_counter() - t0
        return out

    def wrapped_emit(event_type, payload):
        if event_type == "episode":
            row = {
                "project": "HalfCheetah",
                "animation_on": int(animation_on),
                "episode": int(payload.get("episode", 0)),
                "steps": int(payload.get("steps", 0)),
                "learn_seconds": state["current"]["learn"],
                "rollout_seconds": state["current"]["rollout"],
                "eval_seconds": state["current"]["eval"],
                "episode_total_seconds": state["current"]["learn"] + state["current"]["rollout"] + state["current"]["eval"],
            }
            state["rows"].append(row)
            state["current"] = {"learn": 0.0, "rollout": 0.0, "eval": 0.0}
        return orig_emit(event_type, payload)

    hc.SB3PolicyAgent.create_model = wrapped_create_model
    trainer.run_episode = wrapped_run_episode
    trainer.evaluate_policy = wrapped_evaluate
    trainer._emit = wrapped_emit

    try:
        trainer.train(
            {
                "policy": "TD3",
                "device": "CPU",
                "episodes": episodes,
                "max_steps": 1000,
                "gamma": hc.GENERAL_DEFAULTS["gamma"],
                "epsilon_max": 0.0,
                "epsilon_decay": 1.0,
                "epsilon_min": 0.0,
                "moving_average_values": 20,
                "update_rate_episodes": 1,
                "frame_capture_stride": 2,
                "animation_on": animation_on,
                "rollout_full_capture_steps": 120,
                "low_overhead_animation": False,
                "eval_interval": 10,
                "eval_episodes": 1,
                "run_id": f"half_td3_{'anim_on' if animation_on else 'anim_off'}",
                "seed": 42,
                "env_params": dict(hc.ENV_DEFAULTS),
                "specific_params": specific,
            }
        )
    finally:
        hc.SB3PolicyAgent.create_model = orig_create_model

    return state["rows"]


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, Any]:
    pre = [r for r in rows if r["episode"] <= 9]
    post = [r for r in rows if r["episode"] >= 11]
    all_totals = [r["episode_total_seconds"] for r in rows]

    summary = {
        "label": label,
        "episodes": len(rows),
        "avg_total": _safe_mean(all_totals),
        "avg_learn": _safe_mean([r["learn_seconds"] for r in rows]),
        "avg_rollout": _safe_mean([r["rollout_seconds"] for r in rows]),
        "avg_eval": _safe_mean([r["eval_seconds"] for r in rows]),
        "avg_total_pre9": _safe_mean([r["episode_total_seconds"] for r in pre]),
        "avg_total_post11": _safe_mean([r["episode_total_seconds"] for r in post]),
        "avg_learn_pre9": _safe_mean([r["learn_seconds"] for r in pre]),
        "avg_learn_post11": _safe_mean([r["learn_seconds"] for r in post]),
    }
    return summary


def main() -> None:
    all_rows: List[Dict[str, Any]] = []

    print("Running Walker2D TD3 CPU with animation OFF...")
    rows_walker_off = run_walker2d(episodes=12, animation_on=False)
    all_rows.extend(rows_walker_off)

    print("Running Walker2D TD3 CPU with animation ON...")
    rows_walker_on = run_walker2d(episodes=12, animation_on=True)
    all_rows.extend(rows_walker_on)

    print("Running HalfCheetah TD3 CPU with animation OFF...")
    rows_half_off = run_halfcheetah(episodes=12, animation_on=False)
    all_rows.extend(rows_half_off)

    out_csv = ROOT / "td3_cpu_bottleneck_measurements.csv"
    fieldnames = [
        "project",
        "animation_on",
        "episode",
        "steps",
        "learn_seconds",
        "rollout_seconds",
        "eval_seconds",
        "episode_total_seconds",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    summaries = [
        summarize(rows_walker_off, "Walker2D_CPU_anim_off"),
        summarize(rows_walker_on, "Walker2D_CPU_anim_on"),
        summarize(rows_half_off, "HalfCheetah_CPU_anim_off"),
    ]

    out_summary = ROOT / "td3_cpu_bottleneck_summary.csv"
    sum_fields = [
        "label",
        "episodes",
        "avg_total",
        "avg_learn",
        "avg_rollout",
        "avg_eval",
        "avg_total_pre9",
        "avg_total_post11",
        "avg_learn_pre9",
        "avg_learn_post11",
    ]
    with out_summary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=sum_fields)
        writer.writeheader()
        writer.writerows(summaries)

    print(f"WROTE,{out_csv}")
    print(f"WROTE,{out_summary}")
    for item in summaries:
        print(item)


if __name__ == "__main__":
    main()
