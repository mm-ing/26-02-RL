from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from typing import Dict, List

if "--force-cpu" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MountainCar_logic import make_default_trainer


def evaluate_policy(trainer, policy: str, episodes: int = 30, max_steps: int = 200) -> Dict[str, float]:
    rewards: List[float] = []
    successes = 0
    max_positions: List[float] = []
    for _ in range(episodes):
        result = trainer.run_episode(policy=policy, epsilon=0.0, max_steps=max_steps)
        rewards.append(float(result["total_reward"]))
        successes += int(bool(result.get("reached_goal", False)))
        max_positions.append(float(result.get("max_position", -1.2)))
    return {
        "reward_mean": float(statistics.mean(rewards)),
        "success_rate": float(successes / max(1, episodes)),
        "max_pos_mean": float(statistics.mean(max_positions)),
    }


def train_with_decay(
    trainer,
    policy: str,
    episodes: int,
    max_steps: int,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
) -> Dict[str, List[float]]:
    rewards: List[float] = []
    max_positions: List[float] = []
    successes: List[float] = []
    epsilon = epsilon_start
    for _ in range(episodes):
        result = trainer.run_episode(policy=policy, epsilon=epsilon, max_steps=max_steps)
        rewards.append(float(result["total_reward"]))
        max_positions.append(float(result.get("max_position", -1.2)))
        successes.append(float(1.0 if bool(result.get("reached_goal", False)) else 0.0))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
    return {"rewards": rewards, "max_positions": max_positions, "successes": successes}


def train_with_decay_checkpoints(
    trainer,
    policy: str,
    episodes: int,
    max_steps: int,
    epsilon_start: float,
    epsilon_decay: float,
    epsilon_min: float,
    checkpoint_every: int,
) -> Dict[str, object]:
    rewards: List[float] = []
    max_positions: List[float] = []
    successes: List[float] = []
    checkpoints: List[Dict[str, float]] = []
    epsilon = epsilon_start

    for idx in range(episodes):
        result = trainer.run_episode(policy=policy, epsilon=epsilon, max_steps=max_steps)
        rewards.append(float(result["total_reward"]))
        max_positions.append(float(result.get("max_position", -1.2)))
        successes.append(float(1.0 if bool(result.get("reached_goal", False)) else 0.0))
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        ep = idx + 1
        if checkpoint_every > 0 and (ep % checkpoint_every == 0 or ep == episodes):
            window = min(checkpoint_every, len(rewards))
            checkpoints.append(
                {
                    "episode": float(ep),
                    "reward_mean": float(statistics.mean(rewards[-window:])),
                    "max_pos_mean": float(statistics.mean(max_positions[-window:])),
                    "success_rate": float(statistics.mean(successes[-window:])),
                    "epsilon": float(epsilon),
                }
            )

    return {
        "rewards": rewards,
        "max_positions": max_positions,
        "successes": successes,
        "checkpoints": checkpoints,
    }


def run_one(policy: str, episodes: int = 400, max_steps: int = 200, eval_episodes: int = 20) -> Dict[str, float]:
    trainer = make_default_trainer(seed=42)

    p = dict(trainer.policy_defaults[policy])
    eps_start = 1.0
    eps_decay = 0.997 if policy == "Dueling DQN" else (0.998 if policy == "D3QN" else 0.9985)
    eps_min = 0.02

    baseline_eval = evaluate_policy(trainer, policy, episodes=eval_episodes, max_steps=max_steps)
    train_data = train_with_decay(
        trainer,
        policy=policy,
        episodes=episodes,
        max_steps=max_steps,
        epsilon_start=eps_start,
        epsilon_decay=eps_decay,
        epsilon_min=eps_min,
    )
    post_eval = evaluate_policy(trainer, policy, episodes=eval_episodes, max_steps=max_steps)
    train_rewards = train_data["rewards"]
    train_max_pos = train_data["max_positions"]
    train_success = train_data["successes"]

    window = min(50, max(1, len(train_rewards)))
    first_50 = float(statistics.mean(train_rewards[:window]))
    last_50 = float(statistics.mean(train_rewards[-window:]))

    trainer.environment.close()

    return {
        "policy": policy,
        "baseline_greedy": baseline_eval["reward_mean"],
        "post_greedy": post_eval["reward_mean"],
        "baseline_success_rate": baseline_eval["success_rate"],
        "post_success_rate": post_eval["success_rate"],
        "baseline_max_pos": baseline_eval["max_pos_mean"],
        "post_max_pos": post_eval["max_pos_mean"],
        "train_first_50": first_50,
        "train_last_50": last_50,
        "train_first_max_pos": float(statistics.mean(train_max_pos[:window])),
        "train_last_max_pos": float(statistics.mean(train_max_pos[-window:])),
        "train_success_rate": float(statistics.mean(train_success)),
        "delta_train": last_50 - first_50,
        "delta_eval": post_eval["reward_mean"] - baseline_eval["reward_mean"],
        "episodes": float(episodes),
        "max_steps": float(max_steps),
        "epsilon_decay": eps_decay,
        "learning_rate": float(p["learning_rate"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MountainCar training diagnostics")
    parser.add_argument("--episodes", type=int, default=300)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--profile", choices=["default", "fast"], default="fast")
    parser.add_argument("--mode", choices=["standard", "long"], default="standard")
    parser.add_argument("--force-cpu", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=20)
    parser.add_argument("--policies", nargs="*", default=["Dueling DQN", "D3QN", "DDQN+PER"])
    args = parser.parse_args()

    if args.mode == "long":
        if parser.get_default("episodes") == args.episodes:
            args.episodes = 150
        if parser.get_default("eval_episodes") == args.eval_episodes:
            args.eval_episodes = 5
        if parser.get_default("max_steps") == args.max_steps:
            args.max_steps = 200
        if parser.get_default("checkpoint_every") == args.checkpoint_every:
            args.checkpoint_every = 25

    policies = args.policies
    print("=== MountainCar Training Diagnostics ===")
    print("Metric note: less negative reward is better (e.g. -140 > -200).")

    def apply_fast_profile(policy_name: str, trainer_obj) -> None:
        base = dict(trainer_obj.policy_defaults[policy_name])
        base.update(
            {
                "learning_rate": 1e-3,
                "batch_size": 32,
                "replay_size": 20_000,
                "target_update": 100,
                "hidden_layer_size": [128, 128],
            }
        )
        if policy_name == "DDQN+PER":
            base.update(
                {
                    "batch_size": 32,
                    "beta_frames": 60_000,
                }
            )
        trainer_obj.get_or_create_agent(policy_name, overrides=base)

    all_results: List[Dict[str, float]] = []

    for policy in policies:
        print(f"Running policy: {policy}", flush=True)
        if args.profile == "fast":
            trainer = make_default_trainer(seed=42)
            apply_fast_profile(policy, trainer)
            p = dict(trainer.policy_defaults[policy])
            eps_decay = 0.995 if policy == "Dueling DQN" else (0.996 if policy == "D3QN" else 0.997)
            print("  - baseline eval...", flush=True)
            base = evaluate_policy(trainer, policy, episodes=max(1, int(args.eval_episodes)), max_steps=max(1, int(args.max_steps)))
            print("  - training...", flush=True)
            if args.mode == "long":
                train_data = train_with_decay_checkpoints(
                    trainer,
                    policy=policy,
                    episodes=max(1, int(args.episodes)),
                    max_steps=max(1, int(args.max_steps)),
                    epsilon_start=1.0,
                    epsilon_decay=eps_decay,
                    epsilon_min=0.02,
                    checkpoint_every=max(1, int(args.checkpoint_every)),
                )
            else:
                train_data = train_with_decay(
                    trainer,
                    policy=policy,
                    episodes=max(1, int(args.episodes)),
                    max_steps=max(1, int(args.max_steps)),
                    epsilon_start=1.0,
                    epsilon_decay=eps_decay,
                    epsilon_min=0.02,
                )
            print("  - post eval...", flush=True)
            post = evaluate_policy(trainer, policy, episodes=max(1, int(args.eval_episodes)), max_steps=max(1, int(args.max_steps)))
            rewards = train_data["rewards"]
            train_max_pos = train_data["max_positions"]
            train_success = train_data["successes"]
            w = min(50, max(1, len(rewards)))
            result = {
                "policy": policy,
                "baseline_greedy": base["reward_mean"],
                "post_greedy": post["reward_mean"],
                "baseline_success_rate": base["success_rate"],
                "post_success_rate": post["success_rate"],
                "baseline_max_pos": base["max_pos_mean"],
                "post_max_pos": post["max_pos_mean"],
                "train_first_50": float(statistics.mean(rewards[:w])),
                "train_last_50": float(statistics.mean(rewards[-w:])),
                "train_first_max_pos": float(statistics.mean(train_max_pos[:w])),
                "train_last_max_pos": float(statistics.mean(train_max_pos[-w:])),
                "train_success_rate": float(statistics.mean(train_success)),
                "delta_train": float(statistics.mean(rewards[-w:]) - statistics.mean(rewards[:w])),
                "delta_eval": float(post["reward_mean"] - base["reward_mean"]),
                "episodes": float(args.episodes),
                "max_steps": float(args.max_steps),
                "epsilon_decay": eps_decay,
                "learning_rate": float(p["learning_rate"]),
            }
            if args.mode == "long":
                result["checkpoints"] = train_data.get("checkpoints", [])
            trainer.environment.close()
        else:
            result = run_one(
                policy,
                episodes=max(1, int(args.episodes)),
                max_steps=max(1, int(args.max_steps)),
                eval_episodes=max(1, int(args.eval_episodes)),
            )
        print("\n---", result["policy"], "---")
        print(f"baseline greedy mean reward : {result['baseline_greedy']:.2f}")
        print(f"post-train greedy mean reward: {result['post_greedy']:.2f}")
        print(f"train mean first 50 episodes : {result['train_first_50']:.2f}")
        print(f"train mean last 50 episodes  : {result['train_last_50']:.2f}")
        print(f"delta train (last-first)     : {result['delta_train']:+.2f}")
        print(f"delta eval (post-base)       : {result['delta_eval']:+.2f}")
        print(f"baseline success rate        : {result['baseline_success_rate'] * 100:.1f}%")
        print(f"post-train success rate      : {result['post_success_rate'] * 100:.1f}%")
        print(f"train success rate           : {result['train_success_rate'] * 100:.1f}%")
        print(f"baseline mean max position   : {result['baseline_max_pos']:.3f}")
        print(f"post-train mean max position : {result['post_max_pos']:.3f}")
        print(f"train max pos first/last     : {result['train_first_max_pos']:.3f} -> {result['train_last_max_pos']:.3f}")
        print(f"params: lr={result['learning_rate']}, eps_decay={result['epsilon_decay']}")
        if args.mode == "long":
            checkpoints = result.get("checkpoints", [])
            if checkpoints:
                print("checkpoint summary:")
                for cp in checkpoints:
                    print(
                        f"  ep {int(cp['episode']):4d} | reward={cp['reward_mean']:.2f} | "
                        f"max_pos={cp['max_pos_mean']:.3f} | success={cp['success_rate'] * 100:.1f}% | eps={cp['epsilon']:.4f}"
                    )

        all_results.append(result)

    if len(all_results) > 1:
        ranked = sorted(
            all_results,
            key=lambda r: (
                float(r.get("post_success_rate", 0.0)),
                float(r.get("post_max_pos", -1.2)),
                float(r.get("post_greedy", -1e9)),
            ),
            reverse=True,
        )
        best = ranked[0]
        print("\n=== Policy Ranking (best first) ===")
        for idx, row in enumerate(ranked, start=1):
            print(
                f"{idx}. {row['policy']} | success={row.get('post_success_rate', 0.0) * 100:.1f}% | "
                f"post_max_pos={row.get('post_max_pos', -1.2):.3f} | post_reward={row.get('post_greedy', -1e9):.2f}"
            )
        print(f"Recommended policy based on benchmark: {best['policy']}")


if __name__ == "__main__":
    main()
