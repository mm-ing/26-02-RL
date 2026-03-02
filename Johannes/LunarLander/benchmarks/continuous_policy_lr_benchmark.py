import random
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from LunarLander_logic import POLICY_DEFAULTS, Trainer


@dataclass
class Candidate:
    name: str
    learning_rate: float
    lr_strategy: str
    lr_decay: float


def run_benchmark() -> None:
    policies = ["PPO", "A2C", "TRPO", "SAC"]
    seeds = [0, 1, 2]
    episodes = 25
    max_steps = 300

    candidates: Dict[str, List[Candidate]] = {
        "PPO": [
            Candidate("default", 1e-4, "linear", 0.3),
            Candidate("alt", 7.5e-5, "linear", 0.3),
        ],
        "A2C": [
            Candidate("default", 1.5e-4, "exponential", 0.3),
            Candidate("alt", 1e-4, "exponential", 0.3),
        ],
        "TRPO": [
            Candidate("default", 7.5e-5, "linear", 0.4),
            Candidate("alt", 1e-4, "linear", 0.4),
        ],
        "SAC": [
            Candidate("default", 1e-4, "cosine", 0.3),
            Candidate("alt", 7.5e-5, "cosine", 0.3),
        ],
    }

    rows = []
    for policy in policies:
        cfg = POLICY_DEFAULTS[policy]
        for candidate in candidates[policy]:
            tail_scores = []
            for seed in seeds:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                trainer = Trainer()
                trainer.rebuild_environment(-10.0, True, False, 15.0, 1.5)
                trainer.set_policy_config(
                    policy,
                    gamma=cfg.gamma,
                    learning_rate=candidate.learning_rate,
                    replay_size=cfg.replay_size,
                    batch_size=cfg.batch_size,
                    target_update=cfg.target_update,
                    replay_warmup=cfg.replay_warmup,
                    learning_cadence=cfg.learning_cadence,
                    activation_function=cfg.activation_function,
                    hidden_layers=cfg.hidden_layers,
                    lr_strategy=candidate.lr_strategy,
                    lr_decay=candidate.lr_decay,
                    min_learning_rate=cfg.min_learning_rate,
                )
                trainer.set_training_plan(policy, episodes=episodes, max_steps=max_steps)
                trainer.reset_policy_agent(policy)
                rewards = trainer.train(policy=policy, num_episodes=episodes, max_steps=max_steps, epsilon=0.0)
                tail8 = float(np.mean(rewards[-8:]))
                tail_scores.append(tail8)
                trainer.close()

            median_tail8 = float(np.median(tail_scores))
            mean_tail8 = float(np.mean(tail_scores))
            rows.append((policy, candidate, median_tail8, mean_tail8, tail_scores))
            print(
                f"{policy:4s} {candidate.name:7s} "
                f"lr={candidate.learning_rate:.2e} strat={candidate.lr_strategy:11s} decay={candidate.lr_decay:.2f} "
                f"median_tail8={median_tail8:8.3f} mean_tail8={mean_tail8:8.3f} seeds={tail_scores}"
            )

    print("\nWINNERS")
    for policy in policies:
        policy_rows = [row for row in rows if row[0] == policy]
        best = max(policy_rows, key=lambda row: row[2])
        cand = best[1]
        print(
            f"{policy}: winner={cand.name} lr={cand.learning_rate:.2e} "
            f"strat={cand.lr_strategy} decay={cand.lr_decay:.2f} median_tail8={best[2]:.3f}"
        )


if __name__ == "__main__":
    run_benchmark()
