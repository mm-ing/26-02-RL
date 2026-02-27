import random
import numpy as np
import torch
from LunarLander_logic import Trainer, POLICY_DEFAULTS

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

policies = ["PPO", "A2C", "TRPO", "SAC"]
candidates = {
    "PPO": [0.2, 0.3],
    "A2C": [0.2, 0.3],
    "TRPO": [0.3, 0.4],
    "SAC": [0.2, 0.3],
}

episodes = 80
max_steps = 500
results = []

for policy in policies:
    for decay in candidates[policy]:
        trainer = Trainer()
        trainer.rebuild_environment(-10.0, True, False, 15.0, 1.5)
        cfg = POLICY_DEFAULTS[policy]
        trainer.set_policy_config(
            policy,
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            replay_size=cfg.replay_size,
            batch_size=cfg.batch_size,
            target_update=cfg.target_update,
            replay_warmup=cfg.replay_warmup,
            learning_cadence=cfg.learning_cadence,
            activation_function=cfg.activation_function,
            hidden_layers=cfg.hidden_layers,
            lr_strategy=cfg.lr_strategy,
            lr_decay=decay,
            min_learning_rate=cfg.min_learning_rate,
        )
        trainer.set_training_plan(policy, episodes=episodes, max_steps=max_steps)
        trainer.reset_policy_agent(policy)
        rewards = trainer.train(policy=policy, num_episodes=episodes, max_steps=max_steps, epsilon=0.0)
        tail20 = float(np.mean(rewards[-20:]))
        full = float(np.mean(rewards))
        results.append((policy, decay, tail20, full))
        print(f"{policy} decay={decay} tail20={tail20:.3f} mean={full:.3f}")
        trainer.close()

print("BEST")
for policy in policies:
    rows = [r for r in results if r[0] == policy]
    best = max(rows, key=lambda x: x[2])
    print(f"{policy} best_decay={best[1]} tail20={best[2]:.3f} mean={best[3]:.3f}")