import time

from BipedalWalker_logic import (
    BipedalWalkerConfig,
    BipedalWalkerTrainer,
    POLICY_DISPLAY_NAMES,
    get_policy_default_configs,
)


def main() -> None:
    results = []
    defaults = get_policy_default_configs()

    for policy in POLICY_DISPLAY_NAMES:
        cfg = BipedalWalkerConfig(policy=policy)
        for key, value in defaults.get(policy, {}).items():
            setattr(cfg, key, value)

        cfg.episodes = 1

        trainer = BipedalWalkerTrainer(cfg)
        start = time.perf_counter()
        output = trainer.train(collect_transitions=False, run_label=f"bench-{policy}")
        elapsed = time.perf_counter() - start

        episode_count = max(1, len(output.get("reward", [])))
        sec_per_episode = elapsed / episode_count
        steps_per_sec = cfg.max_steps / sec_per_episode if sec_per_episode > 0 else 0.0
        last_reward = output.get("reward", [0.0])[-1] if output.get("reward") else 0.0

        results.append((policy, elapsed, sec_per_episode, steps_per_sec, last_reward))

    print("policy, elapsed_s, sec_per_episode, steps_per_sec, last_reward")
    for policy, elapsed, sec_per_episode, steps_per_sec, last_reward in results:
        print(f"{policy}, {elapsed:.2f}, {sec_per_episode:.2f}, {steps_per_sec:.1f}, {last_reward:.2f}")


if __name__ == "__main__":
    main()
