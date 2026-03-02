import argparse
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

from BipedalWalker_logic import BipedalWalkerConfig, BipedalWalkerTrainer, get_policy_default_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("policy", type=str)
    args = parser.parse_args()

    policy = args.policy
    defaults = get_policy_default_configs()

    cfg = BipedalWalkerConfig(policy=policy)
    for key, value in defaults.get(policy, {}).items():
        setattr(cfg, key, value)
    cfg.episodes = 1

    trainer = BipedalWalkerTrainer(cfg)
    trainer.train(collect_transitions=False, run_label=f"bench-{policy}")


if __name__ == "__main__":
    main()
