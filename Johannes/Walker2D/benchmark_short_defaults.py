import time
from Walker2D_logic import Walker2DTrainer, Walker2DEnvConfig, TrainConfig, POLICY_DEFAULTS, POLICY_SHARED_DEFAULTS


def run_policy(policy: str, episodes: int = 1, max_steps: int = 300) -> tuple[str, float, int, float]:
    env_cfg = Walker2DEnvConfig(env_id="Walker2d-v5", render_mode=None)
    train_cfg = TrainConfig(
        policy_name=policy,
        episodes=episodes,
        max_steps=max_steps,
        animation_on=False,
        collect_transitions=False,
        deterministic_eval_every=10,
        device="CPU",
        shared_params=dict(POLICY_SHARED_DEFAULTS[policy]),
        specific_params=dict(POLICY_DEFAULTS[policy]),
        run_id=f"bench_{policy.lower()}",
    )
    trainer = Walker2DTrainer(env_config=env_cfg, train_config=train_cfg)
    start = time.perf_counter()
    done = trainer.train()
    elapsed = time.perf_counter() - start
    return policy, elapsed, int(done.get("episodes_done", 0)), float(done.get("best_reward", 0.0))


if __name__ == "__main__":
    print("policy,seconds,episodes_done,best_reward")
    for policy_name in ["PPO", "SAC", "TD3"]:
        try:
            policy, seconds, episodes_done, best_reward = run_policy(policy_name)
            print(f"{policy},{seconds:.3f},{episodes_done},{best_reward:.3f}")
        except Exception as exc:
            print(f"{policy_name},ERROR,0,{exc}")
