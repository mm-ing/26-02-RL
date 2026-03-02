"""Debug script to test training job directly."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from bipedal_walker_logic import (
    A2CConfig, AlgorithmType, EpisodeConfig, EnvironmentConfig, EventBus,
    EventType, Event, JobConfig, NetworkConfig, PPOConfig, SACConfig, TD3Config,
    TrainingJob, TuningConfig,
)


def make_fast_config(algo, n_episodes=5):
    ep_cfg = EpisodeConfig(n_episodes=n_episodes, max_steps=50, alpha=3e-4, gamma=0.99)
    net = NetworkConfig(hidden_layers=[32, 32], activation="relu")
    return JobConfig(
        name=f"test-{algo}",
        algorithm=algo,
        env_cfg=EnvironmentConfig(env_name="BipedalWalker-v3", hardcore=False),
        ep_cfg=ep_cfg,
        ppo_cfg=PPOConfig(n_steps=128, batch_size=32, n_epochs=2, network=net),
        a2c_cfg=A2CConfig(n_steps=5, network=net),
        sac_cfg=SACConfig(buffer_size=10_000, batch_size=32, learning_starts=10,
                          train_freq=1, gradient_steps=1, network=net),
        td3_cfg=TD3Config(buffer_size=10_000, batch_size=32, learning_starts=10,
                          train_freq=1, gradient_steps=1, network=net),
    )


# Test PPO
print("Building PPO config...")
cfg = make_fast_config(AlgorithmType.PPO.value, n_episodes=5)
print(f"Config: n_episodes={cfg.ep_cfg.n_episodes}, max_steps={cfg.ep_cfg.max_steps}")
print(f"total_timesteps = {cfg.ep_cfg.n_episodes * cfg.ep_cfg.max_steps}")

bus = EventBus()
episodes_received = []

def on_episode(e: Event):
    episodes_received.append(e.data)
    print(f"Episode {e.data['episode']}: return={e.data['return']:.2f}")

bus.subscribe(EventType.EPISODE_COMPLETED, on_episode)

job = TrainingJob(cfg, bus)
print("Building model...")
job.model = job._build_model()
print("Model built. Starting training thread...")
job.start()

deadline = time.time() + 60
while job.is_alive() and time.time() < deadline:
    bus.drain()
    time.sleep(0.5)
    print(f"  Thread alive: {job.is_alive()}, episodes so far: {len(job.returns)}", flush=True)

print(f"\n=== DONE ===")
print(f"Job status: {job.status}")
print(f"Episodes completed: {len(job.returns)}")
if job.returns:
    print(f"Returns: {job.returns}")
else:
    print("No returns recorded!")
