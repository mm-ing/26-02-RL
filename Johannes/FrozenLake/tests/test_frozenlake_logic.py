from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from frozenlake_logic import DQN, DoubleDQN, DuelingDQN, FrozenLakeEnv, PrioDQN, Trainer, Transition


def test_frozenlake_step_returns_expected_shape() -> None:
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4", success_rate=1.0, render_mode="rgb_array")
    state = env.reset(seed=123)
    assert isinstance(state, int)

    next_state, reward, done, info = env.step(2)
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    env.close()


def test_is_reachable_on_standard_map() -> None:
    env = FrozenLakeEnv(is_slippery=True, map_name="4x4", success_rate=1.0 / 3.0, render_mode="rgb_array")
    assert env.is_reachable() is True
    env.close()


def test_run_episode_for_all_policies() -> None:
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4", success_rate=1.0, render_mode="rgb_array")
    trainer = Trainer(env, base_dir=Path(__file__).resolve().parents[1])

    policies = [
        DQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=8, replay_buffer_size=256, warmup_steps=8),
        DoubleDQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=8, replay_buffer_size=256, warmup_steps=8),
        DuelingDQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=8, replay_buffer_size=256, warmup_steps=8),
        PrioDQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=8, replay_buffer_size=256, warmup_steps=8),
    ]

    for policy in policies:
        result = trainer.run_episode(policy=policy, epsilon=0.3, max_steps=20)
        assert "total_reward" in result
        assert "steps" in result
        assert "transitions" in result
        assert isinstance(result["steps"], int)

    env.close()


def test_learning_updates_weights_dqn() -> None:
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4", success_rate=1.0, render_mode="rgb_array")
    policy = DQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=4, replay_buffer_size=64, warmup_steps=4)

    for _ in range(8):
        tr = Transition(state=0, action=1, next_state=1, reward=0.0, done=0.0)
        policy.observe(tr)

    before = [p.detach().cpu().clone() for p in policy.online_net.parameters()]
    loss = policy.learn()
    after = [p.detach().cpu().clone() for p in policy.online_net.parameters()]

    assert loss is None or isinstance(loss, float)
    changed = any(not np.allclose(b.numpy(), a.numpy()) for b, a in zip(before, after))
    assert changed or loss is None
    env.close()


def test_prioritized_replay_sampling_smoke() -> None:
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4", success_rate=1.0, render_mode="rgb_array")
    policy = PrioDQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=4, replay_buffer_size=64, warmup_steps=4)

    for i in range(12):
        tr = Transition(state=i % env.n_states, action=i % env.n_actions, next_state=(i + 1) % env.n_states, reward=0.0, done=0.0)
        policy.observe(tr)

    loss = policy.learn()
    assert loss is None or isinstance(loss, float)
    assert len(policy.replay_buffer) >= 12
    env.close()


def test_train_saves_csv(tmp_path: Path) -> None:
    env = FrozenLakeEnv(is_slippery=False, map_name="4x4", success_rate=1.0, render_mode="rgb_array")
    trainer = Trainer(env, base_dir=tmp_path)
    policy = DQN(n_states=env.n_states, n_actions=env.n_actions, batch_size=4, replay_buffer_size=64, warmup_steps=4)

    rewards, csv_path = trainer.train(policy=policy, num_episodes=3, max_steps=10, epsilon=0.2, save_csv="unit_test")
    assert len(rewards) == 3
    assert csv_path is not None
    assert csv_path.exists()
    env.close()
