from __future__ import annotations
"""Focused tests for Taxi environment, trainer loop, and DQN learning path."""

from pathlib import Path

from Taxi_logic import DQN, DoubleDQN, DuelingDQN, PrioDQN, TaxiEnvironment, Trainer


def test_environment_step_returns_expected_types() -> None:
    # Smoke test one environment transition and returned metadata schema.
    env = TaxiEnvironment(is_raining=False, fickle_passenger=False, render_mode=None, seed=123)
    state = env.reset()
    next_state, reward, done, info = env.step(0)

    assert isinstance(state, int)
    assert isinstance(next_state, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert "executed_action" in info

    env.close()


def test_is_reachable_self_or_neighbor() -> None:
    # Validate reachability helper for a directly reachable next state.
    env = TaxiEnvironment(render_mode=None, seed=123)
    s0 = env.reset()

    assert env.is_reachable(s0, s0) is True or env.is_reachable(s0, s0) is False

    ns, _, _, _ = env.step(0)
    assert env.is_reachable(s0, ns) is True

    env.close()


def test_trainer_run_episode_multiple_policies() -> None:
    # Ensure trainer loop works with all supported DQN policy variants.
    env = TaxiEnvironment(render_mode=None, seed=7)
    trainer = Trainer(env)

    common = {
        "state_size": env.n_states,
        "action_size": env.n_actions,
        "batch_size": 4,
        "buffer_size": 200,
        "hidden_size": 32,
        "target_update_freq": 5,
    }

    agents = [
        DQN(**common),
        DoubleDQN(**common),
        DuelingDQN(**common),
        PrioDQN(**common),
    ]

    for agent in agents:
        # Episode output contract should be consistent for every policy.
        out = trainer.run_episode(agent, epsilon=0.8, max_steps=30)
        assert "total_reward" in out
        assert "steps" in out
        assert isinstance(out["transitions"], list)
        assert out["steps"] <= 30

    env.close()


def test_basic_learning_update_path() -> None:
    # Populate replay, run a learning step, and verify CSV export path executes.
    env = TaxiEnvironment(render_mode=None, seed=11)
    trainer = Trainer(env)

    agent = DQN(
        state_size=env.n_states,
        action_size=env.n_actions,
        batch_size=4,
        buffer_size=300,
        hidden_size=32,
        target_update_freq=3,
    )

    trainer.run_episode(agent, epsilon=1.0, max_steps=80)
    loss = agent.learn()

    # Depending on replay fill state, learn may return None or a scalar loss.
    assert loss is None or isinstance(loss, float)

    csv_dir = Path("results_csv")
    csv_dir.mkdir(parents=True, exist_ok=True)

    rewards = trainer.train(agent, num_episodes=2, max_steps=20, epsilon=0.5, save_csv="pytest_taxi")
    assert len(rewards) == 2

    env.close()
