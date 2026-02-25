from pathlib import Path

import numpy as np
import pytest
import torch

from CartPole_logic import CartPoleEnvironment, D3QN, DoubleDQN, DuelingDQN, Trainer, set_global_seed


@pytest.fixture(scope="module")
def env():
    set_global_seed(7)
    instance = CartPoleEnvironment(sutton_barto_reward=False, seed=7)
    yield instance
    instance.close()


def test_step_returns_valid_types(env):
    state = env.reset(seed=7)
    assert state.shape == (4,)
    next_state, reward, terminated, truncated, info = env.step(0)
    assert isinstance(next_state, np.ndarray)
    assert next_state.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_is_reachable(env):
    valid_state = np.array([0.0, 0.2, 0.01, -0.3], dtype=np.float32)
    invalid_state = np.array([10.0, 0.0, 0.0, 0.0], dtype=np.float32)
    assert env.is_reachable(valid_state) is True
    assert env.is_reachable(invalid_state) is False


@pytest.mark.parametrize("agent_cls", [DoubleDQN, DuelingDQN, D3QN])
def test_run_episode_for_each_policy(env, agent_cls):
    agent = agent_cls(env.state_size, env.action_size)
    trainer = Trainer(env, output_dir=Path(__file__).resolve().parents[1])

    callbacks = []

    def on_step(step):
        callbacks.append(step)

    reward, transitions = trainer.run_episode(agent, epsilon=0.2, max_steps=20, progress_callback=on_step)
    assert isinstance(reward, float)
    assert len(transitions) >= 1
    assert len(callbacks) >= 1


def test_learning_updates_weights(env):
    set_global_seed(11)
    agent = DoubleDQN(env.state_size, env.action_size)

    for _ in range(1300):
        state = np.random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        next_state = np.random.uniform(-0.05, 0.05, size=(4,)).astype(np.float32)
        action = np.random.randint(0, env.action_size)
        reward = float(np.random.randn())
        done = bool(np.random.rand() < 0.1)
        agent.remember((state, action, reward, next_state, done))

    before = [param.detach().clone() for param in agent.online_net.parameters()]
    loss = agent.learn()
    after = [param.detach().clone() for param in agent.online_net.parameters()]

    assert loss is not None
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
