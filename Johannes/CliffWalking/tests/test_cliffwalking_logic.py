import numpy as np

from cliffwalking_logic import CliffWalkingEnv, DQNetwork, DDQNetwork, Trainer


def test_environment_cliff_and_goal_behavior():
    env = CliffWalkingEnv(slippery=False)

    state = env.reset()
    assert state == env.start_state

    next_state, reward, done, _ = env.step(1)
    assert next_state == env.start_state
    assert reward == -100.0
    assert done is False

    env.reset()
    env.step(0)
    for _ in range(11):
        env.step(1)
    next_state, reward, done, _ = env.step(2)
    assert next_state == env.goal_state
    assert reward == -1.0
    assert done is True

    env.close()


def test_trainer_run_episode_for_dqn_and_ddqn():
    env = CliffWalkingEnv(slippery=False)
    trainer = Trainer(env)

    dqn = DQNetwork(batch_size=2)
    reward_dqn, transitions_dqn = trainer.run_episode(dqn, epsilon=1.0, max_steps=20)
    assert isinstance(reward_dqn, float)
    assert len(transitions_dqn) > 0

    ddqn = DDQNetwork(batch_size=2)
    reward_ddqn, transitions_ddqn = trainer.run_episode(ddqn, epsilon=1.0, max_steps=20)
    assert isinstance(reward_ddqn, float)
    assert len(transitions_ddqn) > 0

    env.close()


def _fill_buffer_and_optimize(policy):
    for _ in range(policy.batch_size + 1):
        state = np.array([0.0, 0.0], dtype=np.float32)
        next_state = np.array([0.0, 0.1], dtype=np.float32)
        policy.remember(state, 1, next_state, -1.0, False)
    return policy.optimize()


def test_dqn_learning_update_changes_weights():
    policy = DQNetwork(batch_size=4)
    before = [p.detach().cpu().clone() for p in policy.online_net.parameters()]

    loss = _fill_buffer_and_optimize(policy)

    after = [p.detach().cpu().clone() for p in policy.online_net.parameters()]
    changed = any(not np.allclose(b.numpy(), a.numpy()) for b, a in zip(before, after))

    assert loss is not None
    assert changed


def test_ddqn_learning_update_changes_weights():
    policy = DDQNetwork(batch_size=4)
    before = [p.detach().cpu().clone() for p in policy.online_net.parameters()]

    loss = _fill_buffer_and_optimize(policy)

    after = [p.detach().cpu().clone() for p in policy.online_net.parameters()]
    changed = any(not np.allclose(b.numpy(), a.numpy()) for b, a in zip(before, after))

    assert loss is not None
    assert changed
