from pathlib import Path

from cliffwalking_logic import DDQNetwork, DQNetwork, GridWorld, Trainer, Transition


def test_step_into_cliff_resets_to_start_and_penalizes():
    env = GridWorld(slippery=False, render_mode=None, seed=42)
    start_state = env.reset(seed=42)
    assert start_state == env.start_state == 36

    next_state, reward, done, info = env.step(1)
    assert next_state == env.start_state
    assert reward == -100.0
    assert done is False
    assert info["is_cliff"] is True
    env.close()


def test_reaching_goal_terminates_episode():
    env = GridWorld(slippery=False, render_mode=None, seed=1)
    env.reset(seed=1)
    env.env.unwrapped.s = 35
    next_state, reward, done, _ = env.step(2)

    assert next_state == env.goal_state
    assert reward == -1.0
    assert done is True
    env.close()


def test_slippery_changes_action_perpendicular_when_forced():
    env = GridWorld(slippery=True, slip_probability=1.0, render_mode=None, seed=7)
    env.reset(seed=7)

    _, _, _, info = env.step(0)
    assert info["intended_action"] == 0
    assert info["actual_action"] in (1, 3)
    env.close()


def test_trainer_run_episode_dqn_executes():
    env = GridWorld(slippery=False, render_mode=None, seed=11)
    policy = DQNetwork(
        n_states=env.n_states,
        n_actions=env.n_actions,
        batch_size=2,
        replay_buffer_size=50,
        hidden_neurons=32,
        target_update_frequency=5,
    )
    trainer = Trainer(env, base_dir=Path(__file__).resolve().parents[1])

    result = trainer.run_episode(policy=policy, epsilon=0.5, max_steps=20)
    assert result["steps"] > 0
    assert isinstance(result["total_reward"], float)
    assert len(result["transitions"]) == result["steps"]
    env.close()


def test_trainer_run_episode_ddqn_executes():
    env = GridWorld(slippery=False, render_mode=None, seed=12)
    policy = DDQNetwork(
        n_states=env.n_states,
        n_actions=env.n_actions,
        batch_size=2,
        replay_buffer_size=50,
        hidden_neurons=32,
        target_update_frequency=5,
    )
    trainer = Trainer(env, base_dir=Path(__file__).resolve().parents[1])

    result = trainer.run_episode(policy=policy, epsilon=0.5, max_steps=20)
    assert result["steps"] > 0
    assert isinstance(result["total_reward"], float)
    env.close()


def test_learning_update_path_for_dqn_and_ddqn():
    env = GridWorld(slippery=False, render_mode=None, seed=123)
    dqn = DQNetwork(n_states=env.n_states, n_actions=env.n_actions, batch_size=2, replay_buffer_size=20, hidden_neurons=16)
    ddqn = DDQNetwork(n_states=env.n_states, n_actions=env.n_actions, batch_size=2, replay_buffer_size=20, hidden_neurons=16)

    sample_transitions = [
        Transition(36, 0, 24, -1.0, 0.0),
        Transition(24, 1, 25, -1.0, 0.0),
        Transition(25, 1, 26, -1.0, 0.0),
    ]

    for transition in sample_transitions:
        dqn.observe(transition)
        ddqn.observe(transition)

    dqn_loss = dqn.learn()
    ddqn_loss = ddqn.learn()

    assert dqn_loss is not None
    assert ddqn_loss is not None
    assert dqn_loss >= 0.0
    assert ddqn_loss >= 0.0
    env.close()
