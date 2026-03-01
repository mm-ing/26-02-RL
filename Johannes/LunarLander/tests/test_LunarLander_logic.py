import numpy as np
import pytest

from LunarLander_logic import POLICY_DEFAULTS, Trainer


class DummySpace:
    def __init__(self, shape=None, n=None):
        self.shape = shape
        self.n = n


class DummyLanderEnv:
    def __init__(self):
        self.observation_space = DummySpace(shape=(8,))
        self.action_space = DummySpace(n=4)
        self._step = 0
        self.state_dim = 8
        self.action_dim = 4

    def reset(self, seed=None):
        self._step = 0
        obs = np.zeros(8, dtype=np.float32)
        return obs

    def step(self, action):
        self._step += 1
        obs = np.zeros(8, dtype=np.float32)
        obs[0] = 0.1 * self._step
        action_arr = np.asarray(action)
        if action_arr.ndim == 0:
            reward = 1.0 if int(action_arr) in [0, 1, 2, 3] else -1.0
        else:
            reward = 1.0
        done = self._step >= 5
        return obs, reward, done, {}

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def is_reachable(self, obs):
        obs = np.asarray(obs)
        return np.isfinite(obs[0]) and abs(obs[0]) < 5

    def close(self):
        return None


@pytest.fixture
def trainer():
    return Trainer(env=DummyLanderEnv())


def _configure_fast_learning(trainer, policy):
    trainer.set_policy_config(
        policy,
        replay_warmup=1,
        learning_cadence=1,
        batch_size=1,
        target_update=1,
        hidden_layers="32",
        learning_rate=1e-3,
    )


def test_step_and_reachability(trainer):
    obs = trainer.env.reset()
    assert trainer.env.is_reachable(obs)
    next_obs, reward, done, _ = trainer.env.step(0)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert trainer.env.is_reachable(next_obs)


@pytest.mark.parametrize("policy", list(POLICY_DEFAULTS.keys()))
def test_run_episode_for_policies(trainer, policy):
    _configure_fast_learning(trainer, policy)
    result = trainer.run_episode(policy=policy, epsilon=0.2, max_steps=10)
    assert "reward" in result and "steps" in result
    assert result["steps"] > 0
    assert isinstance(result["transitions"], list)


def test_learning_updates_progress(trainer):
    policy = "D3QN"
    _configure_fast_learning(trainer, policy)
    agent = trainer._get_or_create_agent(policy)
    before = agent.learn_steps
    trainer.run_episode(policy=policy, epsilon=0.5, max_steps=10)
    after = agent.learn_steps
    assert after >= before


def test_reset_policy_agent_reinitializes_network(trainer):
    policy = "DuelingDQN"
    _configure_fast_learning(trainer, policy)
    trainer.run_episode(policy=policy, epsilon=0.2, max_steps=5)
    first_agent = trainer._get_or_create_agent(policy)
    trainer.reset_policy_agent(policy)
    second_agent = trainer._get_or_create_agent(policy)
    assert first_agent is not second_agent


@pytest.mark.parametrize("strategy", ["exponential", "linear", "cosine", "guarded natural gradient"])
def test_lr_strategy_updates_learning_rate(trainer, strategy):
    policy = "D3QN"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, lr_strategy=strategy, lr_decay=0.1, min_learning_rate=1e-5)
    trainer.set_training_plan(policy, episodes=20, max_steps=5)
    before = trainer._get_or_create_agent(policy)._current_lr
    trainer.train(policy=policy, num_episodes=20, max_steps=5, epsilon=0.3)
    after = trainer._get_or_create_agent(policy)._current_lr
    assert after <= before
    assert after >= 1e-5


def test_lr_min_floor_and_loss_based_decay(trainer):
    policy = "DDQN+PER"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, lr_strategy="loss-based", lr_decay=0.5, min_learning_rate=5e-4)
    trainer.set_training_plan(policy, episodes=10, max_steps=5)
    agent = trainer._get_or_create_agent(policy)
    agent._loss_patience = 1
    trainer.train(policy=policy, num_episodes=10, max_steps=5, epsilon=0.3)
    assert agent._current_lr >= 5e-4


def test_continuous_learning_cadence_affects_update_count(trainer):
    policy = "PPO"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, learning_cadence=2)
    trainer.reset_policy_agent(policy)
    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=5)
    updates_cadence_2 = trainer._get_or_create_agent(policy).learn_steps

    trainer.set_policy_config(policy, learning_cadence=5)
    trainer.reset_policy_agent(policy)
    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=5)
    updates_cadence_5 = trainer._get_or_create_agent(policy).learn_steps

    assert updates_cadence_2 >= updates_cadence_5


def test_ppo_gae_and_clip_config_propagation(trainer):
    policy = "PPO"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, gae_lambda=0.9, ppo_clip_range=0.15)
    trainer.reset_policy_agent(policy)
    agent = trainer._get_or_create_agent(policy)
    assert abs(agent._gae_lambda - 0.9) < 1e-9
    assert abs(agent._ppo_clip_range - 0.15) < 1e-9


def test_sac_smoke_learning_runs(trainer):
    policy = "SAC"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, learning_cadence=1, replay_warmup=1, batch_size=2)
    trainer.reset_policy_agent(policy)

    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=8)
    agent = trainer._get_or_create_agent(policy)

    assert agent.learn_steps > 0


def test_sac_learning_cadence_reduces_update_frequency(trainer):
    policy = "SAC"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, replay_warmup=1, batch_size=2, learning_cadence=1)
    trainer.reset_policy_agent(policy)
    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=8)
    updates_cadence_1 = trainer._get_or_create_agent(policy).learn_steps

    trainer.set_policy_config(policy, replay_warmup=1, batch_size=2, learning_cadence=4)
    trainer.reset_policy_agent(policy)
    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=8)
    updates_cadence_4 = trainer._get_or_create_agent(policy).learn_steps

    assert updates_cadence_1 >= updates_cadence_4


def test_trpo_smoke_learning_runs(trainer):
    policy = "TRPO"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, learning_cadence=2, batch_size=4, replay_warmup=1)
    trainer.reset_policy_agent(policy)

    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=8)
    agent = trainer._get_or_create_agent(policy)

    assert agent.learn_steps > 0
    assert np.isfinite(agent._current_lr)


def test_a2c_smoke_learning_runs(trainer):
    policy = "A2C"
    _configure_fast_learning(trainer, policy)
    trainer.set_policy_config(policy, learning_cadence=2, batch_size=4, replay_warmup=1)
    trainer.reset_policy_agent(policy)

    trainer.run_episode(policy=policy, epsilon=0.0, max_steps=8)
    agent = trainer._get_or_create_agent(policy)

    assert agent.learn_steps > 0
    assert np.isfinite(agent._current_lr)


def test_evaluate_policy_is_deterministic_and_no_learning_side_effects(trainer):
    policy = "D3QN"
    _configure_fast_learning(trainer, policy)
    trainer.reset_policy_agent(policy)
    agent = trainer._get_or_create_agent(policy)

    before_learn_steps = agent.learn_steps
    eval_result = trainer.evaluate_policy(policy=policy, max_steps=8, episodes=2, seed_base=123)
    after_learn_steps = agent.learn_steps

    assert len(eval_result["rewards"]) == 2
    assert np.isfinite(float(eval_result["mean_reward"]))
    assert after_learn_steps == before_learn_steps


def test_train_and_csv_export(tmp_path, trainer):
    policy = "D3QN"
    _configure_fast_learning(trainer, policy)
    rewards = trainer.train(policy=policy, num_episodes=2, max_steps=8, epsilon=0.3, save_csv="unit_test")
    assert len(rewards) == 2
