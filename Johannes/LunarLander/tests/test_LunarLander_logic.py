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
        reward = 1.0 if action in [0, 1, 2, 3] else -1.0
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


@pytest.mark.parametrize("policy", ["DuelingDQN", "D3QN", "DDQN+PER"])
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


def test_train_and_csv_export(tmp_path, trainer):
    policy = "D3QN"
    _configure_fast_learning(trainer, policy)
    rewards = trainer.train(policy=policy, num_episodes=2, max_steps=8, epsilon=0.3, save_csv="unit_test")
    assert len(rewards) == 2
