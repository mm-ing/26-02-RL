import pytest

from bandit_logic import Agent, Environment, ThompsonSampling


class FixedRng:
    def __init__(self, random_value: float, randint_value: int):
        self._random_value = random_value
        self._randint_value = randint_value

    def random(self):
        return self._random_value

    def randint(self, _a, _b):
        return self._randint_value


def test_environment_step_returns_int_reward_and_updates_bandit_stats():
    env = Environment(start_amounts=(20, 40, 80), seed=123)

    reward = env.step(0)

    assert isinstance(reward, int)
    state = env.get_state()[0]
    assert state["pulls"] == 1
    assert state["cumulative_reward"] >= 0


def test_payout_probability_uses_stored_div_100_capped_to_one():
    env = Environment(start_amounts=(20, 40, 80), seed=1)
    bandit = env.bandits[0]
    bandit.stored_coins = 99
    bandit.rng = FixedRng(random_value=0.2, randint_value=10)

    reward = bandit.pull()

    assert reward == 1
    assert bandit.pulls == 1


def test_constant_probability_mode_uses_initialization_based_probability():
    env = Environment(start_amounts=(20, 40, 80), seed=2, probability_mode="constant")
    bandit = env.bandits[0]
    bandit.rng = FixedRng(random_value=0.25, randint_value=5)

    reward = bandit.pull()

    assert reward == 0
    assert bandit.success == 0


def test_reward_is_binary_when_payout_occurs():
    env = Environment(start_amounts=(20, 40, 80), seed=3, probability_mode="variable")
    bandit = env.bandits[1]
    bandit.rng = FixedRng(random_value=0.0, randint_value=50)

    reward = bandit.pull()

    assert reward in (0, 1)
    assert reward == 1


def test_agent_epsilon_greedy_can_choose_best_action_when_epsilon_zero():
    env = Environment(start_amounts=(20, 40, 80), seed=99)
    agent = Agent(environment=env, epsilon_start=0.0, epsilon_decay=0.0)
    agent.set_policy("Epsilon-Greedy")
    agent.q_values = [1.0, 3.0, 2.0]

    action, _reward = agent.step()

    assert action == 1


def test_thompson_sampling_updates_posterior_parameters():
    policy = ThompsonSampling()
    policy.reset(3)

    policy.update(action=0, reward=5)
    policy.update(action=1, reward=0)

    assert policy.alpha[0] == pytest.approx(2.0)
    assert policy.beta[0] == pytest.approx(1.0)
    assert policy.alpha[1] == pytest.approx(1.0)
    assert policy.beta[1] == pytest.approx(2.0)
