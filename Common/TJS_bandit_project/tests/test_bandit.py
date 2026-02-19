import pytest
from TJS_bandit_logic import Bandit, Environment, EpsilonGreedyPolicy, ThompsonSamplingPolicy, Agent

def test_p_calculation():
    bandit = Bandit(start_coins=20)
    assert bandit.p == 0.2  # Probability should be 20% for 20 coins

def test_pull_deterministic():
    bandit = Bandit(start_coins=40)
    bandit.pull()  # Perform a pull
    assert isinstance(bandit.total_reward, int)  # Ensure payout is an integer
    assert bandit.pulls == 1  # Should have one pull now

def test_agent_step_updates_estimates():
    env = Environment(starts=(20, 40, 80))
    policy = EpsilonGreedyPolicy(epsilon=0.9)
    agent = Agent(env, policy)
    initial_estimates = agent.estimates.copy()
    agent.step()  # Perform one step
    assert agent.estimates != initial_estimates  # Estimates should be updated after step

def test_thompson_sampling_policy():
    env = Environment(starts=(20, 40, 80))
    policy = ThompsonSamplingPolicy()
    agent = Agent(env, policy)
    initial_action = agent.step()  # Get initial action
    agent.step()  # Perform another step
    assert agent.last_action != initial_action  # Action should change over time

def test_bandit_success_rate():
    bandit = Bandit(start_coins=80)
    for _ in range(100):
        bandit.pull()  # Simulate multiple pulls
    assert bandit.success_rate >= 0  # Success rate should be non-negative
    assert bandit.success_rate <= 1  # Success rate should not exceed 1