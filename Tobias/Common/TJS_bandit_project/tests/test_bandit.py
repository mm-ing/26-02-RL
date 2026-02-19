import pytest
from TJS_bandit_logic import Bandit, Environment, EpsilonGreedyPolicy, ThompsonSamplingPolicy, Agent

def test_p_calculation():
    bandit = Bandit(start_coins=20)
    expected_p = 0.2  # 20 coins / 100
    assert bandit.p == expected_p

def test_pull_deterministic():
    bandit = Bandit(start_coins=20)
    bandit.pull()  # Call the pull method
    assert isinstance(bandit.total_reward, int)  # Ensure payout is an integer
    assert bandit.pulls == 1  # Ensure pulls count is updated

def test_agent_step_updates_estimates():
    env = Environment(starts=(20, 40, 80))
    policy = EpsilonGreedyPolicy(epsilon=0.9)
    agent = Agent(env, policy)
    
    initial_estimates = [bandit.estimated_mean for bandit in env.bandits]
    agent.step()  # Perform a step
    updated_estimates = [bandit.estimated_mean for bandit in env.bandits]
    
    assert updated_estimates != initial_estimates  # Ensure estimates have been updated