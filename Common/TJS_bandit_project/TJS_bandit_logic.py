class Bandit:
    def __init__(self, start_coins):
        self.start_coins = start_coins
        self.pulls = 0
        self.successes = 0
        self.total_reward = 0

    @property
    def p(self):
        return max(min(self.start_coins / 100, 1), 0)

    def pull(self):
        self.pulls += 1
        if random.random() < self.p:
            payout = random.randint(1, self.start_coins)
            self.successes += 1
        else:
            payout = 0
        self.total_reward += payout
        return payout


class Environment:
    def __init__(self, starts=(20, 40, 80)):
        self.bandits = [Bandit(start) for start in starts]

    def step(self, action):
        return self.bandits[action].pull()


class Policy:
    def select_action(self, agent):
        raise NotImplementedError("This method should be overridden by subclasses.")


class EpsilonGreedyPolicy(Policy):
    def __init__(self, epsilon=0.9, decay=0.01):
        self.epsilon = epsilon
        self.decay = decay

    def select_action(self, agent):
        if random.random() < self.epsilon:
            return random.randint(0, len(agent.environment.bandits) - 1)
        else:
            return max(range(len(agent.environment.bandits)), key=lambda x: agent.estimates[x])


class ThompsonSamplingPolicy(Policy):
    def select_action(self, agent):
        beta_samples = [random.betavariate(agent.successes[i] + 1, agent.pulls[i] - agent.successes[i] + 1) for i in range(len(agent.environment.bandits))]
        return beta_samples.index(max(beta_samples))


class Agent:
    def __init__(self, environment, policy):
        self.environment = environment
        self.policy = policy
        self.estimates = [0] * len(environment.bandits)
        self.pulls = [0] * len(environment.bandits)
        self.successes = [0] * len(environment.bandits)

    def step(self):
        action = self.policy.select_action(self)
        reward = self.environment.step(action)
        self.pulls[action] += 1
        if reward > 0:
            self.successes[action] += 1
        self.estimates[action] = self.successes[action] / self.pulls[action] if self.pulls[action] > 0 else 0


def run_iterations(env, agent, n=100):
    for _ in range(n):
        agent.step()