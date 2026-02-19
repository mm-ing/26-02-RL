import random
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Tuple, List, Set, Dict

State = Tuple[int, int]


class Grid:
    ACTIONS = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}  # up, down, left, right

    def __init__(self, M: int = 5, N: int = 3, blocked: List[State] = None,
                 start: State = (0, 2), target: State = None):
        self.M = M
        self.N = N
        # default blocked fields per spec if caller did not provide any
        if blocked is None:
            default_blocked = [(2, 2), (2, 1)]
            # only keep defaults that fit inside grid
            self.blocked: Set[State] = set((x, y) for (x, y) in default_blocked if 0 <= x < M and 0 <= y < N)
        else:
            self.blocked: Set[State] = set(blocked)
        self.start = start
        self.target = target if target is not None else (M - 1, N - 1)

    def in_bounds(self, s: State) -> bool:
        x, y = s
        return 0 <= x < self.M and 0 <= y < self.N

    def is_blocked(self, s: State) -> bool:
        return s in self.blocked

    def valid(self, s: State) -> bool:
        return self.in_bounds(s) and not self.is_blocked(s)

    def neighbors(self, s: State) -> List[State]:
        nbrs = []
        for dx, dy in self.ACTIONS.values():
            ns = (s[0] + dx, s[1] + dy)
            if self.valid(ns):
                nbrs.append(ns)
        return nbrs

    def is_reachable(self, start: State = None, target: State = None) -> bool:
        """Return True if target is reachable from start given current blocked cells."""
        s0 = start if start is not None else self.start
        t0 = target if target is not None else self.target
        if not self.valid(s0) or not self.valid(t0):
            return False
        # BFS
        from collections import deque
        q = deque([s0])
        seen = {s0}
        while q:
            s = q.popleft()
            if s == t0:
                return True
            for dx, dy in self.ACTIONS.values():
                ns = (s[0] + dx, s[1] + dy)
                if ns not in seen and self.valid(ns):
                    seen.add(ns)
                    q.append(ns)
        return False

    def step(self, s: State, action: int, noise: float = 0.0) -> Tuple[State, int, bool]:
        # Deterministic intended move
        dx, dy = self.ACTIONS.get(action, (0, 0))
        intended = (s[0] + dx, s[1] + dy)

        # Apply stochastic slip with probability noise
        candidates = [intended] if self.valid(intended) else [s]
        if noise > 0:
            nbrs = self.neighbors(s)
            if nbrs:
                # include staying in place as possible slip
                candidates = nbrs
                # choose randomly from neighbors with slip probability
                if random.random() < noise:
                    ns = random.choice(candidates)
                else:
                    ns = intended if self.valid(intended) else s
            else:
                ns = s
        else:
            ns = intended if self.valid(intended) else s

        done = ns == self.target
        reward = 0 if done else -1
        return ns, reward, done


class QLearningAgent:
    def __init__(self, grid: Grid, alpha: float = 0.5, gamma: float = 0.99):
        self.grid = grid
        self.alpha = alpha
        self.gamma = gamma
        self.Q: Dict[Tuple[State, int], float] = defaultdict(float)

    def get_q(self, s: State, a: int) -> float:
        return self.Q[(s, a)]

    def best_action(self, s: State) -> int:
        vals = {a: self.get_q(s, a) for a in self.grid.ACTIONS}
        return max(vals, key=vals.get)

    def update(self, s: State, a: int, r: int, s2: State):
        max_q_next = max(self.get_q(s2, a2) for a2 in self.grid.ACTIONS)
        cur = self.get_q(s, a)
        self.Q[(s, a)] = cur + self.alpha * (r + self.gamma * max_q_next - cur)


class MonteCarloAgent:
    """First-visit Monte Carlo prediction/control using state-value returns (V).

    This agent estimates state values V(s) via first-visit Monte Carlo.
    The trainer uses epsilon-greedy action selection when `best_action` is
    present; `best_action` here picks the action leading to the neighbor
    state with highest estimated V.
    """

    def __init__(self, grid: Grid, gamma: float = 0.99):
        self.grid = grid
        self.gamma = gamma
        self.returns_sum: Dict[State, float] = defaultdict(float)
        self.returns_count: Dict[State, int] = defaultdict(int)
        self.V: Dict[State, float] = defaultdict(float)

    def process_episode(self, episode_states: List[State], rewards: List[int]):
        """Process a completed episode given lists of states and rewards.

        `episode_states` should be a list of length T+1: S0..ST, and
        `rewards` should have length T with rewards r0..r_{T-1} where
        r_t is reward observed after taking action from S_t.
        Performs first-visit updates for states.
        """
        G = 0.0
        visited = set()
        # iterate backwards over time steps
        # rewards[t] corresponds to transition S_t -> S_{t+1}
        for t in range(len(rewards) - 1, -1, -1):
            r = rewards[t]
            s = episode_states[t]
            G = self.gamma * G + r
            if s not in visited:
                visited.add(s)
                self.returns_sum[s] += G
                self.returns_count[s] += 1
                self.V[s] = self.returns_sum[s] / self.returns_count[s]

    def best_action(self, s: State) -> int:
        # choose action that leads to neighbor with highest V
        vals = {}
        for a, (dx, dy) in self.grid.ACTIONS.items():
            ns = (s[0] + dx, s[1] + dy)
            if not self.grid.valid(ns):
                vals[a] = float('-inf')
            else:
                vals[a] = self.V.get(ns, 0.0)
        maxv = max(vals.values())
        bests = [a for a, v in vals.items() if v == maxv]
        return random.choice(bests)

    def get_v(self, s: State) -> float:
        return self.V.get(s, 0.0)


class Trainer:
    def __init__(self, grid: Grid, noise: float = 0.0):
        self.grid = grid
        self.noise = noise

    def run_episode(self, policy, epsilon: float = 0.1, max_steps: int = 1000):
        s = self.grid.start
        episode_states = [s]
        rewards = []
        transitions = []
        total_reward = 0
        for step in range(max_steps):
            if hasattr(policy, 'best_action'):
                # epsilon-greedy for Q-learning
                if random.random() < epsilon:
                    a = random.choice(list(self.grid.ACTIONS.keys()))
                else:
                    a = policy.best_action(s)
            else:
                a = random.choice(list(self.grid.ACTIONS.keys()))

            s2, r, done = self.grid.step(s, a, noise=self.noise)
            transitions.append((s, a, r, s2, done))
            rewards.append(r)
            total_reward += r
            if hasattr(policy, 'update'):
                policy.update(s, a, r, s2)
            s = s2
            episode_states.append(s)
            if done:
                break

        # If Monte Carlo policy (state-value), let it process episode
        # by passing the list of visited states and corresponding rewards.
        if isinstance(policy, MonteCarloAgent):
            policy.process_episode(episode_states, rewards)

        return transitions, total_reward

    def train(self, policy, num_episodes: int = 100, max_steps: int = 100, epsilon: float = 0.1,
              save_csv: str = None):
        all_rewards = []
        import csv, time
        fname = None
        if save_csv:
            ts = int(time.time())
            # ensure results_csv folder exists inside this package folder
            outdir = os.path.join(os.path.dirname(__file__), 'results_csv')
            try:
                os.makedirs(outdir, exist_ok=True)
            except Exception:
                pass
            fname = f"{save_csv}_{ts}.csv"
            fpath = os.path.join(outdir, fname)
            csvfile = open(fpath, 'w', newline='')
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'step', 's_x', 's_y', 'action', 'reward', 'next_s_x', 'next_s_y', 'done'])
        for ep in range(num_episodes):
            transitions, total = self.run_episode(policy, epsilon=epsilon, max_steps=max_steps)
            all_rewards.append(total)
            if save_csv and transitions:
                for step, tr in enumerate(transitions):
                    s, a, r, s2, done = tr
                    writer.writerow([ep, step, s[0], s[1], a, r, s2[0], s2[1], int(done)])
        if save_csv:
            csvfile.close()
        return all_rewards
