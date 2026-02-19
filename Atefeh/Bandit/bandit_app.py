"""Application entry point for bandit GUI."""

import tkinter as tk

from bandit_logic import Environment, Agent, EpsilonGreedyPolicy
from bandit_gui import BanditGUI


def build_agent(probs) -> Agent:
    """Create a fresh default agent; GUI can switch policy from controls."""
    env = Environment(probs.copy())
    policy = EpsilonGreedyPolicy(epsilon=0.9, decay=0.001)
    return Agent(env=env, policy=policy, memory_size=100)


def main():
    probs = [0.2, 0.4, 0.8]
    agent_factory = lambda: build_agent(probs)

    root = tk.Tk()
    root.title("Bandit Comparison")
    BanditGUI(root, agent_factory)  # exactly one GUI instance
    root.mainloop()


if __name__ == "__main__":
    main()