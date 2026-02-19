from __future__ import annotations

import argparse
import tkinter as tk

from bandit_gui import GUI
from bandit_logic import Agent, Environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-Armed Bandit Reinforcement Demo")
    parser.add_argument("--runs", type=int, default=100, help="Anzahl Standard-Loops für 'Agent run n loops'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    environment = Environment(start_amounts=(20, 40, 80))
    agent = Agent(
        environment=environment,
        loops=args.runs,
        memory=0,
        epsilon_start=0.9,
        epsilon_decay=0.01,
    )

    root = tk.Tk()
    GUI(root=root, environment=environment, agent=agent)
    root.mainloop()


if __name__ == "__main__":
    main()
