"""
bipedal_walker_app.py
Entry point for the BipedalWalker RL Workbench.
"""

import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(__file__))

from bipedal_walker_ui import WorkbenchUI


def main() -> None:
    app = WorkbenchUI()
    app.run()


if __name__ == "__main__":
    main()
