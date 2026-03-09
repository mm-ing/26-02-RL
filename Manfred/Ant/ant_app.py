"""Entry point for the Ant RL Workbench."""
import os
import sys

# Ensure the package directory is on the path when launched directly
sys.path.insert(0, os.path.dirname(__file__))

from ant_ui import WorkbenchApp


def main() -> None:
    app = WorkbenchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
