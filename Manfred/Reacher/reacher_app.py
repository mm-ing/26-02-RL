"""Entry point for the Reacher RL Workbench."""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from reacher_ui import WorkbenchApp


def main() -> None:
    app = WorkbenchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
