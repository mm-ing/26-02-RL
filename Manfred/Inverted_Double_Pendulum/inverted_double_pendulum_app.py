"""
Entry point for the InvertedDoublePendulum RL Workbench.
"""
from inverted_double_pendulum_ui import WorkbenchApp


def main() -> None:
    app = WorkbenchApp()
    app.mainloop()


if __name__ == "__main__":
    main()
