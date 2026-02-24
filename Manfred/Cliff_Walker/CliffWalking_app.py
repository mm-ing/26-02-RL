"""
CliffWalking RL Workbench – Application entry point
"""

import tkinter as tk
from CliffWalking_ui import WorkbenchUI


def main():
    root = tk.Tk()
    app = WorkbenchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
