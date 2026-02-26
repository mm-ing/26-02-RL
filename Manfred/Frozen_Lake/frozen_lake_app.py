"""Frozen Lake RL Workbench – App entry."""

import tkinter as tk

from frozen_lake_ui import WorkbenchUI


def main():
    root = tk.Tk()
    WorkbenchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
