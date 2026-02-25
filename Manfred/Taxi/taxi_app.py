"""Taxi RL Workbench – App entry."""

import tkinter as tk

from taxi_ui import WorkbenchUI


def main():
    root = tk.Tk()
    WorkbenchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
