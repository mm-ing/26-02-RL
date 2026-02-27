import tkinter as tk

from lunar_lander_ui import WorkbenchUI


def main() -> None:
    root = tk.Tk()
    WorkbenchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
