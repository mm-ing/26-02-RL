import tkinter as tk

from cartpole_ui import WorkbenchUI


def main() -> None:
    root = tk.Tk()
    WorkbenchUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
