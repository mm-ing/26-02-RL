import tkinter as tk

from LunarLander_gui import LunarLanderGUI


def main() -> None:
    root = tk.Tk()
    LunarLanderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
