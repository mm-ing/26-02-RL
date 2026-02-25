from __future__ import annotations

from tkinter import Tk

from frozenlake_gui import FrozenLakeGUI


def main() -> None:
    root = Tk()
    app = FrozenLakeGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
