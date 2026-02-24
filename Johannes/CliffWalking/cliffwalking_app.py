from tkinter import Tk

from cliffwalking_gui import CliffWalkingGUI


def main() -> None:
    root = Tk()
    app = CliffWalkingGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app.shutdown(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()
