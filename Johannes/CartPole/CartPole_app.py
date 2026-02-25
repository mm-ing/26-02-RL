from tkinter import Tk

from CartPole_gui import CartPoleGUI


def main() -> None:
    root = Tk()
    CartPoleGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
