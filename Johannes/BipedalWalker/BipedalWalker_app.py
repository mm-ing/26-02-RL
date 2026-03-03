import os
import tkinter as tk

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from BipedalWalker_gui import BipedalWalkerGUI


def main():
    root = tk.Tk()
    root.geometry("1500x900")
    BipedalWalkerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()