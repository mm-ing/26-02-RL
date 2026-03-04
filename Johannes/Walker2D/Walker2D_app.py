import os
import platform

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if platform.system().lower().startswith("win"):
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        del os.environ["MUJOCO_GL"]
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import tkinter as tk

from Walker2D_gui import Walker2DGUI


def main() -> None:
    root = tk.Tk()
    root.geometry("1400x900")
    Walker2DGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
