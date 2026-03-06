import os
import platform


os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if platform.system().lower() == "windows":
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        os.environ.pop("MUJOCO_GL", None)
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import tkinter as tk

from CarRacing_gui import CarRacingGUI


def main() -> None:
    root = tk.Tk()
    root.title("CarRacing (SB3)")
    root.geometry("1400x900")
    CarRacingGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
