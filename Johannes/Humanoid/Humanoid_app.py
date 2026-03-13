from __future__ import annotations

import os
import platform

# Startup guards must be applied before importing GUI/ML modules.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if platform.system().lower().startswith("win"):
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        os.environ.pop("MUJOCO_GL", None)
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

import tkinter as tk

from Humanoid_gui import HumanoidGUI


def main() -> None:
    root = tk.Tk()
    HumanoidGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
