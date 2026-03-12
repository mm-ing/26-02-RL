from __future__ import annotations

import os
import platform
import tkinter as tk

# Startup guards should be applied before importing GUI/ML stacks.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if platform.system().lower().startswith("win"):
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        os.environ.pop("MUJOCO_GL", None)
    # Prefer glfw backend on Windows for stable rgb_array rendering.
    os.environ.setdefault("MUJOCO_GL", "glfw")
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from Ant_gui import AntGUI


def main() -> None:
    root = tk.Tk()
    AntGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
