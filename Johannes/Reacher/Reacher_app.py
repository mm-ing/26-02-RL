import os
import sys

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if sys.platform.startswith("win"):
    if os.environ.get("MUJOCO_GL", "").lower() == "angle":
        del os.environ["MUJOCO_GL"]
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from Reacher_gui import build_gui_root


def main() -> None:
    root = build_gui_root()
    root.mainloop()


if __name__ == "__main__":
    main()
