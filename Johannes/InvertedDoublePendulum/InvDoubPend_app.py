import os

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

if os.name == "nt":
    if os.environ.get("MUJOCO_GL", "").strip().lower() == "angle":
        os.environ.pop("MUJOCO_GL", None)
else:
    os.environ.setdefault("MUJOCO_GL", "egl")

from InvDoubPend_gui import launch_gui


if __name__ == "__main__":
    launch_gui()
