import time
import tkinter as tk
from pathlib import Path

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import Walker2D_logic as logic
from Walker2D_gui import Walker2DGUI


class DummyActionSpace:
    def sample(self):
        return np.zeros((6,), dtype=np.float32)


class DummyEnv(gym.Env):
    metadata = {"render_modes": [None, "rgb_array"]}

    def __init__(self, render_mode=None, **kwargs):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(24,), dtype=np.float32)
        self._step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.zeros((24,), dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        done = self._step >= 1000
        obs = np.full((24,), self._step / 1000.0, dtype=np.float32)
        reward = float(1.0 - 0.001 * np.square(action).sum())
        return obs, reward, done, False, {}

    def render(self):
        return np.zeros((64, 64, 3), dtype=np.uint8)

    def close(self):
        return None


def main() -> None:
    orig_make = logic.gym.make
    logic.gym.make = lambda *args, **kwargs: DummyEnv(**kwargs)

    root = tk.Tk()
    root.withdraw()
    app = Walker2DGUI(root)

    app.var_policy.set("TD3")
    app._on_policy_changed(None)
    app.var_episodes.set(12)
    app.var_compare_on.set(False)
    app.var_animation_on.set(False)
    app._on_animation_toggle()

    timings = []
    last_t = None
    orig_handle_episode = app._handle_episode_event

    def wrapped_handle_episode(payload):
        nonlocal last_t
        now = time.perf_counter()
        episode = int(payload.get("episode", -1))
        delta = 0.0 if last_t is None else (now - last_t)
        last_t = now
        timings.append((episode, delta))
        return orig_handle_episode(payload)

    app._handle_episode_event = wrapped_handle_episode

    start = time.perf_counter()
    app._start_training()

    while app.training_active:
        root.update()
        time.sleep(0.002)

    total = time.perf_counter() - start
    root.update_idletasks()
    root.destroy()

    logic.gym.make = orig_make

    out_path = Path("td3_gui_timing_dummy.csv")
    lines = ["episode,delta_seconds"]
    lines.extend([f"{episode},{delta:.6f}" for episode, delta in timings])

    ep9 = next((delta for episode, delta in timings if episode == 9), None)
    ep10 = next((delta for episode, delta in timings if episode == 10), None)
    ep11 = next((delta for episode, delta in timings if episode == 11), None)

    lines.append(f"TOTAL_SECONDS,{total:.6f}")
    lines.append(f"DELTA_EP9,{'' if ep9 is None else ep9:.6f}" if ep9 is not None else "DELTA_EP9,")
    lines.append(f"DELTA_EP10,{'' if ep10 is None else ep10:.6f}" if ep10 is not None else "DELTA_EP10,")
    lines.append(f"DELTA_EP11,{'' if ep11 is None else ep11:.6f}" if ep11 is not None else "DELTA_EP11,")
    if ep9 is not None and ep10 is not None and ep9 > 0:
        lines.append(f"RATIO_EP10_OVER_EP9,{ep10 / ep9:.6f}")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE,{out_path}")


if __name__ == "__main__":
    main()
