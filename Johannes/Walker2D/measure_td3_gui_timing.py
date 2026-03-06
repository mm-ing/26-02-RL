import time
import tkinter as tk
from pathlib import Path

from Walker2D_gui import Walker2DGUI


def main() -> None:
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
        timings.append(
            (
                episode,
                delta,
                float(payload.get("reward", 0.0)),
                float(payload.get("lr", 0.0)),
            )
        )
        return orig_handle_episode(payload)

    app._handle_episode_event = wrapped_handle_episode

    start = time.perf_counter()
    app._start_training()

    while app.training_active:
        root.update()
        time.sleep(0.01)

    total = time.perf_counter() - start
    root.update_idletasks()
    root.destroy()

    lines = []
    lines.append(f"TOTAL_SECONDS,{total:.4f}")
    lines.append("EPISODE,DELTA_SECONDS,REWARD,LR")
    for episode, delta, reward, lr in timings:
        lines.append(f"{episode},{delta:.6f},{reward:.6f},{lr:.8f}")

    ep9 = next((delta for episode, delta, _, _ in timings if episode == 9), None)
    ep10 = next((delta for episode, delta, _, _ in timings if episode == 10), None)
    lines.append(f"DELTA_EP9,{ep9 if ep9 is not None else ''}")
    lines.append(f"DELTA_EP10,{ep10 if ep10 is not None else ''}")
    if ep9 is not None and ep10 is not None and ep9 > 0:
        lines.append(f"RATIO_EP10_OVER_EP9,{ep10 / ep9:.6f}")

    out_path = Path("td3_gui_timing_results.csv")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"WROTE,{out_path}")


if __name__ == "__main__":
    main()
