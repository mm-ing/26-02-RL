"""Manual Multi-Armed Bandit (MAB) Tkinter demo.

Run:
    python3 manual_mab.py

This script creates a simple Tkinter UI with 3 bandit buttons. Each bandit
has a fixed success probability. Clicking a button performs a weighted
random draw, updates Trials and Rewards counters, and shows a brief
success/failure status.

No external libraries required — only Python standard library and Tkinter.
"""

import tkinter as tk
import random


def make_bandit_ui(root, bandit_index, prob, row):
    """Create UI elements for a single bandit and return its control dict.

    Args:
        root: parent Tk widget
        bandit_index: int index (0-based)
        prob: float probability of success (0.0 - 1.0)
        row: grid row to place the bandit UI

    Returns:
        dict with keys: 'button', 'trials_var', 'rewards_var'
    """

    frame = tk.Frame(root, padx=8, pady=6)
    frame.grid(row=row, column=0, sticky="w")

    title = tk.Label(frame, text=f"Bandit {bandit_index + 1}", font=("Arial", 12, "bold"))
    title.grid(row=0, column=0, columnspan=2, sticky="w")

    btn = tk.Button(frame, text=f"Play Bandit {bandit_index + 1}", width=18)
    btn.grid(row=1, column=0, rowspan=2, padx=(0, 12))

    trials_var = tk.StringVar(value="Trials: 0")
    rewards_var = tk.StringVar(value="Rewards: 0")

    trials_label = tk.Label(frame, textvariable=trials_var, font=("Arial", 11))
    trials_label.grid(row=1, column=1, sticky="w")

    rewards_label = tk.Label(frame, textvariable=rewards_var, font=("Arial", 11))
    rewards_label.grid(row=2, column=1, sticky="w")

    prob_label = tk.Label(frame, text=f"P(success) = {prob:.2f}", font=("Arial", 10), fg="#555555")
    prob_label.grid(row=0, column=2, padx=(12,0))

    return {
        "frame": frame,
        "button": btn,
        "trials_var": trials_var,
        "rewards_var": rewards_var,
    }


def main():
    # Bandit probabilities (must be distinct)
    probabilities = [0.2, 0.4, 0.8]

    # Internal counters
    trials = [0 for _ in probabilities]
    rewards = [0 for _ in probabilities]

    root = tk.Tk()
    root.title("Manual Multi-Armed Bandit — Tkinter Demo")
    root.geometry("520x260")
    root.resizable(False, False)

    header = tk.Label(root, text="Manual Multi-Armed Bandit (click buttons to play)", font=("Arial", 14, "bold"))
    header.grid(row=0, column=0, pady=(10, 6), sticky="w", padx=10)

    bandit_controls = []

    status_var = tk.StringVar(value="Ready — press a bandit button to play.")
    status_label = tk.Label(root, textvariable=status_var, font=("Arial", 11), fg="#003366")
    status_label.grid(row=5, column=0, sticky="w", padx=10, pady=(12,0))

    def play(b_idx: int):
        """Handle a play on bandit index b_idx."""

        p = probabilities[b_idx]
        trials[b_idx] += 1

        # Weighted draw: success if random() < p
        if random.random() < p:
            rewards[b_idx] += 1
            status_var.set(f"Bandit {b_idx+1}: SUCCESS! (p={p:.2f})")
            # briefly change status color for success
            status_label.config(fg="#006400")
        else:
            status_var.set(f"Bandit {b_idx+1}: failure. (p={p:.2f})")
            status_label.config(fg="#8B0000")

        # Update UI counters immediately
        bandit_controls[b_idx]["trials_var"].set(f"Trials: {trials[b_idx]}")
        bandit_controls[b_idx]["rewards_var"].set(f"Rewards: {rewards[b_idx]}")

        # Return focus so keyboard remains available
        root.focus()

    # Build UI rows for each bandit
    for i, p in enumerate(probabilities, start=1):
        ctrl = make_bandit_ui(root, i-1, p, row=i)
        bandit_controls.append(ctrl)

    # Wire up buttons after creation to close over correct index
    for idx, ctrl in enumerate(bandit_controls):
        ctrl["button"].config(command=lambda i=idx: play(i))

    # Footer with reset and quit buttons
    footer = tk.Frame(root, pady=8)
    footer.grid(row=6, column=0, sticky="w", padx=10)

    def reset_counts():
        for j in range(len(trials)):
            trials[j] = 0
            rewards[j] = 0
            bandit_controls[j]["trials_var"].set("Trials: 0")
            bandit_controls[j]["rewards_var"].set("Rewards: 0")
        status_var.set("Counters reset. Ready.")
        status_label.config(fg="#003366")

    reset_btn = tk.Button(footer, text="Reset Counters", command=reset_counts)
    reset_btn.grid(row=0, column=0, padx=(0,8))

    quit_btn = tk.Button(footer, text="Quit", command=root.destroy)
    quit_btn.grid(row=0, column=1)

    # Keep UI responsive — Tkinter runs on the main thread here.
    root.mainloop()


if __name__ == "__main__":
    main()
