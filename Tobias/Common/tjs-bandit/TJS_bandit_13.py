import tkinter as tk
import random
from TJS_bandit_logic import Bandit, Environment, Agent, EpsilonGreedyPolicy, ThompsonSamplingPolicy

class BanditFrame(tk.Frame):
    def __init__(self, master, index, p, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.index = index
        self.p = p
        self.clicks = 0
        self.last = 0
        self.total = 0

        self.btn = tk.Button(self, text=f"Bandit {index}", width=16, command=self.on_click)
        self.btn.grid(row=0, column=0, columnspan=2, pady=(0, 6))

        self.p_label = tk.Label(self, text=f"p = {int(self.p * 100)}%", font=("TkDefaultFont", 9))
        self.p_label.grid(row=1, column=0, columnspan=2)

        self.clicks_label = tk.Label(self, text=f"Anzahl Klicks: {self.clicks}")
        self.clicks_label.grid(row=2, column=0, sticky="w", padx=(4, 8))

        self.value_label = tk.Label(self, text=f"Ausgabewert: {self.last} (Summe: {self.total})")
        self.value_label.grid(row=2, column=1, sticky="e", padx=(8, 4))

    def on_click(self):
        self.clicks += 1
        if random.random() < self.p:
            payout = random.randint(1, 100)  # Payout range can be adjusted
        else:
            payout = 0
        self.last = payout
        self.total += payout
        self.update_labels()

    def update_labels(self):
        self.clicks_label.config(text=f"Anzahl Klicks: {self.clicks}")
        self.value_label.config(text=f"Ausgabewert: {self.last} (Summe: {self.total})")

def main():
    root = tk.Tk()
    root.title("3-Banditen")
    root.minsize(640, 480)

    container = tk.Frame(root, padx=12, pady=12)
    container.pack(fill="both", expand=True)

    header_manual = tk.Label(container, text="Viel MANUELLEN Erfolg Dir!!!", font=("TkDefaultFont", 16, "bold"))
    header_manual.grid(row=0, column=0, columnspan=3, pady=(0, 12))

    probabilities = [0.20, 0.40, 0.80]
    bandit_frames = []
    for i, p in enumerate(probabilities, start=1):
        bf = BanditFrame(container, i, p, bd=1, relief="groove", padx=8, pady=6)
        bf.grid(row=1, column=i-1, padx=8, sticky="nsew")
        bandit_frames.append(bf)

    header_agent = tk.Label(container, text="Dies wird vom Agenten gesteuert!!!", font=("TkDefaultFont", 16, "bold"))
    header_agent.grid(row=2, column=0, columnspan=3, pady=(12, 12))

    # Placeholder for agent buttons and functionality
    agent_frame = tk.Frame(container)
    agent_frame.grid(row=3, column=0, columnspan=3, pady=(0, 6))

    # Example buttons for agent actions
    btn_agent_once = tk.Button(agent_frame, text="Agent EINMAL", command=lambda: print("Agent pulls once"))
    btn_agent_once.grid(row=0, column=0, padx=5)

    btn_agent_all = tk.Button(agent_frame, text="Agent ALLE Iterationen", command=lambda: print("Agent pulls all iterations"))
    btn_agent_all.grid(row=0, column=1, padx=5)

    btn_reset = tk.Button(agent_frame, text="RESET STATE", command=lambda: print("Resetting state"))
    btn_reset.grid(row=0, column=2, padx=5)

    for c in range(3):
        container.grid_columnconfigure(c, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()