import tkinter as tk
import random

class BanditFrame(tk.Frame):
    def __init__(self, master, index, start_coins, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.index = index
        self.start_coins = start_coins
        self.clicks = 0
        self.last = 0
        self.total = 0
        self.player_coins = 100

        self.btn = tk.Button(self, text=f"Bandit {index}", width=16, command=self.on_click)
        self.btn.grid(row=0, column=0, columnspan=2, pady=(0, 6))

        self.p_label = tk.Label(self, text=f"p = {self.start_coins / 100:.2f}", font=("TkDefaultFont", 9))
        self.p_label.grid(row=1, column=0, columnspan=2)

        self.clicks_label = tk.Label(self, text=f"Anzahl Klicks: {self.clicks}")
        self.clicks_label.grid(row=2, column=0, sticky="w", padx=(4, 8))

        self.value_label = tk.Label(self, text=f"Ausgabewert: {self.last} (Summe: {self.total})")
        self.value_label.grid(row=2, column=1, sticky="e", padx=(8, 4))

    def on_click(self):
        if self.player_coins > 0:
            self.player_coins -= 1
            self.clicks += 1
            payout = self.pull()
            self.last = payout
            self.total += payout
            self.update_labels()

    def pull(self):
        p = self.start_coins / 100
        if random.random() < p:
            return random.randint(1, self.start_coins)
        return 0

    def update_labels(self):
        self.clicks_label.config(text=f"Anzahl Klicks: {self.clicks}")
        self.value_label.config(text=f"Ausgabewert: {self.last} (Summe: {self.total})")

def main():
    root = tk.Tk()
    root.title("3-Banditen")
    root.minsize(640, 240)

    container = tk.Frame(root, padx=12, pady=12)
    container.pack(fill="both", expand=True)

    header = tk.Label(container, text="Viel Erfolg beim Zocken!", font=("TkDefaultFont", 16, "bold"))
    header.grid(row=0, column=0, columnspan=3, pady=(0, 12))

    start_coins = [20, 40, 80]
    bandit_frames = []
    for i, coins in enumerate(start_coins, start=1):
        bf = BanditFrame(container, i, coins, bd=1, relief="groove", padx=8, pady=6)
        bf.grid(row=1, column=i-1, padx=8, sticky="nsew")
        bandit_frames.append(bf)

    for c in range(3):
        container.grid_columnconfigure(c, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()