# Ein einfaches GUI mit 3 Banditen, die jeweils eine eigene Wahrscheinlichkeit p haben.

# Bandit_TJS.py

import tkinter as tk
import random

class BanditFrame(tk.Frame):
    def __init__(self, master, index, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.index = index
        self.p = random.random()  # Wahrscheinlichkeit ∈ [0,1]
        self.clicks = 0
        self.last_value = None

        self.button = tk.Button(self, text=f"Bandit {index}", width=12, command=self.on_click)
        self.button.grid(row=0, column=0, columnspan=2, pady=(0,6))

        self.p_label = tk.Label(self, text=f"p = {self.p:.2f}", font=("TkDefaultFont", 8))
        self.p_label.grid(row=1, column=0, columnspan=2)

        self.clicks_label = tk.Label(self, text=f"Anzahl Klicks: {self.clicks}")
        self.clicks_label.grid(row=2, column=0, sticky="w", padx=(0,6))

        self.value_label = tk.Label(self, text=f"Ausgabewert: -")
        self.value_label.grid(row=2, column=1, sticky="e")

    def on_click(self):
        self.clicks += 1
        if random.random() < self.p:
            self.last_value = random.randint(50, 100)
        else:
            self.last_value = random.randint(0, 49)
        self.update_labels()

    def update_labels(self):
        self.clicks_label.config(text=f"Anzahl Klicks: {self.clicks}")
        self.value_label.config(text=f"Ausgabewert: {self.last_value}")

def main():
    root = tk.Tk()
    root.title("3-Banditen GUI")
    root.resizable(True, True)
    container = tk.Frame(root, padx=10, pady=10)
    container.pack(fill="both", expand=True)

    bandits = []
    for i in range(1, 4):
        bf = BanditFrame(container, i, bd=1, relief="groove", padx=8, pady=6)
        bf.grid(row=0, column=i-1, padx=6, sticky="n")
        bandits.append(bf)

    # make columns expand evenly
    for col in range(3):
        container.grid_columnconfigure(col, weight=1)

    root.mainloop()

if __name__ == "__main__":
    main()