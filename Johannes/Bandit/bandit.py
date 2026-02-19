# ...existing code...
import tkinter as tk
import random

class BanditMachine:
    def __init__(self, start_coins: int, rng: random.Random = None):
        self.stored = int(start_coins)
        self.attempts = 0
        self.total_returned = 0
        self.rng = rng or random.Random()

    def play(self) -> int:
        """Ein Coin wird eingeworfen, Maschine zahlt zufällig aus.
        Erwartete Auszahlung steigt mit self.stored.
        Gibt die Anzahl zurückgegebener Coins (int)."""
        self.attempts += 1
        # Einwurf kostet 1 Coin, landet im Automaten
        self.stored += 1
        # Wahrscheinlichkeit, dass etwas ausgezahlt wird: stored/100 (max 1.0)
        p = min(float(self.stored) / 100.0, 1.0)
        payout = 0
        if self.rng.random() < p:
            # Wenn Auszahlung erfolgt, wähle eine zufällige Auszahlung zwischen 1 und stored
            # (nicht größer als aktuell im Automaten verfügbare Coins)
            payout = self.rng.randint(1, self.stored)
        # Auszahlung abziehen und Statistik aktualisieren
        self.stored -= payout
        self.total_returned += payout
        return payout

def create_gui(start_amounts=(20, 40, 80)):
    root = tk.Tk()
    root.title("Einarmige Banditen")
    top_label = tk.Label(root, text="Glücksspiel kann süchtig machen!", fg="red")
    top_label.grid(row=0, column=0, columnspan=3, pady=(10, 5))

    machines = [BanditMachine(s) for s in start_amounts]
    attempt_labels = []
    return_labels = []
    last_payout_labels = []

    def make_callback(i):
        def on_click():
            payout = machines[i].play()
            attempt_labels[i]["text"] = f"Versuche: {machines[i].attempts}"
            return_labels[i]["text"] = f"Coins zurückgegeben: {machines[i].total_returned}"
            last_payout_labels[i]["text"] = f"Letzte Auszahlung: {payout}"
        return on_click

    for i in range(3):
        frame = tk.Frame(root, padx=10, pady=10, bd=1, relief=tk.RIDGE)
        frame.grid(row=1, column=i, padx=5, pady=5, sticky="n")
        btn = tk.Button(frame, text=f"Automat {i+1} (1 Coin)", width=18, command=make_callback(i))
        btn.pack(pady=(0,5))
        attempt = tk.Label(frame, text=f"Versuche: {machines[i].attempts}")
        attempt.pack()
        ret = tk.Label(frame, text=f"Coins zurückgegeben: {machines[i].total_returned}")
        ret.pack()
        last = tk.Label(frame, text="Letzte Auszahlung: 0")
        last.pack()
        attempt_labels.append(attempt)
        return_labels.append(ret)
        last_payout_labels.append(last)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
# ...existing code...