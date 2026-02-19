import tkinter as tk
import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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
            payout = random.randint(1, 100)  # Random payout between 1 and 100
        else:
            payout = 0
        self.last = payout
        self.total += payout
        self.update_labels()

    def update_labels(self):
        self.clicks_label.config(text=f"Anzahl Klicks: {self.clicks}")
        self.value_label.config(text=f"Ausgabewert: {self.last} (Summe: {self.total})")

class BanditGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3-Banditen")
        self.minsize(640, 480)

        self.container = tk.Frame(self, padx=12, pady=12)
        self.container.pack(fill="both", expand=True)

        self.header_manual = tk.Label(self.container, text="Viel MANUELLEN Erfolg Dir!!!", font=("TkDefaultFont", 16, "bold"))
        self.header_manual.grid(row=0, column=0, columnspan=3, pady=(0, 12))

        self.header_agent = tk.Label(self.container, text="Dies wird vom Agenten gesteuert!!!", font=("TkDefaultFont", 16, "bold"))
        self.header_agent.grid(row=3, column=0, columnspan=3, pady=(12, 12))

        probabilities = [0.20, 0.40, 0.80]
        self.bandit_frames = []
        for i, p in enumerate(probabilities, start=1):
            bf = BanditFrame(self.container, i, p, bd=1, relief="groove", padx=8, pady=6)
            bf.grid(row=1, column=i-1, padx=8, sticky="nsew")
            self.bandit_frames.append(bf)

        self.agent_buttons_frame = tk.Frame(self.container)
        self.agent_buttons_frame.grid(row=4, column=0, columnspan=3, pady=(12, 0))

        self.agent_once_btn = tk.Button(self.agent_buttons_frame, text="Agent EINMAL", command=self.agent_once)
        self.agent_once_btn.grid(row=0, column=0, padx=5)

        self.agent_all_btn = tk.Button(self.agent_buttons_frame, text="Agent ALLE Iterationen", command=self.agent_all)
        self.agent_all_btn.grid(row=0, column=1, padx=5)

        self.reset_btn = tk.Button(self.agent_buttons_frame, text="RESET STATE", command=self.reset_state)
        self.reset_btn.grid(row=0, column=2, padx=5)

        self.cumulative_rewards = []
        self.iteration_count = 0

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.container)
        self.canvas.get_tk_widget().grid(row=5, column=0, columnspan=3)

    def agent_once(self):
        for bandit_frame in self.bandit_frames:
            bandit_frame.on_click()

    def agent_all(self):
        for _ in range(100):  # Simulate 100 iterations
            self.agent_once()
            self.update_plot()

    def reset_state(self):
        for bandit_frame in self.bandit_frames:
            bandit_frame.clicks = 0
            bandit_frame.total = 0
            bandit_frame.last = 0
            bandit_frame.update_labels()
        self.cumulative_rewards.clear()
        self.ax.clear()
        self.canvas.draw()

    def update_plot(self):
        self.iteration_count += 1
        # Here you would calculate cumulative rewards based on your logic
        # For demonstration, we will just append a random value
        self.cumulative_rewards.append(random.randint(0, 100))
        self.ax.clear()
        self.ax.plot(self.cumulative_rewards, label='Cumulative Reward')
        self.ax.set_title('Cumulative Rewards Over Iterations')
        self.ax.set_xlabel('Iterations')
        self.ax.set_ylabel('Cumulative Reward')
        self.ax.legend()
        self.canvas.draw()

def main():
    app = BanditGUI()
    app.mainloop()

if __name__ == "__main__":
    main()