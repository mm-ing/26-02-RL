import tkinter as tk
from tkinter import ttk
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
            payout = random.randint(1, 100)  # Adjust payout range as needed
        else:
            payout = 0
        self.last = payout
        self.total += payout
        self.update_labels()

    def update_labels(self):
        self.clicks_label.config(text=f"Anzahl Klicks: {self.clicks}")
        self.value_label.config(text=f"Ausgabewert: {self.last} (Summe: {self.total})")

class GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3-Banditen")
        self.minsize(640, 480)

        self.container = tk.Frame(self, padx=12, pady=12)
        self.container.pack(fill="both", expand=True)

        self.header = tk.Label(self.container, text="Viel MANUELLEN Erfolg Dir!!!", font=("TkDefaultFont", 16, "bold"))
        self.header.grid(row=0, column=0, columnspan=3, pady=(0, 12))

        self.bandit_frames = []
        probabilities = [0.20, 0.40, 0.80]
        for i, p in enumerate(probabilities, start=1):
            bf = BanditFrame(self.container, i, p, bd=1, relief="groove", padx=8, pady=6)
            bf.grid(row=1, column=i-1, padx=8, sticky="nsew")
            self.bandit_frames.append(bf)

        self.agent_frame = tk.Frame(self.container)
        self.agent_frame.grid(row=2, column=0, columnspan=3, pady=(12, 0))

        self.agent_label = tk.Label(self.agent_frame, text="Dies wird vom Agenten gesteuert!!!", font=("TkDefaultFont", 16, "bold"))
        self.agent_label.pack()

        self.policy_var = tk.StringVar(value="epsilon")
        self.epsilon_radio = ttk.Radiobutton(self.agent_frame, text="Epsilon-Greedy", variable=self.policy_var, value="epsilon")
        self.thompson_radio = ttk.Radiobutton(self.agent_frame, text="Thompson Sampling", variable=self.policy_var, value="thompson")
        self.epsilon_radio.pack(side=tk.LEFT, padx=5)
        self.thompson_radio.pack(side=tk.LEFT, padx=5)

        self.agent_once_btn = tk.Button(self.agent_frame, text="Agent EINMAL", command=self.agent_once)
        self.agent_all_btn = tk.Button(self.agent_frame, text="Agent ALLE Iterationen", command=self.agent_all)
        self.reset_btn = tk.Button(self.agent_frame, text="RESET STATE", command=self.reset_state)

        self.agent_once_btn.pack(side=tk.LEFT, padx=5)
        self.agent_all_btn.pack(side=tk.LEFT, padx=5)
        self.reset_btn.pack(side=tk.LEFT, padx=5)

        self.figure, self.ax = plt.subplots()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.container)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=3)

        self.ax.set_title("Cumulative Reward vs Iteration")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Cumulative Reward")

    def agent_once(self):
        pass  # Implement agent logic for one iteration

    def agent_all(self):
        pass  # Implement agent logic for multiple iterations

    def reset_state(self):
        for bf in self.bandit_frames:
            bf.clicks = 0
            bf.total = 0
            bf.last = 0
            bf.update_labels()

    def update_plot(self, iterations, rewards):
        self.ax.clear()
        self.ax.plot(iterations, rewards)
        self.ax.set_title("Cumulative Reward vs Iteration")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Cumulative Reward")
        self.canvas.draw_idle()

if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()