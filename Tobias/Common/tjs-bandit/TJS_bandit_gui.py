import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from TJS_bandit_logic import Environment, Agent, EpsilonGreedyPolicy, ThompsonSamplingPolicy

class BanditGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("3-Armed Bandit Game")
        self.master.geometry("800x600")

        self.create_widgets()
        self.setup_plot()

        self.env = Environment(starts=(20, 40, 80))
        self.agent = Agent(self.env, EpsilonGreedyPolicy())
        self.iterations = 100
        self.rewards = []

    def create_widgets(self):
        # Upper section
        upper_frame = ttk.Frame(self.master)
        upper_frame.pack(pady=10)

        upper_label = ttk.Label(upper_frame, text="Viel MANUELLEN Erfolg Dir!!!", font=("Arial", 16))
        upper_label.pack(pady=10)

        self.bandit_buttons = []
        for i in range(3):
            btn = ttk.Button(upper_frame, text=f"Bandit {i + 1}", command=lambda i=i: self.pull_bandit(i))
            btn.pack(side=tk.LEFT, padx=10)
            self.bandit_buttons.append(btn)

        # Lower section
        lower_frame = ttk.Frame(self.master)
        lower_frame.pack(pady=10)

        lower_label = ttk.Label(lower_frame, text="Dies wird vom Agenten gesteuert!!!", font=("Arial", 16))
        lower_label.pack(pady=10)

        self.agent_button_once = ttk.Button(lower_frame, text="Agent EINMAL", command=self.run_agent_once)
        self.agent_button_once.pack(side=tk.LEFT, padx=10)

        self.agent_button_all = ttk.Button(lower_frame, text="Agent ALLE Iterationen", command=self.run_agent_all)
        self.agent_button_all.pack(side=tk.LEFT, padx=10)

        self.reset_button = ttk.Button(lower_frame, text="RESET STATE", command=self.reset_game)
        self.reset_button.pack(side=tk.LEFT, padx=10)

    def setup_plot(self):
        self.figure, self.ax = plt.subplots()
        self.ax.set_title("Cumulative Reward")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Reward")
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.master)
        self.canvas.get_tk_widget().pack()

    def pull_bandit(self, index):
        reward = self.env.step(index)
        self.rewards.append(reward)
        self.update_plot()

    def run_agent_once(self):
        reward = self.agent.step()
        self.rewards.append(reward)
        self.update_plot()

    def run_agent_all(self):
        for _ in range(self.iterations):
            reward = self.agent.step()
            self.rewards.append(reward)
        self.update_plot()

    def reset_game(self):
        self.env.reset()
        self.rewards.clear()
        self.ax.clear()
        self.setup_plot()

    def update_plot(self):
        self.ax.clear()
        self.ax.plot(range(len(self.rewards)), self.rewards, label='Cumulative Reward')
        self.ax.set_title("Cumulative Reward")
        self.ax.set_xlabel("Iterations")
        self.ax.set_ylabel("Reward")
        self.ax.legend()
        self.canvas.draw_idle()

if __name__ == "__main__":
    root = tk.Tk()
    app = BanditGUI(root)
    root.mainloop()