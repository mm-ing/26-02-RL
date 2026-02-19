import tkinter as tk
from TJS_bandit_logic import Environment, Agent, EpsilonGreedyPolicy, ThompsonSamplingPolicy
import threading

class BanditApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3-Armed Bandit Game")
        self.environment = Environment(starts=(20, 40, 80))
        self.policy = EpsilonGreedyPolicy(epsilon=0.9, decay=0.01)
        self.agent = Agent(self.environment, self.policy)
        self.setup_gui()

    def setup_gui(self):
        self.gui = BanditGUI(self.root, self.environment, self.agent)
        self.gui.pack(fill="both", expand=True)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = BanditApp()
    app.run()