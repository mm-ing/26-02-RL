import threading
import tkinter as tk
from tkinter import ttk
from typing import Tuple
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time

try:
    from .gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer
except Exception:
    from gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer


class Tooltip:
    """Simple tooltip for tkinter widgets."""

    def __init__(self, widget, text: str):
        self.widget = widget
        self.text = text
        self.tip = None
        widget.bind('<Enter>', self.show)
        widget.bind('<Leave>', self.hide)

    def show(self, _=None):
        if self.tip:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f'+{x}+{y}')
        label = tk.Label(self.tip, text=self.text, background='yellow', relief='solid', borderwidth=1)
        label.pack()

    def hide(self, _=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


class GridWorldGUI(tk.Tk):
    def __init__(self, grid: Grid):
        super().__init__()
        self.title('GridWorld RL')
        self.grid_obj = grid
        self.trainer = Trainer(grid)
        self.policy = None
        self.agent_instance = None
        self.training = False
        self.dragging = None  # 'agent' or 'target' or None
        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        left = ttk.Frame(self)
        left.grid(row=0, column=0, sticky='nsew')
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky='nsew')

        # Canvas for grid
        self.canvas = tk.Canvas(left, width=400, height=300, bg='white')
        self.canvas.grid(row=0, column=0, padx=10, pady=10)
        self.canvas.bind('<Button-1>', self._on_canvas_click)
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.draw_grid()

        # Controls
        row = 0
        ttk.Label(right, text='alpha').grid(row=row, column=0, sticky='w')
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_e = ttk.Entry(right, textvariable=self.alpha_var, width=10)
        alpha_e.grid(row=row, column=1)
        Tooltip(alpha_e, 'Learning rate for Q-learning updates')

        row += 1
        ttk.Label(right, text='gamma').grid(row=row, column=0, sticky='w')
        self.gamma_var = tk.DoubleVar(value=0.99)
        gamma_e = ttk.Entry(right, textvariable=self.gamma_var, width=10)
        gamma_e.grid(row=row, column=1)
        Tooltip(gamma_e, 'Discount factor for future rewards')

        row += 1
        ttk.Label(right, text='max steps/ep').grid(row=row, column=0, sticky='w')
        self.max_steps_var = tk.IntVar(value=200)
        max_e = ttk.Entry(right, textvariable=self.max_steps_var, width=10)
        max_e.grid(row=row, column=1)
        Tooltip(max_e, 'Maximum steps allowed in a single episode')

        row += 1
        ttk.Label(right, text='episodes').grid(row=row, column=0, sticky='w')
        self.episodes_var = tk.IntVar(value=100)
        episodes_e = ttk.Entry(right, textvariable=self.episodes_var, width=10)
        episodes_e.grid(row=row, column=1)
        Tooltip(episodes_e, 'Number of episodes to train')

        row += 1
        # Policy selection
        self.policy_var = tk.StringVar(value='q')
        ttk.Radiobutton(right, text='Q-learning', variable=self.policy_var, value='q').grid(row=row, column=0)
        ttk.Radiobutton(right, text='Monte Carlo', variable=self.policy_var, value='mc').grid(row=row, column=1)
        Tooltip(self, 'Select the learning algorithm')

        row += 1
        ttk.Button(right, text='Run single step', command=self._run_single_step).grid(row=row, column=0, columnspan=2, pady=3, sticky='ew')
        Tooltip(self, 'Execute a single environment step and update policy')

        row += 1
        ttk.Button(right, text='Run single episode', command=self._run_single_episode).grid(row=row, column=0, columnspan=2, pady=3, sticky='ew')
        Tooltip(self, 'Run one episode of max steps and update policy')

        row += 1
        ttk.Button(right, text='Train and Run', command=self._train_and_run).grid(row=row, column=0, columnspan=2, pady=3, sticky='ew')
        Tooltip(self, 'Train selected policy for configured episodes')

        row += 1
        self.live_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(right, text='Live plot', variable=self.live_plot_var).grid(row=row, column=0, columnspan=2, sticky='w')

        row += 1
        ttk.Button(right, text='Show Value Table', command=self._show_value_table).grid(row=row, column=0, pady=3, sticky='ew')
        ttk.Button(right, text='Show Q-table', command=self._show_q_table).grid(row=row, column=1, pady=3, sticky='ew')

        row += 1
        ttk.Button(right, text='Save samplings CSV', command=self._save_csv).grid(row=row, column=0, columnspan=2, pady=3, sticky='ew')

        # Plot area
        fig = Figure(figsize=(4, 2))
        self.ax = fig.add_subplot(111)
        self.canvas_fig = FigureCanvasTkAgg(fig, master=right)
        self.canvas_fig.get_tk_widget().grid(row=row+1, column=0, columnspan=2)

    def draw_grid(self):
        self.canvas.delete('all')
        w = int(self.canvas['width'])
        h = int(self.canvas['height'])
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = w / cols
        ch = h / rows
        for x in range(cols):
            for y in range(rows):
                x0 = x * cw
                y0 = y * ch
                x1 = x0 + cw
                y1 = y0 + ch
                fill = 'lightgrey' if (x, y) in self.grid_obj.blocked else 'white'
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline='black')

        # target
        tx, ty = self.grid_obj.target
        self.canvas.create_rectangle(tx * cw + 4, ty * ch + 4, (tx + 1) * cw - 4, (ty + 1) * ch - 4,
                                     fill='green', outline='black')
        # agent
        ax, ay = self.grid_obj.start
        self.canvas.create_oval(ax * cw + 10, ay * ch + 10, (ax + 1) * cw - 10, (ay + 1) * ch - 10,
                                fill='blue')

    def _canvas_pos_to_cell(self, xpix, ypix):
        w = int(self.canvas['width'])
        h = int(self.canvas['height'])
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = w / cols
        ch = h / rows
        x = int(xpix // cw)
        y = int(ypix // ch)
        return max(0, min(cols - 1, x)), max(0, min(rows - 1, y))

    def _on_canvas_click(self, event):
        cell = self._canvas_pos_to_cell(event.x, event.y)
        # toggle blocked unless clicking agent or target
        if cell == self.grid_obj.start or cell == self.grid_obj.target:
            return
        if cell in self.grid_obj.blocked:
            self.grid_obj.blocked.remove(cell)
        else:
            self.grid_obj.blocked.add(cell)
        self.draw_grid()

    def _on_press(self, event):
        cell = self._canvas_pos_to_cell(event.x, event.y)
        if cell == self.grid_obj.start:
            self.dragging = 'agent'
        elif cell == self.grid_obj.target:
            self.dragging = 'target'
        else:
            self.dragging = None

    def _on_motion(self, event):
        # optional: provide visual feedback while dragging
        pass

    def _on_release(self, event):
        if not self.dragging:
            return
        cell = self._canvas_pos_to_cell(event.x, event.y)
        if cell in self.grid_obj.blocked:
            self.dragging = None
            return
        if self.dragging == 'agent':
            self.grid_obj.start = cell
        elif self.dragging == 'target':
            self.grid_obj.target = cell
        self.dragging = None
        self.draw_grid()

    def _run_single_step(self):
        policy = self._create_policy()
        # run one step
        transitions, total = self.trainer.run_episode(policy, epsilon=0.1, max_steps=1)
        self._update_plot([total])
        self.draw_grid()

    def _run_single_episode(self):
        policy = self._create_policy()
        transitions, total = self.trainer.run_episode(policy, epsilon=0.1, max_steps=self.max_steps_var.get())
        self._update_plot([total])
        self.draw_grid()

    def _train_and_run(self):
        if self.training:
            return
        self.training = True

        def worker():
            policy = self._create_policy()
            rewards = []
            for ep in range(self.episodes_var.get()):
                transitions, total = self.trainer.run_episode(policy, epsilon=0.1, max_steps=self.max_steps_var.get())
                rewards.append(total)
                if self.live_plot_var.get():
                    self.after(0, lambda r=rewards.copy(): self._update_plot(r))
            # final plot
            self.after(0, lambda r=rewards: self._update_plot(r))
            self.training = False

        threading.Thread(target=worker, daemon=True).start()

    def _create_policy(self):
        if self.policy_var.get() == 'q':
            q = QLearningAgent(self.grid_obj, alpha=self.alpha_var.get(), gamma=self.gamma_var.get())
            self.agent_instance = q
            return q
        else:
            mc = MonteCarloAgent(self.grid_obj, gamma=self.gamma_var.get())
            self.agent_instance = mc
            return mc

    def _update_plot(self, rewards):
        self.ax.clear()
        self.ax.plot(rewards)
        self.ax.set_xlabel('Episode')
        self.ax.set_ylabel('Total Reward')
        self.canvas_fig.draw()

    def _show_q_table(self):
        if not isinstance(self.agent_instance, QLearningAgent):
            tk.messagebox.showinfo('Q-table', 'Q-table only available for Q-learning agent')
            return
        top = tk.Toplevel(self)
        top.title('Q-table')
        text = tk.Text(top, width=60, height=20)
        text.pack()
        for (s, a), v in sorted(self.agent_instance.Q.items(), key=lambda x: (x[0][0], x[0][1])):
            text.insert('end', f's={s}, a={a} -> {v:.3f}\n')

    def _show_value_table(self):
        if not isinstance(self.agent_instance, MonteCarloAgent):
            tk.messagebox.showinfo('Value table', 'Value table available for Monte Carlo agent')
            return
        top = tk.Toplevel(self)
        top.title('Value table')
        text = tk.Text(top, width=40, height=20)
        text.pack()
        for s, v in sorted(self.agent_instance.V.items()):
            text.insert('end', f's={s} -> {v:.3f}\n')

    def _save_csv(self):
        # train and save samplings
        base = f'grid_samplings_M{self.grid_obj.M}_N{self.grid_obj.N}_episodes{self.episodes_var.get()}'
        policy = self._create_policy()
        # use Trainer.train which supports save_csv base name
        def worker():
            self.trainer.train(policy, num_episodes=self.episodes_var.get(), max_steps=self.max_steps_var.get(), epsilon=0.1, save_csv=base)
            self.after(0, lambda: tk.messagebox.showinfo('Saved', f'Samplings saved with base {base}'))

        threading.Thread(target=worker, daemon=True).start()
