import threading
import tkinter as tk
from tkinter import ttk
from typing import Tuple
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import time
import itertools
import random
import os
import datetime
try:
    from PIL import Image, ImageDraw, ImageTk
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

try:
    from .gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer, SARSAAgent, ExpectedSarsaAgent
except Exception:
    from gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer, SARSAAgent, ExpectedSarsaAgent


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
        # resize debounce id
        self._resize_after_id = None
        self.grid_obj = grid
        self.trainer = Trainer(grid)
        self.policy = None
        self.agent_instance = None
        self.training = False
        self._stop_requested = False
        self.dragging = None  # 'agent' or 'target' or None
        self._agent_path = []
        self._animating = False
        self._build_ui()

    def _build_ui(self):
        # make root scalable; keep training params column non-expanding (tight)
        self.rowconfigure(0, weight=1)
        # Controls column will expand, training params column kept tight (weight=0)
        self.columnconfigure(0, weight=9)
        self.columnconfigure(1, weight=0)
        # main layout: Environment (top-left), Controls (below env),
        # Training Parameters (right of controls) and Live Plot (bottom)
        self.rowconfigure(0, weight=1)   # Environment (resizable)
        self.rowconfigure(1, weight=0)   # Controls (fixed)
        self.rowconfigure(2, weight=0)   # Current State (fixed)
        self.rowconfigure(3, weight=1)   # Live Plot (resizable)
        # keep training parameters column tight (non-expanding)
        self.columnconfigure(0, weight=9)
        self.columnconfigure(1, weight=0)

        # Environment panel (contains grid and environment inputs)
        env_frame = ttk.LabelFrame(self, text='Environment', padding=6)
        env_frame.grid(row=0, column=0, columnspan=2, sticky='nsew', padx=8, pady=8)
        env_frame.rowconfigure(0, weight=1)
        env_frame.columnconfigure(0, weight=1)

        # Canvas for grid (resizes with its frame) - make scrollable
        # fixed cell pixel size so cells keep same size when grid dimensions change
        # increased by factor 2 per user request
        self.cell_size = 60
        canvas_wrap = ttk.Frame(env_frame)
        canvas_wrap.grid(row=0, column=0, sticky='nsew')
        canvas_wrap.rowconfigure(0, weight=1)
        canvas_wrap.columnconfigure(0, weight=1)
        self.canvas = tk.Canvas(canvas_wrap, width=400, height=300, bg='white')
        self.canvas.grid(row=0, column=0, sticky='nsew')
        self.vbar = ttk.Scrollbar(canvas_wrap, orient='vertical', command=self.canvas.yview)
        self.vbar.grid(row=0, column=1, sticky='ns')
        self.hbar = ttk.Scrollbar(canvas_wrap, orient='horizontal', command=self.canvas.xview)
        self.hbar.grid(row=1, column=0, sticky='ew')
        self.canvas.configure(yscrollcommand=self.vbar.set, xscrollcommand=self.hbar.set)
        self.canvas.bind('<Configure>', lambda e: self.draw_grid())

        # Environment-specific controls (to the right of canvas, fixed size)
        controls_env = ttk.Frame(env_frame)
        env_frame.columnconfigure(1, weight=0)
        controls_env.grid(row=0, column=1, sticky='nw', padx=(8,6), pady=6)

        # Controls panel (below Environment) with buttons
        controls_frame = ttk.LabelFrame(self, text='Controls', padding=6)
        controls_frame.grid(row=1, column=0, sticky='nsew', padx=8, pady=(0,4))
        controls_frame.columnconfigure(0, weight=1)

        # Current State panel (below Controls, left of Training Parameters)
        current_state = ttk.LabelFrame(self, text='Current State', padding=6)
        current_state.grid(row=2, column=0, sticky='nsew', padx=8, pady=(0,8))
        current_state.columnconfigure(0, weight=1)
        # Single-line current state label (updated in draw_grid)
        self.state_line_label = ttk.Label(current_state, text='')
        self.state_line_label.grid(row=0, column=0, sticky='w', padx=2, pady=1)
        # live counters (episode/step) used to populate the state line
        self.current_episode = 0
        self.current_step = 0

        # Training parameters panel (to the right of Controls)
        training_params = ttk.LabelFrame(self, text='Training Parameters', padding=6)
        # keep this frame as tight as possible: don't expand to fill column
        training_params.grid(row=1, column=1, rowspan=2, sticky='ne', padx=8, pady=(0,8))
        training_params.columnconfigure(0, weight=0)

        # bind canvas mouse events
        self.canvas.bind('<ButtonPress-1>', self._on_press)
        self.canvas.bind('<B1-Motion>', self._on_motion)
        self.canvas.bind('<ButtonRelease-1>', self._on_release)
        self.canvas.bind('<Motion>', self._on_hover)
        self.canvas.bind('<Leave>', lambda e: self.canvas.delete('hover'))
        self.draw_grid()

        # Controls (placed inside controls_env)
        row = 0
        ttk.Label(controls_env, text='grid M (cols)').grid(row=row, column=0, sticky='w')
        self.grid_M_var = tk.IntVar(value=self.grid_obj.M)
        grid_m_e = ttk.Entry(controls_env, textvariable=self.grid_M_var, width=6)
        grid_m_e.grid(row=row, column=1, sticky='w')
        Tooltip(grid_m_e, 'Number of columns (M)')

        row += 1
        ttk.Label(controls_env, text='grid N (rows)').grid(row=row, column=0, sticky='w')
        self.grid_N_var = tk.IntVar(value=self.grid_obj.N)
        grid_n_e = ttk.Entry(controls_env, textvariable=self.grid_N_var, width=6)
        grid_n_e.grid(row=row, column=1, sticky='w')
        Tooltip(grid_n_e, 'Number of rows (N)')

        row += 1
        ttk.Label(controls_env, text='start x,y').grid(row=row, column=0, sticky='w')
        self.start_x_var = tk.IntVar(value=self.grid_obj.start[0])
        self.start_y_var = tk.IntVar(value=self.grid_obj.start[1])
        start_x_e = ttk.Entry(controls_env, textvariable=self.start_x_var, width=4)
        start_x_e.grid(row=row, column=1, sticky='w')
        start_y_e = ttk.Entry(controls_env, textvariable=self.start_y_var, width=4)
        start_y_e.grid(row=row, column=1, sticky='e')
        Tooltip(start_x_e, 'Start X coordinate')
        Tooltip(start_y_e, 'Start Y coordinate')

        row += 1
        ttk.Label(controls_env, text='target x,y').grid(row=row, column=0, sticky='w')
        self.target_x_var = tk.IntVar(value=self.grid_obj.target[0])
        self.target_y_var = tk.IntVar(value=self.grid_obj.target[1])
        target_x_e = ttk.Entry(controls_env, textvariable=self.target_x_var, width=4)
        target_x_e.grid(row=row, column=1, sticky='w')
        target_y_e = ttk.Entry(controls_env, textvariable=self.target_y_var, width=4)
        target_y_e.grid(row=row, column=1, sticky='e')
        Tooltip(target_x_e, 'Target X coordinate')
        Tooltip(target_y_e, 'Target Y coordinate')

        row += 1
        btn_apply = ttk.Button(controls_env, text='Apply grid', command=self._apply_grid_settings)
        btn_apply.grid(row=row, column=0, columnspan=1, pady=3, sticky='w')
        Tooltip(btn_apply, 'Apply grid size and start/target positions')
        reset_btn = ttk.Button(controls_env, text='Reset map', command=self._reset_map)
        reset_btn.grid(row=row, column=1, columnspan=1, sticky='w', padx=(8,0))
        Tooltip(reset_btn, 'Clear agent path on the map and place agent at start')

        # Blocked-field inputs placed to the right within controls_env
        ttk.Label(controls_env, text='blocked x').grid(row=0, column=2, sticky='w', padx=(10, 0))
        self.blocked_x_var = tk.IntVar(value=0)
        blocked_x_e = ttk.Entry(controls_env, textvariable=self.blocked_x_var, width=6)
        blocked_x_e.grid(row=0, column=3, sticky='w')
        Tooltip(blocked_x_e, 'Blocked field X coordinate')

        ttk.Label(controls_env, text='blocked y').grid(row=1, column=2, sticky='w', padx=(10, 0))
        self.blocked_y_var = tk.IntVar(value=0)
        blocked_y_e = ttk.Entry(controls_env, textvariable=self.blocked_y_var, width=6)
        blocked_y_e.grid(row=1, column=3, sticky='w')
        Tooltip(blocked_y_e, 'Blocked field Y coordinate')

        add_btn = ttk.Button(controls_env, text='Add blocked', command=self._add_blocked_from_input)
        add_btn.grid(row=2, column=2, columnspan=2, sticky='w', padx=(10,0))
        Tooltip(add_btn, 'Add a blocked field coordinate')

        self.blocked_listbox = tk.Listbox(controls_env, height=4)
        self.blocked_listbox.grid(row=3, column=2, columnspan=2, rowspan=3, sticky='nsew', padx=(10,0))
        self._refresh_blocked_listbox()

        remove_btn = ttk.Button(controls_env, text='Remove selected', command=self._remove_selected_blocked)
        remove_btn.grid(row=6, column=2, columnspan=2, sticky='w', padx=(10,0), pady=(4,0))
        Tooltip(remove_btn, 'Remove selected blocked field from the list')

        # polish fonts and spacing for env controls
        try:
            default_font = tkfont.nametofont('TkDefaultFont')
            default_font.configure(size=10)
        except Exception:
            default_font = None
        for child in controls_env.winfo_children():
            try:
                child.grid_configure(padx=1, pady=1)
            except Exception:
                pass

        # Training parameters (placed into training_params frame on the right)
        row = 0
        ttk.Label(training_params, text='alpha').grid(row=row, column=0, sticky='w')
        self.alpha_var = tk.DoubleVar(value=0.5)
        alpha_e = ttk.Entry(training_params, textvariable=self.alpha_var, width=10)
        alpha_e.grid(row=row, column=1, sticky='e')
        Tooltip(alpha_e, 'Learning rate for Q-learning updates')

        row += 1
        ttk.Label(training_params, text='gamma').grid(row=row, column=0, sticky='w')
        self.gamma_var = tk.DoubleVar(value=0.99)
        gamma_e = ttk.Entry(training_params, textvariable=self.gamma_var, width=10)
        gamma_e.grid(row=row, column=1, sticky='e')
        Tooltip(gamma_e, 'Discount factor for future rewards')

        row += 1
        ttk.Label(training_params, text='max steps/ep').grid(row=row, column=0, sticky='w')
        self.max_steps_var = tk.IntVar(value=20)
        max_e = ttk.Entry(training_params, textvariable=self.max_steps_var, width=10)
        max_e.grid(row=row, column=1, sticky='e')
        Tooltip(max_e, 'Maximum steps allowed in a single episode')

        row += 1
        ttk.Label(training_params, text='episodes').grid(row=row, column=0, sticky='w')
        self.episodes_var = tk.IntVar(value=100)
        episodes_e = ttk.Entry(training_params, textvariable=self.episodes_var, width=10)
        episodes_e.grid(row=row, column=1, sticky='e')
        Tooltip(episodes_e, 'Number of episodes to train')

        row += 1
        ttk.Label(training_params, text='epsilon min').grid(row=row, column=0, sticky='w')
        self.epsilon_min_var = tk.DoubleVar(value=0.1)
        eps_min_e = ttk.Entry(training_params, textvariable=self.epsilon_min_var, width=10)
        eps_min_e.grid(row=row, column=1, sticky='e')
        Tooltip(eps_min_e, 'Minimum exploration epsilon (used for single-step/single-episode and Q-learning)')

        row += 1
        ttk.Label(training_params, text='epsilon max').grid(row=row, column=0, sticky='w')
        self.epsilon_max_var = tk.DoubleVar(value=0.9)
        eps_max_e = ttk.Entry(training_params, textvariable=self.epsilon_max_var, width=10)
        eps_max_e.grid(row=row, column=1, sticky='e')
        Tooltip(eps_max_e, 'Maximum exploration epsilon (used for Monte Carlo training)')

        row += 1
        # Policy selection (label + dropdown)
        self.policy_var = tk.StringVar(value='Q-learning')
        ttk.Label(training_params, text='policy').grid(row=row, column=0, sticky='w')
        policy_box = ttk.Combobox(training_params, textvariable=self.policy_var, values=['Q-learning', 'Monte Carlo', 'SARSA', 'Expected SARSA'], state='readonly', width=18)
        policy_box.grid(row=row, column=1, sticky='e')
        Tooltip(policy_box, 'Select learning policy')
        # tighten spacing in training_params (small horizontal gap)
        try:
            for child in training_params.winfo_children():
                try:
                    child.grid_configure(padx=(0,2), pady=1)
                except Exception:
                    pass
        except Exception:
            pass

        # Controls buttons arranged in 2 columns x 4 rows
        controls_frame.columnconfigure(0, weight=1)
        controls_frame.columnconfigure(1, weight=1)
        # Row 0
        btn_step = ttk.Button(controls_frame, text='Run single step', command=self._run_single_step)
        btn_step.grid(row=0, column=0, padx=4, pady=3, sticky='ew')
        btn_show_val = ttk.Button(controls_frame, text='Show Value Table', command=self._show_value_table)
        btn_show_val.grid(row=0, column=1, padx=4, pady=3, sticky='ew')
        Tooltip(btn_step, 'Execute a single environment step and update policy')
        Tooltip(btn_show_val, 'Show state-value table (Monte Carlo)')
        # Row 1
        btn_episode = ttk.Button(controls_frame, text='Run single episode', command=self._run_single_episode)
        btn_episode.grid(row=1, column=0, padx=4, pady=3, sticky='ew')
        btn_show_q = ttk.Button(controls_frame, text='Show Q-table', command=self._show_q_table)
        btn_show_q.grid(row=1, column=1, padx=4, pady=3, sticky='ew')
        Tooltip(btn_episode, 'Run one episode of max steps and update policy')
        Tooltip(btn_show_q, 'Show Q-table (Q-learning)')
        # Row 2
        btn_train = ttk.Button(controls_frame, text='Train and Run', command=self._train_and_run)
        btn_train.grid(row=2, column=0, padx=4, pady=3, sticky='ew')
        btn_save = ttk.Button(controls_frame, text='Save samplings CSV', command=self._save_csv)
        btn_save.grid(row=2, column=1, padx=4, pady=3, sticky='ew')
        Tooltip(btn_train, 'Train selected policy for configured episodes')
        Tooltip(btn_save, 'Save sampled transitions to CSV')
        # Row 3
        btn_reset_all = ttk.Button(controls_frame, text='Reset All', command=self._reset_all)
        btn_reset_all.grid(row=3, column=0, padx=4, pady=3, sticky='ew')
        btn_save_plot = ttk.Button(controls_frame, text='Save Plot PNG', command=self._save_plot_png)
        btn_save_plot.grid(row=3, column=1, padx=4, pady=3, sticky='ew')
        Tooltip(btn_reset_all, 'Stop training (if running), clear plot and reset map')
        Tooltip(btn_save_plot, 'Save current live plot to plots/*.png')

        # live plot checkbox remains in training parameters (left)
        self.live_plot_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(training_params, text='Live plot', variable=self.live_plot_var).grid(row=8, column=0, sticky='w')
        # reduced speed toggle (right of live plot) - default active
        self.reduced_speed_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(training_params, text='reduced speed', variable=self.reduced_speed_var).grid(row=8, column=1, sticky='e')

        # Plot area (bottom) as its own labeled frame that resizes
        plot_frame = ttk.LabelFrame(self, text='Live Plot', padding=6)
        plot_frame.grid(row=3, column=0, columnspan=2, sticky='nsew', padx=8, pady=(0,8))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        fig = Figure(figsize=(8, 3), dpi=100)
        self.ax = fig.add_subplot(111)
        self.ax.tick_params(labelsize=9)
        self.canvas_fig = FigureCanvasTkAgg(fig, master=plot_frame)
        self.canvas_fig.get_tk_widget().grid(row=0, column=0, sticky='nsew')
        # timestamp of last plot update to throttle frequent redraws
        self._last_plot_update = 0.0
        # plotting state: keep past runs
        self.plot_runs = []  # each: dict(rewards, label, color, visible)
        self._color_cycle = itertools.cycle(matplotlib.cm.get_cmap('tab10').colors)
        # connect pick handlers for clickable legend
        try:
            self.canvas_fig.mpl_connect('pick_event', self._on_legend_pick)
        except Exception:
            pass

        # refresh live plot when window resized (debounced)
        try:
            self.bind('<Configure>', self._on_root_configure)
        except Exception:
            pass

        # double initial width after layout computed (cap to screen)
        try:
            self.update_idletasks()
            req_w = self.winfo_reqwidth()
            req_h = self.winfo_reqheight()
            screen_w = self.winfo_screenwidth()
            screen_h = self.winfo_screenheight()
            new_w = min(req_w * 2, screen_w - 40)
            new_h = min(req_h, screen_h - 80)
            self.geometry(f'{new_w}x{new_h}')
        except Exception:
            pass

    def _on_root_configure(self, event):
        # debounce configure events and refresh plot after resize
        try:
            if self._resize_after_id:
                try:
                    self.after_cancel(self._resize_after_id)
                except Exception:
                    pass
            self._resize_after_id = self.after(150, lambda: self._update_plot())
        except Exception:
            pass

    def _set_current_counters(self, ep: int, step: int):
        """Set current episode/step on the GUI thread and refresh display."""
        try:
            # update counters but avoid expensive full redraw; update label only
            self.current_episode = ep
            self.current_step = step
            if hasattr(self, 'state_line_label'):
                status = 'Training:' if getattr(self, 'training', False) else 'Idle:'
                try:
                    # space-padded width 4 formatting to keep spacing stable
                    self.state_line_label.config(text=f'{status} step: {self.current_step:4d}  episode: {self.current_episode:4d}')
                except Exception:
                    # fallback simple update
                    self.state_line_label.config(text=f'{status} step: {self.current_step}  episode: {self.current_episode}')
        except Exception:
            pass

    def draw_grid(self):
        self.canvas.delete('all')
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = self.cell_size
        ch = self.cell_size
        total_w = int(cols * cw)
        total_h = int(rows * ch)
        # update scroll region so canvas becomes scrollable when grid larger than viewport
        try:
            self.canvas.configure(scrollregion=(0, 0, total_w, total_h))
        except Exception:
            pass
        # draw cells at fixed pixel size
        for x in range(cols):
            for y in range(rows):
                x0 = x * cw
                y0 = y * ch
                x1 = x0 + cw
                y1 = y0 + ch
                fill = 'lightgrey' if (x, y) in self.grid_obj.blocked else 'white'
                self.canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline='black')

        # target (use anti-aliased image if Pillow available)
        tx, ty = self.grid_obj.target
        margin = max(2, int(min(cw, ch) * 0.12))
        x0 = tx * cw + margin
        y0 = ty * ch + margin
        x1 = (tx + 1) * cw - margin
        y1 = (ty + 1) * ch - margin
        tw = int(x1 - x0)
        th = int(y1 - y0)
        if PIL_AVAILABLE and tw > 0 and th > 0:
            # create a rounded rectangle image sized to the cell
            img = self._create_target_image(max(4, tw), max(4, th))
            # keep reference to avoid GC
            self._target_img = ImageTk.PhotoImage(img, master=self)
            cx = (x0 + x1) / 2
            cy = (y0 + y1) / 2
            self.canvas.create_image(cx, cy, image=self._target_img)
        else:
            # fallback to canvas rounded rect
            self._create_rounded_rect(x0, y0, x1, y1, r=min(cw, ch) * 0.2, fill='green')
        # agent (use anti-aliased image if available)
        ax, ay = self.grid_obj.start
        margin = max(2, int(min(cw, ch) * 0.15))
        aw = int(cw - 2 * margin)
        ah = int(ch - 2 * margin)
        if PIL_AVAILABLE and aw > 0 and ah > 0:
            img = self._create_agent_image(max(4, aw), max(4, ah))
            self._agent_img = ImageTk.PhotoImage(img, master=self)
            cx = (ax * cw) + cw / 2
            cy = (ay * ch) + ch / 2
            self.canvas.create_image(cx, cy, image=self._agent_img)
        else:
            self.canvas.create_oval(ax * cw + margin, ay * ch + margin, (ax + 1) * cw - margin, (ay + 1) * ch - margin,
                                    fill='blue')
        # draw agent path if present
        try:
            self.canvas.delete('agent_path')
        except Exception:
            pass
        if getattr(self, '_agent_path', None):
            pts = self._agent_path
            # draw each segment with an arrow head indicating move direction
            for i in range(len(pts) - 1):
                px, py = pts[i]
                nx, ny = pts[i + 1]
                x0 = px * cw + cw / 2
                y0 = py * ch + ch / 2
                x1 = nx * cw + cw / 2
                y1 = ny * ch + ch / 2
                try:
                    self.canvas.create_line(x0, y0, x1, y1, fill='red', width=2, arrow='last', arrowshape=(10, 12, 4), tags='agent_path')
                except Exception:
                    # fallback to simple line
                    self.canvas.create_line(x0, y0, x1, y1, fill='red', width=2, tags='agent_path')
        # update current state single-line label if present
        try:
            if hasattr(self, 'state_line_label'):
                ep = int(getattr(self, 'current_episode', 0) or 0)
                st = int(getattr(self, 'current_step', 0) or 0)
                status = 'Training:' if getattr(self, 'training', False) else 'Idle:'
                # fixed-width space-padded 4-digit numbers to keep spacing stable
                self.state_line_label.config(text=f'{status} step: {st:4d}  episode: {ep:4d}')
        except Exception:
            pass

    def _create_rounded_rect(self, x0, y0, x1, y1, r=10, **kwargs):
        # draw a rounded rectangle by composing arcs and rectangles
        if r <= 0:
            return self.canvas.create_rectangle(x0, y0, x1, y1, **kwargs)
        # corners
        self.canvas.create_arc(x0, y0, x0 + 2 * r, y0 + 2 * r, start=90, extent=90, style='pieslice', **kwargs)
        self.canvas.create_arc(x1 - 2 * r, y0, x1, y0 + 2 * r, start=0, extent=90, style='pieslice', **kwargs)
        self.canvas.create_arc(x0, y1 - 2 * r, x0 + 2 * r, y1, start=180, extent=90, style='pieslice', **kwargs)
        self.canvas.create_arc(x1 - 2 * r, y1 - 2 * r, x1, y1, start=270, extent=90, style='pieslice', **kwargs)
        # center rectangles to fill
        self.canvas.create_rectangle(x0 + r, y0, x1 - r, y1, **kwargs)
        self.canvas.create_rectangle(x0, y0 + r, x1, y1 - r, **kwargs)

    def _create_target_image(self, w, h, radius_ratio=0.18):
        # Generate an anti-aliased rounded green square using PIL
        size = (max(4, int(w)), max(4, int(h)))
        img = Image.new('RGBA', size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        r = int(min(size) * radius_ratio)
        # draw rounded rectangle with antialias by drawing at double size then downsample
        scale = 4
        big = Image.new('RGBA', (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
        bd = ImageDraw.Draw(big)
        bx0, by0, bx1, by1 = 0, 0, size[0] * scale, size[1] * scale
        br = max(1, int(r * scale))
        # rounded rectangle
        bd.rounded_rectangle([bx0, by0, bx1, by1], radius=br, fill=(34, 139, 34, 255))
        # downsample with ANTIALIAS
        img = big.resize(size, resample=Image.LANCZOS)
        return img

    def _create_agent_image(self, w, h, radius_ratio=0.35):
        # create anti-aliased blue circle with light outline
        size = (max(4, int(w)), max(4, int(h)))
        scale = 4
        big = Image.new('RGBA', (size[0] * scale, size[1] * scale), (0, 0, 0, 0))
        bd = ImageDraw.Draw(big)
        bx0, by0, bx1, by1 = 0, 0, size[0] * scale, size[1] * scale
        # draw circle centered
        br = int(min(size) * radius_ratio * scale)
        cx = (bx0 + bx1) // 2
        cy = (by0 + by1) // 2
        bd.ellipse([cx - br, cy - br, cx + br, cy + br], fill=(30, 144, 255, 255))
        # light stroke
        st = max(1, int(scale))
        bd.ellipse([cx - br + st, cy - br + st, cx + br - st, cy + br - st], outline=(200, 220, 255, 120))
        img = big.resize(size, resample=Image.LANCZOS)
        return img

    def _apply_grid_settings(self):
        try:
            M = int(self.grid_M_var.get())
            N = int(self.grid_N_var.get())
            sx = int(self.start_x_var.get())
            sy = int(self.start_y_var.get())
            tx = int(self.target_x_var.get())
            ty = int(self.target_y_var.get())
        except Exception:
            tk.messagebox.showerror('Invalid input', 'Grid and positions must be integers')
            return
        # filter blocked fields to new bounds
        blocked = {(x, y) for (x, y) in self.grid_obj.blocked if 0 <= x < M and 0 <= y < N}
        # ensure start/target not blocked
        start = (max(0, min(M - 1, sx)), max(0, min(N - 1, sy)))
        target = (max(0, min(M - 1, tx)), max(0, min(N - 1, ty)))
        if start in blocked:
            blocked.remove(start)
        if target in blocked:
            blocked.remove(target)
        # recreate grid
        self.grid_obj = Grid(M=M, N=N, blocked=list(blocked), start=start, target=target)
        self.trainer.grid = self.grid_obj
        # update entry defaults
        self.grid_M_var.set(M)
        self.grid_N_var.set(N)
        self.start_x_var.set(start[0])
        self.start_y_var.set(start[1])
        self.target_x_var.set(target[0])
        self.target_y_var.set(target[1])
        self.draw_grid()

    def _add_blocked_from_input(self):
        try:
            x = int(self.blocked_x_var.get())
            y = int(self.blocked_y_var.get())
        except Exception:
            tk.messagebox.showerror('Invalid', 'Blocked coordinates must be integers')
            return
        cell = (x, y)
        if not (0 <= cell[0] < self.grid_obj.M and 0 <= cell[1] < self.grid_obj.N):
            tk.messagebox.showerror('Invalid', 'Blocked coordinates out of bounds')
            return
        if cell == self.grid_obj.start or cell == self.grid_obj.target:
            tk.messagebox.showwarning('Blocked', 'Cannot block start or target')
            return
        # tentative add and check path
        self.grid_obj.blocked.add(cell)
        if not self.grid_obj.is_reachable():
            self.grid_obj.blocked.remove(cell)
            tk.messagebox.showwarning('Blocked', 'That would block all paths to target. Action cancelled.')
            return
        self._refresh_blocked_listbox()
        self.draw_grid()

    def _remove_selected_blocked(self):
        # remove selected items from listbox and grid
        sel = list(self.blocked_listbox.curselection())
        if not sel:
            return
        to_remove = []
        for i in sel:
            try:
                item = self.blocked_listbox.get(i)
            except Exception:
                continue
            x, y = [int(s) for s in item.split(',')]
            to_remove.append((x, y))
        for cell in to_remove:
            if cell in self.grid_obj.blocked:
                self.grid_obj.blocked.remove(cell)
        self._refresh_blocked_listbox()
        self.draw_grid()

    def _refresh_blocked_listbox(self):
        # update the listbox with current blocked coords
        try:
            self.blocked_listbox.delete(0, 'end')
        except Exception:
            return
        for b in sorted(self.grid_obj.blocked):
            self.blocked_listbox.insert('end', f'{b[0]},{b[1]}')

    def _reset_map(self):
        # clear any drawn agent path
        try:
            # reset agent to configured start coordinates
            sx = int(self.start_x_var.get())
            sy = int(self.start_y_var.get())
            self.grid_obj.start = (sx, sy)
            self.trainer.grid = self.grid_obj
        except Exception:
            pass
        try:
            self._agent_path = []
            self.canvas.delete('agent_path')
            # also clear any ongoing single-step episode accumulation
            self._ongoing_states = None
            self._ongoing_rewards = None
            self._ongoing_transitions = None
        except Exception:
            pass
        self.draw_grid()

    def _canvas_pos_to_cell(self, xpix, ypix):
        # map widget coordinates to canvas coordinates (account for scrolling)
        cx = self.canvas.canvasx(xpix)
        cy = self.canvas.canvasy(ypix)
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = self.cell_size
        ch = self.cell_size
        x = int(cx // cw)
        y = int(cy // ch)
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
        self.press_cell = cell
        if cell == self.grid_obj.start:
            self.dragging = 'agent'
        elif cell == self.grid_obj.target:
            self.dragging = 'target'
        else:
            self.dragging = None

    def _on_motion(self, event):
        if not self.dragging:
            return
        cell = self._canvas_pos_to_cell(event.x, event.y)
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = self.cell_size
        ch = self.cell_size
        # draw or update preview
        if self.dragging == 'agent':
            margin = max(2, int(min(cw, ch) * 0.15))
            x0 = cell[0] * cw + margin
            y0 = cell[1] * ch + margin
            x1 = (cell[0] + 1) * cw - margin
            y1 = (cell[1] + 1) * ch - margin
            # use the agent image for preview (keep PhotoImage ref to avoid GC)
            try:
                aw = int(x1 - x0)
                ah = int(y1 - y0)
                if PIL_AVAILABLE and aw > 0 and ah > 0:
                    img = self._create_agent_image(max(4, aw), max(4, ah))
                    self._preview_agent_img = ImageTk.PhotoImage(img, master=self)
                    cx = cell[0] * cw + cw / 2
                    cy = cell[1] * ch + ch / 2
                    if not self.canvas.find_withtag('preview'):
                        self.canvas.create_image(cx, cy, image=self._preview_agent_img, tags='preview')
                    else:
                        # move and update image
                        items = self.canvas.find_withtag('preview')
                        for it in items:
                            try:
                                self.canvas.coords(it, cx, cy)
                                self.canvas.itemconfig(it, image=self._preview_agent_img)
                            except Exception:
                                pass
                else:
                    if not self.canvas.find_withtag('preview'):
                        self.canvas.create_oval(x0, y0, x1, y1, fill='blue', outline='', stipple='gray50', tags='preview')
                    else:
                        self.canvas.coords('preview', x0, y0, x1, y1)
            except Exception:
                # fallback to simple oval preview
                if not self.canvas.find_withtag('preview'):
                    self.canvas.create_oval(x0, y0, x1, y1, fill='blue', outline='', stipple='gray50', tags='preview')
                else:
                    self.canvas.coords('preview', x0, y0, x1, y1)
        elif self.dragging == 'target':
            margin = max(2, int(min(cw, ch) * 0.12))
            x0 = cell[0] * cw + margin
            y0 = cell[1] * ch + margin
            x1 = (cell[0] + 1) * cw - margin
            y1 = (cell[1] + 1) * ch - margin
            try:
                tw = int(x1 - x0)
                th = int(y1 - y0)
                if PIL_AVAILABLE and tw > 0 and th > 0:
                    img = self._create_target_image(max(4, tw), max(4, th))
                    self._preview_target_img = ImageTk.PhotoImage(img, master=self)
                    cx = cell[0] * cw + cw / 2
                    cy = cell[1] * ch + ch / 2
                    if not self.canvas.find_withtag('preview'):
                        self.canvas.create_image(cx, cy, image=self._preview_target_img, tags='preview')
                    else:
                        items = self.canvas.find_withtag('preview')
                        for it in items:
                            try:
                                self.canvas.coords(it, cx, cy)
                                self.canvas.itemconfig(it, image=self._preview_target_img)
                            except Exception:
                                pass
                else:
                    if not self.canvas.find_withtag('preview'):
                        self.canvas.create_rectangle(x0, y0, x1, y1, fill='green', outline='', stipple='gray50', tags='preview')
                    else:
                        self.canvas.coords('preview', x0, y0, x1, y1)
            except Exception:
                if not self.canvas.find_withtag('preview'):
                    self.canvas.create_rectangle(x0, y0, x1, y1, fill='green', outline='', stipple='gray50', tags='preview')
                else:
                    self.canvas.coords('preview', x0, y0, x1, y1)

    def _on_hover(self, event):
        # show a hover preview indicating if adding a block here would disconnect start->target
        # do nothing if currently dragging
        if getattr(self, 'dragging', None):
            return
        cell = self._canvas_pos_to_cell(event.x, event.y)
        # do not preview on start/target
        if cell == self.grid_obj.start or cell == self.grid_obj.target:
            self.canvas.delete('hover')
            return
        cols = self.grid_obj.M
        rows = self.grid_obj.N
        cw = self.cell_size
        ch = self.cell_size
        x0 = cell[0] * cw
        y0 = cell[1] * ch
        x1 = x0 + cw
        y1 = y0 + ch
        # tentative block
        will_block = False
        if cell not in self.grid_obj.blocked:
            self.grid_obj.blocked.add(cell)
            if not self.grid_obj.is_reachable():
                will_block = True
            self.grid_obj.blocked.remove(cell)
        # draw hover rect
        self.canvas.delete('hover')
        color = 'red' if will_block else 'lightgrey'
        self.canvas.create_rectangle(x0+1, y0+1, x1-1, y1-1, fill=color, outline='', stipple='gray25', tags='hover')

    def _on_release(self, event):
        release_cell = self._canvas_pos_to_cell(event.x, event.y)
        # if we were dragging, move agent/target
        if self.dragging:
            cell = release_cell
            if cell in self.grid_obj.blocked:
                # do not place on blocked
                pass
            else:
                if self.dragging == 'agent':
                    self.grid_obj.start = cell
                    # update input fields
                    try:
                        self.start_x_var.set(cell[0])
                        self.start_y_var.set(cell[1])
                    except Exception:
                        pass
                elif self.dragging == 'target':
                    self.grid_obj.target = cell
                    try:
                        self.target_x_var.set(cell[0])
                        self.target_y_var.set(cell[1])
                    except Exception:
                        pass
        else:
            # treat as click toggle only if press and release on same cell
            if hasattr(self, 'press_cell') and self.press_cell == release_cell:
                cell = release_cell
                if cell != self.grid_obj.start and cell != self.grid_obj.target:
                    # tentative toggle and check reachability
                    if cell in self.grid_obj.blocked:
                        self.grid_obj.blocked.remove(cell)
                        ok = self.grid_obj.is_reachable()
                        if not ok:
                            # shouldn't happen when unblocking, but guard
                            self.grid_obj.blocked.add(cell)
                            tk.messagebox.showwarning('Blocked', 'Action would leave no valid path (kept blocked).')
                    else:
                        self.grid_obj.blocked.add(cell)
                        if not self.grid_obj.is_reachable():
                            self.grid_obj.blocked.remove(cell)
                            tk.messagebox.showwarning('Blocked', 'That would block all paths to target. Action cancelled.')
                        else:
                            pass
                    self._refresh_blocked_listbox()

        # clear preview and redraw
        self.canvas.delete('preview')
        try:
            if hasattr(self, '_preview_agent_img'):
                del self._preview_agent_img
        except Exception:
            pass
        try:
            if hasattr(self, '_preview_target_img'):
                del self._preview_target_img
        except Exception:
            pass
        self.dragging = None
        self.draw_grid()

    def _run_single_step(self):
        if self._animating:
            return
        # ensure we keep a persistent policy instance for ongoing single-step episodes
        if not hasattr(self, 'agent_instance') or self.agent_instance is None:
            policy = self._create_policy()
        else:
            policy = self.agent_instance

        # initialize ongoing episode buffers if not present
        if not hasattr(self, '_ongoing_states') or self._ongoing_states is None:
            self._ongoing_states = [self.grid_obj.start]
            self._ongoing_rewards = []
            self._ongoing_transitions = []

        s = self.grid_obj.start
        # epsilon-greedy selection for single-step (small exploration)
        eps = self.epsilon_min_var.get()
        if hasattr(policy, 'best_action'):
            if random.random() < eps:
                a = random.choice(list(self.grid_obj.ACTIONS.keys()))
            else:
                a = policy.best_action(s)
        else:
            a = random.choice(list(self.grid_obj.ACTIONS.keys()))

        s2, r, done = self.grid_obj.step(s, a, noise=self.trainer.noise)

        # apply any immediate update (QLearning)
        if hasattr(policy, 'update'):
            policy.update(s, a, r, s2)

        # append to ongoing episode
        self._ongoing_transitions.append((s, a, r, s2, done))
        self._ongoing_rewards.append(r)
        self._ongoing_states.append(s2)

        # update live counters for single-step
        try:
            # episode remains as-is (0 or user-set); step is length of transitions
            self.current_step = len(self._ongoing_transitions)
            if not getattr(self, 'current_episode', None):
                self.current_episode = 0
        except Exception:
            pass

        # advance agent position
        try:
            self.grid_obj.start = s2
            self.trainer.grid = self.grid_obj
        except Exception:
            pass

        # animate this single transition without clearing prior path
        self._animate_transitions([(s, a, r, s2, done)], reset_before=False)

        # if episode finished, let MonteCarloAgent process the full episode and clear buffers
        if done:
            if isinstance(policy, MonteCarloAgent):
                policy.process_episode(self._ongoing_states, self._ongoing_rewards)
            # reset ongoing buffers so next single-step starts fresh
            self._ongoing_states = None
            self._ongoing_rewards = None
            self._ongoing_transitions = None

    def _run_single_episode(self):
        if self._animating:
            return
        # reset agent to configured start before running
        try:
            sx = int(self.start_x_var.get())
            sy = int(self.start_y_var.get())
            self.grid_obj.start = (sx, sy)
            self.trainer.grid = self.grid_obj
        except Exception:
            pass
        # clear any ongoing single-step accumulation
        self._ongoing_states = None
        self._ongoing_rewards = None
        self._ongoing_transitions = None
        policy = self._create_policy()
        # set current episode/step for UI (single episode run)
        try:
            self.current_episode = 1
            self.current_step = 0
        except Exception:
            pass
        transitions, total = self.trainer.run_episode(policy, epsilon=self.epsilon_min_var.get(), max_steps=self.max_steps_var.get())
        # animate full episode
        self._agent_path = [self.grid_obj.start]
        self._animate_transitions(transitions, reset_before=False)

    def _train_and_run(self):
        if self.training:
            return
        self.training = True
        self._stop_requested = False

        def worker():
            policy = self._create_policy()
            rewards = []
            episodes = self.episodes_var.get()
            alpha = self.alpha_var.get()
            gamma = self.gamma_var.get()
            # Read epsilon bounds from GUI: use epsilon_max for Monte Carlo, epsilon_min otherwise
            if self.policy_var.get() == 'Monte Carlo':
                epsilon = self.epsilon_max_var.get()
            else:
                epsilon = self.epsilon_min_var.get()
            max_steps = self.max_steps_var.get()
            for ep in range(episodes):
                if self._stop_requested:
                    break
                # update current episode/step via GUI thread helper so it redraws
                try:
                    self.after(0, self._set_current_counters, ep + 1, 0)
                except Exception:
                    pass
                # provide a progress callback so per-step updates refresh the UI
                try:
                    progress_cb = (lambda step, e=ep: self.after(0, self._set_current_counters, e + 1, step + 1))
                except Exception:
                    progress_cb = None
                transitions, total = self.trainer.run_episode(policy, epsilon=epsilon, max_steps=max_steps, progress_callback=progress_cb)
                rewards.append(total)
                if self.live_plot_var.get():
                    self.after(0, lambda r=rewards.copy(): self._update_plot(current_rewards=r))
                # optional reduced-speed delay to make progress observable
                try:
                    if getattr(self, 'reduced_speed_var', None) and self.reduced_speed_var.get():
                        time.sleep(0.033)
                except Exception:
                    pass
            # final: register run and redraw (mark if stopped)
            pol = self.policy_var.get()
            status = 'Stopped' if self._stop_requested else 'Completed'
            label = f"{pol} | α={alpha:.2f}, γ={gamma:.2f}, eps={epsilon}, ep={episodes} [{status}]"
            if rewards:
                self.after(0, lambda r=rewards, lab=label: self._add_plot_run(r, lab))
            # clear current episode/step after training finishes
            try:
                self.after(0, self._set_current_counters, 0, 0)
            except Exception:
                pass
            self.training = False

        threading.Thread(target=worker, daemon=True).start()

    def _create_policy(self):
        p = self.policy_var.get()
        if p == 'Q-learning':
            q = QLearningAgent(self.grid_obj, alpha=self.alpha_var.get(), gamma=self.gamma_var.get())
            self.agent_instance = q
            return q
        elif p == 'Monte Carlo':
            mc = MonteCarloAgent(self.grid_obj, gamma=self.gamma_var.get())
            self.agent_instance = mc
            return mc
        elif p == 'SARSA':
            sarsa = SARSAAgent(self.grid_obj, alpha=self.alpha_var.get(), gamma=self.gamma_var.get())
            self.agent_instance = sarsa
            return sarsa
        elif p == 'Expected SARSA':
            esarsa = ExpectedSarsaAgent(self.grid_obj, alpha=self.alpha_var.get(), gamma=self.gamma_var.get())
            self.agent_instance = esarsa
            return esarsa
        else:
            # fallback to Q-learning
            q = QLearningAgent(self.grid_obj, alpha=self.alpha_var.get(), gamma=self.gamma_var.get())
            self.agent_instance = q
            return q

    def _reset_all(self):
        # request stop for running training
        try:
            self._stop_requested = True
        except Exception:
            pass
        # clear training state and visualizations
        try:
            self.training = False
            self.plot_runs = []
            self._redraw_all_plots()
            # reset the map (also clears agent path)
            self._reset_map()
            self.agent_instance = None
        except Exception:
            pass
        try:
            tk.messagebox.showinfo('Reset All', 'Training stopped (if running) and map reset.')
        except Exception:
            pass

    def _animate_transitions(self, transitions, reset_before=False, delay=150):
        # transitions: list of (s,a,r,s2,done)
        if self._animating:
            return
        self._animating = True
        # optional reset handled by caller
        idx = 0

        def step():
            nonlocal idx
            if idx >= len(transitions):
                self._animating = False
                return
            s, a, r, s2, done = transitions[idx]
            # move agent to s2
            try:
                self.grid_obj.start = s2
                self.trainer.grid = self.grid_obj
            except Exception:
                pass
            # append to path and redraw
            if not hasattr(self, '_agent_path') or self._agent_path is None:
                self._agent_path = [s2]
            else:
                self._agent_path.append(s2)
            # update current step counter for live display
            try:
                self.current_step = idx + 1
            except Exception:
                pass
            self.draw_grid()
            idx += 1
            self.after(delay, step)

        # start animation
        self.after(0, step)

    def _add_plot_run(self, rewards, label):
        # register a completed run
        color = next(self._color_cycle)
        run_id = len(self.plot_runs) + 1
        full_label = f"Run {run_id}: {label}"
        run = {'id': run_id, 'rewards': list(rewards), 'label': full_label, 'color': color, 'visible': True}
        self.plot_runs.append(run)
        self._redraw_all_plots()

    def _redraw_all_plots(self, current_rewards=None):
        self.ax.clear()
        lines = []
        all_vals = []
        for run in self.plot_runs:
            lr = run['rewards']
            if not lr:
                continue
            line, = self.ax.plot(lr, '-o', color=run['color'], label=run['label'], alpha=1.0 if run['visible'] else 0.2, visible=run['visible'])
            lines.append(line)
            all_vals.extend(lr)
        # overlay current running rewards as dashed line
        if current_rewards:
            self.ax.plot(current_rewards, '--', color='gray', label='current (in progress)')
            all_vals.extend(current_rewards)

        self.ax.set_title('Episode rewards', fontsize=12)
        self.ax.set_xlabel('Episode', fontsize=11)
        self.ax.set_ylabel('Total Reward', fontsize=11)
        self.ax.tick_params(axis='both', which='major', labelsize=9)
        # always show horizontal zero line and ensure 0 is a y-tick
        try:
            # determine y-limits to include 0
            if all_vals:
                y_min = min(all_vals)
                y_max = max(all_vals)
            else:
                y_min, y_max = -1, 1
            y_min = min(y_min, 0)
            y_max = max(y_max, 0)
            margin = max(1.0, (y_max - y_min) * 0.1)
            self.ax.set_ylim(y_min - margin, y_max + margin)
            # add zero line
            self.ax.axhline(0, color='black', linestyle='--', linewidth=1)
            # ensure 0 tick visible
            yt = list(self.ax.get_yticks())
            if 0 not in yt:
                yt.append(0)
                yt = sorted(yt)
                self.ax.set_yticks(yt)
        except Exception:
            pass
        # legend at right
        try:
            leg = self.ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            # enable picking on legend lines and map them to run ids via label text
            for legline, text in zip(leg.get_lines(), leg.get_texts()):
                lab = text.get_text()
                # expected run labels start with 'Run <n>:'
                run_idx = None
                if lab.startswith('Run '):
                    try:
                        num = int(lab.split(':', 1)[0].split()[1])
                        run_idx = num - 1
                    except Exception:
                        run_idx = None
                # attach index if valid
                if run_idx is not None and 0 <= run_idx < len(self.plot_runs):
                    legline.set_picker(5)
                    legline._run_index = run_idx
        except Exception:
            pass
        try:
            self.canvas_fig.figure.tight_layout()
        except Exception:
            pass
        self.canvas_fig.draw()

    def _update_plot(self, current_rewards=None):
        # throttle frequent plot updates to avoid UI slowdowns
        try:
            now = time.time()
            last = getattr(self, '_last_plot_update', 0.0) or 0.0
            # minimum interval 150 ms
            if now - last < 0.15:
                return
            self._last_plot_update = now
        except Exception:
            pass
        # keep existing runs and optionally overlay current run
        self._redraw_all_plots(current_rewards=current_rewards)

    def _on_legend_pick(self, event):
        # toggle visibility of run corresponding to picked legend entry
        legline = event.artist
        run_idx = getattr(legline, '_run_index', None)
        if run_idx is None:
            return
        if run_idx < 0 or run_idx >= len(self.plot_runs):
            return
        run = self.plot_runs[run_idx]
        run['visible'] = not run.get('visible', True)
        self._redraw_all_plots()

    def _show_q_table(self):
        if not hasattr(self.agent_instance, 'get_q'):
            tk.messagebox.showinfo('Q-table', 'Q-table only available for agents with Q(s,a) (Q-learning/SARSA/Expected SARSA)')
            return
        top = tk.Toplevel(self)
        top.title('Q-table')
        # use a Treeview to show states as rows and actions as columns
        cols = ['State', 'Up', 'Down', 'Left', 'Right']
        tree = ttk.Treeview(top, columns=cols, show='headings', height=20)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=100, anchor='center')

        # vertical scrollbar
        vsb = ttk.Scrollbar(top, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        # list all states (exclude blocked)
        action_map = {0: 'Up', 1: 'Down', 2: 'Left', 3: 'Right'}
        states = []
        for y in range(self.grid_obj.N):
            for x in range(self.grid_obj.M):
                s = (x, y)
                if not self.grid_obj.is_blocked(s):
                    states.append(s)

        for s in sorted(states, key=lambda t: (t[1], t[0])):
            vals = [f'{s}']
            for a in [0, 1, 2, 3]:
                v = self.agent_instance.get_q(s, a) if hasattr(self.agent_instance, 'get_q') else 0.0
                vals.append(f'{v:.3f}')
            tree.insert('', 'end', values=vals)

    def _show_value_table(self):
        # Show a value table for states. Works for MonteCarloAgent (state-value)
        # and for QLearningAgent by deriving V(s) = max_a Q(s,a).
        top = tk.Toplevel(self)
        top.title('Value table')

        cols = ['State', 'Value']
        tree = ttk.Treeview(top, columns=cols, show='headings', height=20)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=120, anchor='center')

        vsb = ttk.Scrollbar(top, orient='vertical', command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        top.rowconfigure(0, weight=1)
        top.columnconfigure(0, weight=1)

        states = []
        for y in range(self.grid_obj.N):
            for x in range(self.grid_obj.M):
                s = (x, y)
                states.append(s)

        for s in sorted(states, key=lambda t: (t[1], t[0])):
            if self.grid_obj.is_blocked(s):
                val_str = 'BLOCKED'
            else:
                # Prefer explicit state-value if agent provides it
                if hasattr(self.agent_instance, 'get_v'):
                    v = self.agent_instance.get_v(s)
                elif hasattr(self.agent_instance, 'get_q'):
                    # derive V(s) = max_a Q(s,a)
                    v = max(self.agent_instance.get_q(s, a) for a in [0, 1, 2, 3])
                else:
                    v = 0.0
                val_str = f'{v:.3f}'
            tree.insert('', 'end', values=(f'{s}', val_str))

    def _save_csv(self):
        # train and save samplings
        base = f'grid_samplings_M{self.grid_obj.M}_N{self.grid_obj.N}_episodes{self.episodes_var.get()}'
        policy = self._create_policy()
        # use Trainer.train which supports save_csv base name
        def worker():
            self.trainer.train(policy, num_episodes=self.episodes_var.get(), max_steps=self.max_steps_var.get(), epsilon=self.epsilon_min_var.get(), save_csv=base)
            self.after(0, lambda: tk.messagebox.showinfo('Saved', f'Samplings saved with base {base}'))

        threading.Thread(target=worker, daemon=True).start()

    def _save_plot_png(self):
        # create plots directory and save current matplotlib figure
        try:
            plots_dir = os.path.join(os.getcwd(), 'plots')
            os.makedirs(plots_dir, exist_ok=True)
            # gather parameter values for filename
            policy = str(self.policy_var.get()).replace(' ', '_')
            eps_min = str(self.epsilon_min_var.get()).replace('.', 'p')
            eps_max = str(self.epsilon_max_var.get()).replace('.', 'p')
            alpha = str(self.alpha_var.get()).replace('.', 'p')
            gamma = str(self.gamma_var.get()).replace('.', 'p')
            episodes = str(self.episodes_var.get())
            max_steps = str(self.max_steps_var.get())
            ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            fname = f"{policy}_epsmin{eps_min}_epsmax{eps_max}_alpha{alpha}_gamma{gamma}_ep{episodes}_max{max_steps}_{ts}.png"
            path = os.path.join(plots_dir, fname)
            # save figure
            self.canvas_fig.figure.savefig(path, dpi=150)
            tk.messagebox.showinfo('Saved', f'Plot saved to {path}')
        except Exception as e:
            tk.messagebox.showerror('Error', f'Failed saving plot: {e}')
