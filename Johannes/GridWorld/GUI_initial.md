Prompt to recreate `gridworld_gui.py`

Goal
----
Create a single, self-contained Python module `gridworld_gui.py` that implements the GUI for the GridWorld project. The GUI must match the current layout and behavior exactly so it can be dropped into the repository and run together with the existing `gridworld_logic.py` (which provides `Grid`, agents, and `Trainer`). Provide robust, well-structured code that handles layout, threading, plotting, image rendering, and UI responsiveness.

Top-level requirements
---------------------
- Use Python 3 + tkinter/ttk for the GUI.
- Use matplotlib (TkAgg) for the embedded live plot.
- Use Pillow (PIL) if available to render anti-aliased agent/target images; gracefully degrade to canvas drawing if Pillow is not installed.
- Do not block the Tk mainloop: run training in background threads and perform GUI updates via `after()` on the main thread.
- Keep PhotoImage references on the GUI object to avoid garbage collection issues.
- The GUI file should import symbols from `gridworld_logic.py` (Grid, QLearningAgent, MonteCarloAgent, SARSAAgent, ExpectedSarsaAgent, Trainer) when available in the same package.

Layout and widget structure
---------------------------
Arrange the main window using a grid with two columns. The UI is split into labeled panels (ttk.LabelFrame) as follows (top → bottom):

- **Environment** (row 0, spans both columns):
  - Left: A scrollable Canvas that displays the GridWorld. The canvas must use a fixed per-cell pixel size stored in `self.cell_size` (default 60). The canvas is wrapped in a frame with vertical and horizontal scrollbars and the canvas `scrollregion` should be set to the full grid pixel size (cols * cell_size, rows * cell_size).
  - Right (inside the same `Environment` frame): environment inputs (to the right of the canvas) including fields for `grid M (cols)`, `grid N (rows)`, `start x,y`, `target x,y`, `Apply grid`, `Reset map`, and blocked-field inputs (blocked x, blocked y, Add blocked, Remove selected, blocked list). Use small paddings (tight spacing) between labels and inputs.

- **Controls** (below Environment in column 0): a 2×4 grid of buttons, arranged as:
  - Row0: `Run single step` | `Show Value Table`
  - Row1: `Run single episode` | `Show Q-table`
  - Row2: `Train and Run` | `Save samplings CSV`
  - Row3: `Reset All` | `Save Plot PNG`
  Buttons should expand horizontally in their cells (sticky='ew').

- **Current State** (below Controls, column 0): a compact LabelFrame containing a single-line label `state_line_label` which displays the live status in the format: `Training:` or `Idle:` followed by `step: {st:4d}  episode: {ep:4d}` (space-padded width=4). The label must be updated frequently but without redrawing the canvas.

- **Training Parameters** (to the right of Controls, column 1, spanning Controls + Current State rows): a tight LabelFrame with training inputs (do not expand horizontally):
  - `alpha` (default 0.5)
  - `gamma` (default 0.99)
  - `max steps/ep` (default 20)
  - `episodes` (default 100)
  - `epsilon min` (default 0.1)
  - `epsilon max` (default 0.9)   <-- default changed
  - `policy` dropdown values: `['Q-learning', 'Monte Carlo', 'SARSA', 'Expected SARSA']` (readonly)
  - `Live plot` checkbox (default True) aligned left
  - `reduced speed` checkbox (default True) aligned to the right of `Live plot` — when enabled, the training worker sleeps 33ms between sequential episodes to make progress observable.
  Use tight horizontal padding between label and field; right-align the input widgets inside this panel.

- **Live Plot** (bottom, spans both columns): an embedded matplotlib FigureCanvasTkAgg showing episode reward series. Keep past runs in `self.plot_runs` and draw current-in-progress rewards as a dashed line. Provide clickable legend entries to toggle visibility.

Core behaviors and interactions
------------------------------
- Canvas drawing:
  - Draw grid cells at pixel size `cell_size` (x loop cols, y loop rows), with blocked cells filled `lightgrey` and others `white`.
  - Draw target using a rounded rectangle image (PIL) or canvas fallback (green rounded rect).
  - Draw agent using a small anti-aliased circular image (PIL) or canvas oval fallback (blue).
  - Maintain `_agent_img` and `_target_img` attributes to keep PhotoImage references.
  - Draw agent path segments with red lines and arrowheads and tag them `agent_path` so they can be cleared.
  - Maintain `self._agent_path` list and draw it when present.
  - Update canvas `scrollregion` to (0,0,total_w,total_h) where total_w = M * cell_size, total_h = N * cell_size.

- Mouse and drag interactions:
  - Map widget coords to canvas coords using `canvasx`/`canvasy` and compute cell via integer division by `cell_size`.
  - Left-click on a non-agent/target cell toggles blocked (with reachability check via `Grid.is_reachable()`); show hover preview `hover` when moving mouse over cells showing tentative block color (red if it would block the path).
  - Dragging agent/target shows a preview image (use `_preview_agent_img` / `_preview_target_img` to keep references) tagged `preview` and updates its coords; releasing places agent/target unless on blocked cell.

- Controls:
  - `Run single step` keeps a persistent policy instance for single-step accumulation; each click advances by one step (epsilon from epsilon_min), updates model (QL update if present), appends to `_ongoing_transitions`, and animates that single transition without clearing prior path. Update `current_step` using `_set_current_counters` (do not redraw canvas fully just for counters).
  - `Run single episode` resets agent to configured start, runs `Trainer.run_episode(policy, ...)` once and animates the episode transitions sequentially; set `current_episode` to 1 and `current_step` to 0 before running.
  - `Train and Run` executes in a background thread; for each episode:
    - update `current_episode` via `after(0, ...)` helper so GUI label shows progress,
    - pass a `progress_callback` to `Trainer.run_episode` so GUI receives per-step callbacks; schedule per-step updates to the GUI thread with `after(0, ... self._set_current_counters(ep, step))` so `step` updates live,
    - append total reward to `rewards` list and (if `live_plot` active) schedule a plot update with `after(0, lambda: _update_plot(current_rewards=rewards.copy()))`.
    - if `reduced speed` is enabled, sleep 0.033 seconds between episodes to slow progress for observation.
  - `Reset All` requests stop (set `_stop_requested`), clears plot runs and map, and resets `agent_instance`.
  - `Save samplings CSV` calls `Trainer.train(..., save_csv=base)` or equivalent to produce CSV in `results_csv/`.
  - `Save Plot PNG` writes the embedded matplotlib figure to `plots/` with a filename encoding parameters: policy, eps min/max, alpha, gamma, episodes, max_steps, and a timestamp.

Plotting and performance
-----------------------
- Keep a timestamp `self._last_plot_update` and throttle `_update_plot` to no more than once every 150 ms to avoid UI slowdowns.
- Avoid expensive full canvas redraws when updating the `Current State` counters; update the single-line label directly in `_set_current_counters`.
- `draw_grid()` should still be called when agent moves (animation) or when the map changes, but avoid calling it from per-step progress callbacks — instead only update the counters and let animation code invoke `draw_grid()` for visual movement.

Image generation utilities (PIL optional)
--------------------------------------
- Implement `_create_agent_image(w,h)` and `_create_target_image(w,h)` that render anti-aliased images using PIL (draw at a larger scale and downsample using LANCZOS). If PIL not available, draw simple ovals/rounded rects on the canvas.

Legend and plot interactivity
----------------------------
- Make the figure legend clickable. When a legend line is clicked toggle the corresponding run's visibility in `self.plot_runs` and call `_redraw_all_plots()`.

Persistence and defaults
------------------------
- Defaults to set:
  - `cell_size = 60`
  - `alpha = 0.5`
  - `gamma = 0.99`
  - `max_steps = 20`
  - `episodes = 100`
  - `epsilon_min = 0.1`
  - `epsilon_max = 0.9` (default changed)
  - `live_plot = True`
  - `reduced_speed = True`

Robustness and threading
------------------------
- Use `threading.Thread(..., daemon=True)` for background training.
- Use try/except defensively around GUI updates scheduled with `after`.
- Avoid calling matplotlib draw on the worker thread; always call plotting updates on the GUI thread using `after`.

File placement and imports
--------------------------
- Place `gridworld_gui.py` in the same package/folder as `gridworld_logic.py` and import the Trainer and agent classes with a local package import if possible, with a fallback absolute import for running as a script:

  try:
      from .gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer, SARSAAgent, ExpectedSarsaAgent
  except Exception:
      from gridworld_logic import Grid, QLearningAgent, MonteCarloAgent, Trainer, SARSAAgent, ExpectedSarsaAgent

Developer guidance
-------------------
- Keep widget names and variable names reasonably consistent (e.g., `self.canvas`, `self.cell_size`, `self._agent_img`, `self._preview_agent_img`, `self.plot_runs`, `self.state_line_label`, `self.current_step`, `self.current_episode`, `self.reduced_speed_var`, `self.live_plot_var`, `self._last_plot_update`).
- Provide helper functions for coordinate conversions (`_canvas_pos_to_cell`), image creation, drawing helpers (`_create_rounded_rect`), and a small `_set_current_counters(ep, step)` that updates label text on the GUI thread.
- Make initial window geometry reasonably wide (double measured required width) but cap to screen size.

Deliverable
-----------
Write the full `gridworld_gui.py` module implementing the above layout and behaviors so it runs with the existing `gridworld_logic.py` in the repo. The module should be clean, with clear separation between UI wiring and training logic, and should include inline comments describing non-obvious choices (e.g., why we throttle plotting, why we keep PhotoImage refs, etc.).

End of prompt.
