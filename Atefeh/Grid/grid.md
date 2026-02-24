Below is the **updated, Copilot-ready main prompt** with **animated agent movement** added for:

1. Run Single Step
2. Run Single Episode
3. Train & Run

---

# Goal

Create a fully functional Reinforcement Learning project implementing a configurable Gridworld Labyrinth with GUI, Monte Carlo and Q-Learning policies, including **animated movement of the agent** during execution.

---

# About me

* I am a reinforcement learning student.
* I have advanced Python programming knowledge.
* Therefore:

  * Provide short explanations and educational comments regarding the RL logic.
  * Explain important implementation decisions.
  * Do NOT oversimplify the reinforcement learning concepts.
  * Focus on clean architecture and correctness.

---

# Task

Create a Reinforcement Learning project where:

An agent must move from a starting position to a target field in a labyrinth.

The system must allow:

* Dynamic grid configuration
* Interactive GUI
* Policy selection
* Live training visualization
* Saving trajectories
* **Animated agent movement in the GUI during Single Step, Single Episode, and Train & Run**

---

# Specifics

## Grid World

Generate a grid map with:

* Columns: x = M
* Rows: y = N
* Adjustable size
* Default: N = 3, M = 5

Fields:

* White = normal fields
* Grey = blocked fields
* Green (rounded square) = target
* Blue circle = agent

---

## Labyrinth Constraints

* Blocked fields cannot be passed.
* Field positions are adjustable.
* There must always remain at least one valid path from start to target.
* Default blocked fields: (2,0), (2,1)
* Default starting position: (0,2)
* Default target position: (M-1, N-1)

---

# Reinforcement Learning Model

## State (s)

Agent position:

```text
(x, y) as tuple[int, int]
```

---

## Actions (a)

Movement:

```text
0 = up
1 = down
2 = left
3 = right
```

No diagonal movement.

If action results in:

* Outside grid → stay in same state
* Blocked field → stay in same state

---

## Reward (r)

```text
0  → if state == target
-1 → otherwise
```

---

## World Model

Transition model:

```text
p(s, a) → s'
```

Reward model:

```text
p(s', r | s, a)
```

Environment is deterministic.

---

# Policies to Implement

## 1. Monte Carlo

* Episode-based
* Update only after reaching terminal state
* Learn state-value function V(s)
* Use epsilon-greedy
* Use discount factor γ

---

## 2. Q-Learning

* Tabular Q-learning
* Maintain Q-table: Q(s, a)
* Update rule:

```text
Q(s,a) = Q(s,a) + α [ r + γ * max_a' Q(s',a') - Q(s,a) ]
```

* Use epsilon-greedy exploration
* Use learning rate α
* Use discount factor γ

---

# Programming Requirements

## Files to Create

```text
gridworld_app.py      (Entry point)
gridworld_logic.py    (World + RL logic)
gridworld_gui.py      (GUI using tkinter)
```

Use object-oriented programming.

---

# Entry Point (gridworld_app.py)

* Initialize all class objects
* Connect GUI and logic
* Start application

---

# World Logic (gridworld_logic.py)

Create necessary classes:

* GridMap
* Agent
* Environment
* MonteCarlo
* QLearning
* EpsilonScheduler
* Trajectory / Sampling handler

Responsibilities:

* Grid generation
* State transitions
* Reward handling
* Policy execution
* Training loop
* Value and Q-table storage
* CSV export

Include:

* Clean separation between environment and policy
* Proper encapsulation
* Docstrings
* Inline educational explanations

**Important API requirement (for GUI animation):**
Expose step-wise control methods so the GUI can animate movement:

* `reset_episode()` → resets agent to start and returns initial state
* `step()` → executes exactly one transition and returns:

  ```text
  (prev_state, action, next_state, reward, done)
  ```
* `finish_episode()` → (Monte Carlo) applies end-of-episode updates using stored trajectory
* For Q-learning, updates may occur per-step (inside `step()` or immediately after each step)

---

# GUI (gridworld_gui.py)

Use `tkinter`.

---

## Layout

### Left Section: Grid Map

* Show grid at top
* White cells
* Grey blocked cells
* Green rounded square target
* Blue circle agent

Interactive features:

* Blocked fields adjustable by clicking on grid
* Agent starting position adjustable via drag & drop
* Target position adjustable via drag & drop

Required input fields:

* Grid size
* Blocked fields (editable list)
* Start position
* Target position

---

### Right Section: Learning Parameters

At top:

* Live reward plot (matplotlib embedded in tkinter)

Input fields:

* Gamma (γ)
* Alpha (α)
* Max steps per episode
* Number of episodes
* Epsilon max
* Epsilon min
* Epsilon decay

---

# Buttons

* Select Policy: Monte Carlo or Q-learning
* Run single step
* Run single episode
* Train and run
* Show value table
* Show Q-table
* Save samplings into CSV
* Rest

CSV filename must include:

* Policy name
* Alpha
* Gamma
* Episodes
* Timestamp

---

# Animated Movement Requirements

The agent must **visibly move on the grid in real time** for:

## 1) Run Single Step

* Execute exactly ONE environment transition.
* Immediately update/redraw the agent marker at the new position.
* GUI must remain responsive (no freezing).

## 2) Run Single Episode

* Run sequential steps until:

  * goal reached OR
  * max steps reached
* After EACH step:

  * animate agent movement on the grid
  * update status (episode, step, state, action, reward)
  * optional short delay (e.g., 50–150 ms) so movement is visible
* At episode end:

  * Monte Carlo: compute return and update V(s) via `finish_episode()`
  * Q-learning: ensure updates applied properly (per-step)

## 3) Train & Run

* Run N episodes.
* Must not block tkinter mainloop.
* Implement training using `tkinter.after()` scheduling (preferred).
* Provide either:

  * a “Visualize training” checkbox (on/off), OR
  * animate only every k-th episode for performance, OR both
* Reward plot must update live (at least per episode).
* Add a **Stop Training** button that stops gracefully after current step or episode.
* Disable conflicting buttons while training, re-enable when done/stopped.

**Implementation constraint:**

* Do NOT run long loops directly inside button callbacks.
* Use `after()` callbacks for step-by-step execution and animation.
* If you use threads, ensure tkinter updates happen only on the main thread via `after()`.

---

# Live Reward Plot

* X-axis: Episode number
* Y-axis: Episode reward (return)
* Must update live during training
* Use matplotlib (TkAgg embedding)
* Update plot efficiently (update data + `draw_idle()`)

---

# Additional Requirements

* Ensure at least one valid path exists before training
* Validate inputs
* Use proper exception handling
* Provide short RL explanations in comments
* Include unit tests
* Provide example command usage
* Provide README.md
* Provide requirements.txt

---

# Testing

Write tests for:

* Grid boundary logic
* Blocked field transitions
* Reward correctness
* Monte Carlo return calculation
* Q-learning update rule
* Epsilon decay
* Deterministic transition correctness (stay in place on blocked/out-of-bounds)

---

# Educational Requirement

Because I am learning reinforcement learning:

* Add short explanatory comments near:

  * Monte Carlo return computation
  * Q-learning update rule
  * Epsilon-greedy selection
  * Discounting logic
  * Why GUI uses `after()` for animation (event loop / responsiveness)

Keep explanations precise and technical.

---

# Final Deliverable

Generate:

* Complete working code
* All three Python files
* README
* requirements.txt
* Unit tests
* Example run instructions

The implementation must be correct, modular, professional, and include **animated agent movement** during Single Step, Single Episode, and Train & Run.
