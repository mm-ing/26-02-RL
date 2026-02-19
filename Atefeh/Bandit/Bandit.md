You are an AI coding assistant. Build/extend a Python project for a Multi-Armed Bandit (Reinforcement Learning) demo with a functional Tkinter GUI. Follow all requirements exactly.

CONCEPTS / TERMINOLOGY
- In this Bandit task we operate in STEPS (not episodes). One STEP = choose one bandit (one arm pull) + receive reward.
- An EPISODE is a sequence of n steps, but this task has no natural terminal end, so we will LIMIT the number of steps by a user-defined “Agent Loops / N steps”.

CORE REQUIREMENTS
1) Rewards MUST be Bernoulli:
   - Each bandit returns reward r ∈ {0, 1} only (win/lose). No multi-coin payouts, no arbitrary reward magnitudes.

2) Agent memory:
   - Maintain an “agent memory” of past samples with a maximum size of 100 entries.
   - If memory size is set to 0, treat it as “use full memory”.
   - The policy/agent may use this memory/history to compute its internal estimates.

3) Policies (implement BOTH):
   A) Epsilon-Greedy policy
      - Provide epsilon (ε) parameter controlling exploration vs exploitation.
      - Start with default ε = 0.9 to make the logic obvious (mostly explore at the beginning).
      - Provide an epsilon decay parameter (decay) so ε decreases over steps.
      - Example default decay = 0.001 (slow decrease).
      - If decay = 0, epsilon should be allowed to be small (e.g., 0.1) and remain constant.
      - Make it easy to experiment with epsilon and decay.

   B) Thompson Sampling policy
      - Implement Thompson Sampling as an alternative policy.
      - Thompson Sampling must NOT use epsilon or decay; it decides purely algorithmically from the sampled history/probabilities.

ARCHITECTURE / FILE STRUCTURE (class-based, clean separation)
Create/extend these files (project name prefix “bandit_” is fine):
- bandit_app.py
  - Entry point only:
    - instantiate Logic
    - instantiate GUI
    - inject Logic into GUI
    - start GUI mainloop
    - handle shutdown/cleanup
  - Keep this file minimal (no big logic here).

- bandit_logic.py
  - Implement as classes (do NOT bury logic inside GUI):
    1) class Environment
       - Represents the bandit environment (the slot machines).
       - Holds N bandits with fixed Bernoulli reward probabilities.
       - Provides a method to “pull” a chosen bandit and return reward 0/1.

    2) class Agent
       - Interacts with Environment.
       - Uses a Policy to select actions.
       - Updates internal statistics/estimates each step.
       - Stores memory (max 100 entries).
       - Exposes data needed by the GUI for display/plot.

    3) Policy classes / interface
       - EpsilonGreedyPolicy (with epsilon and optional decay)
       - ThompsonSamplingPolicy (no epsilon/decay)
       - The Agent can switch between policies.

- bandit_gui.py
  - Tkinter GUI implementation (functional, not “pretty”).
  - One SINGLE GUI window, structured as:
    - TOP: three manual bandit buttons (Bandit 1/2/3) that allow the user to pull each bandit manually.
      - Manual button presses should still update the same “current state” statistics (counts, wins, etc.).
    - BELOW: agent controls and displays described below.

GUI CONTROLS (Buttons)
- “Single Step” button:
  - Executes exactly one agent step (select bandit via current policy, pull it, update stats).
  - Designed so the user can watch step-by-step behavior.

- “Run N Steps” (Agent Loops) button:
  - Runs N steps (N user-configurable) and updates displays/plot along the way.

- “Reset” button:
  - Resets environment/agent statistics and GUI state so parameter experiments can restart cleanly.

CURRENT STATE DISPLAY (runtime, always visible)
Show, at runtime (update every step):
- Pull counts per bandit (e.g., bandit1 pulled 3 times, bandit2 2 times, …)
- Win counts per bandit
- Success rate per bandit (wins / pulls) computed at runtime
- Current epsilon value (and/or epsilon schedule state) for epsilon-greedy
- Cumulative reward so far
- Currently selected policy name

PLOT REQUIREMENTS
- Include a plot in the GUI using matplotlib (embed in Tkinter).
- The plot must have:
  - A legend indicating the policy used
  - Different colors for different policies/series
- Prefer a LIVE plot:
  - As the agent runs steps/loops, update the plot incrementally (e.g., each loop/step adds a point/segment),
  - rather than only plotting once at the end.

USABILITY / STYLE
- The interface must be functional and easy to operate for algorithm experiments; it does not need to be beautiful.
- Keep code modular and readable.
- Provide sensible defaults (ε=0.9, decay=0.001, memory max 100, and three bandits).

OUTPUT
- Provide complete runnable Python code for all files:
  - bandit_app.py, bandit_logic.py, bandit_gui.py
- Include brief instructions to run the program (e.g., python bandit_app.py).
