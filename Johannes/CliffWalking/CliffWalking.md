Goal
----
Set up a cliff walking reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial.md`.

High-level summary
------------------
- project-name: `CliffWalking`
- 2 policies/approaches: 
    - vanilla deep Q-learning (DQN)
    - double DQN
- expand GUI by additional DNN panel

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up cliff walking environment; use command: `gym.make('CliffWalking-v1')` and pygame for animation
- The cliff can be chosen to be slippery (disabled by default) so the player may move perpendicular to the intended direction sometimes
- Policies:
    - vanilla DQN 
        - class `DQNetwork`
        - suggest and implement default settings
    - double DQN
        - class `DDQNetwork`
        - suggest and implement default settings

GUI behavior and layout:
- in the evironment panel:
    - show the gymnasium animation of cliff walking
    - add toggle "slippery cliff"
- between panels `Controls` and `Training Parameters` add a `DNN Parameters` panel with tight width, same height as `Training Parameters` 
- add relevant input fields for `Training Parameters` and `DNN Parameters`
- in the live plot add a moving average plot of the reward with doubled line width




