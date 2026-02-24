Goal
----
Set up a taxi reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial.md`.

High-level summary
------------------
- project-name: `Taxi`
- 4 policies/approaches: 
    - vanilla deep Q-learning (DQN)
    - double DQN
    - dueling DQN
    - proritzed DQN
- expand GUI by additional DNN panel

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up `Taxi` environment; use command: `gym.make('Taxi-v3')` and pygame for animation
- endable different setups for the game: `is_raining` (default false), `fickle_passenger` (default false)
- Policies:
    - vanilla DQN 
        - class `DQN`
        - suggest and implement default settings
    - double DQN
        - class `DoubleDQN`
        - suggest and implement default settings
    - dueling DQN
        - class `DuelingDQN`
        - suggest and implement default settings
    - prioritized DQN
        - class `PrioDQN`
        - suggest and implement default settings

GUI behavior and layout:
- in the evironment panel:
    - show the gymnasium animation of taxi
    - add toggle "raining" which adjusts `is_raining` of the gymnasium setup
    - add toggle "running man" which adjusts `fickle_passenger` of the gymnasium setup
- between panels `Controls` and `Training Parameters` add a `DNN Parameters` panel with tight width, same height as `Training Parameters` 
- add relevant input fields for `Training Parameters` and `DNN Parameters`
- in the live plot add a moving average plot (average over 20 values) of the reward with doubled line width; add input field `MA N values` to `Training Parameters` panel to adjust the average




