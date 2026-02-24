Goal
----
Set up a frozen lake reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial.md`.

High-level summary
------------------
- project-name: `FrozenLake`
- 4 policies/approaches: 
    - vanilla deep Q-learning (DQN)
    - double DQN
    - dueling DQN
    - proritzed DQN
- expand GUI by additional DNN panel

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up `Frozen Lake` environment; use command: `gym.make('FrozenLake-v1')` and pygame for animation
- endable different setups for the game: `is_slippery` (default true), `map_name` (default 4x4), `success_rate`(default 1.0/3.0)
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
    - show the gymnasium animation of frozen lake
    - add toggle "slippery" which adjusts `is_slippery` of the gymnasium setup
    - add input field `slippery rate` which adjusts `success_rate` of the gymnasium setup
    - add dropdown menu `map size` with options `4x4` and `8x8` which adjusts `map_name` of the gymnasium setup
- between panels `Controls` and `Training Parameters` add a `DNN Parameters` panel with tight width, same height as `Training Parameters` 
- add relevant input fields for `Training Parameters` and `DNN Parameters`
- in the live plot add a moving average plot (average over 20 values) of the reward with doubled line width; add input field `MA N values` to `Training Parameters` panel to adjust the average




