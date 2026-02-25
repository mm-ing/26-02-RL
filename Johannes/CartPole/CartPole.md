Goal
----
Set up a cart pole reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial.md`.

High-level summary
------------------
- project-name: `CartPole`
- 3 policies/approaches:
    - double DQN
    - dueling DQN
    - combined double + dueling DQN (D3QN)
- expand GUI by additional DNN panel

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up `Cart Pole` environment; use command: `gym.make('CartPole-v1')` and pygame for animation
- endable different setups for the game: `sutton_barto_reward` (default false)
- Policies:
    - double DQN
        - class `DoubleDQN`
        - suggest and implement default settings
    - dueling DQN
        - class `DuelingDQN`
        - suggest and implement default settings
    - combined double + dueling DQN
        - class `D3QN`
        - suggest and implement default settings

GUI behavior and layout:
- in the evironment panel:
    - show the gymnasium animation of taxi
    - add toggle "sutton_barto_reward" which adjusts `sutton_barto_reward` of the gymnasium setup
- between panels `Controls` and `Training Parameters` add a `DNN Parameters` panel with tight width, same height as `Training Parameters` 
- add relevant input fields for `Training Parameters` and `DNN Parameters`





