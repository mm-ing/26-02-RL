Goal
----
Set up a mountain car reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial_2.0.md`.

High-level summary
------------------
- project-name: `MountainCar`
- 3 policies/approaches:
    - double DQN
    - dueling DQN
    - combined double + dueling DQN (D3QN)

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up `Mountain Car` environment; use command: `gym.make('MountainCar-v0')` and pygame for animation
- endable different setups for the game: 
    - `goal_velocity` (default = 0)
    - `x_init` (default = pi)
    - `y_init` (default = 1.0)
- Policies:
    - dueling DQN
        - class `DuelingDQN`
    - combined double + dueling DQN
        - class `D3QN`
    - double DQN + Prioritized Experience Replay
        - class `DDQN+PER`
    - suggest and implement default settings for each policy

GUI behavior and layout:
- in the `Evironment` panel:
    - show the gymnasium animation of mountain car
- in the `Parameter` panel, in the group `Environment`:
    - add input "goal velocity" which adjusts `goal_velocity` of the gymnasium setup
    - add input "x" which adjusts `x_init` of the gymnasium setup
    - add input "y" which adjusts `y_init` of the gymnasium setup





