Goal
----
Set up a lunar lander reinforcement learning project based on the gymnasium API. Use the project architecture and GUI design defined in `RL_initial_2.0.md`.

High-level summary
------------------
- project-name: `LunarLander`
- 3 policies/approaches:
    - double DQN
    - dueling DQN
    - combined double + dueling DQN (D3QN)

Functional requirements (detailed)
----------------------------------
Environment & logic:
- set up `Lunar Lander` environment; use command: `gym.make('LunarLander-v3')` and pygame for animation
- allways use `continuous=False`
- endable different setups for the game: 
    - `gravity` (default = -10.0)
    - `enable_wind` (default = False)
    - `wind_power` (default = 15.0)
    - `turbulence_power` (default = 1.5)
- Policies:
    - dueling DQN
        - class `DuelingDQN`
    - combined double + dueling DQN
        - class `D3QN`
    - double DQN + Prioritized Experience Replay
        - class `DDQN+PER`
    - suggest optimal parameter settings for this project and implement them as default settings for each policy

GUI behavior and layout:
- in the `Evironment` panel:
    - show the gymnasium animation of lunar lander
- in the `Parameter` panel, in the group `Environment`:
    - add input "gravity" which adjusts `gravity` of the gymnasium setup
    - add toggle "wind on" which adjust `enable_wind` of the gymnasium setup
    - add input "wind power" which adjusts `wind_power` of the gymnasium setup
    - add input "turbulence" which adjusts `turbulence_power` of the gymnasium setup





