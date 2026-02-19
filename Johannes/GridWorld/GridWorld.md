# Labyrint

## Goal
About me:
- i am a reinforcement learning student, therefor give me advices and explanations regarding your solution for the following task
- i have advanced python programming knowledge

Task: 
- create a reinforcement project of a labyrinth 
- agent has to move to the target field from starting position

## Specifics
- generate a grid map with size (columns: x = M, rows: y = N)
    - fields are shown white
    - size is adjustable
    - default is N = 3, M = 5
- making a labyrinth from that by blocking some fields
    - blocked fields are shown grey
    - blocked fields can not be passed by the agent
    - field position is adjustable
    - there must always stay at least one path passable by the agent from start to target
    - default blocked fields: (2, 0), (2, 1)
- default starting position is in field (0, 2)
- default target position is in field (M, N)
- state (s): agent position on map (x-y-coordinates as tuple of int)
- actions (a): 
    - agent moves along the grid 
    - movement: up = 0, down = 1, left = 2, right = 3
    - no diagonal movement
- reward (r): 0 if position = target, -1 else
- world model: 
    - transition model
    - probability to get to next state s': p(s,a) --> s'
    - reward r --> p(s',r|s,a)
- policies: 
    - Monte Carlo
    - Q-learning
- learning rate alpha
- discount factor gamma

## Programming
- create 3 python files: gridworld_gui.py; gridworld_app.py; gridworld_logic.py
- use object oriented programming
- run tests
- write example commands, readme, requirements
### Entry point (gridworld_app.py)
- initialize all class objects
- access other scripts
### World logic (gridworld_logic.py)
- create necessary classe based on the specifics above such as agent, map, Monte Carlo, Q-learning
### GUI (gridworld_gui.py)
- use tkinter
- show tooltips for buttons with short explanations of what this value does
- grid map:
    - place this section on the left
    - show the grid map at the top of this section
    - show the blocked fields (grey)
    - show the target as green square with rounded corner
    - show the agent as blue circle
    - required input fields:
        - grid size
        - blocked fields, additionaly adjustable by clicking on the grid map
        - agent starting position, additionaly adjustable by drag and drop
        - target position, additionaly adjustable by drag and drop
- learning params:
    - show the learning section to the right
    - show plot in the top of the section
    - input fields:
        - gamma
        - alpha
        - max steps per episode
        - number of episodes
    - buttons:
        - select policy Monte Carlo or Q-learning
        - run single step
        - run single episode
        - train and run
        - live reward plot (matplotlib)
        - show value table
        - show Q-table
        - save samplings into csv; create file name based params