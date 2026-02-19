# tjs-bandit Project

## Overview
The tjs-bandit project is a three-armed bandit game implemented in Python using Tkinter for the graphical user interface (GUI) and reinforcement learning techniques to optimize the agent's performance. The game allows users to interact with three bandits, each with different payout probabilities, and includes an agent that learns to maximize rewards through exploration and exploitation strategies.

## Project Structure
```
tjs-bandit
├── TJS_bandit_13.py          # Standalone GUI for the bandit game
├── TJS_bandit_app.py         # Entry point for the application
├── TJS_bandit_gui.py         # GUI class with manual and agent-controlled sections
├── TJS_bandit_logic.py       # Core logic for the bandit game
├── requirements.txt          # Project dependencies
├── README.md                 # Project documentation
├── tests                     # Unit tests for the project
│   └── test_bandit.py        # Tests for bandit logic
└── docs                      # Documentation
    └── design.md             # Design decisions and architecture
```

## Setup Instructions
1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd tjs-bandit
   ```

2. **Install Dependencies**
   Ensure you have Python 3 installed. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**
   To start the game, run the following command:
   ```
   python TJS_bandit_app.py
   ```

## Usage
- The GUI consists of two sections:
  - **Manual Section**: Users can click on the buttons representing each bandit to play the game manually.
  - **Agent Section**: An agent can be activated to automatically play the game and learn which bandit yields the most coins.

- Each bandit has a fixed payout probability (20%, 40%, 80%) and uses Bernoulli logic for payouts.

- The game tracks the number of clicks and total payouts for each bandit, updating the display in real-time.

## Game Mechanics
- Each play costs 1 coin, and the payout is determined by the bandit's probability.
- The agent employs reinforcement learning strategies, including Epsilon-Greedy and Thompson Sampling, to optimize its actions over 100 iterations.

## Testing
Unit tests are provided in the `tests` directory. To run the tests, use:
```
pytest tests/test_bandit.py
```

## Documentation
For detailed design decisions and architecture, refer to `docs/design.md`. 

This project aims to provide an engaging way to explore the concepts of probability, reinforcement learning, and user interaction through a fun game format.