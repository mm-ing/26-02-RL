# TJS Bandit Project

This project implements a three-armed bandit game using a graphical user interface (GUI) and reinforcement learning techniques. The game simulates a slot machine environment where players can interact with three different bandits, each with its own payout probability. The project includes both manual interaction and an automated agent that learns to maximize rewards.

## Project Structure

```
TJS_bandit_project
├── TJS_bandit_13.py         # Standalone GUI application for the bandit game
├── TJS_bandit_logic.py      # Core logic for the bandit game, including classes for Bandit, Environment, and Agent
├── TJS_bandit_gui.py        # Extended GUI functionality with agent controls and live plotting
├── TJS_bandit_app.py        # Entry point for initializing the application
├── requirements.txt          # Required Python packages for the project
├── .gitignore                # Files and directories to be ignored by version control
├── README.md                 # Documentation for the project
└── tests                     # Directory containing test files
    └── test_bandit.py       # Test stubs for the project
```

## Setup Instructions

1. **Clone the Repository**: 
   Clone this repository to your local machine using:
   ```
   git clone <repository-url>
   ```

2. **Navigate to the Project Directory**:
   ```
   cd TJS_bandit_project
   ```

3. **Install Requirements**:
   Install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

4. **Run the Application**:
   To start the standalone GUI application, run:
   ```
   python TJS_bandit_13.py
   ```
   or to run the application with reinforcement learning features:
   ```
   python TJS_bandit_app.py
   ```

## Usage

- The GUI will display three bandit buttons. Clicking on a button will simulate a pull of that bandit, costing one coin and potentially yielding a payout based on the bandit's probability.
- The agent can be controlled to perform actions automatically, with options to run a single iteration or multiple iterations.
- The cumulative rewards can be visualized in real-time if matplotlib is installed.

## Testing

To run the tests, use pytest:
```
pytest tests/test_bandit.py
```

## Additional Information

- Ensure that you have Python 3.x installed on your machine.
- This project uses Tkinter for the GUI and may require additional libraries for plotting.
- For any issues or contributions, please open an issue or submit a pull request on the repository.