# TJS Bandit Project

## Overview
The TJS Bandit Project is a Python application that simulates a multi-armed bandit environment using a graphical user interface (GUI). The project allows users to interact with three one-armed bandits, each with different payout probabilities, and includes an agent that can automatically interact with the bandits to maximize rewards. The application also features live plotting of cumulative rewards.

## Project Structure
```
TJS_bandit_project
├── TJS_bandit_13.py        # Standalone GUI application for user interaction with bandits
├── TJS_bandit_app.py       # Entry point for initializing the environment and GUI
├── TJS_bandit_gui.py       # GUI class that manages the layout and user interactions
├── TJS_bandit_logic.py     # Core logic for bandit environment and agent policies
├── requirements.txt         # List of dependencies for the project
├── README.md                # Documentation for setup and usage
└── tests                    # Directory containing test cases for the project
    └── test_bandit.py      # Test cases for validating bandit logic and functionality
```

## Setup Instructions
1. **Clone the Repository**
   ```
   git clone <repository-url>
   cd TJS_bandit_project
   ```

2. **Install Dependencies**
   Ensure you have Python 3 installed. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Run the Application**
   To start the application, run:
   ```
   python TJS_bandit_app.py
   ```

## Usage
- The GUI will display a title "Viel Erfolg beim Zocken!" and three buttons corresponding to the bandits.
- Each button represents a bandit with a payout probability of 20%, 40%, or 80%.
- Users can manually click the buttons to interact with the bandits, and the application will update click counts and payout values accordingly.
- The agent can also perform actions automatically through the provided buttons in the GUI.
- Live plots will visualize cumulative rewards during agent interactions.

## Testing
To ensure the functionality of the project, run the tests located in the `tests` directory:
```
pytest tests/test_bandit.py
```

## Acknowledgments
This project utilizes Tkinter for the GUI and matplotlib for live plotting. Special thanks to the contributors and the open-source community for their support.