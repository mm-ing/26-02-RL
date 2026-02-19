# Design Document for TJS Bandit Project

## Overview
The TJS Bandit project is a simulation of a three-armed bandit game that incorporates both manual user interaction and automated agent strategies using reinforcement learning. The project is designed to provide an engaging user experience while demonstrating key concepts in probability, statistics, and machine learning.

## Architecture
The project is structured into several key components, each responsible for specific functionalities:

1. **TJS_bandit_13.py**: 
   - Serves as the standalone GUI for the three-armed bandit game.
   - Initializes the Tkinter application and sets up the user interface.
   - Manages user interactions, displaying click counts and payout values for each bandit.

2. **TJS_bandit_app.py**: 
   - Acts as the entry point for the application.
   - Initializes the environment and GUI components.
   - Starts the main Tkinter loop to handle user input and display updates.

3. **TJS_bandit_gui.py**: 
   - Contains the GUI class that extends the functionality of the bandit game.
   - Implements the layout for both manual and agent-controlled sections.
   - Integrates live plotting using matplotlib to visualize cumulative rewards over iterations.

4. **TJS_bandit_logic.py**: 
   - Defines the core logic of the bandit game.
   - **Class Bandit**: Represents a single bandit, encapsulating properties for payout probability and click counts, along with methods for pulling the lever.
   - **Class Environment**: Manages the collection of bandits and handles the overall game state.
   - **Class Agent**: Implements the logic for the agent to interact with the bandits, including action selection and reward tracking.
   - **Class Policy**: Defines the policy interface and implements strategies such as Epsilon-Greedy and Thompson Sampling.

## Design Decisions
- **User Interface**: The GUI is designed to be responsive, allowing for real-time updates of click counts and payout values. The layout is divided into two sections: one for manual interaction and another for agent control.
- **Payout Logic**: The game employs Bernoulli logic for determining payouts based on the assigned probabilities for each bandit. This simulates the randomness inherent in real slot machines.
- **Reinforcement Learning**: The agent uses Epsilon-Greedy and Thompson Sampling strategies to explore and exploit the bandits, aiming to maximize the total reward over a specified number of iterations (100).
- **Live Plotting**: The integration of matplotlib allows for dynamic visualization of the agent's performance, providing insights into the effectiveness of different policies over time.

## Future Enhancements
- Additional policies could be implemented to further explore the effectiveness of different strategies.
- User customization options for starting coins and payout ranges could enhance the gameplay experience.
- A more detailed analytics dashboard could be added to provide users with insights into their gameplay and the agent's performance.

## Conclusion
The TJS Bandit project serves as an educational tool for understanding the mechanics of probability, statistics, and reinforcement learning. Through its interactive GUI and underlying logic, it provides a comprehensive simulation of a three-armed bandit game, making it an engaging experience for users and a valuable resource for learning.