import tkinter as tk
from TJS_bandit_logic import Environment
from TJS_bandit_gui import GUI

def main():
    # Initialize the environment with starting coins for each bandit
    env = Environment(starts=(20, 40, 80))
    
    # Create the main application window
    root = tk.Tk()
    root.title("3-Armed Bandit Game")
    
    # Initialize the GUI with the environment
    app = GUI(root, env)
    
    # Start the Tkinter main loop
    root.mainloop()

if __name__ == "__main__":
    main()