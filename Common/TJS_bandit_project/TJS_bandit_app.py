import tkinter as tk
from TJS_bandit_gui import BanditGUI

def main():
    root = tk.Tk()
    root.title("3-Banditen Spiel")
    app = BanditGUI(root)
    app.pack(fill="both", expand=True)
    root.mainloop()

if __name__ == "__main__":
    main()