from __future__ import annotations

import tkinter as tk

from Taxi_gui import TaxiGUI
from Taxi_logic import TaxiEnvironment, Trainer


def main() -> None:
    env = TaxiEnvironment(render_mode="rgb_array")
    trainer = Trainer(env, results_dir="results_csv")

    root = tk.Tk()
    app = TaxiGUI(root, env, trainer)

    def on_close() -> None:
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
