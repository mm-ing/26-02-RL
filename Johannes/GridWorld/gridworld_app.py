try:
    from .gridworld_logic import Grid
    from .gridworld_gui import GridWorldGUI
except Exception:
    from gridworld_logic import Grid
    from gridworld_gui import GridWorldGUI


def main():
    grid = Grid()
    app = GridWorldGUI(grid)
    app.mainloop()


if __name__ == '__main__':
    main()
