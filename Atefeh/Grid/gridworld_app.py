from __future__ import annotations

try:
    from .gridworld_gui import launch_gui
    from .gridworld_logic import GridMap, GridWorldLab
except ImportError:
    from gridworld_gui import launch_gui
    from gridworld_logic import GridMap, GridWorldLab


def main() -> None:
    grid_map = GridMap(rows=3, cols=5, blocked={(2, 0), (2, 1)}, start=(0, 2), target=(4, 2))
    lab = GridWorldLab(grid_map=grid_map)
    launch_gui(lab)


if __name__ == "__main__":
    main()
