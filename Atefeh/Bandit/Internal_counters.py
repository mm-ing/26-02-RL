"""Launcher for the Manual MAB demo.

This file previously contained partial UI code which referenced
variables (like `probabilities`) that weren't defined here. To avoid
duplication and NameError issues, this launcher imports and runs the
complete implementation in `manual_mab.py`.

Run this file directly or run `manual_mab.py`:

    python3 Internal_counters.py
    python3 manual_mab.py
"""

import sys

def main():
    try:
        # manual_mab.py is in the same directory
        from manual_mab import main as mab_main
    except Exception as exc:
        print("Error: failed to import manual_mab.py:", exc)
        print("Make sure manual_mab.py exists in the same folder and is error-free.")
        sys.exit(1)

    # Delegate to the actual main function
    mab_main()


if __name__ == "__main__":
    main()