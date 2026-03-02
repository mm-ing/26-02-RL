from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--rootdir",
        ".",
        "--confcutdir",
        ".",
        "tests/test_BipedalWalker_logic.py",
        "tests/test_BipedalWalker_gui.py",
    ]

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=project_dir)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
