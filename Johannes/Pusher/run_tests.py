from __future__ import annotations

import subprocess
import sys


if __name__ == "__main__":
    cmd = [sys.executable, "-m", "pytest", "-q", "--rootdir", ".", "--confcutdir", ".", "tests"]
    raise SystemExit(subprocess.call(cmd))
