import subprocess
import sys


def main() -> int:
    cmd = [sys.executable, "-m", "pytest", "-q", "--rootdir", ".", "--confcutdir", ".", "tests"]
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
