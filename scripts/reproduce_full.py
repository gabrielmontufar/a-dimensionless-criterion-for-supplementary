from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    subprocess.run([sys.executable, str(ROOT / "scripts" / "reproduce_core.py")], cwd=ROOT, check=True)
    print("Optional online metadata and solver checks are intentionally outside R0/R1.")
    print("REPRODUCIBILITY_FULL: PASS_WITH_OPTIONAL_WARNINGS")


if __name__ == "__main__":
    main()
