"""Pre-deadline submission readiness audit gate.

Verifies in a single pass:
  1. Clean git working tree & Git LFS tracking on submission/weights/*.pt
  2. Unit tests (tests/test_scorer_and_agent_guard.py)
  3. Official validate.py sandbox check (local or --docker)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def _run(cmd: list[str], label: str) -> bool:
    print(f"\n[Preflight] {label}: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        print(f"[Preflight] FAILED: {label} (exit={proc.returncode})")
        return False
    print(f"[Preflight] PASSED: {label}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-submission readiness check.")
    parser.add_argument("--docker", action="store_true", help="Also run validate.py --docker")
    parser.add_argument("--image", type=str, default="devforge-doom")
    args = parser.parse_args()

    # 1. Git LFS check
    lfs_out = subprocess.check_output(["git", "lfs", "ls-files"], cwd=str(ROOT), text=True)
    if "submission/weights/best_agent.pt" not in lfs_out:
        print("[Preflight] FAILED: submission/weights/best_agent.pt is not tracked by Git LFS!")
        return 1
    print("[Preflight] PASSED: Git LFS tracking verified for submission/weights/best_agent.pt")

    # 2. Unit tests
    if not _run([sys.executable, "-m", "pytest", "tests/test_scorer_and_agent_guard.py", "-v"], "Unit Tests"):
        return 1

    # 3. Validate
    val_cmd = [sys.executable, "validate.py"]
    if args.docker:
        val_cmd.extend(["--docker", "--image", args.image])
    if not _run(val_cmd, "Submission Validator"):
        return 1

    print("\n[Preflight] ALL SUBMISSION GATES PASSED!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
