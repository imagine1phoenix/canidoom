"""Stress-test and verify procedural MapGen WADs inside DoomEnv."""

from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "template_reference" / "env"))

from doom_env import make_env  # noqa: E402
from training.mapgen.tiers import MapGen  # noqa: E402


def verify_maps(count: int = 500, steps_per_map: int = 10) -> tuple[int, int]:
    """Generate `count` maps across Tiers 1-4, load each in DoomEnv, and step `steps_per_map` times."""
    passed = 0
    failed = 0
    rng = np.random.default_rng(0)

    with tempfile.TemporaryDirectory(prefix="yugaantar_verify_") as tmp:
        tmp_dir = Path(tmp)
        for i in range(count):
            tier = (i % 4) + 1
            seed = 100_000 + i
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    wad_path = MapGen.sample(tier=tier, seed=seed, out_dir=tmp_dir)
                    env = make_env(wad_path, "MAP01", episode_ticks=200, render_buffers=False)
                    try:
                        env.reset(seed=0)
                        for _ in range(steps_per_map):
                            act = rng.integers(0, 2, size=9, dtype=np.int8)
                            _, _, term, trunc, _ = env.step(act)
                            if term or trunc:
                                break
                    finally:
                        env.close()
                if caught:
                    print(f"[WARN] tier={tier} seed={seed}: {[str(w.message) for w in caught]}")
                    failed += 1
                else:
                    passed += 1
            except Exception as exc:
                print(f"[FAIL] tier={tier} seed={seed}: {exc}")
                failed += 1
            if (i + 1) % 100 == 0:
                print(f"Progress: {i + 1}/{count} (passed={passed}, failed={failed})")
    return passed, failed


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify generated Tier 1-4 maps in DoomEnv.")
    parser.add_argument("--count", type=int, default=500, help="Number of maps to generate and test.")
    parser.add_argument("--steps", type=int, default=10, help="Random steps per map.")
    args = parser.parse_args()

    passed, failed = verify_maps(count=args.count, steps_per_map=args.steps)
    print(f"Verification complete: {passed}/{passed + failed} passed, {failed} failed.")
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
