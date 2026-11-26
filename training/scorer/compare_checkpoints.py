"""Compare candidate checkpoints on the held-out suite and optionally promote the winner.

Usage:
    python -m training.scorer.compare_checkpoints \
      --candidates submission/weights/best_agent.pt runs/staged_main/checkpoints/best.pt \
      --suite training/scorer/eval_suite.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.scorer.score_checkpoint import score_checkpoint


def compare_checkpoints(
    candidates: list[str],
    suite_path: str | None = None,
    device: str = "cpu",
    promote_to: str | None = None,
    out_json: str | None = None,
) -> dict[str, Any]:
    reports: list[dict[str, Any]] = []
    for ckpt in candidates:
        p = Path(ckpt)
        if not p.is_file():
            print(f"[Compare] Skipping missing checkpoint: {ckpt}")
            continue
        summary = score_checkpoint(
            checkpoint_path=str(p),
            eval_suite_path=suite_path,
            device=device,
        )
        reports.append(summary)

    if not reports:
        raise FileNotFoundError("No valid candidate checkpoints found.")

    reports.sort(key=lambda r: float(r["total_score"]), reverse=True)
    winner = reports[0]

    print("\n=== Checkpoint Leaderboard ===")
    print(f"{'Checkpoint':<45} | {'Score':>8} | {'Exit%':>6} | {'Kills':>5} | {'Doors':>5} | {'Deaths':>6} | {'ms':>6}")
    print("-" * 98)
    for r in reports:
        diag = r.get("diagnostics", {})
        print(
            f"{str(r['checkpoint']):<45} | "
            f"{r['total_score']:>8.2f} | "
            f"{100.0 * r['exit_rate']:>5.1f}% | "
            f"{diag.get('total_kills', 0):>5} | "
            f"{diag.get('total_doors_opened', 0):>5} | "
            f"{diag.get('deaths', 0):>6} | "
            f"{r['mean_act_ms']:>6.2f}"
        )

    if promote_to and winner.get("passed_limits", False):
        dest = Path(promote_to)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = Path(winner["checkpoint"])
        if src.resolve() != dest.resolve():
            shutil.copy2(src, dest)
            print(f"\n[Compare] Promoted winner {src} -> {dest} (score={winner['total_score']})")
        else:
            print(f"\n[Compare] Current {dest} is already the winner (score={winner['total_score']}).")

    payload = {"winner": winner["checkpoint"], "leaderboard": reports}
    if out_json:
        out_p = Path(out_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare candidate checkpoints on held-out suite.")
    parser.add_argument("--candidates", nargs="+", required=True, help="Paths to .pt checkpoints")
    parser.add_argument("--suite", type=str, default="training/scorer/eval_suite.json")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--promote-to", type=str, default=None, help="Copy winning checkpoint here if valid")
    parser.add_argument("--out", type=str, default=None, help="Optional JSON leaderboard output")
    args = parser.parse_args()

    compare_checkpoints(
        candidates=args.candidates,
        suite_path=args.suite,
        device=args.device,
        promote_to=args.promote_to,
        out_json=args.out,
    )
