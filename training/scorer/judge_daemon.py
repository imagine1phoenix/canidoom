"""Automated LOQ Checkpoint Judge Daemon (Phase 4).

Continuously monitors runs/incoming/ (or runs/checkpoints/) for new checkpoints
synced from OMEN, evaluates them on the held-out suite (EVALUATION.md), tracks
the team champion, and logs results.

Usage:
    python -m training.scorer.judge_daemon --incoming-dir runs/incoming --suite training/scorer/eval_suite.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from training.scorer.score_checkpoint import score_checkpoint


def run_judge_daemon(
    incoming_dir: Path | str = "runs/incoming",
    eval_suite_path: Path | str = "training/scorer/eval_suite.json",
    poll_interval_s: int = 30,
    dry_run: bool = False,
) -> None:
    inc_dir = Path(incoming_dir)
    inc_dir.mkdir(parents=True, exist_ok=True)
    history_file = inc_dir / "scored_checkpoints.json"

    scored_history = {}
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                scored_history = json.load(f)
        except Exception:
            scored_history = {}

    best_score = max([v.get("total_score", 0.0) for v in scored_history.values()], default=0.0)
    print(f"[JudgeDaemon] Starting watch on {inc_dir.resolve()}. Current best score: {best_score:.2f}")

    if dry_run:
        print("[JudgeDaemon] Dry run mode enabled — verified daemon startup cleanly.")
        return

    try:
        while True:
            # Find all .pt files
            ckpts = sorted(list(inc_dir.glob("*.pt")), key=lambda p: p.stat().st_mtime)

            for ckpt in ckpts:
                ckpt_key = ckpt.name
                if ckpt_key in scored_history:
                    continue

                print(f"\n[JudgeDaemon] >>> New checkpoint detected: {ckpt.name}")
                try:
                    summary = score_checkpoint(
                        checkpoint_path=ckpt,
                        eval_suite_path=eval_suite_path,
                        device="cpu",
                    )
                    scored_history[ckpt_key] = summary

                    # Check for new champion
                    if summary["total_score"] > best_score:
                        old_best = best_score
                        best_score = summary["total_score"]
                        print(
                            f"\n[JudgeDaemon] ★★★ NEW TEAM CHAMPION! ★★★\n"
                            f"  Checkpoint: {ckpt.name}\n"
                            f"  Score: {best_score:.2f} (prev: {old_best:.2f})\n"
                            f"  Mean Latency: {summary['mean_act_ms']:.2f} ms\n"
                            f"  Exit Rate: {summary['exit_rate'] * 100:.1f}%\n"
                        )

                    # Save updated history
                    with open(history_file, "w", encoding="utf-8") as f:
                        json.dump(scored_history, f, indent=2)

                except Exception as e:
                    print(f"[JudgeDaemon] Error scoring {ckpt.name}: {e}")

            time.sleep(poll_interval_s)

    except KeyboardInterrupt:
        print("\n[JudgeDaemon] Stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOQ Checkpoint Judge Daemon.")
    parser.add_argument("--incoming-dir", type=str, default="runs/incoming")
    parser.add_argument("--suite", type=str, default="training/scorer/eval_suite.json")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--dry_run", "--dry-run", action="store_true", help="Perform smoke test and exit")
    args = parser.parse_args()

    run_judge_daemon(args.incoming_dir, args.suite, args.interval, dry_run=args.dry_run)
