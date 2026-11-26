"""Combine and consolidate specialist checkpoints trained across curriculum stages.

Evaluates each available specialist checkpoint on the held-out suite, forms a
softmax-weighted model soup across matching tensor shapes, optionally runs a
short Stage-4 (all-tiers random) consolidation fine-tune so recurrent GRU and
actor-critic heads co-adapt, and promotes to `submission/weights/best_agent.pt`
only when the held-out score exceeds `--promote-if-above`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from training.ppo import _load_cfg_dict, load_checkpoint, train as ppo_train
from training.scorer.score_checkpoint import score_checkpoint


def weight_average_state_dicts(
    state_dicts: Sequence[dict[str, torch.Tensor]],
    scores: Sequence[float],
    temperature: float = 50.0,
) -> dict[str, torch.Tensor]:
    """Compute a softmax score-weighted average over matching tensors across checkpoints."""
    if not state_dicts:
        raise ValueError("state_dicts must be non-empty")
    if len(state_dicts) != len(scores):
        raise ValueError("state_dicts and scores must have the same length")
    if len(state_dicts) == 1:
        return {k: v.clone() for k, v in state_dicts[0].items()}

    sc = np.asarray(scores, dtype=np.float64)
    temp = max(1e-3, float(temperature))
    scaled = (sc - np.max(sc)) / temp
    weights = np.exp(scaled)
    weights = weights / np.sum(weights)

    ref = state_dicts[0]
    merged: dict[str, torch.Tensor] = {}
    for key, ref_t in ref.items():
        compatible = all(
            (key in sd and isinstance(sd[key], torch.Tensor) and sd[key].shape == ref_t.shape)
            for sd in state_dicts
        )
        if not compatible:
            merged[key] = ref_t.clone()
            continue

        if ref_t.is_floating_point():
            acc = torch.zeros_like(ref_t, dtype=torch.float32)
            for sd, w in zip(state_dicts, weights):
                acc += float(w) * sd[key].detach().to(dtype=torch.float32, device=ref_t.device)
            merged[key] = acc.to(dtype=ref_t.dtype)
        else:
            merged[key] = ref_t.clone()
    return merged


def combine_specialists(
    checkpoint_paths: Sequence[str | Path],
    suite_path: str | Path = "training/scorer/eval_suite.json",
    out_path: str | Path = "runs/combined/best_combined.pt",
    config_path: str | Path = "training/configs/ppo_staged.yaml",
    finetune_steps: int = 16384,
    device: str = "cpu",
    promote_if_above: float = 504.13,
    soup_temperature: float = 50.0,
    submission_path: str | Path = "submission/weights/best_agent.pt",
) -> dict[str, Any]:
    existing = [Path(p) for p in checkpoint_paths if Path(p).is_file()]
    if not existing:
        raise FileNotFoundError(f"No existing checkpoints found in: {list(checkpoint_paths)}")

    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    report_dir = out_p.parent

    # 1. Score each existing specialist on the held-out suite
    specialist_results: list[dict[str, Any]] = []
    state_dicts: list[dict[str, torch.Tensor]] = []
    model_cfg: Optional[dict[str, Any]] = None
    max_step = 0

    for p in existing:
        res = score_checkpoint(
            checkpoint_path=p,
            eval_suite_path=suite_path,
            device="cpu",
        )
        ckpt = load_checkpoint(p, map_location="cpu")
        if model_cfg is None:
            model_cfg = ckpt.get("model_config")
        max_step = max(max_step, int(ckpt.get("global_step", 0)))
        state_dicts.append(ckpt["model_state_dict"])
        specialist_results.append(
            {
                "path": str(p),
                "global_step": int(ckpt.get("global_step", 0)),
                "total_score": float(res["total_score"]),
                "exit_rate": float(res["exit_rate"]),
                "mean_act_ms": float(res["mean_act_ms"]),
                "passed_limits": bool(res["passed_limits"]),
            }
        )

    best_single = max(specialist_results, key=lambda r: r["total_score"])
    candidates: list[dict[str, Any]] = [
        {"name": "best_single", "path": best_single["path"], "scorecard": best_single}
    ]

    # 2. Form weighted soup checkpoint
    scores = [r["total_score"] for r in specialist_results]
    soup_sd = weight_average_state_dicts(state_dicts, scores, temperature=soup_temperature)
    soup_path = report_dir / "soup_raw.pt"
    torch.save(
        {
            "model_state_dict": soup_sd,
            "model_config": model_cfg
            or {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True},
            "global_step": max_step,
            "update_idx": 0,
        },
        soup_path,
    )
    soup_eval = score_checkpoint(checkpoint_path=soup_path, eval_suite_path=suite_path, device="cpu")
    candidates.append({"name": "soup_raw", "path": str(soup_path), "scorecard": soup_eval})

    # 3. Optional Stage-4 (all tiers random) consolidation fine-tune
    if finetune_steps > 0:
        ft_dir = report_dir / "finetune_stage4"
        cfg = _load_cfg_dict(config_path)
        cfg.update(
            {
                "curriculum": "staged",
                "stage_lock": 4,
                "stage_start": 4,
                "stage_max": 4,
                "resume_checkpoint": str(soup_path),
                "max_steps": int(finetune_steps),
                "output_dir": str(ft_dir),
                "best_out": "",
                "device": str(device),
                "use_llm_coach": False,
            }
        )
        ft_res = ppo_train(cfg)
        ft_ckpt = ft_res.get("best_checkpoint") or str(ft_dir / "checkpoints" / "latest.pt")
        if Path(ft_ckpt).is_file():
            ft_eval = score_checkpoint(checkpoint_path=ft_ckpt, eval_suite_path=suite_path, device="cpu")
            candidates.append({"name": "soup_finetuned", "path": str(ft_ckpt), "scorecard": ft_eval})

    # 4. Pick overall winner
    winner = max(candidates, key=lambda c: float(c["scorecard"]["total_score"]))
    winner_score = float(winner["scorecard"]["total_score"])
    shutil.copy2(winner["path"], out_p)

    promoted = False
    sub_p = Path(submission_path)
    if winner_score > float(promote_if_above):
        sub_p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out_p, sub_p)
        promoted = True

    report = {
        "specialists": specialist_results,
        "candidates": [
            {"name": c["name"], "path": c["path"], "total_score": float(c["scorecard"]["total_score"])}
            for c in candidates
        ],
        "winner": winner["name"],
        "winner_path": str(out_p),
        "winner_score": winner_score,
        "promote_if_above": float(promote_if_above),
        "promoted": promoted,
        "submission_path": str(sub_p),
    }
    report_path = report_dir / "combine_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[Combiner] Winner={winner['name']} score={winner_score:.2f} promoted={promoted} -> {report_path}")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score, merge, and consolidate specialist DOOM checkpoints.")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Paths to specialist checkpoints (gracefully skips missing paths).",
    )
    parser.add_argument("--suite", type=str, default="training/scorer/eval_suite.json")
    parser.add_argument("--config", type=str, default="training/configs/ppo_staged.yaml")
    parser.add_argument("--finetune-steps", type=int, default=16384)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out", type=str, default="runs/combined/best_combined.pt")
    parser.add_argument("--promote-if-above", type=float, default=504.13)
    parser.add_argument("--soup-temperature", type=float, default=50.0)
    parser.add_argument("--submission-path", type=str, default="submission/weights/best_agent.pt")
    args = parser.parse_args(argv)

    combine_specialists(
        checkpoint_paths=args.checkpoints,
        suite_path=args.suite,
        out_path=args.out,
        config_path=args.config,
        finetune_steps=args.finetune_steps,
        device=args.device,
        promote_if_above=args.promote_if_above,
        soup_temperature=args.soup_temperature,
        submission_path=args.submission_path,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
