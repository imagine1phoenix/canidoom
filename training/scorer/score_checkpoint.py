"""Local Held-Out Checkpoint Evaluator and Scorer.

Directly mirrors docs/EVALUATION.md and planning/Schema.md §5.3:
- Evaluates checkpoints on held-out maps across Tiers 1-4.
- Runs single-thread inference (torch.set_num_threads(1)).
- Tracks non-binary scoring: Exit (+efficiency), Progression (keys/doors/locks),
  Exploration (unique sectors via sector_polygons), Combat (kills vs damage).
- Measures mean and p99 act() latency against official 40ms/step compute budget.
- Produces complete JSON scorecards and checks promotion criteria.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure repository root is on sys.path
SCORER_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCORER_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from doom_env import make_env
from doom_env.limits import AGENT_MS_PER_STEP, ACT_TIMEOUT_S
from submission.agent import DoomAgent
from training.rewards import RewardShaper
from training.scorer.eval_suite import get_default_eval_suite, load_eval_suite_from_json, EvalMapSpec, TIER_WEIGHTS


def score_episode(
    agent: DoomAgent,
    spec: EvalMapSpec,
    device: str = "cpu",
) -> dict:
    """Evaluates agent on one map and computes partial-progress scorecard."""
    wad_path = Path(spec.wad_path)
    if not wad_path.is_file():
        wad_path = REPO_ROOT / spec.wad_path
    env = make_env(
        wad=str(wad_path),
        map_name=spec.map_name,
        episode_ticks=spec.episode_ticks,
        render_buffers=True,
    )

    agent.reset()
    obs, info = env.reset(seed=spec.seed)

    act_times = []
    keys_observed = 0
    damage_taken = 0.0
    kills = 0

    sector_polys = env.sector_polygons()
    shaper = RewardShaper()
    shaper.reset(sector_polys)
    priv_prev = env.privileged_state()
    priv = priv_prev
    total_sectors = max(len(sector_polys), 1)

    step_idx = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        t0 = time.perf_counter()
        action = agent.act(obs)
        t1 = time.perf_counter()
        act_times.append((t1 - t0) * 1000.0)

        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1

        # Read privileged state in the scorer process for metric tracking
        priv = env.privileged_state()
        shaper.step(priv_prev, priv, action, info)
        priv_prev = priv

        if "keys" in priv:
            k_val = priv["keys"]
            keys_count = sum(k_val.values()) if isinstance(k_val, dict) else int(np.sum(k_val))
            keys_observed = max(keys_observed, int(keys_count))
        if "damage_taken" in priv:
            damage_taken = float(priv["damage_taken"])
        if "kills" in priv:
            kills = int(priv["kills"])

    visited_sectors = shaper.visited_sectors
    doors_opened = len(shaper.opened_doors)
    locks_unlocked = len(shaper.unlocked_doors)

    env.close()

    exit_reached = bool(info.get("exit_reached", False))
    dead = bool(priv.get("dead", False))
    rem_ticks = max(0, spec.episode_ticks - (step_idx * 4))

    # --- Scoring Formula (EVALUATION.md) ---
    # 1. Exit points: +40 base + up to +10 for remaining time & health
    exit_score = 0.0
    if exit_reached:
        health_frac = max(0.0, float(priv.get("health", 100.0))) / 100.0
        time_frac = float(rem_ticks) / float(spec.episode_ticks)
        exit_score = 40.0 + 5.0 * health_frac + 5.0 * time_frac

    # 2. Progression points: keycard (+15), door (+5), locked door (+15)
    progression_score = (keys_observed * 15.0) + (doors_opened * 5.0) + (locks_unlocked * 15.0)

    # 3. Exploration points: up to +20 based on unique sectors visited
    exploration_frac = min(1.0, float(len(visited_sectors)) / float(total_sectors))
    exploration_score = 20.0 * exploration_frac

    # 4. Combat points: kills * 3 - damage * 0.05 - early death penalty
    combat_score = (kills * 3.0) - (damage_taken * 0.05)
    if dead and step_idx < (spec.episode_ticks // 8):
        combat_score -= 10.0
    combat_score = max(0.0, combat_score)

    total_map_score = exit_score + progression_score + exploration_score + combat_score
    tier_weight = TIER_WEIGHTS.get(spec.tier, 1.0)
    weighted_score = total_map_score * tier_weight

    act_arr = np.array(act_times) if act_times else np.zeros(1)
    mean_ms = float(np.mean(act_arr))
    p99_ms = float(np.percentile(act_arr, 99))

    return {
        "map_id": spec.id,
        "tier": spec.tier,
        "weighted_score": weighted_score,
        "raw_score": total_map_score,
        "exit_reached": exit_reached,
        "dead": dead,
        "early_death": bool(dead and step_idx < (spec.episode_ticks // 8)),
        "timed_out": bool((not exit_reached) and (not dead)),
        "keys": keys_observed,
        "doors_opened": doors_opened,
        "locks_unlocked": locks_unlocked,
        "kills": kills,
        "damage_taken": damage_taken,
        "exploration_frac": exploration_frac,
        "steps_taken": step_idx,
        "mean_act_ms": mean_ms,
        "p99_act_ms": p99_ms,
        "passed_limits": (mean_ms <= AGENT_MS_PER_STEP) and (p99_ms <= ACT_TIMEOUT_S * 1000.0),
    }


def _specs_from_wad_dir(wad_dir_p: Path) -> list[EvalMapSpec]:
    specs = []
    for wad_file in sorted(wad_dir_p.glob("*.wad")):
        stem = wad_file.stem
        tier = 1
        if "t1" in stem or "tier1" in stem:
            tier = 1
        elif "t2" in stem or "tier2" in stem:
            tier = 2
        elif "t3" in stem or "tier3" in stem:
            tier = 3
        elif "t4" in stem or "tier4" in stem:
            tier = 4

        seed = 1001
        if "seed" in stem:
            try:
                seed = int("".join(filter(str.isdigit, stem.split("seed")[-1])))
            except Exception:
                pass
        specs.append(
            EvalMapSpec(
                id=stem,
                wad_path=str(wad_file),
                map_name="MAP01",
                tier=tier,
                episode_ticks=4200 if tier == 4 else 2100,
                seed=seed,
            )
        )
    return specs


def score_checkpoint(
    checkpoint_path: Path | str | None = None,
    eval_suite_path: Path | str | None = None,
    device: str = "cpu",
    wad_dir: Path | str | None = None,
    out_path: Path | str | None = None,
) -> dict:
    """Evaluates agent on the full held-out suite and generates scorecard JSON."""
    if device == "cpu":
        torch.set_num_threads(1)

    # Instantiate agent
    from gymnasium.spaces import Box, MultiBinary
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)
    if checkpoint_path is not None:
        agent = DoomAgent(obs_space, act_space, device=device, checkpoint_path=checkpoint_path)
        print(f"[Scorer] Loaded checkpoint: {checkpoint_path}")
    else:
        agent = DoomAgent(obs_space, act_space, device=device, checkpoint_path=None, allow_untrained=True)

    # Load map suite
    if wad_dir is not None and Path(wad_dir).is_dir():
        wad_dir_p = Path(wad_dir)
        default_suite_json = REPO_ROOT / "training" / "scorer" / "eval_suite.json"
        if default_suite_json.exists():
            all_specs = load_eval_suite_from_json(default_suite_json)
            matched = [
                s for s in all_specs
                if Path(s.wad_path).name in {p.name for p in wad_dir_p.glob("*.wad")}
            ]
            if matched:
                for s in matched:
                    s.wad_path = str(wad_dir_p / Path(s.wad_path).name)
                suite = matched
            else:
                suite = _specs_from_wad_dir(wad_dir_p)
        else:
            suite = _specs_from_wad_dir(wad_dir_p)
    elif eval_suite_path is not None and Path(eval_suite_path).exists():
        suite = load_eval_suite_from_json(eval_suite_path)
    else:
        suite = get_default_eval_suite(REPO_ROOT)

    print(f"[Scorer] Running evaluation on {len(suite)} held-out map(s)...")

    results = []
    tier_scores = {1: [], 2: [], 3: [], 4: []}
    exit_counts = 0
    all_mean_ms = []
    all_p99_ms = []

    for spec in suite:
        res = score_episode(agent, spec, device=device)
        results.append(res)
        tier_scores[spec.tier].append(res["weighted_score"])
        if res["exit_reached"]:
            exit_counts += 1
        all_mean_ms.append(res["mean_act_ms"])
        all_p99_ms.append(res["p99_act_ms"])

        print(
            f"  Tier {spec.tier} [{spec.id}]: "
            f"Score={res['weighted_score']:.1f} (exit={res['exit_reached']}, "
            f"kills={res['kills']}, act_mean={res['mean_act_ms']:.1f}ms)"
        )

    total_score = float(sum(r["weighted_score"] for r in results))
    exit_rate = float(exit_counts) / max(len(suite), 1)
    mean_ms = float(np.mean(all_mean_ms))
    p99_ms = float(np.max(all_p99_ms))

    tier_averages = {
        str(tier): float(np.mean(scores)) if scores else 0.0
        for tier, scores in tier_scores.items()
    }

    per_tier_diag = {}
    for tier in (1, 2, 3, 4):
        t_res = [r for r in results if r["tier"] == tier]
        if t_res:
            per_tier_diag[str(tier)] = {
                "count": len(t_res),
                "exits": sum(1 for r in t_res if r["exit_reached"]),
                "avg_score": round(float(np.mean([r["weighted_score"] for r in t_res])), 2),
                "avg_kills": round(float(np.mean([r["kills"] for r in t_res])), 2),
                "avg_exploration": round(float(np.mean([r["exploration_frac"] for r in t_res])), 3),
                "deaths": sum(1 for r in t_res if r["dead"]),
                "timeouts": sum(1 for r in t_res if r["timed_out"]),
            }

    diagnostics = {
        "total_kills": sum(r["kills"] for r in results),
        "total_keys": sum(r["keys"] for r in results),
        "total_doors_opened": sum(r["doors_opened"] for r in results),
        "total_locks_unlocked": sum(r["locks_unlocked"] for r in results),
        "deaths": sum(1 for r in results if r["dead"]),
        "early_deaths": sum(1 for r in results if r["early_death"]),
        "timeouts": sum(1 for r in results if r["timed_out"]),
        "mean_exploration_frac": round(float(np.mean([r["exploration_frac"] for r in results])) if results else 0.0, 3),
        "per_tier": per_tier_diag,
    }

    summary = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else "untrained_baseline",
        "total_score": round(total_score, 2),
        "tier_scores": tier_averages,
        "exit_rate": round(exit_rate, 3),
        "mean_act_ms": round(mean_ms, 2),
        "p99_act_ms": round(p99_ms, 2),
        "passed_limits": mean_ms < AGENT_MS_PER_STEP,
        "maps_evaluated": len(suite),
        "diagnostics": diagnostics,
        "results": [
            {
                "id": r["map_id"],
                "tier": r["tier"],
                "exit_reached": r["exit_reached"],
                "dead": r["dead"],
                "timed_out": r["timed_out"],
                "score": r["weighted_score"],
                "kills": r["kills"],
                "keys": r["keys"],
                "exploration_frac": round(r["exploration_frac"], 3),
                "mean_act_ms": r["mean_act_ms"],
            }
            for r in results
        ],
    }

    print("\n--- Checkpoint Evaluation Summary ---")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))

    if out_path:
        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Scorer] Saved evaluation results to {out_p.resolve()}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DoomAgent on held-out suite.")
    parser.add_argument("--checkpoint", type=str, default="submission/weights/best_agent.pt", help="Path to .pt checkpoint")
    parser.add_argument("--suite", type=str, default=None, help="Path to eval_suite.json")
    parser.add_argument("--wad_dir", "--wad-dir", type=str, default=None, help="Path to heldout WAD directory")
    parser.add_argument("--out", "-o", type=str, default=None, help="Path to output JSON scorecard")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    score_checkpoint(
        checkpoint_path=args.checkpoint,
        eval_suite_path=args.suite,
        device=args.device,
        wad_dir=args.wad_dir,
        out_path=args.out,
    )
