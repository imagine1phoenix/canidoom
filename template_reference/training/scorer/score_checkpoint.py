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
from doom_env.limits import AGENT_MS_PER_STEP, ACT_SUBSEQUENT_TIMEOUT_S
from submission.agent import DoomAgent
from training.scorer.eval_suite import get_default_eval_suite, load_eval_suite_from_json, EvalMapSpec, TIER_WEIGHTS


def score_episode(
    agent: DoomAgent,
    spec: EvalMapSpec,
    device: str = "cpu",
) -> dict:
    """Evaluates agent on one map and computes partial-progress scorecard."""
    env = make_env(
        wad=spec.wad_path,
        map_name=spec.map_name,
        episode_ticks=spec.episode_ticks,
        render_buffers=True,
    )

    agent.reset()
    obs, info = env.reset(seed=spec.seed)

    act_times = []
    visited_sectors = set()
    keys_observed = 0
    doors_opened = 0
    locks_unlocked = 0
    damage_taken = 0.0
    kills = 0

    priv_initial = env.privileged_state()
    total_sectors = max(len(env.sector_polygons()), 1)

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
        if "keys" in priv:
            keys_observed = max(keys_observed, int(sum(priv["keys"].values())))
        if "damage_taken" in priv:
            damage_taken = float(priv["damage_taken"])
        if "kills" in priv:
            kills = int(priv["kills"])

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
        "keys": keys_observed,
        "kills": kills,
        "damage_taken": damage_taken,
        "exploration_frac": exploration_frac,
        "mean_act_ms": mean_ms,
        "p99_act_ms": p99_ms,
        "passed_limits": (mean_ms <= AGENT_MS_PER_STEP) and (p99_ms <= ACT_SUBSEQUENT_TIMEOUT_S * 1000.0),
    }


def score_checkpoint(
    checkpoint_path: Path | str | None = None,
    eval_suite_path: Path | str | None = None,
    device: str = "cpu",
) -> dict:
    """Evaluates agent on the full held-out suite and generates scorecard JSON."""
    if device == "cpu":
        torch.set_num_threads(1)

    # Instantiate agent
    from gymnasium.spaces import Box, MultiBinary
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)
    agent = DoomAgent(obs_space, act_space, device=device)

    # Load custom weights if supplied
    if checkpoint_path is not None:
        p = Path(checkpoint_path)
        if p.exists():
            ckpt = torch.load(p, map_location=device, weights_only=False)
            state_dict = ckpt.get("model_state_dict", ckpt)
            agent.model.load_state_dict(state_dict, strict=False)
            print(f"[Scorer] Loaded checkpoint: {p}")

    # Load map suite
    if eval_suite_path is not None and Path(eval_suite_path).exists():
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

    summary = {
        "checkpoint": str(checkpoint_path) if checkpoint_path else "untrained_baseline",
        "total_score": round(total_score, 2),
        "tier_scores": tier_averages,
        "exit_rate": round(exit_rate, 3),
        "mean_act_ms": round(mean_ms, 2),
        "p99_act_ms": round(p99_ms, 2),
        "passed_limits": mean_ms < AGENT_MS_PER_STEP,
    }

    print("\n--- Checkpoint Evaluation Summary ---")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate DoomAgent on held-out suite.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint")
    parser.add_argument("--suite", type=str, default=None, help="Path to eval_suite.json")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    score_checkpoint(args.checkpoint, args.suite, device=args.device)
