"""Staged (easy -> hard) curriculum for the DOOM PPO trainer.

Replaces the hard-biased PLR sampling (tier_weights [1, 1.5, 2.5, 3.0] -> ~75% Tier 3/4 maps)
with an explicit ladder the agent has to climb:

    0  single_map   ONE fixed map, no enemies (learn: find the exit)
    1  easy         Tier 1, random seeds, a few weak enemies
    2  medium       Tier 2, random seeds, doors
    3  hard         Tier 3 (keycards) + Tier 4 (multi-lock) mix
    4  random       every tier mixed (final generalisation stage)

A stage is "mastered" when the rolling exit-rate over the last `window` episodes reaches its
threshold (after `min_episodes`), or when `max_steps_per_stage` is spent (so it never stalls).
`stage_lock` pins a run to one stage; `stage_max` caps the highest stage index reachable.
Stages >= 1 also apply a balanced within-stage replay buffer (softened temperature=0.7,
staleness_coef=0.35) so hard levels do not collapse replay onto a single seed.

Training seeds always come from [100_000, 999_999] (disjoint from held-out [1_000, 9_999]).
Uses no network / no LLM. It is a plain feedback rule on the training GPU run's own telemetry.
"""

from __future__ import annotations

import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

TRAIN_SEED_MIN = 100_000
TRAIN_SEED_MAX = 999_999
FIXED_MAP_SEED = 123_457  # the single map of stage 0 (always the same layout)


@dataclass
class StageSpec:
    name: str
    tiers: Sequence[int]
    tier_probs: Sequence[float]
    knobs: dict[str, Any]
    exit_threshold: float
    scaffolding_alpha: float
    fixed_seed: Optional[int] = None


DEFAULT_STAGES: list[StageSpec] = [
    StageSpec(
        name="single_map",
        tiers=[1],
        tier_probs=[1.0],
        knobs={"scenario": "open_arena", "num_rooms": 1, "enemies_per_room": 0.0, "ammo_mult": 1.5},
        exit_threshold=0.80,
        scaffolding_alpha=1.0,
        fixed_seed=FIXED_MAP_SEED,
    ),
    StageSpec(
        name="easy",
        tiers=[1],
        tier_probs=[1.0],
        knobs={"num_rooms": 2, "enemies_per_room": 1.0, "ammo_mult": 1.3},
        exit_threshold=0.60,
        scaffolding_alpha=0.8,
    ),
    StageSpec(
        name="medium",
        tiers=[1, 2],
        tier_probs=[0.25, 0.75],
        knobs={"enemies_per_room": 1.5, "ammo_mult": 1.1},
        exit_threshold=0.45,
        scaffolding_alpha=0.5,
    ),
    StageSpec(
        name="hard",
        tiers=[2, 3, 4],
        tier_probs=[0.2, 0.5, 0.3],
        knobs={},
        exit_threshold=0.30,
        scaffolding_alpha=0.25,
    ),
    StageSpec(
        name="random",
        tiers=[1, 2, 3, 4],
        tier_probs=[0.25, 0.25, 0.25, 0.25],
        knobs={},
        exit_threshold=1.01,  # final stage never "completes"
        scaffolding_alpha=0.0,
    ),
]


@dataclass
class StageCurriculum:
    stages: list[StageSpec] = field(default_factory=lambda: list(DEFAULT_STAGES))
    stage_idx: int = 0
    stage_lock: Optional[int] = None
    stage_max: Optional[int] = None
    window: int = 64
    min_episodes: int = 64
    min_steps_per_stage: int = 100_000
    max_steps_per_stage: int = 1_500_000
    replay_prob: float = 0.4
    temperature: float = 0.7
    staleness_coef: float = 0.35
    buffer_capacity: int = 128
    seed: int = 7
    state_path: Optional[Path] = None
    stage_start_step: Optional[int] = None

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self._exits: deque[float] = deque(maxlen=self.window)
        self._episodes_in_stage = 0
        self._stage_start_initialized = self.stage_start_step is not None
        self._stage_start_step = 0 if self.stage_start_step is None else int(self.stage_start_step)
        self._replay_tick = 0
        self._stage_buffer: dict[str, dict[str, Any]] = {}
        if self.stage_lock is not None:
            self.stage_idx = int(self.stage_lock)
        cap = len(self.stages) - 1 if self.stage_max is None else min(len(self.stages) - 1, int(self.stage_max))
        self.stage_idx = max(0, min(max(0, cap), int(self.stage_idx)))
        self._used: set[int] = set()

    # ------------------------------------------------------------------ queries
    @property
    def spec(self) -> StageSpec:
        return self.stages[self.stage_idx]

    @property
    def name(self) -> str:
        return f"{self.stage_idx}:{self.spec.name}"

    def exit_rate(self) -> float:
        return float(np.mean(self._exits)) if self._exits else 0.0

    # ------------------------------------------------------------------ sampling
    def _fresh_seed(self) -> int:
        for _ in range(64):
            s = int(self.rng.integers(TRAIN_SEED_MIN, TRAIN_SEED_MAX + 1))
            if s not in self._used:
                self._used.add(s)
                return s
        return int(self.rng.integers(TRAIN_SEED_MIN, TRAIN_SEED_MAX + 1))

    def _sample_from_buffer(self, sp: StageSpec) -> Optional[dict[str, Any]]:
        entries = list(self._stage_buffer.values())
        n = len(entries)
        if n < 8:
            return None
        scores = np.array([max(1e-6, float(e["score"])) for e in entries], dtype=np.float64)
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        temp = max(1e-3, float(self.temperature))
        p_score = (1.0 / ranks) ** (1.0 / temp)
        p_score /= p_score.sum()

        stale = np.array([max(0.0, float(self._replay_tick - int(e["last_seen"]))) for e in entries], dtype=np.float64)
        p_stale = stale / stale.sum() if stale.sum() > 0 else np.full(n, 1.0 / n, dtype=np.float64)

        sc = max(0.0, min(1.0, float(self.staleness_coef)))
        probs = (1.0 - sc) * p_score + sc * p_stale
        probs /= probs.sum()
        idx = int(self.rng.choice(n, p=probs))
        chosen = entries[idx]
        chosen["last_seen"] = self._replay_tick
        return {
            "level_id": chosen["level_id"],
            "tier": int(chosen["tier"]),
            "knobs": dict(sp.knobs),
            "seed": int(chosen["seed"]),
            "scaffolding_alpha": float(sp.scaffolding_alpha),
        }

    def sample(self) -> dict[str, Any]:
        """Return a level dict compatible with `PLRBuffer.sample_level` consumers in ppo.py."""
        self._replay_tick += 1
        sp = self.spec
        if sp.fixed_seed is None and len(self._stage_buffer) >= 8 and self.rng.random() < float(self.replay_prob):
            replayed = self._sample_from_buffer(sp)
            if replayed is not None:
                return replayed

        p = np.asarray(sp.tier_probs, dtype=np.float64)
        tier = int(self.rng.choice(list(sp.tiers), p=p / p.sum()))
        seed = sp.fixed_seed if sp.fixed_seed is not None else self._fresh_seed()
        level_id = f"stage{self.stage_idx}_t{tier}_s{seed}"
        if sp.fixed_seed is None:
            if len(self._stage_buffer) >= max(8, int(self.buffer_capacity)):
                evict_id = min(self._stage_buffer, key=lambda k: float(self._stage_buffer[k]["score"]))
                self._stage_buffer.pop(evict_id, None)
            self._stage_buffer[level_id] = {
                "level_id": level_id,
                "tier": int(tier),
                "seed": int(seed),
                "score": 1.0,
                "last_seen": self._replay_tick,
            }
        return {
            "level_id": level_id,
            "tier": tier,
            "knobs": dict(sp.knobs),
            "seed": int(seed),
            "scaffolding_alpha": float(sp.scaffolding_alpha),
        }

    # ------------------------------------------------------------------ feedback
    def report_episode(
        self,
        exit_reached: bool,
        level_id: Optional[str] = None,
        score: float = 1.0,
    ) -> None:
        self._exits.append(1.0 if exit_reached else 0.0)
        self._episodes_in_stage += 1
        if level_id and level_id in self._stage_buffer:
            s_val = float(score) if not math.isnan(float(score)) else 1.0
            self._stage_buffer[level_id]["score"] = max(0.0, s_val)
            self._stage_buffer[level_id]["last_seen"] = self._replay_tick

    def maybe_advance(self, global_step: int) -> Optional[str]:
        """Advance to the next stage when mastered (or out of budget). Returns a log line if it moved."""
        cap = (
            len(self.stages) - 1
            if self.stage_max is None
            else min(len(self.stages) - 1, int(self.stage_max))
        )
        if self.stage_lock is not None or self.stage_idx >= cap:
            return None
        if not self._stage_start_initialized and self._episodes_in_stage == 0:
            self._stage_start_step = global_step
            self._stage_start_initialized = True
        spent = global_step - self._stage_start_step
        mastered = (
            self._episodes_in_stage >= self.min_episodes
            and spent >= self.min_steps_per_stage
            and self.exit_rate() >= self.spec.exit_threshold
        )
        out_of_budget = spent >= self.max_steps_per_stage
        if not (mastered or out_of_budget):
            return None
        old = self.name
        old_rate = self.exit_rate()
        self.stage_idx += 1
        self._exits.clear()
        self._stage_buffer.clear()
        self._episodes_in_stage = 0
        self._stage_start_step = global_step
        self._stage_start_initialized = True
        why = "MASTERED" if mastered else "step budget spent"
        msg = f"[Stage] {old} -> {self.name} ({why}; exit={old_rate * 100:.0f}% @ step {global_step:,})"
        self.save()
        return msg

    # ------------------------------------------------------------------ persistence
    def save(self) -> None:
        if self.state_path is None:
            return
        sp = self.spec
        payload = {
            "stage_idx": int(self.stage_idx),
            "stage": str(sp.name),
            "tiers": [int(t) for t in sp.tiers],
            "knobs": dict(sp.knobs),
            "fixed_seed": None if sp.fixed_seed is None else int(sp.fixed_seed),
            "scaffolding_alpha": float(sp.scaffolding_alpha),
            "exit_rate": round(self.exit_rate(), 4),
        }
        try:
            self.state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def state_dict(self) -> dict[str, Any]:
        return {
            "stage_idx": int(self.stage_idx),
            "stage_lock": self.stage_lock,
            "stage_max": self.stage_max,
            "exits": list(self._exits),
            "episodes_in_stage": int(self._episodes_in_stage),
            "stage_start_step": int(self._stage_start_step),
            "replay_tick": int(self._replay_tick),
            "stage_buffer": {k: dict(v) for k, v in self._stage_buffer.items()},
        }
