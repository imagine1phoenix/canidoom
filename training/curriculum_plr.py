"""Self-evolving Prioritized Level Replay (PLR) curriculum and scaffolding scheduler.

Maintains a bounded buffer (capacity=256) of procedural levels weighted by |GAE| score
and staleness, enforcing strict disjointness between training seeds (>= 100_000)
and held-out evaluation seeds ([1_000, 9_999]). Matches planning/Schema.md §5.2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator, Optional, Sequence

import numpy as np

from .mapgen.tiers import ALL_SCENARIOS, TIER_SCENARIOS

TRAIN_SEED_MIN = 100_000
TRAIN_SEED_MAX = 999_999


def _sanitize_score(val: Any) -> float:
    try:
        f = float(val)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    if math.isinf(f):
        return 10.0 if f > 0.0 else 0.0
    return max(0.0, f)


@dataclass
class PLREntry:
    """PLR level descriptor supporting dict access, attribute access, and `(entry, alpha)` unpacking."""

    level_id: str
    tier: int
    knobs: dict[str, Any]
    seed: int
    score_gae: float = 1.0
    last_seen_update: int = 0
    visits: int = 1
    scaffolding_alpha: float = 0.0

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and hasattr(self, key)

    def __iter__(self) -> Iterator[Any]:
        # Enables `entry, alpha = plr.sample_level(...)` while `lvl["seed"]` also works directly
        yield self
        yield float(self.scaffolding_alpha)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level_id": self.level_id,
            "tier": self.tier,
            "knobs": dict(self.knobs),
            "seed": self.seed,
            "score_gae": self.score_gae,
            "last_seen_update": self.last_seen_update,
            "visits": self.visits,
            "scaffolding_alpha": self.scaffolding_alpha,
        }


class PLRBuffer:
    """Self-evolving Prioritized Level Replay buffer across extreme procedural Tiers 1-4."""

    def __init__(
        self,
        capacity: int = 256,
        tier_weights: Sequence[float] = (1.0, 1.5, 2.5, 3.0),
        seed_min: int = TRAIN_SEED_MIN,
        seed_max: int = TRAIN_SEED_MAX,
        replay_prob: float = 0.5,
        staleness_coef: float = 0.1,
        temperature: float = 0.1,
        scaffolding_end_step: int = 12_000_000,
        seed: int = 42,
    ) -> None:
        if seed_min < TRAIN_SEED_MIN:
            raise ValueError(
                f"Training seed_min must be >= {TRAIN_SEED_MIN} to stay disjoint from held-out suite, got {seed_min}"
            )
        self.capacity = max(1, int(capacity))
        self.tier_weights = [float(w) for w in tier_weights]
        if len(self.tier_weights) != 4:
            raise ValueError("tier_weights must have 4 entries for Tiers 1..4")
        self.seed_min = int(seed_min)
        self.seed_max = int(seed_max)
        self.replay_prob = float(replay_prob)
        self.staleness_coef = float(staleness_coef)
        self.temperature = max(1e-3, float(temperature))
        self.scaffolding_end_step = max(1, int(scaffolding_end_step))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.entries: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._by_seed: dict[int, dict[str, Any]] = {}
        self._used_seeds: set[int] = set()
        self.update_idx: int = 0

        self.global_knobs: dict[str, Any] = {
            "enemies_per_room_mult": 1.0,
            "ammo_mult": 1.0,
            "num_rooms_offset": 0.0,
            "num_rooms_override": None,
            "enemies_per_room_override": None,
        }

    def __len__(self) -> int:
        return len(self.entries)

    def scaffolding_alpha(self, global_step: int, scaffolding_end_step: Optional[int] = None) -> float:
        """Linearly decay training scaffolding 1.0 -> 0.0 over `scaffolding_end_step` steps."""
        end = max(1, int(scaffolding_end_step or self.scaffolding_end_step))
        frac = float(global_step) / float(end)
        return float(max(0.0, min(1.0, 1.0 - frac)))

    def _new_seed(self) -> int:
        for _ in range(64):
            s = int(self.rng.integers(self.seed_min, self.seed_max + 1))
            if s not in self._used_seeds:
                self._used_seeds.add(s)
                return s
        s = int(self.rng.integers(self.seed_min, self.seed_max + 1))
        self._used_seeds.add(s)
        return s

    def _sample_tier(self) -> int:
        w = np.maximum(np.asarray(self.tier_weights, dtype=np.float64), 1e-6)
        probs = w / w.sum()
        return int(self.rng.choice([1, 2, 3, 4], p=probs))

    def _default_knobs(self, tier: int) -> dict[str, Any]:
        scenario = str(self.rng.choice(TIER_SCENARIOS[tier]))
        if self.global_knobs.get("num_rooms_override") is not None:
            num_rooms = int(self.global_knobs["num_rooms_override"])
        else:
            base_rooms = {1: 2, 2: 4, 3: 5, 4: 6}[tier]
            num_rooms = int(
                max(1 if tier == 1 else 3, min(12, round(base_rooms + float(self.global_knobs["num_rooms_offset"]))))
            )

        if self.global_knobs.get("enemies_per_room_override") is not None:
            enemies_per_room = float(self.global_knobs["enemies_per_room_override"])
        else:
            base_enemies = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.5}[tier]
            enemies_per_room = round(
                float(max(0.0, min(12.0, base_enemies * float(self.global_knobs["enemies_per_room_mult"])))),
                2,
            )

        ammo_mult = round(float(max(0.3, min(2.5, float(self.global_knobs["ammo_mult"])))), 2)
        return {
            "num_rooms": num_rooms,
            "num_keys": 0 if tier < 3 else (1 if tier == 3 else int(self.rng.choice([1, 2]))),
            "enemies_per_room": enemies_per_room,
            "ammo_mult": ammo_mult,
            "scenario": scenario,
        }

    @staticmethod
    def _to_plr_entry(raw: dict[str, Any], alpha: float) -> PLREntry:
        return PLREntry(
            level_id=str(raw["level_id"]),
            tier=int(raw["tier"]),
            knobs=dict(raw["knobs"]),
            seed=int(raw["seed"]),
            score_gae=float(raw["score_gae"]),
            last_seen_update=int(raw["last_seen_update"]),
            visits=int(raw["visits"]),
            scaffolding_alpha=float(alpha),
        )

    def add(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Validate and insert or update a PLR entry, evicting lowest score on capacity overflow."""
        tier = int(entry["tier"])
        if tier not in (1, 2, 3, 4):
            raise ValueError(f"PLR tier must be in 1..4, got {tier}")

        seed_val = int(entry["seed"])
        if not (self.seed_min <= seed_val <= self.seed_max):
            raise ValueError(
                f"PLR seed {seed_val} outside allowed training range [{self.seed_min}, {self.seed_max}]"
            )

        level_id = str(entry.get("level_id") or f"proc_t{tier}_s{seed_val}")
        clean = {
            "level_id": level_id,
            "tier": tier,
            "knobs": dict(entry.get("knobs") or self._default_knobs(tier)),
            "seed": seed_val,
            "score_gae": _sanitize_score(entry.get("score_gae", 1.0)),
            "last_seen_update": int(entry.get("last_seen_update", 0)),
            "visits": max(1, int(entry.get("visits", 1))),
        }
        self._used_seeds.add(seed_val)

        if level_id in self._by_id:
            existing = self._by_id[level_id]
            existing.update(clean)
            self._by_seed[seed_val] = existing
            return dict(existing)

        if len(self.entries) >= self.capacity:
            evict_idx = int(np.argmin([float(e["score_gae"]) for e in self.entries]))
            removed = self.entries.pop(evict_idx)
            self._by_id.pop(removed["level_id"], None)
            self._by_seed.pop(int(removed["seed"]), None)

        self.entries.append(clean)
        self._by_id[level_id] = clean
        self._by_seed[seed_val] = clean
        return dict(clean)

    insert = add
    register = add

    def _replay_weights(self, update_idx: int) -> np.ndarray:
        n = len(self.entries)
        scores = np.array(
            [max(1e-6, abs(_sanitize_score(e["score_gae"]))) for e in self.entries],
            dtype=np.float64,
        )
        order = np.argsort(-scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        p_score = (1.0 / ranks) ** (1.0 / self.temperature)
        p_score /= p_score.sum()

        staleness = np.array(
            [max(0.0, float(update_idx - int(e["last_seen_update"]))) for e in self.entries],
            dtype=np.float64,
        )
        if staleness.sum() > 0:
            p_stale = staleness / staleness.sum()
        else:
            p_stale = np.full(n, 1.0 / n, dtype=np.float64)

        mix = (1.0 - self.staleness_coef) * p_score + self.staleness_coef * p_stale
        mix /= mix.sum()
        return mix

    def sample(self, update_idx: int = 0, global_step: int = 0) -> PLREntry:
        """Prioritized sample from stored levels when non-empty; safe fresh fallback when empty."""
        alpha = self.scaffolding_alpha(global_step)
        if len(self.entries) > 0:
            probs = self._replay_weights(update_idx)
            idx = int(self.rng.choice(len(self.entries), p=probs))
            entry = self.entries[idx]
            entry["visits"] += 1
            entry["last_seen_update"] = int(update_idx)
            return self._to_plr_entry(entry, alpha)
        return self.sample_level(global_step=global_step, update_idx=update_idx)

    def sample_level(self, global_step: int = 0, update_idx: Optional[int] = None) -> PLREntry:
        """Training mix: replay stored frontier levels with `eff_replay_p`, else generate fresh level."""
        u_idx = self.update_idx if update_idx is None else int(update_idx)
        alpha = self.scaffolding_alpha(global_step)
        eff_replay_p = self.replay_prob * min(1.0, len(self.entries) / max(1.0, 0.25 * self.capacity))
        if len(self.entries) > 0 and self.rng.random() < eff_replay_p:
            return self.sample(update_idx=u_idx, global_step=global_step)

        tier = self._sample_tier()
        lvl_seed = self._new_seed()
        knobs = self._default_knobs(tier)
        level_id = f"proc_t{tier}_s{lvl_seed}"
        stored = self.add(
            {
                "level_id": level_id,
                "tier": tier,
                "knobs": knobs,
                "seed": lvl_seed,
                "score_gae": 1.0,
                "last_seen_update": u_idx,
                "visits": 1,
            }
        )
        return self._to_plr_entry(stored, alpha)

    def update_score(
        self,
        level_id: str | int,
        gae_score: float,
        update_idx: Optional[int] = None,
    ) -> None:
        """Update stored GAE priority score and staleness timestamp by level_id or seed."""
        if update_idx is None:
            self.update_idx += 1
            u_idx = self.update_idx
        else:
            u_idx = int(update_idx)
            self.update_idx = max(self.update_idx, u_idx)

        if isinstance(level_id, int):
            entry = self._by_seed.get(level_id)
        else:
            entry = self._by_id.get(level_id)
            if entry is None and level_id.isdigit():
                entry = self._by_seed.get(int(level_id))

        if entry is None:
            return
        old = float(entry["score_gae"])
        clean_s = _sanitize_score(gae_score)
        entry["score_gae"] = float(0.5 * old + 0.5 * clean_s)
        entry["last_seen_update"] = u_idx

    def update_knobs(self, knob_deltas: dict[str, Any]) -> dict[str, Any]:
        """Apply bounded adjustments to curriculum knobs and tier weights (e.g. from llm_coach)."""
        if "num_rooms" in knob_deltas:
            self.global_knobs["num_rooms_override"] = int(max(1, min(12, int(knob_deltas["num_rooms"]))))
        if "num_rooms_delta" in knob_deltas:
            self.global_knobs["num_rooms_override"] = None
            self.global_knobs["num_rooms_offset"] = float(
                max(-2.0, min(3.0, float(self.global_knobs["num_rooms_offset"]) + float(knob_deltas["num_rooms_delta"])))
            )

        if "enemies_per_room" in knob_deltas:
            val = float(knob_deltas["enemies_per_room"])
            self.global_knobs["enemies_per_room_override"] = float(max(0.0, min(12.0, val)))
        if "enemies_per_room_delta" in knob_deltas:
            self.global_knobs["enemies_per_room_override"] = None
            self.global_knobs["enemies_per_room_mult"] = float(
                max(
                    0.4,
                    min(2.0, float(self.global_knobs["enemies_per_room_mult"]) + float(knob_deltas["enemies_per_room_delta"])),
                )
            )
        if "enemies_per_room_mult" in knob_deltas:
            self.global_knobs["enemies_per_room_override"] = None
            self.global_knobs["enemies_per_room_mult"] = float(
                max(0.4, min(2.0, float(knob_deltas["enemies_per_room_mult"])))
            )

        if "ammo_mult_delta" in knob_deltas:
            self.global_knobs["ammo_mult"] = float(
                max(0.3, min(2.5, float(self.global_knobs["ammo_mult"]) + float(knob_deltas["ammo_mult_delta"])))
            )
        if "ammo_mult" in knob_deltas:
            self.global_knobs["ammo_mult"] = float(max(0.3, min(2.5, float(knob_deltas["ammo_mult"]))))

        if "tier_weights" in knob_deltas:
            tw = list(knob_deltas["tier_weights"])
            if len(tw) == 4:
                self.tier_weights = [float(max(0.2, min(5.0, float(x)))) for x in tw]

        return {
            "global_knobs": dict(self.global_knobs),
            "tier_weights": list(self.tier_weights),
            "scenarios": list(ALL_SCENARIOS),
        }

    apply_knobs = update_knobs
    set_knobs = update_knobs

    def state_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity,
            "tier_weights": list(self.tier_weights),
            "seed_min": self.seed_min,
            "seed_max": self.seed_max,
            "replay_prob": self.replay_prob,
            "staleness_coef": self.staleness_coef,
            "temperature": self.temperature,
            "scaffolding_end_step": self.scaffolding_end_step,
            "update_idx": int(self.update_idx),
            "used_seeds": sorted(int(s) for s in self._used_seeds),
            "rng_state": self.rng.bit_generator.state,
            "global_knobs": dict(self.global_knobs),
            "entries": [{**e, "knobs": dict(e["knobs"])} for e in self.entries],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.capacity = max(1, int(state.get("capacity", self.capacity)))
        self.tier_weights = [float(x) for x in state.get("tier_weights", self.tier_weights)]
        self.update_idx = int(state.get("update_idx", self.update_idx))
        self.global_knobs.update(dict(state.get("global_knobs", {})))
        self.entries = []
        self._by_id = {}
        self._by_seed = {}
        self._used_seeds = set(int(s) for s in state.get("used_seeds", []))
        for raw in state.get("entries", []):
            self.add(dict(raw))
        if "rng_state" in state and state["rng_state"] is not None:
            self.rng.bit_generator.state = state["rng_state"]
