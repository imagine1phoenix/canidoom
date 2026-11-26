"""Evaluation Suite Manifest for Zero-Shot Generalization Benchmarking.

Defines the held-out map suite across all 4 tiers:
- Tier 1: Combat & Survival
- Tier 2: Multi-Room Exploration
- Tier 3: Long-Horizon Progression (Keycard -> Locked Door -> Exit)
- Tier 4: Full Gauntlet
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

TIER_WEIGHTS = {
    1: 1.0,
    2: 1.5,
    3: 2.5,
    4: 3.0,
}


@dataclass
class EvalMapSpec:
    id: str
    wad_path: str
    map_name: str
    tier: int
    episode_ticks: int = 2100
    seed: int = 1001
    source: str = "heldout"


def get_default_eval_suite(repo_root: Path | None = None) -> list[EvalMapSpec]:
    """Returns default held-out evaluation suite using available test maps."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent.parent

    maps_dir = repo_root / "maps"
    fixtures_dir = repo_root / "tests" / "fixtures"

    specs = []

    # Two rooms map (Tier 2 exploration baseline)
    two_rooms_wad = maps_dir / "two_rooms.wad"
    if two_rooms_wad.exists():
        specs.append(
            EvalMapSpec(
                id="two_rooms_baseline",
                wad_path=str(two_rooms_wad),
                map_name="MAP01",
                tier=2,
                episode_ticks=2100,
                seed=1001,
            )
        )

    # Fixture maps for multi-room / doors
    if fixtures_dir.exists():
        wad_files = sorted(list(fixtures_dir.glob("*.wad")))
        for i, wad_path in enumerate(wad_files[:4]):
            tier = 1 if i == 0 else (2 if i < 3 else 3)
            specs.append(
                EvalMapSpec(
                    id=f"fixture_tier{tier}_{wad_path.stem[:8]}",
                    wad_path=str(wad_path),
                    map_name="MAP01",
                    tier=tier,
                    episode_ticks=2100,
                    seed=1002 + i,
                )
            )

    return specs


def load_eval_suite_from_json(json_path: Path | str) -> list[EvalMapSpec]:
    """Loads map specifications from a JSON manifest (Schema.md §5.1)."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    specs = []
    for item in data:
        specs.append(
            EvalMapSpec(
                id=item["id"],
                wad_path=item["wad"],
                map_name=item.get("map_name", "MAP01"),
                tier=int(item["tier"]),
                episode_ticks=int(item.get("episode_ticks", 2100)),
                seed=int(item.get("seed", 1001)),
                source=item.get("source", "procedural_heldout"),
            )
        )
    return specs
