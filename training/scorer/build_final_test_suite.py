"""Generate a strictly disjoint Final Test Suite (`training/scorer/final_test_suite.json`).

Three disjoint map splits:
1. Training:   procedural seeds in [100_000, 999_999]
2. Validation: `training/scorer/eval_suite.json` (procedural seeds [1_001, 1_024] + freedoom2 MAP01-MAP03)
3. Final Test: `training/scorer/final_test_suite.json` (procedural seeds [50_001, 50_016] + freedoom2 MAP04-MAP05)

Checks for zero overlap by both (seed, wad, map_name) and SHA-256 file hash.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from training.mapgen.tiers import MapGen

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VAL_MANIFEST = REPO_ROOT / "training" / "scorer" / "eval_suite.json"
TEST_MANIFEST = REPO_ROOT / "training" / "scorer" / "final_test_suite.json"
FINAL_TEST_DIR = REPO_ROOT / "maps" / "final_test"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_disjoint_splits(
    val_manifest: Path = VAL_MANIFEST,
    test_manifest: Path = TEST_MANIFEST,
    train_seed_min: int = 100_000,
    train_seed_max: int = 999_999,
) -> dict[str, int]:
    val_items = json.loads(val_manifest.read_text(encoding="utf-8"))
    test_items = json.loads(test_manifest.read_text(encoding="utf-8"))

    val_hashes = {item["sha256"] for item in val_items if item.get("sha256")}
    val_keys = {(item["wad"], item.get("map_name", "MAP01")) for item in val_items}
    val_seeds = {int(item["seed"]) for item in val_items if item.get("source") != "freedoom2"}

    for item in test_items:
        seed = int(item["seed"])
        key = (item["wad"], item.get("map_name", "MAP01"))
        if key in val_keys:
            raise AssertionError(f"Overlap between validation and final_test on {key}")
        if item.get("source") != "freedoom2":
            if seed in val_seeds:
                raise AssertionError(f"Seed {seed} appears in both validation and final_test")
            if train_seed_min <= seed <= train_seed_max:
                raise AssertionError(f"Final test seed {seed} overlaps training seed range")
            if item.get("sha256") and item["sha256"] in val_hashes:
                raise AssertionError(f"SHA-256 overlap on {item['id']}")

    return {"validation_maps": len(val_items), "final_test_maps": len(test_items)}


def build_final_test_suite() -> Path:
    FINAL_TEST_DIR.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []

    # 16 fresh procedural maps: 4 per Tier (1..4), seeds 50001..50016
    seed = 50001
    for tier in (1, 2, 3, 4):
        for _ in range(4):
            wad_path, map_name = MapGen.sample(tier=tier, seed=seed, out_dir=str(FINAL_TEST_DIR))
            target_wad = FINAL_TEST_DIR / f"final_t{tier}_seed{seed}.wad"
            if Path(wad_path).resolve() != target_wad.resolve():
                Path(wad_path).replace(target_wad)
            rel_wad = f"maps/final_test/{target_wad.name}"
            items.append(
                {
                    "id": f"final_test_t{tier}_{seed}",
                    "wad": rel_wad,
                    "map_name": map_name,
                    "tier": tier,
                    "episode_ticks": 4200 if tier == 4 else 2100,
                    "seed": seed,
                    "sha256": _file_sha256(target_wad),
                    "source": "procedural_final_test",
                    "split": "final_test",
                }
            )
            seed += 1

    # Add 2 community BSD-licensed Freedoom2 maps not used in validation (MAP04, MAP05)
    items.append(
        {
            "id": "final_test_freedoom2_map04",
            "wad": "freedoom2.wad",
            "map_name": "MAP04",
            "tier": 3,
            "episode_ticks": 4200,
            "seed": 50104,
            "sha256": "",
            "source": "freedoom2",
            "split": "final_test",
        }
    )
    items.append(
        {
            "id": "final_test_freedoom2_map05",
            "wad": "freedoom2.wad",
            "map_name": "MAP05",
            "tier": 4,
            "episode_ticks": 4200,
            "seed": 50105,
            "sha256": "",
            "source": "freedoom2",
            "split": "final_test",
        }
    )

    TEST_MANIFEST.write_text(json.dumps(items, indent=2) + "\n", encoding="utf-8")
    counts = verify_disjoint_splits(VAL_MANIFEST, TEST_MANIFEST)
    print(f"[FinalTestSuite] Built {TEST_MANIFEST} -> {counts}")
    return TEST_MANIFEST


if __name__ == "__main__":
    build_final_test_suite()
