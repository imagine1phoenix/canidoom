"""Contract, solvability, reward, and seed-disjointness tests for MapGen, Rewards, and PLR."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from repo root even if doom_env is inside template_reference/env
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "template_reference" / "env"))

from doom_env import make_env  # noqa: E402
from training.curriculum_plr import TRAIN_SEED_MAX, TRAIN_SEED_MIN, PLRBuffer  # noqa: E402
from training.mapgen.tiers import MapGen  # noqa: E402
from training.mapgen.udmf_builder import (  # noqa: E402
    LOCK_BLUE,
    THING_PLAYER1,
    SectorSpec,
    ThingSpec,
    UDMFMapBuilder,
)
from training.rewards import RewardShaper  # noqa: E402


def test_ray_recognizes_door_and_locked_door(tmp_path: Path) -> None:
    """Verify raycaster reports hit_type == 2 (door) and 3 (locked_door)."""
    # 1. Normal door test
    b_door = UDMFMapBuilder("MAP01")
    b_door.add_sector(SectorSpec(0, 0, 256, 256))
    b_door.add_sector(SectorSpec(256, 80, 320, 176, is_door=True, door_lock=0))
    b_door.add_sector(SectorSpec(320, 0, 576, 256))
    b_door.add_thing(ThingSpec(160, 128, 0, THING_PLAYER1))
    wad_door = b_door.write_wad(tmp_path / "door.wad")

    env = make_env(wad_door, "MAP01", episode_ticks=200, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert int(obs["rays"]["hit_type"][0]) == 2, f"Expected door (2), got {obs['rays']['hit_type'][0]}"
    finally:
        env.close()

    # 2. Locked door test
    b_lock = UDMFMapBuilder("MAP01")
    b_lock.add_sector(SectorSpec(0, 0, 256, 256))
    b_lock.add_sector(SectorSpec(256, 80, 320, 176, is_door=True, door_lock=LOCK_BLUE))
    b_lock.add_sector(SectorSpec(320, 0, 576, 256))
    b_lock.add_thing(ThingSpec(160, 128, 0, THING_PLAYER1))
    wad_lock = b_lock.write_wad(tmp_path / "locked_door.wad")

    env = make_env(wad_lock, "MAP01", episode_ticks=200, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert int(obs["rays"]["hit_type"][0]) == 3, f"Expected locked_door (3), got {obs['rays']['hit_type'][0]}"
    finally:
        env.close()


def test_exit_linedef_terminates_episode(tmp_path: Path) -> None:
    """Walking across generated exit alcove triggers terminated=True and info['exit_reached']=True."""
    b = UDMFMapBuilder("MAP01")
    b.add_sector(SectorSpec(0, 0, 256, 256))
    b.add_sector(SectorSpec(256, 96, 320, 160, is_exit=True, tex_floor="GATE3"))
    b.add_thing(ThingSpec(200, 128, 0, THING_PLAYER1))
    wad_exit = b.write_wad(tmp_path / "exit.wad")

    env = make_env(wad_exit, "MAP01", episode_ticks=200, render_buffers=False)
    try:
        env.reset(seed=0)
        shaper = RewardShaper()
        shaper.reset(env)
        action = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)  # MOVE_FORWARD
        reached = False
        for _ in range(30):
            _, _, terminated, truncated, info = env.step(action)
            r, _ = shaper.step(env, action)
            assert np.isfinite(r)
            if terminated:
                assert info["exit_reached"] is True
                assert r > 10.0
                reached = True
                break
            assert not truncated
        assert reached, "Player failed to trigger exit alcove within 30 steps"
    finally:
        env.close()


def test_rewards_on_two_rooms() -> None:
    """RewardShaper initializes and computes finite rewards on template two_rooms.wad."""
    wad = REPO_ROOT / "template_reference" / "maps" / "two_rooms.wad"
    env = make_env(wad, "MAP01", episode_ticks=400, render_buffers=False)
    try:
        env.reset(seed=0)
        shaper = RewardShaper()
        shaper.reset(env)
        assert shaper.visited_sector_count >= 1
        rng = np.random.default_rng(0)
        for _ in range(20):
            act = rng.integers(0, 2, size=9, dtype=np.int8)
            _, _, terminated, truncated, _ = env.step(act)
            r, _ = shaper.step(env, act)
            assert np.isfinite(r)
            if terminated or truncated:
                break
    finally:
        env.close()


def test_train_heldout_seed_disjoint() -> None:
    """Ensure zero intersection between held-out eval_suite.json seeds and PLR training seeds."""
    eval_path = REPO_ROOT / "training" / "scorer" / "eval_suite.json"
    assert eval_path.exists(), "eval_suite.json must exist"
    suite = json.loads(eval_path.read_text())
    heldout_seeds = {entry["seed"] for entry in suite if entry.get("seed") is not None}
    assert all(1000 <= s <= 9999 for s in heldout_seeds)

    plr = PLRBuffer(capacity=256, seed=123)
    for step in range(0, 100_000, 1000):
        res = plr.sample_level(global_step=step)
        if isinstance(res, tuple):
            entry, alpha = res
            seed_val = entry.seed
            lvl_id = entry.seed
        else:
            seed_val = int(res["seed"])
            alpha = float(res["scaffolding_alpha"])
            lvl_id = res["level_id"]
        assert TRAIN_SEED_MIN <= seed_val <= TRAIN_SEED_MAX
        assert seed_val not in heldout_seeds
        assert 0.0 <= alpha <= 1.0
        plr.update_score(lvl_id, 0.5, update_idx=step // 1000)


@pytest.mark.parametrize("tier", [1, 2, 3, 4])
def test_mapgen_all_tiers_load_and_step(tmp_path: Path, tier: int) -> None:
    """Each tier generates a valid WAD that loads in DoomEnv and steps cleanly."""
    res = MapGen.sample(tier=tier, seed=100_000 + tier, out_dir=tmp_path)
    wad = res[0] if isinstance(res, tuple) else res
    env = make_env(wad, "MAP01", episode_ticks=200, render_buffers=False)
    try:
        obs, _ = env.reset(seed=0)
        assert obs["rays"]["distance"].shape == (64,)
        for _ in range(5):
            act = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
            _, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break
    finally:
        env.close()


def test_eval_suite_loads_all_entries_including_freedoom2() -> None:
    """Verify eval_suite.json loads all 24 procedural WADs + 3 freedoom2 maps and paths exist."""
    from training.scorer.eval_suite import load_eval_suite_from_json

    specs = load_eval_suite_from_json(REPO_ROOT / "training" / "scorer" / "eval_suite.json")
    assert len(specs) == 27
    for s in specs:
        assert Path(s.wad_path).exists(), f"Missing WAD: {s.wad_path}"


def test_mapgen_reuses_existing_wad_and_handles_replace_permission_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replaying a seed reuses existing WADs without replacing open files, and tolerates WinError 5."""
    wad1, _ = MapGen.sample(tier=3, seed=161_128, scaffolding_alpha=0.25, out_dir=tmp_path)
    assert wad1.is_file()

    def _deny_replace(self: Path, target: Path | str) -> Path:
        raise PermissionError("Access is denied")

    monkeypatch.setattr(Path, "replace", _deny_replace)
    wad2, _ = MapGen.sample(tier=3, seed=161_128, scaffolding_alpha=0.25, out_dir=tmp_path)
    assert wad2 == wad1

    wad3, _ = MapGen.sample(tier=3, seed=161_129, scaffolding_alpha=0.25, out_dir=tmp_path)
    assert wad3.is_file() and wad3.stat().st_size >= 12
