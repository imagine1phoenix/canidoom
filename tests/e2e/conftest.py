"""Shared fixtures and opaque-box helpers for the 4-tier E2E test suite.

Rules:
- NEVER import `training.*` at module top level so `pytest --collect-only` and
  `pytest -m "not e2e"` always succeed while M2-M4 are under development.
- NEVER use `pytest.importorskip` for `training.*` modules.
- NEVER read `.env` or hardcode API keys/secrets.
- NEVER hardcode fixed world XYZ coordinates.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.fixtures import fixture_wad

REPO_ROOT = Path(__file__).resolve().parents[2]
MAPS_DIR = REPO_ROOT / "maps"
TWO_ROOMS_WAD = MAPS_DIR / "two_rooms.wad"
SUBMISSION_DIR = REPO_ROOT / "submission"
TRAINING_DIR = REPO_ROOT / "training"
CONFIGS_DIR = TRAINING_DIR / "configs"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "e2e: opaque-box end-to-end tests across Tiers 1-4",
    )


def import_training(submodule: str) -> Any:
    """Lazy import helper for `training.<submodule>` inside test bodies/fixtures."""
    return importlib.import_module(f"training.{submodule}")


def wad_from_grid(grid: str) -> str:
    """Resolve a pre-built SHA1-hashed fixture WAD path from its ASCII layout."""
    return fixture_wad(grid)


def freedoom_wad_path() -> Path:
    """Return path to bundled `freedoom2.wad` inside the installed `vizdoom` package."""
    import vizdoom

    return Path(os.path.dirname(vizdoom.__file__)) / "freedoom2.wad"


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"Expected YAML mapping in {path}, got {type(data)}"
    return data


def make_synthetic_obs(
    *,
    frame_shape: tuple[int, int] = (120, 160),
    render_buffers: bool = True,
    health: float = 100.0,
    armor: float = 0.0,
    selected_weapon: float = 2.0,
    selected_weapon_ammo: float = 50.0,
    ammo: tuple[float, float, float, float] = (50.0, 0.0, 0.0, 0.0),
    keys: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    vel: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ray_distances: np.ndarray | None = None,
    ray_hit_types: np.ndarray | None = None,
    num_entities: int = 2,
    entity_features: np.ndarray | None = None,
    entity_types: np.ndarray | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """Build a contract-compliant `doom_env` observation dict for `(120,160)` or `(48,64)`."""
    rng = np.random.default_rng(seed)
    h, w = frame_shape
    if render_buffers:
        rgb = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        depth = rng.integers(0, 256, size=(h, w), dtype=np.uint8)
    else:
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.uint8)

    vitals = np.zeros(17, dtype=np.float32)
    vitals[0] = health
    vitals[1] = armor
    vitals[2] = selected_weapon
    vitals[3] = selected_weapon_ammo
    vitals[4:8] = ammo
    vitals[8:14] = keys
    vitals[14:17] = vel

    if ray_distances is None:
        r_dist = np.full(64, 0.5, dtype=np.float32)
    else:
        r_dist = np.asarray(ray_distances, dtype=np.float32).copy()

    if ray_hit_types is None:
        r_type = np.ones(64, dtype=np.int64)
    else:
        r_type = np.asarray(ray_hit_types, dtype=np.int64).copy()

    ent_feats = np.zeros((32, 5), dtype=np.float32)
    ent_types = np.zeros(32, dtype=np.int64)
    ent_mask = np.zeros(32, dtype=np.int8)

    if entity_features is not None:
        n = min(32, len(entity_features))
        ent_feats[:n] = entity_features[:n]
        ent_mask[:n] = 1
        if entity_types is not None:
            ent_types[:n] = entity_types[:n]
        else:
            ent_types[:n] = 2
    elif num_entities > 0:
        n = min(32, num_entities)
        for i in range(n):
            angle = -0.3 + 0.2 * i
            ent_feats[i] = [0.2 + 0.1 * i, np.sin(angle), np.cos(angle), 0.5, -0.2]
            ent_types[i] = 2 if i % 2 == 0 else 5
            ent_mask[i] = 1

    return {
        "rgb": rgb,
        "depth": depth,
        "vitals": vitals,
        "rays": {
            "distance": r_dist,
            "hit_type": r_type,
        },
        "visible_entities": {
            "features": ent_feats,
            "entity_type": ent_types,
            "mask": ent_mask,
        },
    }


def make_dummy_model_inputs(
    *,
    batch_size: int = 2,
    num_entities: int = 3,
    zero_rgb: bool = False,
    zero_grid: bool = False,
    device: str = "cpu",
    seed: int = 0,
) -> Any:
    """Build a `training.preprocessing.ModelInputs` batch lazily."""
    import torch

    prep = import_training("preprocessing")
    ModelInputs = prep.ModelInputs
    rng = np.random.default_rng(seed)
    B = batch_size

    rays = np.zeros((B, 8, 64), dtype=np.float32)
    dist = rng.uniform(0.1, 0.95, size=(B, 64)).astype(np.float32)
    rays[:, 0, :] = dist
    rays[:, 1, :] = 1.0 - dist
    hit_idx = rng.integers(0, 6, size=(B, 64))
    for b in range(B):
        for r in range(64):
            rays[b, 2 + hit_idx[b, r], r] = 1.0

    ent_feats = np.zeros((B, 32, 5), dtype=np.float32)
    ent_types = np.zeros((B, 32), dtype=np.int64)
    ent_mask = np.zeros((B, 32), dtype=bool)
    if num_entities > 0:
        k = min(32, num_entities)
        ent_feats[:, :k, :] = rng.normal(0.0, 0.5, size=(B, k, 5)).astype(np.float32)
        ent_feats[:, :k, 0] = np.abs(ent_feats[:, :k, 0]) + 0.05
        ent_types[:, :k] = rng.integers(1, 12, size=(B, k))
        ent_mask[:, :k] = True

    vitals = rng.normal(0.0, 0.3, size=(B, 36)).astype(np.float32)
    rgb_depth = (
        np.zeros((B, 4, 48, 64), dtype=np.float32)
        if zero_rgb
        else rng.uniform(0.0, 1.0, size=(B, 4, 48, 64)).astype(np.float32)
    )
    grid = (
        np.zeros((B, 6, 32, 32), dtype=np.float32)
        if zero_grid
        else rng.uniform(0.0, 1.0, size=(B, 6, 32, 32)).astype(np.float32)
    )
    action_mask = np.ones((B, 9), dtype=bool)

    return ModelInputs(
        rays=torch.as_tensor(rays, dtype=torch.float32, device=device),
        ent_feats=torch.as_tensor(ent_feats, dtype=torch.float32, device=device),
        ent_types=torch.as_tensor(ent_types, dtype=torch.int64, device=device),
        ent_mask=torch.as_tensor(ent_mask, dtype=torch.bool, device=device),
        vitals=torch.as_tensor(vitals, dtype=torch.float32, device=device),
        rgb_depth=torch.as_tensor(rgb_depth, dtype=torch.float32, device=device),
        grid=torch.as_tensor(grid, dtype=torch.float32, device=device),
        action_mask=torch.as_tensor(action_mask, dtype=torch.bool, device=device),
    )


def write_synthetic_npz(
    path: Path,
    *,
    num_steps: int = 40,
    seed: int = 0,
    done_indices: tuple[int, ...] | None = None,
    sparse_use_attack: bool = True,
    tier: int = 1,
) -> Path:
    """Write a synthetic `.npz` episode archive strictly conforming to `Schema.md §3`."""
    rng = np.random.default_rng(seed)
    T = num_steps
    rgb = rng.integers(0, 256, size=(T, 48, 64, 3), dtype=np.uint8)
    depth = rng.integers(0, 256, size=(T, 48, 64), dtype=np.uint8)
    vitals = np.zeros((T, 17), dtype=np.float32)
    vitals[:, 0] = 100.0
    vitals[:, 2] = 2.0
    vitals[:, 3] = 50.0
    vitals[:, 4] = 50.0
    vitals[:, 14] = rng.uniform(-5.0, 10.0, size=T).astype(np.float32)

    rays_distance = rng.uniform(0.1, 1.0, size=(T, 64)).astype(np.float32)
    rays_hit_type = rng.integers(0, 6, size=(T, 64), dtype=np.uint8)
    ent_features = np.zeros((T, 32, 5), dtype=np.float32)
    ent_type = np.zeros((T, 32), dtype=np.uint8)
    ent_mask = np.zeros((T, 32), dtype=np.uint8)
    ent_features[:, 0] = [0.3, 0.0, 1.0, 0.0, 0.0]
    ent_type[:, 0] = 2
    ent_mask[:, 0] = 1

    actions = np.zeros((T, 9), dtype=np.uint8)
    actions[:, 0] = 1  # MOVE_FORWARD
    if sparse_use_attack and T >= 6:
        actions[2, 6] = 1  # ATTACK
        actions[4, 7] = 1  # USE

    rewards = np.full(T, -0.002, dtype=np.float32)
    rewards[-1] = 15.0

    dones = np.zeros(T, dtype=bool)
    if done_indices is None:
        dones[-1] = True
    else:
        for idx in done_indices:
            if 0 <= idx < T:
                dones[idx] = True

    meta = {
        "wad": "synthetic_t1.wad",
        "map": "MAP01",
        "tier": tier,
        "source": "synthetic",
        "exit_reached": True,
        "ticks": int(T * 4),
        "contract_version": "1.0.0-draft",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        rgb=rgb,
        depth=depth,
        vitals=vitals,
        rays_distance=rays_distance,
        rays_hit_type=rays_hit_type,
        ent_features=ent_features,
        ent_type=ent_type,
        ent_mask=ent_mask,
        actions=actions,
        rewards=rewards,
        dones=dones,
        meta_json=np.array(json.dumps(meta)),
    )
    return path


def make_priv_state(
    *,
    tic: int = 4,
    episode_ticks: int = 2100,
    x: float = 0.0,
    y: float = 0.0,
    z: float = 0.0,
    angle: float = 0.0,
    health: float = 100.0,
    kills: int = 0,
    items: int = 0,
    damage_taken: float = 0.0,
    keys: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    dead: bool = False,
    timeout: bool = False,
    exit_reached: bool = False,
    finished: bool = False,
    sector_floor: list[float] | np.ndarray | None = None,
    sector_ceiling: list[float] | np.ndarray | None = None,
    door_lock: list[int] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Create a privileged_state dict matching `DoomEnv.privileged_state()`."""
    return {
        "tic": tic,
        "episode_ticks": episode_ticks,
        "x": float(x),
        "y": float(y),
        "z": float(z),
        "angle": float(angle),
        "health": float(health),
        "kills": int(kills),
        "items": int(items),
        "damage_taken": float(damage_taken),
        "keys": np.asarray(keys, dtype=np.float32),
        "dead": bool(dead),
        "timeout": bool(timeout),
        "exit_reached": bool(exit_reached),
        "finished": bool(finished),
        "sector_floor": np.asarray([0.0, 0.0] if sector_floor is None else sector_floor, dtype=np.float64),
        "sector_ceiling": np.asarray([128.0, 0.0] if sector_ceiling is None else sector_ceiling, dtype=np.float64),
        "door_lock": np.asarray([0, 0] if door_lock is None else door_lock, dtype=np.int64),
    }


def make_square_sector_polygons(
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    size: float = 256.0,
) -> list[tuple[int, np.ndarray]]:
    """Build two adjacent square sectors `[0, size]` and `[size, 2*size]` offset by `(origin_x, origin_y)`."""
    def square_lines(x0: float, y0: float, s: float) -> np.ndarray:
        return np.array(
            [
                [x0, y0, x0 + s, y0],
                [x0 + s, y0, x0 + s, y0 + s],
                [x0 + s, y0 + s, x0, y0 + s],
                [x0, y0 + s, x0, y0],
            ],
            dtype=np.float64,
        )

    return [
        (0, square_lines(origin_x, origin_y, size)),
        (1, square_lines(origin_x + size, origin_y, size)),
    ]


@pytest.fixture
def two_rooms_wad() -> Path:
    assert TWO_ROOMS_WAD.is_file(), f"Missing {TWO_ROOMS_WAD}"
    return TWO_ROOMS_WAD


@pytest.fixture
def make_live_env():
    """Factory fixture creating `doom_env.make_env` instances and closing them on teardown."""
    from doom_env import make_env

    created = []

    def _factory(wad: str | Path, map_name: str = "MAP01", **kwargs: Any):
        env = make_env(str(wad), map_name=map_name, **kwargs)
        created.append(env)
        return env

    yield _factory
    for env in created:
        try:
            env.close()
        except Exception:
            pass
