"""Coordinate-free ego-centric observation preprocessing and 360-degree scan-matching odometry grid.

Strictly ego-centric and translation/rotation-invariant (zero fixed world XYZ assumptions).
Matches planning/Schema.md §2.1 and §2.2, and stays compatible with submission/agent.py.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch

NUM_RAYS = 64
MAX_ENTITIES = 32
GRID_SIZE = 32
GRID_CHANNELS = 6
GRID_CENTER = 16
CELL_SIZE = 48.0
FRAME_SKIP = 4
RAY_RANGE = 1024.0
RGB_HEIGHT = 48
RGB_WIDTH = 64
NUM_VITALS_RAW = 17
NUM_VITALS_PREPROCESSED = 36
NUM_ACTIONS = 9
FRONT_RAY_INDICES = (60, 61, 62, 63, 0, 1, 2, 3, 4)
DOOR_HIT_TYPES = (2, 3)

# Ray angles: index 0 = straight ahead, positive CCW (16 = +90 deg left, 32 = behind, 48 = right)
_RAY_ANGLES = np.arange(NUM_RAYS, dtype=np.float32) * (2.0 * np.pi / NUM_RAYS)
_RAY_COS = np.cos(_RAY_ANGLES)
_RAY_SIN = np.sin(_RAY_ANGLES)


@dataclass
class OdomGridState:
    """Persistent per-episode egocentric odometry & action-cooldown state."""

    grid: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    )
    prev_ray_dist: np.ndarray = field(
        default_factory=lambda: np.ones((NUM_RAYS,), dtype=np.float32)
    )
    prev_ray_type: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_RAYS,), dtype=np.int64)
    )
    prev_action: np.ndarray = field(
        default_factory=lambda: np.zeros((NUM_ACTIONS,), dtype=np.float32)
    )
    use_cooldown: int = 0
    initialized: bool = False
    last_dtheta: float = 0.0
    use_consumed: bool = False
    cooldown_pre_armed: bool = False

    @classmethod
    def zeros(cls) -> "OdomGridState":
        return cls()

    def copy(self) -> "OdomGridState":
        return OdomGridState(
            grid=self.grid.copy(),
            prev_ray_dist=self.prev_ray_dist.copy(),
            prev_ray_type=self.prev_ray_type.copy(),
            prev_action=np.asarray(self.prev_action, dtype=np.float32).copy(),
            use_cooldown=int(self.use_cooldown),
            initialized=bool(self.initialized),
            last_dtheta=float(self.last_dtheta),
            use_consumed=bool(self.use_consumed),
            cooldown_pre_armed=bool(self.cooldown_pre_armed),
        )


@dataclass
class ModelInputs:
    """Preprocessed neural input tensors matching planning/Schema.md §2.1."""

    rays: torch.Tensor        # (B, 8, 64) float32
    ent_feats: torch.Tensor   # (B, 32, 5) float32
    ent_types: torch.Tensor   # (B, 32) int64
    ent_mask: torch.Tensor    # (B, 32) bool
    vitals: torch.Tensor      # (B, 36) float32
    rgb_depth: torch.Tensor   # (B, 4, 48, 64) float32
    grid: torch.Tensor        # (B, 6, 32, 32) float32
    action_mask: torch.Tensor # (B, 9) bool
    d_theta: Optional[torch.Tensor] = None  # (B,) float32 optional odometry yaw delta

    def to(self, device: torch.device | str) -> "ModelInputs":
        return ModelInputs(
            rays=self.rays.to(device),
            ent_feats=self.ent_feats.to(device),
            ent_types=self.ent_types.to(device),
            ent_mask=self.ent_mask.to(device),
            vitals=self.vitals.to(device),
            rgb_depth=self.rgb_depth.to(device),
            grid=self.grid.to(device),
            action_mask=self.action_mask.to(device),
            d_theta=self.d_theta.to(device) if self.d_theta is not None else None,
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {
            "rays": self.rays,
            "ent_feats": self.ent_feats,
            "ent_types": self.ent_types,
            "ent_mask": self.ent_mask,
            "vitals": self.vitals,
            "rgb_depth": self.rgb_depth,
            "grid": self.grid,
            "action_mask": self.action_mask,
        }
        if self.d_theta is not None:
            out["d_theta"] = self.d_theta
        return out


def stack_model_inputs(items: Sequence[ModelInputs], dim: int = 0) -> ModelInputs:
    """Concatenate a sequence of ModelInputs along `dim`."""
    if not items:
        raise ValueError("Cannot stack empty sequence of ModelInputs")
    has_dtheta = all(x.d_theta is not None for x in items)
    return ModelInputs(
        rays=torch.cat([x.rays for x in items], dim=dim),
        ent_feats=torch.cat([x.ent_feats for x in items], dim=dim),
        ent_types=torch.cat([x.ent_types for x in items], dim=dim),
        ent_mask=torch.cat([x.ent_mask for x in items], dim=dim),
        vitals=torch.cat([x.vitals for x in items], dim=dim),
        rgb_depth=torch.cat([x.rgb_depth for x in items], dim=dim),
        grid=torch.cat([x.grid for x in items], dim=dim),
        action_mask=torch.cat([x.action_mask for x in items], dim=dim),
        d_theta=torch.cat([x.d_theta for x in items if x.d_theta is not None], dim=dim) if has_dtheta else None,
    )


def _estimate_dtheta(
    ray_dist: np.ndarray,
    ray_type: np.ndarray,
    prev_ray_dist: np.ndarray,
    prev_ray_type: np.ndarray,
    prev_action: np.ndarray,
    num_visible_entities: int,
) -> float:
    """Estimate ego yaw change Δθ (radians, CCW positive) via 64-ray circular scan-matching."""
    ray_dist = np.clip(np.nan_to_num(ray_dist, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    prev_ray_dist = np.clip(np.nan_to_num(prev_ray_dist, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    prev_act = np.nan_to_num(prev_action, nan=0.0, posinf=0.0, neginf=0.0)

    k_prior = int(round(float(prev_act[4]) - float(prev_act[5])))
    prior_weight = 5e-5 if num_visible_entities > 2 else 2e-5

    # Filter out moving enemies (hit_type == 4) and max-range misses
    cur_static = (ray_dist < 0.98) & (ray_type != 4)
    prev_static = (prev_ray_dist < 0.98) & (prev_ray_type != 4)

    candidates = range(k_prior - 4, k_prior + 5)
    mses: list[float] = []
    scores: list[float] = []
    valid_counts: list[int] = []

    # When turning CCW by +k bins, world features shift CW by -k indices: curr[i] ~= prev[i + k]
    for k in candidates:
        shifted_dist = np.roll(prev_ray_dist, -k)
        shifted_static = np.roll(prev_static, -k)
        joint = cur_static & shifted_static
        n_valid = int(joint.sum())
        valid_counts.append(n_valid)
        if n_valid >= 4:
            diff = ray_dist[joint] - shifted_dist[joint]
            mse = float(np.mean(diff * diff))
            mses.append(mse)
            scores.append(mse + prior_weight * float((k - k_prior) ** 2))
        else:
            mses.append(1e6)
            scores.append(1e6 + prior_weight * float((k - k_prior) ** 2))

    if max(valid_counts) < 4:
        return float(k_prior) * (2.0 * math.pi / NUM_RAYS)

    best_idx = int(np.argmin(scores))
    best_k = (k_prior - 4) + best_idx

    # 3-point parabolic refinement on pure scan-matching MSE when interior minimum is valid
    delta = 0.0
    if 0 < best_idx < len(mses) - 1:
        y_m, y_0, y_p = mses[best_idx - 1], mses[best_idx], mses[best_idx + 1]
        if y_m < 1e5 and y_p < 1e5:
            denom = y_m - 2.0 * y_0 + y_p
            if denom > 1e-8:
                delta = 0.5 * (y_m - y_p) / denom
                delta = max(-0.5, min(0.5, delta))

    return (float(best_k) + delta) * (2.0 * math.pi / NUM_RAYS)


def _update_egocentric_grid(
    prev_grid: np.ndarray,
    vel_forward: float,
    vel_right: float,
    dtheta: float,
    ray_dist: np.ndarray,
    ray_type: np.ndarray,
) -> np.ndarray:
    """Warp previous (6, 32, 32) grid by ego motion and splat current 64 ray endpoints."""
    # Decay existing grid slowly; keep visited (0) & locked_door (3) longer
    grid = np.nan_to_num(prev_grid, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=True)
    grid[0] *= 0.998
    grid[1] *= 0.992
    grid[2] *= 0.992
    grid[3] *= 0.998
    grid[4] *= 0.992
    grid[5] *= 0.992

    if not math.isfinite(vel_forward):
        vel_forward = 0.0
    if not math.isfinite(vel_right):
        vel_right = 0.0
    if not math.isfinite(dtheta):
        dtheta = 0.0

    dx_cells = (vel_forward * FRAME_SKIP) / CELL_SIZE
    dy_cells = (vel_right * FRAME_SKIP) / CELL_SIZE

    c = math.cos(dtheta)
    s = math.sin(dtheta)
    # Image coordinates: col = 16 + right_cells, row = 16 - fwd_cells
    # Shift old points by (-dx_cells, -dy_cells) then rotate by -dtheta
    tx = -c * (GRID_CENTER + dy_cells) + s * (GRID_CENTER - dx_cells) + GRID_CENTER
    ty = -s * (GRID_CENTER + dy_cells) - c * (GRID_CENTER - dx_cells) + GRID_CENTER
    warp_mat = np.array([[c, -s, tx], [s, c, ty]], dtype=np.float32)

    hwc = np.transpose(grid, (1, 2, 0))
    warped_hwc = cv2.warpAffine(
        hwc,
        warp_mat,
        (GRID_SIZE, GRID_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    grid = np.clip(np.transpose(warped_hwc, (2, 0, 1)), 0.0, 1.0).astype(np.float32)

    # Mark current player position as visited
    grid[0, GRID_CENTER, GRID_CENTER] = 1.0

    # Splat ray endpoints (hit_type != 0 and hit distance < 0.99)
    hit_mask = (ray_type > 0) & (ray_dist < 0.99)
    if np.any(hit_mask):
        r_cells = (ray_dist[hit_mask] * RAY_RANGE) / CELL_SIZE
        cos_h = _RAY_COS[hit_mask]
        sin_h = _RAY_SIN[hit_mask]
        # theta = CCW from forward => fwd = r*cos, right = -r*sin
        rows = np.rint(GRID_CENTER - r_cells * cos_h).astype(np.int64)
        cols = np.rint(GRID_CENTER - r_cells * sin_h).astype(np.int64)
        in_bounds = (rows >= 0) & (rows < GRID_SIZE) & (cols >= 0) & (cols < GRID_SIZE)
        rows = rows[in_bounds]
        cols = cols[in_bounds]
        types = ray_type[hit_mask][in_bounds]

        # Channel mapping: 1=wall, 2=door, 3=locked_door, 4=item(5), 5=enemy(4)
        for ch, target_type in ((1, 1), (2, 2), (3, 3), (4, 5), (5, 4)):
            sel = types == target_type
            if np.any(sel):
                grid[ch, rows[sel], cols[sel]] = 1.0

    return grid


def _pack_rgb_depth(obs: dict[str, Any]) -> np.ndarray:
    """Convert raw RGB (120x160x3 or 48x64x3) and Depth (120x160 or 48x64) to (4, 48, 64) float32."""
    rgb_raw = np.asarray(obs["rgb"])
    if np.issubdtype(rgb_raw.dtype, np.floating):
        rgb = np.clip(np.nan_to_num(rgb_raw, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)
    else:
        rgb = rgb_raw.astype(np.uint8, copy=False)

    depth_raw = np.asarray(obs["depth"])
    if np.issubdtype(depth_raw.dtype, np.floating):
        depth = np.clip(np.nan_to_num(depth_raw, nan=0.0, posinf=255.0, neginf=0.0), 0.0, 255.0).astype(np.uint8)
    else:
        depth = depth_raw.astype(np.uint8, copy=False)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if rgb.shape[:2] != (RGB_HEIGHT, RGB_WIDTH):
        rgb = cv2.resize(rgb, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_LINEAR)
    if depth.shape[:2] != (RGB_HEIGHT, RGB_WIDTH):
        depth = cv2.resize(depth, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_LINEAR)

    out = np.empty((4, RGB_HEIGHT, RGB_WIDTH), dtype=np.float32)
    out[:3] = np.transpose(rgb, (2, 0, 1)).astype(np.float32) * (1.0 / 255.0)
    out[3] = depth.astype(np.float32) * (1.0 / 255.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def _validate_obs_contract(obs: dict[str, Any]) -> None:
    """Verify observation keys and expected tensor shapes before encoding."""
    if not isinstance(obs, dict):
        raise ValueError(f"obs must be a dict, got {type(obs).__name__}")
    for key in ("rgb", "depth", "vitals", "rays", "visible_entities"):
        if key not in obs:
            raise KeyError(f"obs missing required key '{key}'")

    rgb = np.asarray(obs["rgb"])
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] not in ((120, 160), (48, 64)):
        raise ValueError(f"Invalid rgb shape {rgb.shape}; expected (120, 160, 3) or (48, 64, 3)")

    depth = np.asarray(obs["depth"])
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth_hw = depth.shape[:2]
    elif depth.ndim == 2:
        depth_hw = depth.shape
    else:
        raise ValueError(f"Invalid depth shape {depth.shape}; expected (120, 160) or (48, 64)")
    if depth_hw not in ((120, 160), (48, 64)):
        raise ValueError(f"Invalid depth spatial shape {depth_hw}; expected (120, 160) or (48, 64)")

    vitals = np.asarray(obs["vitals"])
    if vitals.shape != (NUM_VITALS_RAW,):
        raise ValueError(f"Invalid vitals shape {vitals.shape}; expected (17,)")

    rays = obs["rays"]
    if "distance" not in rays or "hit_type" not in rays:
        raise KeyError("obs['rays'] must contain 'distance' and 'hit_type'")
    if np.asarray(rays["distance"]).shape != (NUM_RAYS,) or np.asarray(rays["hit_type"]).shape != (NUM_RAYS,):
        raise ValueError("obs['rays'] fields must have shape (64,)")

    ve = obs["visible_entities"]
    for sub in ("features", "entity_type", "mask"):
        if sub not in ve:
            raise KeyError(f"obs['visible_entities'] missing '{sub}'")
    if np.asarray(ve["features"]).shape != (MAX_ENTITIES, 5):
        raise ValueError("visible_entities['features'] must have shape (32, 5)")
    if np.asarray(ve["entity_type"]).shape != (MAX_ENTITIES,) or np.asarray(ve["mask"]).shape != (MAX_ENTITIES,):
        raise ValueError("visible_entities['entity_type'] and ['mask'] must have shape (32,)")


def compute_action_mask(obs: dict[str, Any], state: OdomGridState) -> np.ndarray:
    """Compute observation-legal action mask of shape (9,) bool."""
    mask = np.ones((NUM_ACTIONS,), dtype=bool)
    if int(state.use_cooldown) > 0:
        mask[7] = False
    v = np.nan_to_num(
        np.asarray(obs.get("vitals", np.zeros(NUM_VITALS_RAW, dtype=np.float32)), dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if len(v) >= 4:
        selected_weapon = int(round(float(v[2])))
        selected_ammo = float(v[3])
        if selected_weapon > 1 and selected_ammo <= 0.0:
            mask[6] = False
    return mask


def preprocess(
    obs: dict[str, Any],
    state: Optional[OdomGridState] = None,
    device: torch.device | str = "cpu",
    use_grid: bool = True,
) -> tuple[ModelInputs, OdomGridState]:
    """Convert a single raw doom_env observation dict to (ModelInputs, updated OdomGridState).

    Cleanly initializes state when `state is None`. Strictly ego-centric and coordinate-free.
    """
    _validate_obs_contract(obs)

    if state is None:
        cur_state = OdomGridState.zeros()
    else:
        cur_state = state

    # --- 1. Rays (1, 8, 64) ---
    raw_ray_dist = np.nan_to_num(
        np.asarray(obs["rays"]["distance"], dtype=np.float32),
        nan=1.0,
        posinf=1.0,
        neginf=0.0,
    )
    ray_dist = np.clip(raw_ray_dist, 0.0, 1.0)
    raw_ray_type = np.nan_to_num(np.asarray(obs["rays"]["hit_type"]), nan=0.0).astype(np.int64)
    ray_type = np.clip(raw_ray_type, 0, 5)
    rays_arr = np.zeros((8, NUM_RAYS), dtype=np.float32)
    rays_arr[0] = ray_dist
    rays_arr[1] = 1.0 - ray_dist
    rays_arr[2 + ray_type, np.arange(NUM_RAYS)] = 1.0

    # --- 2. Visible Entities (1, 32, 5), (1, 32), (1, 32) ---
    ve = obs["visible_entities"]
    raw_mask = np.asarray(ve["mask"]).astype(bool)
    raw_feats = np.nan_to_num(
        np.asarray(ve["features"], dtype=np.float32),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).copy()
    raw_types = np.clip(np.nan_to_num(np.asarray(ve["entity_type"]), nan=0.0).astype(np.int64), 0, 11).copy()

    # sensors.py:295 emits raw rel_vel_fwd, rel_vel_right in map units/tic; normalize by 30.0
    raw_feats[:, 3:5] *= 1.0 / 30.0
    raw_feats[~raw_mask] = 0.0
    raw_types[~raw_mask] = 0

    # --- 3. Vitals + Action Context (1, 36) ---
    v = np.nan_to_num(np.asarray(obs["vitals"], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    vitals_arr = np.zeros((NUM_VITALS_PREPROCESSED,), dtype=np.float32)
    vitals_arr[0] = np.clip(v[0] / 100.0, 0.0, 2.0)  # health
    vitals_arr[1] = np.clip(v[1] / 100.0, 0.0, 2.0)  # armor
    vitals_arr[2] = np.log1p(np.clip(v[3], 0.0, 500.0)) / 6.2  # selected_weapon_ammo
    vitals_arr[3:7] = np.log1p(np.clip(v[4:8], 0.0, 500.0)) / 6.2  # ammo 2,3,5,6
    vitals_arr[7:13] = np.clip(v[8:14], 0.0, 1.0)  # 6 keys
    vitals_arr[13:16] = np.clip(v[14:17] / 30.0, -2.0, 2.0)  # vel fwd, right, up

    selected_weapon = int(round(float(v[2])))
    weapon_idx = max(0, min(7, selected_weapon))
    vitals_arr[16 + weapon_idx] = 1.0

    prev_act = np.nan_to_num(
        np.asarray(cur_state.prev_action, dtype=np.float32).reshape(NUM_ACTIONS),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    use_pressed = int(round(float(prev_act[7]))) == 1
    current_cooldown = max(0, int(cur_state.use_cooldown))
    next_use_consumed = bool(cur_state.use_consumed)

    if current_cooldown > 0:
        if not cur_state.cooldown_pre_armed:
            current_cooldown -= 1
    elif use_pressed and not next_use_consumed:
        idx = np.asarray(FRONT_RAY_INDICES, dtype=np.int64)
        front_door = bool(np.any(((ray_type[idx] == 2) | (ray_type[idx] == 3)) & (ray_dist[idx] < 0.15)))
        if front_door:
            current_cooldown = 6
            next_use_consumed = True

    if not use_pressed:
        next_use_consumed = False

    vitals_arr[24] = min(1.0, float(current_cooldown) / 6.0)
    vitals_arr[25] = 1.0 if float(np.sum(v[8:14])) > 0.5 else 0.0
    selected_ammo = float(v[3])
    vitals_arr[26] = 1.0 if (selected_weapon > 1 and selected_ammo <= 5.0) else 0.0
    vitals_arr[27:36] = prev_act

    # --- 4. RGB + Depth (1, 4, 48, 64) ---
    rgb_depth_arr = _pack_rgb_depth(obs)

    # --- 5. Scan-Matched Egocentric Occupancy Grid (1, 6, 32, 32) ---
    if use_grid:
        if cur_state.initialized:
            dtheta = _estimate_dtheta(
                ray_dist=ray_dist,
                ray_type=ray_type,
                prev_ray_dist=cur_state.prev_ray_dist,
                prev_ray_type=cur_state.prev_ray_type,
                prev_action=prev_act,
                num_visible_entities=int(raw_mask.sum()),
            )
        else:
            dtheta = 0.0
        next_grid = _update_egocentric_grid(
            prev_grid=cur_state.grid,
            vel_forward=float(v[14]),
            vel_right=float(v[15]),
            dtheta=dtheta,
            ray_dist=ray_dist,
            ray_type=ray_type,
        )
    else:
        dtheta = 0.0
        next_grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)

    # --- 6. Observation-Legal Action Mask (1, 9) ---
    action_mask_arr = np.ones((NUM_ACTIONS,), dtype=bool)
    if current_cooldown > 0:
        action_mask_arr[7] = False  # USE masked during cooldown
    if selected_weapon > 1 and selected_ammo <= 0.0:
        action_mask_arr[6] = False  # ATTACK masked when non-melee weapon is out of ammo

    next_state = OdomGridState(
        grid=next_grid,
        prev_ray_dist=ray_dist.copy(),
        prev_ray_type=ray_type.copy(),
        prev_action=prev_act.copy(),
        use_cooldown=current_cooldown,
        initialized=True,
        last_dtheta=float(dtheta),
        use_consumed=next_use_consumed,
        cooldown_pre_armed=False,
    )

    dev = torch.device(device)
    model_inputs = ModelInputs(
        rays=torch.from_numpy(np.ascontiguousarray(rays_arr[None, ...])).to(dev),
        ent_feats=torch.from_numpy(np.ascontiguousarray(raw_feats[None, ...])).to(dev),
        ent_types=torch.from_numpy(np.ascontiguousarray(raw_types[None, ...])).to(dev),
        ent_mask=torch.from_numpy(np.ascontiguousarray(raw_mask[None, ...])).to(dev),
        vitals=torch.from_numpy(np.ascontiguousarray(vitals_arr[None, ...])).to(dev),
        rgb_depth=torch.from_numpy(np.ascontiguousarray(rgb_depth_arr[None, ...])).to(dev),
        grid=torch.from_numpy(np.ascontiguousarray(next_grid[None, ...])).to(dev),
        action_mask=torch.from_numpy(np.ascontiguousarray(action_mask_arr[None, ...])).to(dev),
        d_theta=torch.tensor([dtheta], dtype=torch.float32, device=dev),
    )
    return model_inputs, next_state


def update_action_state(
    state: OdomGridState,
    action: np.ndarray | Sequence[int],
    obs: Optional[dict[str, Any]] = None,
) -> OdomGridState:
    """Update state with executed `action` and trigger 6-step USE cooldown if facing a door."""
    act = np.nan_to_num(
        np.asarray(action, dtype=np.float32).reshape(NUM_ACTIONS),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    next_state = state.copy()
    next_state.prev_action = act.copy()

    if int(round(float(act[7]))) == 1:
        if obs is not None and next_state.use_cooldown == 0 and not state.use_consumed:
            r_dist = np.clip(
                np.nan_to_num(np.asarray(obs["rays"]["distance"], dtype=np.float32), nan=1.0, posinf=1.0, neginf=0.0),
                0.0,
                1.0,
            )
            r_type = np.nan_to_num(np.asarray(obs["rays"]["hit_type"]), nan=0.0).astype(np.int64)
            idx = np.asarray(FRONT_RAY_INDICES, dtype=np.int64)
            front_door = bool(np.any(((r_type[idx] == 2) | (r_type[idx] == 3)) & (r_dist[idx] < 0.15)))
            if front_door:
                next_state.use_cooldown = 6
                next_state.cooldown_pre_armed = True
                next_state.use_consumed = True
    else:
        next_state.use_consumed = False

    return next_state
