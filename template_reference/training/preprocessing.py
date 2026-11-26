"""Observation Preprocessing and Legal Action Masking for DOOM Agent.

Implements the contract defined in:
- docs/CONTRACT.md
- planning/Schema.md §2
- planning/Design.md §2

Exports:
- ModelInputs: Dataclass of batched PyTorch tensors for HybridDoomNet.
- OdomGridState: Per-environment tracker for odometry grid and action cooldowns.
- preprocess: Converts raw doom_env observation dicts into ModelInputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch
import cv2

from .odometry_grid import (
    OdomGridState,
    update_odometry_and_grid,
    NUM_RAYS,
    GRID_SIZE,
    GRID_CHANNELS,
)

# Constants from Schema.md
RGB_HEIGHT = 48
RGB_WIDTH = 64
MAX_ENTITIES = 32
NUM_VITALS_RAW = 17
NUM_VITALS_PREPROCESSED = 36
NUM_ACTIONS = 9

# Front ray indices for door USE cooldown check (360° circular rays: 0 is straight ahead)
FRONT_RAY_INDICES = (60, 61, 62, 63, 0, 1, 2, 3, 4)
DOOR_HIT_TYPES = (2, 3)  # 2: door, 3: locked_door


@dataclass
class ModelInputs:
    rays: torch.Tensor          # (B, 8, 64) float32
    ent_feats: torch.Tensor     # (B, 32, 5) float32
    ent_types: torch.Tensor     # (B, 32) int64
    ent_mask: torch.Tensor      # (B, 32) bool
    vitals: torch.Tensor        # (B, 36) float32
    rgb_depth: torch.Tensor     # (B, 4, 48, 64) float32
    grid: torch.Tensor          # (B, 6, 32, 32) float32
    action_mask: torch.Tensor   # (B, 9) bool
    d_theta: torch.Tensor       # (B,) float32


def preprocess_rays(rays_dict: dict) -> np.ndarray:
    """Packs 64 rays into (8, 64) array:

    - Channel 0: distance [0, 1]
    - Channel 1: proximity (1.0 - distance)
    - Channels 2..7: 6-way one-hot encoding of hit_type {0..5}
    """
    dist = rays_dict.get("distance", np.ones(NUM_RAYS, dtype=np.float32)).astype(np.float32)
    hit_type = rays_dict.get("hit_type", np.zeros(NUM_RAYS, dtype=np.int64)).astype(np.int64)

    proximity = 1.0 - dist
    hit_one_hot = np.zeros((6, NUM_RAYS), dtype=np.float32)
    valid_mask = (hit_type >= 0) & (hit_type < 6)
    hit_one_hot[hit_type[valid_mask], np.where(valid_mask)[0]] = 1.0

    return np.vstack([dist[np.newaxis, :], proximity[np.newaxis, :], hit_one_hot])


def preprocess_entities(ent_dict: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Packs visible entities into features (32, 5), types (32,), mask (32,)."""
    feats = ent_dict.get("features", np.zeros((MAX_ENTITIES, 5), dtype=np.float32)).copy()
    types = ent_dict.get("entity_type", np.zeros(MAX_ENTITIES, dtype=np.int64)).copy()
    mask = ent_dict.get("mask", np.zeros(MAX_ENTITIES, dtype=np.int8)) > 0

    # Normalize relative velocities (indices 3, 4) by / 30.0, clipped to [-2.0, 2.0]
    if feats.shape[-1] >= 5:
        feats[:, 3] = np.clip(feats[:, 3] / 30.0, -2.0, 2.0)
        feats[:, 4] = np.clip(feats[:, 4] / 30.0, -2.0, 2.0)

    return feats.astype(np.float32), types.astype(np.int64), mask.astype(bool)


def preprocess_vitals(
    vitals: np.ndarray,
    state: OdomGridState,
) -> np.ndarray:
    """Preprocesses 17 scalar vitals + weapon one-hot + context into a 36-dim vector."""
    out = np.zeros(NUM_VITALS_PREPROCESSED, dtype=np.float32)

    if len(vitals) >= 17:
        # 0: health, 1: armor (/ 100.0)
        out[0] = float(np.clip(vitals[0] / 100.0, 0.0, 2.0))
        out[1] = float(np.clip(vitals[1] / 100.0, 0.0, 2.0))

        # 2..9: selected_weapon one-hot (8 slots: 0..7)
        w_slot = int(np.clip(vitals[2], 0, 7))
        out[2 + w_slot] = 1.0

        # 10..14: ammo slots (indices 3..7 in raw vitals)
        for i in range(5):
            ammo_val = float(np.clip(vitals[3 + i], 0.0, 500.0))
            out[10 + i] = float(np.log1p(ammo_val) / 6.2)

        # 15..20: 6 key flags (indices 8..13 in raw vitals)
        out[15:21] = vitals[8:14].astype(np.float32)

        # 21..23: velocities (indices 14..16 in raw vitals)
        out[21] = float(np.clip(vitals[14] / 30.0, -2.0, 2.0))
        out[22] = float(np.clip(vitals[15] / 30.0, -2.0, 2.0))
        out[23] = float(np.clip(vitals[16] / 30.0, -2.0, 2.0))

        # Context features
        out[24] = float(state.use_cooldown / 6.0)
        out[25] = 1.0 if np.any(vitals[8:14] > 0.5) else 0.0
        out[26] = 1.0 if (vitals[2] > 1 and vitals[3] <= 5.0) else 0.0

    # 27..35: previous action (9 buttons)
    if state.prev_action is not None and len(state.prev_action) >= 9:
        out[27:36] = state.prev_action[:9].astype(np.float32)

    return out


def preprocess_rgb_depth(obs: dict) -> np.ndarray:
    """Downsamples RGB (120x160x3) and Depth (120x160) to (4, 48, 64)."""
    rgb = obs.get("rgb")
    depth = obs.get("depth")

    out = np.zeros((4, RGB_HEIGHT, RGB_WIDTH), dtype=np.float32)

    if rgb is not None and rgb.shape[0] == 120 and rgb.shape[1] == 160:
        rgb_resized = cv2.resize(rgb, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_LINEAR)
        # Transpose HWC -> CHW and normalize to [0, 1]
        out[0:3] = np.transpose(rgb_resized, (2, 0, 1)).astype(np.float32) / 255.0

    if depth is not None and depth.shape[0] == 120 and depth.shape[1] == 160:
        depth_resized = cv2.resize(depth, (RGB_WIDTH, RGB_HEIGHT), interpolation=cv2.INTER_LINEAR)
        out[3] = depth_resized.astype(np.float32) / 255.0

    return out


def compute_action_mask(
    obs: dict,
    state: OdomGridState,
) -> np.ndarray:
    """Computes observation-legal action mask of shape (9,) bool.

    - USE (index 7): disabled while use_cooldown > 0 to prevent door-reversing bounces.
    - ATTACK (index 6): disabled when selected_weapon > 1 (not melee) and ammo <= 0.
    """
    mask = np.ones(NUM_ACTIONS, dtype=bool)

    # 1. Door USE cooldown
    if state.use_cooldown > 0:
        mask[7] = False

    # 2. Ammo ATTACK mask
    vitals = obs.get("vitals", np.zeros(NUM_VITALS_RAW, dtype=np.float32))
    if len(vitals) >= 4:
        weapon = int(vitals[2])
        ammo = float(vitals[3])
        # Weapons: 0: fist, 1: chainsaw (infinite melee ammo). Weapons > 1 consume ammo.
        if weapon > 1 and ammo <= 0.0:
            mask[6] = False

    return mask


def update_use_cooldown(
    obs: dict,
    state: OdomGridState,
) -> None:
    """Updates the door-reversing cooldown tracker."""
    # Decrement cooldown counter if active
    if state.use_cooldown > 0:
        state.use_cooldown -= 1

    # If USE was pressed in the previous step
    if state.prev_action is not None and len(state.prev_action) > 7 and state.prev_action[7] == 1:
        rays = obs.get("rays", {})
        dists = rays.get("distance", np.ones(NUM_RAYS, dtype=np.float32))
        types = rays.get("hit_type", np.zeros(NUM_RAYS, dtype=np.int64))

        # Check if front rays hit a door or locked door close by (~150 units)
        for idx in FRONT_RAY_INDICES:
            if int(types[idx]) in DOOR_HIT_TYPES and float(dists[idx]) < 0.15:
                # Trigger a 6-step cooldown (~0.7 s at 8.75 steps/s) so door opens smoothly
                state.use_cooldown = 6
                break


def preprocess(
    obs: dict,
    state: OdomGridState | None = None,
    device: str = "cpu",
    use_grid: bool = True,
) -> tuple[ModelInputs, OdomGridState]:
    """Single-observation preprocessor entrypoint.

    Converts raw observation dict into ModelInputs on specified device.
    """
    if state is None:
        state = OdomGridState()

    # 1. Update cooldown from previous action
    update_use_cooldown(obs, state)

    # 2. Update odometry and egocentric occupancy grid
    if use_grid:
        grid, d_theta = update_odometry_and_grid(state, obs)
    else:
        grid = np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
        d_theta = 0.0

    # 3. Preprocess sensory streams
    rays_arr = preprocess_rays(obs.get("rays", {}))
    ent_feats, ent_types, ent_mask = preprocess_entities(obs.get("visible_entities", {}))
    vitals_arr = preprocess_vitals(obs.get("vitals", np.zeros(NUM_VITALS_RAW, dtype=np.float32)), state)
    rgb_depth_arr = preprocess_rgb_depth(obs)
    action_mask_arr = compute_action_mask(obs, state)

    # 4. Pack into batched PyTorch tensors (batch size 1)
    dev = torch.device(device)
    inputs = ModelInputs(
        rays=torch.from_numpy(rays_arr).unsqueeze(0).to(dev),
        ent_feats=torch.from_numpy(ent_feats).unsqueeze(0).to(dev),
        ent_types=torch.from_numpy(ent_types).unsqueeze(0).to(dev),
        ent_mask=torch.from_numpy(ent_mask).unsqueeze(0).to(dev),
        vitals=torch.from_numpy(vitals_arr).unsqueeze(0).to(dev),
        rgb_depth=torch.from_numpy(rgb_depth_arr).unsqueeze(0).to(dev),
        grid=torch.from_numpy(grid).unsqueeze(0).to(dev),
        action_mask=torch.from_numpy(action_mask_arr).unsqueeze(0).to(dev),
        d_theta=torch.tensor([d_theta], dtype=torch.float32, device=dev),
    )

    return inputs, state
