"""360° Ray Scan-Matching Odometry & Egocentric Occupancy Grid Builder.

Implements the specification in Design.md §2.2:
- Circular cross-correlation scan matching over 64 360° rays to estimate yaw rotation (d_theta).
- Ego velocity integration (vel_forward, vel_right) for translation (dx, dy).
- Affine warping of a 32x32 6-channel semantic occupancy grid centered on the player.
- Ray endpoint splatting into semantic channels (visited, wall, door, locked_door, item, enemy).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
import numpy as np
import cv2

# Grid settings from Design.md / Schema.md
GRID_SIZE = 32
GRID_CHANNELS = 6
CELL_SIZE = 48.0  # map units per grid cell
RAY_RANGE = 1024.0
NUM_RAYS = 64
RAY_ANGLE_STEP = 2.0 * math.pi / NUM_RAYS  # 5.625 degrees per ray bin

# Channel definitions
CH_VISITED = 0
CH_WALL = 1
CH_DOOR = 2
CH_LOCKED_DOOR = 3
CH_ITEM = 4
CH_ENEMY = 5


@dataclass
class OdomGridState:
    grid: np.ndarray = field(
        default_factory=lambda: np.zeros((GRID_CHANNELS, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    )
    prev_ray_dist: np.ndarray | None = None
    prev_ray_type: np.ndarray | None = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(9, dtype=np.int8))
    use_cooldown: int = 0
    d_theta: float = 0.0


def estimate_yaw_rotation(
    curr_dist: np.ndarray,
    curr_type: np.ndarray,
    prev_dist: np.ndarray | None,
    prev_action: np.ndarray | None,
) -> float:
    """Estimates yaw rotation d_theta (radians, positive CCW / turning left)

    from consecutive 64-ray 360° scans and previous action prior.
    """
    if prev_dist is None:
        return 0.0

    # Prior shift from turn buttons: index 4 = TURN_LEFT, index 5 = TURN_RIGHT
    k_prior = 0
    if prev_action is not None and len(prev_action) > 5:
        k_prior = int(prev_action[4]) - int(prev_action[5])

    # Search window around prior shift
    search_radius = 4
    candidates = range(k_prior - search_radius, k_prior + search_radius + 1)

    costs = []
    # Mask rays that hit static geometry (walls/doors/locked_doors: types 1, 2, 3)
    curr_valid = (curr_dist < 0.98) & (curr_type >= 1) & (curr_type <= 3)

    for k in candidates:
        shifted_prev = np.roll(prev_dist, k)
        valid = curr_valid & (shifted_prev < 0.98)

        if np.sum(valid) >= 6:
            diff = np.abs(curr_dist[valid] - shifted_prev[valid])
            cost = float(np.mean(diff))
        else:
            # Fallback if too few rays hit geometry
            cost = 1.0

        # Small quadratic prior as a tie-breaker when rays are uniform
        cost += 0.002 * float((k - k_prior) ** 2)
        costs.append(cost)

    best_idx = int(np.argmin(costs))
    k_star = candidates[best_idx]

    # 3-point parabolic interpolation for sub-bin continuous accuracy
    delta = 0.0
    if 0 < best_idx < len(costs) - 1:
        c_prev, c_curr, c_next = costs[best_idx - 1], costs[best_idx], costs[best_idx + 1]
        denom = c_prev - 2.0 * c_curr + c_next
        if abs(denom) > 1e-6:
            delta = 0.5 * (c_prev - c_next) / denom
            delta = float(np.clip(delta, -0.5, 0.5))

    d_theta = (float(k_star) + delta) * RAY_ANGLE_STEP
    return d_theta


def update_odometry_and_grid(
    state: OdomGridState,
    obs: dict,
    frame_skip: int = 4,
) -> tuple[np.ndarray, float]:
    """Updates the egocentric occupancy grid and estimates d_theta for one step.

    Returns:
        (updated_grid, d_theta)
    """
    vitals = obs.get("vitals", np.zeros(17, dtype=np.float32))
    rays = obs.get("rays", {})
    curr_dist = rays.get("distance", np.ones(NUM_RAYS, dtype=np.float32))
    curr_type = rays.get("hit_type", np.zeros(NUM_RAYS, dtype=np.int64))

    # 1. Yaw rotation estimation (d_theta)
    d_theta = estimate_yaw_rotation(
        curr_dist=curr_dist,
        curr_type=curr_type,
        prev_dist=state.prev_ray_dist,
        prev_action=state.prev_action,
    )
    state.d_theta = d_theta

    # 2. Ego translation: vel_forward, vel_right (indices 14, 15 in vitals)
    vel_forward = float(vitals[14]) if len(vitals) > 14 else 0.0
    vel_right = float(vitals[15]) if len(vitals) > 15 else 0.0

    # Displacement in map units over frame_skip game tics
    dx_units = vel_forward * frame_skip
    dy_units = vel_right * frame_skip

    # In grid cells (center at 16, 16):
    # Forward is -row in image frame (up is row 0), Right is +col (right is col 31)
    dx_cells = dx_units / CELL_SIZE
    dy_cells = dy_units / CELL_SIZE

    # 3. Grid Decay
    grid = state.grid
    grid[CH_WALL] *= 0.992
    grid[CH_DOOR] *= 0.992
    grid[CH_ITEM] *= 0.992
    grid[CH_ENEMY] *= 0.950  # Enemies move, decay fast
    grid[CH_VISITED] *= 0.998  # Visited locations remembered longer
    grid[CH_LOCKED_DOOR] *= 0.998  # Locked doors retained across long exploration

    # 4. Affine Warp into new egocentric frame
    # Rotation angle in degrees: when player rotates CCW (+d_theta), world rotates CW (-d_theta)
    center = (GRID_SIZE / 2.0, GRID_SIZE / 2.0)
    angle_deg = -math.degrees(d_theta)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    # Add translation to transformation matrix:
    # Ego forward motion moves past obstacles down (+row direction), ego right motion moves obstacles left (-col direction)
    M[0, 2] -= dy_cells
    M[1, 2] += dx_cells

    # Warp each channel using OpenCV affine transform
    # cv2 expects (H, W, C) for multi-channel warp
    grid_hwc = np.transpose(grid, (1, 2, 0))
    warped_hwc = cv2.warpAffine(
        grid_hwc,
        M,
        (GRID_SIZE, GRID_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    grid = np.transpose(warped_hwc, (2, 0, 1)).astype(np.float32)

    # 5. Splat Current Ray Endpoints
    center_idx = GRID_SIZE // 2  # 16
    for i in range(NUM_RAYS):
        dist = float(curr_dist[i]) * RAY_RANGE
        hit = int(curr_type[i])
        if dist >= RAY_RANGE * 0.98 or hit <= 0:
            continue

        angle_i = i * RAY_ANGLE_STEP
        # In player coordinate system: forward is +x, left is +y
        forward_dist = dist * math.cos(angle_i)
        left_dist = dist * math.sin(angle_i)

        # Map to grid: row = center - forward / CELL_SIZE, col = center - left / CELL_SIZE
        row = int(round(center_idx - forward_dist / CELL_SIZE))
        col = int(round(center_idx - left_dist / CELL_SIZE))

        if 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE:
            if hit == 1:  # wall
                grid[CH_WALL, row, col] = 1.0
            elif hit == 2:  # door
                grid[CH_DOOR, row, col] = 1.0
            elif hit == 3:  # locked_door
                grid[CH_LOCKED_DOOR, row, col] = 1.0
            elif hit == 4:  # enemy
                grid[CH_ENEMY, row, col] = 1.0
            elif hit == 5:  # item
                grid[CH_ITEM, row, col] = 1.0

    # Mark current position as visited
    grid[CH_VISITED, center_idx, center_idx] = 1.0

    # Save state for next step
    state.grid = grid
    state.prev_ray_dist = curr_dist.copy()
    state.prev_ray_type = curr_type.copy()

    return grid, d_theta
