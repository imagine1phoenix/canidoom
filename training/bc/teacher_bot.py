"""Training-only Privileged Teacher Bot for BC / DAgger distillation.

STRICTLY TRAINING-ONLY — NEVER imported by `submission/agent.py`.
Uses `env.privileged_state()`, `env.sector_polygons()`, and UDMF `TEXTMAP`
geometry to compute shortest-path navigation (keys -> locked doors -> exit),
door interaction (`USE`), and enemy hitbox aiming/strafe-dodging across
procedurally generated Tier 1–4 maps.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Doom UDMF keycard/skull thing types -> priv["keys"] index (0..5)
KEY_THING_TYPES: dict[int, int] = {
    5: 0,   # BlueCard
    6: 1,   # YellowCard
    13: 2,  # RedCard
    40: 3,  # BlueSkull
    39: 4,  # YellowSkull
    38: 5,  # RedSkull
}


@dataclass
class MapLayout:
    exit_pos: tuple[float, float]
    exit_is_use: bool
    keys: list[tuple[float, float, int]]  # (x, y, key_idx)
    doors: list[tuple[float, float, int]]  # (x, y, lock_id)
    sectors_aabb: list[tuple[int, float, float, float, float]]  # (sid, xmin, xmax, ymin, ymax)
    grid_origin: tuple[float, float]
    grid_res: float
    grid_w: int
    grid_h: int
    walkable: np.ndarray  # (H, W) bool


def _parse_udmf_objectives(wad_path: Path) -> tuple[tuple[float, float], bool, list[tuple[float, float, int]], list[tuple[float, float, int]]]:
    raw = wad_path.read_bytes()
    idx = raw.find(b"namespace")
    if idx < 0:
        return (0.0, 0.0), False, [], []
    text = raw[idx:].decode("latin1", errors="ignore")

    verts: list[tuple[float, float]] = []
    for m in re.finditer(r"vertex\s*\{([^}]*)\}", text):
        blk = m.group(1)
        xm = re.search(r"\bx\s*=\s*([-+]?\d+(?:\.\d+)?)", blk)
        ym = re.search(r"\by\s*=\s*([-+]?\d+(?:\.\d+)?)", blk)
        if xm and ym:
            verts.append((float(xm.group(1)), float(ym.group(1))))

    keys: list[tuple[float, float, int]] = []
    for m in re.finditer(r"thing\s*\{([^}]*)\}", text):
        blk = m.group(1)
        tm = re.search(r"\btype\s*=\s*(\d+)", blk)
        if not tm:
            continue
        t_id = int(tm.group(1))
        if t_id in KEY_THING_TYPES:
            xm = re.search(r"\bx\s*=\s*([-+]?\d+(?:\.\d+)?)", blk)
            ym = re.search(r"\by\s*=\s*([-+]?\d+(?:\.\d+)?)", blk)
            if xm and ym:
                keys.append((float(xm.group(1)), float(ym.group(1)), KEY_THING_TYPES[t_id]))

    exit_pts: list[tuple[float, float]] = []
    exit_is_use = False
    doors: list[tuple[float, float, int]] = []

    for m in re.finditer(r"linedef\s*\{([^}]*)\}", text):
        blk = m.group(1)
        sm = re.search(r"\bspecial\s*=\s*(\d+)", blk)
        if not sm:
            continue
        sp = int(sm.group(1))
        v1m = re.search(r"\bv1\s*=\s*(\d+)", blk)
        v2m = re.search(r"\bv2\s*=\s*(\d+)", blk)
        if not (v1m and v2m):
            continue
        v1, v2 = int(v1m.group(1)), int(v2m.group(1))
        if not (0 <= v1 < len(verts) and 0 <= v2 < len(verts)):
            continue
        mx = 0.5 * (verts[v1][0] + verts[v2][0])
        my = 0.5 * (verts[v1][1] + verts[v2][1])
        if sp in (243, 11):
            exit_pts.append((mx, my))
            if "playeruse" in blk.lower():
                exit_is_use = True
        elif sp in (12, 13, 202):
            am = re.search(r"\barg3\s*=\s*(\d+)", blk)
            lock_id = int(am.group(1)) if am else 0
            doors.append((mx, my, lock_id))

    if exit_pts:
        ex = float(np.mean([p[0] for p in exit_pts]))
        ey = float(np.mean([p[1] for p in exit_pts]))
        exit_pos = (ex, ey)
    else:
        exit_pos = (0.0, 0.0)

    return exit_pos, exit_is_use, keys, doors


def build_map_layout(wad_path: str | Path, sector_polygons: list[tuple[int, np.ndarray]], res: float = 24.0) -> MapLayout:
    exit_pos, exit_is_use, keys, doors = _parse_udmf_objectives(Path(wad_path))

    sectors_aabb: list[tuple[int, float, float, float, float]] = []
    xs: list[float] = []
    ys: list[float] = []
    for sid, segs_raw in sector_polygons:
        segs = np.asarray(segs_raw, dtype=np.float64)
        if segs.ndim != 2 or segs.shape[1] != 4 or len(segs) == 0:
            continue
        xmin = float(min(segs[:, 0].min(), segs[:, 2].min()))
        xmax = float(max(segs[:, 0].max(), segs[:, 2].max()))
        ymin = float(min(segs[:, 1].min(), segs[:, 3].min()))
        ymax = float(max(segs[:, 1].max(), segs[:, 3].max()))
        sectors_aabb.append((int(sid), xmin, xmax, ymin, ymax))
        xs.extend([xmin, xmax])
        ys.extend([ymin, ymax])

    if not xs:
        return MapLayout(exit_pos, exit_is_use, keys, doors, [], (0.0, 0.0), res, 1, 1, np.ones((1, 1), dtype=bool))

    ox = math.floor(min(xs) / res) * res - res
    oy = math.floor(min(ys) / res) * res - res
    gx_max = math.ceil(max(xs) / res) * res + res
    gy_max = math.ceil(max(ys) / res) * res + res
    gw = max(1, int(round((gx_max - ox) / res)) + 1)
    gh = max(1, int(round((gy_max - oy) / res)) + 1)
    walkable = np.zeros((gh, gw), dtype=bool)

    # Mark interior of each rectangular/polygonal sector walkable (small margin keeps corridors connected)
    margin = 6.0
    for _, xmin, xmax, ymin, ymax in sectors_aabb:
        ix0 = max(0, int(math.floor((xmin + margin - ox) / res)))
        ix1 = min(gw - 1, int(math.ceil((xmax - margin - ox) / res)))
        iy0 = max(0, int(math.floor((ymin + margin - oy) / res)))
        iy1 = min(gh - 1, int(math.ceil((ymax - margin - oy) / res)))
        if ix1 < ix0:
            ix0 = ix1 = max(0, min(gw - 1, int(round((0.5 * (xmin + xmax) - ox) / res))))
        if iy1 < iy0:
            iy0 = iy1 = max(0, min(gh - 1, int(round((0.5 * (ymin + ymax) - oy) / res))))
        walkable[iy0 : iy1 + 1, ix0 : ix1 + 1] = True

    # Ensure doors, keys, and exit are walkable cells
    for px, py in [exit_pos] + [(k[0], k[1]) for k in keys] + [(d[0], d[1]) for d in doors]:
        ix = max(0, min(gw - 1, int(round((px - ox) / res))))
        iy = max(0, min(gh - 1, int(round((py - oy) / res))))
        walkable[max(0, iy - 1) : min(gh, iy + 2), max(0, ix - 1) : min(gw, ix + 2)] = True

    return MapLayout(exit_pos, exit_is_use, keys, doors, sectors_aabb, (ox, oy), res, gw, gh, walkable)


def _astar(
    layout: MapLayout,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    blocked_points: Optional[list[tuple[float, float]]] = None,
) -> list[tuple[float, float]]:
    ox, oy = layout.grid_origin
    res = layout.grid_res
    gw, gh = layout.grid_w, layout.grid_h

    def to_cell(x: float, y: float) -> tuple[int, int]:
        return (
            max(0, min(gw - 1, int(round((x - ox) / res)))),
            max(0, min(gh - 1, int(round((y - oy) / res)))),
        )

    def to_world(ix: int, iy: int) -> tuple[float, float]:
        return (ox + ix * res, oy + iy * res)

    sx, sy = to_cell(*start_xy)
    gx, gy = to_cell(*goal_xy)
    if (sx, sy) == (gx, gy):
        return [goal_xy]

    blocked_cells: set[tuple[int, int]] = set()
    for bx, by in (blocked_points or []):
        cx, cy = to_cell(bx, by)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                blocked_cells.add((cx + dx, cy + dy))
    blocked_cells.discard((sx, sy))
    blocked_cells.discard((gx, gy))

    pq: list[tuple[float, float, int, int]] = [(0.0, 0.0, sx, sy)]
    came: dict[tuple[int, int], tuple[int, int]] = {}
    g_cost: dict[tuple[int, int], float] = {(sx, sy): 0.0}
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
            (-1, -1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (1, 1, 1.414)]

    found = False
    while pq:
        _, g, cx, cy = heapq.heappop(pq)
        if (cx, cy) == (gx, gy):
            found = True
            break
        if g > g_cost.get((cx, cy), 1e18):
            continue
        for dx, dy, step_c in nbrs:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < gw and 0 <= ny < gh):
                continue
            if (nx, ny) in blocked_cells:
                continue
            if not layout.walkable[ny, nx] and (nx, ny) != (gx, gy):
                continue
            ng = g + step_c
            if ng < g_cost.get((nx, ny), 1e18):
                g_cost[(nx, ny)] = ng
                came[(nx, ny)] = (cx, cy)
                h = math.hypot(gx - nx, gy - ny)
                heapq.heappush(pq, (ng + h, ng, nx, ny))

    if not found:
        return [goal_xy]

    path: list[tuple[float, float]] = []
    cur = (gx, gy)
    while cur != (sx, sy):
        path.append(to_world(*cur))
        cur = came[cur]
    path.reverse()
    path.append(goal_xy)
    return path


class PrivilegedTeacherBot:
    """Training-only scripted expert that solves Tier 1–4 procedural maps cleanly."""

    def __init__(self, wad_path: str | Path, sector_polygons: list[tuple[int, np.ndarray]]) -> None:
        self.layout = build_map_layout(wad_path, sector_polygons)
        self._step = 0
        self._use_cooldown = 0
        self._last_xy: Optional[tuple[float, float]] = None
        self._stuck_count = 0
        self._unstuck_dir = 1
        self._unstuck_timer = 0
        self._cached_path: list[tuple[float, float]] = []
        self._replan_every = 4

    def act(self, obs: dict[str, Any], priv: dict[str, Any]) -> np.ndarray:
        """Return MultiBinary(9) action: [FWD, BAK, L, R, TL, TR, ATK, USE, WPN]."""
        act = np.zeros(9, dtype=np.int8)
        self._step += 1
        if self._use_cooldown > 0:
            self._use_cooldown -= 1

        px = float(priv.get("x", 0.0))
        py = float(priv.get("y", 0.0))
        ang_deg = float(priv.get("angle", 0.0))
        keys_held = np.asarray(priv.get("keys", np.zeros(6)), dtype=np.float32)

        # Track stuckness
        if self._last_xy is not None:
            disp = math.hypot(px - self._last_xy[0], py - self._last_xy[1])
            if disp < 2.2:
                self._stuck_count += 1
            else:
                self._stuck_count = max(0, self._stuck_count - 1)
        self._last_xy = (px, py)

        if self._stuck_count >= 5:
            self._unstuck_timer = 6
            self._unstuck_dir = -self._unstuck_dir
            self._stuck_count = 0

        # 1. Check visible enemies from obs["visible_entities"]
        ent_feats = obs["visible_entities"]["features"]
        ent_types = obs["visible_entities"]["entity_type"]
        ent_mask = obs["visible_entities"]["mask"]
        ammo = float(obs["vitals"][2]) if len(obs["vitals"]) > 2 else 1.0

        best_enemy = None  # (dist, rel_angle_rad)
        for i in range(len(ent_mask)):
            if not ent_mask[i]:
                continue
            etype = int(ent_types[i])
            if etype in (1, 2, 3, 4):  # enemy types
                d = float(ent_feats[i, 0])
                sin_a = float(ent_feats[i, 1])
                cos_a = float(ent_feats[i, 2])
                rel_ang = math.atan2(sin_a, cos_a)  # +left, -right
                if best_enemy is None or d < best_enemy[0]:
                    best_enemy = (d, rel_ang)

        # 2. Pick current navigation goal: uncollected keycard first, else exit
        # Map UDMF lock_id (1=red, 2=blue, 3=yellow, 129..131) -> keys_held check
        def _has_lock_key(lock_id: int) -> bool:
            if lock_id <= 0:
                return True
            if lock_id in (2, 130):
                return float(keys_held[0]) > 0.5 or float(keys_held[3]) > 0.5
            if lock_id in (3, 131):
                return float(keys_held[1]) > 0.5 or float(keys_held[4]) > 0.5
            if lock_id in (1, 129):
                return float(keys_held[2]) > 0.5 or float(keys_held[5]) > 0.5
            return float(np.sum(keys_held)) > 0.5

        blocked_doors = [(dx, dy) for (dx, dy, lid) in self.layout.doors if not _has_lock_key(lid)]

        uncollected_keys = [
            (kx, ky) for (kx, ky, kidx) in self.layout.keys
            if kidx < len(keys_held) and float(keys_held[kidx]) < 0.5
        ]
        if uncollected_keys:
            goal = min(uncollected_keys, key=lambda k: math.hypot(k[0] - px, k[1] - py))
        else:
            goal = self.layout.exit_pos

        if self._step % self._replan_every == 1 or not self._cached_path:
            self._cached_path = _astar(self.layout, (px, py), goal, blocked_points=blocked_doors)

        # Advance along waypoints
        target = goal
        for wp in self._cached_path:
            if math.hypot(wp[0] - px, wp[1] - py) > 36.0:
                target = wp
                break

        # Compute bearing to target in player frame (+left, -right)
        desired_deg = math.degrees(math.atan2(target[1] - py, target[0] - px))
        diff_deg = (desired_deg - ang_deg + 180.0) % 360.0 - 180.0

        # 3. Combat override only if enemy is close & threatening, else shoot on the move
        engaged_combat = False
        if best_enemy is not None and ammo > 0.0:
            edist, e_ang = best_enemy
            if edist < 0.36:
                engaged_combat = True
                if e_ang > 0.10:
                    act[4] = 1  # TURN_LEFT
                elif e_ang < -0.10:
                    act[5] = 1  # TURN_RIGHT
                if abs(e_ang) < 0.24:
                    act[6] = 1  # ATTACK
                # Strafe dodge while shooting
                act[2 if (self._step // 8) % 2 == 0 else 3] = 1
                if edist < 0.12:
                    act[1] = 1
            elif edist < 0.58 and abs(e_ang) < 0.22:
                act[6] = 1  # opportunistic shot while continuing navigation

        # 4. Navigation steering when not locked in close combat
        if not engaged_combat:
            if diff_deg > 10.0:
                act[4] = 1  # TURN_LEFT
            elif diff_deg < -10.0:
                act[5] = 1  # TURN_RIGHT
            if abs(diff_deg) < 60.0:
                act[0] = 1  # MOVE_FORWARD

            rays_d = obs["rays"]["distance"]
            front_d = float(min(rays_d[0], rays_d[1], rays_d[63]))
            left_d = float(np.mean(rays_d[3:8]))
            right_d = float(np.mean(rays_d[56:61]))
            if front_d < 0.055:
                act[2 if left_d > right_d else 3] = 1

        # 5. Unstuck recovery (sidestep + forward)
        if self._unstuck_timer > 0:
            self._unstuck_timer -= 1
            act[2 if self._unstuck_dir > 0 else 3] = 1
            act[0] = 1

        # 6. Door & Exit USE interaction — ONLY when a door ray is directly ahead (closed) or exit switch
        rays_d = obs["rays"]["distance"]
        rays_t = obs["rays"]["hit_type"]
        front_ray_idxs = (0, 1, 2, 62, 63)
        door_ray_ahead = any(
            int(rays_t[idx]) in (2, 3) and float(rays_d[idx]) < 0.14
            for idx in front_ray_idxs
        )
        near_exit_use = self.layout.exit_is_use and (
            math.hypot(self.layout.exit_pos[0] - px, self.layout.exit_pos[1] - py) < 72.0
        )
        if self._use_cooldown == 0 and (door_ray_ahead or near_exit_use):
            act[7] = 1  # USE
            self._use_cooldown = 14

        return act
