"""Coordinate-free dense training reward shaper and 8-sector ego-directional compass attribution.

Uses ONLY relative deltas from `env.privileged_state()` and translation-invariant
point-in-polygon lookups on `env.sector_polygons()`. Zero fixed world XYZ assumptions.
Matches planning/Schema.md §2.4 and Design.md §3, and supports both dict-delta and DoomEnv callers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np


@dataclass
class RewardWeights:
    exit_base: float = 15.0
    exit_speed_bonus: float = 3.0
    new_sector: float = 0.40
    key_pickup: float = 4.00
    door_open: float = 1.20
    locked_door_unlock: float = 5.00
    kill: float = 1.00
    item_pickup: float = 0.15
    damage_per_hp: float = -0.02
    death: float = -4.00
    door_reversal_bounce: float = -0.30
    stuck_wall_scrape: float = -0.06
    wasted_shot: float = -0.015
    step_cost: float = -0.002
    # Anti-loop / exploration shaping (training-only, relative displacement)
    new_cell: float = 0.06
    loop_penalty: float = -0.03
    conflict_buttons: float = -0.01
    backward_move: float = -0.004
    forward_progress: float = 0.004


def _point_in_segments(x: float, y: float, segs: np.ndarray) -> bool:
    """Even-odd horizontal ray-casting test on unordered (K, 4) boundary segments [x1, y1, x2, y2]."""
    if len(segs) == 0:
        return False
    x1 = segs[:, 0]
    y1 = segs[:, 1]
    x2 = segs[:, 2]
    y2 = segs[:, 3]
    cond = (y1 > y) != (y2 > y)
    if not np.any(cond):
        return False
    x_intersect = (x2[cond] - x1[cond]) * (y - y1[cond]) / (y2[cond] - y1[cond]) + x1[cond]
    crossings = int(np.sum(x < x_intersect))
    return (crossings % 2) == 1


def _polygon_area_from_segments(segs: np.ndarray, aabb_area: float) -> float:
    """Compute rotation-invariant polygon area via shoelace over traced boundary rings.

    Falls back to `aabb_area` only when segments do not form closed polygon ring(s).
    """
    n_segs = len(segs)
    if n_segs < 3:
        return aabb_area

    def _vkey(x: float, y: float) -> tuple[float, float]:
        return (round(float(x), 6), round(float(y), 6))

    adj: dict[tuple[float, float], list[tuple[tuple[float, float], int, tuple[float, float], tuple[float, float]]]] = {}
    keys: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(n_segs):
        x1, y1, x2, y2 = float(segs[i, 0]), float(segs[i, 1]), float(segs[i, 2]), float(segs[i, 3])
        u = _vkey(x1, y1)
        v = _vkey(x2, y2)
        keys.append((u, v))
        if u == v:
            continue
        adj.setdefault(u, []).append((v, i, (x1, y1), (x2, y2)))
        adj.setdefault(v, []).append((u, i, (x2, y2), (x1, y1)))

    used = [False] * n_segs
    rings: list[tuple[float, tuple[float, float], np.ndarray]] = []
    all_closed = True

    for i in range(n_segs):
        if used[i]:
            continue
        u, v = keys[i]
        used[i] = True
        if u == v:
            continue
        p1 = (float(segs[i, 0]), float(segs[i, 1]))
        p2 = (float(segs[i, 2]), float(segs[i, 3]))
        pts: list[tuple[float, float]] = [p1, p2]
        ring_segs: list[list[float]] = [[p1[0], p1[1], p2[0], p2[1]]]
        cur = v
        closed = False
        while True:
            if cur == u:
                closed = True
                break
            nxt_edge = None
            for cand in adj.get(cur, ()):
                if not used[cand[1]]:
                    nxt_edge = cand
                    break
            if nxt_edge is None:
                break
            nxt_k, j, p_cur, p_nxt = nxt_edge
            used[j] = True
            pts.append(p_nxt)
            ring_segs.append([p_cur[0], p_cur[1], p_nxt[0], p_nxt[1]])
            cur = nxt_k

        if not closed or len(pts) < 4:
            all_closed = False
            break

        arr = np.asarray(pts[:-1], dtype=np.float64)
        cx = float(arr[:, 0].mean())
        cy = float(arr[:, 1].mean())
        rx = arr[:, 0] - cx
        ry = arr[:, 1] - cy
        ring_area = 0.5 * abs(float(np.sum(rx * np.roll(ry, -1) - np.roll(rx, -1) * ry)))
        rings.append((ring_area, pts[0], np.asarray(ring_segs, dtype=np.float64)))

    if not all_closed or not rings:
        return aabb_area

    if len(rings) == 1:
        area = rings[0][0]
        return max(1.0, area) if area > 1e-6 else aabb_area

    max_idx = max(range(len(rings)), key=lambda idx: rings[idx][0])
    outer_area, _, outer_segs = rings[max_idx]
    total_area = outer_area
    for idx, (r_area, sample_pt, _) in enumerate(rings):
        if idx == max_idx:
            continue
        if _point_in_segments(sample_pt[0], sample_pt[1], outer_segs):
            total_area -= r_area
        else:
            total_area += r_area
    return max(1.0, total_area) if total_area > 1e-6 else aabb_area


class RewardShaper:
    """Dense coordinate-free training reward shaper built over doom_env privileged deltas."""

    def __init__(self, weights: Optional[dict[str, float] | RewardWeights] = None) -> None:
        if isinstance(weights, RewardWeights):
            self.w_exit = float(weights.exit_base)
            self.w_exit_speed = float(weights.exit_speed_bonus)
            self.w_sector = float(weights.new_sector)
            self.w_key = float(weights.key_pickup)
            self.w_door = float(weights.door_open)
            self.w_locked_door = float(weights.locked_door_unlock)
            self.w_kill = float(weights.kill)
            self.w_item = float(weights.item_pickup)
            self.w_damage = float(weights.damage_per_hp)
            self.w_death = float(weights.death)
            self.w_door_bounce = float(weights.door_reversal_bounce)
            self.w_stuck = float(weights.stuck_wall_scrape)
            self.w_wasted_shot = float(weights.wasted_shot)
            self.w_step = float(weights.step_cost)
            self.w_cell = float(weights.new_cell)
            self.w_loop = float(weights.loop_penalty)
            self.w_conflict = float(weights.conflict_buttons)
            self.w_back = float(weights.backward_move)
            self.w_fwd = float(weights.forward_progress)
        else:
            w = dict(weights or {})
            self.w_exit = float(w.get("exit", 15.0))
            self.w_exit_speed = float(w.get("exit_speed", 3.0))
            self.w_sector = float(w.get("sector", 0.40))
            self.w_key = float(w.get("key", 4.00))
            self.w_door = float(w.get("door", 1.20))
            self.w_locked_door = float(w.get("locked_door", 5.00))
            self.w_kill = float(w.get("kill", 1.00))
            self.w_item = float(w.get("item", 0.15))
            self.w_damage = float(w.get("damage", -0.02))
            self.w_death = float(w.get("death", -4.00))
            self.w_door_bounce = float(w.get("door_bounce", -0.30))
            self.w_stuck = float(w.get("stuck", -0.02))
            self.w_wasted_shot = float(w.get("wasted_shot", -0.015))
            self.w_step = float(w.get("step", -0.002))
            self.w_cell = float(w.get("new_cell", 0.06))
            self.w_loop = float(w.get("loop_penalty", -0.015))
            self.w_conflict = float(w.get("conflict_buttons", -0.01))
            self.w_back = float(w.get("backward_move", -0.004))
            self.w_fwd = float(w.get("forward_progress", 0.004))

        self._sectors: list[tuple[int, float, float, float, float, np.ndarray, float]] = []
        self.visited_sectors: set[int] = set()
        self.opened_doors: set[int] = set()
        self.unlocked_doors: set[int] = set()
        self._prev_door_vel: dict[int, float] = {}
        self.stuck_steps: int = 0
        self._exit_awarded: bool = False
        self._death_penalized: bool = False
        self._spawn_seeded: bool = False
        self._env_prev_priv: Optional[dict[str, Any]] = None
        self.last_components: dict[str, float] = {}
        self._cells: set[tuple[int, int]] = set()
        self._pos_hist: list[tuple[float, float]] = []

    @property
    def visited_sector_count(self) -> int:
        return len(self.visited_sectors)

    def reset(self, sector_polygons: Optional[Any] = None) -> None:
        """Initialize per-episode sector geometry and episodic trackers.

        Accepts either `sector_polygons: list[tuple[int, np.ndarray]] | None` (Schema.md §2.4)
        or a `DoomEnv` instance directly.
        """
        env_obj = None
        raw_polys = sector_polygons
        if sector_polygons is not None and hasattr(sector_polygons, "sector_polygons") and hasattr(sector_polygons, "privileged_state"):
            env_obj = sector_polygons
            raw_polys = env_obj.sector_polygons()

        self._sectors = []
        if raw_polys:
            for sid, segs_raw in raw_polys:
                segs = np.asarray(segs_raw, dtype=np.float64)
                if segs.ndim != 2 or segs.shape[1] != 4 or len(segs) == 0:
                    continue
                xmin = float(min(segs[:, 0].min(), segs[:, 2].min()))
                xmax = float(max(segs[:, 0].max(), segs[:, 2].max()))
                ymin = float(min(segs[:, 1].min(), segs[:, 3].min()))
                ymax = float(max(segs[:, 1].max(), segs[:, 3].max()))
                aabb_area = max(1.0, (xmax - xmin) * (ymax - ymin))
                area = _polygon_area_from_segments(segs, aabb_area)
                weight = max(0.2, min(1.5, area / (512.0 * 512.0)))
                self._sectors.append((int(sid), xmin, xmax, ymin, ymax, segs, weight))

        self.visited_sectors = set()
        self.opened_doors = set()
        self.unlocked_doors = set()
        self._prev_door_vel = {}
        self.stuck_steps = 0
        self._exit_awarded = False
        self._death_penalized = False
        self._spawn_seeded = False
        self._env_prev_priv = None
        self.last_components = {}
        self._cells = set()
        self._pos_hist = []

        if env_obj is not None:
            priv0 = env_obj.privileged_state()
            self._env_prev_priv = priv0
            spawn_hit = self._find_sector(float(priv0.get("x", 0.0)), float(priv0.get("y", 0.0)))
            if spawn_hit is not None:
                self.visited_sectors.add(spawn_hit[0])
            self._spawn_seeded = True

    def _find_sector(self, x: float, y: float) -> Optional[tuple[int, float]]:
        for sid, xmin, xmax, ymin, ymax, segs, weight in self._sectors:
            if (xmin - 1e-9) <= x <= (xmax + 1e-9) and (ymin - 1e-9) <= y <= (ymax + 1e-9):
                if _point_in_segments(x, y, segs):
                    return sid, weight
        return None

    def _compute_step(
        self,
        priv_prev: dict[str, Any],
        priv_now: dict[str, Any],
        action: np.ndarray | list[int],
        info: Optional[dict[str, Any]] = None,
    ) -> float:
        """Compute dense shaping reward strictly from relative deltas between `priv_prev` and `priv_now`."""
        act = np.asarray(action, dtype=np.int64).reshape(-1)
        info_d = info or {}

        comp: dict[str, float] = {
            "exit": 0.0,
            "sector": 0.0,
            "key": 0.0,
            "door": 0.0,
            "locked_door": 0.0,
            "door_bounce": 0.0,
            "kill": 0.0,
            "item": 0.0,
            "damage": 0.0,
            "death": 0.0,
            "stuck": 0.0,
            "step": self.w_step,
            "explore": 0.0,
            "loop": 0.0,
            "conflict": 0.0,
            "backward": 0.0,
            "forward": 0.0,
        }

        # 1. Exit bonus (guarded once per episode)
        exit_now = bool(info_d.get("exit_reached", False) or priv_now.get("exit_reached", False))
        exit_prev = bool(priv_prev.get("exit_reached", False))
        if exit_now and not exit_prev and not self._exit_awarded:
            max_ticks = max(1, int(priv_now.get("episode_ticks", 2100)))
            tic = max(0, int(priv_now.get("tic", max_ticks)))
            rem_frac = max(0.0, float(max_ticks - tic) / float(max_ticks))
            comp["exit"] = self.w_exit + self.w_exit_speed * rem_frac
            self._exit_awarded = True

        # 2. Unique sector discovery (translation-invariant point-in-polygon)
        if not self._spawn_seeded:
            self._spawn_seeded = True
            spawn_hit = self._find_sector(
                float(priv_prev.get("x", 0.0)),
                float(priv_prev.get("y", 0.0)),
            )
            if spawn_hit is not None:
                self.visited_sectors.add(spawn_hit[0])

        x_now = float(priv_now.get("x", 0.0))
        y_now = float(priv_now.get("y", 0.0))
        sec_hit = self._find_sector(x_now, y_now)
        if sec_hit is not None:
            sid, weight = sec_hit
            if sid not in self.visited_sectors:
                self.visited_sectors.add(sid)
                comp["sector"] = self.w_sector * weight

        # 3. Keycard / skull pickup delta
        keys_prev = np.asarray(priv_prev.get("keys", np.zeros(6)), dtype=np.float32)
        keys_now = np.asarray(priv_now.get("keys", np.zeros(6)), dtype=np.float32)
        key_delta = max(0.0, float(np.sum(keys_now - keys_prev)))
        if key_delta > 0.0:
            comp["key"] = self.w_key * key_delta

        # 4. Door open / locked door unlock / door reversal bounce
        fl_prev = priv_prev.get("sector_floor")
        cl_prev = priv_prev.get("sector_ceiling")
        fl_now = priv_now.get("sector_floor")
        cl_now = priv_now.get("sector_ceiling")
        locks = priv_now.get("door_lock")
        if fl_prev is not None and cl_prev is not None and fl_now is not None and cl_now is not None and locks is not None:
            open_prev = np.asarray(cl_prev, dtype=np.float64) - np.asarray(fl_prev, dtype=np.float64)
            open_now = np.asarray(cl_now, dtype=np.float64) - np.asarray(fl_now, dtype=np.float64)
            locks_arr = np.asarray(locks, dtype=np.int64)
            n_sec = min(len(open_prev), len(open_now), len(locks_arr))
            use_pressed = len(act) > 7 and int(act[7]) == 1
            bounced = False
            for s in range(n_sec):
                lock_val = int(locks_arr[s])
                if lock_val < 0:
                    continue
                op_p = float(open_prev[s])
                op_n = float(open_now[s])
                vel = op_n - op_p
                prev_vel = self._prev_door_vel.get(s, 0.0)
                if op_n > 8.0 and op_p <= 8.0:
                    if lock_val > 0 and s not in self.unlocked_doors:
                        self.unlocked_doors.add(s)
                        self.opened_doors.add(s)
                        comp["locked_door"] += self.w_locked_door
                    elif lock_val == 0 and s not in self.opened_doors:
                        self.opened_doors.add(s)
                        comp["door"] += self.w_door
                if use_pressed and prev_vel > 0.5 and vel < -0.5 and not bounced:
                    comp["door_bounce"] = self.w_door_bounce
                    bounced = True
                self._prev_door_vel[s] = vel

        # 5. Combat, pickups & death
        kill_delta = max(0, int(priv_now.get("kills", 0)) - int(priv_prev.get("kills", 0)))
        if kill_delta > 0:
            comp["kill"] = self.w_kill * float(kill_delta)
        elif len(act) > 6 and int(act[6]) == 1 and not bool(priv_now.get("finished", False)):
            comp["wasted_shot"] = self.w_wasted_shot

        item_delta = max(0, int(priv_now.get("items", 0)) - int(priv_prev.get("items", 0)))
        if item_delta > 0:
            comp["item"] = self.w_item * float(item_delta)

        dmg_delta = max(0.0, float(priv_now.get("damage_taken", 0.0)) - float(priv_prev.get("damage_taken", 0.0)))
        if dmg_delta > 0.0:
            comp["damage"] = self.w_damage * dmg_delta

        if bool(priv_now.get("dead", False)) and not self._death_penalized:
            self._death_penalized = True
            comp["death"] = self.w_death

        # 6. Anti-wall-scrape / stuck penalty
        dx = x_now - float(priv_prev.get("x", 0.0))
        dy = y_now - float(priv_prev.get("y", 0.0))
        disp = math.hypot(dx, dy)
        moving_fwd = len(act) > 0 and int(act[0]) == 1
        if moving_fwd and disp < 2.0 and not bool(priv_now.get("finished", False)):
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        if self.stuck_steps > 8:
            comp["stuck"] = self.w_stuck

        # 7. Anti-loop exploration shaping: reward new 80-unit cells, punish dithering in place
        cell = (int(math.floor(x_now / 80.0)), int(math.floor(y_now / 80.0)))
        if cell not in self._cells:
            self._cells.add(cell)
            if len(self._cells) > 1:
                comp["explore"] = self.w_cell
        self._pos_hist.append((x_now, y_now))
        if len(self._pos_hist) > 40:
            self._pos_hist.pop(0)
        if len(self._pos_hist) == 40 and not bool(priv_now.get("finished", False)):
            ox, oy = self._pos_hist[0]
            if math.hypot(x_now - ox, y_now - oy) < 48.0:
                comp["loop"] = self.w_loop
        if len(act) > 5:
            n_conf = int(act[0] == 1 and act[1] == 1) + int(act[2] == 1 and act[3] == 1) + int(act[4] == 1 and act[5] == 1)
            if n_conf:
                comp["conflict"] = self.w_conflict * float(n_conf)
            if int(act[1]) == 1 and int(act[0]) == 0:
                comp["backward"] = self.w_back
            if moving_fwd and int(act[1]) == 0 and disp >= 3.0:
                comp["forward"] = self.w_fwd

        self.last_components = comp
        return float(sum(comp.values()))

    def step(
        self,
        priv_prev: Any,
        priv_now: Any,
        action: Optional[np.ndarray | list[int]] = None,
        info: Optional[dict[str, Any]] = None,
    ) -> float | tuple[float, dict[str, float]]:
        """Compute dense shaping reward.

        Supports both:
        - `step(priv_prev, priv_now, action, info=None) -> float` (Schema.md §2.4)
        - `step(env, action) -> tuple[float, dict[str, float]]` (DoomEnv helper)
        """
        if action is None and hasattr(priv_prev, "privileged_state"):
            if self._env_prev_priv is None:
                raise RuntimeError("Call RewardShaper.reset(env) before step(env, action).")
            cur_priv = priv_prev.privileged_state()
            r = self._compute_step(
                self._env_prev_priv,
                cur_priv,
                priv_now,
                info={"exit_reached": bool(cur_priv.get("exit_reached", False))},
            )
            self._env_prev_priv = cur_priv
            return r, dict(self.last_components)

        if action is None:
            raise TypeError("RewardShaper.step(priv_prev, priv_now, action, info=None) requires `action`")
        return self._compute_step(priv_prev, priv_now, action, info=info)


def attribute_sector_rewards(
    priv_prev: dict[str, Any],
    priv_now: dict[str, Any],
    reward_components: Optional[dict[str, float] | float] = None,
) -> np.ndarray:
    """Attribute step reward to 8 ego-centric angular sectors (0 = ahead, CCW 45 deg/sector).

    Strictly coordinate-free and translation/rotation-invariant for supervising
    `HybridDoomNet` `value_compass: (B, 8)`.
    """
    compass = np.zeros((8,), dtype=np.float32)
    if reward_components is None:
        val = 0.0
    elif isinstance(reward_components, dict):
        val = float(
            sum(
                float(v)
                for k, v in reward_components.items()
                if k not in ("step", "stuck", "door_bounce")
            )
        )
    else:
        val = float(reward_components)

    if abs(val) < 1e-8:
        return compass

    dx = float(priv_now.get("x", 0.0)) - float(priv_prev.get("x", 0.0))
    dy = float(priv_now.get("y", 0.0)) - float(priv_prev.get("y", 0.0))
    heading = math.radians(float(priv_prev.get("angle", 0.0)))

    if math.hypot(dx, dy) > 1e-3:
        rel_angle = (math.atan2(dy, dx) - heading + math.pi) % (2.0 * math.pi) - math.pi
    else:
        rel_angle = 0.0

    sector = int(math.floor(((rel_angle + math.pi / 8.0) % (2.0 * math.pi)) / (math.pi / 4.0))) % 8
    compass[(sector - 1) % 8] += np.float32(0.15 * val)
    compass[sector] += np.float32(0.70 * val)
    compass[(sector + 1) % 8] += np.float32(0.15 * val)
    return compass
