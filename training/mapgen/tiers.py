"""Extreme domain-randomized ZDoom UDMF procedural map generator (Tiers 1-4).

Generates non-overlapping rectangular macro-grid rooms, connectors, doors, locked doors,
and walkover exit alcoves with:
- Zero fixed world coordinates (global origin_x/origin_y offsets in [-2048, 2048],
  90-degree rotations, randomized cell_size 160..384, corridor widths 64..128,
  ceiling/step heights, lighting, and textures).
- Extreme scenario archetypes: open_arena, tight_ambush, branching_maze, scarce_ammo,
  dense_waves, multi_room_doors, keycard_progression.
- BFS reachability verification (start -> keycard(s) -> locked door(s) -> exit).
- Pure-Python WAD packaging via doom_env.wad.write_wad + optional /usr/bin/zdbsp
  (zero omgifol dependency, < 2048 sectors).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from doom_env.wad import Lump, write_wad

ALL_SCENARIOS = (
    "open_arena",
    "tight_ambush",
    "branching_maze",
    "scarce_ammo",
    "dense_waves",
    "multi_room_doors",
    "keycard_progression",
)

TIER_SCENARIOS: dict[int, tuple[str, ...]] = {
    1: ("open_arena", "tight_ambush", "scarce_ammo", "dense_waves"),
    2: ("multi_room_doors", "branching_maze", "tight_ambush", "scarce_ammo", "open_arena"),
    3: ("keycard_progression", "branching_maze", "tight_ambush", "scarce_ammo", "multi_room_doors"),
    4: ("keycard_progression", "dense_waves", "branching_maze", "tight_ambush", "scarce_ammo", "open_arena"),
}

FLOOR_FLATS = ("FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLAT14", "FLAT17", "FLAT1_1", "FLAT1_2", "FLAT8")
CEIL_FLATS = ("CEIL1_1", "CEIL1_2", "CEIL3_1", "CEIL5_1", "CEIL5_2")
WALL_TEXTURES = ("STARTAN2", "STONE2", "BROWN1", "GRAY1")

# Verified Lock specs matching doom_env.mapinfo & Freedoom Thing DoomEdNums
KEY_SPECS = (
    {"thing_type": 13, "lock": 129, "door_tex": "DOORRED", "name": "red"},
    {"thing_type": 5, "lock": 130, "door_tex": "DOORBLU", "name": "blue"},
    {"thing_type": 6, "lock": 131, "door_tex": "DOORYEL", "name": "yellow"},
)

ENEMY_WEAK = (3004, 9)            # Zombieman, ShotgunGuy
ENEMY_MID = (3001, 3002, 65)      # Imp, Demon, ChaingunGuy
ENEMY_HEAVY = (3001, 3002, 3005)  # Imp, Demon, Cacodemon


class GeneratedMap(tuple):
    """`(Path, str)` tuple that also implements `os.PathLike` for single-Path callers."""

    def __new__(cls, wad_path: Path, map_name: str = "MAP01") -> "GeneratedMap":
        return super().__new__(cls, (wad_path, map_name))

    @property
    def path(self) -> Path:
        return self[0]

    def __fspath__(self) -> str:
        return os.fspath(self[0])

    def __getattr__(self, attr: str) -> Any:
        return getattr(self[0], attr)


@dataclass
class _RectSector:
    sid: int
    x0: int
    y0: int
    x1: int
    y1: int
    floor_h: int
    ceil_h: int
    light: int
    tex_floor: str
    tex_ceil: str
    tex_wall: str
    kind: str = "room"  # "room", "corridor", "door", "exit"
    door_lock: int = 0
    door_tex: str = "BIGDOOR2"


@dataclass
class _Thing:
    x: float
    y: float
    angle: int
    thing_type: int


def _rotate_rect(x0: int, y0: int, x1: int, y1: int, rot_k: int) -> tuple[int, int, int, int]:
    pts = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    rpts = []
    for x, y in pts:
        if rot_k == 0:
            rpts.append((x, y))
        elif rot_k == 1:
            rpts.append((-y, x))
        elif rot_k == 2:
            rpts.append((-x, -y))
        else:
            rpts.append((y, -x))
    xs = [p[0] for p in rpts]
    ys = [p[1] for p in rpts]
    return min(xs), min(ys), max(xs), max(ys)


def _rotate_pt(x: float, y: float, rot_k: int) -> tuple[float, float]:
    if rot_k == 0:
        return x, y
    if rot_k == 1:
        return -y, x
    if rot_k == 2:
        return -x, -y
    return y, -x


def _bfs_reachable(
    start: int,
    adj: dict[int, list[tuple[int, int]]],
    held_locks: set[int],
) -> set[int]:
    visited = {start}
    q = deque([start])
    while q:
        u = q.popleft()
        for v, lock in adj.get(u, []):
            if v in visited:
                continue
            if lock == 0 or lock in held_locks:
                visited.add(v)
                q.append(v)
    return visited


def _compile_udmf(
    sectors: list[_RectSector],
    portals: dict[tuple[int, int], tuple[str, int, str]],
    things: list[_Thing],
) -> str:
    """Compile non-overlapping axis-aligned RectSectors + portals into valid ZDoom UDMF TEXTMAP."""
    v_map: dict[tuple[int, int], int] = {}
    vertices: list[tuple[int, int]] = []

    def vid(x: int, y: int) -> int:
        key = (int(x), int(y))
        idx = v_map.get(key)
        if idx is None:
            idx = len(vertices)
            v_map[key] = idx
            vertices.append(key)
        return idx

    sidedefs: list[str] = []
    linedefs: list[str] = []

    def add_side(sec_id: int, tex_top: str = "-", tex_mid: str = "-", tex_bot: str = "-") -> int:
        idx = len(sidedefs)
        sidedefs.append(
            f'sidedef {{ sector = {sec_id}; texturetop = "{tex_top}"; '
            f'texturemiddle = "{tex_mid}"; texturebottom = "{tex_bot}"; }}'
        )
        return idx

    def emit_edge(
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        sec_right: Optional[_RectSector],
        sec_left: Optional[_RectSector],
    ) -> None:
        if sec_right is None and sec_left is None:
            return
        if sec_right is not None and sec_left is None:
            s_front = add_side(sec_right.sid, tex_mid=sec_right.tex_wall)
            linedefs.append(
                f"linedef {{ v1 = {vid(x1, y1)}; v2 = {vid(x2, y2)}; sidefront = {s_front}; "
                f"blocking = true; dontpegbottom = true; }}"
            )
            return
        if sec_left is not None and sec_right is None:
            s_front = add_side(sec_left.sid, tex_mid=sec_left.tex_wall)
            linedefs.append(
                f"linedef {{ v1 = {vid(x2, y2)}; v2 = {vid(x1, y1)}; sidefront = {s_front}; "
                f"blocking = true; dontpegbottom = true; }}"
            )
            return

        assert sec_right is not None and sec_left is not None
        pair = (min(sec_right.sid, sec_left.sid), max(sec_right.sid, sec_left.sid))
        pinfo = portals.get(pair)
        if pinfo is None:
            pinfo = ("open", 0, "-")

        # Ensure sidefront is the room (and sideback is door/exit) when one side is door or exit
        front_sec, back_sec = sec_right, sec_left
        va, vb = (x1, y1), (x2, y2)
        if sec_right.kind in ("door", "exit") and sec_left.kind not in ("door", "exit"):
            front_sec, back_sec = sec_left, sec_right
            va, vb = (x2, y2), (x1, y1)

        pkind, lock_num, door_tex = pinfo
        if pkind == "door":
            s_front = add_side(front_sec.sid, tex_top=door_tex, tex_mid="-", tex_bot="-")
            s_back = add_side(back_sec.sid, tex_top="-", tex_mid="-", tex_bot="-")
            if lock_num > 0:
                linedefs.append(
                    f"linedef {{ v1 = {vid(*va)}; v2 = {vid(*vb)}; sidefront = {s_front}; sideback = {s_back}; "
                    f"twosided = true; special = 13; arg0 = 0; arg1 = 16; arg2 = 0; arg3 = {lock_num}; "
                    f"playeruse = true; monsteruse = false; repeatspecial = true; }}"
                )
            else:
                linedefs.append(
                    f"linedef {{ v1 = {vid(*va)}; v2 = {vid(*vb)}; sidefront = {s_front}; sideback = {s_back}; "
                    f"twosided = true; special = 12; arg0 = 0; arg1 = 16; arg2 = 150; "
                    f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
                )
        elif pkind == "exit":
            s_front = add_side(front_sec.sid)
            s_back = add_side(back_sec.sid)
            linedefs.append(
                f"linedef {{ v1 = {vid(*va)}; v2 = {vid(*vb)}; sidefront = {s_front}; sideback = {s_back}; "
                f"twosided = true; special = 243; playercross = true; }}"
            )
        else:
            top_tex = front_sec.tex_wall if front_sec.ceil_h != back_sec.ceil_h else "-"
            bot_tex = front_sec.tex_wall if front_sec.floor_h != back_sec.floor_h else "-"
            s_front = add_side(front_sec.sid, tex_top=top_tex, tex_bot=bot_tex)
            s_back = add_side(back_sec.sid, tex_top=top_tex, tex_bot=bot_tex)
            linedefs.append(
                f"linedef {{ v1 = {vid(*va)}; v2 = {vid(*vb)}; sidefront = {s_front}; sideback = {s_back}; "
                f"twosided = true; }}"
            )

    # Split vertical edges at each x
    x_coords = sorted({s.x0 for s in sectors} | {s.x1 for s in sectors})
    for x in x_coords:
        relevant = [s for s in sectors if s.x0 == x or s.x1 == x]
        ys = sorted({s.y0 for s in relevant} | {s.y1 for s in relevant})
        for i in range(len(ys) - 1):
            ya, yb = ys[i], ys[i + 1]
            if ya >= yb:
                continue
            west = next((s for s in relevant if s.x1 == x and s.y0 <= ya and yb <= s.y1), None)
            east = next((s for s in relevant if s.x0 == x and s.y0 <= ya and yb <= s.y1), None)
            emit_edge(x, yb, x, ya, west, east)

    # Split horizontal edges at each y
    y_coords = sorted({s.y0 for s in sectors} | {s.y1 for s in sectors})
    for y in y_coords:
        relevant = [s for s in sectors if s.y0 == y or s.y1 == y]
        xs = sorted({s.x0 for s in relevant} | {s.x1 for s in relevant})
        for i in range(len(xs) - 1):
            xa, xb = xs[i], xs[i + 1]
            if xa >= xb:
                continue
            south = next((s for s in relevant if s.y1 == y and s.x0 <= xa and xb <= s.x1), None)
            north = next((s for s in relevant if s.y0 == y and s.x0 <= xa and xb <= s.x1), None)
            emit_edge(xa, y, xb, y, south, north)

    lines_out = ['namespace = "zdoom";']
    for vx, vy in vertices:
        lines_out.append(f"vertex {{ x = {float(vx):.1f}; y = {float(vy):.1f}; }}")
    for sec in sectors:
        lines_out.append(
            f'sector {{ texturefloor = "{sec.tex_floor}"; textureceiling = "{sec.tex_ceil}"; '
            f"heightfloor = {sec.floor_h}; heightceiling = {sec.ceil_h}; lightlevel = {sec.light}; }}"
        )
    lines_out.extend(sidedefs)
    lines_out.extend(linedefs)
    for th in things:
        lines_out.append(
            f"thing {{ x = {th.x:.1f}; y = {th.y:.1f}; angle = {int(th.angle) % 360}; type = {th.thing_type}; "
            f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true; }}"
        )
    return "\n".join(lines_out) + "\n"


class MapGen:
    """Extreme domain-randomized procedural UDMF map generator across Tiers 1-4."""

    @staticmethod
    def sample(
        tier: int,
        knobs: Optional[dict[str, Any]] = None,
        seed: Optional[int] = 0,
        scaffolding_alpha: float = 0.0,
        *,
        out_dir: Optional[str | Path] = None,
    ) -> GeneratedMap:
        """Generate a BFS-verified ZDoom UDMF WAD and return `GeneratedMap(wad_path, 'MAP01')`."""
        tier_i = int(tier)
        if tier_i not in (1, 2, 3, 4):
            raise ValueError(f"tier must be in {{1, 2, 3, 4}}, got {tier}")
        tier = tier_i
        k = dict(knobs or {})
        alpha = max(0.0, min(1.0, float(scaffolding_alpha)))
        if seed is None:
            seed = int(np.random.default_rng().integers(100_000, 1_000_000))
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)

        scenario = str(k.get("scenario") or rng.choice(TIER_SCENARIOS[tier]))
        if scenario not in ALL_SCENARIOS:
            scenario = str(rng.choice(TIER_SCENARIOS[tier]))

        # Wide domain randomization of scales & offsets (zero fixed XYZ coordinates!)
        if scenario == "tight_ambush":
            room_size = int(k.get("cell_size", int(rng.integers(160, 225))))
            corridor_w = int(k.get("corridor_w", 64))
        elif scenario == "open_arena":
            room_size = int(k.get("cell_size", int(rng.integers(272, 385))))
            corridor_w = int(k.get("corridor_w", int(rng.choice([96, 128]))))
        else:
            room_size = int(k.get("cell_size", int(rng.integers(192, 353))))
            corridor_w = int(k.get("corridor_w", int(rng.choice([64, 96, 128]))))
        room_size = max(160, (room_size // 16) * 16)
        corridor_w = min(room_size - 32, max(64, (corridor_w // 16) * 16))
        gap = 64
        stride = room_size + gap

        origin_x = int(k.get("origin_x", int(rng.integers(-128, 129)) * 16))
        origin_y = int(k.get("origin_y", int(rng.integers(-128, 129)) * 16))
        rot_k = int(k.get("rot_k", int(rng.integers(0, 4))))

        default_rooms = {1: 2, 2: 4, 3: 5, 4: 6}[tier]
        num_rooms = int(k.get("num_rooms", max(2 if tier > 1 else 1, default_rooms + int(rng.integers(-1, 2)))))
        if scenario == "open_arena" and tier == 1:
            num_rooms = max(1, min(3, num_rooms))
        if scenario in ("branching_maze", "multi_room_doors"):
            num_rooms = max(3, num_rooms)
        if tier >= 3:
            num_rooms = max(4, num_rooms)

        default_enemies = {1: 1.5, 2: 2.0, 3: 2.5, 4: 3.5}[tier]
        enemies_per_room = float(k.get("enemies_per_room", default_enemies))
        if scenario == "dense_waves":
            enemies_per_room *= 1.6
        elif scenario == "tight_ambush":
            enemies_per_room = max(2.0, enemies_per_room * 1.2)

        ammo_mult = float(k.get("ammo_mult", 0.45 if scenario == "scarce_ammo" else 1.0))

        # --- Build connected macro-grid room graph ---
        nodes: list[tuple[int, int]] = [(0, 0)]
        occupied = {(0, 0): 0}
        edges: list[tuple[int, int]] = []

        if scenario == "open_arena" and tier == 1 and num_rooms == 1:
            pass
        else:
            dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
            while len(nodes) < num_rooms:
                if scenario == "branching_maze":
                    parent_idx = int(rng.integers(0, len(nodes)))
                else:
                    parent_idx = len(nodes) - 1 if rng.random() < 0.65 else int(rng.integers(0, len(nodes)))
                px, py = nodes[parent_idx]
                perm = rng.permutation(4)
                added = False
                for di in perm:
                    dx, dy = dirs[int(di)]
                    nx, ny = px + dx, py + dy
                    if (nx, ny) not in occupied:
                        nid = len(nodes)
                        nodes.append((nx, ny))
                        occupied[(nx, ny)] = nid
                        edges.append((parent_idx, nid))
                        added = True
                        break
                if not added:
                    free_found = False
                    for pi in rng.permutation(len(nodes)):
                        px2, py2 = nodes[int(pi)]
                        for dx, dy in dirs:
                            nx2, ny2 = px2 + dx, py2 + dy
                            if (nx2, ny2) not in occupied:
                                nid = len(nodes)
                                nodes.append((nx2, ny2))
                                occupied[(nx2, ny2)] = nid
                                edges.append((int(pi), nid))
                                free_found = True
                                break
                        if free_found:
                            break
                    if not free_found:
                        break

        tree_adj: dict[int, list[int]] = {i: [] for i in range(len(nodes))}
        for u, v in edges:
            tree_adj[u].append(v)
            tree_adj[v].append(u)

        dist = {0: 0}
        parent_map = {0: -1}
        q = deque([0])
        while q:
            u = q.popleft()
            for v in tree_adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    parent_map[v] = u
                    q.append(v)

        exit_room = max(range(len(nodes)), key=lambda i: dist[i])
        path_to_exit: list[int] = []
        cur = exit_room
        while cur != -1:
            path_to_exit.append(cur)
            cur = parent_map[cur]
        path_to_exit.reverse()

        num_locks = 0
        if tier == 3 or scenario == "keycard_progression":
            num_locks = 1 if len(nodes) >= 3 else 0
        elif tier == 4:
            num_locks = 2 if len(nodes) >= 6 else (1 if len(nodes) >= 3 else 0)

        locked_edges: dict[tuple[int, int], dict[str, Any]] = {}
        key_placements: list[tuple[int, dict[str, Any]]] = []

        if num_locks >= 1 and len(path_to_exit) >= 2:
            lock_u, lock_v = path_to_exit[-2], path_to_exit[-1]
            spec1 = KEY_SPECS[int(rng.integers(0, len(KEY_SPECS)))]
            edge_key1 = (min(lock_u, lock_v), max(lock_u, lock_v))

            adj_no_lock: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(nodes))}
            for u, v in edges:
                ek = (min(u, v), max(u, v))
                if ek != edge_key1:
                    adj_no_lock[u].append((v, 0))
                    adj_no_lock[v].append((u, 0))
            pre_lock_nodes = sorted(_bfs_reachable(0, adj_no_lock, set()) - {0})
            if pre_lock_nodes:
                key_room1 = max(pre_lock_nodes, key=lambda n: (n not in path_to_exit, dist[n]))
                locked_edges[edge_key1] = spec1
                key_placements.append((key_room1, spec1))

                if num_locks == 2 and len(path_to_exit) >= 4:
                    lock2_u, lock2_v = path_to_exit[-3], path_to_exit[-2]
                    edge_key2 = (min(lock2_u, lock2_v), max(lock2_u, lock2_v))
                    rem_specs = [sp for sp in KEY_SPECS if sp["lock"] != spec1["lock"]]
                    spec2 = rem_specs[int(rng.integers(0, len(rem_specs)))]
                    adj_pre2: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(nodes))}
                    for u, v in edges:
                        ek = (min(u, v), max(u, v))
                        if ek not in (edge_key1, edge_key2):
                            adj_pre2[u].append((v, 0))
                            adj_pre2[v].append((u, 0))
                    pre2_nodes = sorted(_bfs_reachable(0, adj_pre2, set()) - {0})
                    if pre2_nodes:
                        key_room2 = max(pre2_nodes, key=lambda n: (n != key_room1, n not in path_to_exit, dist[n]))
                        locked_edges[edge_key2] = spec2
                        key_placements = [(key_room2, spec2), (lock2_v, spec1)]

        if scenario in ("open_arena", "branching_maze") and len(nodes) >= 4 and len(locked_edges) == 0:
            for u in range(len(nodes)):
                ux, uy = nodes[u]
                for dx, dy in ((1, 0), (0, 1)):
                    v = occupied.get((ux + dx, uy + dy))
                    if v is not None and v not in tree_adj[u] and rng.random() < 0.35:
                        edges.append((u, v))
                        tree_adj[u].append(v)
                        tree_adj[v].append(u)

        sectors: list[_RectSector] = []
        portals: dict[tuple[int, int], tuple[str, int, str]] = {}
        room_sector_ids: dict[int, int] = {}

        base_ceil = int(rng.choice([128, 144, 160, 192]))
        base_light = int(rng.integers(128, 225))
        wall_tex = str(rng.choice(WALL_TEXTURES))
        ceil_tex = str(rng.choice(CEIL_FLATS))

        for nid, (gx, gy) in enumerate(nodes):
            x0 = gx * stride
            y0 = gy * stride
            x1 = x0 + room_size
            y1 = y0 + room_size
            fl_h = 0 if nid == 0 else int(rng.choice([0, 0, 8, 16]))
            cl_h = max(fl_h + 112, base_ceil + int(rng.choice([0, 16, 32])))
            lt = max(112, min(232, base_light + int(rng.integers(-24, 25))))
            sec = _RectSector(
                sid=len(sectors),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                floor_h=fl_h,
                ceil_h=cl_h,
                light=lt,
                tex_floor=str(rng.choice(FLOOR_FLATS)),
                tex_ceil=ceil_tex,
                tex_wall=wall_tex,
                kind="room",
            )
            sectors.append(sec)
            room_sector_ids[nid] = sec.sid

        verify_adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(len(nodes))}
        has_normal_door = False
        for edge_idx, (u, v) in enumerate(edges):
            ux, uy = nodes[u]
            vx, vy = nodes[v]
            su = sectors[room_sector_ids[u]]
            sv = sectors[room_sector_ids[v]]
            ek = (min(u, v), max(u, v))
            lock_spec = locked_edges.get(ek)

            if lock_spec is not None:
                is_door = True
                lock_num = int(lock_spec["lock"])
                door_tex = str(lock_spec["door_tex"])
            elif scenario == "open_arena":
                is_door = False
                lock_num = 0
                door_tex = "BIGDOOR2"
            else:
                door_prob = 0.75 if (tier >= 2 or scenario == "multi_room_doors") else 0.35
                is_door = bool(rng.random() < door_prob)
                if scenario == "multi_room_doors" and not has_normal_door and edge_idx == len(edges) - 1:
                    is_door = True
                lock_num = 0
                door_tex = "BIGDOOR2"

            if is_door and lock_num == 0:
                has_normal_door = True

            scaff_open = False
            if is_door and alpha > 0.0 and lock_num == 0 and rng.random() < (0.5 * alpha):
                scaff_open = True
            elif is_door and alpha > 0.5 and lock_num > 0 and rng.random() < (0.35 * (alpha - 0.5) * 2.0):
                lock_num = 0
                door_tex = "BIGDOOR2"
                scaff_open = True

            if ux != vx:
                x_lo = min(ux, vx) * stride + room_size
                x_hi = x_lo + gap
                cy = uy * stride + room_size // 2
                cw = 64 if is_door else corridor_w
                y_lo = cy - cw // 2
                y_hi = y_lo + cw
            else:
                y_lo = min(uy, vy) * stride + room_size
                y_hi = y_lo + gap
                cx = ux * stride + room_size // 2
                cw = 64 if is_door else corridor_w
                x_lo = cx - cw // 2
                x_hi = x_lo + cw

            fl_conn = min(su.floor_h, sv.floor_h)
            if is_door:
                conn_ceil = base_ceil if scaff_open else fl_conn
                conn_kind = "door"
                conn_wall = "DOORTRAK"
                conn_floor = "FLAT5_4"
            else:
                conn_ceil = min(su.ceil_h, sv.ceil_h)
                conn_kind = "corridor"
                conn_wall = wall_tex
                conn_floor = su.tex_floor

            conn = _RectSector(
                sid=len(sectors),
                x0=x_lo,
                y0=y_lo,
                x1=x_hi,
                y1=y_hi,
                floor_h=fl_conn,
                ceil_h=conn_ceil,
                light=su.light,
                tex_floor=conn_floor,
                tex_ceil=ceil_tex,
                tex_wall=conn_wall,
                kind=conn_kind,
                door_lock=lock_num,
                door_tex=door_tex,
            )
            sectors.append(conn)
            p_mode = ("door", lock_num, door_tex) if is_door else ("open", 0, "-")
            portals[(min(su.sid, conn.sid), max(su.sid, conn.sid))] = p_mode
            portals[(min(sv.sid, conn.sid), max(sv.sid, conn.sid))] = p_mode
            verify_adj[u].append((v, lock_num))
            verify_adj[v].append((u, lock_num))

        # --- Internal BFS Solvability Verification ---
        held: set[int] = set()
        progress = True
        key_by_room: dict[int, list[int]] = {}
        for kr, spec in key_placements:
            key_by_room.setdefault(kr, []).append(int(spec["lock"]))
        while progress:
            progress = False
            reach = _bfs_reachable(0, verify_adj, held)
            for r in reach:
                for lk in key_by_room.get(r, []):
                    if lk not in held:
                        held.add(lk)
                        progress = True
        final_reach = _bfs_reachable(0, verify_adj, held)
        if exit_room not in final_reach:
            raise RuntimeError(f"Generated map seed={seed} tier={tier} failed BFS reachability check")

        # --- Attach Walkover Exit Alcove ---
        ex_gx, ex_gy = nodes[exit_room]
        ex_sec = sectors[room_sector_ids[exit_room]]
        exit_w = 96 if room_size >= 160 else 64
        exit_d = 64
        chosen_dir = None
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            if (ex_gx + dx, ex_gy + dy) not in occupied:
                chosen_dir = (dx, dy)
                break
        if chosen_dir is None:
            for nid_cand in reversed(path_to_exit):
                cgx, cgy = nodes[nid_cand]
                for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    if (cgx + dx, cgy + dy) not in occupied:
                        ex_sec = sectors[room_sector_ids[nid_cand]]
                        chosen_dir = (dx, dy)
                        break
                if chosen_dir is not None:
                    break
        assert chosen_dir is not None
        dx, dy = chosen_dir
        if dx == 1:
            ex0, ex1 = ex_sec.x1, ex_sec.x1 + exit_d
            cy = (ex_sec.y0 + ex_sec.y1) // 2
            ey0, ey1 = cy - exit_w // 2, cy + exit_w // 2
        elif dx == -1:
            ex0, ex1 = ex_sec.x0 - exit_d, ex_sec.x0
            cy = (ex_sec.y0 + ex_sec.y1) // 2
            ey0, ey1 = cy - exit_w // 2, cy + exit_w // 2
        elif dy == 1:
            ey0, ey1 = ex_sec.y1, ex_sec.y1 + exit_d
            cx = (ex_sec.x0 + ex_sec.x1) // 2
            ex0, ex1 = cx - exit_w // 2, cx + exit_w // 2
        else:
            ey0, ey1 = ex_sec.y0 - exit_d, ex_sec.y0
            cx = (ex_sec.x0 + ex_sec.x1) // 2
            ex0, ex1 = cx - exit_w // 2, cx + exit_w // 2

        exit_alcove = _RectSector(
            sid=len(sectors),
            x0=ex0,
            y0=ey0,
            x1=ex1,
            y1=ey1,
            floor_h=ex_sec.floor_h,
            ceil_h=ex_sec.ceil_h,
            light=min(240, ex_sec.light + 24),
            tex_floor="GATE3",
            tex_ceil=ceil_tex,
            tex_wall="GRAY1",
            kind="exit",
        )
        sectors.append(exit_alcove)
        portals[(min(ex_sec.sid, exit_alcove.sid), max(ex_sec.sid, exit_alcove.sid))] = ("exit", 0, "-")

        # --- Populate Things ---
        things: list[_Thing] = []
        s0 = sectors[room_sector_ids[0]]
        p_x = 0.5 * (s0.x0 + s0.x1)
        p_y = 0.5 * (s0.y0 + s0.y1)
        p_angle = 0
        if tree_adj[0]:
            nb = tree_adj[0][0]
            nbx, nby = nodes[nb]
            dx0, dy0 = nbx - nodes[0][0], nby - nodes[0][1]
            if dx0 > 0:
                p_angle = 0
            elif dy0 > 0:
                p_angle = 90
            elif dx0 < 0:
                p_angle = 180
            else:
                p_angle = 270
        things.append(_Thing(p_x, p_y, p_angle, 1))

        if (tier >= 2 or scenario in ("open_arena", "dense_waves")) and scenario != "scarce_ammo":
            w_type = 2001 if rng.random() < 0.75 else 2002
            things.append(_Thing(p_x + 32.0, p_y, 0, w_type))
            things.append(_Thing(p_x - 32.0, p_y, 0, 2008 if w_type == 2001 else 2007))

        for kr, spec in key_placements:
            ks = sectors[room_sector_ids[kr]]
            things.append(_Thing(0.5 * (ks.x0 + ks.x1), 0.5 * (ks.y0 + ks.y1), 0, int(spec["thing_type"])))

        for nid in range(len(nodes)):
            rs = sectors[room_sector_ids[nid]]
            cx = 0.5 * (rs.x0 + rs.x1)
            cy = 0.5 * (rs.y0 + rs.y1)
            margin = max(40.0, 0.25 * room_size)

            if nid == 0 and len(nodes) > 1:
                if alpha > 0.0 and rng.random() < alpha:
                    things.append(_Thing(cx, cy - margin * 0.5, 0, 2011))
                continue

            if "enemies_per_room" in k and float(k["enemies_per_room"]) <= 0.0:
                n_en = 0
            else:
                n_en = int(rng.poisson(max(0.5, enemies_per_room)))
                if nid == 0 and len(nodes) == 1:
                    n_en = max(1, n_en)
                if scenario == "dense_waves":
                    n_en = max(2, n_en)
                n_en = min(6, n_en)

            pool = ENEMY_WEAK if tier == 1 else (ENEMY_WEAK + ENEMY_MID if tier <= 2 else ENEMY_MID + ENEMY_HEAVY)
            for _ in range(n_en):
                ex = float(rng.uniform(rs.x0 + margin, rs.x1 - margin))
                ey = float(rng.uniform(rs.y0 + margin, rs.y1 - margin))
                if nid == 0 and math.hypot(ex - p_x, ey - p_y) < 96.0:
                    ex = rs.x1 - margin
                etype = int(rng.choice(pool))
                things.append(_Thing(ex, ey, int(rng.integers(0, 360)), etype))

            if rng.random() < min(1.0, 0.65 * ammo_mult + 0.4 * alpha):
                ix = float(rng.uniform(rs.x0 + margin, rs.x1 - margin))
                iy = float(rng.uniform(rs.y0 + margin, rs.y1 - margin))
                itype = int(rng.choice([2007, 2008, 2011, 2012, 2018]))
                things.append(_Thing(ix, iy, 0, itype))

            if scenario == "scarce_ammo" and rng.random() < 0.5:
                bx = float(rng.uniform(rs.x0 + margin, rs.x1 - margin))
                by = float(rng.uniform(rs.y0 + margin, rs.y1 - margin))
                things.append(_Thing(bx, by, 0, 2035))

        # --- Apply Global Rotation & Translation (Zero Fixed Coordinates!) ---
        for sec in sectors:
            rx0, ry0, rx1, ry1 = _rotate_rect(sec.x0, sec.y0, sec.x1, sec.y1, rot_k)
            sec.x0 = rx0 + origin_x
            sec.y0 = ry0 + origin_y
            sec.x1 = rx1 + origin_x
            sec.y1 = ry1 + origin_y

        for th in things:
            tx, ty = _rotate_pt(th.x, th.y, rot_k)
            th.x = tx + origin_x
            th.y = ty + origin_y
            th.angle = (th.angle + 90 * rot_k) % 360

        textmap = _compile_udmf(sectors, portals, things)

        target_dir = Path(out_dir) if out_dir is not None else Path(k.get("out_dir", "tmp_wads"))
        target_dir.mkdir(parents=True, exist_ok=True)
        knobs_for_hash = {str(key): val for key, val in k.items() if key != "out_dir"}
        knobs_hash = hashlib.sha1(
            json.dumps(knobs_for_hash, sort_keys=True, default=str).encode("utf-8")
        ).hexdigest()[:8]
        tag = f"t{tier}_{scenario}_s{int(seed)}_a{int(round(alpha * 100))}_k{knobs_hash}_p{os.getpid()}"
        raw_wad = target_dir / f"{tag}_raw.wad"
        final_wad = target_dir / f"{tag}.wad"

        write_wad(
            raw_wad,
            [
                Lump("MAP01", b""),
                Lump("TEXTMAP", textmap.encode("utf-8")),
                Lump("ENDMAP", b""),
            ],
        )

        zdbsp = shutil.which("zdbsp")
        if zdbsp:
            proc = subprocess.run(
                [zdbsp, "-o", str(final_wad), str(raw_wad)],
                capture_output=True,
                text=True,
                timeout=15,
            )
            if proc.returncode == 0 and final_wad.is_file():
                raw_wad.unlink(missing_ok=True)
                return GeneratedMap(final_wad, "MAP01")

        raw_wad.replace(final_wad)
        return GeneratedMap(final_wad, "MAP01")


def wad_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
