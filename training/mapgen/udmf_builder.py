"""Procedural UDMF (TEXTMAP) PWAD builder for ViZDoom / Freedoom2.

Generates watertight, non-self-intersecting orthogonal sector layouts in ZDoom
UDMF format (`MAP01` + `TEXTMAP` + `ENDMAP`) so ViZDoom 1.3.3 builds nodes
cleanly without requiring an external BSP binary.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

# ZDoom lock numbers matching env/doom_env/mapinfo.py
LOCK_RED = 1
LOCK_BLUE = 2
LOCK_YELLOW = 3

# Doom thing type IDs (Freedoom2 / Vanilla Doom II compatible)
THING_PLAYER1 = 1
THING_ZOMBIEMAN = 3004
THING_SHOTGUNGUY = 9
THING_IMP = 3001
THING_CHAINGUNGUY = 65
THING_DEMON = 3002

THING_STIMPACK = 2011
THING_MEDIKIT = 2012
THING_ARMOR_GREEN = 2018
THING_CLIP = 2007
THING_SHELLS = 2008
THING_AMMO_BOX = 2048
THING_SHELL_BOX = 2049
THING_SHOTGUN = 2001
THING_CHAINGUN = 2002

THING_KEY_BLUE = 5
THING_KEY_YELLOW = 6
THING_KEY_RED = 13

DOOR_TEXTURE_BY_LOCK = {
    0: "BIGDOOR2",
    LOCK_RED: "BIGDOOR2",
    LOCK_BLUE: "BIGDOOR2",
    LOCK_YELLOW: "BIGDOOR2",
}


@dataclass
class PillarSpec:
    """Solid 1-sided rectangular column strictly inside a sector."""

    x1: float
    y1: float
    x2: float
    y2: float
    texture: str = "METAL"


@dataclass
class SectorSpec:
    """Axis-aligned rectangular sector in Doom map units."""

    x1: float
    y1: float
    x2: float
    y2: float
    floor_h: int = 0
    ceil_h: int = 128
    light: int = 192
    tex_floor: str = "FLOOR0_1"
    tex_ceil: str = "CEIL1_1"
    tex_wall: str = "STARTAN2"
    is_door: bool = False
    door_lock: int = 0  # 0 = unlocked, 1 = Red, 2 = Blue, 3 = Yellow
    pre_opened: bool = False  # PLR scaffolding knob
    is_exit: bool = False
    pillars: list[PillarSpec] = field(default_factory=list)


@dataclass
class ThingSpec:
    """Actor / item / player spawn placed inside the map."""

    x: float
    y: float
    angle: int
    thing_type: int
    ambush: bool = False


def _write_wad(path: Path, lumps: list[tuple[str, bytes]]) -> None:
    """Write a PWAD file directly (zero external dependencies)."""
    body = b""
    directory = b""
    offset = 12
    for name, data in lumps:
        uname = name.upper().encode("ascii")
        if len(uname) > 8:
            raise ValueError(f"Lump name too long: {name}")
        directory += struct.pack("<ii8s", offset, len(data), uname.ljust(8, b"\0"))
        body += data
        offset += len(data)
    header = b"PWAD" + struct.pack("<ii", len(lumps), offset)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header + body + directory)


class UDMFMapBuilder:
    """Builds a valid ZDoom UDMF PWAD from non-overlapping rectangular sectors."""

    def __init__(self, map_name: str = "MAP01") -> None:
        self.map_name = map_name.upper()
        self.sectors: list[SectorSpec] = []
        self.things: list[ThingSpec] = []

    def add_sector(self, spec: SectorSpec) -> int:
        if spec.x2 <= spec.x1 or spec.y2 <= spec.y1:
            raise ValueError(f"Invalid sector bounds: {spec}")
        idx = len(self.sectors)
        self.sectors.append(spec)
        return idx

    def add_thing(self, spec: ThingSpec) -> None:
        self.things.append(spec)

    def build_textmap(self) -> str:
        vertices: list[tuple[float, float]] = []
        v_map: dict[tuple[float, float], int] = {}

        def get_vid(x: float, y: float) -> int:
            key = (round(float(x), 3), round(float(y), 3))
            if key not in v_map:
                v_map[key] = len(vertices)
                vertices.append(key)
            return v_map[key]

        # Collect all split coordinates along each sector's 4 edges.
        # Horizontal edges (y fixed): split by x coordinates of touching sectors.
        # Vertical edges (x fixed): split by y coordinates of touching sectors.
        all_xs = sorted({s.x1 for s in self.sectors} | {s.x2 for s in self.sectors})
        all_ys = sorted({s.y1 for s in self.sectors} | {s.y2 for s in self.sectors})

        # Directed atomic edge -> (sector_idx, tex_wall, is_door, door_lock, is_exit)
        # Convention: interior of sector is to the RIGHT of directed edge p1 -> p2 (Doom front side).
        # Bottom edge (y1): (x_b, y1) -> (x_a, y1) goes left; interior (+y) is on the right?
        # Wait: in Doom (x right, y up), if vector is p1->p2, right-hand normal is (dy, -dx).
        # Let's verify:
        #   - Left edge (x=x1): p1=(x1,y1) -> p2=(x1,y2), dx=0, dy>0 => normal = (+dy, 0) -> points +x (INSIDE).
        #   - Top edge (y=y2): p1=(x1,y2) -> p2=(x2,y2), dx>0, dy=0 => normal = (0, -dx) -> points -y (INSIDE).
        #   - Right edge (x=x2): p1=(x2,y2) -> p2=(x2,y1), dx=0, dy<0 => normal = (dy, 0) -> points -x (INSIDE).
        #   - Bottom edge (y=y1): p1=(x2,y1) -> p2=(x1,y1), dx<0, dy=0 => normal = (0, -dx) -> points +y (INSIDE).
        # That matches two_rooms.wad!
        directed_edges: dict[
            tuple[tuple[float, float], tuple[float, float]], int
        ] = {}

        for s_idx, s in enumerate(self.sectors):
            xs = [x for x in all_xs if s.x1 <= x <= s.x2]
            ys = [y for y in all_ys if s.y1 <= y <= s.y2]

            # Left edge: x = s.x1, y1 -> y2
            for a, b in zip(ys[:-1], ys[1:]):
                directed_edges[((s.x1, a), (s.x1, b))] = s_idx
            # Top edge: y = s.y2, x1 -> x2
            for a, b in zip(xs[:-1], xs[1:]):
                directed_edges[((a, s.y2), (b, s.y2))] = s_idx
            # Right edge: x = s.x2, y2 -> y1
            for a, b in zip(ys[:-1], ys[1:]):
                directed_edges[((s.x2, b), (s.x2, a))] = s_idx
            # Bottom edge: y = s.y1, x2 -> x1
            for a, b in zip(xs[:-1], xs[1:]):
                directed_edges[((b, s.y1), (a, s.y1))] = s_idx

        sidedefs: list[str] = []
        linedefs: list[str] = []
        visited_pairs: set[tuple[tuple[float, float], tuple[float, float]]] = set()

        for (p1, p2), front_sec_idx in directed_edges.items():
            if (p1, p2) in visited_pairs:
                continue
            rev = (p2, p1)
            front_sec = self.sectors[front_sec_idx]

            if rev in directed_edges:
                back_sec_idx = directed_edges[rev]
                visited_pairs.add((p1, p2))
                visited_pairs.add(rev)
                back_sec = self.sectors[back_sec_idx]

                # Ensure door or exit sector special is emitted cleanly on the portal
                v1 = get_vid(*p1)
                v2 = get_vid(*p2)

                is_door_portal = front_sec.is_door or back_sec.is_door
                is_exit_portal = front_sec.is_exit or back_sec.is_exit

                if is_door_portal:
                    door_sec = front_sec if front_sec.is_door else back_sec
                    lock = door_sec.door_lock
                    door_tex = DOOR_TEXTURE_BY_LOCK.get(lock, "BIGDOOR2")
                    # Front & back sidedefs: upper texture is door face
                    sf_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {front_sec_idx}; texturetop = "{door_tex}"; texturemiddle = "-"; texturebottom = "-"; }}'
                    )
                    sb_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {back_sec_idx}; texturetop = "{door_tex}"; texturemiddle = "-"; texturebottom = "-"; }}'
                    )
                    if door_sec.pre_opened:
                        # Scaffolding mode: door already open, no special needed
                        linedefs.append(
                            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; sideback = {sb_idx}; twosided = true; }}"
                        )
                    elif lock > 0:
                        linedefs.append(
                            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; sideback = {sb_idx}; "
                            f"twosided = true; special = 13; arg0 = 0; arg1 = 16; arg2 = 150; arg3 = {lock}; "
                            f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
                        )
                    else:
                        linedefs.append(
                            f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; sideback = {sb_idx}; "
                            f"twosided = true; special = 12; arg0 = 0; arg1 = 16; arg2 = 150; "
                            f"playeruse = true; monsteruse = true; repeatspecial = true; }}"
                        )
                elif is_exit_portal:
                    sf_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {front_sec_idx}; texturetop = "-"; texturemiddle = "-"; texturebottom = "-"; }}'
                    )
                    sb_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {back_sec_idx}; texturetop = "-"; texturemiddle = "-"; texturebottom = "-"; }}'
                    )
                    linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; sideback = {sb_idx}; "
                        f"twosided = true; special = 243; playercross = true; }}"
                    )
                else:
                    # Open portal between two adjacent sectors (supports height steps)
                    upper_tex = front_sec.tex_wall
                    lower_tex = front_sec.tex_wall
                    sf_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {front_sec_idx}; texturetop = "{upper_tex}"; texturemiddle = "-"; texturebottom = "{lower_tex}"; }}'
                    )
                    sb_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {back_sec_idx}; texturetop = "{back_sec.tex_wall}"; texturemiddle = "-"; texturebottom = "{back_sec.tex_wall}"; }}'
                    )
                    linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; sideback = {sb_idx}; twosided = true; }}"
                    )
            else:
                # 1-sided outer wall
                visited_pairs.add((p1, p2))
                v1 = get_vid(*p1)
                v2 = get_vid(*p2)
                wall_tex = "DOORTRAK" if front_sec.is_door else front_sec.tex_wall
                sf_idx = len(sidedefs)
                sidedefs.append(
                    f'sidedef {{ sector = {front_sec_idx}; texturetop = "-"; texturemiddle = "{wall_tex}"; texturebottom = "-"; }}'
                )
                linedefs.append(
                    f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; blocking = true; dontpegbottom = true; }}"
                )

        # Interior pillars (void islands inside sectors: front side faces OUT of pillar, i.e., into sector)
        for s_idx, s in enumerate(self.sectors):
            for pil in s.pillars:
                # Clockwise loop around pillar so right-hand normal points OUT into the room
                pts = [
                    (pil.x1, pil.y1),
                    (pil.x2, pil.y1),
                    (pil.x2, pil.y2),
                    (pil.x1, pil.y2),
                ]
                for idx in range(4):
                    a = pts[idx]
                    b = pts[(idx + 1) % 4]
                    v1 = get_vid(*a)
                    v2 = get_vid(*b)
                    sf_idx = len(sidedefs)
                    sidedefs.append(
                        f'sidedef {{ sector = {s_idx}; texturetop = "-"; texturemiddle = "{pil.texture}"; texturebottom = "-"; }}'
                    )
                    linedefs.append(
                        f"linedef {{ v1 = {v1}; v2 = {v2}; sidefront = {sf_idx}; blocking = true; dontpegbottom = true; }}"
                    )

        lines: list[str] = ['namespace = "zdoom";']
        for x, y in vertices:
            lines.append(f"vertex {{ x = {x:.1f}; y = {y:.1f}; }}")
        for s in self.sectors:
            ceil_h = s.ceil_h if (not s.is_door or s.pre_opened) else s.floor_h
            lines.append(
                f'sector {{ texturefloor = "{s.tex_floor}"; textureceiling = "{s.tex_ceil}"; '
                f"heightfloor = {s.floor_h}; heightceiling = {ceil_h}; lightlevel = {s.light}; }}"
            )
        lines.extend(sidedefs)
        lines.extend(linedefs)
        for t in self.things:
            ambush_str = " ambush = true;" if t.ambush else ""
            lines.append(
                f"thing {{ x = {t.x:.1f}; y = {t.y:.1f}; angle = {t.angle}; type = {t.thing_type}; "
                f"skill1 = true; skill2 = true; skill3 = true; skill4 = true; skill5 = true; single = true;{ambush_str} }}"
            )
        return "\n".join(lines) + "\n"

    def write_wad(self, out_path: str | Path) -> Path:
        path = Path(out_path)
        textmap = self.build_textmap().encode("utf-8")
        lumps = [
            (self.map_name, b""),
            ("TEXTMAP", textmap),
            ("ENDMAP", b""),
        ]
        _write_wad(path, lumps)
        return path
