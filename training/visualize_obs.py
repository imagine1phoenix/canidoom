"""Live DOOM Spectator + Neural Telemetry Command Center (pygame).

Layout:
- Left: Large First-Person Game Viewport (720x540, 4:3) + Action Pills + Vitals Gauges + 3D Wireframe Sensor
- Right (Data Console Beside Game):
    - Top-Left: 360deg Polar Radar + 8-Sector Value Compass
    - Top-Right: 32x32 Egocentric Neural Occupancy Grid
    - Bottom-Left: 9-Action Policy Confidence Bars & Action Masking
    - Bottom-Right: Live Telemetry Analytics (Critic V(s), Ep Return, Policy Entropy, Ray Breakdown, Latency)

Supports TWO modes:
1. Live Agent Spectator:
     python -m training.visualize_obs --checkpoint submission/weights/best_agent.pt --watch --random-maps
2. Manual Keyboard Mode (no --checkpoint):
     python -m training.visualize_obs --tier 3 --seed 1001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pygame

try:
    import cv2  # type: ignore
except ImportError:
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "template_reference" / "env"))

from doom_env import make_env  # noqa: E402
from training.mapgen.tiers import MapGen  # noqa: E402
from training.odometry_grid import (  # noqa: E402
    CH_DOOR,
    CH_ENEMY,
    CH_ITEM,
    CH_LOCKED_DOOR,
    CH_VISITED,
    CH_WALL,
)
from training.rewards import RewardShaper  # noqa: E402

# Color Palette
CLR_BG = (10, 12, 17)
CLR_PANEL_BG = (17, 21, 30)
CLR_PANEL_HEADER = (23, 29, 42)
CLR_PANEL_BORDER = (38, 46, 64)
CLR_TEXT_PRIMARY = (230, 236, 245)
CLR_TEXT_MUTED = (140, 150, 170)
CLR_ACCENT_CYAN = (0, 215, 235)
CLR_ACCENT_GREEN = (45, 225, 115)
CLR_ACCENT_GOLD = (255, 195, 45)
CLR_ACCENT_RED = (255, 70, 70)

# Hit-type colors matching the live debug palette (RGB for pygame)
HIT_COLORS = {
    0: (35, 40, 52),       # none
    1: (220, 226, 238),    # wall (white/gray)
    2: (60, 170, 255),     # door (cyan/blue)
    3: (255, 205, 45),     # locked door (gold)
    4: (255, 70, 70),      # enemy (red)
    5: (50, 235, 115),     # item (green)
}

HIT_NAMES = {0: "None", 1: "Wall", 2: "Door", 3: "Locked", 4: "Enemy", 5: "Item"}
ACT_LABELS = ("FWD", "BAK", "L", "R", "TL", "TR", "ATK", "USE", "WPN")


def _read_keyboard_action() -> np.ndarray:
    keys = pygame.key.get_pressed()
    act = np.zeros(9, dtype=np.int8)
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        act[0] = 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        act[1] = 1
    if keys[pygame.K_a]:
        act[2] = 1
    if keys[pygame.K_d]:
        act[3] = 1
    if keys[pygame.K_LEFT]:
        act[4] = 1
    if keys[pygame.K_RIGHT]:
        act[5] = 1
    if keys[pygame.K_SPACE] or keys[pygame.K_f]:
        act[6] = 1
    if keys[pygame.K_e]:
        act[7] = 1
    if keys[pygame.K_TAB]:
        act[8] = 1
    return act


def _draw_card(screen: pygame.Surface, x: int, y: int, w: int, h: int, title: str, font: pygame.font.Font, title_color: tuple[int, int, int] = CLR_ACCENT_CYAN) -> tuple[int, int, int, int]:
    """Draw a styled dark cyberpunk card with header bar and return content rect (cx, cy, cw, ch)."""
    pygame.draw.rect(screen, CLR_PANEL_BG, (x, y, w, h), border_radius=6)
    pygame.draw.rect(screen, CLR_PANEL_BORDER, (x, y, w, h), 1, border_radius=6)
    
    # Header strip
    header_h = 28
    pygame.draw.rect(screen, CLR_PANEL_HEADER, (x + 1, y + 1, w - 2, header_h), border_top_left_radius=5, border_top_right_radius=5)
    pygame.draw.line(screen, CLR_PANEL_BORDER, (x, y + header_h), (x + w, y + header_h), 1)
    
    # Accent indicator dot
    pygame.draw.circle(screen, title_color, (x + 14, y + header_h // 2), 4)
    txt = font.render(title, True, CLR_TEXT_PRIMARY)
    screen.blit(txt, (x + 24, y + (header_h - txt.get_height()) // 2))
    
    return x + 10, y + header_h + 8, w - 20, h - header_h - 16


def _draw_game_stage(
    screen: pygame.Surface,
    obs: dict[str, Any],
    act: np.ndarray,
    hud_meta: dict[str, Any],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    bold_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Draw the primary game viewport with status header, action pills, wireframe strip, and vitals."""
    # Top banner bar
    banner_h = 32
    pygame.draw.rect(screen, (14, 18, 26), (x, y, w, banner_h), border_radius=4)
    pygame.draw.rect(screen, CLR_PANEL_BORDER, (x, y, w, banner_h), 1, border_radius=4)
    
    # Mode pill
    mode_str = hud_meta.get("mode", "AGENT LIVE")
    mode_col = CLR_ACCENT_GREEN if "AGENT" in mode_str else CLR_ACCENT_GOLD
    pygame.draw.rect(screen, mode_col, (x + 8, y + 5, 96, 22), border_radius=3)
    mode_txt = small_font.render(mode_str, True, (10, 18, 12))
    screen.blit(mode_txt, (x + 8 + (96 - mode_txt.get_width()) // 2, y + 9))
    
    map_txt = font.render(f"MAP: {hud_meta.get('map_label', 'MAP01')}", True, CLR_TEXT_PRIMARY)
    screen.blit(map_txt, (x + 116, y + 7))
    
    ckpt_txt = small_font.render(f"Ckpt: {hud_meta.get('ckpt_step', 0):,} stp (u{hud_meta.get('ckpt_upd', 0)})", True, CLR_TEXT_MUTED)
    screen.blit(ckpt_txt, (x + 280, y + 9))
    
    fps_str = hud_meta.get("fps_str", "35 FPS")
    perf_txt = small_font.render(f"Act: {hud_meta.get('act_ms', 0.0):.2f} ms | {fps_str} | Ep #{hud_meta.get('episode', 1)} Stp {hud_meta.get('ep_step', 0)}", True, CLR_ACCENT_CYAN)
    screen.blit(perf_txt, (x + w - perf_txt.get_width() - 10, y + 9))

    # Main First-Person Game Viewport (720x524)
    vp_x = x
    vp_y = y + banner_h + 8
    vp_w = w
    vp_h = 524

    if cv2 is not None:
        up = cv2.resize(obs["rgb"], (vp_w, vp_h), interpolation=cv2.INTER_CUBIC)
        blur = cv2.GaussianBlur(up, (0, 0), sigmaX=1.2)
        sharp = cv2.addWeighted(up, 1.45, blur, -0.45, 0)
        rgb_scaled = pygame.surfarray.make_surface(np.transpose(sharp, (1, 0, 2)))
    else:
        rgb_surf = pygame.surfarray.make_surface(np.transpose(obs["rgb"], (1, 0, 2)))
        rgb_scaled = pygame.transform.smoothscale(rgb_surf, (vp_w, vp_h))
    screen.blit(rgb_scaled, (vp_x, vp_y))
    pygame.draw.rect(screen, CLR_PANEL_BORDER, (vp_x, vp_y, vp_w, vp_h), 1)

    # Crosshair reticle
    cx, cy = vp_x + vp_w // 2, vp_y + vp_h // 2
    pygame.draw.line(screen, (0, 255, 140), (cx - 10, cy), (cx + 10, cy), 1)
    pygame.draw.line(screen, (0, 255, 140), (cx, cy - 10), (cx, cy + 10), 1)
    pygame.draw.circle(screen, (0, 255, 140), (cx, cy), 2)

    # Bottom Dashboard Deck (Under Game Viewport)
    deck_y = vp_y + vp_h + 8
    deck_h = h - (deck_y - y)
    pygame.draw.rect(screen, CLR_PANEL_BG, (x, deck_y, w, deck_h), border_radius=6)
    pygame.draw.rect(screen, CLR_PANEL_BORDER, (x, deck_y, w, deck_h), 1, border_radius=6)

    # Left Section of Deck: Action Pills & Hit Legend
    deck_left_w = 340
    pills_y = deck_y + 12
    screen.blit(bold_font.render("ACTIVE ACTIONS:", True, CLR_TEXT_MUTED), (x + 12, pills_y))
    
    ax = x + 12
    ay = pills_y + 22
    for i, lbl in enumerate(ACT_LABELS):
        active = int(act[i]) == 1
        bg = CLR_ACCENT_GREEN if active else (28, 34, 46)
        fg = (10, 20, 12) if active else (130, 140, 160)
        bw = 34
        pygame.draw.rect(screen, bg, (ax, ay, bw, 22), border_radius=3)
        lbl_surf = small_font.render(lbl, True, fg)
        screen.blit(lbl_surf, (ax + (bw - lbl_surf.get_width()) // 2, ay + 4))
        ax += bw + 3

    # Color Legend Row
    lx = x + 12
    ly = ay + 30
    screen.blit(small_font.render("Hit:", True, CLR_TEXT_MUTED), (lx, ly))
    lx += 32
    for ht in (1, 2, 3, 4, 5):
        col = HIT_COLORS[ht]
        pygame.draw.rect(screen, col, (lx, ly + 2, 9, 9), border_radius=2)
        txt = small_font.render(HIT_NAMES[ht], True, CLR_TEXT_PRIMARY)
        screen.blit(txt, (lx + 12, ly))
        lx += 14 + txt.get_width() + 6

    # 3D Wireframe Raycast Preview Strip (Miniaturized Sensor Sight)
    wf_x = x + 12
    header_txt = small_font.render("3D Raycast Depth Sight", True, CLR_TEXT_MUTED)
    screen.blit(header_txt, (wf_x, ly + 20))
    wf_y = ly + 35
    wf_w = deck_left_w - 24
    wf_h = deck_h - (wf_y - deck_y) - 10
    pygame.draw.rect(screen, (10, 12, 18), (wf_x, wf_y, wf_w, wf_h), border_radius=3)
    pygame.draw.rect(screen, (34, 40, 56), (wf_x, wf_y, wf_w, wf_h), 1, border_radius=3)

    rays_dist = obs["rays"]["distance"]
    rays_type = obs["rays"]["hit_type"]
    fov_indices = list(range(8, -1, -1)) + list(range(63, 55, -1))
    col_w = wf_w / float(len(fov_indices))
    mid_wf_y = wf_y + wf_h // 2
    for ci, r_idx in enumerate(fov_indices):
        d = float(rays_dist[r_idx])
        ht = int(rays_type[r_idx])
        wall_h = int(min(wf_h - 4, (wf_h - 4) * (0.16 / max(0.06, d))))
        shade = max(0.3, min(1.0, 1.05 - 0.78 * d))
        col = tuple(int(c * shade) for c in HIT_COLORS.get(ht, (180, 180, 180)))
        rx = int(wf_x + ci * col_w)
        if ht != 0:
            pygame.draw.rect(screen, col, (rx, mid_wf_y - wall_h // 2, max(2, int(col_w)), max(2, wall_h)))

    # Divider Line between Left & Right Deck
    pygame.draw.line(screen, CLR_PANEL_BORDER, (x + deck_left_w + 6, deck_y + 10), (x + deck_left_w + 6, deck_y + deck_h - 10), 1)

    # Right Section of Deck: High-Visibility Player Vitals & Mission Status
    vx0 = x + deck_left_w + 20
    vw = w - (deck_left_w + 32)
    v = obs["vitals"]
    
    # Row 1: Vitals Gauges (Health, Armor, Ammo)
    hp = float(v[0])
    armor = float(v[1])
    ammo = float(v[3])
    
    vy = deck_y + 12
    # Health bar
    hp_col = CLR_ACCENT_GREEN if hp > 50 else (CLR_ACCENT_GOLD if hp > 25 else CLR_ACCENT_RED)
    screen.blit(bold_font.render(f"HEALTH: {hp:3.0f}%", True, hp_col), (vx0, vy))
    pygame.draw.rect(screen, (30, 36, 48), (vx0, vy + 20, 170, 12), border_radius=2)
    pygame.draw.rect(screen, hp_col, (vx0, vy + 20, int(170 * max(0.0, min(1.0, hp / 100.0))), 12), border_radius=2)

    # Armor bar
    screen.blit(bold_font.render(f"ARMOR: {armor:3.0f}%", True, CLR_ACCENT_CYAN), (vx0 + 190, vy))
    pygame.draw.rect(screen, (30, 36, 48), (vx0 + 190, vy + 20, 160, 12), border_radius=2)
    pygame.draw.rect(screen, CLR_ACCENT_CYAN, (vx0 + 190, vy + 20, int(160 * max(0.0, min(1.0, armor / 100.0))), 12), border_radius=2)

    # Ammo bar
    vy += 42
    screen.blit(bold_font.render(f"AMMO: {ammo:3.0f}", True, CLR_ACCENT_GOLD), (vx0, vy))
    pygame.draw.rect(screen, (30, 36, 48), (vx0, vy + 20, 170, 12), border_radius=2)
    pygame.draw.rect(screen, CLR_ACCENT_GOLD, (vx0, vy + 20, int(170 * max(0.0, min(1.0, ammo / 50.0))), 12), border_radius=2)

    # Keycards Inventory Display
    screen.blit(bold_font.render("KEYCARDS:", True, CLR_TEXT_PRIMARY), (vx0 + 190, vy))
    has_blue = int(v[8]) > 0
    has_red = int(v[9]) > 0
    has_yellow = int(v[10]) > 0
    
    kx = vx0 + 190
    ky = vy + 20
    for key_name, has_k, col in (("BLUE", has_blue, (60, 150, 255)), ("RED", has_red, (255, 60, 60)), ("YELLOW", has_yellow, (255, 210, 40))):
        k_bg = col if has_k else (32, 38, 50)
        k_fg = (10, 15, 20) if has_k else (100, 110, 130)
        pygame.draw.rect(screen, k_bg, (kx, ky, 50, 16), border_radius=3)
        ktxt = small_font.render(key_name, True, k_fg)
        screen.blit(ktxt, (kx + (50 - ktxt.get_width()) // 2, ky + 1))
        kx += 55

    # Mission Telemetry Summary Pills
    vy += 42
    stats_box = (
        (f"KILLS: {hud_meta.get('kills', 0)}", CLR_ACCENT_RED),
        (f"EXITS: {hud_meta.get('exits', 0)}", CLR_ACCENT_GREEN),
        (f"STEPS: {hud_meta.get('ep_step', 0)}", CLR_TEXT_PRIMARY),
    )
    sx = vx0
    for s_title, s_col in stats_box:
        pygame.draw.rect(screen, (24, 30, 42), (sx, vy, 110, 24), border_radius=3)
        pygame.draw.rect(screen, CLR_PANEL_BORDER, (sx, vy, 110, 24), 1, border_radius=3)
        stxt = bold_font.render(s_title, True, s_col)
        screen.blit(stxt, (sx + (110 - stxt.get_width()) // 2, vy + 4))
        sx += 120


def _draw_radar_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    compass: Optional[np.ndarray],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Card 1: 360deg Polar Radar + 8-Sector Value Compass."""
    cx0, cy0, cw, ch = _draw_card(screen, x, y, w, h, "360° Radar & Value Compass", font, CLR_ACCENT_GOLD)

    radar_cx = cx0 + cw // 2
    radar_cy = cy0 + ch // 2 + 6
    radius = min(cw // 2 - 14, ch // 2 - 14)

    # Concentric distance rings
    pygame.draw.circle(screen, (28, 34, 48), (radar_cx, radar_cy), radius, 1)
    pygame.draw.circle(screen, (24, 28, 40), (radar_cx, radar_cy), int(radius * 0.66), 1)
    pygame.draw.circle(screen, (20, 24, 34), (radar_cx, radar_cy), int(radius * 0.33), 1)

    # Cross axis
    pygame.draw.line(screen, (28, 34, 48), (radar_cx - radius, radar_cy), (radar_cx + radius, radar_cy), 1)
    pygame.draw.line(screen, (28, 34, 48), (radar_cx, radar_cy - radius), (radar_cx, radar_cy + radius), 1)

    # 8-Sector Value Compass ("plus-point" directional advantage)
    if compass is not None and len(compass) == 8:
        c_pos = np.maximum(0.0, compass)
        c_max = float(np.max(c_pos))
        if c_max > 1e-6:
            c_norm = c_pos / c_max
            for sec in range(8):
                val_n = float(c_norm[sec])
                if val_n < 0.08:
                    continue
                ang_center = sec * (math.pi / 4.0)
                a0 = ang_center - math.pi / 8.0
                a1 = ang_center + math.pi / 8.0
                r_sec = int(radius * (0.25 + 0.72 * val_n))
                pts = [
                    (radar_cx, radar_cy),
                    (radar_cx - int(math.sin(a0) * r_sec), radar_cy - int(math.cos(a0) * r_sec)),
                    (radar_cx - int(math.sin(ang_center) * r_sec), radar_cy - int(math.cos(ang_center) * r_sec)),
                    (radar_cx - int(math.sin(a1) * r_sec), radar_cy - int(math.cos(a1) * r_sec)),
                ]
                pygame.draw.polygon(screen, (100, 75, 20), pts)
                pygame.draw.polygon(screen, CLR_ACCENT_GOLD, pts, 1)

    # 64 Radar depth rays
    rays_dist = obs["rays"]["distance"]
    rays_type = obs["rays"]["hit_type"]
    for i in range(64):
        d = float(rays_dist[i])
        ht = int(rays_type[i])
        theta = (i / 64.0) * 2.0 * math.pi
        px = radar_cx - int(math.sin(theta) * d * radius)
        py = radar_cy - int(math.cos(theta) * d * radius)
        col = HIT_COLORS.get(ht, (180, 180, 180))
        pygame.draw.line(screen, (col[0] // 3, col[1] // 3, col[2] // 3), (radar_cx, radar_cy), (px, py), 1)
        if ht != 0:
            pygame.draw.circle(screen, col, (px, py), 3)

    # Center agent marker
    pygame.draw.circle(screen, (255, 255, 0), (radar_cx, radar_cy), 4)
    # Forward heading arrow
    pygame.draw.line(screen, (255, 255, 0), (radar_cx, radar_cy), (radar_cx, radar_cy - 12), 2)


def _draw_neural_grid_panel(
    screen: pygame.Surface,
    grid: Optional[np.ndarray],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Card 2: 32x32 Egocentric Neural Occupancy Grid."""
    cx0, cy0, cw, ch = _draw_card(screen, x, y, w, h, "32×32 Egocentric Neural Grid", font, CLR_ACCENT_CYAN)

    grid_box = min(cw - 10, ch - 26)
    gx0 = cx0 + (cw - grid_box) // 2
    gy0 = cy0 + (ch - grid_box) // 2 - 4

    if grid is not None and grid.shape == (6, 32, 32):
        comp = np.zeros((32, 32, 3), dtype=np.float32)
        comp[:, :, 2] += grid[CH_VISITED] * 0.50      # blue visited floor
        comp[:, :, :] += grid[CH_WALL][..., None] * 0.70  # white walls
        comp[:, :, 0] += grid[CH_DOOR] * 0.90
        comp[:, :, 1] += grid[CH_DOOR] * 0.90         # yellow unlocked door
        comp[:, :, 0] += grid[CH_LOCKED_DOOR] * 1.00  # red locked door
        comp[:, :, 1] += grid[CH_ITEM] * 0.85
        comp[:, :, 2] += grid[CH_ITEM] * 0.90         # cyan item
        comp[:, :, 0] += grid[CH_ENEMY] * 1.00        # red enemy
        comp[:, :, 1] += grid[CH_ENEMY] * 0.20
        img_u8 = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(img_u8, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, (grid_box, grid_box))
        screen.blit(scaled, (gx0, gy0))
        pygame.draw.rect(screen, CLR_PANEL_BORDER, (gx0, gy0, grid_box, grid_box), 1)

        # Agent position & heading
        pcx, pcy = gx0 + grid_box // 2, gy0 + grid_box // 2
        pygame.draw.circle(screen, (255, 255, 0), (pcx, pcy), 4)
        pygame.draw.line(screen, (255, 255, 0), (pcx, pcy), (pcx, pcy - 12), 2)
    else:
        pygame.draw.rect(screen, (20, 24, 32), (gx0, gy0, grid_box, grid_box))
        screen.blit(small_font.render("Awaiting Grid State...", True, CLR_TEXT_MUTED), (gx0 + 20, gy0 + grid_box // 2))

    footer_lbl = small_font.render("Coordinate-Free Scan-Matched Memory", True, CLR_TEXT_MUTED)
    screen.blit(footer_lbl, (cx0 + (cw - footer_lbl.get_width()) // 2, cy0 + ch - 16))


def _draw_policy_panel(
    screen: pygame.Surface,
    act: np.ndarray,
    probs: Optional[np.ndarray],
    action_mask: Optional[np.ndarray],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Card 3: 9-Action Policy Probabilities & Action Masks."""
    cx0, cy0, cw, ch = _draw_card(screen, x, y, w, h, "Action Policy Probabilities", font, CLR_ACCENT_GREEN)

    by = cy0 + 6
    bar_x = cx0 + 50
    bar_w = cw - 108

    for i, lbl in enumerate(ACT_LABELS):
        p = float(probs[i]) if (probs is not None and i < len(probs)) else float(act[i])
        allowed = bool(action_mask[i]) if (action_mask is not None and i < len(action_mask)) else True
        fired = int(act[i]) == 1

        label_col = CLR_TEXT_PRIMARY if allowed else (130, 95, 95)
        screen.blit(small_font.render(lbl, True, label_col), (cx0 + 8, by))
        
        # Background bar
        pygame.draw.rect(screen, (28, 34, 46), (bar_x, by + 2, bar_w, 14), border_radius=3)
        
        fill_w = int(bar_w * max(0.0, min(1.0, p)))
        fill_col = CLR_ACCENT_GREEN if fired else ((70, 145, 235) if allowed else (130, 55, 55))
        if fill_w > 0:
            pygame.draw.rect(screen, fill_col, (bar_x, by + 2, fill_w, 14), border_radius=3)
            
        pct_str = "MASK" if not allowed else f"{int(round(p * 100)):3d}%"
        pct_col = (255, 80, 80) if not allowed else CLR_TEXT_PRIMARY
        screen.blit(small_font.render(pct_str, True, pct_col), (bar_x + bar_w + 8, by))
        by += 26


def _draw_sparkline(
    screen: pygame.Surface,
    vals: deque[float],
    title: str,
    color: tuple[int, int, int],
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (20, 24, 34), (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, (38, 46, 64), (x, y, w, h), 1, border_radius=4)
    cur = vals[-1] if vals else 0.0
    screen.blit(small_font.render(f"{title}: {cur:+.2f}", True, color), (x + 8, y + 4))

    if len(vals) >= 2:
        arr = np.asarray(vals, dtype=np.float32)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        span = max(1e-3, hi - lo)
        pad_x, pad_top, pad_bot = 8, 20, 6
        gx_w = max(1, w - 2 * pad_x)
        gy_h = max(1, h - pad_top - pad_bot)
        pts = []
        n = len(arr)
        for idx, val in enumerate(arr):
            px = x + pad_x + int((idx / float(max(1, n - 1))) * gx_w)
            py = y + pad_top + int((1.0 - (float(val) - lo) / span) * gy_h)
            pts.append((px, py))
        pygame.draw.lines(screen, color, False, pts, 2)


def _draw_analytics_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    hud_meta: dict[str, Any],
    val_hist: deque[float],
    ret_hist: deque[float],
    ent_hist: deque[float],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    """Card 4: Live Telemetry Analytics & Sensor Breakdown."""
    cx0, cy0, cw, ch = _draw_card(screen, x, y, w, h, "Learning Dynamics & Sensors", font, CLR_ACCENT_CYAN)

    # Rolling Sparklines (Critic Value, Return, Entropy)
    sh = 48
    gap = 8
    _draw_sparkline(screen, val_hist, "Critic V(s)", CLR_ACCENT_CYAN, small_font, cx0, cy0 + 2, cw, sh)
    _draw_sparkline(screen, ret_hist, "Ep Return", CLR_ACCENT_GREEN, small_font, cx0, cy0 + sh + gap, cw, sh)
    _draw_sparkline(screen, ent_hist, "Policy Entropy", CLR_ACCENT_GOLD, small_font, cx0, cy0 + 2 * (sh + gap), cw, sh)

    # 360deg Sensor Breakdown
    ry = cy0 + 3 * (sh + gap) + 12
    screen.blit(font.render("360° Sensor Breakdown", True, CLR_TEXT_PRIMARY), (cx0, ry))
    ry += 22

    rays_type = obs["rays"]["hit_type"]
    rays_dist = obs["rays"]["distance"]
    for ht in (1, 2, 3, 4, 5):
        mask = rays_type == ht
        cnt = int(np.sum(mask))
        frac = cnt / 64.0
        min_d = f"{float(np.min(rays_dist[mask])):.2f}" if cnt > 0 else "--"
        col = HIT_COLORS[ht]
        
        screen.blit(small_font.render(f"{HIT_NAMES[ht]:<6s}", True, col), (cx0, ry))
        bx = cx0 + 64
        bw = cw - 138
        pygame.draw.rect(screen, (28, 34, 46), (bx, ry + 3, bw, 10), border_radius=2)
        if cnt > 0:
            pygame.draw.rect(screen, col, (bx, ry + 3, max(4, int(bw * frac)), 10), border_radius=2)
        screen.blit(small_font.render(f"{cnt:2d} ({min_d})", True, CLR_TEXT_PRIMARY), (bx + bw + 8, ry))
        ry += 18


def _load_agent_weights(agent: Any, ckpt_path: Path) -> tuple[int, int]:
    import torch

    ckpt = torch.load(ckpt_path, map_location=agent.device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt)
    agent.model.load_state_dict(state_dict, strict=False)
    agent.model.eval()
    return int(ckpt.get("global_step", 0)), int(ckpt.get("update_idx", 0))


def _attach_telemetry_hook(agent: Any, telemetry: dict[str, Any]) -> None:
    """Wrap agent.model.forward so agent.act() records logits, V(s), entropy, and value_compass."""
    import torch

    orig_forward = agent.model.forward

    def _hooked_forward(*args: Any, **kwargs: Any) -> Any:
        kwargs["return_aux"] = True
        out = orig_forward(*args, **kwargs)
        logits, value, item2, item3 = out
        aux = item2 if isinstance(item2, dict) else (item3 if isinstance(item3, dict) else None)
        with torch.no_grad():
            probs = torch.sigmoid(logits.squeeze(0)).clamp(1e-6, 1.0 - 1e-6)
            ent = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs)).sum().item()
            telemetry["probs"] = probs.cpu().numpy()
            telemetry["value"] = float(value.reshape(-1)[0].item())
            telemetry["entropy"] = float(ent)
            if isinstance(aux, dict) and "value_compass" in aux:
                telemetry["compass"] = aux["value_compass"].squeeze(0).cpu().numpy()
            if args and hasattr(args[0], "action_mask"):
                telemetry["action_mask"] = args[0].action_mask.squeeze(0).cpu().numpy()
            elif "inputs" in kwargs and hasattr(kwargs["inputs"], "action_mask"):
                telemetry["action_mask"] = kwargs["inputs"].action_mask.squeeze(0).cpu().numpy()
        return out

    agent.model.forward = _hooked_forward


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="DOOM Spectator + Neural Telemetry Dashboard.")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3, 4], help="Procedural map tier.")
    parser.add_argument("--seed", type=int, default=1001, help="Initial map seed.")
    parser.add_argument("--wad", type=str, default=None, help="Optional fixed WAD path instead of MapGen.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint to drive DoomAgent.")
    parser.add_argument("--watch", action="store_true", help="Auto-reload checkpoint when modified during training.")
    parser.add_argument("--teacher", action="store_true", help="Watch the training-only Privileged Teacher Bot live.")
    parser.add_argument("--random-maps", action="store_true", help="Sample fresh Tier 1-4 maps on each episode.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fps", type=int, default=35, help="Spectator render FPS (default: 35 for native DOOM speed; set 0 for uncapped).")
    parser.add_argument("--save-screenshot", type=str, default=None, help="Run N steps and save PNG frame then exit.")
    parser.add_argument("--screenshot-steps", type=int, default=12, help="Steps before --save-screenshot captures.")
    args = parser.parse_args(argv)

    tmp_wad_dir = Path(tempfile.mkdtemp(prefix="doom-vis-"))
    rng = np.random.default_rng(int(time.time()) % 100_000)

    agent = None
    telemetry: dict[str, Any] = {}
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    stage_file = (ckpt_path.parent.parent / "stage_state.json") if ckpt_path is not None else None
    last_mtime = -1.0
    last_stage_idx: Optional[int] = None
    ckpt_step, ckpt_upd = 0, 0

    def _read_stage_state() -> Optional[dict[str, Any]]:
        if stage_file is None or not stage_file.is_file():
            return None
        try:
            return json.loads(stage_file.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _next_wad(ep_idx: int) -> tuple[Path, str]:
        nonlocal last_stage_idx
        if args.wad:
            return Path(args.wad), Path(args.wad).name
        st = _read_stage_state()
        if st is not None:
            last_stage_idx = int(st.get("stage_idx", 0))
            st_name = str(st.get("stage", "staged"))
            tiers = st.get("tiers") or [1]
            knobs = {**dict(st.get("knobs") or {}), "out_dir": str(tmp_wad_dir)}
            alpha = float(st.get("scaffolding_alpha", 0.0))
            if st.get("fixed_seed") is not None:
                seed = int(st["fixed_seed"])
                tier = int(tiers[0])
                wad_p, _ = MapGen.sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=alpha)
                return Path(wad_p), f"Stage {last_stage_idx}:{st_name} (s={seed})"
            tier = int(rng.choice(tiers))
            seed = int(rng.integers(200_000, 999_999))
            wad_p, _ = MapGen.sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=alpha)
            return Path(wad_p), f"Stage {last_stage_idx}:{st_name} T{tier} s={seed}"
        if args.random_maps:
            tier = int(rng.choice([1, 2, 2, 3, 4]))
            seed = int(rng.integers(200_000, 999_999))
        else:
            tier = args.tier
            seed = args.seed + ep_idx - 1
        wad_p, _ = MapGen.sample(tier=tier, knobs={"out_dir": str(tmp_wad_dir)}, seed=seed)
        return Path(wad_p), f"T{tier} s={seed}"

    if ckpt_path is not None:
        from gymnasium.spaces import Box, MultiBinary
        from submission.agent import DoomAgent

        agent = DoomAgent(Box(-np.inf, np.inf, (1,), np.float32), MultiBinary(9), device=args.device)
        _attach_telemetry_hook(agent, telemetry)
        if ckpt_path.is_file():
            try:
                ckpt_step, ckpt_upd = _load_agent_weights(agent, ckpt_path)
                last_mtime = ckpt_path.stat().st_mtime
                print(f"[Spectator] Loaded {ckpt_path} (step={ckpt_step:,}, upd={ckpt_upd})", flush=True)
            except Exception as exc:
                print(f"[Spectator] Waiting for readable checkpoint ({exc})...", flush=True)

    ep_idx = 1
    exits_count = 0
    ep_step = 0
    ep_ret = 0.0
    wad_path, map_label = _next_wad(ep_idx)
    env = make_env(wad_path, "MAP01", episode_ticks=2100, render_buffers=True)

    pygame.init()
    
    # High-Definition 1420x760 Command Center Layout
    # Left: 740px Game Screen Stage
    # Right: 660px Telemetry Console Grid (Two 320px columns)
    win_w = 1420
    win_h = 760
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"DevForge DOOM Neural Command Center — {map_label}")
    
    font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 13)
    small_font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 11)
    bold_font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 12, bold=True)
    clock = pygame.time.Clock()

    shaper = RewardShaper()
    obs, _ = env.reset(seed=0)
    shaper.reset(env.sector_polygons())
    priv_prev = env.privileged_state()
    teacher_bot = None
    if args.teacher:
        from training.bc.teacher_bot import PrivilegedTeacherBot
        teacher_bot = PrivilegedTeacherBot(wad_path, env.sector_polygons())
    if agent is not None:
        agent.reset()

    val_hist: deque[float] = deque(maxlen=90)
    ret_hist: deque[float] = deque(maxlen=90)
    ent_hist: deque[float] = deque(maxlen=90)

    last_watch_check = time.perf_counter()
    current_fps = max(0, args.fps)
    paused = False
    running = True
    try:
        while running:
            force_new_map = False
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        running = False
                    elif event.key == pygame.K_r:
                        force_new_map = True
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                    elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_RIGHTBRACKET, pygame.K_UP):
                        current_fps = min(120, (35 if current_fps == 0 else current_fps + 10))
                    elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET, pygame.K_DOWN):
                        current_fps = max(5, (35 if current_fps == 0 else current_fps - 5))
                    elif event.key in (pygame.K_0, pygame.K_u):
                        current_fps = 0  # Uncapped max speed!

            if paused:
                clock.tick(15)
                continue

            # Hot-reload checkpoint every ~2 seconds if --watch is active
            if agent is not None and args.watch and ckpt_path is not None:
                now = time.perf_counter()
                if now - last_watch_check >= 2.0:
                    last_watch_check = now
                    if ckpt_path.is_file():
                        try:
                            mt = ckpt_path.stat().st_mtime
                            if mt > last_mtime:
                                ckpt_step, ckpt_upd = _load_agent_weights(agent, ckpt_path)
                                last_mtime = mt
                                print(f"[Spectator] Hot-reloaded weights -> step={ckpt_step:,} upd={ckpt_upd}", flush=True)
                        except Exception:
                            pass
                    if not args.wad:
                        st_now = _read_stage_state()
                        if st_now is not None and int(st_now.get("stage_idx", 0)) != last_stage_idx:
                            force_new_map = True

            t0 = time.perf_counter()
            if agent is not None and ckpt_step > 0:
                act = agent.act(obs)
            elif teacher_bot is not None:
                act = teacher_bot.act(obs, priv_prev)
            elif agent is not None:
                act = agent.act(obs)
            else:
                act = _read_keyboard_action()
            act_ms = (time.perf_counter() - t0) * 1000.0
            obs, _, terminated, truncated, info = env.step(act)
            priv_now = env.privileged_state()
            step_r = float(shaper.step(priv_prev, priv_now, act, info))
            priv_prev = priv_now
            ep_ret += step_r
            ep_step += 1

            val_hist.append(float(telemetry.get("value", 0.0)))
            ret_hist.append(ep_ret)
            ent_hist.append(float(telemetry.get("entropy", 0.0)))

            if terminated or truncated or force_new_map:
                if info.get("exit_reached", False):
                    exits_count += 1
                ep_idx += 1
                ep_step = 0
                ep_ret = 0.0
                if args.random_maps or force_new_map:
                    env.close()
                    wad_path, map_label = _next_wad(ep_idx)
                    env = make_env(wad_path, "MAP01", episode_ticks=2100, render_buffers=True)
                    pygame.display.set_caption(f"DevForge DOOM Neural Command Center — {map_label}")
                obs, _ = env.reset(seed=ep_idx)
                shaper.reset(env.sector_polygons())
                priv_prev = env.privileged_state()
                if args.teacher:
                    from training.bc.teacher_bot import PrivilegedTeacherBot
                    teacher_bot = PrivilegedTeacherBot(wad_path, env.sector_polygons())
                if agent is not None:
                    agent.reset()

            # Fill master canvas background
            screen.fill(CLR_BG)

            hud_meta = {
                "mode": "AGENT WATCH" if (agent and ckpt_step > 0) else ("TEACHER LIVE" if teacher_bot else ("AGENT LIVE" if agent else "MANUAL")),
                "map_label": map_label,
                "ckpt_step": ckpt_step,
                "ckpt_upd": ckpt_upd,
                "episode": ep_idx,
                "ep_step": ep_step,
                "exits": exits_count,
                "act_ms": act_ms,
                "fps_str": (f"{current_fps} FPS (+/-)" if current_fps > 0 else "MAX SPEED [0]") + (" [PAUSED]" if paused else ""),
                "kills": int(priv_now.get("kills", 0)),
            }

            # 1. Primary Left Stage: Large First-Person Game Viewport + HUD Controls
            stage_w = 730
            _draw_game_stage(screen, obs, act, hud_meta, font, small_font, bold_font, 10, 10, stage_w, win_h - 20)

            # 2. Right Telemetry Console Beside Game Screen: 2 Columns of Cards
            right_x0 = 10 + stage_w + 12
            card_w = 328
            card_h = (win_h - 20 - 12) // 2  # ~364px

            # Top-Left: Radar & Compass
            _draw_radar_panel(
                screen,
                obs,
                telemetry.get("compass"),
                font,
                small_font,
                right_x0,
                10,
                card_w,
                card_h,
            )

            # Top-Right: Neural Occupancy Grid
            grid = agent.odom_state.grid if (agent is not None and agent.odom_state is not None) else None
            _draw_neural_grid_panel(
                screen,
                grid,
                font,
                small_font,
                right_x0 + card_w + 12,
                10,
                card_w,
                card_h,
            )

            # Bottom-Left: Action Policy Probabilities
            _draw_policy_panel(
                screen,
                act,
                telemetry.get("probs"),
                telemetry.get("action_mask"),
                font,
                small_font,
                right_x0,
                10 + card_h + 12,
                card_w,
                card_h,
            )

            # Bottom-Right: Learning Dynamics & 360° Sensor Breakdown
            _draw_analytics_panel(
                screen,
                obs,
                hud_meta,
                val_hist,
                ret_hist,
                ent_hist,
                font,
                small_font,
                right_x0 + card_w + 12,
                10 + card_h + 12,
                card_w,
                card_h,
            )

            pygame.display.flip()

            if args.save_screenshot and ep_step >= args.screenshot_steps:
                out_png = Path(args.save_screenshot)
                out_png.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(out_png))
                print(f"[Spectator] Saved screenshot: {out_png}", flush=True)
                break

            if current_fps > 0:
                clock.tick(current_fps)
    finally:
        env.close()
        pygame.quit()
        import shutil
        shutil.rmtree(tmp_wad_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
