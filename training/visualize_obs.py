"""Live 4-panel DOOM + 3D Raycast Wall Projection + Neural Grid + Analytics Dashboard (pygame).

Supports TWO modes:
1. Live Agent Spectator (recommended while training!):
     python -m training.visualize_obs --checkpoint runs/ppo_flagship/checkpoints/latest.pt --watch --random-maps
   Renders 4 panels side-by-side:
     - Panel 1: First-person RGB view + Action pills + Hit Legend
     - Panel 2: Pseudo-3D Wireframe Raycast Wall Projection ("wall view") + 360deg Radar & Value Compass
     - Panel 3: 32x32 Egocentric Occupancy Grid + Live 9-Action Probability Bars
     - Panel 4: Live Analytics Screen (Critic V(s), Ep Return, Entropy sparklines + Ray composition + Latency)
   Automatically hot-reloads new weights whenever the checkpoint updates!

2. Manual Keyboard Mode (no --checkpoint):
     python -m training.visualize_obs --tier 3 --seed 1001
   Controls: W/S (fwd/back), A/D (strafe), Left/Right (turn), Space/F (shoot), E (use), R (new random map), Q/Esc (quit).
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

# Hit-type colors matching the live debug palette (RGB for pygame)
HIT_COLORS = {
    0: (35, 38, 46),       # none
    1: (220, 224, 234),    # wall (white/gray)
    2: (60, 165, 255),     # door (cyan/blue)
    3: (255, 200, 40),     # locked door (gold)
    4: (255, 65, 65),      # enemy (red)
    5: (60, 235, 115),     # item (green)
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


def _draw_left_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    act: np.ndarray,
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    w: int,
    h: int,
) -> None:
    rgb_h = h - 72
    rgb_surf = pygame.surfarray.make_surface(np.transpose(obs["rgb"], (1, 0, 2)))
    rgb_scaled = pygame.transform.scale(rgb_surf, (w, rgb_h))
    screen.blit(rgb_scaled, (0, 0))

    # Subtle reticle on viewport
    cx, cy = w // 2, rgb_h // 2
    pygame.draw.line(screen, (0, 255, 140), (cx - 8, cy), (cx + 8, cy), 1)
    pygame.draw.line(screen, (0, 255, 140), (cx, cy - 8), (cx, cy + 8), 1)

    # Bottom HUD bar with action pills & color legend
    pygame.draw.rect(screen, (14, 16, 22), (0, rgb_h, w, 72))
    pygame.draw.line(screen, (45, 52, 70), (0, rgb_h), (w, rgb_h), 1)

    ax = 10
    ay = rgb_h + 8
    for i, lbl in enumerate(ACT_LABELS):
        active = int(act[i]) == 1
        bg = (45, 200, 105) if active else (36, 40, 52)
        fg = (10, 15, 10) if active else (140, 148, 165)
        bw = 42
        pygame.draw.rect(screen, bg, (ax, ay, bw, 22), border_radius=3)
        screen.blit(small_font.render(lbl, True, fg), (ax + 6, ay + 4))
        ax += bw + 4

    # Color legend row
    lx = 10
    ly = rgb_h + 42
    for ht in (1, 2, 3, 4, 5):
        col = HIT_COLORS[ht]
        pygame.draw.rect(screen, col, (lx, ly + 2, 10, 10))
        txt = small_font.render(HIT_NAMES[ht], True, (190, 198, 215))
        screen.blit(txt, (lx + 14, ly))
        lx += 20 + txt.get_width() + 10


def _draw_mid_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    compass: Optional[np.ndarray],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (14, 16, 22), (x0, 0, w, h))

    # 1. Top half: Pseudo-3D Wireframe Raycast Wall Projection ("wall view") + entities
    fov_h = h // 2
    vx, vy, vw, vh = x0 + 8, 8, w - 16, fov_h - 14
    pygame.draw.rect(screen, (10, 12, 18), (vx, vy, vw, vh))

    mid_y = vy + vh // 2
    max_wall_h = vh - 18

    # Perspective ceiling/floor shading + depth reference bands
    pygame.draw.rect(screen, (15, 18, 26), (vx, vy, vw, vh // 2))
    pygame.draw.rect(screen, (22, 24, 32), (vx, mid_y, vw, vh - vh // 2))
    for d_ref in (0.25, 0.5, 0.75):
        ref_half = int(min(max_wall_h // 2, (max_wall_h // 2) * (0.14 / d_ref)))
        pygame.draw.line(screen, (32, 36, 48), (vx, mid_y - ref_half), (vx + vw, mid_y - ref_half), 1)
        pygame.draw.line(screen, (32, 36, 48), (vx, mid_y + ref_half), (vx + vw, mid_y + ref_half), 1)
    # Perspective diagonal floor/ceiling grid rays
    for frac in (-0.45, -0.22, 0.22, 0.45):
        ex = int(vx + vw * (0.5 + frac))
        pygame.draw.line(screen, (28, 32, 42), (vx + vw // 2, mid_y), (ex, vy), 1)
        pygame.draw.line(screen, (28, 32, 42), (vx + vw // 2, mid_y), (ex, vy + vh), 1)

    rays_dist = obs["rays"]["distance"]
    rays_type = obs["rays"]["hit_type"]
    # Front ~95 deg arc: indices 8..0 (left->center), 63..56 (center->right)
    fov_indices = list(range(8, -1, -1)) + list(range(63, 55, -1))
    n_cols = len(fov_indices)
    col_w = vw / float(n_cols)

    top_pts: list[tuple[int, int]] = []
    bot_pts: list[tuple[int, int]] = []
    prev_d: Optional[float] = None
    prev_ht: Optional[int] = None

    for ci, r_idx in enumerate(fov_indices):
        d = float(rays_dist[r_idx])
        ht = int(rays_type[r_idx])
        color = HIT_COLORS.get(ht, (200, 200, 200))
        wall_h = int(min(max_wall_h, max_wall_h * (0.16 / max(0.055, d))))
        shade = max(0.22, min(1.0, 1.05 - 0.78 * d))
        shaded = tuple(int(c * shade) for c in color)

        x_left = int(vx + ci * col_w)
        x_right = int(vx + (ci + 1) * col_w)
        y_top = mid_y - wall_h // 2
        y_bot = mid_y + wall_h // 2

        if ht != 0:
            pygame.draw.rect(screen, shaded, (x_left, y_top, max(2, x_right - x_left), max(2, wall_h)))
            # Bright floor/ceiling edge cap on wall strip
            edge_col = tuple(min(255, int(c * 1.15)) for c in shaded)
            pygame.draw.line(screen, edge_col, (x_left, y_top), (x_right, y_top), 2)
            pygame.draw.line(screen, edge_col, (x_left, y_bot), (x_right, y_bot), 2)

        # Draw crisp vertical wall corner edges at depth discontinuities or material switches
        if prev_d is not None and (abs(d - prev_d) > 0.08 or ht != prev_ht):
            corner_col = (240, 245, 255) if ht in (1, 2, 3) or prev_ht in (1, 2, 3) else (150, 160, 180)
            pygame.draw.line(screen, corner_col, (x_left, min(y_top, top_pts[-1][1])), (x_left, max(y_bot, bot_pts[-1][1])), 1)

        top_pts.append((x_left + max(1, (x_right - x_left) // 2), y_top))
        bot_pts.append((x_left + max(1, (x_right - x_left) // 2), y_bot))
        prev_d, prev_ht = d, ht

    if len(top_pts) >= 2:
        pygame.draw.lines(screen, (200, 210, 225), False, top_pts, 1)
        pygame.draw.lines(screen, (170, 180, 200), False, bot_pts, 1)

    # Visible entities bounding boxes + distance label
    ent_feats = obs["visible_entities"]["features"]
    ent_types = obs["visible_entities"]["entity_type"]
    ent_mask = obs["visible_entities"]["mask"]
    for i in range(len(ent_mask)):
        if not ent_mask[i]:
            continue
        dist_norm, sin_a, cos_a, _, _ = ent_feats[i]
        rel_angle = math.atan2(sin_a, cos_a)
        if abs(rel_angle) <= math.radians(48):
            frac_x = 0.5 - (rel_angle / math.radians(90))
            ex = int(vx + frac_x * vw)
            box_sz = int(min(max_wall_h * 0.72, max(12, 24 / max(0.08, dist_norm))))
            etype = int(ent_types[i])
            ecolor = (255, 65, 65) if etype in (1, 2, 3, 4) else (60, 235, 115)
            pygame.draw.rect(screen, ecolor, (ex - box_sz // 2, mid_y - box_sz // 2, box_sz, box_sz), 2)
            lbl = small_font.render(f"{dist_norm:.2f}", True, ecolor)
            screen.blit(lbl, (ex - lbl.get_width() // 2, mid_y - box_sz // 2 - 13))

    pygame.draw.rect(screen, (55, 62, 82), (vx, vy, vw, vh), 1)
    title_surf = font.render("3D Wall Projection + Entities", True, (210, 220, 235))
    pygame.draw.rect(screen, (14, 16, 23), (vx + 4, vy + 2, title_surf.get_width() + 8, title_surf.get_height() + 4), border_radius=3)
    screen.blit(title_surf, (vx + 8, vy + 4))

    # 2. Bottom half: 360deg Radar + 8-Sector Value Compass overlay
    cx = x0 + w // 2
    cy = fov_h + (h - fov_h) // 2 + 4
    radius = min(w // 2 - 20, (h - fov_h) // 2 - 18)
    pygame.draw.circle(screen, (32, 36, 48), (cx, cy), radius, 1)
    pygame.draw.circle(screen, (28, 32, 42), (cx, cy), radius // 2, 1)

    # 8-Sector Value Compass ("plus point" directional prediction)
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
                r_sec = int(radius * (0.28 + 0.68 * val_n))
                pts = [
                    (cx, cy),
                    (cx - int(math.sin(a0) * r_sec), cy - int(math.cos(a0) * r_sec)),
                    (cx - int(math.sin(ang_center) * r_sec), cy - int(math.cos(ang_center) * r_sec)),
                    (cx - int(math.sin(a1) * r_sec), cy - int(math.cos(a1) * r_sec)),
                ]
                pygame.draw.polygon(screen, (95, 72, 22), pts)
                pygame.draw.polygon(screen, (255, 195, 50), pts, 1)

    for i in range(64):
        d = float(rays_dist[i])
        ht = int(rays_type[i])
        theta = (i / 64.0) * 2.0 * math.pi
        px = cx - int(math.sin(theta) * d * radius)
        py = cy - int(math.cos(theta) * d * radius)
        col = HIT_COLORS.get(ht, (180, 180, 180))
        pygame.draw.line(screen, (col[0] // 3, col[1] // 3, col[2] // 3), (cx, cy), (px, py), 1)
        if ht != 0:
            pygame.draw.circle(screen, col, (px, py), 3)

    pygame.draw.circle(screen, (255, 255, 0), (cx, cy), 4)
    screen.blit(font.render("360deg Radar + Value Compass", True, (200, 210, 225)), (x0 + 12, fov_h + 6))


def _draw_grid_panel(
    screen: pygame.Surface,
    obs: dict[str, Any],
    act: np.ndarray,
    probs: Optional[np.ndarray],
    action_mask: Optional[np.ndarray],
    grid: Optional[np.ndarray],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (18, 20, 28), (x0, 0, w, h))
    pygame.draw.line(screen, (40, 46, 62), (x0, 0), (x0, h), 1)

    # Top: 32x32 Egocentric Occupancy Grid
    grid_box = min(w - 28, h // 2 - 22)
    gx0 = x0 + (w - grid_box) // 2
    gy0 = 26
    if grid is not None and grid.shape == (6, 32, 32):
        comp = np.zeros((32, 32, 3), dtype=np.float32)
        comp[:, :, 2] += grid[CH_VISITED] * 0.45      # blue visited
        comp[:, :, :] += grid[CH_WALL][..., None] * 0.68
        comp[:, :, 0] += grid[CH_DOOR] * 0.90
        comp[:, :, 1] += grid[CH_DOOR] * 0.90
        comp[:, :, 0] += grid[CH_LOCKED_DOOR] * 1.00
        comp[:, :, 1] += grid[CH_ITEM] * 0.85
        comp[:, :, 2] += grid[CH_ITEM] * 0.90
        comp[:, :, 1] += grid[CH_ENEMY] * 1.00
        img_u8 = (np.clip(comp, 0.0, 1.0) * 255.0).astype(np.uint8)
        surf = pygame.surfarray.make_surface(np.transpose(img_u8, (1, 0, 2)))
        scaled = pygame.transform.scale(surf, (grid_box, grid_box))
        screen.blit(scaled, (gx0, gy0))
        pygame.draw.rect(screen, (60, 70, 90), (gx0, gy0, grid_box, grid_box), 1)
        pcx, pcy = gx0 + grid_box // 2, gy0 + grid_box // 2
        pygame.draw.circle(screen, (255, 255, 0), (pcx, pcy), 4)
        pygame.draw.line(screen, (255, 255, 0), (pcx, pcy), (pcx, pcy - 11), 2)
    else:
        pygame.draw.rect(screen, (26, 28, 36), (gx0, gy0, grid_box, grid_box))

    screen.blit(font.render("32x32 Egocentric Neural Grid", True, (200, 210, 225)), (x0 + 12, 6))

    # Bottom: Policy Action Confidence Bars (9 Bernoulli outputs)
    by = gy0 + grid_box + 14
    screen.blit(font.render("Action Policy Probabilities", True, (200, 210, 225)), (x0 + 12, by))
    by += 22
    bar_x = x0 + 54
    bar_w = w - 108
    for i, lbl in enumerate(ACT_LABELS):
        p = float(probs[i]) if (probs is not None and i < len(probs)) else float(act[i])
        allowed = bool(action_mask[i]) if (action_mask is not None and i < len(action_mask)) else True
        fired = int(act[i]) == 1

        screen.blit(small_font.render(lbl, True, (210, 215, 225) if allowed else (120, 90, 90)), (x0 + 12, by))
        pygame.draw.rect(screen, (32, 36, 48), (bar_x, by + 2, bar_w, 12), border_radius=2)
        fill_w = int(bar_w * max(0.0, min(1.0, p)))
        fill_col = (55, 215, 115) if fired else ((75, 145, 235) if allowed else (120, 55, 55))
        if fill_w > 0:
            pygame.draw.rect(screen, fill_col, (bar_x, by + 2, fill_w, 12), border_radius=2)
        pct_str = "MASK" if not allowed else f"{int(round(p * 100)):3d}%"
        screen.blit(small_font.render(pct_str, True, (185, 195, 210)), (bar_x + bar_w + 6, by))
        by += 21


def _draw_sparkline(
    screen: pygame.Surface,
    vals: deque[float],
    title: str,
    color: tuple[int, int, int],
    font: pygame.font.Font,
    small_font: pygame.font.Font,
    x: int,
    y: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (22, 25, 34), (x, y, w, h), border_radius=4)
    pygame.draw.rect(screen, (48, 54, 72), (x, y, w, h), 1, border_radius=4)
    cur = vals[-1] if vals else 0.0
    screen.blit(small_font.render(f"{title}: {cur:+.2f}", True, color), (x + 6, y + 3))

    if len(vals) >= 2:
        arr = np.asarray(vals, dtype=np.float32)
        lo, hi = float(np.min(arr)), float(np.max(arr))
        span = max(1e-3, hi - lo)
        pad_x, pad_top, pad_bot = 6, 18, 6
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
    x0: int,
    w: int,
    h: int,
) -> None:
    pygame.draw.rect(screen, (14, 16, 23), (x0, 0, w, h))
    pygame.draw.line(screen, (45, 52, 70), (x0, 0), (x0, h), 1)
    screen.blit(font.render("Analytics Screen", True, (110, 215, 255)), (x0 + 12, 6))

    v = obs["vitals"]
    ly = 28
    stats = [
        f"Mode:   {hud_meta.get('mode', 'MANUAL')}",
        f"Map:    {hud_meta.get('map_label', 'MAP01')}",
        f"Ckpt:   {hud_meta.get('ckpt_step', 0):,} stp (u{hud_meta.get('ckpt_upd', 0)})",
        f"Ep #{hud_meta.get('episode', 1):<3d} Step {hud_meta.get('ep_step', 0):<4d} Exit {hud_meta.get('exits', 0)}",
        f"HP {v[0]:3.0f} | Arm {v[1]:3.0f} | Ammo {v[3]:3.0f} | Key {int(v[8]+v[9]+v[10])}",
        f"Act Latency: {hud_meta.get('act_ms', 0.0):.2f} ms | Kills {hud_meta.get('kills', 0)}",
    ]
    for s in stats:
        screen.blit(small_font.render(s, True, (215, 222, 235)), (x0 + 12, ly))
        ly += 18

    # Rolling sparklines (Critic Value, Episode Return, Policy Entropy)
    sw = w - 24
    sh = 70
    _draw_sparkline(screen, val_hist, "Critic V(s)", (100, 210, 255), font, small_font, x0 + 12, ly + 4, sw, sh)
    _draw_sparkline(screen, ret_hist, "Ep Return", (80, 235, 130), font, small_font, x0 + 12, ly + sh + 10, sw, sh)
    _draw_sparkline(screen, ent_hist, "Policy Entropy", (255, 195, 70), font, small_font, x0 + 12, ly + 2 * (sh + 10), sw, sh)

    # Ray hit composition breakdown (64 rays)
    ry = ly + 3 * (sh + 10) + 10
    rays_type = obs["rays"]["hit_type"]
    rays_dist = obs["rays"]["distance"]
    screen.blit(font.render("360deg Sensor Breakdown", True, (185, 195, 215)), (x0 + 12, ry))
    ry += 22
    for ht in (1, 2, 3, 4, 5):
        mask = rays_type == ht
        cnt = int(np.sum(mask))
        frac = cnt / 64.0
        min_d = f"{float(np.min(rays_dist[mask])):.2f}" if cnt > 0 else "--"
        col = HIT_COLORS[ht]
        screen.blit(small_font.render(f"{HIT_NAMES[ht]:<6s}", True, col), (x0 + 12, ry))
        bx, bw = x0 + 68, w - 148
        pygame.draw.rect(screen, (30, 34, 46), (bx, ry + 3, bw, 9), border_radius=2)
        if cnt > 0:
            pygame.draw.rect(screen, col, (bx, ry + 3, max(3, int(bw * frac)), 9), border_radius=2)
        screen.blit(small_font.render(f"{cnt:2d} ({min_d})", True, (180, 188, 205)), (bx + bw + 6, ry))
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
        # Force return_aux=True for spectator analytics while preserving return tuple order
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
    parser = argparse.ArgumentParser(description="4-panel DOOM live 3D wall + neural grid + analytics dashboard.")
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2, 3, 4], help="Procedural map tier.")
    parser.add_argument("--seed", type=int, default=1001, help="Initial map seed.")
    parser.add_argument("--wad", type=str, default=None, help="Optional fixed WAD path instead of MapGen.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pt checkpoint to drive DoomAgent.")
    parser.add_argument("--watch", action="store_true", help="Auto-reload checkpoint when modified during training.")
    parser.add_argument("--teacher", action="store_true", help="Watch the training-only Privileged Teacher Bot live (switches to --checkpoint once created).")
    parser.add_argument("--random-maps", action="store_true", help="Sample fresh Tier 1-4 maps on each episode.")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--fps", type=int, default=15, help="Spectator render FPS.")
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
    left_w, mid_w, grid_w, analytics_w, win_h = 420, 340, 310, 290, 496
    screen = pygame.display.set_mode((left_w + mid_w + grid_w + analytics_w, win_h))
    pygame.display.set_caption(f"DevForge DOOM Live Analytics — {map_label}")
    font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 13)
    small_font = pygame.font.SysFont("dejavusansmono,menlo,consolas,monospace", 11)
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
                    pygame.display.set_caption(f"DevForge DOOM Live Analytics — {map_label}")
                obs, _ = env.reset(seed=ep_idx)
                shaper.reset(env.sector_polygons())
                priv_prev = env.privileged_state()
                if args.teacher:
                    from training.bc.teacher_bot import PrivilegedTeacherBot
                    teacher_bot = PrivilegedTeacherBot(wad_path, env.sector_polygons())
                if agent is not None:
                    agent.reset()

            _draw_left_panel(screen, obs, act, font, small_font, left_w, win_h)
            _draw_mid_panel(screen, obs, telemetry.get("compass"), font, small_font, left_w, mid_w, win_h)
            grid = agent.odom_state.grid if (agent is not None and agent.odom_state is not None) else None
            _draw_grid_panel(
                screen,
                obs,
                act,
                telemetry.get("probs"),
                telemetry.get("action_mask"),
                grid,
                font,
                small_font,
                left_w + mid_w,
                grid_w,
                win_h,
            )
            hud_meta = {
                "mode": "AGENT WATCH" if (agent and ckpt_step > 0) else ("TEACHER LIVE" if teacher_bot else ("AGENT LIVE" if agent else "MANUAL")),
                "map_label": map_label,
                "ckpt_step": ckpt_step,
                "ckpt_upd": ckpt_upd,
                "episode": ep_idx,
                "ep_step": ep_step,
                "exits": exits_count,
                "act_ms": act_ms,
                "kills": int(priv_now.get("kills", 0)),
            }
            _draw_analytics_panel(
                screen,
                obs,
                hud_meta,
                val_hist,
                ret_hist,
                ent_hist,
                font,
                small_font,
                left_w + mid_w + grid_w,
                analytics_w,
                win_h,
            )
            pygame.display.flip()

            if args.save_screenshot and ep_step >= args.screenshot_steps:
                out_png = Path(args.save_screenshot)
                out_png.parent.mkdir(parents=True, exist_ok=True)
                pygame.image.save(screen, str(out_png))
                print(f"[Spectator] Saved screenshot: {out_png}", flush=True)
                break

            clock.tick(max(1, args.fps))
    finally:
        env.close()
        pygame.quit()
        import shutil
        shutil.rmtree(tmp_wad_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
