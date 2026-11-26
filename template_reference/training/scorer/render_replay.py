"""Side-by-Side Brain-View Replay Renderer for Stage 2 Desk Review.

Renders high-impact visual MP4 replays containing 3 synchronized panels:
1. First-Person RGB View (with action & health overlay)
2. 360° Polar Ray Radar (color-coded by hit type: wall, door, locked_door, enemy, item)
3. 32x32 Egocentric Occupancy Grid (live topological neural memory map)

Usage:
    python -m training.scorer.render_replay --wad maps/two_rooms.wad --output replay.mp4
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure repo root is on sys.path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.agent import DoomAgent
from training.odometry_grid import (
    GRID_SIZE,
    NUM_RAYS,
    RAY_ANGLE_STEP,
    CH_VISITED,
    CH_WALL,
    CH_DOOR,
    CH_LOCKED_DOOR,
    CH_ITEM,
    CH_ENEMY,
)

ACTION_NAMES = (
    "FWD", "BACK", "STRAFE_L", "STRAFE_R", "TURN_L", "TURN_R", "ATTACK", "USE", "WEAPON"
)


def draw_polar_radar(rays: dict, width: int = 320, height: int = 240) -> np.ndarray:
    """Renders a top-down polar radar of the 64 rays."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = width // 2, height // 2
    max_radius = min(cx, cy) - 15

    # Draw range rings
    for r_frac in (0.33, 0.66, 1.0):
        cv2.circle(canvas, (cx, cy), int(max_radius * r_frac), (40, 40, 40), 1)

    # Crosshair axes
    cv2.line(canvas, (cx, cy - max_radius), (cx, cy + max_radius), (50, 50, 50), 1)
    cv2.line(canvas, (cx - max_radius, cy), (cx + max_radius, cy), (50, 50, 50), 1)

    dists = rays.get("distance", np.ones(NUM_RAYS, dtype=np.float32))
    types = rays.get("hit_type", np.zeros(NUM_RAYS, dtype=np.int64))

    # Color map for hit types: (B, G, R)
    COLOR_MAP = {
        0: (60, 60, 60),      # none (dark)
        1: (180, 180, 180),  # wall (white/gray)
        2: (0, 215, 255),    # door (yellow)
        3: (0, 0, 255),      # locked door (red)
        4: (0, 255, 0),      # enemy (green)
        5: (255, 200, 0),    # item (cyan/blue)
    }

    # Plot rays
    for i in range(NUM_RAYS):
        angle = i * RAY_ANGLE_STEP
        # In radar view: ray 0 is straight UP (y decreases)
        d_frac = min(1.0, float(dists[i]))
        r = max_radius * d_frac
        # Angle 0 is forward (-y), 90 deg is left (-x)
        px = int(cx - r * math.sin(angle))
        py = int(cy - r * math.cos(angle))

        h_type = int(types[i])
        color = COLOR_MAP.get(h_type, (150, 150, 150))
        cv2.line(canvas, (cx, cy), (px, py), color, 1)
        cv2.circle(canvas, (px, py), 2, color, -1)

    # Player position
    cv2.circle(canvas, (cx, cy), 4, (0, 255, 255), -1)
    cv2.putText(canvas, "360deg Ray Radar", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return canvas


def draw_grid_map(grid: np.ndarray, width: int = 320, height: int = 240) -> np.ndarray:
    """Renders the 32x32 egocentric occupancy grid as an upscaled color map."""
    # Build RGB composite from 6 channels
    # Grid shape: (6, 32, 32)
    composite = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)

    # Visited: dark blue
    composite[:, :, 0] += grid[CH_VISITED] * 0.4
    # Wall: gray
    composite[:, :, 0] += grid[CH_WALL] * 0.6
    composite[:, :, 1] += grid[CH_WALL] * 0.6
    composite[:, :, 2] += grid[CH_WALL] * 0.6
    # Door: yellow (G + R)
    composite[:, :, 1] += grid[CH_DOOR] * 0.9
    composite[:, :, 2] += grid[CH_DOOR] * 0.9
    # Locked door: red
    composite[:, :, 2] += grid[CH_LOCKED_DOOR] * 1.0
    # Item: cyan (B + G)
    composite[:, :, 0] += grid[CH_ITEM] * 0.9
    composite[:, :, 1] += grid[CH_ITEM] * 0.8
    # Enemy: green
    composite[:, :, 1] += grid[CH_ENEMY] * 1.0

    # Clip and convert to uint8
    img_grid = (np.clip(composite, 0.0, 1.0) * 255.0).astype(np.uint8)

    # Upscale to panel size
    upscaled = cv2.resize(img_grid, (width, height), interpolation=cv2.INTER_NEAREST)

    # Player position marker at center (16, 16)
    cx, cy = width // 2, height // 2
    cv2.circle(upscaled, (cx, cy), 5, (0, 255, 255), -1)
    # Forward direction indicator (arrow pointing UP)
    cv2.arrowedLine(upscaled, (cx, cy), (cx, cy - 12), (0, 255, 255), 2, tipLength=0.4)

    cv2.putText(upscaled, "Egocentric Neural Grid (32x32)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return upscaled


def render_replay(
    wad_path: str,
    map_name: str = "MAP01",
    checkpoint_path: str | None = None,
    output_path: str = "replay_showcase.mp4",
    max_steps: int = 500,
    fps: int = 15,
) -> None:
    """Records and writes an MP4 replay combining RGB + Radar + Grid."""
    print(f"[RenderReplay] Loading map: {wad_path} ({map_name})...")
    env = make_env(wad=wad_path, map_name=map_name, episode_ticks=max_steps * 4, render_buffers=True)

    from gymnasium.spaces import Box, MultiBinary
    obs_space = Box(-np.inf, np.inf, (1,), np.float32)
    act_space = MultiBinary(9)
    agent = DoomAgent(obs_space, act_space, device="cpu")

    if checkpoint_path and Path(checkpoint_path).exists():
        import torch
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent.model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=False)
        print(f"[RenderReplay] Loaded weights from {checkpoint_path}")

    agent.reset()
    obs, info = env.reset(seed=42)

    panel_w, panel_h = 320, 240
    total_w, total_h = panel_w * 3, panel_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (total_w, total_h))

    step_idx = 0
    terminated = False
    truncated = False

    print(f"[RenderReplay] Rendering video to {output_path}...")
    while not (terminated or truncated) and step_idx < max_steps:
        action = agent.act(obs)

        # Panel 1: First-Person RGB
        raw_rgb = obs.get("rgb", np.zeros((120, 160, 3), dtype=np.uint8))
        rgb_bgr = cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR)
        panel_rgb = cv2.resize(rgb_bgr, (panel_w, panel_h), interpolation=cv2.INTER_LINEAR)

        # HUD Overlay on RGB panel
        vitals = obs.get("vitals", np.zeros(17))
        health = int(vitals[0]) if len(vitals) > 0 else 100
        active_actions = [ACTION_NAMES[i] for i in range(9) if action[i] == 1]
        act_str = "+".join(active_actions) if active_actions else "IDLE"

        cv2.putText(panel_rgb, f"HP: {health}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(panel_rgb, f"Step: {step_idx} | {act_str}", (10, panel_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

        # Panel 2: 360° Polar Ray Radar
        panel_radar = draw_polar_radar(obs.get("rays", {}), panel_w, panel_h)

        # Panel 3: 32x32 Egocentric Grid
        current_grid = agent.odom_state.grid if agent.odom_state else np.zeros((6, 32, 32))
        panel_grid = draw_grid_map(current_grid, panel_w, panel_h)

        # Concatenate 3 panels side-by-side
        frame = np.hstack([panel_rgb, panel_radar, panel_grid])
        out_video.write(frame)

        obs, reward, terminated, truncated, info = env.step(action)
        step_idx += 1

    env.close()
    out_video.release()
    print(f"[RenderReplay] Completed {step_idx} steps. Video saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 3-panel side-by-side agent replay.")
    parser.add_argument("--wad", type=str, default="maps/two_rooms.wad", help="Path to WAD")
    parser.add_argument("--map", type=str, default="MAP01", help="Map lump name")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--output", type=str, default="replay_showcase.mp4", help="Output MP4 path")
    parser.add_argument("--steps", type=int, default=300, help="Max steps to record")
    args = parser.parse_args()

    render_replay(
        wad_path=args.wad,
        map_name=args.map,
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        max_steps=args.steps,
    )
