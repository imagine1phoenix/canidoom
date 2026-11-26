"""Generate a 3-panel Held-Out Benchmark & Diagnostic PNG chart using OpenCV + NumPy.

Panels:
  1. Training Progression (Init -> 77.8k -> 557k -> 806.9k -> Staged 377.9k)
  2. Sampling Temperature (tau) Ablation on 27 held-out maps
  3. Per-Tier Breakdown (Tiers 1-4: Avg Score, Exploration %, Exits)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _draw_panel(canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, title: str) -> None:
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (36, 28, 24), -1)
    cv2.rectangle(canvas, (x0, y0), (x1, y1), (85, 65, 55), 2)
    cv2.putText(canvas, title, (x0 + 16, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (245, 240, 235), 2, cv2.LINE_AA)


def generate_chart(diag_json: Path, out_png: Path) -> None:
    data = json.loads(diag_json.read_text(encoding="utf-8"))
    per_tier = data.get("diagnostics", {}).get("per_tier", {})

    w, h = 1560, 540
    canvas = np.full((h, w, 3), (25, 18, 15), dtype=np.uint8)

    cv2.putText(
        canvas,
        "DevForge DOOM — Held-Out Benchmark, Temperature Ablation & Per-Tier Diagnostics (27 Maps)",
        (26, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.74,
        (255, 248, 245),
        2,
        cv2.LINE_AA,
    )

    base_y = 445
    plot_h = 305

    # Panel 1: Score Progression
    p1 = (24, 58, 512, 514)
    _draw_panel(canvas, *p1, "1. Held-Out Score Progression")
    prog = [
        ("Init", 0.0, (135, 118, 110)),
        ("77.8k", 344.28, (230, 150, 90)),
        ("557k", 386.14, (230, 150, 90)),
        ("806.9k", 504.13, (245, 180, 120)),
        ("Staged", float(data.get("total_score", 1440.45)), (113, 204, 46)),
    ]
    max_s = max(v for _, v, _ in prog) * 1.15
    bar_w, gap, start_x = 58, 28, p1[0] + 38
    for idx, (lbl, val, col) in enumerate(prog):
        bx0 = start_x + idx * (bar_w + gap)
        bx1 = bx0 + bar_w
        bh = int((val / max_s) * plot_h) if max_s > 0 else 0
        by0 = base_y - bh
        cv2.rectangle(canvas, (bx0, by0), (bx1, base_y), col, -1)
        cv2.putText(canvas, f"{val:.0f}", (bx0 + 4, by0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 244, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (bx0 + 4, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 200, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "377.9k", (start_x + 4 * (bar_w + gap) + 4, base_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 230, 160), 1, cv2.LINE_AA)

    # Panel 2: Sampling Temperature Ablation
    p2 = (536, 58, 1024, 514)
    _draw_panel(canvas, *p2, "2. Sampling Temperature (tau) Ablation")
    taus = [
        ("0.50", "7.4% ex", 1217.46, (230, 160, 100)),
        ("0.65", "3.7% ex", 1050.80, (230, 160, 100)),
        ("0.85", "14.8% ex", float(data.get("total_score", 1440.45)), (113, 204, 46)),
        ("1.00", "11.1% ex", 1085.30, (230, 160, 100)),
    ]
    max_t = 1650.0
    bar_w2, gap2, start_x2 = 76, 30, p2[0] + 44
    for idx, (lbl, sub, val, col) in enumerate(taus):
        bx0 = start_x2 + idx * (bar_w2 + gap2)
        bx1 = bx0 + bar_w2
        bh = int((val / max_t) * plot_h)
        by0 = base_y - bh
        cv2.rectangle(canvas, (bx0, by0), (bx1, base_y), col, -1)
        cv2.putText(canvas, f"{val:.1f}", (bx0 + 6, by0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (250, 244, 240), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"t={lbl}", (bx0 + 10, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 200, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, sub, (bx0 + 6, base_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (195, 195, 195), 1, cv2.LINE_AA)

    # Panel 3: Per-Tier Diagnostics
    p3 = (1048, 58, 1536, 514)
    _draw_panel(canvas, *p3, "3. Per-Tier Breakdown (Score & Exploration)")
    cv2.rectangle(canvas, (p3[0] + 20, 100), (p3[0] + 34, 114), (245, 165, 75), -1)
    cv2.putText(canvas, "Avg Score", (p3[0] + 42, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 210, 200), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (p3[0] + 135, 100), (p3[0] + 149, 114), (65, 180, 245), -1)
    cv2.putText(canvas, "Exploration %", (p3[0] + 157, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (225, 210, 200), 1, cv2.LINE_AA)

    bar_w3, start_x3, slot_w = 40, p3[0] + 42, 108
    for idx, t in enumerate(("1", "2", "3", "4")):
        td = per_tier.get(t, {})
        avg_s = float(td.get("avg_score", 0.0))
        expl = float(td.get("avg_exploration", 0.0)) * 100.0
        exits = int(td.get("exits", 0))
        cnt = int(td.get("count", 0))
        kills = float(td.get("avg_kills", 0.0))

        cx = start_x3 + idx * slot_w
        h_s = int((min(avg_s, 80.0) / 80.0) * plot_h)
        cv2.rectangle(canvas, (cx, base_y - h_s), (cx + bar_w3, base_y), (245, 165, 75), -1)
        cv2.putText(canvas, f"{avg_s:.1f}", (cx + 2, base_y - h_s - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (255, 210, 160), 1, cv2.LINE_AA)

        h_e = int((min(expl, 100.0) / 100.0) * plot_h)
        ex_x0 = cx + bar_w3 + 6
        cv2.rectangle(canvas, (ex_x0, base_y - h_e), (ex_x0 + bar_w3, base_y), (65, 180, 245), -1)
        cv2.putText(canvas, f"{expl:.0f}%", (ex_x0 + 2, base_y - h_e - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (130, 215, 255), 1, cv2.LINE_AA)

        cv2.putText(canvas, f"T{t} ({exits}/{cnt}ex)", (cx, base_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (215, 200, 190), 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{kills:.1f} kills", (cx + 2, base_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (185, 185, 185), 1, cv2.LINE_AA)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), canvas)
    print(f"[Plot] Saved diagnostics chart to {out_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render held-out diagnostics PNG chart.")
    parser.add_argument("--input", type=str, default="eval_results_diagnostics.json")
    parser.add_argument("--out", type=str, default="docs/heldout_diagnostics.png")
    args = parser.parse_args()
    generate_chart(Path(args.input), Path(args.out))
