"""Unit tests for DoomAgent NaN-safety/tau config and scorer diagnostics/comparator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from submission.agent import DoomAgent
from training.scorer.compare_checkpoints import compare_checkpoints


def _make_nan_obs() -> dict:
    return {
        "rgb": np.zeros((120, 160, 3), dtype=np.uint8),
        "depth": np.zeros((120, 160), dtype=np.uint8),
        "vitals": np.full((17,), np.nan, dtype=np.float32),
        "rays": {
            "distance": np.full((64,), np.nan, dtype=np.float32),
            "hit_type": np.zeros((64,), dtype=np.int64),
        },
        "visible_entities": {
            "features": np.full((32, 5), np.nan, dtype=np.float32),
            "entity_type": np.zeros((32,), dtype=np.int64),
            "mask": np.zeros((32,), dtype=np.int8),
        },
    }


def test_agent_nan_obs_returns_valid_action() -> None:
    agent = DoomAgent(None, None, device="cpu", tau=0.85)
    agent.reset()
    act = agent.act(_make_nan_obs())
    assert isinstance(act, np.ndarray)
    assert act.shape == (9,)
    assert set(np.unique(act)).issubset({0, 1})


def test_compare_checkpoints_sorts_and_promotes(tmp_path: Path) -> None:
    c1 = tmp_path / "c1.pt"
    c2 = tmp_path / "c2.pt"
    dest = tmp_path / "best_agent.pt"
    out_json = tmp_path / "board.json"
    c1.write_bytes(b"ckpt1")
    c2.write_bytes(b"ckpt2")

    fake_summaries = {
        str(c1): {
            "checkpoint": str(c1),
            "total_score": 500.0,
            "tier_scores": {"1": 20.0},
            "exit_rate": 0.05,
            "mean_act_ms": 5.0,
            "p99_act_ms": 12.0,
            "passed_limits": True,
            "diagnostics": {"total_kills": 10, "total_doors_opened": 5, "deaths": 20},
        },
        str(c2): {
            "checkpoint": str(c2),
            "total_score": 1440.45,
            "tier_scores": {"1": 39.7},
            "exit_rate": 0.148,
            "mean_act_ms": 5.66,
            "p99_act_ms": 18.9,
            "passed_limits": True,
            "diagnostics": {"total_kills": 56, "total_doors_opened": 35, "deaths": 20},
        },
    }

    with patch(
        "training.scorer.compare_checkpoints.score_checkpoint",
        side_effect=lambda checkpoint_path, **kwargs: fake_summaries[str(checkpoint_path)],
    ):
        res = compare_checkpoints(
            candidates=[str(c1), str(c2)],
            promote_to=str(dest),
            out_json=str(out_json),
        )

    assert res["winner"] == str(c2)
    assert dest.read_bytes() == b"ckpt2"
    saved = json.loads(out_json.read_text(encoding="utf-8"))
    assert saved["leaderboard"][0]["total_score"] == 1440.45


def test_plot_diagnostics_creates_png(tmp_path: Path) -> None:
    from training.scorer.plot_diagnostics import generate_chart

    diag_json = tmp_path / "diag.json"
    out_png = tmp_path / "chart.png"
    diag_json.write_text(
        json.dumps(
            {
                "total_score": 1440.45,
                "diagnostics": {
                    "per_tier": {
                        "1": {"count": 6, "exits": 3, "avg_score": 39.7, "avg_kills": 2.0, "avg_exploration": 0.667},
                        "2": {"count": 7, "exits": 1, "avg_score": 42.36, "avg_kills": 1.86, "avg_exploration": 0.533},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    generate_chart(diag_json, out_png)
    assert out_png.is_file()
    assert out_png.stat().st_size > 10_000


def test_render_replay_policy_bar_shape() -> None:
    from training.scorer.render_replay import draw_policy_bar

    probs = np.linspace(0.1, 0.9, 9, dtype=np.float32)
    action = np.array([1, 0, 0, 0, 1, 0, 1, 0, 0], dtype=np.int8)
    strip = draw_policy_bar(probs, action, step_idx=12, total_w=960, bar_h=64)
    assert strip.shape == (64, 960, 3)
    assert strip.dtype == np.uint8
