"""Targeted audit & regression tests for strict checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from submission.agent import DoomAgent, load_strict_checkpoint
from training.model import HybridDoomNet, RecurrentState
from training.ppo import bptt_unroll, masked_bernoulli_log_prob_entropy
from training.rewards import RewardShaper, RewardWeights
from training.scorer.score_checkpoint import score_checkpoint


def test_task1_strict_loading_fails_on_missing_lfs_corrupt_and_mismatched(tmp_path: Path) -> None:
    model = HybridDoomNet(use_rgb=True, use_grid=True, rgb_dropout_p=0.0, gru_dim=256, num_slots=4, slot_dim=8)

    # 1. Missing file
    missing = tmp_path / "does_not_exist.pt"
    with pytest.raises(FileNotFoundError):
        load_strict_checkpoint(missing, model)
    with pytest.raises(FileNotFoundError):
        DoomAgent(checkpoint_path=missing)
    with pytest.raises(FileNotFoundError):
        score_checkpoint(checkpoint_path=missing)

    # 2. Git LFS pointer file
    lfs_ptr = tmp_path / "lfs_pointer.pt"
    lfs_ptr.write_text(
        "version https://git-lfs.github.com/spec/v1\noid sha256:deadbeef\nsize 123456\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Git LFS pointer"):
        DoomAgent(checkpoint_path=lfs_ptr)

    # 3. Corrupt binary file
    corrupt = tmp_path / "corrupt.pt"
    corrupt.write_bytes(b"\x00\xffnot a valid pickle or zip archive")
    with pytest.raises(RuntimeError, match="Corrupt or unreadable"):
        DoomAgent(checkpoint_path=corrupt)

    # 4. Missing / unexpected / shape-mismatched keys
    bad_sd = dict(model.state_dict())
    first_key = next(iter(bad_sd))
    bad_sd.pop(first_key)
    bad_sd["unexpected_extra_weight"] = torch.zeros(3)
    bad_ckpt = tmp_path / "mismatched.pt"
    torch.save({"model_state_dict": bad_sd}, bad_ckpt)
    with pytest.raises(RuntimeError):
        DoomAgent(checkpoint_path=bad_ckpt)

    # 5. Explicit allow_untrained=True opt-in works, while default refuses None
    with pytest.raises(FileNotFoundError):
        DoomAgent(checkpoint_path=None)
    untrained = DoomAgent(checkpoint_path=None, allow_untrained=True)
    assert untrained.model is not None


def test_task2_no_training_config_writes_to_submission_weights() -> None:
    cfg_dir = Path("training/configs")
    for yml in sorted(cfg_dir.glob("*.yaml")):
        cfg = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
        best_out = str(cfg.get("best_out") or "")
        assert "submission/weights" not in best_out.replace("\\", "/"), (
            f"{yml} must not write directly to submission/weights/ (got best_out={best_out!r})"
        )


def test_task6_rollout_vs_ppo_update_log_probs_match_with_rgb_dropout_p_0() -> None:
    torch.manual_seed(42)
    model = HybridDoomNet(use_rgb=True, use_grid=True, rgb_dropout_p=0.0, gru_dim=64, num_slots=2, slot_dim=4)
    model.eval()

    T, B = 6, 2
    obs_seq = {
        "rays": torch.randn(B, T, 8, 64),
        "ent_feats": torch.randn(B, T, 32, 5),
        "ent_types": torch.zeros(B, T, 32, dtype=torch.long),
        "ent_mask": torch.ones(B, T, 32, dtype=torch.bool),
        "vitals": torch.randn(B, T, 36),
        "rgb_depth": torch.randn(B, T, 4, 48, 64),
        "grid": torch.randn(B, T, 6, 32, 32),
        "action_mask": torch.ones(B, T, 9, dtype=torch.bool),
        "d_theta": torch.zeros(B, T),
    }
    actions = torch.randint(0, 2, (B, T, 9)).float()
    dones = torch.zeros(B, T)
    h0 = RecurrentState.zeros(B, 64, 2, 4)

    h = h0
    rollout_lps = []
    with torch.no_grad():
        for t in range(T):
            step_in = {k: v[:, t] for k, v in obs_seq.items()}
            logits_t, _, _, h = model(step_in, h, return_aux=False)
            lp_t, _ = masked_bernoulli_log_prob_entropy(logits_t, step_in["action_mask"], actions[:, t])
            rollout_lps.append(lp_t)
    rollout_lp = torch.stack(rollout_lps, dim=1)

    model.train()
    upd_logits, _, _, _ = bptt_unroll(model, obs_seq, dones, init_h=h0, return_aux=False, rgb_dropout_p=0.0)
    upd_lp, _ = masked_bernoulli_log_prob_entropy(upd_logits, obs_seq["action_mask"], actions)
    assert torch.allclose(rollout_lp, upd_lp, atol=1e-5)


def test_task7_spawn_relative_exploration_cells_are_translation_invariant() -> None:
    w = RewardWeights(new_cell=0.06, loop_penalty=-0.03, forward_progress=0.004)
    sa, sb = RewardShaper(w), RewardShaper(w)
    sa.reset([])
    sb.reset([])

    shift_x, shift_y = 37.0, 19.0
    pts = [(10.0, 10.0), (50.0, 10.0), (95.0, 10.0), (140.0, 10.0)]
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        pa0 = {"x": x0, "y": y0, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
        pa1 = {"x": x1, "y": y1, "kills": 0, "items": 0, "damage_taken": 0.0, "dead": False, "finished": False}
        pb0 = {**pa0, "x": x0 + shift_x, "y": y0 + shift_y}
        pb1 = {**pa1, "x": x1 + shift_x, "y": y1 + shift_y}
        ra = sa.step(pa0, pa1, fwd)
        rb = sb.step(pb0, pb1, fwd)
        assert ra == pytest.approx(rb, abs=1e-6)
