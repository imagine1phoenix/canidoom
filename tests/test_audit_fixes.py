"""Targeted audit & regression tests for strict checkpoint loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from submission.agent import DoomAgent, load_strict_checkpoint
from training.model import HybridDoomNet
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
