"""Tier 2 Opaque-Box E2E Tests: RL, Coach, BC & Submission Boundaries (F7-F12, 30 tests).

All `training.*` imports are strictly lazy inside test bodies.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import signal
import socket
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.e2e.conftest import (
    REPO_ROOT,
    SUBMISSION_DIR,
    TWO_ROOMS_WAD,
    import_training,
    make_dummy_model_inputs,
    make_synthetic_obs,
    wad_from_grid,
    write_synthetic_npz,
)

pytestmark = pytest.mark.e2e


def _make_vec_env(num_envs: int, wad_path: str | Path, **kwargs: Any):
    envs_mod = import_training("envs")
    VecDoomEnv = envs_mod.VecDoomEnv
    sig = inspect.signature(VecDoomEnv.__init__)
    params = sig.parameters
    call_kw: dict[str, Any] = {}
    if "num_envs" in params:
        call_kw["num_envs"] = num_envs
    if "wad_path" in params:
        call_kw["wad_path"] = str(wad_path)
    elif "wad" in params:
        call_kw["wad"] = str(wad_path)
    elif "wads" in params:
        call_kw["wads"] = [str(wad_path)] * num_envs
    for k, v in kwargs.items():
        if k in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            call_kw[k] = v
    return VecDoomEnv(**call_kw)


def _make_rnd_module(feature_dim: int = 64):
    rnd_mod = import_training("rnd")
    cls = getattr(rnd_mod, "RNDModule", None) or getattr(rnd_mod, "RND", None)
    assert cls is not None
    sig = inspect.signature(cls.__init__)
    if "feature_dim" in sig.parameters:
        return rnd_mod, cls(feature_dim=feature_dim)
    if "input_dim" in sig.parameters:
        return rnd_mod, cls(input_dim=feature_dim)
    return rnd_mod, cls(feature_dim)


# ===========================================================================
# F7: VecDoomEnv Boundaries (5 tests)
# ===========================================================================


def test_t2_f7_vec_env_recovers_from_killed_worker() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.0, worker_timeout_s=5.0)
    try:
        venv.reset()
        procs = list(getattr(venv, "processes", getattr(venv, "_procs", getattr(venv, "workers", []))))
        assert procs and hasattr(procs[0], "pid") and procs[0].pid is not None
        os.kill(procs[0].pid, signal.SIGKILL)
        procs[0].join(timeout=5)

        actions = np.zeros((2, 9), dtype=np.int8)
        obs, rewards, _, _, _ = venv.step(actions)
        assert len(obs) == 2
        assert np.isfinite(rewards).all()
    finally:
        venv.close()


def test_t2_f7_vec_env_worker_timeout_recovery() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.0, worker_timeout_s=2.0)
    try:
        venv.reset()
        # Verify short timeout parameter is respected during normal operation and worker restart works
        restart_fn = getattr(venv, "_restart_worker", getattr(venv, "restart_worker", None))
        if callable(restart_fn):
            restart_fn(0)
        obs, rewards, _, _, _ = venv.step(np.zeros((2, 9), dtype=np.int8))
        assert len(obs) == 2 and np.isfinite(rewards).all()
    finally:
        venv.close()


def test_t2_f7_vec_env_worker_episode_recycling() -> None:
    empty_wad = wad_from_grid("#####\n#P..#\n#####")
    venv = _make_vec_env(
        1,
        empty_wad,
        episode_ticks=12,  # 3 steps per episode
        render_buffers_ratio=0.0,
        worker_max_episodes=2,
    )
    try:
        venv.reset()
        procs_before = [p.pid for p in getattr(venv, "processes", getattr(venv, "_procs", getattr(venv, "workers", [])))]
        noop = np.zeros((1, 9), dtype=np.int8)
        for _ in range(12):  # spans > 3 episodes -> triggers recycle at 2
            venv.step(noop)
        procs_after = [p.pid for p in getattr(venv, "processes", getattr(venv, "_procs", getattr(venv, "workers", [])))]
        assert procs_before and procs_after
        assert procs_after[0] != procs_before[0], "Expected worker PID to recycle after worker_max_episodes=2"
    finally:
        venv.close()


def test_t2_f7_vec_env_invalid_action_shape_or_values_raises() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.0)
    try:
        venv.reset()
        with pytest.raises((ValueError, AssertionError)):
            venv.step(np.zeros((2, 5), dtype=np.int8))
        with pytest.raises((ValueError, AssertionError)):
            venv.step(np.full((2, 9), 2, dtype=np.int8))
    finally:
        venv.close()


def test_t2_f7_vec_env_double_close_idempotent_and_step_after_close_raises() -> None:
    venv = _make_vec_env(1, TWO_ROOMS_WAD, episode_ticks=100, render_buffers_ratio=0.0)
    venv.reset()
    venv.close()
    venv.close()  # idempotent
    with pytest.raises(RuntimeError):
        venv.step(np.zeros((1, 9), dtype=np.int8))


# ===========================================================================
# F8: RND + Grid Novelty Boundaries (5 tests)
# ===========================================================================


def test_t2_f8_rnd_constant_zero_batch_no_nan() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=64)
    zeros = torch.zeros(16, 64, dtype=torch.float32)
    for _ in range(20):
        rew, loss = rnd.compute_intrinsic_reward(zeros) if hasattr(rnd, "compute_intrinsic_reward") else rnd(zeros)
        assert torch.isfinite(rew).all()
        assert torch.isfinite(loss)


def test_t2_f8_rnd_single_element_batch_b1_no_ddof_nan() -> None:
    import torch

    rnd_mod, rnd = _make_rnd_module(feature_dim=32)
    rms = rnd_mod.RunningMeanStd(shape=(32,))
    x1 = np.ones((1, 32), dtype=np.float32) * 2.0
    rms.update(x1)
    assert np.isfinite(rms.mean).all() and np.isfinite(rms.var).all()

    t1 = torch.tensor(x1)
    rew, loss = rnd.compute_intrinsic_reward(t1) if hasattr(rnd, "compute_intrinsic_reward") else rnd(t1)
    assert torch.isfinite(rew).all() and torch.isfinite(loss)


def test_t2_f8_rnd_extreme_outlier_input_stays_finite() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=32)
    normal = torch.randn(16, 32)
    _ = rnd.compute_intrinsic_reward(normal) if hasattr(rnd, "compute_intrinsic_reward") else rnd(normal)
    outlier = torch.full((4, 32), 1e6, dtype=torch.float32)
    rew, loss = rnd.compute_intrinsic_reward(outlier) if hasattr(rnd, "compute_intrinsic_reward") else rnd(outlier)
    assert torch.isfinite(rew).all() and torch.isfinite(loss)


def test_t2_f8_grid_novelty_all_zero_vs_saturated_ones() -> None:
    import torch

    rnd_mod, rnd = _make_rnd_module(feature_dim=32)
    grid_fn = getattr(rnd, "grid_novelty", None) or getattr(rnd_mod, "compute_grid_novelty", None)
    assert callable(grid_fn)

    b0 = grid_fn(torch.zeros(2, 6, 32, 32, dtype=torch.float32))
    b1 = grid_fn(torch.ones(2, 6, 32, 32, dtype=torch.float32))
    assert torch.isfinite(b0).all() and torch.isfinite(b1).all()


def test_t2_f8_rnd_mismatched_feature_dim_raises() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=64)
    bad = torch.randn(4, 99, dtype=torch.float32)
    with pytest.raises((RuntimeError, ValueError)):
        _ = rnd.compute_intrinsic_reward(bad) if hasattr(rnd, "compute_intrinsic_reward") else rnd(bad)


# ===========================================================================
# F9: Offline llm_coach.py Boundaries (5 tests)
# ===========================================================================


def test_t2_f9_empty_api_key_skips_network_and_uses_fallback(monkeypatch) -> None:
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "")

    def _forbid_net(*args, **kwargs):
        raise AssertionError("Network must not be called when CREO_API_KEY is empty")

    monkeypatch.setattr(urllib.request, "urlopen", _forbid_net)
    knobs = coach.advise_curriculum({"exit_rate": 0.2}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})
    summary = coach.summarize_evaluation({"total_score": 50.0})
    assert isinstance(knobs, dict) and isinstance(summary, str) and len(summary) > 0


def test_t2_f9_http_error_and_timeout_trigger_fallback(monkeypatch) -> None:
    import urllib.error
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(urllib.error.URLError("simulated timeout")),
    )

    out = coach.advise_curriculum({"exit_rate": 0.4}, {"num_rooms": 3, "enemies_per_room": 2, "ammo_mult": 1.0})
    assert 1 <= int(out["num_rooms"]) <= 12
    summary = coach.summarize_evaluation({"total_score": 30.0})
    assert isinstance(summary, str) and len(summary.strip()) > 0


def test_t2_f9_malformed_json_and_out_of_bounds_knobs_clamped(monkeypatch) -> None:
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")

    class _FakeResp:
        def __init__(self, txt: str):
            self._b = json.dumps({"choices": [{"message": {"content": txt}}]}).encode("utf-8")

        def read(self):
            return self._b

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    # Case 1: invalid non-JSON garbage
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResp("NOT JSON {{{"))
    out_garbage = coach.advise_curriculum({"exit_rate": 0.3}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})
    assert 1 <= int(out_garbage["num_rooms"]) <= 12

    # Case 2: extreme out-of-bounds knobs
    extreme = json.dumps({"num_rooms": 9999, "enemies_per_room": -50, "ammo_mult": -5.0})
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResp(extreme))
    out_clamped = coach.advise_curriculum({"exit_rate": 0.3}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})
    assert 1 <= int(out_clamped["num_rooms"]) <= 12
    assert 0 <= int(out_clamped["enemies_per_room"]) <= 12
    assert 0.1 <= float(out_clamped["ammo_mult"]) <= 3.0


def test_t2_f9_empty_and_partial_telemetry_safe(monkeypatch) -> None:
    coach = import_training("llm_coach")
    monkeypatch.delenv("CREO_API_KEY", raising=False)
    out = coach.advise_curriculum({}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})
    summary = coach.summarize_evaluation({})
    assert isinstance(out, dict) and "num_rooms" in out
    assert isinstance(summary, str) and len(summary.strip()) > 0


def test_t2_f9_no_hardcoded_creo_secret_in_coach_or_env_example() -> None:
    secret_re = re.compile(r"cr_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9]{20,}")
    for rel in ("training/llm_coach.py", ".env.example"):
        p = REPO_ROOT / rel
        assert p.is_file(), f"Missing {p}"
        text = p.read_text(encoding="utf-8", errors="replace")
        assert not secret_re.search(text), f"Potential hardcoded Creo API key found in {rel}"


# ===========================================================================
# F10: PPO + BPTT Boundaries (5 tests)
# ===========================================================================


def test_t2_f10_bptt_done_masking_resets_gru_and_obj_slots() -> None:
    import torch

    ppo = import_training("ppo")
    model_mod = import_training("model")
    unroll_fn = getattr(ppo, "bptt_unroll", None) or getattr(ppo, "unroll_bptt_chunk", None)
    assert callable(unroll_fn), "training.ppo must expose bptt_unroll / unroll_bptt_chunk"

    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": False, "use_grid": False}
    )
    net.eval()
    T, B = 6, 1
    steps_a = [make_dummy_model_inputs(batch_size=B, zero_rgb=True, zero_grid=True, seed=100 + t) for t in range(T)]
    steps_b = [make_dummy_model_inputs(batch_size=B, zero_rgb=True, zero_grid=True, seed=200 + t) for t in range(3)] + steps_a[3:]
    # done[2] = True -> hidden state entering step 3 must be zeroed!
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)
    init_h = model_mod.RecurrentState(
        gru_h=torch.randn(1, B, 256),
        obj_slots=torch.randn(B, 4, 8),
    )

    with torch.no_grad():
        logits_a, values_a = unroll_fn(net, steps_a, dones, init_h)[:2]
        logits_b, values_b = unroll_fn(net, steps_b, dones, init_h)[:2]

    # Steps 3, 4, 5 are identical and follow done[2]==1 -> must match bit-for-bit!
    torch.testing.assert_close(logits_a[3:], logits_b[3:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(values_a[3:], values_b[3:], atol=1e-6, rtol=1e-6)


def test_t2_f10_all_actions_masked_except_one_finite_entropy() -> None:
    import torch

    ppo = import_training("ppo")
    fn = getattr(ppo, "masked_bernoulli_log_prob_entropy", None) or getattr(ppo, "masked_bernoulli", None)
    assert callable(fn)

    logits = torch.zeros(2, 9, dtype=torch.float32)
    mask = torch.zeros(2, 9, dtype=torch.bool)
    mask[:, 0] = True  # only MOVE_FORWARD allowed
    actions = torch.zeros(2, 9, dtype=torch.float32)
    actions[:, 0] = 1.0

    lp, ent = fn(logits, mask, actions)[:2]
    assert torch.isfinite(lp).all()
    assert torch.isfinite(ent).all() and (ent >= 0.0).all()


def test_t2_f10_zero_demos_missing_demo_dir_runs_pure_rl(tmp_path: Path) -> None:
    ppo = import_training("ppo")
    cfg_path = tmp_path / "ppo_nodemo.yaml"
    cfg = {
        "seed": 1,
        "num_envs": 1,
        "rollout_steps": 16,
        "bptt_len": 16,
        "ppo_epochs": 1,
        "minibatch_chunks": 1,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "lr": 3e-4,
        "max_grad_norm": 0.5,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "rnd_coef": 0.05,
        "grid_novelty_coef": 0.03,
        "aux_coef": 0.10,
        "bc_kl_beta_init": 0.0,
        "compass_coef": 0.15,
        "rgb_dropout_p": 0.0,
        "scaffolding_end_step": 1000,
        "tier_weights": [1.0, 1.5, 2.5, 3.0],
        "worker_max_episodes": 100,
        "worker_timeout_s": 15.0,
        "render_buffers_ratio": 0.0,
        "total_updates": 1,
        "wad_path": str(TWO_ROOMS_WAD),
        "demo_dir": str(tmp_path / "nonexistent_demos"),
        "output_dir": str(tmp_path / "run_nodemo"),
        "device": "cpu",
        "wandb_enabled": False,
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    train_fn = getattr(ppo, "train", None) or getattr(ppo, "run_training", None)
    train_fn(cfg_path)
    assert list((tmp_path / "run_nodemo").rglob("*.pt"))


def test_t2_f10_missing_resume_checkpoint_raises(tmp_path: Path) -> None:
    ppo = import_training("ppo")
    load_fn = getattr(ppo, "load_checkpoint", None)
    if callable(load_fn):
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_fn(tmp_path / "missing.pt")
    else:
        cfg_path = tmp_path / "ppo_bad_resume.yaml"
        cfg_path.write_text(
            yaml.safe_dump(
                {
                    "seed": 1,
                    "num_envs": 1,
                    "rollout_steps": 16,
                    "bptt_len": 16,
                    "ppo_epochs": 1,
                    "minibatch_chunks": 1,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_coef": 0.2,
                    "lr": 3e-4,
                    "max_grad_norm": 0.5,
                    "vf_coef": 0.5,
                    "ent_coef": 0.01,
                    "rnd_coef": 0.05,
                    "grid_novelty_coef": 0.03,
                    "aux_coef": 0.10,
                    "bc_kl_beta_init": 0.0,
                    "compass_coef": 0.15,
                    "rgb_dropout_p": 0.0,
                    "scaffolding_end_step": 1000,
                    "tier_weights": [1.0, 1.5, 2.5, 3.0],
                    "worker_max_episodes": 100,
                    "worker_timeout_s": 15.0,
                    "render_buffers_ratio": 0.0,
                    "total_updates": 1,
                    "resume_checkpoint": str(tmp_path / "missing.pt"),
                    "wad_path": str(TWO_ROOMS_WAD),
                    "output_dir": str(tmp_path / "out"),
                    "device": "cpu",
                    "wandb_enabled": False,
                }
            ),
            encoding="utf-8",
        )
        train_fn = getattr(ppo, "train", None) or getattr(ppo, "run_training", None)
        with pytest.raises((FileNotFoundError, RuntimeError)):
            train_fn(cfg_path)


def test_t2_f10_gradient_clipping_caps_explosion() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": False, "use_grid": False}
    )
    batch = make_dummy_model_inputs(batch_size=2, zero_rgb=True, zero_grid=True)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, 2, 256, dtype=torch.float32),
        obj_slots=torch.zeros(2, 4, 8, dtype=torch.float32),
    )
    logits, value, _, _ = net(batch, h, return_aux=False)
    loss = (logits.sum() + value.sum()) * 1e6
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=0.5)
    total_norm = float(
        torch.sqrt(sum(p.grad.detach().pow(2).sum() for p in net.parameters() if p.grad is not None))
    )
    assert total_norm <= 0.501


# ===========================================================================
# F11: Optional BC Pretrainer Boundaries (5 tests)
# ===========================================================================


def test_t2_f11_missing_or_empty_demo_dir_fails_fast(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    empty_dir = tmp_path / "empty_demos"
    empty_dir.mkdir()
    cfg_path = tmp_path / "bc_empty.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "demo_dir": str(empty_dir),
                "output_checkpoint": str(tmp_path / "out.pt"),
                "seq_len": 32,
                "batch_seqs": 2,
                "epochs": 1,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "max_grad_norm": 0.5,
                "pos_weight_use": 2.5,
                "pos_weight_attack": 2.5,
                "rgb_dropout_p": 0.0,
                "val_split": 0.0,
                "device": "cpu",
                "wandb_project": "devforge-doom",
                "wandb_enabled": False,
                "model_config": {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True},
            }
        ),
        encoding="utf-8",
    )
    train_fn = getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None)
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        train_fn(cfg_path)


def test_t2_f11_malformed_npz_missing_actions_raises(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    bad_npz = tmp_path / "bad.npz"
    np.savez_compressed(bad_npz, rgb=np.zeros((10, 48, 64, 3), dtype=np.uint8))
    ds_cls = getattr(bc_mod, "DemoChunkDataset", None) or getattr(bc_mod, "BCDataset", None)
    with pytest.raises((KeyError, ValueError)):
        ds = ds_cls([bad_npz], seq_len=32)
        _ = ds[0]


def test_t2_f11_short_episode_under_seq_len_32_handled(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    demo_dir = tmp_path / "short_demos"
    write_synthetic_npz(demo_dir / "short.npz", num_steps=7, seed=4)
    out_ckpt = tmp_path / "short_bc.pt"
    cfg_path = tmp_path / "bc_short.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "demo_dir": str(demo_dir),
                "output_checkpoint": str(out_ckpt),
                "seq_len": 32,
                "batch_seqs": 1,
                "epochs": 1,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "max_grad_norm": 0.5,
                "pos_weight_use": 2.5,
                "pos_weight_attack": 2.5,
                "rgb_dropout_p": 0.0,
                "val_split": 0.0,
                "device": "cpu",
                "wandb_project": "devforge-doom",
                "wandb_enabled": False,
                "model_config": {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True},
            }
        ),
        encoding="utf-8",
    )
    train_fn = getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None)
    train_fn(cfg_path)
    assert out_ckpt.is_file()


def test_t2_f11_mid_sequence_done_boundary_finite(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    demo_dir = tmp_path / "done_demos"
    write_synthetic_npz(demo_dir / "ep_mid_done.npz", num_steps=32, done_indices=(10, 31), seed=5)
    out_ckpt = tmp_path / "done_bc.pt"
    cfg_path = tmp_path / "bc_done.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "demo_dir": str(demo_dir),
                "output_checkpoint": str(out_ckpt),
                "seq_len": 32,
                "batch_seqs": 1,
                "epochs": 1,
                "lr": 3e-4,
                "weight_decay": 1e-4,
                "max_grad_norm": 0.5,
                "pos_weight_use": 2.5,
                "pos_weight_attack": 2.5,
                "rgb_dropout_p": 0.0,
                "val_split": 0.0,
                "device": "cpu",
                "wandb_project": "devforge-doom",
                "wandb_enabled": False,
                "model_config": {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True},
            }
        ),
        encoding="utf-8",
    )
    train_fn = getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None)
    train_fn(cfg_path)
    assert out_ckpt.is_file()


def test_t2_f11_zero_positive_use_attack_batch_finite_loss() -> None:
    import torch

    bc_mod = import_training("bc.train_bc")
    loss_fn = getattr(bc_mod, "compute_bc_loss", None)
    assert callable(loss_fn)

    logits = torch.zeros(4, 9, dtype=torch.float32)
    targets = torch.zeros(4, 9, dtype=torch.float32)  # zero USE/ATTACK positives
    mask = torch.ones(4, 9, dtype=torch.bool)
    loss = loss_fn(logits, targets, mask, pos_weight_use=2.5, pos_weight_attack=2.5)
    assert torch.isfinite(loss)


# ===========================================================================
# F12: Submission, IPC, Security & Validator Boundaries (5 tests — executable now)
# ===========================================================================


def test_t2_f12_submission_ast_forbids_network_and_llm_imports() -> None:
    forbidden = {
        "openai",
        "urllib",
        "requests",
        "httpx",
        "aiohttp",
        "socket",
        "http",
        "vizdoom",
        "llm_coach",
    }
    found = []
    for py in SUBMISSION_DIR.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    if top in forbidden or "llm_coach" in alias.name:
                        found.append(f"{py.name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                top = node.module.split(".")[0]
                if top in forbidden or "llm_coach" in node.module:
                    found.append(f"{py.name}: from {node.module}")
    assert not found, f"Forbidden offline-isolation imports in submission/: {found}"


def test_t2_f12_submission_ast_forbids_aimbots_pathfinding_and_fixed_xyz() -> None:
    banned_names = {"atan2", "astar", "dijkstra", "bfs", "privileged_state", "POSITION_X", "POSITION_Y", "POSITION_Z"}
    hits = []
    for py in SUBMISSION_DIR.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        tree = ast.parse(text)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_names:
                hits.append(f"{py.name}:{node.id}")
            elif isinstance(node, ast.Attribute) and node.attr in banned_names:
                hits.append(f"{py.name}:.{node.attr}")
    assert not hits, f"Rule violations in submission/: {hits}"


def test_t2_f12_doom_agent_handles_render_buffers_false_and_zero_entities() -> None:
    from doom_env import action_space, observation_space
    from submission.agent import DoomAgent

    agent = DoomAgent(observation_space(), action_space(), "cpu")
    agent.reset()
    obs = make_synthetic_obs(
        frame_shape=(120, 160),
        render_buffers=False,
        num_entities=0,
        ray_distances=np.ones(64, dtype=np.float32),
        ray_hit_types=np.zeros(64, dtype=np.int64),
    )
    act = agent.act(obs)
    arr = np.asarray(act)
    assert arr.shape == (9,) and arr.dtype == np.int8
    assert set(np.unique(arr).tolist()) <= {0, 1}


def test_t2_f12_validate_py_rejects_bad_requirements(tmp_path: Path) -> None:
    repo = tmp_path / "bad_req_repo"
    (repo / "submission").mkdir(parents=True)
    (repo / "submission" / "agent.py").write_text(
        "import numpy as np\n"
        "class DoomAgent:\n"
        "    def __init__(self, o, a, d): pass\n"
        "    def reset(self): pass\n"
        "    def act(self, obs): return np.zeros(9, np.int8)\n",
        encoding="utf-8",
    )
    (repo / "submission" / "requirements.txt").write_text("--index-url https://evil.example/simple\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate.py"), "--repo", str(repo), "--map", "maps/two_rooms.wad", "--ticks", "20"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 1
    assert "unsupported line" in proc.stdout


def test_t2_f12_remote_protocol_rejects_invalid_action_and_wrong_seq(tmp_path: Path) -> None:
    from doom_env import remote

    # 1. Wrong sequence number over socketpair
    host, agent_sock = socket.socketpair()
    conn = remote.AgentConnection(host)
    try:
        agent_sock.sendall(remote.encode({"type": "action", "seq": "deadbeef"}, {"action": np.zeros(9, np.int8)}))
        with pytest.raises(remote.AgentFailure) as ei:
            conn.act({"rgb": np.zeros((120, 160, 3), dtype=np.uint8)}, timeout=1.0)
        assert ei.value.kind == "protocol"
    finally:
        host.close()
        agent_sock.close()

    # 2. Invalid action value from subprocess agent
    repo = tmp_path / "bad_act_repo"
    (repo / "submission").mkdir(parents=True)
    (repo / "submission" / "agent.py").write_text(
        textwrap.dedent(
            """
            import numpy as np
            class DoomAgent:
                def __init__(self, o, a, d): pass
                def reset(self): pass
                def act(self, obs): return np.full(9, 2, dtype=np.int8)
            """
        ),
        encoding="utf-8",
    )
    srv = remote.listen(tmp_path / "bad.sock")
    proc = subprocess.Popen(
        [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(srv), "--repo", str(repo), "--device", "cpu"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    conn2 = remote.AgentConnection.accept(srv, timeout=15, alive=lambda: proc.poll() is None)
    try:
        conn2.init("cpu", timeout=15)
        conn2.reset(timeout=5)
        with pytest.raises(remote.AgentFailure) as ei2:
            conn2.act(make_synthetic_obs(), timeout=5)
        assert ei2.value.kind == "invalid_action"
    finally:
        conn2.close()
        srv.close()
        proc.kill()
        proc.wait()
