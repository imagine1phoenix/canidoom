"""Tier 1 Opaque-Box E2E Tests: RL Core, Coach, BC & Submission (F7-F12, 30 tests).

All `training.*` imports are strictly lazy inside test bodies.
"""

from __future__ import annotations

import inspect
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.e2e.conftest import (
    REPO_ROOT,
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


# ===========================================================================
# F7: Subprocess Vectorized VecDoomEnv (5 tests)
# ===========================================================================


def test_t1_f7_vec_env_init_reset_and_step() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200)
    try:
        obs, infos = venv.reset()
        assert len(obs) == 2
        actions = np.zeros((2, 9), dtype=np.int8)
        actions[:, 0] = 1
        obs2, rewards, terminateds, truncateds, infos2 = venv.step(actions)
        assert len(obs2) == 2
        assert np.asarray(rewards).shape == (2,)
        assert np.asarray(terminateds).shape == (2,)
        assert np.asarray(truncateds).shape == (2,)
        assert np.isfinite(rewards).all()
        assert len(infos2) == 2
    finally:
        venv.close()


def test_t1_f7_vec_env_mixed_render_buffers_ratio() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.5)
    try:
        obs, _ = venv.reset()
        rgb_sums = [int(np.asarray(o["rgb"]).sum()) for o in obs]
        assert any(s > 0 for s in rgb_sums) and any(s == 0 for s in rgb_sums)
    finally:
        venv.close()


def test_t1_f7_vec_env_auto_reset_exposes_terminal_privileged_state() -> None:
    corridor_wad = wad_from_grid("#######\n#P...E#\n#######")
    venv = _make_vec_env(1, corridor_wad, episode_ticks=400, render_buffers_ratio=0.0)
    try:
        venv.reset()
        fwd = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
        terminal_info = None
        for _ in range(120):
            _, _, term, trunc, infos = venv.step(fwd)
            if bool(term[0]) or bool(trunc[0]):
                terminal_info = infos[0]
                break
        assert terminal_info is not None, "Expected agent to reach exit in straight corridor"
        assert terminal_info.get("exit_reached") is True or (
            isinstance(terminal_info.get("terminal_privileged_state"), dict)
            and terminal_info["terminal_privileged_state"].get("exit_reached") is True
        ) or (
            isinstance(terminal_info.get("privileged_state"), dict)
            and terminal_info["privileged_state"].get("exit_reached") is True
        )
    finally:
        venv.close()


def test_t1_f7_vec_env_reset_with_wad_swaps_maps() -> None:
    from doom_env.sensors import RAY_LOCKED

    locked_wad = wad_from_grid("#######\n#P..B.E#\n#######")
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.0)
    try:
        venv.reset()
        swapped = False
        if hasattr(venv, "reset_with_wad"):
            sig = inspect.signature(venv.reset_with_wad)
            if "env_idx" in sig.parameters:
                obs0, _ = venv.reset_with_wad(0, str(locked_wad), "MAP01")
                assert int(obs0["rays"]["hit_type"][0]) == RAY_LOCKED
                swapped = True
            else:
                obs_all, _ = venv.reset_with_wad([str(locked_wad), str(locked_wad)])
                assert int(obs_all[0]["rays"]["hit_type"][0]) == RAY_LOCKED
                swapped = True
        assert swapped, "VecDoomEnv must expose reset_with_wad"
    finally:
        venv.close()


def test_t1_f7_vec_env_close_reaps_workers() -> None:
    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=200, render_buffers_ratio=0.0)
    venv.reset()
    procs = list(getattr(venv, "processes", getattr(venv, "_procs", getattr(venv, "workers", []))))
    venv.close()
    for p in procs:
        if hasattr(p, "is_alive"):
            assert not p.is_alive()


# ===========================================================================
# F8: RND + Egocentric Grid Novelty (5 tests)
# ===========================================================================


def _make_rnd_module(feature_dim: int = 100):
    rnd_mod = import_training("rnd")
    cls = getattr(rnd_mod, "RNDModule", None) or getattr(rnd_mod, "RND", None)
    assert cls is not None, "training.rnd must expose RNDModule or RND"
    sig = inspect.signature(cls.__init__)
    params = list(sig.parameters.keys())[1:]
    if "feature_dim" in sig.parameters:
        return rnd_mod, cls(feature_dim=feature_dim)
    if "input_dim" in sig.parameters:
        return rnd_mod, cls(input_dim=feature_dim)
    if params:
        return rnd_mod, cls(feature_dim)
    return rnd_mod, cls()


def test_t1_f8_rnd_intrinsic_reward_and_differentiable_loss() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=100)
    feats = torch.randn(8, 100, dtype=torch.float32)

    if hasattr(rnd, "compute_intrinsic_reward"):
        rew, loss = rnd.compute_intrinsic_reward(feats)
    else:
        rew, loss = rnd(feats)

    assert rew.shape == (8,)
    assert torch.isfinite(rew).all() and (rew >= 0.0).all()
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()


def test_t1_f8_rnd_predictor_loss_decreases_on_repeated_inputs() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=100)
    opt = torch.optim.Adam([p for p in rnd.parameters() if p.requires_grad], lr=1e-2)
    fixed_x = torch.ones(16, 100, dtype=torch.float32) * 0.5

    losses = []
    for _ in range(15):
        opt.zero_grad()
        _, loss = rnd.compute_intrinsic_reward(fixed_x) if hasattr(rnd, "compute_intrinsic_reward") else rnd(fixed_x)
        loss.backward()
        opt.step()
        losses.append(float(loss.item()))

    assert losses[-1] < losses[0]


def test_t1_f8_rnd_target_parameters_stay_frozen() -> None:
    import torch

    _, rnd = _make_rnd_module(feature_dim=64)
    target = getattr(rnd, "target", getattr(rnd, "target_net", None))
    assert target is not None, "RND module must expose frozen target network"
    before = [p.detach().clone() for p in target.parameters()]
    assert all(not p.requires_grad for p in target.parameters())

    opt = torch.optim.Adam([p for p in rnd.parameters() if p.requires_grad], lr=1e-2)
    x = torch.randn(8, 64)
    _, loss = rnd.compute_intrinsic_reward(x) if hasattr(rnd, "compute_intrinsic_reward") else rnd(x)
    opt.zero_grad()
    loss.backward()
    opt.step()

    for p_old, p_new in zip(before, target.parameters()):
        assert torch.equal(p_old, p_new)


def test_t1_f8_grid_novelty_rewards_unvisited_cells_higher() -> None:
    import torch

    rnd_mod, rnd = _make_rnd_module(feature_dim=64)
    grid_fn = getattr(rnd, "grid_novelty", None) or getattr(rnd_mod, "compute_grid_novelty", None)
    assert callable(grid_fn), "training.rnd must expose grid_novelty / compute_grid_novelty"

    revisit_grid = torch.zeros(1, 6, 32, 32, dtype=torch.float32)
    revisit_grid[0, 0, 16, 16] = 1.0

    novel_grid = torch.zeros(1, 6, 32, 32, dtype=torch.float32)
    novel_grid[0, 0, 16, 16] = 1.0
    novel_grid[0, 0, 8:14, 14:20] = 1.0  # newly explored cells in front

    b_revisit = grid_fn(revisit_grid)
    b_novel = grid_fn(novel_grid)
    assert float(b_novel.mean()) > float(b_revisit.mean())


def test_t1_f8_welford_running_mean_std_accuracy() -> None:
    rnd_mod = import_training("rnd")
    rms_cls = getattr(rnd_mod, "RunningMeanStd", None)
    assert rms_cls is not None, "training.rnd must expose RunningMeanStd"

    rms = rms_cls(shape=(4,))
    rng = np.random.default_rng(42)
    b1 = rng.normal(loc=3.0, scale=2.0, size=(100, 4)).astype(np.float32)
    b2 = rng.normal(loc=3.0, scale=2.0, size=(100, 4)).astype(np.float32)
    all_b = np.concatenate([b1, b2], axis=0)

    rms.update(b1)
    rms.update(b2)
    np.testing.assert_allclose(np.asarray(rms.mean), all_b.mean(axis=0), atol=1e-4)
    np.testing.assert_allclose(np.asarray(rms.var), all_b.var(axis=0), atol=1e-3)


# ===========================================================================
# F9: Offline / Training llm_coach.py (5 tests, strictly mocked HTTP)
# ===========================================================================


class _MockHTTPResponse:
    def __init__(self, payload: dict[str, Any]):
        self._bytes = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._bytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_t1_f9_advise_curriculum_parses_mocked_response(monkeypatch) -> None:
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")
    monkeypatch.setenv("CREO_BASE_URL", "https://example.invalid/v1")
    monkeypatch.setenv("CREO_MODEL", "@cf/openai/gpt-oss-120b")

    llm_reply = {"num_rooms": 5, "enemies_per_room": 3, "ammo_mult": 0.9}
    chat_payload = {"choices": [{"message": {"content": json.dumps(llm_reply)}}]}
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _MockHTTPResponse(chat_payload))

    telemetry = {"exit_rate": 0.35, "sector_coverage": 0.62, "kills": 2.1, "stuck_rate": 0.15}
    current = {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0}
    out = coach.advise_curriculum(telemetry, current)
    assert isinstance(out, dict)
    assert out["num_rooms"] == 5
    assert out["enemies_per_room"] == 3
    assert float(out["ammo_mult"]) == pytest.approx(0.9)


def test_t1_f9_summarize_evaluation_parses_mocked_completion(monkeypatch) -> None:
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")
    chat_payload = {
        "choices": [{"message": {"content": "Tier 3 keycard backtracking improved; watch stuck rate."}}]
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda req, timeout=None: _MockHTTPResponse(chat_payload))

    summary = coach.summarize_evaluation({"total_score": 68.4, "exit_rate": 0.55})
    assert isinstance(summary, str) and len(summary.strip()) > 10
    assert "Tier 3" in summary


def test_t1_f9_request_uses_env_url_model_and_bearer_header(monkeypatch) -> None:
    import urllib.request

    coach = import_training("llm_coach")
    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-xyz")
    monkeypatch.setenv("CREO_BASE_URL", "https://mock.creo.invalid/v1")
    monkeypatch.setenv("CREO_MODEL", "@cf/openai/gpt-oss-120b")

    captured: dict[str, Any] = {}

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["auth"] = req.get_header("Authorization")
        captured["body"] = json.loads(req.data.decode("utf-8"))
        return _MockHTTPResponse({"choices": [{"message": {"content": json.dumps({"num_rooms": 4})}}]})

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    coach.advise_curriculum({"exit_rate": 0.5}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})

    assert captured["url"].startswith("https://mock.creo.invalid/v1")
    assert captured["auth"] == "Bearer test-dummy-token-xyz"
    assert captured["body"]["model"] == "@cf/openai/gpt-oss-120b"


def test_t1_f9_cli_help_and_offline_invocation_exit_zero() -> None:
    env = {k: v for k, v in __import__("os").environ.items() if k != "CREO_API_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "training.llm_coach", "--help"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr


def test_t1_f9_deterministic_offline_fallback_without_api_key(monkeypatch) -> None:
    coach = import_training("llm_coach")
    monkeypatch.delenv("CREO_API_KEY", raising=False)

    current = {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0}
    telemetry = {"exit_rate": 0.1, "sector_coverage": 0.25, "stuck_rate": 0.4}
    k1 = coach.advise_curriculum(telemetry, current)
    k2 = coach.advise_curriculum(telemetry, current)
    s1 = coach.summarize_evaluation({"total_score": 42.0, "exit_rate": 0.2, "mean_act_ms": 11.0})

    assert k1 == k2 and isinstance(k1, dict)
    assert 1 <= int(k1["num_rooms"]) <= 12
    assert 0.1 <= float(k1["ammo_mult"]) <= 3.0
    assert isinstance(s1, str) and len(s1.strip()) > 0


# ===========================================================================
# F10: 100% Self-Evolving Recurrent PPO + BPTT + Value Compass (5 tests)
# ===========================================================================


def test_t1_f10_self_evolving_single_room_ppo_zero_demos(tmp_path: Path) -> None:
    ppo = import_training("ppo")
    cfg_path = tmp_path / "ppo_smoke.yaml"
    cfg = {
        "seed": 42,
        "num_envs": 1,
        "rollout_steps": 32,
        "bptt_len": 16,
        "ppo_epochs": 2,
        "minibatch_chunks": 2,
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
        "rgb_dropout_p": 0.25,
        "scaffolding_end_step": 1000,
        "tier_weights": [1.0, 1.5, 2.5, 3.0],
        "worker_max_episodes": 100,
        "worker_timeout_s": 15.0,
        "render_buffers_ratio": 0.0,
        "total_updates": 2,
        "wad_path": wad_from_grid("#######\n#P...E#\n#######"),
        "output_dir": str(tmp_path / "run"),
        "device": "cpu",
        "wandb_enabled": False,
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    train_fn = getattr(ppo, "train", None) or getattr(ppo, "run_training", None)
    assert callable(train_fn), "training.ppo must expose train()"
    result = train_fn(cfg_path)
    ckpts = list((tmp_path / "run").rglob("*.pt"))
    assert ckpts, f"Expected PPO to save at least one .pt checkpoint, result={result}"


def test_t1_f10_gae_calculation_matches_analytical_values() -> None:
    import torch

    ppo = import_training("ppo")
    gae_fn = getattr(ppo, "compute_gae", None)
    assert callable(gae_fn), "training.ppo must expose compute_gae()"

    rewards = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float32)  # (T=3, B=1)
    values = torch.tensor([[0.5], [1.0], [1.5]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)
    next_value = torch.tensor([2.0], dtype=torch.float32)
    gamma, lam = 0.99, 0.95

    # Hand calculation from t=2 down to t=0:
    # t=2 (not done): delta2 = 3.0 + 0.99*2.0 - 1.5 = 3.48; adv2 = 3.48
    # t=1 (done=1 -> mask=0): delta1 = 2.0 + 0 - 1.0 = 1.0; adv1 = 1.0
    # t=0 (not done): delta0 = 1.0 + 0.99*1.0 - 0.5 = 1.49; adv0 = 1.49 + 0.99*0.95*1.0 = 2.4305
    expected_adv = torch.tensor([[2.4305], [1.0], [3.48]], dtype=torch.float32)

    adv, ret = gae_fn(rewards, values, dones, next_value, gamma=gamma, gae_lambda=lam)
    torch.testing.assert_close(adv.reshape(3, 1), expected_adv, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(ret.reshape(3, 1), expected_adv + values, atol=1e-4, rtol=1e-4)


def test_t1_f10_masked_bernoulli_excludes_masked_logits() -> None:
    import torch

    ppo = import_training("ppo")
    fn = getattr(ppo, "masked_bernoulli_log_prob_entropy", None) or getattr(ppo, "masked_bernoulli", None)
    assert callable(fn), "training.ppo must expose masked_bernoulli_log_prob_entropy / masked_bernoulli"

    logits = torch.tensor([[1.2, -0.4, 0.0, 0.3, -1.0, 0.5, 50.0, 50.0, 0.1]], dtype=torch.float32)
    mask = torch.tensor([[True, True, True, True, True, True, False, False, True]], dtype=torch.bool)
    actions = torch.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    out = fn(logits, mask, actions)
    log_prob, entropy = out[0], out[1]
    assert torch.isfinite(log_prob).all() and torch.isfinite(entropy).all()

    # Perturbing masked logits (indices 6, 7) must not change log_prob or entropy
    logits2 = logits.clone()
    logits2[0, 6] = -50.0
    logits2[0, 7] = -50.0
    out2 = fn(logits2, mask, actions)
    torch.testing.assert_close(log_prob, out2[0], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(entropy, out2[1], atol=1e-6, rtol=1e-6)


def test_t1_f10_value_compass_and_next_rays_aux_losses_finite() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    )
    B = 4
    batch = make_dummy_model_inputs(batch_size=B)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )
    _, _, aux, _ = net(batch, h, return_aux=True)

    target_rays = torch.rand(B, 64)
    target_compass = torch.randn(B, 8)
    aux_loss = torch.nn.functional.smooth_l1_loss(aux["next_rays"], target_rays)
    compass_loss = torch.nn.functional.mse_loss(aux["value_compass"], target_compass)
    total = 0.10 * aux_loss + 0.15 * compass_loss
    assert torch.isfinite(total)
    total.backward()


def test_t1_f10_checkpoint_conforms_to_schema_md_4(tmp_path: Path) -> None:
    import torch

    ppo = import_training("ppo")
    model_mod = import_training("model")
    model_cfg = {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    net = model_mod.HybridDoomNet(model_cfg)
    ckpt_path = tmp_path / "ckpt_test.pt"

    save_fn = getattr(ppo, "save_checkpoint", None)
    assert callable(save_fn), "training.ppo must expose save_checkpoint()"
    save_fn(
        ckpt_path,
        model=net,
        model_config=model_cfg,
        global_step=1024,
        update_idx=1,
        scorer_summary={"total_score": 0.0, "mean_act_ms": 0.0, "p99_act_ms": 0.0},
    )
    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    required = {
        "contract_version",
        "git_sha",
        "global_step",
        "update_idx",
        "model_config",
        "model_state_dict",
        "scorer_summary",
    }
    assert required <= set(payload.keys())


# ===========================================================================
# F11: Optional / Synthetic-Compatible BC Pretrainer (5 tests)
# ===========================================================================


def test_t1_f11_synthetic_npz_dataset_loads_schema_md_3(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    npz_path = write_synthetic_npz(tmp_path / "demos" / "ep001.npz", num_steps=40, seed=1)
    ds_cls = getattr(bc_mod, "DemoChunkDataset", None) or getattr(bc_mod, "BCDataset", None)
    assert ds_cls is not None, "training.bc.train_bc must expose DemoChunkDataset or BCDataset"

    ds = ds_cls([npz_path], seq_len=32)
    assert len(ds) >= 1


def test_t1_f11_train_bc_runs_one_epoch_and_writes_checkpoint(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    demo_dir = tmp_path / "demos"
    write_synthetic_npz(demo_dir / "ep001.npz", num_steps=36, seed=1)
    write_synthetic_npz(demo_dir / "ep002.npz", num_steps=36, seed=2)
    out_ckpt = tmp_path / "out" / "bc_warmstart.pt"

    cfg_path = tmp_path / "bc_smoke.yaml"
    cfg = {
        "seed": 42,
        "demo_dir": str(demo_dir),
        "output_checkpoint": str(out_ckpt),
        "seq_len": 32,
        "batch_seqs": 2,
        "epochs": 1,
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "max_grad_norm": 0.5,
        "pos_weight_use": 2.5,
        "pos_weight_attack": 2.5,
        "rgb_dropout_p": 0.15,
        "val_split": 0.0,
        "device": "cpu",
        "wandb_project": "devforge-doom",
        "wandb_enabled": False,
        "model_config": {
            "d_model": 256,
            "gru_dim": 256,
            "num_slots": 4,
            "slot_dim": 8,
            "use_rgb": True,
            "use_grid": True,
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    train_fn = getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None)
    assert callable(train_fn), "training.bc.train_bc must expose train() or train_bc()"
    train_fn(cfg_path)
    assert out_ckpt.is_file()


def test_t1_f11_bc_checkpoint_conforms_to_schema_md_4_and_loads_in_net(tmp_path: Path) -> None:
    import torch

    bc_mod = import_training("bc.train_bc")
    model_mod = import_training("model")
    demo_dir = tmp_path / "demos"
    write_synthetic_npz(demo_dir / "ep001.npz", num_steps=32, seed=3)
    out_ckpt = tmp_path / "bc_warmstart.pt"
    cfg_path = tmp_path / "bc.yaml"
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
                "model_config": {
                    "d_model": 256,
                    "gru_dim": 256,
                    "num_slots": 4,
                    "slot_dim": 8,
                    "use_rgb": True,
                    "use_grid": True,
                },
            }
        ),
        encoding="utf-8",
    )
    train_fn = getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None)
    train_fn(cfg_path)

    payload = torch.load(out_ckpt, map_location="cpu", weights_only=False)
    for k in ("contract_version", "git_sha", "global_step", "update_idx", "model_config", "model_state_dict", "scorer_summary"):
        assert k in payload
    net = model_mod.HybridDoomNet(payload["model_config"])
    net.load_state_dict(payload["model_state_dict"])


def test_t1_f11_positive_class_weight_increases_use_attack_loss() -> None:
    import torch

    bc_mod = import_training("bc.train_bc")
    loss_fn = getattr(bc_mod, "compute_bc_loss", None)
    assert callable(loss_fn), "training.bc.train_bc must expose compute_bc_loss()"

    logits = torch.zeros(2, 9, dtype=torch.float32)
    targets = torch.zeros(2, 9, dtype=torch.float32)
    targets[:, 6] = 1.0  # ATTACK
    targets[:, 7] = 1.0  # USE
    mask = torch.ones(2, 9, dtype=torch.bool)

    l_unweighted = loss_fn(logits, targets, mask, pos_weight_use=1.0, pos_weight_attack=1.0)
    l_weighted = loss_fn(logits, targets, mask, pos_weight_use=2.5, pos_weight_attack=2.5)
    assert float(l_weighted) > float(l_unweighted)


def test_t1_f11_bc_accepts_48x64_npz_frames_without_resize_error(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    npz_path = write_synthetic_npz(tmp_path / "ep_48x64.npz", num_steps=32, seed=9)
    with np.load(npz_path) as data:
        assert data["rgb"].shape == (32, 48, 64, 3)
        assert data["depth"].shape == (32, 48, 64)

    ds_cls = getattr(bc_mod, "DemoChunkDataset", None) or getattr(bc_mod, "BCDataset", None)
    ds = ds_cls([npz_path], seq_len=32)
    sample = ds[0]
    assert sample is not None


# ===========================================================================
# F12: Offline DoomAgent, doom_env.remote, validate.py & Security (5 tests)
# ===========================================================================


def test_t1_f12_doom_agent_init_reset_act_contract(make_live_env) -> None:
    from doom_env import action_space, observation_space
    from submission.agent import DoomAgent

    agent = DoomAgent(observation_space(), action_space(), "cpu")
    agent.reset()
    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=200)
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        act = agent.act(obs)
        arr = np.asarray(act)
        assert arr.shape == (9,)
        assert arr.dtype == np.int8
        assert set(np.unique(arr).tolist()) <= {0, 1}
        obs, _, term, trunc, _ = env.step(arr)
        if term or trunc:
            break


def test_t1_f12_doom_agent_remote_ipc_multi_map_reset(tmp_path: Path, make_live_env) -> None:
    from doom_env import remote
    from tests.e2e.conftest import wad_from_grid

    sock_path = tmp_path / "agent.sock"
    srv = remote.listen(sock_path)
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "doom_env.remote",
            "serve",
            "--socket",
            remote.address(srv),
            "--repo",
            str(REPO_ROOT),
            "--device",
            "cpu",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    conn = remote.AgentConnection.accept(srv, timeout=20, alive=lambda: proc.poll() is None)
    try:
        ready = conn.init("cpu", timeout=20)
        assert ready["device"] == "cpu"
        for wad in (TWO_ROOMS_WAD, wad_from_grid("#######\n#P...E#\n#######")):
            env = make_live_env(wad, episode_ticks=80)
            obs, _ = env.reset(seed=0)
            conn.reset(timeout=5)
            for _ in range(5):
                action = conn.act(obs, timeout=5)
                assert action.shape == (9,) and action.dtype == np.int8
                assert set(np.unique(action).tolist()) <= {0, 1}
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break
    finally:
        conn.close()
        srv.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def test_t1_f12_validate_py_passes_on_repo() -> None:
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate.py"), "--repo", str(REPO_ROOT), "--map", "maps/two_rooms.wad", "--ticks", "600"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "PASSED" in proc.stdout


def test_t1_f12_validate_check_imports_zero_warnings() -> None:
    import validate

    validate.results.clear()
    validate.check_imports(REPO_ROOT)
    warnings_or_fails = [r for r in validate.results if r[0] in (validate.WARN, validate.FAIL)]
    assert not warnings_or_fails, f"check_imports flagged issues: {warnings_or_fails}"


def test_t1_f12_doom_agent_single_thread_latency_budget() -> None:
    import torch
    from doom_env import action_space, observation_space
    from submission.agent import DoomAgent

    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        agent = DoomAgent(observation_space(), action_space(), "cpu")
        agent.reset()
        obs = make_synthetic_obs(frame_shape=(120, 160), seed=12)
        lat_ms = []
        for _ in range(200):
            t0 = time.perf_counter()
            act = agent.act(obs)
            lat_ms.append((time.perf_counter() - t0) * 1000.0)
            assert act.shape == (9,)
        assert np.mean(lat_ms) < 25.0, f"Mean latency too high: {np.mean(lat_ms):.2f} ms"
        assert np.max(lat_ms) < 50.0, f"Max latency spike too high: {np.max(lat_ms):.2f} ms"
    finally:
        torch.set_num_threads(prev_threads)
