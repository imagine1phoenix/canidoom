"""Unit and integration tests for Vedesh's RL Core (`training/envs.py`, `ppo.py`, `rnd.py`, `llm_coach.py`)."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from doom_env.sensors import RAY_LOCKED
from fixtures import fixture_wad
from training.envs import VecDoomEnv
from training.llm_coach import advise_curriculum, summarize_evaluation
from training.model import HybridDoomNet, RecurrentState
from training.ppo import bptt_unroll, compute_gae, masked_bernoulli_log_prob_entropy, train as train_ppo
from training.preprocessing import preprocess
from training.rnd import RNDModule, RunningMeanStd, compute_grid_novelty

ROOT = Path(__file__).resolve().parent.parent
TWO_ROOMS_WAD = ROOT / "maps" / "two_rooms.wad"


def test_bptt_done_masking() -> None:
    """Verify hidden state `gru_h` and `obj_slots` zero cleanly at `done[t-1] == 1` inside BPTT."""
    net = HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": False, "use_grid": False}
    ).eval()
    obs = {
        "rgb": np.zeros((48, 64, 3), dtype=np.uint8),
        "depth": np.zeros((48, 64), dtype=np.uint8),
        "vitals": np.zeros((17,), dtype=np.float32),
        "rays": {"distance": np.full(64, 0.4, dtype=np.float32), "hit_type": np.ones(64, dtype=np.int64)},
        "visible_entities": {
            "features": np.zeros((32, 5), dtype=np.float32),
            "entity_type": np.zeros((32,), dtype=np.int64),
            "mask": np.zeros((32,), dtype=np.int8),
        },
    }
    inp0, _ = preprocess(obs, None, use_grid=False)
    obs2 = {**obs, "rays": {"distance": np.full(64, 0.8, dtype=np.float32), "hit_type": np.ones(64, dtype=np.int64)}}
    inp1, _ = preprocess(obs2, None, use_grid=False)

    seq_a = [inp0, inp0, inp0, inp0, inp0, inp0]
    seq_b = [inp1, inp1, inp1, inp0, inp0, inp0]
    dones = torch.tensor([[0.0], [0.0], [1.0], [0.0], [0.0], [0.0]], dtype=torch.float32)
    h0 = RecurrentState(gru_h=torch.randn(1, 1, 256), obj_slots=torch.randn(1, 4, 8))

    with torch.no_grad():
        logits_a, vals_a, _, _ = bptt_unroll(net, seq_a, dones, init_h=h0)
        logits_b, vals_b, _, _ = bptt_unroll(net, seq_b, dones, init_h=h0)

    torch.testing.assert_close(logits_a[3:], logits_b[3:], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(vals_a[3:], vals_b[3:], atol=1e-6, rtol=1e-6)


def test_ppo_overfit_single_room(tmp_path: Path) -> None:
    """Verify 100% self-evolving PPO runs end-to-end on a single corridor WAD and emits checkpoints."""
    corridor = fixture_wad("#######\n#P...E#\n#######")
    cfg_path = tmp_path / "ppo_smoke.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 7,
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
                "compass_coef": 0.15,
                "bc_kl_beta_init": 0.0,
                "rgb_dropout_p": 0.25,
                "scaffolding_end_step": 1000,
                "tier_weights": [1.0, 1.5, 2.5, 3.0],
                "worker_max_episodes": 50,
                "worker_timeout_s": 15.0,
                "render_buffers_ratio": 0.0,
                "total_updates": 2,
                "wad_path": str(corridor),
                "output_dir": str(tmp_path / "run"),
                "device": "cpu",
                "wandb_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    res = train_ppo(cfg_path)
    assert res["global_step"] == 64
    assert np.isfinite(res["last_loss"])
    assert res["checkpoint"] is not None and Path(res["checkpoint"]).is_file()


def test_subprocess_env_reload() -> None:
    """Verify VecDoomEnv cleanly swaps WADs across 5 resets and reaps workers on close."""
    locked_wad = fixture_wad("#######\n#P..B.E#\n#######")
    venv = VecDoomEnv(num_envs=2, wad_path=TWO_ROOMS_WAD, episode_ticks=150, render_buffers_ratio=0.0)
    try:
        venv.reset()
        for idx in range(5):
            target_wad = locked_wad if idx % 2 == 0 else TWO_ROOMS_WAD
            obs0, _ = venv.reset_with_wad(0, target_wad, "MAP01")
            if idx % 2 == 0:
                assert int(obs0["rays"]["hit_type"][0]) == RAY_LOCKED
        procs = list(venv.processes)
    finally:
        venv.close()
    assert all(not p.is_alive() for p in procs)


def test_rnd_gae_mask_and_llm_coach_offline(monkeypatch) -> None:
    """Verify RND curiosity, masked Bernoulli, GAE, and offline/mocked llm_coach."""
    rnd = RNDModule(feature_dim=32)
    x = torch.randn(4, 32)
    rew, loss = rnd.compute_intrinsic_reward(x)
    assert rew.shape == (4,) and torch.isfinite(rew).all() and (rew >= 0).all()
    assert torch.isfinite(loss)

    g = torch.zeros(2, 6, 32, 32)
    g[1, 0, 10:15, 10:15] = 1.0
    nov = compute_grid_novelty(g)
    assert float(nov[1]) > float(nov[0])

    rms = RunningMeanStd(shape=(2,))
    rms.update(np.array([[1.0, 2.0]], dtype=np.float32))
    assert np.isfinite(rms.mean).all()

    logits = torch.tensor([[0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 50.0, 50.0, 0.0]])
    mask = torch.tensor([[True, True, True, True, True, True, False, False, True]])
    acts = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    lp, ent = masked_bernoulli_log_prob_entropy(logits, mask, acts)
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()

    adv, ret = compute_gae(
        torch.ones(2, 1), torch.zeros(2, 1), torch.tensor([[0.0], [1.0]]), torch.ones(1)
    )
    assert adv.shape == (2, 1) and ret.shape == (2, 1)

    monkeypatch.delenv("CREO_API_KEY", raising=False)
    k = advise_curriculum({"exit_rate": 0.1, "stuck_rate": 0.4}, {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0})
    assert 1 <= k["num_rooms"] <= 12
    s = summarize_evaluation({"total_score": 55.0, "exit_rate": 0.5})
    assert "55.00" in s
