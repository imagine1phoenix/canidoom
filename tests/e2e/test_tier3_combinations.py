"""Tier 3 Opaque-Box E2E Tests: Pairwise Cross-Feature Combinations (14 tests across F1-F12).

All `training.*` imports are strictly lazy inside test bodies.
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.e2e.conftest import (
    CONFIGS_DIR,
    REPO_ROOT,
    TWO_ROOMS_WAD,
    import_training,
    load_yaml,
    make_synthetic_obs,
    wad_from_grid,
    write_synthetic_npz,
)

pytestmark = pytest.mark.e2e


def _sample_map(tier: int, knobs: dict | None = None, seed: int = 100001, scaffolding_alpha: float = 0.0):
    tiers_mod = import_training("mapgen.tiers")
    MapGen = tiers_mod.MapGen
    sig = inspect.signature(MapGen.sample)
    if "self" in sig.parameters:
        return MapGen().sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)
    return MapGen.sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)


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


def test_t3_01_f1_config_with_f3_hybrid_doom_net() -> None:
    import torch

    bc_cfg = load_yaml(CONFIGS_DIR / "bc.yaml")
    model_mod = import_training("model")
    prep = import_training("preprocessing")

    net = model_mod.HybridDoomNet(bc_cfg["model_config"])
    inp, _ = prep.preprocess(make_synthetic_obs(frame_shape=(120, 160)), None)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, 1, bc_cfg["model_config"]["gru_dim"]),
        obj_slots=torch.zeros(1, bc_cfg["model_config"]["num_slots"], bc_cfg["model_config"]["slot_dim"]),
    )
    logits, val, aux, _ = net(inp, h, return_aux=True)
    assert logits.shape == (1, 9) and torch.isfinite(val).all()
    assert aux["value_compass"].shape == (1, 8)


def test_t3_02_f2_preprocess_feeds_f3_hybrid_doom_net_live_and_npz(make_live_env) -> None:
    import torch

    prep = import_training("preprocessing")
    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    )
    net.eval()

    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=100, render_buffers=True)
    live_obs, _ = env.reset(seed=0)
    npz_obs = make_synthetic_obs(frame_shape=(48, 64), seed=4)

    h = model_mod.RecurrentState(gru_h=torch.zeros(1, 1, 256), obj_slots=torch.zeros(1, 4, 8))
    with torch.no_grad():
        inp_live, state = prep.preprocess(live_obs, None)
        l1, v1, aux1, h1 = net(inp_live, h, return_aux=True)
        inp_npz, _ = prep.preprocess(npz_obs, state)
        l2, v2, aux2, _ = net(inp_npz, h1, return_aux=True)

    assert torch.isfinite(l1).all() and torch.isfinite(l2).all()
    assert torch.isfinite(aux1["value_compass"]).all() and torch.isfinite(aux2["value_compass"]).all()


def test_t3_03_f5_mapgen_with_f4_reward_shaper_exploration(make_live_env) -> None:
    rewards_mod = import_training("rewards")
    wad_path, map_name = _sample_map(tier=2, seed=100303, scaffolding_alpha=0.5)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    env.reset(seed=0)

    shaper = rewards_mod.RewardShaper()
    shaper.reset(env.sector_polygons())
    priv_prev = env.privileged_state()
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    total_r = 0.0
    for _ in range(15):
        _, _, term, trunc, info = env.step(fwd)
        priv_now = env.privileged_state()
        r = shaper.step(priv_prev, priv_now, fwd, info)
        assert np.isfinite(r)
        total_r += r
        priv_prev = priv_now
        if term or trunc:
            break
    assert np.isfinite(total_r)


def test_t3_04_f6_plr_buffer_drives_f5_mapgen_across_scaffolding_alphas(make_live_env) -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256, scaffolding_end_step=1_000_000)

    add_fn = getattr(buf, "add", None) or getattr(buf, "insert", None) or getattr(buf, "register", None)
    add_fn(
        {
            "level_id": "proc_t2_s100401",
            "tier": 2,
            "knobs": {"num_rooms": 3, "num_keys": 0, "enemies_per_room": 2, "ammo_mult": 1.0},
            "seed": 100401,
            "score_gae": 0.9,
            "last_seen_update": 1,
            "visits": 1,
        }
    )
    alpha_fn = getattr(buf, "scaffolding_alpha", None) or getattr(buf, "get_scaffolding_alpha", None)
    sample = buf.sample()
    sample_dict = sample if isinstance(sample, dict) else vars(sample)

    for step in (0, 500_000, 1_000_000):
        alpha = float(alpha_fn(step))
        wad, map_name = _sample_map(
            tier=sample_dict["tier"],
            knobs=sample_dict["knobs"],
            seed=sample_dict["seed"],
            scaffolding_alpha=alpha,
        )
        env = make_live_env(wad, map_name=map_name, episode_ticks=80, render_buffers=False)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs)


def test_t3_05_f9_llm_coach_proposals_apply_to_f6_plr_buffer(monkeypatch) -> None:
    coach = import_training("llm_coach")
    plr_mod = import_training("curriculum_plr")

    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")
    proposal = {"num_rooms": 5, "enemies_per_room": 3, "ammo_mult": 0.85}

    class _Resp:
        def read(self):
            return json.dumps({"choices": [{"message": {"content": json.dumps(proposal)}}]}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    buf = plr_mod.PLRBuffer(capacity=256)
    updated_knobs = coach.advise_curriculum(
        {"exit_rate": 0.45, "sector_coverage": 0.70},
        {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0},
    )
    updater = getattr(buf, "apply_knobs", None) or getattr(buf, "set_knobs", None) or getattr(buf, "update_knobs", None)
    updater(updated_knobs)
    sampled = buf.sample()
    sd = sampled if isinstance(sampled, dict) else vars(sampled)
    assert sd["knobs"]["num_rooms"] == 5
    assert float(sd["knobs"]["ammo_mult"]) == pytest.approx(0.85)


def test_t3_06_f5_mapgen_wads_hotswap_in_f7_vec_env() -> None:
    w1, _ = _sample_map(tier=1, seed=100501, scaffolding_alpha=0.0)
    w2, _ = _sample_map(tier=2, seed=100502, scaffolding_alpha=0.0)
    venv = _make_vec_env(2, w1, episode_ticks=150, render_buffers_ratio=0.0)
    try:
        venv.reset()
        sig = inspect.signature(venv.reset_with_wad)
        if "env_idx" in sig.parameters:
            venv.reset_with_wad(0, str(w2), "MAP01")
            venv.reset_with_wad(1, str(w2), "MAP01")
        else:
            venv.reset_with_wad([str(w2), str(w2)])
        obs, rewards, _, _, _ = venv.step(np.zeros((2, 9), dtype=np.int8))
        assert len(obs) == 2 and np.isfinite(rewards).all()
    finally:
        venv.close()


def test_t3_07_f4_reward_shaper_inside_f7_vec_env() -> None:
    corridor_wad = wad_from_grid("#######\n#P...E#\n#######")
    venv = _make_vec_env(1, corridor_wad, episode_ticks=400, render_buffers_ratio=0.0)
    try:
        venv.reset()
        fwd = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.int8)
        saw_exit_reward = False
        for _ in range(120):
            _, rewards, term, trunc, _ = venv.step(fwd)
            if bool(term[0]) and float(rewards[0]) >= 10.0:
                saw_exit_reward = True
                break
            if bool(trunc[0]):
                break
        assert saw_exit_reward, "Expected VecDoomEnv + RewardShaper to emit exit bonus >= 10.0"
    finally:
        venv.close()


def test_t3_08_f7_vec_env_obs_feed_f8_rnd_and_f2_preprocess() -> None:
    import torch

    prep = import_training("preprocessing")
    rnd_mod = import_training("rnd")
    cls = getattr(rnd_mod, "RNDModule", None) or getattr(rnd_mod, "RND", None)

    venv = _make_vec_env(2, TWO_ROOMS_WAD, episode_ticks=150, render_buffers_ratio=0.5)
    try:
        obs_list, _ = venv.reset()
        obs_list, _, _, _, _ = venv.step(np.zeros((2, 9), dtype=np.int8))
        preps = [prep.preprocess(o, None)[0] for o in obs_list]
        feats = torch.cat([torch.cat([p.rays.flatten(1), p.vitals], dim=-1) for p in preps], dim=0)
        grids = torch.cat([p.grid for p in preps], dim=0)

        sig = inspect.signature(cls.__init__)
        rnd = cls(feature_dim=feats.shape[-1]) if "feature_dim" in sig.parameters else cls(feats.shape[-1])
        rew, loss = rnd.compute_intrinsic_reward(feats) if hasattr(rnd, "compute_intrinsic_reward") else rnd(feats)
        grid_fn = getattr(rnd, "grid_novelty", None) or getattr(rnd_mod, "compute_grid_novelty", None)
        g_bonus = grid_fn(grids)
        assert rew.shape == (2,) and torch.isfinite(rew).all()
        assert g_bonus.shape[0] == 2 and torch.isfinite(g_bonus).all()
        assert torch.isfinite(loss)
    finally:
        venv.close()


def test_t3_09_f2_preprocess_and_f10_ppo_masked_bernoulli() -> None:
    import torch

    prep = import_training("preprocessing")
    ppo = import_training("ppo")
    obs = make_synthetic_obs(selected_weapon=2.0, selected_weapon_ammo=0.0)
    inp, _ = prep.preprocess(obs, None)
    assert bool(inp.action_mask[0, 6]) is False

    fn = getattr(ppo, "masked_bernoulli_log_prob_entropy", None) or getattr(ppo, "masked_bernoulli", None)
    logits = torch.zeros(1, 9, dtype=torch.float32)
    logits[0, 6] = 100.0
    actions = torch.zeros(1, 9, dtype=torch.float32)
    lp, ent = fn(logits, inp.action_mask, actions)[:2]
    assert torch.isfinite(lp).all() and torch.isfinite(ent).all()


def test_t3_10_f3_value_compass_with_f10_ppo_aux_update() -> None:
    import torch

    model_mod = import_training("model")
    prep = import_training("preprocessing")
    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    inp, _ = prep.preprocess(make_synthetic_obs(frame_shape=(120, 160)), None)
    h = model_mod.RecurrentState(gru_h=torch.zeros(1, 1, 256), obj_slots=torch.zeros(1, 4, 8))

    logits, val, aux, _ = net(inp, h, return_aux=True)
    target_compass = torch.linspace(-0.5, 1.0, 8).unsqueeze(0)
    target_rays = torch.rand(1, 64)
    loss = (
        0.10 * torch.nn.functional.smooth_l1_loss(aux["next_rays"], target_rays)
        + 0.15 * torch.nn.functional.mse_loss(aux["value_compass"], target_compass)
        + 0.01 * logits.pow(2).mean()
        + 0.5 * val.pow(2).mean()
    )
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
    opt.step()
    assert torch.isfinite(loss)


def test_t3_11_f8_rnd_intrinsic_combines_with_f10_ppo_gae() -> None:
    import torch

    rnd_mod = import_training("rnd")
    ppo = import_training("ppo")
    cls = getattr(rnd_mod, "RNDModule", None) or getattr(rnd_mod, "RND", None)
    rnd = cls(feature_dim=64) if "feature_dim" in inspect.signature(cls.__init__).parameters else cls(64)

    feats = torch.randn(8, 64)
    intr, _ = rnd.compute_intrinsic_reward(feats) if hasattr(rnd, "compute_intrinsic_reward") else rnd(feats)
    extr = torch.full((8,), -0.002)
    combined = (extr + 0.05 * intr.detach()).unsqueeze(1)  # (T=8, B=1)
    values = torch.zeros(8, 1)
    dones = torch.zeros(8, 1)
    adv, ret = ppo.compute_gae(combined, values, dones, torch.zeros(1), gamma=0.99, gae_lambda=0.95)
    assert torch.isfinite(adv).all() and torch.isfinite(ret).all()


def test_t3_12_f11_bc_warmstart_loads_into_f10_ppo_with_bc_kl(tmp_path: Path) -> None:
    bc_mod = import_training("bc.train_bc")
    ppo = import_training("ppo")
    demo_dir = tmp_path / "demos"
    write_synthetic_npz(demo_dir / "ep1.npz", num_steps=32, seed=7)
    bc_ckpt = tmp_path / "bc_warmstart.pt"

    bc_cfg = tmp_path / "bc.yaml"
    bc_cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "demo_dir": str(demo_dir),
                "output_checkpoint": str(bc_ckpt),
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
    (getattr(bc_mod, "train", None) or getattr(bc_mod, "train_bc", None))(bc_cfg)

    ppo_cfg = tmp_path / "ppo_bc.yaml"
    ppo_cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
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
                "bc_kl_beta_init": 0.20,
                "compass_coef": 0.15,
                "rgb_dropout_p": 0.0,
                "scaffolding_end_step": 1000,
                "tier_weights": [1.0, 1.5, 2.5, 3.0],
                "worker_max_episodes": 100,
                "worker_timeout_s": 15.0,
                "render_buffers_ratio": 0.0,
                "total_updates": 1,
                "wad_path": str(TWO_ROOMS_WAD),
                "resume_checkpoint": str(bc_ckpt),
                "bc_anchor_checkpoint": str(bc_ckpt),
                "output_dir": str(tmp_path / "ppo_run"),
                "device": "cpu",
                "wandb_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (getattr(ppo, "train", None) or getattr(ppo, "run_training", None))(ppo_cfg)
    assert list((tmp_path / "ppo_run").rglob("*.pt"))


def test_t3_13_f10_ppo_checkpoint_loads_into_f3_net_and_f12_remote_ipc(tmp_path: Path) -> None:
    import torch
    from doom_env import remote

    ppo = import_training("ppo")
    model_mod = import_training("model")
    model_cfg = {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    net = model_mod.HybridDoomNet(model_cfg)
    ckpt = tmp_path / "agent_ckpt.pt"
    ppo.save_checkpoint(
        ckpt,
        model=net,
        model_config=model_cfg,
        global_step=500,
        update_idx=1,
        scorer_summary={"total_score": 10.0, "mean_act_ms": 5.0, "p99_act_ms": 8.0},
    )

    repo = tmp_path / "temp_sub_repo"
    (repo / "submission").mkdir(parents=True)
    (repo / "submission" / "agent.py").write_text(
        f"""
from pathlib import Path
import numpy as np
import torch
from training.model import HybridDoomNet, RecurrentState
from training.preprocessing import preprocess

class DoomAgent:
    def __init__(self, observation_space, action_space, device):
        torch.set_num_threads(1)
        self.device = device
        payload = torch.load({str(ckpt)!r}, map_location=device, weights_only=False)
        self.net = HybridDoomNet(payload["model_config"]).to(device).eval()
        self.net.load_state_dict(payload["model_state_dict"])
        self.state = None
        self.h = None
        self.rng = np.random.default_rng(0)
    def reset(self):
        self.state = None
        self.h = RecurrentState(gru_h=torch.zeros(1, 1, 256, device=self.device), obj_slots=torch.zeros(1, 4, 8, device=self.device))
        self.rng = np.random.default_rng(0)
    def act(self, obs):
        inp, self.state = preprocess(obs, self.state)
        with torch.inference_mode():
            logits, _, _, self.h = self.net(inp, self.h, return_aux=False)
        probs = torch.sigmoid(logits[0] / 0.85).cpu().numpy()
        act = (self.rng.random(9) < probs).astype(np.int8)
        self.state.prev_action = act.astype(np.float32)
        return act
""",
        encoding="utf-8",
    )

    srv = remote.listen(tmp_path / "ipc.sock")
    env_vars = {**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT)}
    proc = subprocess.Popen(
        [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(srv), "--repo", str(repo), "--device", "cpu"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env=env_vars,
    )
    conn = remote.AgentConnection.accept(srv, timeout=20, alive=lambda: proc.poll() is None)
    try:
        conn.init("cpu", timeout=20)
        conn.reset(timeout=5)
        action = conn.act(make_synthetic_obs(), timeout=5)
        assert action.shape == (9,) and action.dtype == np.int8
    finally:
        conn.close()
        srv.close()
        proc.kill()
        proc.wait()


def test_t3_14_f1_eval_yaml_with_f12_validate_py_and_coach_summary(monkeypatch) -> None:
    eval_cfg = load_yaml(CONFIGS_DIR / "eval.yaml")
    coach = import_training("llm_coach")
    monkeypatch.delenv("CREO_API_KEY", raising=False)

    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate.py"), "--repo", str(REPO_ROOT), "--map", "maps/two_rooms.wad", "--ticks", "600"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    scorecard = {
        "total_score": 50.0,
        "cpu_threads": eval_cfg["cpu_threads"],
        "max_vram_mb": eval_cfg["max_vram_mb"],
        "max_ram_mb": eval_cfg["max_ram_mb"],
        "passed_limits": True,
    }
    summary = coach.summarize_evaluation(scorecard)
    assert isinstance(summary, str) and len(summary.strip()) > 0
