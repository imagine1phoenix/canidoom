"""Tier 4 Opaque-Box E2E Tests: Full End-to-End Realistic Workloads & Security Sweep (7 tests).

All `training.*` imports are strictly lazy inside test bodies.
Tests `test_t4_06_*` and `test_t4_07_*` exercise already-present `submission/`, `env/doom_env/`,
`validate.py`, and repo security invariants so they execute and pass immediately.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.e2e.conftest import (
    REPO_ROOT,
    SUBMISSION_DIR,
    TWO_ROOMS_WAD,
    freedoom_wad_path,
    import_training,
    make_dummy_model_inputs,
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


def test_t4_01_self_evolving_rl_from_scratch_zero_demos(tmp_path: Path) -> None:
    import torch

    ppo = import_training("ppo")
    wad_path, _ = _sample_map(tier=1, seed=101001, scaffolding_alpha=0.5)
    run_dir = tmp_path / "t4_01_run"
    cfg_path = tmp_path / "ppo_t4_01.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
                "num_envs": 2,
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
                "render_buffers_ratio": 0.5,
                "total_updates": 2,
                "wad_path": str(wad_path),
                "demo_dir": str(tmp_path / "no_demos_needed"),
                "output_dir": str(run_dir),
                "device": "cpu",
                "wandb_enabled": False,
            }
        ),
        encoding="utf-8",
    )

    train_fn = getattr(ppo, "train", None) or getattr(ppo, "run_training", None)
    train_fn(cfg_path)

    ckpts = sorted(run_dir.rglob("*.pt"))
    assert ckpts, "T4-01 failed to emit checkpoint from pure self-evolving RL"
    payload = torch.load(ckpts[-1], map_location="cpu", weights_only=False)
    for key in ("contract_version", "git_sha", "global_step", "update_idx", "model_config", "model_state_dict", "scorer_summary"):
        assert key in payload


def test_t4_02_extreme_random_map_generalization_and_coordinate_free_invariance(make_live_env) -> None:
    import torch

    prep = import_training("preprocessing")
    model_mod = import_training("model")
    rewards_mod = import_training("rewards")

    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    )
    net.eval()

    extreme_specs = [
        (1, {"num_rooms": 1, "enemies_per_room": 4, "ammo_mult": 0.5}, 102001),
        (2, {"num_rooms": 5, "enemies_per_room": 2, "ammo_mult": 1.0}, 102002),
        (3, {"num_rooms": 4, "num_keys": 1, "enemies_per_room": 3, "ammo_mult": 0.4}, 102003),
        (4, {"num_rooms": 3, "enemies_per_room": 5, "ammo_mult": 0.25}, 102004),
    ]
    for tier, knobs, seed in extreme_specs:
        wad_path, map_name = _sample_map(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=0.0)
        env = make_live_env(wad_path, map_name=map_name, episode_ticks=120, render_buffers=True)
        obs, _ = env.reset(seed=0)
        state = None
        h = model_mod.RecurrentState(gru_h=torch.zeros(1, 1, 256), obj_slots=torch.zeros(1, 4, 8))
        for _ in range(5):
            inp, state = prep.preprocess(obs, state)
            with torch.no_grad():
                logits, val, aux, h = net(inp, h, return_aux=True)
            assert torch.isfinite(logits).all() and torch.isfinite(val).all()
            assert torch.isfinite(aux["value_compass"]).all()
            act = (torch.sigmoid(logits[0]) > 0.5).to(torch.int8).numpy()
            state.prev_action = act.astype(np.float32)
            obs, _, term, trunc, _ = env.step(act)
            if term or trunc:
                break

    # Coordinate-free translation invariance across shifted fixture WADs
    grid = "#########\n#P.....z#\n#..D....#\n#########"
    shifted = "\n".join(["#" * 13] * 3 + ["####" + row for row in grid.splitlines()])
    ea = make_live_env(wad_from_grid(grid), episode_ticks=200, render_buffers=False)
    eb = make_live_env(wad_from_grid(shifted), episode_ticks=200, render_buffers=False)
    ea.reset(seed=3)
    eb.reset(seed=3)
    sa, sb = rewards_mod.RewardShaper(), rewards_mod.RewardShaper()
    sa.reset(ea.sector_polygons())
    sb.reset(eb.sector_polygons())
    pa_prev, pb_prev = ea.privileged_state(), eb.privileged_state()
    assert pa_prev["x"] != pb_prev["x"]

    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    for _ in range(8):
        _, _, ta, tra, ia = ea.step(fwd)
        _, _, tb, trb, ib = eb.step(fwd)
        pa_now, pb_now = ea.privileged_state(), eb.privileged_state()
        ra = sa.step(pa_prev, pa_now, fwd, ia)
        rb = sb.step(pb_prev, pb_now, fwd, ib)
        assert ra == pytest.approx(rb, abs=1e-4)
        pa_prev, pb_prev = pa_now, pb_now
        if ta or tra or tb or trb:
            break


def test_t4_03_plr_curriculum_scaffolding_annealing_and_mocked_llm_coach(
    tmp_path: Path, monkeypatch
) -> None:
    plr_mod = import_training("curriculum_plr")
    coach = import_training("llm_coach")
    envs_mod = import_training("envs")

    monkeypatch.setenv("CREO_API_KEY", "test-dummy-token-000")

    class _Resp:
        def read(self):
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({"num_rooms": 3, "enemies_per_room": 2, "ammo_mult": 0.9})}}]}
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())

    buf = plr_mod.PLRBuffer(capacity=256, scaffolding_end_step=1_000_000)
    add_fn = getattr(buf, "add", None) or getattr(buf, "insert", None) or getattr(buf, "register", None)
    add_fn(
        {
            "level_id": "proc_t2_s103001",
            "tier": 2,
            "knobs": {"num_rooms": 4, "num_keys": 0, "enemies_per_room": 2, "ammo_mult": 1.0},
            "seed": 103001,
            "score_gae": 1.5,
            "last_seen_update": 1,
            "visits": 1,
        }
    )
    knobs = coach.advise_curriculum(
        {"exit_rate": 0.4, "sector_coverage": 0.65, "stuck_rate": 0.1},
        {"num_rooms": 4, "enemies_per_room": 2, "ammo_mult": 1.0},
    )
    updater = getattr(buf, "apply_knobs", None) or getattr(buf, "set_knobs", None) or getattr(buf, "update_knobs", None)
    updater(knobs)

    alpha_fn = getattr(buf, "scaffolding_alpha", None) or getattr(buf, "get_scaffolding_alpha", None)
    alphas = [float(alpha_fn(s)) for s in (0, 500_000, 1_000_000)]
    assert alphas[0] == pytest.approx(1.0) and alphas[-1] == pytest.approx(0.0)

    sampled = buf.sample()
    sd = sampled if isinstance(sampled, dict) else vars(sampled)
    wad_vanilla, map_name = _sample_map(tier=sd["tier"], knobs=sd["knobs"], seed=sd["seed"], scaffolding_alpha=alphas[-1])

    sig = inspect.signature(envs_mod.VecDoomEnv.__init__)
    kw = {"num_envs": 1, "episode_ticks": 120, "render_buffers_ratio": 0.0}
    kw["wad_path" if "wad_path" in sig.parameters else "wad"] = str(wad_vanilla)
    venv = envs_mod.VecDoomEnv(**kw)
    try:
        venv.reset()
        _, rew, _, _, _ = venv.step(np.zeros((1, 9), dtype=np.int8))
        assert np.isfinite(rew).all()
    finally:
        venv.close()


def test_t4_04_ego_value_compass_spatial_grid_and_bptt_done_boundary(make_live_env) -> None:
    import torch

    prep = import_training("preprocessing")
    model_mod = import_training("model")
    ppo = import_training("ppo")

    net = model_mod.HybridDoomNet(
        {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    )
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)

    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=120, render_buffers=True)
    obs, _ = env.reset(seed=0)
    state = None
    inputs_seq = []
    dones = []
    for step_idx in range(8):
        inp, state = prep.preprocess(obs, state)
        inputs_seq.append(inp)
        act = np.array([1, 0, 0, 0, int(step_idx % 2 == 0), 0, 0, int(step_idx == 2), 0], dtype=np.int8)
        state.prev_action = act.astype(np.float32)
        obs, _, term, trunc, _ = env.step(act)
        dones.append([1.0 if (term or trunc or step_idx == 3) else 0.0])
        if term or trunc:
            obs, _ = env.reset(seed=1)
            state = None

    unroll_fn = getattr(ppo, "bptt_unroll", None) or getattr(ppo, "unroll_bptt_chunk", None)
    init_h = model_mod.RecurrentState(gru_h=torch.zeros(1, 1, 256), obj_slots=torch.zeros(1, 4, 8))
    out = unroll_fn(net, inputs_seq, torch.tensor(dones, dtype=torch.float32), init_h)
    logits_seq, values_seq = out[0], out[1]
    assert torch.isfinite(logits_seq).all() and torch.isfinite(values_seq).all()

    # Aux compass + next_rays loss check
    _, _, aux, _ = net(inputs_seq[-1], init_h, return_aux=True)
    loss = (
        0.15 * torch.nn.functional.mse_loss(aux["value_compass"], torch.zeros(1, 8))
        + 0.10 * torch.nn.functional.smooth_l1_loss(aux["next_rays"], torch.zeros(1, 64))
        + 0.01 * logits_seq.pow(2).mean()
    )
    opt.zero_grad()
    loss.backward()
    opt.step()
    assert torch.isfinite(loss)


def test_t4_05_optional_synthetic_bc_warmstart_to_ppo_to_agent(tmp_path: Path, make_live_env) -> None:
    import torch

    bc_mod = import_training("bc.train_bc")
    ppo = import_training("ppo")
    model_mod = import_training("model")
    prep = import_training("preprocessing")

    demo_dir = tmp_path / "demos"
    write_synthetic_npz(demo_dir / "synth_ep.npz", num_steps=32, seed=42)
    bc_ckpt = tmp_path / "bc_warmstart.pt"
    bc_cfg = tmp_path / "bc.yaml"
    bc_cfg.write_text(
        yaml.safe_dump(
            {
                "seed": 42,
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
                "rgb_dropout_p": 0.15,
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
    assert bc_ckpt.is_file()

    ppo_out = tmp_path / "ppo_out"
    ppo_cfg = tmp_path / "ppo.yaml"
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
                "rgb_dropout_p": 0.25,
                "scaffolding_end_step": 1000,
                "tier_weights": [1.0, 1.5, 2.5, 3.0],
                "worker_max_episodes": 100,
                "worker_timeout_s": 15.0,
                "render_buffers_ratio": 0.0,
                "total_updates": 1,
                "wad_path": str(TWO_ROOMS_WAD),
                "resume_checkpoint": str(bc_ckpt),
                "bc_anchor_checkpoint": str(bc_ckpt),
                "output_dir": str(ppo_out),
                "device": "cpu",
                "wandb_enabled": False,
            }
        ),
        encoding="utf-8",
    )
    (getattr(ppo, "train", None) or getattr(ppo, "run_training", None))(ppo_cfg)

    final_ckpts = sorted(ppo_out.rglob("*.pt"))
    assert final_ckpts
    payload = torch.load(final_ckpts[-1], map_location="cpu", weights_only=False)
    net = model_mod.HybridDoomNet(payload["model_config"]).eval()
    net.load_state_dict(payload["model_state_dict"])

    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=200, render_buffers=True)
    obs, _ = env.reset(seed=0)
    state = None
    h = model_mod.RecurrentState(gru_h=torch.zeros(1, 1, 256), obj_slots=torch.zeros(1, 4, 8))
    rng = np.random.default_rng(0)
    for _ in range(20):
        inp, state = prep.preprocess(obs, state)
        with torch.inference_mode():
            logits, _, _, h = net(inp, h, return_aux=False)
        probs = torch.sigmoid(logits[0] / 0.85).numpy()
        act = (rng.random(9) < probs).astype(np.int8)
        state.prev_action = act.astype(np.float32)
        obs, _, term, trunc, _ = env.step(act)
        if term or trunc:
            break


def test_t4_06_end_to_end_remote_ipc_freedoom_and_validate_sandbox(tmp_path: Path, make_live_env) -> None:
    """End-to-end IPC, Freedoom MAP01 + two_rooms.wad execution, latency, and validate.py sandbox check."""
    from doom_env import limits as L
    from doom_env import remote

    sock_path = tmp_path / "e2e_agent.sock"
    srv = remote.listen(sock_path)
    env_vars = {**os.environ, "PYTHONUNBUFFERED": "1", **L.THREAD_ENV}
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
        env=env_vars,
    )
    conn = remote.AgentConnection.accept(srv, timeout=L.INIT_TIMEOUT_S, alive=lambda: proc.poll() is None)
    try:
        ready = conn.init("cpu", timeout=L.INIT_TIMEOUT_S)
        assert ready["device"] == "cpu"

        for wad, map_name in ((TWO_ROOMS_WAD, "MAP01"), (freedoom_wad_path(), "MAP01")):
            env = make_live_env(wad, map_name=map_name, episode_ticks=200)
            obs, _ = env.reset(seed=0)
            conn.reset(timeout=L.RESET_TIMEOUT_S)
            lat_ms = []
            for step_i in range(25):
                t0 = time.perf_counter()
                act = conn.act(obs, timeout=L.FIRST_ACT_TIMEOUT_S if step_i == 0 else L.ACT_TIMEOUT_S)
                lat_ms.append((time.perf_counter() - t0) * 1000.0)
                assert act.shape == (9,) and act.dtype == np.int8
                assert set(np.unique(act).tolist()) <= {0, 1}
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
            assert np.mean(lat_ms) < 25.0, f"IPC mean step latency exceeded 25ms: {np.mean(lat_ms):.2f}ms"
    finally:
        conn.close()
        srv.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    val_proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "validate.py"), "--repo", str(REPO_ROOT), "--map", "maps/two_rooms.wad", "--ticks", "600"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert val_proc.returncode == 0, val_proc.stdout + val_proc.stderr
    assert "PASSED" in val_proc.stdout


def test_t4_07_security_anti_cheat_offline_isolation_and_rule_sweep() -> None:
    """Repo-wide security, secret hygiene, offline submission isolation, and rule-compliance sweep."""
    import validate

    # 1. .env must be gitignored and NEVER tracked in git
    if (REPO_ROOT / ".git").exists():
        ig = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-v", ".env"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert ig.returncode == 0, ".env is NOT matched by .gitignore!"
        tracked_env = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", ".env"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert tracked_env.stdout.strip() == "", f".env is tracked in git: {tracked_env.stdout}"

        # 2. Zero raw secrets in git-tracked files (excluding tests/e2e string patterns if any)
        ls_proc = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            timeout=30,
        )
        assert ls_proc.returncode == 0
        secret_re = re.compile(r"cr_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9]{20,}")
        leaks = []
        for rel in ls_proc.stdout.decode("utf-8", "replace").split("\0"):
            if not rel:
                continue
            p = REPO_ROOT / rel
            if not p.is_file() or p.suffix in {".wad", ".pt", ".npz", ".png", ".mp4"}:
                continue
            try:
                content = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if secret_re.search(content):
                leaks.append(rel)
        assert not leaks, f"Potential secret token found in tracked files: {leaks}"

    # 3. .env.example contains placeholders only
    env_ex = REPO_ROOT / ".env.example"
    assert env_ex.is_file()
    ex_text = env_ex.read_text(encoding="utf-8")
    assert "CREO_BASE_URL=" in ex_text and "CREO_MODEL=" in ex_text and "CREO_API_KEY=" in ex_text
    assert "cr_" not in ex_text

    # 4. submission/ offline isolation & zero banned rule constructs
    forbidden_imports = {"openai", "urllib", "requests", "httpx", "aiohttp", "socket", "http", "vizdoom", "llm_coach"}
    banned_symbols = {"atan2", "astar", "dijkstra", "bfs", "privileged_state", "POSITION_X", "POSITION_Y", "POSITION_Z"}
    for py in SUBMISSION_DIR.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in forbidden_imports and "llm_coach" not in alias.name, alias.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in forbidden_imports and "llm_coach" not in node.module, node.module
            elif isinstance(node, ast.Name):
                assert node.id not in banned_symbols, f"Banned symbol {node.id} in {py}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in banned_symbols, f"Banned attribute {node.attr} in {py}"

    # 5. validate.py requirements & check_imports clean
    validate.results.clear()
    validate.check_requirements(REPO_ROOT)
    validate.check_imports(REPO_ROOT)
    bad = [r for r in validate.results if r[0] in (validate.WARN, validate.FAIL)]
    assert not bad, f"validate.py check_requirements/check_imports issues: {bad}"
