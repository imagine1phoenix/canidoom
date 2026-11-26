"""Tier 1 Opaque-Box E2E Tests: Core Modules (F1-F6, 30 tests = 5 per feature).

All `training.*` imports are strictly lazy inside test bodies.
"""

from __future__ import annotations

import inspect
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.e2e.conftest import (
    CONFIGS_DIR,
    TWO_ROOMS_WAD,
    import_training,
    load_yaml,
    make_dummy_model_inputs,
    make_priv_state,
    make_square_sector_polygons,
    make_synthetic_obs,
)

pytestmark = pytest.mark.e2e


# ---------------------------------------------------------------------------
# Helpers for opaque-box contract interaction
# ---------------------------------------------------------------------------
def _sample_map(tier: int, knobs: dict | None = None, seed: int = 100001, scaffolding_alpha: float = 0.0):
    tiers_mod = import_training("mapgen.tiers")
    MapGen = tiers_mod.MapGen
    sig = inspect.signature(MapGen.sample)
    if "self" in sig.parameters:
        return MapGen().sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)
    return MapGen.sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)


def _get_config_validator(cfg_mod: Any, kind: str):
    for name in (f"validate_{kind}_config", "validate_config", f"load_{kind}_config", "load_config"):
        fn = getattr(cfg_mod, name, None)
        if callable(fn):
            return name, fn
    return None, None


# ===========================================================================
# F1: YAML Configs & Schema Validation (5 tests)
# ===========================================================================


def test_t1_f1_ppo_curriculum_yaml_schema() -> None:
    cfg_path = CONFIGS_DIR / "ppo_curriculum.yaml"
    assert cfg_path.is_file(), f"Missing {cfg_path}"
    cfg = load_yaml(cfg_path)

    expected_keys = {
        "seed",
        "num_envs",
        "rollout_steps",
        "bptt_len",
        "ppo_epochs",
        "minibatch_chunks",
        "gamma",
        "gae_lambda",
        "clip_coef",
        "lr",
        "max_grad_norm",
        "vf_coef",
        "ent_coef",
        "rnd_coef",
        "grid_novelty_coef",
        "aux_coef",
        "bc_kl_beta_init",
        "compass_coef",
        "rgb_dropout_p",
        "scaffolding_end_step",
        "tier_weights",
        "worker_max_episodes",
        "worker_timeout_s",
        "render_buffers_ratio",
    }
    missing = expected_keys - set(cfg.keys())
    assert not missing, f"ppo_curriculum.yaml missing Schema.md §7 keys: {missing}"
    assert isinstance(cfg["seed"], int)
    assert isinstance(cfg["num_envs"], int) and cfg["num_envs"] >= 1
    assert isinstance(cfg["rollout_steps"], int) and cfg["rollout_steps"] >= cfg["bptt_len"]
    assert isinstance(cfg["bptt_len"], int) and cfg["bptt_len"] >= 1
    assert 0.0 < float(cfg["gamma"]) <= 1.0
    assert 0.0 < float(cfg["gae_lambda"]) <= 1.0
    assert 0.0 < float(cfg["clip_coef"]) < 1.0
    assert float(cfg["lr"]) > 0.0
    assert float(cfg["max_grad_norm"]) > 0.0
    assert 0.0 <= float(cfg["rgb_dropout_p"]) <= 1.0
    assert 0.0 <= float(cfg["render_buffers_ratio"]) <= 1.0


def test_t1_f1_bc_yaml_schema() -> None:
    cfg_path = CONFIGS_DIR / "bc.yaml"
    assert cfg_path.is_file(), f"Missing {cfg_path}"
    cfg = load_yaml(cfg_path)

    expected_keys = {
        "seed",
        "demo_dir",
        "output_checkpoint",
        "seq_len",
        "batch_seqs",
        "epochs",
        "lr",
        "weight_decay",
        "max_grad_norm",
        "pos_weight_use",
        "pos_weight_attack",
        "rgb_dropout_p",
        "val_split",
        "device",
        "wandb_project",
        "wandb_enabled",
        "model_config",
    }
    missing = expected_keys - set(cfg.keys())
    assert not missing, f"bc.yaml missing Schema.md §7 keys: {missing}"
    mc = cfg["model_config"]
    assert isinstance(mc, dict)
    for k in ("d_model", "gru_dim", "num_slots", "slot_dim", "use_rgb", "use_grid"):
        assert k in mc, f"model_config missing {k}"
    assert mc["d_model"] == 256 and mc["gru_dim"] == 256
    assert mc["num_slots"] == 4 and mc["slot_dim"] == 8
    assert float(cfg["pos_weight_use"]) == pytest.approx(2.5)
    assert float(cfg["pos_weight_attack"]) == pytest.approx(2.5)


def test_t1_f1_eval_yaml_schema_and_pr2_limits() -> None:
    cfg_path = CONFIGS_DIR / "eval.yaml"
    assert cfg_path.is_file(), f"Missing {cfg_path}"
    cfg = load_yaml(cfg_path)

    expected_keys = {
        "suite_path",
        "output_dir",
        "device",
        "cpu_threads",
        "max_vram_mb",
        "max_ram_mb",
        "agent_ms_per_step",
        "latency_target_ms",
        "tier_weights",
        "deterministic_seed",
        "run_docker_validate",
    }
    missing = expected_keys - set(cfg.keys())
    assert not missing, f"eval.yaml missing Schema.md §7 keys: {missing}"
    assert cfg["cpu_threads"] == 1
    assert cfg["max_vram_mb"] <= 2048
    assert cfg["max_ram_mb"] <= 8192
    assert float(cfg["agent_ms_per_step"]) == pytest.approx(40.0)
    assert 0.0 < float(cfg["latency_target_ms"]) <= float(cfg["agent_ms_per_step"])


def test_t1_f1_self_evolving_rl_defaults_and_loader() -> None:
    cfg = load_yaml(CONFIGS_DIR / "ppo_curriculum.yaml")
    assert float(cfg["bc_kl_beta_init"]) == pytest.approx(0.0)
    assert float(cfg["compass_coef"]) > 0.0
    assert float(cfg["rnd_coef"]) > 0.0
    assert float(cfg["grid_novelty_coef"]) >= 0.0

    configs_mod = import_training("configs")
    assert configs_mod is not None
    # Verify any exported loader/validator accepts the shipped configs cleanly
    for kind, fname in (("ppo", "ppo_curriculum.yaml"), ("bc", "bc.yaml"), ("eval", "eval.yaml")):
        name, fn = _get_config_validator(configs_mod, kind)
        if fn is not None:
            if name.startswith("load"):
                loaded = fn(CONFIGS_DIR / fname)
            else:
                sig = inspect.signature(fn)
                if len(sig.parameters) >= 2:
                    loaded = fn(load_yaml(CONFIGS_DIR / fname), kind)
                else:
                    loaded = fn(load_yaml(CONFIGS_DIR / fname))
            assert loaded is not None or name.startswith("validate")


def test_t1_f1_tier_weights_and_rollout_chunk_consistency() -> None:
    ppo_cfg = load_yaml(CONFIGS_DIR / "ppo_curriculum.yaml")
    eval_cfg = load_yaml(CONFIGS_DIR / "eval.yaml")

    for name, cfg in (("ppo_curriculum.yaml", ppo_cfg), ("eval.yaml", eval_cfg)):
        tw = cfg["tier_weights"]
        assert isinstance(tw, list) and len(tw) == 4, f"{name} tier_weights invalid: {tw}"
        assert all(isinstance(w, (int, float)) and w > 0.0 for w in tw)

    assert ppo_cfg["rollout_steps"] % ppo_cfg["bptt_len"] == 0
    total_chunks = ppo_cfg["num_envs"] * (ppo_cfg["rollout_steps"] // ppo_cfg["bptt_len"])
    assert total_chunks >= ppo_cfg["minibatch_chunks"] >= 1


# ===========================================================================
# F2: Coordinate-Free Preprocessing, Ego Grid & Action Masking (5 tests)
# ===========================================================================


def test_t1_f2_preprocess_live_120x160_obs_shapes(make_live_env) -> None:
    import torch

    prep = import_training("preprocessing")
    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=200, render_buffers=True)
    obs, _ = env.reset(seed=0)

    inputs, state = prep.preprocess(obs, None)
    assert isinstance(state, prep.OdomGridState)
    assert isinstance(inputs, prep.ModelInputs)
    assert inputs.rays.shape == (1, 8, 64) and inputs.rays.dtype == torch.float32
    assert inputs.ent_feats.shape == (1, 32, 5) and inputs.ent_feats.dtype == torch.float32
    assert inputs.ent_types.shape == (1, 32) and inputs.ent_types.dtype == torch.int64
    assert inputs.ent_mask.shape == (1, 32) and inputs.ent_mask.dtype == torch.bool
    assert inputs.vitals.shape == (1, 36) and inputs.vitals.dtype == torch.float32
    assert inputs.rgb_depth.shape == (1, 4, 48, 64) and inputs.rgb_depth.dtype == torch.float32
    assert inputs.grid.shape == (1, 6, 32, 32) and inputs.grid.dtype == torch.float32
    assert inputs.action_mask.shape == (1, 9) and inputs.action_mask.dtype == torch.bool
    assert torch.isfinite(inputs.rays).all()
    assert torch.isfinite(inputs.vitals).all()
    assert torch.isfinite(inputs.rgb_depth).all()
    assert torch.isfinite(inputs.grid).all()


def test_t1_f2_preprocess_npz_48x64_passthrough() -> None:
    prep = import_training("preprocessing")
    obs = make_synthetic_obs(frame_shape=(48, 64), seed=7)
    inputs, state = prep.preprocess(obs, None)

    assert inputs.rgb_depth.shape == (1, 4, 48, 64)
    expected_rgb = obs["rgb"].transpose(2, 0, 1).astype(np.float32) / 255.0
    expected_depth = obs["depth"].astype(np.float32)[None, ...] / 255.0
    np.testing.assert_allclose(inputs.rgb_depth[0, :3].numpy(), expected_rgb, atol=1e-5)
    np.testing.assert_allclose(inputs.rgb_depth[0, 3:4].numpy(), expected_depth, atol=1e-5)
    assert state.grid.shape == (6, 32, 32)


def test_t1_f2_rays_channels_and_one_hot_encoding() -> None:
    prep = import_training("preprocessing")
    dists = np.linspace(0.05, 1.0, 64, dtype=np.float32)
    hits = np.arange(64, dtype=np.int64) % 6
    obs = make_synthetic_obs(ray_distances=dists, ray_hit_types=hits)

    inputs, _ = prep.preprocess(obs, None)
    r = inputs.rays[0].numpy()
    np.testing.assert_allclose(r[0], dists, atol=1e-5)
    np.testing.assert_allclose(r[1], 1.0 - dists, atol=1e-5)
    one_hot = r[2:8]
    np.testing.assert_allclose(one_hot.sum(axis=0), np.ones(64, dtype=np.float32), atol=1e-6)
    np.testing.assert_array_equal(one_hot.argmax(axis=0), hits)


def test_t1_f2_vitals_packing_and_prev_action_propagation() -> None:
    prep = import_training("preprocessing")
    obs0 = make_synthetic_obs(health=80.0, armor=50.0, selected_weapon=2.0, keys=(1, 0, 0, 0, 0, 0))
    inputs0, state0 = prep.preprocess(obs0, None)
    np.testing.assert_allclose(inputs0.vitals[0, 27:36].numpy(), np.zeros(9, dtype=np.float32))

    taken = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    state0.prev_action = taken.copy()
    inputs1, _ = prep.preprocess(obs0, state0)
    np.testing.assert_allclose(inputs1.vitals[0, 27:36].numpy(), taken, atol=1e-6)
    assert inputs1.vitals[0, 0].item() == pytest.approx(0.80, abs=1e-4)
    assert inputs1.vitals[0, 1].item() == pytest.approx(0.50, abs=1e-4)


def test_t1_f2_egocentric_grid_updates_over_live_steps(make_live_env) -> None:
    prep = import_training("preprocessing")
    env = make_live_env(TWO_ROOMS_WAD, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=1)
    inputs, state = prep.preprocess(obs, None)

    fwd_turn = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8)
    for _ in range(10):
        obs, _, term, trunc, _ = env.step(fwd_turn)
        state.prev_action = fwd_turn.astype(np.float32)
        inputs, state = prep.preprocess(obs, state)
        if term or trunc:
            break

    grid = inputs.grid[0].numpy()
    assert grid.shape == (6, 32, 32)
    assert np.isfinite(grid).all()
    assert (grid >= 0.0).all() and (grid <= 1.0 + 1e-5).all()
    assert grid[0, 16, 16] == pytest.approx(1.0, abs=1e-5)


# ===========================================================================
# F3: HybridDoomNet + 8-Sector Ego Value Compass (5 tests)
# ===========================================================================


def _default_model_config(**overrides: Any) -> dict[str, Any]:
    cfg = {
        "d_model": 256,
        "gru_dim": 256,
        "num_slots": 4,
        "slot_dim": 8,
        "use_rgb": True,
        "use_grid": True,
    }
    cfg.update(overrides)
    return cfg


def test_t1_f3_model_forward_shapes_and_dtypes() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_default_model_config())
    B = 3
    batch = make_dummy_model_inputs(batch_size=B, num_entities=4)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )

    logits, value, aux, next_h = net(batch, h, return_aux=False)
    assert logits.shape == (B, 9) and logits.dtype == torch.float32
    assert value.shape in {(B, 1), (B,)} and value.dtype == torch.float32
    assert aux is None
    assert next_h.gru_h.shape == (1, B, 256)
    assert next_h.obj_slots.shape == (B, 4, 8)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()


def test_t1_f3_return_aux_includes_8_sector_value_compass() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_default_model_config())
    B = 2
    batch = make_dummy_model_inputs(batch_size=B)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )

    logits, value, aux, next_h = net(batch, h, return_aux=True)
    assert isinstance(aux, dict)
    assert "next_rays" in aux and aux["next_rays"].shape == (B, 64)
    assert "value_compass" in aux and aux["value_compass"].shape == (B, 8)
    assert torch.isfinite(aux["next_rays"]).all()
    assert torch.isfinite(aux["value_compass"]).all()


def test_t1_f3_gradient_flow_across_policy_value_and_compass_heads() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_default_model_config())
    B = 2
    batch = make_dummy_model_inputs(batch_size=B)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )

    logits, value, aux, next_h = net(batch, h, return_aux=True)
    loss = logits.sum() + value.sum() + aux["next_rays"].sum() + aux["value_compass"].sum()
    loss.backward()

    grads = [p.grad for p in net.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum().item() > 0.0 for g in grads)
    assert all(g is None or torch.isfinite(g).all() for g in grads)


def test_t1_f3_recurrent_state_evolves_over_sequence() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_default_model_config())
    net.eval()
    B = 1
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )
    with torch.no_grad():
        b0 = make_dummy_model_inputs(batch_size=B, seed=10)
        _, _, _, h1 = net(b0, h, return_aux=False)
        b1 = make_dummy_model_inputs(batch_size=B, seed=11)
        _, _, _, h2 = net(b1, h1, return_aux=False)

    assert not torch.allclose(h1.gru_h, h.gru_h)
    assert not torch.allclose(h2.gru_h, h1.gru_h)
    assert not torch.allclose(h2.obj_slots, h.obj_slots)


def test_t1_f3_circular_ray_wraparound_produces_finite_distinct_compass() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_default_model_config(use_rgb=False, use_grid=False))
    net.eval()
    B = 1
    batch = make_dummy_model_inputs(batch_size=B, num_entities=0, zero_rgb=True, zero_grid=True, seed=3)
    # Create an asymmetric wall opening spanning across ray 63 -> ray 0 seam
    batch.rays.zero_()
    batch.rays[0, 0, :] = 0.2
    batch.rays[0, 0, [62, 63, 0, 1]] = 0.95
    batch.rays[0, 1, :] = 1.0 - batch.rays[0, 0, :]
    batch.rays[0, 3, :] = 1.0  # wall
    batch.rays[0, 3, [62, 63, 0, 1]] = 0.0
    batch.rays[0, 2, [62, 63, 0, 1]] = 1.0  # open

    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )
    with torch.no_grad():
        logits_a, _, aux_a, _ = net(batch, h, return_aux=True)
        # Rotate rays by 180 degrees (32 bins)
        batch_rot = make_dummy_model_inputs(batch_size=B, num_entities=0, zero_rgb=True, zero_grid=True, seed=3)
        batch_rot.rays = torch.roll(batch.rays, shifts=32, dims=-1)
        batch_rot.vitals = batch.vitals.clone()
        logits_b, _, aux_b, _ = net(batch_rot, h, return_aux=True)

    assert torch.isfinite(aux_a["value_compass"]).all()
    assert torch.isfinite(aux_b["value_compass"]).all()
    assert not torch.allclose(logits_a, logits_b, atol=1e-5)


# ===========================================================================
# F4: Coordinate-Free Dense RewardShaper (5 tests)
# ===========================================================================


def test_t1_f4_neutral_step_applies_step_time_cost() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset(make_square_sector_polygons())

    p0 = make_priv_state(tic=4, x=64.0, y=64.0)
    p1 = make_priv_state(tic=8, x=64.0, y=64.0)
    r = shaper.step(p0, p1, np.zeros(9, dtype=np.int8), {"exit_reached": False})
    assert isinstance(r, float)
    assert r == pytest.approx(-0.002, abs=1e-4)


def test_t1_f4_unique_sector_discovery_awarded_once_only() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset(make_square_sector_polygons(size=256.0))

    p0 = make_priv_state(tic=4, x=128.0, y=128.0)
    p1 = make_priv_state(tic=8, x=384.0, y=128.0)  # enters sector 1
    p2 = make_priv_state(tic=12, x=400.0, y=128.0)  # stays in sector 1

    r_enter = shaper.step(p0, p1, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    r_stay = shaper.step(p1, p2, np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    assert r_enter > 0.05
    assert r_stay == pytest.approx(-0.002, abs=1e-4)


def test_t1_f4_exit_bonus_and_speed_efficiency() -> None:
    rewards_mod = import_training("rewards")
    noop = np.zeros(9, dtype=np.int8)

    s_fast = rewards_mod.RewardShaper()
    s_fast.reset(make_square_sector_polygons())
    r_fast = s_fast.step(
        make_priv_state(tic=100, x=64.0, y=64.0),
        make_priv_state(tic=104, x=64.0, y=64.0, exit_reached=True, finished=True),
        noop,
        {"exit_reached": True},
    )

    s_slow = rewards_mod.RewardShaper()
    s_slow.reset(make_square_sector_polygons())
    r_slow = s_slow.step(
        make_priv_state(tic=1900, x=64.0, y=64.0),
        make_priv_state(tic=1904, x=64.0, y=64.0, exit_reached=True, finished=True),
        noop,
        {"exit_reached": True},
    )
    assert r_fast >= 15.0
    assert r_slow >= 15.0
    assert r_fast > r_slow


def test_t1_f4_key_door_and_locked_door_deltas() -> None:
    rewards_mod = import_training("rewards")
    noop = np.zeros(9, dtype=np.int8)

    # Keycard pickup
    s1 = rewards_mod.RewardShaper()
    s1.reset(make_square_sector_polygons())
    r_key = s1.step(
        make_priv_state(x=64.0, y=64.0, keys=(0, 0, 0, 0, 0, 0)),
        make_priv_state(x=64.0, y=64.0, keys=(1, 0, 0, 0, 0, 0)),
        noop,
    )
    assert r_key >= 3.0

    # Normal door open (ceiling rises from 0 to 64, door_lock == 0)
    s2 = rewards_mod.RewardShaper()
    s2.reset(make_square_sector_polygons())
    r_door = s2.step(
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 0.0], door_lock=[0, 0]),
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 64.0], door_lock=[0, 0]),
        noop,
    )
    assert r_door > 0.3

    # Locked door unlock (ceiling rises from 0 to 64, door_lock > 0)
    s3 = rewards_mod.RewardShaper()
    s3.reset(make_square_sector_polygons())
    r_lock = s3.step(
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 0.0], door_lock=[0, 129]),
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 64.0], door_lock=[0, 129]),
        noop,
    )
    assert r_lock > r_door
    assert r_lock >= 4.0


def test_t1_f4_combat_pickup_damage_and_death_deltas() -> None:
    rewards_mod = import_training("rewards")
    noop = np.zeros(9, dtype=np.int8)

    shaper = rewards_mod.RewardShaper()
    shaper.reset(make_square_sector_polygons())
    r_kill = shaper.step(make_priv_state(x=64.0, y=64.0, kills=0), make_priv_state(x=64.0, y=64.0, kills=1), noop)
    r_item = shaper.step(make_priv_state(x=64.0, y=64.0, items=0), make_priv_state(x=64.0, y=64.0, items=1), noop)
    r_dmg = shaper.step(
        make_priv_state(x=64.0, y=64.0, damage_taken=0.0),
        make_priv_state(x=64.0, y=64.0, damage_taken=25.0),
        noop,
    )
    r_death = shaper.step(
        make_priv_state(x=64.0, y=64.0, dead=False),
        make_priv_state(x=64.0, y=64.0, dead=True, finished=True),
        noop,
    )

    assert r_kill == pytest.approx(1.0 - 0.002, abs=0.05)
    assert r_item == pytest.approx(0.15 - 0.002, abs=0.05)
    assert r_dmg < -0.2
    assert r_death <= -3.5


# ===========================================================================
# F5: Extreme Domain-Randomized Procedural MapGen (5 tests)
# ===========================================================================


def test_t1_f5_tier1_generates_loadable_wad_under_2048_sectors(make_live_env) -> None:
    wad_path, map_name = _sample_map(tier=1, seed=100001, scaffolding_alpha=0.0)
    assert Path(wad_path).is_file()
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    assert len(env.sector_polygons()) < 2048
    obs, _, _, _, _ = env.step(np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8))
    assert env.observation_space.contains(obs)


def test_t1_f5_tier2_contains_normal_doors(make_live_env) -> None:
    wad_path, map_name = _sample_map(tier=2, seed=100002, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert len(env.sector_polygons()) < 2048
    assert len(env.mapinfo.door_sectors) >= 1
    assert any(lock == 0 for lock in env.mapinfo.door_sectors.values())


def test_t1_f5_tier3_contains_keycard_and_locked_door(make_live_env) -> None:
    wad_path, map_name = _sample_map(tier=3, seed=100003, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert len(env.sector_polygons()) < 2048
    assert any(lock > 0 for lock in env.mapinfo.door_sectors.values())


def test_t1_f5_tier4_gauntlet_loadable_and_multi_sector(make_live_env) -> None:
    wad_path, map_name = _sample_map(tier=4, seed=100004, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=400, render_buffers=False)
    obs, _ = env.reset(seed=0)
    polys = env.sector_polygons()
    assert 3 <= len(polys) < 2048
    for _ in range(5):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break


def test_t1_f5_extreme_knobs_deterministic_by_seed(make_live_env) -> None:
    knobs_small = {"num_rooms": 2, "enemies_per_room": 1, "ammo_mult": 0.5}
    knobs_large = {"num_rooms": 6, "enemies_per_room": 4, "ammo_mult": 1.5}

    w1, m1 = _sample_map(tier=2, knobs=knobs_small, seed=100055, scaffolding_alpha=0.0)
    w2, m2 = _sample_map(tier=2, knobs=knobs_small, seed=100055, scaffolding_alpha=0.0)
    w3, m3 = _sample_map(tier=2, knobs=knobs_large, seed=100056, scaffolding_alpha=0.0)

    e1 = make_live_env(w1, map_name=m1, episode_ticks=100, render_buffers=False)
    e2 = make_live_env(w2, map_name=m2, episode_ticks=100, render_buffers=False)
    e3 = make_live_env(w3, map_name=m3, episode_ticks=100, render_buffers=False)
    e1.reset(seed=0)
    e2.reset(seed=0)
    e3.reset(seed=0)

    assert len(e1.sector_polygons()) == len(e2.sector_polygons())
    assert len(e3.sector_polygons()) >= len(e1.sector_polygons())


# ===========================================================================
# F6: Self-Evolving PLRBuffer & Scaffolding Schedule (5 tests)
# ===========================================================================


def _make_plr_entry(seed: int, score_gae: float, tier: int = 2, last_seen_update: int = 1, visits: int = 1):
    return {
        "level_id": f"proc_t{tier}_s{seed}",
        "tier": tier,
        "knobs": {"num_rooms": 4, "num_keys": 1, "enemies_per_room": 2, "ammo_mult": 1.0},
        "seed": seed,
        "score_gae": score_gae,
        "last_seen_update": last_seen_update,
        "visits": visits,
    }


def _plr_add(buf: Any, entry: dict) -> None:
    for name in ("add", "insert", "register", "push", "update"):
        fn = getattr(buf, name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            if len(sig.parameters) == 1:
                fn(entry)
                return
            try:
                fn(**entry)
                return
            except TypeError:
                pass
    pytest.fail("PLRBuffer missing add/insert/register/update method")


def _plr_sample(buf: Any, update_idx: int = 10) -> dict:
    fn = getattr(buf, "sample", None)
    assert callable(fn), "PLRBuffer missing sample()"
    sig = inspect.signature(fn)
    if "update_idx" in sig.parameters:
        res = fn(update_idx=update_idx)
    elif len(sig.parameters) >= 1 and next(iter(sig.parameters.values())).default is inspect.Parameter.empty:
        res = fn(update_idx)
    else:
        res = fn()
    if isinstance(res, dict):
        return res
    return vars(res)


def test_t1_f6_plr_buffer_add_and_entry_schema() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    for i in range(10):
        _plr_add(buf, _make_plr_entry(seed=100000 + i, score_gae=0.1 * (i + 1)))

    sampled = _plr_sample(buf, update_idx=5)
    for key in ("tier", "knobs", "seed"):
        assert key in sampled, f"Sampled PLR entry missing {key}: {sampled}"
    assert 100_000 <= int(sampled["seed"]) <= 999_999


def test_t1_f6_plr_prioritizes_higher_gae_levels() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    _plr_add(buf, _make_plr_entry(seed=100001, score_gae=0.01, last_seen_update=99, visits=10))
    _plr_add(buf, _make_plr_entry(seed=100002, score_gae=5.00, last_seen_update=1, visits=1))

    counts = Counter(_plr_sample(buf, update_idx=100)["seed"] for _ in range(200))
    assert counts[100002] > counts[100001]


def test_t1_f6_scaffolding_alpha_monotonic_decay() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256, scaffolding_end_step=12_000_000)

    fn = getattr(buf, "scaffolding_alpha", None) or getattr(buf, "get_scaffolding_alpha", None)
    if fn is None:
        fn = getattr(plr_mod, "scaffolding_alpha", None)
    assert callable(fn), "Missing scaffolding_alpha schedule on PLRBuffer/curriculum_plr"

    a0 = float(fn(0))
    a_mid = float(fn(6_000_000))
    a_end = float(fn(12_000_000))
    assert a0 == pytest.approx(1.0, abs=1e-6)
    assert 0.0 < a_mid < a0
    assert a_end == pytest.approx(0.0, abs=1e-6)


def test_t1_f6_plr_applies_bounded_knob_updates() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    updater = getattr(buf, "apply_knobs", None) or getattr(buf, "set_knobs", None) or getattr(buf, "update_knobs", None)
    assert callable(updater), "PLRBuffer missing apply_knobs/set_knobs/update_knobs"

    new_knobs = {"num_rooms": 6, "enemies_per_room": 4, "ammo_mult": 0.75}
    updater(new_knobs)
    sampled = _plr_sample(buf, update_idx=1)
    assert sampled["knobs"]["num_rooms"] == 6
    assert sampled["knobs"]["enemies_per_room"] == 4
    assert float(sampled["knobs"]["ammo_mult"]) == pytest.approx(0.75)


def test_t1_f6_plr_serialization_roundtrip() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    _plr_add(buf, _make_plr_entry(seed=100010, score_gae=1.25))
    _plr_add(buf, _make_plr_entry(seed=100020, score_gae=0.50))

    if hasattr(buf, "state_dict") and hasattr(buf, "load_state_dict"):
        state = buf.state_dict()
        restored = plr_mod.PLRBuffer(capacity=256)
        restored.load_state_dict(state)
    elif hasattr(buf, "to_dict") and hasattr(plr_mod.PLRBuffer, "from_dict"):
        state = buf.to_dict()
        restored = plr_mod.PLRBuffer.from_dict(state)
    else:
        pytest.fail("PLRBuffer missing state_dict/load_state_dict or to_dict/from_dict")

    assert len(restored) == 2 if hasattr(restored, "__len__") else True
    s = _plr_sample(restored, update_idx=10)
    assert s["seed"] in {100010, 100020}
