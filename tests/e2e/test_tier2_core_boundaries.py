"""Tier 2 Opaque-Box E2E Tests: Core Boundary, Corner & Negative Cases (F1-F6, 30 tests).

All `training.*` imports are strictly lazy inside test bodies.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.e2e.conftest import (
    CONFIGS_DIR,
    import_training,
    load_yaml,
    make_dummy_model_inputs,
    make_priv_state,
    make_square_sector_polygons,
    make_synthetic_obs,
)

pytestmark = pytest.mark.e2e


def _validate_cfg(kind: str, cfg: dict[str, Any]) -> None:
    configs_mod = import_training("configs")
    for name in (f"validate_{kind}_config", "validate_config"):
        fn = getattr(configs_mod, name, None)
        if callable(fn):
            sig = inspect.signature(fn)
            if len(sig.parameters) >= 2 and name == "validate_config":
                fn(cfg, kind)
            else:
                fn(cfg)
            return
    pytest.fail(f"training.configs must expose validate_{kind}_config or validate_config")


def _sample_map(tier: int, knobs: dict | None = None, seed: int = 100001, scaffolding_alpha: float = 0.0):
    tiers_mod = import_training("mapgen.tiers")
    MapGen = tiers_mod.MapGen
    sig = inspect.signature(MapGen.sample)
    if "self" in sig.parameters:
        return MapGen().sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)
    return MapGen.sample(tier=tier, knobs=knobs, seed=seed, scaffolding_alpha=scaffolding_alpha)


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


def _plr_sample(buf: Any, update_idx: int = 1) -> dict:
    fn = getattr(buf, "sample", None)
    assert callable(fn)
    sig = inspect.signature(fn)
    if "update_idx" in sig.parameters:
        res = fn(update_idx=update_idx)
    elif len(sig.parameters) >= 1 and next(iter(sig.parameters.values())).default is inspect.Parameter.empty:
        res = fn(update_idx)
    else:
        res = fn()
    return res if isinstance(res, dict) else vars(res)


# ===========================================================================
# F1: YAML Config Boundaries (5 tests)
# ===========================================================================


def test_t2_f1_invalid_num_envs_and_bptt_len_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "ppo_curriculum.yaml")
    for key, bad_val in (("num_envs", 0), ("bptt_len", -1)):
        bad = dict(base)
        bad[key] = bad_val
        with pytest.raises(ValueError):
            _validate_cfg("ppo", bad)


def test_t2_f1_eval_exceeding_pr2_vram_or_threads_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "eval.yaml")
    for key, bad_val in (("max_vram_mb", 4096), ("max_ram_mb", 16384), ("cpu_threads", 4)):
        bad = dict(base)
        bad[key] = bad_val
        with pytest.raises(ValueError):
            _validate_cfg("eval", bad)


def test_t2_f1_missing_yaml_path_raises_file_not_found(tmp_path: Path) -> None:
    configs_mod = import_training("configs")
    missing = tmp_path / "does_not_exist.yaml"
    loader = getattr(configs_mod, "load_config", None) or getattr(configs_mod, "load_ppo_config", None)
    if callable(loader):
        with pytest.raises(FileNotFoundError):
            loader(missing)
    else:
        with pytest.raises(FileNotFoundError):
            load_yaml(missing)


def test_t2_f1_non_divisible_rollout_steps_by_bptt_len_rejected() -> None:
    bad = load_yaml(CONFIGS_DIR / "ppo_curriculum.yaml")
    bad["rollout_steps"] = 130
    bad["bptt_len"] = 32
    with pytest.raises(ValueError):
        _validate_cfg("ppo", bad)


def test_t2_f1_render_buffers_ratio_out_of_bounds_rejected() -> None:
    base = load_yaml(CONFIGS_DIR / "ppo_curriculum.yaml")
    for ratio in (-0.1, 1.5):
        bad = dict(base)
        bad["render_buffers_ratio"] = ratio
        with pytest.raises(ValueError):
            _validate_cfg("ppo", bad)


# ===========================================================================
# F2: Preprocessing & Odometry Boundaries (5 tests)
# ===========================================================================


def test_t2_f2_use_cooldown_masks_use_for_six_steps_near_door() -> None:
    prep = import_training("preprocessing")
    dists = np.full(64, 0.5, dtype=np.float32)
    hits = np.ones(64, dtype=np.int64)
    dists[0] = 0.10  # < 0.15 front door
    hits[0] = 2  # RAY_DOOR
    obs_near = make_synthetic_obs(ray_distances=dists, ray_hit_types=hits)

    # USE=1 when door is far (>= 0.15) should NOT trigger cooldown
    dists_far = dists.copy()
    dists_far[0] = 0.50
    obs_far = make_synthetic_obs(ray_distances=dists_far, ray_hit_types=hits)
    _, st_far = prep.preprocess(obs_far, None)
    st_far.prev_action = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    inp_far, st_far = prep.preprocess(obs_far, st_far)
    assert bool(inp_far.action_mask[0, 7]) is True

    # USE=1 near door (< 0.15) triggers 6-step cooldown masking action_mask[7] == False
    _, st = prep.preprocess(obs_near, None)
    st.prev_action = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.float32)
    for step_idx in range(6):
        inp, st = prep.preprocess(obs_near, st)
        assert bool(inp.action_mask[0, 7]) is False, f"Expected USE masked on cooldown step {step_idx}"
        st.prev_action = np.zeros(9, dtype=np.float32)

    inp_after, st = prep.preprocess(obs_near, st)
    assert bool(inp_after.action_mask[0, 7]) is True


def test_t2_f2_attack_mask_melee_vs_ranged_zero_ammo() -> None:
    prep = import_training("preprocessing")
    for melee_slot in (0.0, 1.0):
        obs_melee = make_synthetic_obs(selected_weapon=melee_slot, selected_weapon_ammo=0.0)
        inp_m, _ = prep.preprocess(obs_melee, None)
        assert bool(inp_m.action_mask[0, 6]) is True

    obs_pistol_empty = make_synthetic_obs(selected_weapon=2.0, selected_weapon_ammo=0.0)
    inp_p, _ = prep.preprocess(obs_pistol_empty, None)
    assert bool(inp_p.action_mask[0, 6]) is False


def test_t2_f2_zero_entities_zero_buffers_and_extreme_velocities() -> None:
    import torch

    prep = import_training("preprocessing")
    obs = make_synthetic_obs(
        render_buffers=False,
        num_entities=0,
        vel=(1e4, -1e4, 1e4),
    )
    inp, state = prep.preprocess(obs, None)
    assert int(inp.ent_mask.sum()) == 0
    assert torch.all(inp.rgb_depth == 0.0)
    assert torch.isfinite(inp.vitals).all()
    assert torch.isfinite(inp.grid).all()
    assert (inp.grid >= 0.0).all() and (inp.grid <= 1.0 + 1e-5).all()


def test_t2_f2_all_none_rays_and_enemy_occlusion_safe() -> None:
    import torch

    prep = import_training("preprocessing")
    obs0 = make_synthetic_obs(
        ray_distances=np.ones(64, dtype=np.float32),
        ray_hit_types=np.zeros(64, dtype=np.int64),
        num_entities=0,
    )
    inp0, st = prep.preprocess(obs0, None)
    assert torch.allclose(inp0.rays[0, 2], torch.ones(64))

    # Next step with moving enemies on all rays (hit_type=4) and TURN_LEFT
    st.prev_action = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    obs1 = make_synthetic_obs(
        ray_distances=np.full(64, 0.3, dtype=np.float32),
        ray_hit_types=np.full(64, 4, dtype=np.int64),
        vel=(5.0, -3.0, 0.0),
    )
    inp1, _ = prep.preprocess(obs1, st)
    assert torch.isfinite(inp1.grid).all()


def test_t2_f2_invalid_obs_shape_or_missing_key_raises() -> None:
    prep = import_training("preprocessing")
    bad_shape_obs = make_synthetic_obs()
    bad_shape_obs["rgb"] = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises((ValueError, RuntimeError)):
        prep.preprocess(bad_shape_obs, None)

    missing_key_obs = make_synthetic_obs()
    del missing_key_obs["rays"]
    with pytest.raises((KeyError, ValueError)):
        prep.preprocess(missing_key_obs, None)


# ===========================================================================
# F3: HybridDoomNet + Value Compass Boundaries (5 tests)
# ===========================================================================


def _cfg(**kw: Any) -> dict[str, Any]:
    d = {"d_model": 256, "gru_dim": 256, "num_slots": 4, "slot_dim": 8, "use_rgb": True, "use_grid": True}
    d.update(kw)
    return d


def test_t2_f3_zero_entities_mixed_batch_no_nan() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_cfg())
    net.eval()
    batch = make_dummy_model_inputs(batch_size=2, num_entities=5)
    batch.ent_mask[0, :] = False  # Sample 0 has zero entities; sample 1 has 5
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, 2, 256, dtype=torch.float32),
        obj_slots=torch.zeros(2, 4, 8, dtype=torch.float32),
    )
    with torch.no_grad():
        logits, value, aux, next_h = net(batch, h, return_aux=True)

    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(aux["value_compass"]).all()
    assert torch.isfinite(aux["next_rays"]).all()
    assert torch.isfinite(next_h.gru_h).all()
    assert torch.isfinite(next_h.obj_slots).all()


def test_t2_f3_full_32_entities_permutation_invariance() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_cfg())
    net.eval()
    b1 = make_dummy_model_inputs(batch_size=1, num_entities=32, seed=99)
    perm = torch.randperm(32)
    b2 = make_dummy_model_inputs(batch_size=1, num_entities=32, seed=99)
    b2.ent_feats = b1.ent_feats[:, perm, :].clone()
    b2.ent_types = b1.ent_types[:, perm].clone()
    b2.ent_mask = b1.ent_mask[:, perm].clone()

    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, 1, 256, dtype=torch.float32),
        obj_slots=torch.zeros(1, 4, 8, dtype=torch.float32),
    )
    with torch.no_grad():
        l1, v1, aux1, _ = net(b1, h, return_aux=True)
        l2, v2, aux2, _ = net(b2, h, return_aux=True)

    torch.testing.assert_close(l1, l2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(v1, v2, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(aux1["value_compass"], aux2["value_compass"], atol=1e-4, rtol=1e-4)


def test_t2_f3_use_rgb_false_and_use_grid_false_fallback() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_cfg(use_rgb=False, use_grid=False))
    B = 2
    batch = make_dummy_model_inputs(batch_size=B, zero_rgb=True, zero_grid=True)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )
    logits, value, aux, next_h = net(batch, h, return_aux=True)
    assert logits.shape == (B, 9) and torch.isfinite(logits).all()
    assert aux["value_compass"].shape == (B, 8) and torch.isfinite(aux["value_compass"]).all()


def test_t2_f3_extreme_magnitude_inputs_stay_finite() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_cfg())
    net.eval()
    B = 2
    batch = make_dummy_model_inputs(batch_size=B, num_entities=4)
    batch.vitals.fill_(1e4)
    batch.ent_feats.fill_(-1e4)
    h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, B, 256, dtype=torch.float32),
        obj_slots=torch.zeros(B, 4, 8, dtype=torch.float32),
    )
    with torch.no_grad():
        logits, value, aux, _ = net(batch, h, return_aux=True)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(value).all()
    assert torch.isfinite(aux["value_compass"]).all()


def test_t2_f3_mismatched_recurrent_batch_size_raises() -> None:
    import torch

    model_mod = import_training("model")
    net = model_mod.HybridDoomNet(_cfg())
    batch = make_dummy_model_inputs(batch_size=2)
    bad_h = model_mod.RecurrentState(
        gru_h=torch.zeros(1, 3, 256, dtype=torch.float32),
        obj_slots=torch.zeros(3, 4, 8, dtype=torch.float32),
    )
    with pytest.raises((RuntimeError, ValueError)):
        net(batch, bad_h, return_aux=False)


# ===========================================================================
# F4: RewardShaper Boundaries (5 tests)
# ===========================================================================


def test_t2_f4_strict_world_translation_invariance() -> None:
    rewards_mod = import_training("rewards")
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    s_a = rewards_mod.RewardShaper()
    s_a.reset(make_square_sector_polygons(origin_x=0.0, origin_y=0.0, size=256.0))
    ra = s_a.step(
        make_priv_state(tic=4, x=128.0, y=128.0),
        make_priv_state(tic=8, x=300.0, y=128.0, kills=1, damage_taken=10.0),
        fwd,
    )

    s_b = rewards_mod.RewardShaper()
    s_b.reset(make_square_sector_polygons(origin_x=5000.0, origin_y=-5000.0, size=256.0))
    rb = s_b.step(
        make_priv_state(tic=4, x=5128.0, y=-4872.0),
        make_priv_state(tic=8, x=5300.0, y=-4872.0, kills=1, damage_taken=10.0),
        fwd,
    )
    assert ra == pytest.approx(rb, abs=1e-6)


def test_t2_f4_door_reversal_bounce_penalty() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset(make_square_sector_polygons())
    use_act = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=np.int8)

    # Step 1: door starts opening (0 -> 20)
    shaper.step(
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 0.0]),
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 20.0]),
        np.zeros(9, dtype=np.int8),
    )
    # Step 2: USE reverses opening door (20 -> 12)
    r_bounce = shaper.step(
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 20.0]),
        make_priv_state(x=64.0, y=64.0, sector_ceiling=[128.0, 12.0]),
        use_act,
    )
    assert r_bounce <= -0.1


def test_t2_f4_stuck_wall_scrape_penalty_after_eight_steps() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset(make_square_sector_polygons())
    fwd = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)

    rewards = []
    for step in range(12):
        p_prev = make_priv_state(tic=4 * step, x=64.0, y=64.0)
        p_now = make_priv_state(tic=4 * (step + 1), x=64.5, y=64.0)
        rewards.append(shaper.step(p_prev, p_now, fwd))

    assert rewards[0] == pytest.approx(-0.002, abs=1e-4)
    assert rewards[-1] < rewards[0] - 0.01


def test_t2_f4_empty_sector_polygons_zero_div_safe() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset([])
    r = shaper.step(
        make_priv_state(x=64.0, y=64.0),
        make_priv_state(x=128.0, y=64.0),
        np.zeros(9, dtype=np.int8),
    )
    assert np.isfinite(r)


def test_t2_f4_none_polygons_and_simultaneous_death_exit_finite() -> None:
    rewards_mod = import_training("rewards")
    shaper = rewards_mod.RewardShaper()
    shaper.reset(None)
    r = shaper.step(
        make_priv_state(x=64.0, y=64.0),
        make_priv_state(x=64.0, y=64.0, dead=True, exit_reached=True, finished=True),
        np.zeros(9, dtype=np.int8),
        {"exit_reached": True},
    )
    assert isinstance(r, float) and np.isfinite(r)


# ===========================================================================
# F5: MapGen Extreme Domain Randomization Boundaries (5 tests)
# ===========================================================================


def test_t2_f5_extreme_claustrophobic_ambush_layout(make_live_env) -> None:
    knobs = {"num_rooms": 1, "enemies_per_room": 4, "ammo_mult": 0.8}
    wad_path, map_name = _sample_map(tier=1, knobs=knobs, seed=100201, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert len(env.sector_polygons()) < 2048
    assert env.observation_space.contains(obs)


def test_t2_f5_extreme_scarce_ammo_dense_gauntlet(make_live_env) -> None:
    knobs = {"num_rooms": 3, "enemies_per_room": 6, "ammo_mult": 0.25}
    wad_path, map_name = _sample_map(tier=4, knobs=knobs, seed=100202, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    obs, _ = env.reset(seed=0)
    assert len(env.sector_polygons()) < 2048
    assert env.observation_space.contains(obs)


def test_t2_f5_multi_branch_keycard_maze(make_live_env) -> None:
    knobs = {"num_rooms": 6, "num_keys": 1, "enemies_per_room": 2, "ammo_mult": 1.0}
    wad_path, map_name = _sample_map(tier=3, knobs=knobs, seed=100203, scaffolding_alpha=0.0)
    env = make_live_env(wad_path, map_name=map_name, episode_ticks=200, render_buffers=False)
    env.reset(seed=0)
    assert len(env.sector_polygons()) < 2048
    assert len(env.mapinfo.door_sectors) >= 1


def test_t2_f5_invalid_tier_raises_value_error() -> None:
    for bad_tier in (0, 5):
        with pytest.raises(ValueError):
            _sample_map(tier=bad_tier, seed=100204, scaffolding_alpha=0.0)


def test_t2_f5_scaffolding_alpha_zero_keeps_vanilla_locked_doors(make_live_env) -> None:
    w_van, m_van = _sample_map(tier=3, seed=100777, scaffolding_alpha=0.0)
    w_scaf, m_scaf = _sample_map(tier=3, seed=100777, scaffolding_alpha=1.0)

    e_van = make_live_env(w_van, map_name=m_van, episode_ticks=100, render_buffers=False)
    e_scaf = make_live_env(w_scaf, map_name=m_scaf, episode_ticks=100, render_buffers=False)
    e_van.reset(seed=0)
    e_scaf.reset(seed=0)
    assert any(lock > 0 for lock in e_van.mapinfo.door_sectors.values())
    assert len(e_scaf.sector_polygons()) < 2048


# ===========================================================================
# F6: PLRBuffer Boundaries (5 tests)
# ===========================================================================


def test_t2_f6_sample_from_empty_buffer_falls_back_safely() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    sampled = _plr_sample(buf, update_idx=0)
    assert 1 <= int(sampled["tier"]) <= 4
    assert 100_000 <= int(sampled["seed"]) <= 999_999


def test_t2_f6_capacity_overflow_evicts_to_256() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    for i in range(300):
        _plr_add(
            buf,
            {
                "level_id": f"proc_t2_s{100000 + i}",
                "tier": 2,
                "knobs": {"num_rooms": 4, "num_keys": 1, "enemies_per_room": 2, "ammo_mult": 1.0},
                "seed": 100000 + i,
                "score_gae": float(i % 10),
                "last_seen_update": i,
                "visits": 1,
            },
        )
    if hasattr(buf, "__len__"):
        assert len(buf) <= 256
    elif hasattr(buf, "entries"):
        assert len(buf.entries) <= 256


def test_t2_f6_rejects_heldout_seed_range() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    with pytest.raises(ValueError):
        _plr_add(
            buf,
            {
                "level_id": "heldout_leak",
                "tier": 2,
                "knobs": {"num_rooms": 4, "num_keys": 1, "enemies_per_room": 2, "ammo_mult": 1.0},
                "seed": 1500,
                "score_gae": 1.0,
                "last_seen_update": 1,
                "visits": 1,
            },
        )


def test_t2_f6_nan_and_inf_score_gae_guarded() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256)
    _plr_add(
        buf,
        {
            "level_id": "proc_t1_s100001",
            "tier": 1,
            "knobs": {"num_rooms": 2, "num_keys": 0, "enemies_per_room": 1, "ammo_mult": 1.0},
            "seed": 100001,
            "score_gae": float("nan"),
            "last_seen_update": 1,
            "visits": 1,
        },
    )
    _plr_add(
        buf,
        {
            "level_id": "proc_t1_s100002",
            "tier": 1,
            "knobs": {"num_rooms": 2, "num_keys": 0, "enemies_per_room": 1, "ammo_mult": 1.0},
            "seed": 100002,
            "score_gae": float("inf"),
            "last_seen_update": 2,
            "visits": 1,
        },
    )
    sampled = _plr_sample(buf, update_idx=3)
    assert sampled["seed"] in {100001, 100002}


def test_t2_f6_scaffolding_alpha_clamps_beyond_end_step() -> None:
    plr_mod = import_training("curriculum_plr")
    buf = plr_mod.PLRBuffer(capacity=256, scaffolding_end_step=12_000_000)
    fn = getattr(buf, "scaffolding_alpha", None) or getattr(buf, "get_scaffolding_alpha", None)
    assert callable(fn)
    assert float(fn(20_000_000)) == pytest.approx(0.0, abs=1e-6)
    assert float(fn(0)) == pytest.approx(1.0, abs=1e-6)
