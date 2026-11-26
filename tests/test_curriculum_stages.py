from training.curriculum_stages import (
    DEFAULT_STAGES,
    FIXED_MAP_SEED,
    TRAIN_SEED_MAX,
    TRAIN_SEED_MIN,
    StageCurriculum,
)


def test_stage0_is_one_fixed_map_without_enemies():
    cur = StageCurriculum()
    levels = [cur.sample() for _ in range(20)]
    assert {lv["seed"] for lv in levels} == {FIXED_MAP_SEED}
    assert all(lv["tier"] == 1 and lv["knobs"]["enemies_per_room"] == 0.0 for lv in levels)


def test_advances_only_after_mastery_and_walks_the_ladder():
    cur = StageCurriculum(min_episodes=10, window=10, min_steps_per_stage=0, max_steps_per_stage=10**12)
    seen = [cur.spec.name]
    step = 0
    for _ in range(len(DEFAULT_STAGES) * 3):
        for _ in range(10):
            cur.report_episode(True)
        step += 1000
        cur.maybe_advance(step)
        if cur.spec.name != seen[-1]:
            seen.append(cur.spec.name)
    assert seen == [s.name for s in DEFAULT_STAGES]


def test_does_not_advance_when_failing_until_budget_runs_out():
    cur = StageCurriculum(min_episodes=5, window=5, min_steps_per_stage=0, max_steps_per_stage=5000)
    for _ in range(20):
        cur.report_episode(False)
    assert cur.maybe_advance(1000) is None and cur.stage_idx == 0
    assert cur.maybe_advance(6000) is not None and cur.stage_idx == 1


def test_stage_lock_never_advances_and_seeds_stay_out_of_heldout_range():
    cur = StageCurriculum(stage_lock=2, min_episodes=1, window=2, min_steps_per_stage=0)
    for _ in range(5):
        cur.report_episode(True)
    assert cur.maybe_advance(10**9) is None and cur.stage_idx == 2
    for _ in range(50):
        assert TRAIN_SEED_MIN <= cur.sample()["seed"] <= TRAIN_SEED_MAX


def test_stage_max_caps_progression_and_saves_rich_state(tmp_path):
    state_file = tmp_path / "stage_state.json"
    cur = StageCurriculum(
        stage_idx=1,
        stage_max=2,
        min_episodes=2,
        window=4,
        min_steps_per_stage=0,
        max_steps_per_stage=10**9,
        state_path=state_file,
    )
    for _ in range(4):
        lv = cur.sample()
        cur.report_episode(True, level_id=lv["level_id"], score=10.0)
    assert cur.maybe_advance(1000) is not None and cur.stage_idx == 2
    for _ in range(4):
        lv = cur.sample()
        cur.report_episode(True, level_id=lv["level_id"], score=10.0)
    assert cur.maybe_advance(2000) is None and cur.stage_idx == 2
    import json
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["stage_idx"] == 2 and saved["stage"] == "medium" and "tiers" in saved


def test_weight_average_state_dicts_combines_hybrid_nets():
    import torch
    from training.model import HybridDoomNet
    from training.combine_specialists import weight_average_state_dicts

    m1 = HybridDoomNet({"d_model": 64, "gru_dim": 64, "num_slots": 2, "slot_dim": 4, "use_rgb": False, "use_grid": False})
    m2 = HybridDoomNet({"d_model": 64, "gru_dim": 64, "num_slots": 2, "slot_dim": 4, "use_rgb": False, "use_grid": False})
    with torch.no_grad():
        for p in m1.parameters():
            p.fill_(1.0)
        for p in m2.parameters():
            p.fill_(3.0)
    merged = weight_average_state_dicts([m1.state_dict(), m2.state_dict()], [100.0, 100.0], temperature=50.0)
    first_key = next(iter(merged))
    assert torch.allclose(merged[first_key], torch.full_like(merged[first_key], 2.0), atol=1e-5)


def test_resumed_stage_respects_stage_start_step():
    cur = StageCurriculum(
        stage_idx=2,
        stage_max=4,
        min_episodes=32,
        window=40,
        min_steps_per_stage=16_384,
        max_steps_per_stage=400_000,
        stage_start_step=931_328,
    )
    for _ in range(4):
        cur.report_episode(True)
    assert cur.maybe_advance(932_352) is None and cur.stage_idx == 2
    assert cur.maybe_advance(931_328 + 400_000) is not None and cur.stage_idx == 3
