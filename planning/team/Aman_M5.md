# Aman (MacBook Air M5) — Step-by-Step Runbook

## 1. Role & Ownership Summary (3-Person Team)
* **Role:** Extreme Domain-Randomized Procedural Map Generator, Coordinate-Free Dense Reward Engineer, Self-Evolving PLR Curriculum Architect & Narrative Lead (`README.md` + 2-Page Brief PDF) (**Free Tier — Ration Pro Turns Carefully!**).
* **Files You Own:**
  * `training/mapgen/udmf_builder.py` (Procedural PWAD / UDMF writer + `zdbsp` node compiler wrapper)
  * `training/mapgen/tiers.py` (`MapGen.sample(tier, knobs=None, seed=0, scaffolding_alpha=0.0) -> tuple[Path, str]` with **wide domain randomization & extreme scenarios**: tight claustrophobic ambushes, large open arenas, multi-branch corridor mazes, scarce ammo survival, dense monster waves, randomized geometry scales + internal BFS solvability verifier)
  * `training/rewards.py` (Coordinate-free dense training-only reward shaper using relative `env.privileged_state()` deltas & `env.sector_polygons()` — zero fixed XYZ assumptions)
  * `training/curriculum_plr.py` (Self-evolving Prioritized Level Replay buffer + scaffolding annealing schedule)
  * `training/scorer/eval_suite.json` & `maps/heldout/*.wad` (Locked disjoint extreme random-map benchmark maps for Aryan's LOQ judge)
  * Root `README.md`, `training/README.md`, and **2-Page Project Brief PDF**
  * `tests/test_mapgen_and_rewards.py`
* **Interfaces You Provide & Consume (`Schema.md`):**
  * **Provide to Vedesh:** `MapGen.sample(...)`, `RewardShaper.reset(sector_polygons)` + `step(priv_prev, priv_now, action, info=None)`, and `PLRBuffer` plugged into `envs.py` & self-evolving `ppo.py`.
  * **Provide to Aryan:** `eval_suite.json` + pre-generated `maps/heldout/` extreme random WADs (`seed ∈ [1000, 9999]`, strictly disjoint from training seeds `≥ 100000`) + curated community/Freedoom benchmark list.
  * **Consume from Aryan & Vedesh:** Held-out scorecards, ablation tables, and replay screenshots (`render_replay.py`) for the 2-Page Project Brief PDF.

---

## 2. Phase 0: Pre-Event Machine Setup (Complete Before H0 — Zero Project Code!)

Run on **MacBook Air M5 (macOS Terminal)**:
```bash
# 1. Install Homebrew dependencies, Git LFS & Tailscale
brew install cmake sdl2 boost git git-lfs tailscale
git lfs install

# 2. Install uv + Python 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.13

# 3. Check WAD node-builder availability on macOS (or compile zdbsp once from source into ~/.local/bin)
# Note: also verify zdbsp / bsp works over SSH on Vedesh's OMEN WSL2!
which zdbsp || which bsp

# 4. Smoke-test vizdoom==1.3.3 & doom_env on Apple Silicon (delete /tmp/setup-check before H0!)
git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
cd /tmp/setup-check
uv sync
uv run python validate.py
cd ~ && rm -rf /tmp/setup-check

# 5. Verify Tailscale SSH to Vedesh's OMEN & Aryan's LOQ
tailscale status
ssh <vedesh-user>@<omen-tailscale-ip> "which zdbsp bsp"
```
* **AI Quota Discipline (Free Account):** Save **Gemini Pro (High)** strictly for designing orthogonal room-graph generation without overlapping linedefs (`tiers.py`) and PLR math (`curriculum_plr.py`). Use **Gemini Flash** for WAD binary/UDMF formatting, `rewards.py`, map manifests, and `/browser` WAD lookups!

---

## 3. Phase-by-Phase Execution Checklist

### Phase 1 (H0–H2): WAD Builder & Tier 1–2 Procedural Generator v0
* **AI Model & Command:** `Gemini Pro (High)` + `/plan` for room-grid layout design, then **`Gemini Flash`** for WAD writer implementation.
1. Clone team repo, run `uv sync` + `uv add --dev omgifol pyyaml`, checkout `feat/aman-mapgen-v0`.
2. **Design Choice That Saves Hours — Grid-Cell Room Graph:**
   * Carve rooms and corridors on a coarse integer macro-grid (e.g., `8×8` macro-tiles where each tile is `256×256` or `384×384` Doom units). Orthogonal non-overlapping grid rectangles **guarantee zero self-intersecting linedefs** and `< 150` sectors (`<< 2048` limit)!
   * Emit UDMF `TEXTMAP` (or classic Doom `THINGS/LINEDEFS/SIDEDEFS/VERTEXES/SECTORS` via `omgifol`) and run `zdbsp -o out.wad in.wad` so ViZDoom loads sectors and doors cleanly.
   * **Verify Door Linedef Specials:** Confirm `doom_env/mapinfo.py` recognizes your door linedef special (`1` DR Door Open Wait Close) as `hit_type == 2 (door)` and locked-door specials (`26/27/28` Blue/Yellow/Red DR Door) as `hit_type == 3 (locked_door)`! *(Test this directly by inspecting `obs["rays"]["hit_type"]` in `make_env`!)*
3. Implement Tier 1 (1–2 open arena rooms with pillars, enemies, health/ammo, exit switch/pad) and Tier 2 (3–5 connected rooms with regular doors and dead-end branches).
* **Done Criteria (H2 Gate):** `make_env` loads generated Tier 1 & Tier 2 WADs with zero warnings, rays report `door` (`2`) accurately, and walking across the exit linedef sets `info["exit_reached"] == True`.

### Phase 2 (H2–H4): Dense Privileged Reward Shaper & Held-Out Eval Suite v0
* **AI Model & Command:** **`Gemini Flash`** + `/plan`.
1. Checkout `feat/aman-rewards`. Implement `training/rewards.py` matching `Design.md §3`:
   * Track visited sector IDs per episode via point-in-polygon / sector lookup on `env.sector_polygons()` and `privileged_state()["x", "y"]`.
   * Reward exit (`+15 + 3 * time_frac`), new sector (`+0.4`), key pickup (`+4.0`), door open (`+0.6`), locked door unlock (`+5.0`), kill (`+1.0`), pickup (`+0.15`), minus damage (`-0.02/HP`), death (`-4.0`), door-reversal bounce (`-0.3`), stuck wall-scrape (`-0.02`), step cost (`-0.002`).
2. Generate `24` fixed held-out WADs (`seed ∈ [1001..1024]`) + add selected `freedoom2.wad` maps (`MAP01–MAP03`) into `training/scorer/eval_suite.json` and push for Aryan's scorer.
* **Done Criteria (H4 Gate):** `rewards.py` unit-tested on `two_rooms.wad`; Aryan's `score_checkpoint.py` runs end-to-end on `eval_suite.json`.

### Phase 3 (H4–H6): Tier 3 Keycard Puzzles, Tier 4 Gauntlets, PLR & Demo Support
* **AI Model & Command:** `Gemini Pro (High)` + `/plan` for Tier 3 DAG + PLR, then **`Gemini Flash` + `/goal`** for 500-map validation sweep.
1. Checkout `feat/aman-tier34-plr`. Add **Tier 3** & **Tier 4** to `training/mapgen/tiers.py`:
   * **Tier 3 (Keycard Backtracking):** Spawn room connects to (Branch A: unlocked rooms leading to Blue/Yellow/Red Keycard guarded by 1–2 monsters) and (Branch B: visible Locked Door requiring that exact keycard, leading to Exit room). Assert graph solvability via internal generator check before writing WAD!
   * **Tier 4 (Full Gauntlet):** 5–8 rooms, 1–2 sequential key locks, ambush monster closets/corners, tighter health/ammo placement (`episode_ticks = 4200`).
2. Run `/goal` with `Gemini Flash`: generate 500 random maps across Tiers 1–4, load each in `make_env`, run 10 random steps, and fix any geometry edge cases until `500/500` pass with 0 warnings.
3. Implement `training/curriculum_plr.py`:
   * Maintain buffer of `256` active seeds across Tiers 1–4 scored by mean `|GAE|` + staleness bonus.
   * Implement scaffolding knob `scaffolding_alpha` (`1.0 → 0.0` over `12M` steps): at high `alpha`, fraction `alpha * 0.5` of Tier 2/3 doors spawn pre-opened or unlocked with extra ammo packs so early PPO discovers exits, decaying smoothly to `0.0` (100% vanilla).
4. **H5–H5.5:** Stress-test extreme domain randomization (tight ambushes, wide open arenas, multi-branch mazes, scarce ammo, dense waves, randomized wall/ceiling heights & textures) so the self-evolving policy generalizes zero-shot without human demos.
* **Done Criteria (H6 Gate):** 500/500 extreme random generated maps load & raycast `locked_door (3)` properly; `curriculum_plr.py` merged for Vedesh's Long Run #1.

### Phase 4 & 5 (H6–H12): Out-of-Distribution WAD Curation & Reward Audit
* **AI Model & Command:** `Gemini Flash` + `/browser`.
1. Use `/browser` to locate 5–10 small, permissive vanilla/Freedoom-compatible community WADs (`< 1500` sectors, single-player) and add them to **held-out evaluation** (`eval_suite.json` v2) so we measure true generalization beyond our own generator!
2. Add visual & geometric variety to `tiers.py` (varied Freedoom wall/floor textures, L-shaped hallways, safe height steps `≤ 16` units, light level randomization `112–224`) so Aryan's RGB branch never overfits one wall texture.
3. Audit Vedesh's Run #1 W&B reward curves (`rollout/sector_coverage`, `rollout/doors_unlocked`) to spot any reward hacking (e.g., farming door opens or suiciding) and adjust coefficients if needed.
* **Done Criteria (H12 Gate):** Expanded `eval_suite.json` (40 diverse maps) locked; zero reward-hacking pathologies.

### Phase 6, 7 & 8 (H12–H24): Tier 3/4 Replay Diagnosis, `README.md`, `training/README.md` & 2-Page Brief PDF
* **AI Model & Command:** `Gemini Flash` (Pro for Brief PDF polish + optional `training/llm_coach.py` scorecard summarizer).
1. **H12–H16:** Watch replays from Aryan's `render_replay.py` on held-out extreme Tier 3 & Tier 4 failures; make final pre-freeze tweaks to PLR tier weights before **H16 Feature Freeze** and draft Page 1–2 of the **2-Page Project Brief PDF**.
2. **H16–H20:** Write clear, thorough `README.md` & `training/README.md` explaining the extreme domain-randomized generator, ray-special verification, coordinate-free reward shaping formulas, self-evolving PLR curriculum, and exact CLI commands; populate final held-out scorecard & ablation tables into the 2-Page Brief PDF.
3. **H20–H23.5:** Ensure `uv run pytest` passes 100%, export final **2-Page Project Brief PDF (`≤ 2 pages`)**, and lead the 5-min Desk-Review Q&A rehearsal!

---

## 4. Tests You Must Write (`tests/test_mapgen_and_rewards.py`)
* `test_ray_recognizes_door_and_locked_door()`: Spawn player facing generated normal door and generated locked door → assert `obs["rays"]["hit_type"][0] == 2` and `== 3` respectively! *(Critical contract check!)*
* `test_exit_linedef_terminates_episode()`: Teleport/walk straight into generated exit switch/pad → assert `terminated == True` and `info["exit_reached"] == True`.
* `test_train_heldout_seed_disjoint()`: Assert zero intersection between `eval_suite.json` seeds/WAD hashes and `PLRBuffer` allowed seed range.

---

## 5. If Things Go Wrong (Aman's Fallbacks)
* **`zdbsp` unavailable or quirky on macOS:** Generate classic Doom binary lumps with `omgifol` and run `bsp`/`zdbsp` remotely on Vedesh's OMEN WSL2 (`envs.py` generates maps directly on Linux anyway!), or pre-compile a bank of `2,000` WADs on OMEN in 90 seconds.
* **Custom locked doors don't register as `hit_type == 3`:** Inspect `env/doom_env/mapinfo.py` line-by-line (allowed to read!) and emit the exact linedef special integer / lock number `mapinfo.py` parses in `tests/fixtures/*.wad`.
* **Procedural generator blocks Vedesh at H2:** Start Vedesh on `maps/two_rooms.wad` + `tests/fixtures/*.wad` + `freedoom2.wad` (`MAP01–MAP10`) immediately while finishing `tiers.py`.
