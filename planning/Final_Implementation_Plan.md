# Master Phase-Wise Implementation Plan (24 Hours — 3-Person Team)

---

## Phase 0: Environment Readiness (H0 Active — Template `63b90c4` Synced)
* **Vedesh (OMEN):** WSL2 Ubuntu 24.04 + CUDA PyTorch + `uv` + Python 3.13 + Docker Desktop (`devforge-doom` image built) + `zdbsp` + Tailscale + W&B team workspace + gitignored `.env` for optional `llm_coach.py`.
* **Aryan (LOQ):** Identical WSL2 Ubuntu 24.04 + CUDA PyTorch + `uv` + Docker Desktop (`devforge-doom` cached) + Tailscale + W&B login.
* **Aman (M5):** macOS `uv` + Python 3.13 + `vizdoom==1.3.3` + `zdbsp` verified + Tailscale SSH to OMEN tested.
* **Gate 0:** Baseline `uv run pytest` and `uv run python validate.py` pass with exit code `0` on the synced repo (`origin/main`).

---

## Phase 1: H0–H2 — Repo Sync, Schema Lock & Runnable `v0` Interfaces
* **Vedesh (OMEN):**
  1. Sync `Reinforce-SST/devforge-doom-template` (`63b90c4`) into root & `template_reference/` on `origin/main` (`https://github.com/aryanbhojgaria/canitdoom.git`), remove the 4th-member roster doc & stray `notepad main.py`, add `.env` to `.gitignore` + `.env.example`, and refresh all planning/schema docs.
  2. Create `training/configs/{ppo_curriculum.yaml,bc.yaml,eval.yaml}` & `training/envs.py` (multiprocess `spawn` `doom_env` workers, episode auto-reset, map-reload hook, extreme domain-randomization & scaffolding hooks).
* **Aryan (LOQ):**
  1. Implement `training/preprocessing.py` v0 (coordinate-free rays, entities, `(B, 36)` vitals, `64×48` RGB+depth supporting `(120,160)` and `(48,64)`, legal `USE`/`ATTACK` mask; dummy zero grid placeholder).
  2. Implement `training/model.py` v0 (Ray 1D Conv + NaN-safe Entity Attention + Vitals MLP + RGB CNN + GRU + Objective Slots + **8-sector Ego-Directional Value Compass `value_compass: (B, 8)`** + 9 Bernoulli heads + Critic) and wire `submission/agent.py` (strictly offline, zero `openai`/`llm_coach` imports).
* **Aman (M5):**
  1. Implement `training/mapgen/udmf_builder.py` (`zdbsp` node builder) and `tiers.py` v0 generating extreme domain-randomized Tier 1–2 maps (tight ambushes, open arenas, variable scales, zero fixed XYZ assumptions).
  2. Assemble initial practice map pool (`two_rooms.wad`, `tests/fixtures/*.wad`, ViZDoom scenarios).
* **H2 Gate:** `python validate.py --docker --image devforge-doom` (PR #3 auto-fallback + `check_imports()` zero warnings, PR #2 `2 GB VRAM / 8 GB RAM / 1 CPU thread` limits) passes on `submission/agent.py`; `envs.py` steps 16 workers on OMEN at `> 1,000` steps/sec.

---

## Phase 2: H2–H4 — 100% Self-Evolving PPO, `llm_coach.py`, Dense Rewards & Scorer
* **Vedesh (OMEN):**
  1. Implement **100% self-evolving** `training/ppo.py` (custom GAE, truncated BPTT `L=32` with `done` hidden-state masking, clipped policy/value losses, 9 Bernoulli heads, `value_compass` & `next_rays` aux losses — zero human demos required!), `training/rnd.py`, and `training/llm_coach.py` (`@cf/openai/gpt-oss-120b` via `CREO_API_KEY` env var with offline deterministic fallback).
  2. Run first end-to-end self-evolving PPO smoke run on `two_rooms.wad` + Aman's Tier 1–2 extreme maps; stream metrics to W&B.
* **Aryan (LOQ):**
  1. Implement `training/scorer/eval_suite.py`, `score_checkpoint.py` (mirroring `EVALUATION.md` on unseen extreme random maps), and `render_replay.py`.
  2. Write `tests/test_model_and_agent.py` verifying shapes, `value_compass`, reset isolation, and `torch.set_num_threads(1)` (`CPU_THREADS = 1` + `THREAD_ENV`) latency (`< 20 ms`).
* **Aman (M5):**
  1. Implement coordinate-free `training/rewards.py` using relative `privileged_state()` deltas & `sector_polygons()` (dense sector discovery, kills, door open, anti-wall-scrape, anti-`USE`-bounce).
  2. Lock fixed held-out extreme random benchmark WADs (`seed ∈ [1000, 9999]`) for Aryan's `eval_suite.json`.
* **H4 Gate:** Self-evolving PPO smoke test shows rising episode return and kill/door counts on W&B within 15 minutes; `score_checkpoint.py` produces a complete JSON scorecard on LOQ.

---

## Phase 3: H4–H6 — Odometry Grid, Extreme Tiers 3–4, PLR & Optional BC Compatibility
* **Vedesh (OMEN):**
  1. Implement optional/synthetic-compatible `training/bc/train_bc.py` + optional decaying BC-KL (`bc_kl_beta_init = 0.0` default for pure self-evolving RL) and comprehensive `tests/test_envs_and_ppo.py` (covering BPTT `done` masking, single-room PPO overfit, subprocess WAD reload FD check, `value_compass`, and `llm_coach` mocked HTTP tests).
* **Aryan (LOQ):**
  1. Implement `training/odometry_grid.py` (360° circular ray scan-matching + `cv2.warpAffine` `32×32` 6-ch semantic occupancy grid) + Objective Memory Slots + `value_compass` auxiliary head.
  2. Visually sanity-check warped grids on `tests/fixtures` and benchmark single-thread latency.
* **Aman (M5):**
  1. Complete Tier 3 (keycard branch → locked door → exit backtracking) and Tier 4 (multi-lock + tight ambush + wide arena + maze + scarce ammo gauntlet) extreme domain-randomized generators with internal BFS solvability assertions.
  2. Implement `training/curriculum_plr.py` (self-evolving PLR + scaffolding knob annealing).
* **H6 Gate:** Self-evolving PPO + PLR + Value Compass + Odometry Grid running end-to-end; Tiers 1–4 extreme procedural WADs pass 500-map stress load test with zero segfaults.

---

## Phase 4: H6–H8 — Launch Long Self-Evolving Run #1 & Automated LOQ Judge
* **Vedesh (OMEN):**
  1. Launch **Long Run #1** (`100% Self-Evolving Recurrent PPO + RND + Value Compass + PLR + Scaffolding`) inside `tmux` on OMEN.
  2. Setup checkpoint export every `250k` env steps to `runs/run1/checkpoints/`, querying `llm_coach.py` on telemetry summaries for bounded PLR knob diagnostics.
* **Aryan (LOQ):**
  1. Start automated watch loop (`rsync` new checkpoints from OMEN every 20–30 min → run `score_checkpoint.py` on extreme random held-out suite → post `eval/*` to W&B → if new best & passes `validate.py --docker`, copy to staging).
* **Aman (M5):**
  1. Curate additional held-out Freedoom 2 maps (`MAP01–MAP05`) and permissive community single-level WADs (`< 2048` sectors) to test out-of-distribution generalization.
  2. Inspect Run #1 W&B reward breakdowns for reward-hacking (e.g., spinning or farming) and draft root `README.md` updates.
* **H8 Gate:** Run #1 stable past `2M+` steps on OMEN with zero memory leaks; LOQ automated judge has scored ≥ 3 checkpoints onto W&B.

---

## Phase 5: H8–H12 — Overnight / Mid-Event Training, Ablations & Brief Draft
* **Vedesh (OMEN):**
  1. Keep OMEN running at max sustainable worker throughput; inspect W&B every 1–2 hours (or via `/schedule` + `llm_coach.py`).
  2. Tune `ent_coef`, `rnd_coef`, `compass_coef`, or PLR weights and launch **Run #2** from best checkpoint if needed.
* **Aryan (LOQ):**
  1. Run a controlled **Ablation Training Run** on LOQ (`use_grid=False` / no `value_compass`) so we have genuine quantitative ablation rows for the 2-Page Brief.
  2. Capture side-by-side comparison replays via `render_replay.py` (RGB + Egocentric Occupancy Grid).
* **Aman (M5):**
  1. Expand extreme procedural geometry & texture variations (L-shaped hallways, pillars, height steps ≤ 24 units, variable lighting, extreme arena/corridor aspect ratios).
  2. Draft Page 1 of the **2-Page Project Brief PDF** (USP, self-evolving Hybrid + Scan-Match Grid + Ego Value Compass architecture, Rule compliance).
* **H12 Gate:** Held-out Tier 1 survival `> 80%`, Tier 2 sector coverage `> 65%`, and Tier 3 keycard pickups/unlocks observed on unseen random maps.

---

## Phase 6: H12–H16 — Full Architecture Push & Feature Freeze
* **Vedesh (OMEN):**
  1. Launch/continue **Run #3 (Flagship Full Config)** with RGB modality dropout (`p=0.25`) + full grid + objective slots + 8-sector `value_compass`.
* **Aryan (LOQ):**
  1. Profile exact single-thread CPU & GPU memory inside `validate.py --docker --image devforge-doom` (`CPU_THREADS = 1`, `MAX_VRAM_MB = 2048`, `MAX_RAM_MB = 8192`, `RamSampler`, `one_cpu()`). If mean latency `> 20 ms`, optimize conv channels before H16 freeze.
* **Aman (M5):**
  1. Watch replay recordings from `render_replay.py` on held-out extreme Tier 3 & Tier 4 failures; adjust `rewards.py` / PLR weights before H16 freeze.
  2. Draft Page 2 of Brief PDF (self-evolving pipeline, random-map ablation table skeleton, `llm_coach.py` offline curriculum design, honest limitations).
* **H16 Gate (HARD FEATURE FREEZE):** All code changes to `model.py`, `preprocessing.py`, `odometry_grid.py`, and `agent.py` **freeze at H16**. Only weight checkpoints, training configs, tests, and documentation change after H16.

---

## Phase 7: H16–H20 — Mandatory Vanilla-Only Fine-Tune & Checkpoint Lock
* **Vedesh (OMEN):**
  1. Take the highest-scoring checkpoint on held-out eval at H17–H18 and run the **Final Vanilla-Only Fine-Tune** (`scaffolding_alpha = 0.0`, 100% vanilla doors/ammo/health, `SKILL = 3`, `bc_kl = 0.0`, lower `lr = 8e-5`) for `2–4 hours`.
* **Aryan (LOQ):**
  1. Run an expanded 40-map extreme random held-out tournament across top 5 candidate checkpoints + 4,200-tick endurance test.
  2. Promote the winning checkpoint to `submission/weights/best_agent.pt` via **Git LFS** and verify `python validate.py --docker`.
* **Aman (M5):**
  1. Polish `training/README.md`, root `README.md`, and populate final quantitative results & ablation numbers into the **2-Page Project Brief PDF** (using `llm_coach.py` CLI summary if helpful).
* **H20 Gate (CHECKPOINT LOCK):** `submission/weights/best_agent.pt` committed with Git LFS on `main`, passing `validate.py --docker --image devforge-doom` (`2 GB VRAM / 8 GB RAM / 1 CPU thread`, zero `check_imports()` warnings) with `< 25 ms/step`.

---

## Phase 8: H20–H24 — Cold-Clone Verification, Brief Export & Final Submission
* **Vedesh (OMEN):**
  1. Perform a **fresh `git clone` into a clean verification folder** (testing LFS pull!), run `docker build -t devforge-doom .` and `python validate.py --docker --image devforge-doom`.
  2. Verify `git status` clean, `git lfs ls-files` lists weights, total repo size `< 1 GB`, zero secrets or `.env` files in commit history, `uv run pytest` 100% pass, repository set to **Public**.
* **Aryan (LOQ):**
  1. Audit every line under `submission/` against `RULES.md` (zero `atan2`, zero `A*`/BFS, zero fixed XYZ assumptions, zero `openai`/`llm_coach`/network imports, empty or strictly valid `submission/requirements.txt` passing `check_imports()`).
  2. Run independent cold-clone `validate.py --docker --image devforge-doom` check on LOQ + polish showcase MP4 replay.
* **Aman (M5):**
  1. Export final **2-Page Project Brief PDF** (strictly `≤ 2 pages`) and prepare the 10-question Desk-Review Q&A sheet & 5-min team dry-run.
* **H23.5 Gate (HARD SUBMISSION CUTOFF):** Final commit SHA recorded, GitHub URL + 2-Page PDF submitted on portal 30 minutes ahead of deadline.

---

## Final Pre-Submission Checklist
- [ ] Repository is **Public** and synced from `Reinforce-SST/devforge-doom-template` (`63b90c4`) on `origin/main` (`https://github.com/aryanbhojgaria/canitdoom.git`).
- [ ] `git lfs ls-files` shows `submission/weights/best_agent.pt` (repo size `< 1 GB`).
- [ ] Zero secrets committed (`.env` ignored; only `.env.example` with placeholders committed).
- [ ] `submission/requirements.txt` has only comment lines (or pinned `pkg==ver` passing `check_imports()` and tested in Docker).
- [ ] Fresh shallow clone passes `python validate.py --docker --image devforge-doom` with zero warnings (including `check_imports()` & PR #2 `2 GB VRAM / 8 GB RAM / 1 CPU thread` limits) and `< 25 ms/step` latency.
- [ ] `submission/` contains zero hardcoded rule trees, pathfinders, fixed XYZ assumptions, `atan2` aimbots, or `openai`/network calls.
- [ ] `training/` contains complete, documented extreme MapGen, PLR, RND, `llm_coach.py`, and 100% self-evolving Recurrent PPO source code.
- [ ] 2-Page Project Brief PDF is `≤ 2` pages, proofread, and uploaded alongside the repo URL.
