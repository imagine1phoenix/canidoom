# Aryan (Lenovo LOQ — i5-13th Gen + RTX 4050) — Step-by-Step Runbook

## 1. Role & Ownership Summary (3-Person Team)
* **Role:** Neural Architecture, Spatial Memory, Inference Contract (`DoomAgent`), Automated Random-Map Judge & Replay Visualizer (**Free Tier — Ration Pro Turns Carefully!**).
* **Files You Own:**
  * `training/preprocessing.py` (Strictly coordinate-free observation scaling, `(120,160)` & `(48,64)` RGB+depth support, legal `USE`/`ATTACK` action mask)
  * `training/odometry_grid.py` (360° circular ray scan-matching yaw estimator + `cv2.warpAffine` `32×32` 6-ch occupancy grid)
  * `training/model.py` (`HybridDoomNet`: Circular 1D Ray Conv, NaN-safe Masked Entity Attention, `(B, 36)` Vitals MLP, Compact RGB+Depth CNN w/ dropout, Grid CNN, GRU + 4 Objective Memory Slots, 9 Bernoulli Heads, Critic, Aux Heads incl. **8-sector Ego-Directional Value Compass `value_compass: (B, 8)`**)
  * `submission/agent.py` (`DoomAgent`: `__init__`, `reset`, `act` — `torch.set_num_threads(1)`, **offline `--network none`**, zero `openai`/`llm_coach` imports, strict `< 25 ms/step` target)
  * `training/scorer/eval_suite.py`, `training/scorer/score_checkpoint.py` (Local mirror of `EVALUATION.md` on unseen extreme random maps + automated promoter) & `training/scorer/render_replay.py` (Side-by-side RGB + Egocentric Grid replay visualizer)
  * `tests/test_model_and_agent.py`
* **Interfaces You Provide & Consume (`Schema.md`):**
  * **Provide to Vedesh:** `preprocess(obs, state=None)` and `HybridDoomNet` (with `value_compass: (B, 8)` when `return_aux=True`) matching `Schema.md §2` shapes.
  * **Consume from Aman:** Held-out extreme random-map benchmark manifest & WAD files (`maps/heldout/*.wad` + `eval_suite.json`).
  * **Consume from Vedesh:** Incoming `.pt` checkpoints from OMEN via Tailscale (`runs/incoming/`).

---

## 2. Phase 0: Pre-Event Machine Setup (Complete Before H0 — Zero Project Code!)

Run inside **WSL2 Ubuntu 24.04** on LOQ:
```bash
# 1. System packages + Docker WSL2 integration check (note libglib2.0-0t64 on Ubuntu 24.04)
sudo apt update && sudo apt install -y ca-certificates curl wget git git-lfs libgl1 libglib2.0-0t64 libsdl2-2.0-0 build-essential cmake libsdl2-dev rsync tmux htop
git lfs install
nvidia-smi     # Confirm RTX 4050 is visible

# 2. Install uv + Python 3.13
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv python install 3.13

# 3. Pre-build official Docker image & test validate.py (delete /tmp/setup-check before H0!)
git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
cd /tmp/setup-check
docker build -t devforge-doom .
uv sync
uv run python validate.py --docker --image devforge-doom
cd ~ && rm -rf /tmp/setup-check

# 4. Tailscale + test rsync from Vedesh's OMEN + W&B login
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
ssh <vedesh-user>@<omen-tailscale-ip> "echo connected"
uv tool run wandb login
```
* **AI Quota Discipline (Free Account):** Save **Gemini Pro (High) / Claude Opus** strictly for Phase 1 (`model.py`) and Phase 3 (`odometry_grid.py`). Use **Gemini Flash** for `score_checkpoint.py`, latency scripts, Docker checks, and unit tests!

---

## 3. Phase-by-Phase Execution Checklist

### Phase 1 (H0–H2): Preprocessor v0, `HybridDoomNet` v0 & `submission/agent.py`
* **AI Model & Command:** `Gemini Pro (High)` + `/plan` (after 10-min H0 schema lock with Vedesh).
1. Clone team repo created by Vedesh, run `uv sync`, checkout `feat/aryan-model-v0`.
2. Implement `training/preprocessing.py` v0 matching `Schema.md §2`:
   * Pack `rays` `(B, 8, 64)`, `ent_feats` `(B, 32, 5)`, `ent_types` `(B, 32)`, `ent_mask` `(B, 32)`, `vitals` `(B, 36)`, `rgb_depth` `(B, 4, 48, 64)` via `cv2.resize`.
   * Implement legal `action_mask`: `USE` 6-step cooldown after front-door `USE=1`; `ATTACK` masked when `selected_weapon > 1 and ammo <= 0`.
   * Pass zero placeholder `(B, 6, 32, 32)` for `grid` in v0 so Vedesh can start PPO immediately.
3. Implement `training/model.py` (`HybridDoomNet` with `use_grid` and `use_rgb` config toggles; handle `ent_mask.sum(dim=1) == 0` safely with `masked_fill` before softmax so 0 visible entities never produces `NaN`!).
4. Implement `submission/agent.py` (`DoomAgent`):
   * Call `torch.set_num_threads(1)` when `device == "cpu"`.
   * Load checkpoint from `submission/weights/best_agent.pt` (save an initial untrained checkpoint via Git LFS for smoke validation).
   * Reset `OdomGridState`, `RecurrentState`, and `np.random.default_rng(0)` in `reset()`.
5. Run `uv run python validate.py --docker --image devforge-doom` and open PR to `main`.
* **Done Criteria (H2 Gate):** `validate.py --docker` passes 100% green; mean single-thread CPU `act()` time `< 15 ms`.

### Phase 2 (H2–H4): Held-Out Local Scorer (`score_checkpoint.py`)
* **AI Model & Command:** **`Gemini Flash`** + `/plan` *(save Pro quota for Phase 3!)*.
1. Checkout `feat/aryan-scorer`. Implement `training/scorer/eval_suite.py` & `score_checkpoint.py`:
   * Load Aman's `eval_suite.json` (disjoint held-out maps across Tiers 1–4).
   * Step `DoomAgent` sequentially while reading `env.privileged_state()` and `env.sector_polygons()` in the **scorer loop** (never exposed to agent) to compute:
     * **Exit points** (`+40` + health/time efficiency up to `+10`),
     * **Progression points** (`+15` keycard, `+5` door opened, `+15` locked door unlocked),
     * **Exploration points** (`+20 × unique_sectors_visited / total_reachable_sectors`),
     * **Combat points** (`kills × 3 - damage_taken × 0.05 - (10 if early_death else 0)`),
     * Weighted across Tiers 1–4 (`[1.0, 1.5, 2.5, 3.0]`).
   * Record `mean_act_ms`, `p99_act_ms`, and verify against `env/doom_env/limits.py`.
* **Done Criteria (H4 Gate):** `score_checkpoint.py` scores random vs untrained baseline cleanly and logs `eval/*` to W&B.

### Phase 3 (H4–H6): 360° Scan-Matching Odometry Grid + Objective Slots
* **AI Model & Command:** **`Gemini Pro (High)` or `Claude Opus`** + **`/boost`** *(spend your saved Pro turns here!)*.
1. Checkout `feat/aryan-grid-slots`. Implement `training/odometry_grid.py` (`Design.md §2.2`):
   * Circular shift search `k ∈ [k_prior - 4, k_prior + 4]` across 64 rays + 3-point parabolic refinement for continuous `Δθ`.
   * `cv2.warpAffine` on `(32, 32, 6)` grid shifted by `(vel_forward * 4 / 48.0, vel_right * 4 / 48.0)` and rotated by `-Δθ`, then splat current 64 ray hits into `visited, wall, door, locked_door, item, enemy_recent`.
2. Enable `Objective Memory Slots (4, 8)` with 2D rotation by `-Δθ` and training-only `aux_preds` (`next_rays`, **`value_compass: (B, 8)`**, `door_prox`, `key_prox`, `exit_prox`) in `model.py`.
3. Visualize 1 episode of the 6 grid channels as PNGs on `maps/two_rooms.wad` and build `training/scorer/render_replay.py` (side-by-side RGB + Ego Grid replay capture)!
* **Done Criteria (H6 Gate):** Walls align cleanly across 90° turns in PNG debug plot; single-thread CPU `act()` with Grid+RGB stays `< 20 ms`; merged before Vedesh launches Run #1.

### Phase 4 (H6–H8): Automated Checkpoint Judge Daemon on LOQ
* **AI Model & Command:** `Gemini Flash` + **`/schedule`**.
1. Set up automated watcher script + `/schedule` every 30 minutes:
   * `rsync -avz <vedesh>@<omen-ip>:~/repo/runs/ ~/repo/runs/incoming/`
   * Evaluate any unseen `.pt` checkpoint on `eval_suite.json`, log `eval/*` to W&B, and compare `total_score` to current champion.
   * If `total_score > best_score` AND `mean_act_ms < 25.0`, automatically test `validate.py --docker` and notify team chat!
* **Done Criteria (H8 Gate):** First 3 checkpoints from Vedesh's Run #1 automatically scored and plotted on W&B.

### Phase 5 & 6 (H8–H16): Controlled Ablation Run & Latency Optimization
* **AI Model & Command:** `Gemini Flash` + `/goal`.
1. Use LOQ's RTX 4050 between checkpoint evaluations (`nice -n 10`, `8` workers) to train a **No-Grid / No-Compass Ablation (`use_grid=False`)** for 4–5M steps on the exact same extreme random-map stream.
   * *Why:* Gives Aman undeniable quantitative proof for Brief Page 2 that Scan-Matched Grid + Value Compass improves Tier 2/3 scores, **and** gives us a tested backup checkpoint if Grid ever misbehaves!
2. Profile `DoomAgent` inside `docker run --cpus=1 --memory=8g` using `validate.py --docker`. Trim conv channels or enable `torch.inference_mode()` optimizations before **H16 Feature Freeze**.
* **Done Criteria (H16 Gate):** Ablation metrics recorded; `DoomAgent` frozen at `< 18 ms/step` on 1 CPU core in Docker.

### Phase 7 & 8 (H16–H24): 40-Map Final Tournament, LFS Promotion & Rule Audit
* **AI Model & Command:** `Gemini Flash`.
1. Between H18–H20, run the expanded 40-map held-out tournament across top Run #3 and Vanilla Fine-Tune checkpoints (including 4,200-tick endurance maps).
2. Copy the winning `.pt` to `submission/weights/best_agent.pt`, verify `git lfs track`, commit & push to `main`, and send final score breakdown table + replay captures to Aman for the 2-Page Brief.
3. **Rule Compliance Audit (H21):** Read every single line in `submission/` and `training/preprocessing.py` + `odometry_grid.py` to confirm zero banned constructs (`RULES.md §1.1`), zero fixed XYZ assumptions, zero `openai`/`llm_coach`/network imports, and confirm `submission/requirements.txt` has zero extra packages (`check_imports()` passes).
4. Run independent clean-clone `python validate.py --docker` check on LOQ by **H22.5**.

---

## 4. Tests You Must Write (`tests/test_model_and_agent.py`)
* `test_zero_entities_no_nan()`: Feed `ent_mask = zeros(32)` and `rays.distance = ones(64)` → verify zero NaNs in logits/value.
* `test_reset_clears_memory()`: Run 10 steps on map A, call `agent.reset()`, verify `agent` output on step 0 of map B is bit-for-bit identical to a fresh `DoomAgent` instance.
* `test_single_thread_latency()`: Run 200 sequential `act()` calls under `torch.set_num_threads(1)` and assert `mean < 25 ms` and `max < 50 ms`.
* `test_action_contract()`: Assert output `dtype == np.int8`, `shape == (9,)`, and `set(np.unique(act)) <= {0, 1}`.

---

## 5. If Things Go Wrong (Aryan's Fallbacks)
* **Scan-matching smears walls during fast combat turns:** Weight the turn-button prior (`k_prior`) higher (`0.10`) when `ent_mask.sum() > 2` (moving enemies perturb rays; filter out rays where `hit_type == 4 (enemy)` before scan-matching!). If still noisy by H14, promote the `use_grid=False` GRU+Slots model.
* **CPU 1-thread latency creeps over `25 ms`:** Downsample RGB to `32×24` or update `odometry_grid` every `2nd` step while still running GRU every step.
* **Free-tier Gemini Pro quota exhausted mid-debug:** Commit branch `feat/aryan-...`, ping Vedesh, and have Vedesh run `/boost` on OMEN's Paid Pro account.
