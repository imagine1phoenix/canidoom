# Vedesh (HP OMEN 16 — i9-14900HX + RTX 5060 8 GB) — Step-by-Step Runbook

## 1. Role & Ownership Summary (3-Person Team)
* **Role:** Team Lead, Repository Gatekeeper & Primary RL Core Engineer (**Paid Google AI Pro Account**).
* **Files You Own:**
  * Repository setup, `origin/main` milestone gates, `.gitignore` (incl. `.env`), `.env.example`, Git LFS config
  * `training/configs/bc.yaml`, `training/configs/ppo_curriculum.yaml`, `training/configs/eval.yaml`
  * `training/envs.py` (Subprocess vectorized `doom_env` workers + extreme domain-randomization & scaffolding hooks + procedural WAD hot-swap)
  * `training/ppo.py` (**100% self-evolving** Custom Recurrent PPO: GAE, truncated BPTT `L=32`, 9 Bernoulli heads, action masking, **8-sector Ego-Directional Value Compass `value_compass`** & `next_rays` aux losses, optional decaying BC-KL)
  * `training/rnd.py` (Random Network Distillation + grid cell-visit novelty bonus)
  * `training/llm_coach.py` (**Training/Offline-Only** `@cf/openai/gpt-oss-120b` Curriculum & Scorecard Advisor reading `CREO_BASE_URL`, `CREO_MODEL`, `CREO_API_KEY` from env/`.env` with deterministic offline fallback; **NEVER imported in `submission/`**)
  * `training/bc/train_bc.py` (Optional/synthetic-compatible Behavior Cloning warm-start trainer — never a blocker for self-evolving RL)
  * `tests/test_envs_and_ppo.py` (+ `value_compass` & `llm_coach` mocked HTTP / offline fallback tests)
* **Interfaces You Provide & Consume (`Schema.md`):**
  * **Consume from Aryan:** `preprocess(obs, state=None)` (`training/preprocessing.py`) and `HybridDoomNet.forward(batch, h, return_aux=False)` (`training/model.py`; pass `return_aux=True` inside PPO to receive `next_rays` + `value_compass: (B, 8)` + `door_prox`/`key_prox`/`exit_prox`).
  * **Consume from Aman:** `MapGen.sample(tier, knobs=None, seed=0, scaffolding_alpha=0.0)` (`training/mapgen/tiers.py`), `RewardShaper.reset(sector_polygons=None)` + `RewardShaper.step(priv_prev, priv_now, action, info=None)` (`training/rewards.py`), and `PLRBuffer` (`training/curriculum_plr.py`).
  * **Provide to Aryan:** Periodic `.pt` checkpoints in `runs/<run_name>/checkpoints/` synced to LOQ for held-out random-map evaluation.

---

## 2. M1 Machine Setup (WSL2 Ubuntu 24.04 + `uv` + Docker)

Run inside **Windows PowerShell** then **WSL2 Ubuntu 24.04**:
```bash
# 1. WSL2 Ubuntu 24.04 (in Windows PowerShell)
wsl -d Ubuntu

# 2. Inside Ubuntu 24.04: core packages, node builder, SSH, tmux (note libglib2.0-0t64 on Ubuntu 24.04)
sudo apt update && sudo apt install -y ca-certificates curl wget git git-lfs libgl1 libglib2.0-0t64 libsdl2-2.0-0 build-essential cmake libsdl2-dev zdbsp tmux htop rsync
git lfs install

# 3. Check exact logical CPU count & GPU visibility
nproc          # 32 logical threads on i9-14900HX; set num_envs = 16
nvidia-smi     # Confirm RTX 5060 8GB is visible

# 4. Install uv + Python 3.13 + configure ext4 UV_PROJECT_ENVIRONMENT
curl -LsSf https://astral.sh/uv/install.sh | sh
ln -sf /root/.local/bin/uv /usr/local/bin/uv
ln -sf /root/.local/bin/uvx /usr/local/bin/uvx
export UV_PROJECT_ENVIRONMENT="/root/.venvs/devforge-doom"
uv python install 3.13

# 5. Sync synced repo root (installs training/ + tests/ deps from pyproject.toml) & validate
cd "/mnt/c/Users/Vedesh/Documents/Hackathons/DevForge X Scaler"
uv sync --python 3.13
uv run pytest
uv run python validate.py
uv run python -c "import torch, vizdoom, yaml, cv2; print('CUDA:', torch.cuda.is_available(), 'ViZDoom:', vizdoom.__version__)"

# From Windows PowerShell (or WSL2 if Docker socket integrated):
docker build -t devforge-doom .
python validate.py --docker --image devforge-doom

# 6. Tailscale + W&B + local .env (NEVER commit .env!)
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up
tailscale ip -4   # Post this IP in team chat so Aryan & Aman can SSH/rsync
uv tool run wandb login
cp .env.example .env  # Add CREO_API_KEY locally only; verify git check-ignore -v .env
```

---

## 3. Phase-by-Phase Execution Checklist

### Phase 1 (H0–H2): Repo Sync, Configs & Parallel Subprocess Envs
* **AI Model & Command:** `Gemini Flash` + `/plan`.
1. **H0 Sync:** Sync `Reinforce-SST/devforge-doom-template` (`63b90c4`) into `origin/main` (`https://github.com/aryanbhojgaria/canitdoom.git`); push completed, verified milestones directly to `main` after `uv run pytest` and `validate.py` gates pass.
2. Add `.env`, `runs/`, `data/demos/`, `/tmp_wads/`, `tmp_wads/`, `.venv/` to `.gitignore`; commit `.env.example` (placeholders only); verify `.gitattributes` tracks `*.pt`, `*.safetensors`, `*.onnx` via Git LFS.
3. Implement `training/configs/{ppo_curriculum.yaml,bc.yaml,eval.yaml}` and `training/envs.py`:
   * Use `multiprocessing.get_context("spawn")` workers wrapping `doom_env.make_env`.
   * Return `(obs, shaped_reward, terminated, truncated, privileged_summary)` per step (zero fixed XYZ assumptions).
   * Support `reset_with_wad(wad_path, map_name, seed, episode_ticks)` and extreme domain-randomization/scaffolding hooks (`scaffolding_alpha`) so PLR swaps random maps cleanly between episodes.
4. Benchmark `16` parallel workers stepping random actions (`render_buffers=True` vs `False`).
* **Done Criteria (H2 Gate):** `envs.py` achieves `> 1,000` steps/sec across 16 workers with zero deadlocks; `validate.py --docker --image devforge-doom` (PR #2 `2 GB VRAM / 8 GB RAM / 1 CPU thread`, PR #3 `check_imports()` zero warnings) passes.

### Phase 2 (H2–H4): 100% Self-Evolving Recurrent PPO + RND + `llm_coach.py`
* **AI Model & Command:** **`Gemini Pro (High)` or `Claude Opus`** + **`/boost`**.
1. Implement `training/ppo.py` (**100% self-evolving without requiring human demos**):
   * Rollout buffer storing `(num_envs=16, rollout_steps=128)` transitions + initial `RecurrentState` every `bptt_len=32` steps.
   * **Crucial Recurrent Rule:** Inside truncated BPTT (`T=32`), multiply `gru_h` and `obj_slots` by `(1.0 - done[t-1])` at episode boundaries so memory never bleeds across maps!
   * Masked `Bernoulli(logits=masked_logits)` log-probs summed across the `9` buttons + entropy bonus + GAE (`γ=0.99, λ=0.95`) + clipped value/policy loss + **8-sector Ego-Directional Value Compass (`value_compass`)** & `next_rays` auxiliary losses.
2. Implement `training/rnd.py` (fixed random target MLP vs predictor MLP on normalized ray+vitals features; Welford running mean/std normalization + grid novelty).
3. Implement `training/llm_coach.py` (stdlib `urllib.request` + optional `openai` client, deterministic offline fallback) reading `CREO_BASE_URL`, `CREO_MODEL`, `CREO_API_KEY` from env/`.env` for bounded PLR knob advising & scorecard summarization.
4. Wire Aman's `rewards.py` and run a self-evolving PPO smoke run on `maps/two_rooms.wad` + Aman's extreme Tier 1–2 maps logging to W&B.
* **Done Criteria (H4 Gate):** Smoke run completes without NaNs; `rollout/ep_return` climbs on W&B; `tests/test_envs_and_ppo.py` passes.

### Phase 3 (H4–H6): Optional/Synthetic BC Compatibility & Comprehensive Pytest Suite
* **AI Model & Command:** `Gemini Flash` (`/plan`).
1. Implement `training/bc/train_bc.py` (optional/synthetic-compatible `.npz` loader chunked into `T=32` sequences, sequential `preprocess()` supporting `(48, 64)` frames directly, masked `BCEWithLogitsLoss` with `pos_weight=2.5` on `USE`/`ATTACK`, saving `bc_warmstart.pt` in `Schema.md §4` format) + optional decaying `bc_kl_beta` (`0.0` default for pure self-evolving RL).
2. Expand `tests/test_envs_and_ppo.py` to test `value_compass` loss/gradients and `llm_coach.py` (mocked HTTP + offline fallback + zero hardcoded key checks).
* **Done Criteria (H6 Gate):** 100% self-evolving PPO + PLR + Value Compass running across extreme Tiers 1–4; all unit & E2E tests green.

### Phase 4 (H6–H8): Launch Long Self-Evolving Run #1 (`tmux`)
* **AI Model & Command:** `Gemini Flash` + `/schedule` + `llm_coach.py`.
1. Launch **Long Run #1** inside `tmux new -s run1`:
   * Full `HybridDoomNet` (Rays + Entities + Vitals + `64×48` RGB w/ `0.25` dropout + `32×32` Scan-Match Grid + Slots + 8-sector `value_compass`), self-evolving PPO from scratch, extreme PLR over Tiers 1–4, RND + Grid novelty active.
2. Save checkpoints every `250k` steps (`runs/run1/ckpt_*.pt`) so Aryan's LOQ pulls them automatically via Tailscale.
3. Periodically pass rollout/scorer telemetry JSON through `training/llm_coach.py` to diagnose plateaus and refine PLR bounds.
* **Done Criteria (H8 Gate):** Run #1 past `2M+` steps smoothly; GPU utilization healthy; zero RAM creep.

### Phase 5 & 6 (H8–H16): Mid-Event Iteration, Pro Quota Support & Feature Freeze
* **AI Model & Command:** **`Gemini Pro (High)` / `Claude Opus`** + `/boost` for diagnostics & helping Aryan/Aman if they hit Free-tier limits.
1. Monitor W&B `eval/tier1..4` curves coming from Aryan's LOQ random-map judge.
2. Tune `ent_coef`, `rnd_coef`, `compass_coef`, or extreme PLR weights and launch **Run #2 / Run #3** warm-started from best self-evolving checkpoint.
3. **Act as Team Paid-Pro Backup:** If Aryan or Aman hits a Free-tier reasoning quota wall before H16, pull their branch on OMEN and run `/boost` for them.
4. **Enforce H16 Feature Freeze:** Freeze model/mapgen/preprocessing architecture by **H16:00**.
* **Done Criteria (H16 Gate):** Architecture frozen; strong upward held-out curves across extreme random Tiers 1–3.

### Phase 7 (H16–H20): Mandatory Vanilla-Only Fine-Tune
* **AI Model & Command:** `Gemini Pro (High)`.
1. At H17–H18, take the highest-scoring held-out checkpoint and launch `run_vanilla_finetune`:
   * `scaffolding_alpha = 0.0` (100% vanilla DOOM doors, ammo, health, `SKILL=3`), `bc_kl_beta = 0.0`, `lr = 8e-5`, `ent_coef = 0.005`, matching unseen evaluation extreme random maps.
2. Stream checkpoints every `150k` steps to Aryan's LOQ 40-map tournament.
* **Done Criteria (H20 Gate):** Vanilla fine-tuned checkpoint wins LOQ tournament and is committed to `submission/weights/best_agent.pt` via Git LFS.

### Phase 8 (H20–H24): Clean-Clone Docker Audit & Final Submission
* **AI Model & Command:** `Gemini Flash`.
1. Clean `training/` scripts, verify `uv run pytest` passes 100%, push final docs/brief updates.
2. **Mandatory Cold-Clone Test on OMEN:**
   ```bash
   git clone https://github.com/aryanbhojgaria/canitdoom.git clean-audit
   cd clean-audit
   git lfs ls-files          # Must list submission/weights/best_agent.pt
   du -sh .                  # Must be well under 1 GB
   python validate.py --docker --image devforge-doom  # PR #2 limits + PR #3 check_imports()
   ```
3. Confirm repo is **Public**, zero secrets/keys/`.env` in git log, and submit GitHub URL + Aman's 2-Page PDF by **H23.5**.

---

## 4. Tests You Must Write (`tests/test_envs_and_ppo.py`)
* `test_bptt_done_masking()`: Verify hidden state `gru_h` and `obj_slots` zero out cleanly at step `t` when `done[t-1] == True` inside a 32-step chunk.
* `test_ppo_overfit_single_room()`: 60-second sanity test proving pure self-evolving PPO loss decreases and action probabilities shift toward positive reward.
* `test_subprocess_env_reload()`: Verify worker swaps between two different WAD paths across `5` resets without leaking file descriptors.
* `test_value_compass_aux_loss()` & `test_llm_coach_mocked_and_offline()`: Verify 8-sector ego-directional compass output `(B, 8)` & loss finiteness, plus `llm_coach.py` mocked HTTP `gpt-oss-120b` parsing and deterministic offline fallback when `CREO_API_KEY` is unset.

---

## 5. If Things Go Wrong (Vedesh's Fallbacks)
* **Exploration stalls on extreme Tier 3/4 random maps:** Increase `compass_coef` (`0.15 → 0.25`) and `grid_novelty_coef` (`0.03 → 0.06`), and run `llm_coach.py` on recent rollout stats to rebalance PLR scenario weights.
* **ViZDoom workers leak RAM or hang after 500 episodes:** Recycle subprocess worker every `100` episodes inside `envs.py` (`worker_max_episodes = 100`) and wrap `step()` in a 15s pipe timeout that restarts the single worker transparently.
* **Hybrid RGB slows OMEN SPS below 500:** Run 12 workers with `render_buffers=False` (leveraging `rgb_dropout` zeroing!) and 4 workers with `render_buffers=True`, doubling throughput while keeping visual branch calibrated.
