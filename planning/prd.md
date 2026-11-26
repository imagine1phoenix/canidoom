# Product Requirements Document (PRD) — Track 1: Zero-Shot DOOM Agent

## 1. Problem & Opportunity

Standard reinforcement learning agents in 3D FPS games memorize pixel sequences or fixed world coordinates on training maps and fail immediately when walls move, textures change, corridors branch unpredictably, or extreme random scenarios appear. Track 1 (*"Can It Play DOOM?"*) evaluates autonomous agents **zero-shot on procedurally / randomly generated maps and extreme unseen scenarios** across four escalating tiers: Combat & Survival, Multi-Room Exploration, Long-Horizon Keycard Progression, and Full Gauntlet.

Because scoring is **non-binary** (weighted sum of Exit, Key/Door Progression, Unique Sector Exploration, and Combat) and manual code/replay reviews shortlist the **Top 2 finalists**, a **100% self-evolving generalist agent** pairing **ego-centric scan-matching spatial memory + 8-sector ego-directional value compass** with **extreme domain-randomized procedural training** has a decisive edge over off-the-shelf pixel RL baselines (with zero reliance on human demos or fixed coordinates).

---

## 2. Primary Goal & Stakeholders

* **Goal:** Finish **#1 on the Automated Leaderboard** (advancing safely in Top 10), pass the **Top 4–5 Manual Code & Replay Audit** with zero rule flags, and secure **Top 2 in Desk Review** to win **Round 3 Finals**.
* **Target Evaluators:**
  1. **Stage 1 Automated Scorer:** Headless sandboxed evaluator testing unseen maps, timeouts (`40 ms/step`), memory caps, and partial-progress metrics.
  2. **Stage 2 Desk Reviewers:** Technical organizers running live demos, inspecting `training/` & replays, and grilling the team on architecture and RL decisions.
  3. **Round 3 Final Judges:** Panel rewarding genuine architectural innovation (bespoke spatial memory, curiosity, procedural generalization) and live presentation clarity.

---

## 3. Target Success Metrics (Measured on Held-Out Local Scorer)

| Metric | Target Threshold (by H20 Lock) | Why It Matters |
| :--- | :--- | :--- |
| **Tier 1 (Combat & Survival)** | `≥ 85%` episode survival/completion, `≥ 80%` enemy kill ratio | Secures reliable foundational points and proves combat competence. |
| **Tier 2 (Multi-Room Exploration)** | `≥ 75%` unique sector coverage, `≥ 60%` exit rate, `0` door-bounce traps | Separates spatial explorers from corner-spinning pixel bots. |
| **Tier 3 (Keycard → Locked Door → Exit)** | `≥ 45%` keycard pickup + door unlock rate, `≥ 25%` full exit rate | High tier multiplier; almost no standard PPO bots score progression here. |
| **Tier 4 (Full Gauntlet)** | Positive net combat + `≥ 50%` sector coverage + opportunistic exits | Highest tier multiplier; surviving ambushes while exploring maximizes partial points. |
| **Single-Thread `act()` Latency** | Mean **`< 20 ms`**, p99 **`< 35 ms`** (`torch.set_num_threads(1)`) | Safe headroom below official `40 ms/step` compute budget and `5 s` per-call cap. |
| **Memory & Repo Footprint** | Peak VRAM `< 1.5 GB`, RAM `< 4 GB`, Repo + LFS `< 250 MB` (`≤ 1 GB` cap) | Passes PR #2 locked sandbox limits (`MAX_VRAM_MB = 2048`, `MAX_RAM_MB = 8192`, `CPU_THREADS = 1`) with margin. |
| **Validator & Test Pass Rate** | `100%` pass on `python validate.py --docker --image devforge-doom` (auto-fallback per PR #3), `check_imports()` (zero warnings), and `pytest` | Eliminates silent 0-score container, missing-import, or schema failures (with PR #3 Windows TCP loopback fallback in `doom_env.remote`). |

---

## 4. Functional Requirements

| ID | Requirement | Owner | Priority |
| :--- | :--- | :--- | :--- |
| **FR-1** | **Contract-Compliant `DoomAgent` (`submission/agent.py`):** Implement `__init__(obs_space, act_space, device)`, `reset()`, and `act(obs) -> MultiBinary(9)` loading LFS weights from `submission/weights/` with zero network/write calls and zero `openai`/`llm_coach` imports. | Aryan | P0 |
| **FR-2** | **Hybrid Sensory Encoder (`training/model.py`):** Fuse circular 1D CNN over 64 360° rays (+ hit-type embeddings), permutation-invariant masked attention over `≤ 32` visible entities (zero-entity NaN-safe), vitals MLP `(B, 36)`, and compact `64×48` RGB+depth CNN trained with RGB modality dropout. | Aryan | P0 |
| **FR-3** | **360° Scan-Matching Grid, Objective Slots & Ego Value Compass (`odometry_grid.py`, `model.py`):** Estimate yaw `Δθ` from consecutive 360° ray scans + turn history, warp a `32×32` 6-channel semantic occupancy grid using ego velocities, maintain GRU + 4 Objective Memory Slots, and train an **8-sector Ego-Directional Value / Advantage Compass (`value_compass: (B, 8)`)** auxiliary head predicting short/medium-horizon "plus-point" reward/novelty per direction. | Aryan + Vedesh | P0 |
| **FR-4** | **Observation-Legal Action Masking (`preprocessing.py`):** Strictly coordinate-free preprocessing supporting `(120,160)` and `(48,64)` frames; prevent door-reversing `USE` spam via cooldown mask and mask `ATTACK` when non-melee ammo is zero. | Aryan | P0 |
| **FR-5** | **Parallel `doom_env` Rollout Engine (`training/envs.py`):** Multi-worker `spawn` subprocess wrapper running `make_env` across 12–20 CPU workers on OMEN, supporting extreme domain-randomized WAD hot-reloading, worker recycling, 15s timeout recovery, and training-only scaffolding toggles. | Vedesh | P0 |
| **FR-6** | **100% Self-Evolving Recurrent PPO + Curiosity (`ppo.py`, `rnd.py`):** Pure self-evolving PyTorch recurrent PPO from scratch (GAE, truncated BPTT with `done` masking, 9 Bernoulli heads, `value_compass` & `next_rays` auxiliary losses, RND + grid-novelty intrinsic bonus) — works end-to-end without any human demos. | Vedesh | P0 |
| **FR-7** | **Optional/Synthetic BC & Offline LLM Curriculum Coach (`bc/train_bc.py`, `llm_coach.py`):** Keep `train_bc.py` optional and synthetic-compatible (never a blocker); integrate `training/llm_coach.py` (`@cf/openai/gpt-oss-120b` via `CREO_API_KEY` env var with offline deterministic fallback) for training-time PLR advising & scorecard brief summarization. | Vedesh | P1 |
| **FR-8** | **Extreme Domain-Randomized Procedural MapGen (`mapgen/`):** Generate solvable, node-built `.wad` levels across extreme scenarios (tight ambushes, open arenas, multi-branch mazes, scarce ammo, dense waves, randomized scales; `< 512` sectors, BFS-verified inside generator) plus disjoint held-out random-map suites. | Aman | P0 |
| **FR-9** | **Coordinate-Free Privileged Reward Shaper & Self-Evolving PLR (`rewards.py`, `curriculum_plr.py`):** Shape training rewards purely from relative `privileged_state()` deltas & `sector_polygons()` (zero fixed XYZ assumptions), prioritize high-learning-potential seeds via PLR, anneal scaffolding to zero, and finish with vanilla-only fine-tuning. | Aman + Vedesh | P0 |
| **FR-10** | **Automated Random-Map Scorer, Replays & Brief Package (`scorer/`, `render_replay.py`, `README.md`, `Brief.pdf`):** Continuous LOQ checkpoint evaluator mirroring `EVALUATION.md` on unseen extreme random maps + replay visualizer (Aryan), paired with root `README.md` & 2-page PDF brief (Aman). | Aryan + Aman | P0 |

---

## 5. Non-Functional Requirements

* **NFR-1 (Strict Single-Thread Determinism & Speed):** `DoomAgent` enforces `CPU_THREADS = 1` (`torch.set_num_threads(1)` + PR #2 `THREAD_ENV` from `env/doom_env/limits.py`), `MAX_VRAM_MB = 2048`, `MAX_RAM_MB = 8192`, and completes 1,050 steps (a full 4,200-tick map) in `< 25 s` wall-clock agent time (budget: `42 s`).
* **NFR-2 (Zero Extra Runtime Dependencies):** `submission/requirements.txt` remains empty (or minimal pinned pure-Python wheels if strictly unavoidable) so container build/startup cannot fail.
* **NFR-3 (Graceful Fallback Architecture):** Setting `config.use_grid = False` or `config.use_rgb = False` cleanly bypasses the grid or RGB branch without breaking checkpoint loading or `agent.py` interfaces.
* **NFR-4 (Strict Train / Eval Separation):** Held-out map seeds, `freedoom2` benchmark maps, and held-out community WADs in `eval_suite.json` are mathematically disjoint from training seeds/files.
* **NFR-5 (Audit-Ready Code Quality):** Clean module boundaries, type hints, docstrings explaining math, and unit tests in `tests/` so manual code reviewers verify originality in minutes.

---

## 6. Out of Scope

* Training purely from high-resolution RGB pixels without spatial telemetry.
* Classical graph planners (`A*`, Dijkstra, Frontier BFS) or hardcoded combat state machines inside `submission/`.
* Multi-node distributed cloud clusters (unless both OMEN and LOQ suffer hardware failure).
* Custom C++ modifications to ViZDoom binaries (all training scaffolding is done via Python wrappers, WAD flags, or ACS).

---

## 7. Milestone Gates Summary

* **H2:** H0 active — template `63b90c4` synced at repo root (`origin/main`); interfaces frozen; `validate.py` (`--docker` with `devforge-doom` fallback) green; `envs.py` > 1,000 SPS.
* **H4:** Custom self-evolving PPO smoke test learning on `two_rooms.wad` + extreme MapGen v0; local random-map scorer running on LOQ.
* **H6:** Self-evolving PPO + PLR active across extreme domain-randomized Tiers 1–4 (optional synthetic/BC warm-start tested if needed).
* **H8:** Long Training Run #1 active on OMEN (Self-Evolving PPO + Value Compass + PLR + RND); LOQ judge posting to W&B; `llm_coach.py` analyzing rollout/eval telemetry.
* **H12:** Scan-matched Occupancy Grid + Objective Slots integrated; ablation launched on LOQ.
* **H16 (Feature Freeze):** Architecture locked; Tier 1–3 progression verified on held-out maps.
* **H20 (Checkpoint Lock):** Final vanilla-only fine-tune completed, scored, Docker-validated, and promoted via Git LFS.
* **H23.5 (Submission Cutoff):** Code cleaned, 2-Page Brief PDF exported, replays captured, repo public & submitted.
