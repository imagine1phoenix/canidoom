# Track 1 ("Can It Play DOOM?") — Official & Team Rulebook

> **H0 Active Rule:** Hackathon **H0** has started and `Reinforce-SST/devforge-doom-template` (`63b90c4`) is synced at the repository root (`origin/main`: `https://github.com/aryanbhojgaria/canitdoom.git`). All training, model, and submission work follows `docs/RULES.md` and `Schema.md`.

---

## 1. Official Hackathon Rules (Condensed from Template `docs/`)

### 1.1 Trained Neural Parameters Only (`docs/RULES.md §1`)
* **Every action decision at evaluation MUST originate from trained neural network weights.**
* **Strictly Banned inside `submission/`:**
  * Hand-written `if/else` rule trees or finite-state machines that select actions.
  * Classical pathfinding or graph search (`A*`, `Dijkstra`, `BFS`, wall-following scripts, frontier planners).
  * Trigonometric aimbots (e.g., computing `atan2` on `visible_entities` angles to turn/shoot directly).
  * Unmodified, off-the-shelf `Stable-Baselines3` (or equivalent plug-and-play baselines with default hyperparameters).
* **Explicitly Allowed (`docs/RULES.md §1`):**
  * Observation preprocessing and feature encoding (e.g., normalizing vitals, circular ray formatting, building an observation-derived egocentric occupancy grid via scan-matching).
  * **Action masking derived strictly from observation fields** (e.g., masking `USE` for a few steps after button-down to prevent door reversal, or masking `ATTACK` when selected non-melee weapon ammo is `0`).
  * Frame stacking and learned recurrent / slot memory modules.
  * Any training-time scaffolding (auto-opening doors, ACS scripts, infinite ammo, dense `privileged_state()` rewards) — **during training only**.

### 1.2 Training vs. Evaluation Isolation (`docs/CONTRACT.md §1`, `docs/RULES.md §2`)
* **Evaluation runs strictly on the unmodified vanilla ViZDoom `1.3.3` engine**, `freedoom2.wad`, `SKILL = 3` ("Hurt me plenty"), `FRAME_SKIP = 4`, and **unseen maps** across Tiers 1–4.
* The scorer **ignores `env/` in your repository** and mounts your repo **read-only** with **no internet access** (only `/tmp` is writable, up to 2 GB).
* `DoomAgent` receives **only** the contract observation dictionary (`rgb`, `depth`, `vitals`, `rays`, `visible_entities`). It never receives `vizdoom.DoomGame` or `privileged_state()`.
* Attempting to inspect the host, read evaluation WAD files, access the network, or exhaust shared GPU/CPU/disk resources results in **immediate disqualification**.

### 1.3 Hardware, Latency & Submission Constraints (`docs/EVALUATION.md`, `docs/SUBMISSION.md`)
* **Locked Evaluation Hardware Limits (PR #2 `4901b13`, `env/doom_env/limits.py`):** **`CPU_THREADS = 1`** (`torch.set_num_threads(1)` + `THREAD_ENV`), **`MAX_VRAM_MB = 2048` (2 GB VRAM; `≤ 1.5 GB` safe target)**, **`MAX_RAM_MB = 8192` (8 GB RAM incl. `/tmp` and `/dev/shm`)**.
* **Hard Timeouts (`env/doom_env/limits.py`):**
  * `DoomAgent.__init__`: `≤ 60 s`
  * `DoomAgent.reset`: `≤ 10 s`
  * First `act()` on each map: `≤ 10 s`
  * Subsequent `act()` calls: `≤ 5 s`
  * Average compute budget per map: **`40 ms × decision_steps`** (our internal promotion target: **`< 25 ms/step`** mean on 1 CPU thread).
  * *Penalty on failure:* If `act()` crashes, times out, or returns anything other than shape `(9,)` values in `{0, 1}`, the player stands still for the remainder of that map.
* **Repository & Weights:**
  * Public GitHub repository synced from `Reinforce-SST/devforge-doom-template` (`63b90c4`) at `https://github.com/aryanbhojgaria/canitdoom.git` (`origin/main`).
  * Total repo size (including Git LFS objects) **`≤ 1 GB`**.
  * Model weights under `submission/weights/` tracked via **Git LFS**.
  * Extra packages in `submission/requirements.txt` must use strict `pkg==version` lines (`≤ 50` lines, no URLs/flags). PR #3 (`validate.py:check_imports()`) AST-scans `submission/**/*.py` and warns on any third-party import missing from `docker/requirements.lock` + `submission/requirements.txt`. **Team policy: keep `submission/requirements.txt` empty (comments only)** by relying solely on packages pre-installed in `docker/requirements.lock`.
  * `training/` must contain our real training, mapgen, and curriculum code — **organizers manually audit the code and replays of the top 4–5 teams**.
  * `python validate.py --docker` (auto-falling back to local `devforge-doom` image per PR #3) must pass 100% cleanly on the final pushed commit before deadline, accompanied by the **2-Page Project Brief PDF**.

---

## 2. Team Engineering Rules

| Rule | Policy | Owner / Enforcer |
| :--- | :--- | :--- |
| **Branching & Direct Milestone Pushes** | Completed milestones are verified and pushed directly to `origin/main` (`https://github.com/aryanbhojgaria/canitdoom.git`) only after `uv run pytest`, `uv run python validate.py` (and `--docker` when available), and verification gates pass; optionally use `feat/<owner>-<module>` branches (`vedesh`, `aryan`, `aman`) during parallel work. | Everyone / Vedesh (OMEN) |
| **Gated `main` Hygiene** | Never push unverified code, secrets, or `.gitignore` / Git LFS violations to `main`. Zero `check_imports()` warnings or `check_committed` failures allowed. | Vedesh (OMEN) |
| **Interface Lock** | `Schema.md` tensor shapes, dictionary keys, and function signatures are locked. Any schema change requires agreement from both producer and consumer before editing. | Vedesh + Aryan + Aman |
| **Checkpoint Hygiene** | **Never commit intermediate `.pt` checkpoints to Git.** Sync intermediate checkpoints between OMEN and LOQ via Tailscale (`rsync`/`scp` into `runs/` which is `.gitignore`d). Only Aryan's promoted winner goes into `submission/weights/best_agent.pt` via Git LFS. | Aryan (LOQ) |
| **Zero Secrets in Git** | Never hardcode or commit W&B API keys, `CREO_API_KEY` (`gpt-oss-120b`), Tailscale keys, SSH keys, or `.env` files. Only `.env.example` with placeholders may be committed; pass `WANDB_API_KEY` / `CREO_API_KEY` via `.env` (gitignored) or shell env only. | Everyone |
| **2-Hour Sync Pulse** | Every 2 hours on the even hour (`H2, H4, ..., H22`), post a 3-bullet update in team chat: *(1) Done, (2) Next 2h, (3) Blockers / FPS / Score*. | Everyone (3-person team) |
| **30-Minute Escalation** | **If stuck on any bug, segfault, or setup issue for > 30 minutes, escalate immediately** in team chat and jump on a 5-minute screen share. Never burn 2 hours silently. | Everyone |
| **Time Gates** | **H16 = Feature Freeze** (no new neural layers or mapgen rewrites). **H20 = Checkpoint Lock** (final vanilla fine-tune promoted). **H23.5 = Hard Push Cutoff** (30-min buffer before official deadline). | Vedesh (OMEN) |

---

## 3. AI-Assistant Rules (Antigravity & `llm_coach.py` Usage)

1. **Active H0 Verification Rule:** H0 is active (`Reinforce-SST/devforge-doom-template` `63b90c4` synced). All generated code must conform to `Schema.md`, pass `uv run pytest` and `validate.py`, and avoid dummy/stub cheats or fixed-XYZ coordinate assumptions.
2. **One Workspace, One Owner per Module:** Coordinate edits cleanly across the 3 teammates (`Vedesh`, `Aryan`, `Aman`). Never have two teammates' AI sessions edit the same file simultaneously.
3. **Desk-Review Ownership Rule:** During Stage 2 Desk Review, judges ask live technical questions to verify understanding. **You may not merge/push AI-generated code unless you can explain how every function works** (e.g., GAE/BPTT masking, RND, & `llm_coach.py` for Vedesh; scan-matching, `value_compass`, & attention masking for Aryan; extreme domain-randomized UDMF sectors, `rewards.py`, & PLR for Aman).
4. **No Runtime AI / Network in `submission/`:** `submission/agent.py` is a pure self-contained offline PyTorch policy. **NEVER** import `openai`, `urllib.request`, or `training/llm_coach.py` inside `submission/` or `DoomAgent`, and never make network calls at inference.
5. **Mandatory Rule Audit Prompt before Merge:** Before pushing any milestone touching `submission/`, run a review check verifying zero `atan2`, zero `A*`/BFS, zero fixed-XYZ assumptions, zero network/LLM imports, and zero action-selecting `if/else` branches exist in `submission/`.
