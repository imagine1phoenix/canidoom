# Technology Stack & Architectural Choices

## 1. Selected Stack

| Layer | Technology | Why We Chose It | Primary Owner |
| :--- | :--- | :--- | :--- |
| **Simulation & Wrapper** | `vizdoom==1.3.3` + Official `doom_env` (`freedoom2.wad`, Skill 3, `FRAME_SKIP=4`) | Exact evaluation contract (`docs/CONTRACT.md`); runs thousands of spatial raycast ticks/sec on CPU cores. | Vedesh / All |
| **Deep Learning Core** | **PyTorch** (`torch`, CUDA 12 in training, 1-thread CPU / CUDA in eval) + `einops` | Pre-installed in official `Dockerfile`; clean dynamic graphs for variable-length entity masks, scan-matching grid warps (`grid_sample`), and custom BPTT. | Aryan & Vedesh |
| **Sensory Encoders** | Circular 1D Conv (64 rays) + Masked Multi-Head Attention (≤32 entities) + MLP (17 vitals) + Depthwise CNN (`64×48` RGB+Depth w/ modality dropout) | Exploits 360° exact geometry immediately while keeping visual projectile/texture awareness without pixel-RL sample inefficiency. | Aryan |
| **Spatial & Long-Horizon Memory** | **360° Ray Scan-Matching Odometry** + `32×32` 6-ch Egocentric Occupancy Grid + `GRU(256)` + 4 Explicit Objective Slots | Gives true room-level memory and locked-door/keycard bearing retention without needing forbidden global `(x, y, angle)` coordinates or classical `A*`. | Aryan |
| **RL Algorithm** | **100% Self-Evolving Custom Recurrent PPO** (GAE, Clipped Surrogate, Truncated BPTT `L=32`, 9 Bernoulli Heads, Obs Action Masking, 8-sector Ego Value Compass Aux Loss, Optional Decaying BC-KL) | Trains purely from scratch on arbitrary random maps without requiring human demos; avoids SB3 penalties and supports recurrent hidden states + custom action masks natively. | Vedesh |
| **Intrinsic Exploration** | **Random Network Distillation (RND)** on spatial-ray embeddings + Egocentric Grid Cell-Visit Novelty + **8-Sector Ego Value Compass** | Drives the agent out of spawn rooms and dead ends while attributing short/medium-horizon "plus points" to ego-centric directions on unseen random layouts. | Vedesh & Aryan |
| **Offline/Training LLM Coach** | **`training/llm_coach.py`** (`@cf/openai/gpt-oss-120b` via `CREO_BASE_URL` / `CREO_API_KEY` env vars, stdlib `urllib.request` + optional `openai` client + deterministic offline fallback) | Training-only curriculum/reward advisor from rollout/scorer JSON telemetry and held-out scorecard/brief summarizer. **NEVER imported or called inside `submission/agent.py`** (`--network none`). | Vedesh |
| **Optional Warm-Start BC** | **Optional / Synthetic-Compatible Behavior Cloning** (`training/bc/train_bc.py`) | Never required for training; kept optional/synthetic-compatible if trajectories are ever supplied, while primary RL pipeline is 100% self-evolving. | Vedesh |
| **Extreme Procedural Training Maps** | **Extreme Domain-Randomized PWAD/UDMF Generator** (`zdbsp`) + ViZDoom Scenarios + `freedoom2` + Community WADs | Wide domain randomization (tight ambushes, huge open arenas, multi-branch mazes, scarce ammo, dense enemies, randomized scales) with zero fixed XYZ assumptions. | Aman |
| **Curriculum Engine** | **Self-Evolving Prioritized Level Replay (PLR)** + Annealed Engine Scaffolding → Mandatory Vanilla Fine-Tune | Concentrates compute on frontier difficulty seeds (plus optional `llm_coach` knob proposals) while guaranteeing final weights run cleanly on vanilla DOOM rules. | Aman & Vedesh |
| **Checkpoint Selection & Replays** | **LOQ Automated Random-Map Held-Out Scorer** mirroring `EVALUATION.md` + `validate.py --docker` + `render_replay.py` | Selects weights by zero-shot generalization on unseen extreme random maps and verified `< 25 ms` latency (`2 GB VRAM / 8 GB RAM / 1 CPU`). | Aryan |
| **Experiment Tracking** | **Weights & Biases (`wandb`)** + local TensorBoard backup | Shared real-time dashboard across all 3 laptops for FPS, losses, and tier-by-tier held-out scores. | Vedesh & Aryan |
| **Infra, Sync & Storage** | **WSL2 Ubuntu 24.04**, **Tailscale** (`rsync`/SSH), **GitHub** (`origin/main`), **Git LFS** | Linux `spawn` stability for 16+ ViZDoom workers on Windows laptops; instant cross-OS file sync with MacBook M5; `< 1 GB` LFS compliance; `.env` gitignored. | Vedesh |
| **Testing & QA** | **`pytest`** + Official `validate.py --docker --image devforge-doom` | Fast regression checks on shapes, BPTT `done` masking, value compass, `llm_coach` mocked HTTP, reset isolation, and container limits. | Vedesh & All |
| **Brief & Diagrams** | **Root `README.md` + Markdown + Mermaid** → 2-Page PDF (Typst / Google Docs / Pandoc) | Crisp vector diagrams and real random-map ablation tables tailored for Stage 2 Desk Review and Finals. | Aman |
| **AI Coding Workflow** | **Antigravity IDE** (`Gemini Pro High` / `Claude Opus` + `Gemini Flash`) + `training/llm_coach.py` CLI (`gpt-oss-120b`) | Maximizes 3-person team velocity within quota limits while keeping human owners in full control of architecture. | All |

---

## 2. Rejected Alternatives (And Why)

| Rejected Option | Reason Rejected |
| :--- | :--- |
| **`Stable-Baselines3` (PPO / RecurrentPPO)** | Explicitly called out as low-scoring / penalized in the problem brief and `docs/RULES.md §1`; awkward dict+mask+custom-memory customization. |
| **Sample Factory (Async APPO)** | Fast ViZDoom framework, but heavy external dependency not in base `Dockerfile` (bloats `requirements.txt` and obscures bespoke architecture during code review). |
| **Custom Async IMPALA / V-Trace** | Higher theoretical throughput, but off-policy recurrent V-trace + shared-memory race conditions routinely burn 6+ hours of debugging in a 24h hackathon. |
| **Pure `120×160` RGB Pixel CNN** | Wastes ~90% of compute learning basic wall/edge filters, runs 5–8× slower on laptop GPUs, and overfits wall textures on unseen Freedoom WADs. |
| **Cloud GPU/CPU Instances (AWS / Azure)** | Local OMEN i9 HX + RTX 5060 plus LOQ RTX 4050 provide plenty of spatial+tiny-RGB throughput with zero network lag, quota delays, or S3 sync friction (Azure kept strictly as disaster backup). |
| **Classical `A*` / Frontier Planner / `atan2` Aiming** | Explicitly prohibited by `docs/RULES.md §1` — instant disqualification during top-5 manual code review. |
