# Hardware, Software & Pre-Event Setup Requirements

> **H0 Active:** Repo is synced from `Reinforce-SST/devforge-doom-template` (`63b90c4`) on `origin/main` (`https://github.com/aryanbhojgaria/canitdoom.git`) for the **3-person team (Vedesh, Aryan, Aman)**.

---

## 1. Team Hardware & OS Allocation (3-Person Roster)

| Teammate | Machine | Specs | Target OS / Runtime | Primary Role |
| :--- | :--- | :--- | :--- | :--- |
| **Vedesh** | HP OMEN 16 | Intel Core i9-14900HX (24c/32t), RTX 5060 8 GB, Windows 11 | **WSL2 Ubuntu 24.04** + Docker Desktop | Primary RL Trainer (`envs.py`, self-evolving `ppo.py`, `rnd.py`, `llm_coach.py`, optional `bc/`), Repo Lead & `pytest` suite (`tests/`) |
| **Aryan** | Lenovo LOQ | Intel Core i5-13th Gen, RTX 4050, Windows 11 | **WSL2 Ubuntu 24.04** + Docker Desktop | `preprocessing.py`, `odometry_grid.py`, `model.py` (8-sector `value_compass` + objective slots), `submission/agent.py`, Held-Out Random-Map Scorer (`scorer/`) & `render_replay.py` |
| **Aman** | MacBook Air M5 | Apple Silicon M5, macOS | **macOS native (`arm64`)** + SSH to OMEN via Tailscale | Extreme Domain-Randomized Procedural MapGen (`mapgen/`), Dense Reward Shaper (`rewards.py`), Self-Evolving PLR (`curriculum_plr.py`), Root `README.md` & 2-Page Brief PDF |

---

## 2. Software Dependencies Breakdown

### 2.1 Pre-Installed in Official Evaluation Image (`Dockerfile` / `docker/requirements.in`)
These packages are **guaranteed present at evaluation** (Python `3.13` inside Ubuntu `24.04` `/opt/venv`). **`submission/agent.py` must ONLY import from these so `submission/requirements.txt` stays empty:**
* Python `3.13`
* `vizdoom==1.3.3` (includes `freedoom2.wad`)
* `gymnasium>=1.0`
* `numpy>=2.0`
* `scipy`
* `opencv-python-headless` (`cv2` — used for fast `64×48` RGB downsampling)
* `torch` (CUDA 12 wheel in Docker; supports both `"cuda"` and `"cpu"`)
* `einops`
* `jax[cuda12]`, `flax`, `optax` *(unused by us; we standardize on PyTorch)*
* `tqdm`, `tensorboard`
* `doom_env` (official wrapper installed from `/opt/doom_env_src`)

### 2.2 Training & Development Extras (Root `pyproject.toml` / Local `uv` Env Only — Never in `submission/agent.py`)
* **Python packages (installed automatically by root `uv sync`):**
  * `torch>=2.0`, `pyyaml>=6.0`, `opencv-python-headless>=4.8`, `scipy>=1.11`, `einops>=0.7`, `tqdm>=4.66`
  * `[dependency-groups].dev`: `pytest>=8` (unit/integration test runner), `wandb>=0.18` (shared training & eval dashboard)
  * *Note:* `training/llm_coach.py` uses Python stdlib `urllib.request` (+ optional `openai` client if installed) so `submission/requirements.txt` stays empty. Optional packages (`omgifol`, `pygame`) can be installed locally on Mac/WSL2 if needed.
* **System / CLI utilities:**
  * `uv` (`>=0.9`) — fast Python 3.13 environment & lockfile manager (`UV_PROJECT_ENVIRONMENT=/root/.venvs/devforge-doom` on WSL2 ext4)
  * `git` + `git-lfs` — version control and model weight storage
  * `docker` (Docker Desktop on OMEN & LOQ) — official container validation (`devforge-doom`)
  * `tailscale` — zero-config mesh VPN linking all 3 laptops (`omen`, `loq`, `aman-m5`)
  * `zdbsp` (`apt install zdbsp` on Ubuntu 24.04, with `libglib2.0-0t64` replacing `libglib2.0-0`) — fast BSP node builder so procedural UDMF/Doom WADs load cleanly in ViZDoom

---

## 3. Pre-Event Setup Checklist & Verification Commands

### 3.1 Vedesh (OMEN) & Aryan (LOQ) — Windows + WSL2 + CUDA + Docker
1. **Install WSL2 Ubuntu 24.04 & System Packages:**
   ```bash
   wsl --install -d Ubuntu-24.04
   # Inside Ubuntu-24.04 terminal:
   sudo apt update && sudo apt install -y ca-certificates curl wget git git-lfs libgl1 libglib2.0-0t64 libsdl2-2.0-0 build-essential cmake libsdl2-dev zdbsp tmux htop rsync
   git lfs install
   ```
2. **Verify CPU Cores & NVIDIA GPU inside WSL2:**
   ```bash
   nproc                     # Vedesh: note exact logical core count to size num_envs!
   nvidia-smi                # Must show RTX 5060 (OMEN) / RTX 4050 (LOQ)
   ```
3. **Install `uv` & Python 3.13:**
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.13
   ```
4. **Install Docker Desktop (Windows) & Pre-Build Official Eval Image:**
   * Enable *Settings → Resources → WSL Integration → Ubuntu-24.04*.
   * Clone template to `/tmp/setup-check` (delete before H0!) and cache the Docker build:
   ```bash
   git clone https://github.com/Reinforce-SST/devforge-doom-template.git /tmp/setup-check
   cd /tmp/setup-check
   docker build -t devforge-doom .
   uv sync
   uv run python validate.py --docker --image devforge-doom
   uv run python -c "import torch, vizdoom; print('CUDA:', torch.cuda.is_available(), 'ViZDoom:', vizdoom.__version__)"
   cd ~ && rm -rf /tmp/setup-check
   ```
5. **Enable SSH Server on OMEN WSL2 + Tailscale:**
   ```bash
   sudo apt install -y openssh-server && sudo service ssh start
   curl -fsSL https://tailscale.com/install.sh | sh
   sudo tailscale up
   tailscale ip -4           # Share this IP with Aryan & Aman
   ```
6. **Login to Weights & Biases & Configure `.env` (gitignored):**
   ```bash
   uv tool run wandb login   # Or export WANDB_API_KEY in ~/.bashrc
   cp .env.example .env      # Fill CREO_API_KEY locally if using training/llm_coach.py (NEVER commit .env!)
   ```

### 3.2 Aman (MacBook Air M5) — macOS Setup
1. **Install Homebrew Utilities, `uv`, Git LFS & Tailscale:**
   ```bash
   brew install cmake sdl2 boost git git-lfs tailscale
   git lfs install
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv python install 3.13
   ```
   *(Check if `brew install zdbsp` or building `zdbsp` from source works on macOS; alternatively build pure classic Doom binary lumps in Python or test `zdbsp` node builds over SSH on OMEN.)*
2. **Sync Repo & Smoke-Test `vizdoom==1.3.3` on Apple Silicon:**
   ```bash
   git clone https://github.com/aryanbhojgaria/canitdoom.git
   cd canitdoom
   uv sync
   uv run pytest
   uv run python validate.py
   uv run python -c "import torch, vizdoom; print('MPS:', torch.backends.mps.is_available(), 'ViZDoom:', vizdoom.__version__)"
   ```
3. **Verify Tailscale SSH into Vedesh's OMEN:**
   ```bash
   tailscale status
   ssh <vedesh-user>@<omen-tailscale-ip> "nvidia-smi"
   ```
