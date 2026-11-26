# Training Pipeline (100% Self-Evolving Generalist RL)

Put your training pipeline here: environment setup, extreme domain-randomized procedural maps, reward shaping, curriculum, model code, and logs. Judges review this folder during manual review.

You may modify the engine, maps, and game variables freely **for training**. Evaluation always uses the vanilla engine through the official `doom_env` wrapper on **procedurally / randomly generated maps and extreme unseen scenarios**, so train strictly on ego-centric `doom_env` observations with **zero hardcoding and zero fixed XYZ coordinate assumptions**.

## Architecture Highlights

- **100% Self-Evolving RL Core (`training/ppo.py`, `training/rnd.py`, `training/envs.py`, `training/curriculum_plr.py`):**
  Trains purely from scratch via self-evolving Recurrent PPO (`L=32` truncated BPTT with episode `done` masking) + RND + egocentric grid novelty + Prioritized Level Replay (PLR) over extreme domain-randomized maps. Human demos / Behavior Cloning (`training/bc/train_bc.py`) are strictly optional and synthetic-compatible — never a training blocker.
- **Ego-Directional Value / Advantage Compass (`training/model.py`):**
  Alongside `next_rays` and `obj_slots`, `HybridDoomNet.forward(..., return_aux=True)` predicts `value_compass: (B, 8)` — expected short/medium-horizon reward and novelty ("plus points") across 8 ego-centric angular sectors so the agent navigates arbitrary unseen random layouts without fixed world coordinates.
- **Extreme Domain-Randomized Procedural Maps (`training/mapgen/`):**
  Generates wide-distribution random layouts (tight claustrophobic ambushes, large open arenas, multi-branch corridor mazes, scarce-ammo survival, dense enemy waves, randomized scales) with annealed training scaffolding (`scaffolding_alpha -> 0.0`) followed by vanilla-only fine-tuning.
- **Offline / Training-Only LLM Curriculum & Brief Coach (`training/llm_coach.py`):**
  Optional training-time telemetry advisor and scorecard summarizer querying an OpenAI-compatible `@cf/openai/gpt-oss-120b` endpoint via environment variables (`CREO_BASE_URL`, `CREO_MODEL`, `CREO_API_KEY` — see `.env.example`, `.env` is `.gitignore`d), with deterministic offline fallback when `CREO_API_KEY` is unset. **`submission/agent.py` runs completely offline (`--network none`) and NEVER imports `openai` or makes network calls.**
