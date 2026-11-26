# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

DevForge × Scaler Hackathon 2026, Track 1 ("Can It Play DOOM?"): a zero-shot ViZDoom agent trained with recurrent PPO on procedurally generated maps. Python ≥3.13, managed with `uv`.

## Commands

```bash
uv sync                                   # install (env/ is the editable `doom-env` package)
uv run pytest                             # all tests (testpaths=tests, pythonpath=.)
uv run pytest tests/test_model_and_agent.py::test_name   # single test
uv run python validate.py                 # pre-submission check: contract, <=40 ms/step on 1 CPU thread, RAM/VRAM, repo size
uv run python -c "from submission.agent import DoomAgent; print('OK')"

# Train (offline smoke test; set CREO_API_KEY + --use-llm-coach for the LLM coach)
CREO_API_KEY="" uv run python -m training.ppo --config training/configs/ppo_curriculum.yaml --max-steps 500
# Resume / write best checkpoint into the submission
uv run python -m training.ppo --config <yaml> --resume runs/<name>/checkpoints/latest.pt --best-out submission/weights/best_agent.pt

# Score a checkpoint on the 27-map held-out suite
uv run python -m training.scorer.score_checkpoint --checkpoint submission/weights/best_agent.pt \
  --suite training/scorer/eval_suite.json --out eval_results_baseline.json --device cpu

# Live visualizer
uv run python training/visualize_obs.py --checkpoint submission/weights/best_agent.pt --watch --random-maps
```

`ppo.py` CLI overrides config YAML values (`--max-steps`, `--num-envs`, `--output-dir`, `--device`, `--curriculum {plr,staged}`, `--stage-lock N`, `--latest-interval`, `--coach-interval`, …). Configs live in `training/configs/` (`ppo_curriculum`, `ppo_staged`, `ppo_antiloop`, `ppo_fast_start`, `bc`, `eval`).

## Architecture

- **`env/doom_env/`** – the official observation/action contract (spaces, sensors, ray casting, limits, remote engine isolation). The scorer ignores this folder in our repo; `template_reference/` is a pristine copy of the template. Don't change contract behavior — `docs/CONTRACT.md` is the spec.
- **Observation pipeline** (`training/preprocessing.py`, `odometry_grid.py`): converts contract observations (64 rays, vitals, visible entities) into ego-centric `ModelInputs`, a scan-matched 32×32 occupancy grid (from yaw/velocity odometry), and an action legality mask. World XYZ must never reach the network.
- **Model** (`training/model.py`): `HybridDoomNet` — circular 1-D conv ray encoder + vitals encoder → GRU core + objective memory slots; heads: 9-action MultiBinary policy, value, 8-sector "value compass" aux, next-rays aux.
- **Training** (`training/ppo.py`): recurrent PPO (truncated BPTT, done masking) over `envs.py` `VecDoomEnv` (multiprocess), with RND curiosity + grid novelty (`rnd.py`), shaped rewards (`rewards.py`, coordinate-free, sector-based, anti-loop terms), optional BC warmstart (`bc/`).
- **Curriculum** – two mutually exclusive modes, chosen by `curriculum:` in config: `plr` (Prioritized Level Replay in `curriculum_plr.py`, optionally steered by `llm_coach.py`) or `staged` (`curriculum_stages.py` easy→hard ladder; persists `stage_state.json` in the run dir; `--stage-lock` pins one stage for parallel specialists). The LLM coach is disabled when staged. Maps come from `training/mapgen/` (4-tier procedural UDMF generator + verifier).
- **Submission** (`submission/agent.py`): `DoomAgent` (`act`/`reset`) loads `submission/weights/best_agent.pt` (Git LFS) and imports `training.model`/`training.preprocessing` by adding the repo root to `sys.path` — so changes to those modules affect the submission and must stay compatible with the checkpoint. Runs offline, single CPU thread, must never import network libs.
- **Scoring** (`training/scorer/`, `eval_suite.json`, `maps/heldout/`): held-out benchmark; results recorded in `eval_results_*.json`. Training runs and logs go to `runs/` (`metrics.csv` per run).

## Hard rules (docs/RULES.md; violations disqualify)

- Actions must come from trained network parameters: no if/else rule trees choosing actions, no A*/BFS pathfinding, no `atan2` aimbots, no hardcoded coordinates or button sequences. Observation-derived action masking and preprocessing are allowed.
- Training may modify engine/maps/game variables; evaluation uses vanilla engine on unseen maps. Train on `doom_env` ego-centric observations.
- Budgets: ≤40 ms/step, 1 CPU thread (`torch.set_num_threads(1)`), ≤2 GB VRAM, ≤8 GB RAM, repo <1 GB. Re-run `validate.py` after touching the model, preprocessing, or agent.
- LLM coach config via `.env` (`CREO_API_KEY`, `CREO_BASE_URL`, `CREO_MODEL`; see `.env.example`); it has a deterministic offline fallback.
