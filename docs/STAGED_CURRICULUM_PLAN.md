# Staged Curriculum Restart — 4-Person Specialist & Combiner Workflow

**Why we restarted cleanly:**
1. The original `ppo_curriculum.yaml` used `tier_weights: [1.0, 1.5, 2.5, 3.0]` (~75% Tier 3–4 hard maps from step 0) and aggressive PLR (`temperature: 0.1, staleness_coef: 0.1`), which trapped replay onto the hardest maps with `exit=0%`.
2. Resuming from the old `806.9k` checkpoint carried over its ingrained spinning/looping habit. Starting **fresh** (`seed: 42`, zero `--resume` of the looping weights) on Stage 0 (`single_map`) reaches `100%` exits immediately and lets every teammate's weights stay mutually compatible for merging.

**5-Stage Progressive Ladder** (`training/curriculum_stages.py`, `training/configs/ppo_staged.yaml`):

| Stage | `--stage-lock` / `--stage-max` | Maps | Advances when rolling exit-rate ≥ |
|---|---|---|---|
| **0: `single_map`** | `0` | ONE fixed map (`seed=123457`), `0` enemies (learn: reach exit fast) | `80%` |
| **1: `easy`** | `1` | Tier 1, random seeds, ~1 weak enemy/room, generous ammo | `60%` |
| **2: `medium`** | `2` | Tiers 1–2, unlocked doors, ~1.5 enemies/room | `45%` |
| **3: `hard`** | `3` | Tiers 2–4, keycards & multi-room exploration | `30%` |
| **4: `random`** | `4` | All Tiers 1–4 equally mixed (zero-shot generalization) | Final stage |

Within stages `>= 1`, balanced replay (`temperature: 0.7`, `staleness_coef: 0.35`) ensures the buffer stays diverse. Training seeds remain strictly in `[100_000, 999_999]` (held-out `[1_000, 9_999]` untouched). `best_out` is blank so nothing overwrites `submission/weights/best_agent.pt` prematurely.

---

## Division of Work (All 4 Train Fresh with `seed: 42` — NO `--resume`!)

First, every teammate pulls the latest code:
```bash
git pull origin main && git lfs pull
```

| Person | Hardware | Role / Stages | Exact Command |
|---|---|---|---|
| **Team Member 1** | RTX GPU | Full backbone (`0 → 1 → 2 → 3 → 4`) + final merge/eval | `python -u -m training.ppo --config training/configs/ppo_staged.yaml --device cuda --output-dir runs/staged_main --use-llm-coach` |
| **Team Member 2** | Mac | Specialist 1: **Single Map → Easy → Medium** (`0 → 2`) | `python -u -m training.ppo --config training/configs/ppo_staged.yaml --stage-start 0 --stage-max 2 --num-envs 8 --device cpu --output-dir runs/stage_specialist1` |
| **Team Member 3** | Mac | Specialist 2: **Single Map → Medium → Hard** (`0 → 3`) | `python -u -m training.ppo --config training/configs/ppo_staged.yaml --stage-start 0 --stage-max 3 --num-envs 8 --device cpu --output-dir runs/stage_specialist2` |
| **Team Member 4** | Mac | Specialist 3: **Further Moving → Random** (`0 → 4`, faster step cap) | `python -u -m training.ppo --config training/configs/ppo_staged.yaml --stage-start 0 --stage-max 4 --stage-max-steps 150000 --num-envs 8 --device cpu --output-dir runs/stage_specialist3` |

Everyone commits `runs/<name>/metrics.csv` (small) and sends their best `runs/<name>/checkpoints/best.pt` (or `latest.pt`) to Team Member 1 when ready. **Do not push directly over `submission/weights/best_agent.pt`.**

---

## Combine & Promote (`training/combine_specialists.py`)

Once checkpoints are collected, Team Member 1 runs the automated combiner on the RTX GPU:
```bash
python -m training.combine_specialists \
  --checkpoints runs/staged_main/checkpoints/best.pt runs/stage_specialist1/checkpoints/best.pt runs/stage_specialist2/checkpoints/best.pt runs/stage_specialist3/checkpoints/best.pt \
  --suite training/scorer/eval_suite.json \
  --finetune-steps 16384 \
  --device cuda \
  --out runs/combined/best_combined.pt \
  --promote-if-above 504.13
```

What this does:
1. Scores every existing specialist on the 27-map held-out suite (`eval_suite.json`).
2. Builds a softmax score-weighted parameter soup (`soup_raw.pt`).
3. Runs a Stage-4 (`random` all-tiers) consolidation fine-tune (`16,384` steps) on CUDA so the GRU and actor-critic heads co-adapt (`soup_finetuned`).
4. Compares `best_single`, `soup_raw`, and `soup_finetuned`, writes `runs/combined/combine_report.json`, and promotes to `submission/weights/best_agent.pt` **only** if it beats our current `504.13` submission (then run `python validate.py`).
