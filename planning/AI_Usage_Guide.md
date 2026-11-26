# Antigravity & `gpt-oss-120b` Coach Playbook (3-Person Team: 1 Paid Pro + 2 Free + Local LLM Coach)

## 1. Quota & Tooling Strategy Across the 3-Person Team

* **Vedesh (OMEN) has Paid Google AI Pro:** Highest quota for **Gemini Pro (High thinking)** and **Claude Opus** in Antigravity.
* **Aryan (LOQ) & Aman (M5) are on Free Tiers:** Limited daily/hourly turns on heavy reasoning models (**Gemini Pro High / Claude Opus**), generous turns on **Gemini Flash**.
* **Shared Offline/Training LLM Coach (`training/llm_coach.py` — `@cf/openai/gpt-oss-120b`):** Configured via `.env` (`CREO_BASE_URL`, `CREO_MODEL`, `CREO_API_KEY` — see `.env.example`; `.env` is strictly `.gitignore`d and never committed). Gives Vedesh, Aryan, and Aman a local CLI helper (`python -m training.llm_coach ...`) to analyze rollout/eval scorecards, propose bounded PLR knob updates, and draft 2-Page Brief insights without burning Antigravity Pro turns. **Never imported by `submission/agent.py`.**

### Golden Rule of Model Selection
1. **Default to `Gemini Flash` (~75–80% of turns on Free accounts):** Use Flash for boilerplate, data loaders, config YAML files, `pytest` unit tests, WAD lump packing tables, W&B logging wrappers, and Markdown/README drafting.
2. **Save `Gemini Pro (High thinking)` / `Claude Opus` for Hard Algorithmic Spikes (~20–25% of turns):**
   * **Vedesh (Paid):** Keep **Gemini Pro (High)** or **Claude Opus** on for `ppo.py` (self-evolving GAE + truncated BPTT hidden-state masking + 8-sector `value_compass` loss), debugging RL training instabilities, and reviewing milestones before pushing to `origin/main`.
   * **Aryan (Free — ration carefully!):** Use **Gemini Pro (High)** / **Claude Opus** *only* for `odometry_grid.py` (circular scan-matching + affine warp) and `model.py` (masked entity attention + 8-sector ego-directional value compass + slot updates). Switch immediately to **Gemini Flash** for `score_checkpoint.py`, `render_replay.py`, latency scripts, and tests.
   * **Aman (Free — ration carefully!):** Use **Gemini Pro (High)** *only* for designing extreme domain-randomized Tier 1–4 layout topology (`tiers.py`), `curriculum_plr.py`, and polishing the **2-Page Project Brief PDF**. Use **Gemini Flash** for WAD byte/UDMF builders, texture lists, `rewards.py`, and `README.md`.
3. **Cross-Team Escalation Trick:** If Aryan or Aman hits a Free-tier Pro rate limit on a gnarly math/RL bug, push/share the branch so **Vedesh pulls it on OMEN and runs `/boost`**, or run `training/llm_coach.py` (`gpt-oss-120b`) for telemetry/code analysis!

---

## 2. Task-to-Model Quick Reference Table

| Module / Task | Owner | Recommended Model | Recommended Tool / Command |
| :--- | :--- | :--- | :--- |
| `training/envs.py` subprocess workers | Vedesh | Gemini Flash → Pro review | `/plan` |
| `training/ppo.py` (self-evolving GAE + BPTT + `value_compass`) & `rnd.py` | Vedesh | **Gemini Pro (High) / Claude Opus** | `/boost` |
| `training/llm_coach.py` (`gpt-oss-120b` advisor & CLI) | Vedesh | Gemini Flash | `/plan` |
| `tests/*.py` comprehensive RL & unit test suites | Vedesh | **Gemini Flash** | `/plan` |
| `training/odometry_grid.py` (360° scan-match + warp) | Aryan | **Gemini Pro (High) / Claude Opus** | `/boost` |
| `training/model.py` (`value_compass` + slots) & `submission/agent.py` | Aryan | **Gemini Pro (High)** | `/plan` |
| `training/scorer/score_checkpoint.py` & `render_replay.py` | Aryan | **Gemini Flash** | `/plan` |
| Continuous LOQ checkpoint scoring every 30m | Aryan | Gemini Flash + `llm_coach.py` | `/schedule` |
| `training/mapgen/` (extreme domain-randomized Tier 1–4 maps) | Aman | Pro for graph logic, **Flash** for WAD I/O | `/plan` then `/goal` |
| Stress-generate & validate 1,000 solvable random WADs | Aman | **Gemini Flash** | `/goal` |
| `training/rewards.py` & `curriculum_plr.py` | Aman | Gemini Pro (High) / Flash | `/plan` |
| Root `README.md` & 2-Page Brief PDF | Aman | **Gemini Pro (High)** + `llm_coach.py` | `/plan` |
| Diagnosing stalled PPO entropy / reward collapse | Vedesh | **Gemini Pro (High)** + `llm_coach.py` | `/boost` |

---

## 3. Antigravity Slash-Command Playbook

* **`/plan` — Use Before Every New Module:** Type `/plan` before writing `ppo.py`, `model.py`, `tiers.py`, or `score_checkpoint.py`. Reviewing a 10-line plan catches tensor-shape mismatches (`Schema.md`) before wasting tokens rewriting 300 lines of code.
* **`/boost` — Reserve for the 3 Hardest Problems:** Triggers deep multi-angle reasoning and rigorous verification. Use it **only** on:
  1. `training/ppo.py` (recurrent BPTT mask, GAE alignment, & 8-sector ego value compass loss),
  2. `training/odometry_grid.py` (coordinate-frame signs on scan-matching + `warpAffine`), and
  3. Diagnosing mid-event RL plateaus (paste W&B metrics on Vedesh's paid Pro account or query `llm_coach.py`).
* **`/goal` — Unattended Thorough Execution:** Ideal when you have a clear automated pass criterion:
  * *Aman:* `"/goal Generate 200 extreme random maps per tier (800 total), run BFS solvability + doom_env reset/step smoke test on every map, and fix any sector/node bugs until 100% pass."`
  * *Aryan:* `"/goal Profile DoomAgent.act under torch.set_num_threads(1) and optimize until mean latency < 18ms and validate.py --docker --image devforge-doom passes."`
* **`/schedule` — Live Hackathon Automation:**
  * *On Aryan's LOQ (H6–H20):* `"/schedule Every 30 minutes check runs/incoming/ for new checkpoints from OMEN, run score_checkpoint.py + validate.py, log to W&B, and alert if a new best total_score is achieved."`
  * *On Vedesh's OMEN (overnight H8–H16):* `"/schedule Every 60 minutes inspect the latest training log / W&B metrics for NaNs, worker crashes, or collapsed entropy (< 0.3) and summarize run health."`
* **`/learn` — Persist Tricky Machine Quirks:** Once WSL2 + `vizdoom==1.3.3` + `zdbsp` paths or `validate.py --docker` flags are working on OMEN/LOQ, run `/learn` so Antigravity remembers exact binary paths and `uv run` invocations across every new chat session!
* **`/compact` — Save Context & Quota:** Run `/compact` whenever a coding thread exceeds ~25–30 turns so responses stay fast and Free-tier token limits last all 24 hours.
