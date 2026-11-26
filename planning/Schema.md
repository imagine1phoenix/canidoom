# Data Contracts, Tensor Schemas & Config Specifications

> **Freeze Rule:** H0 is active and all shapes, keys, signatures, and file formats in this document are **resolved and locked**. Vedesh, Aryan, and Aman (3-person team) code independently against these exact schemas, with **zero hardcoding and zero fixed XYZ coordinate assumptions**.

---

## 1. Official `doom_env` Contract (`env/doom_env/spaces.py`)

### 1.1 Raw Observation Dictionary (`obs: dict`)
| Key | Sub-key | Shape | Dtype | Range / Values | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `rgb` | — | `(120, 160, 3)` | `uint8` | `[0, 255]` | HWC RGB24, HUD/crosshair hidden (`zeros` if `render_buffers=False`). |
| `depth` | — | `(120, 160)` | `uint8` | `[0, 255]` | HW aligned with `rgb` (`zeros` if `render_buffers=False`). |
| `vitals` | — | `(17,)` | `float32` | `(-inf, inf)` | Exact index mapping below (§1.2). |
| `rays` | `distance` | `(64,)` | `float32` | `[0.0, 1.0]` | `hit_dist / 1024.0` (`1.0` = no hit). Index `0` = straight ahead, CCW (`16`=left, `32`=behind, `48`=right). |
| `rays` | `hit_type` | `(64,)` | `int64` | `{0..5}` | `0: none`, `1: wall`, `2: door`, `3: locked_door`, `4: enemy`, `5: item`. |
| `visible_entities` | `features` | `(32, 5)` | `float32` | `(-inf, inf)` | `[dist/1024, sin(rel_angle), cos(rel_angle), rel_vel_fwd, rel_vel_right]`, nearest first (`90°` FOV & LOS). |
| `visible_entities` | `entity_type` | `(32,)` | `int64` | `{0..11}` | `0: pad`, `1: enemy_melee`, `2: enemy_hitscan`, `3: enemy_projectile`, `4: projectile`, `5: health`, `6: armor`, `7: ammo`, `8: weapon`, `9: key`, `10: barrel`, `11: other`. |
| `visible_entities` | `mask` | `(32,)` | `int8`/`bool` | `{0, 1}` | `1` = valid entity slot, `0` = zero-padded empty slot. |

### 1.2 `vitals` Index Table (`VITALS_FIELDS`)
| Index | Field | Scale Normalization in `preprocess()` |
| :--- | :--- | :--- |
| `0` | `health` | `/ 100.0` (clipped `[0, 2.0]`) |
| `1` | `armor` | `/ 100.0` (clipped `[0, 2.0]`) |
| `2` | `selected_weapon` | One-hot `(8,)` over slots `0..7` (or `/ 7.0` + embedding) |
| `3` | `selected_weapon_ammo` | `log1p(clip(x, 0, 500)) / 6.2` |
| `4..7` | `ammo_bullets`, `ammo_shells`, `ammo_rockets`, `ammo_cells` | `log1p(clip(x, 0, 500)) / 6.2` |
| `8..13` | `key_blue_card`, `key_yellow_card`, `key_red_card`, `key_blue_skull`, `key_yellow_skull`, `key_red_skull` | Already `{0.0, 1.0}` |
| `14..16` | `vel_forward`, `vel_right`, `vel_up` | `/ 30.0` (clipped `[-2.0, 2.0]`) |

### 1.3 Official Action Array (`action: np.ndarray`)
* Shape `(9,)`, `dtype=np.int8`, values strictly in `{0, 1}`:
  * `0: MOVE_FORWARD`, `1: MOVE_BACKWARD`, `2: MOVE_LEFT`, `3: MOVE_RIGHT`, `4: TURN_LEFT`, `5: TURN_RIGHT`, `6: ATTACK`, `7: USE`, `8: SELECT_NEXT_WEAPON`.

---

## 2. Preprocessed Neural Inputs (`ModelInputs`) & Memory State

*(For single-step inference `B = 1`; for PPO BPTT training flatten `(B * T, ...)` in encoders then reshape `(B, T, D)` for GRU.)*

### 2.1 `ModelInputs` Tensor Dict / Dataclass
| Field | Shape | Dtype | Construction |
| :--- | :--- | :--- | :--- |
| `rays` | `(B, 8, 64)` | `float32` | Channel 0: `distance`; Channel 1: `1.0 - distance` (proximity); Channels 2–7: 6-way one-hot of `hit_type`. |
| `ent_feats` | `(B, 32, 5)` | `float32` | `[dist, sin_a, cos_a, vel_fwd / 30.0, vel_right / 30.0]`. |
| `ent_types` | `(B, 32)` | `int64` | Raw `entity_type` indices `0..11` (`0 = pad`). |
| `ent_mask` | `(B, 32)` | `bool` | `mask > 0`. |
| `vitals` | `(B, 36)` | `float32` | `27` vitals/derived scalars (`16` normalized scalars + `8` weapon one-hot + `use_cooldown` (1) + `has_any_key` (1) + `ammo_low` (1)) concatenated with `prev_action (B, 9)` (`float32`) → `(B, 36)` `float32`. |
| `rgb_depth` | `(B, 4, 48, 64)` | `float32` | Accepts both `(120, 160)` live `doom_env` frames (`cv2.resize` bilinear `160×120 → 64×48`) and `(48, 64)` `.npz` demo frames (passed through without redundant resize); channels `0..2 = rgb / 255.0`, channel `3 = depth / 255.0`. Zeroed with prob `p = 0.25` during training (modality dropout). |
| `grid` | `(B, 6, 32, 32)` | `float32` | Egocentric `32×32` local map (`48` map units/cell, agent at center `(16, 16)` facing `+row/up`): `0: visited_decay`, `1: wall`, `2: door`, `3: locked_door`, `4: item_or_key`, `5: enemy_recent`. |
| `action_mask` | `(B, 9)` | `bool` | `True` = button allowed to be `1`; `False` = forced `0` (`logits = -1e8`). `mask[7] (USE) = False` while `use_cooldown_steps > 0`; `mask[6] (ATTACK) = False` when `selected_weapon > 1 and selected_weapon_ammo <= 0`. |

### 2.2 `OdomGridState` & `preprocess()` Signature (`training/preprocessing.py`)
Signature:
```python
def preprocess(obs: dict, state: Optional[OdomGridState] = None) -> tuple[ModelInputs, OdomGridState]
```
* When `state is None`, `preprocess()` cleanly initializes a fresh `OdomGridState` (zero grid, zero `prev_action`, `use_cooldown = 0`, initial ray history). Reset on `DoomAgent.reset()` and whenever `done == True` in vectorized rollouts:
* `grid`: `np.ndarray` `(6, 32, 32)` `float32`
* `prev_ray_dist`: `np.ndarray` `(64,)` `float32` (or `None` on step 0)
* `prev_ray_type`: `np.ndarray` `(64,)` `int64`
* `prev_action`: `np.ndarray` `(9,)` `float32` (zeros on reset; updated on step)
* `use_cooldown`: `int` (counts down `6 → 0` after `action[7] == 1` when any front ray `[-4..+4]` sees `door`/`locked_door` within `dist < 0.15`)

### 2.3 `RecurrentState` & `HybridDoomNet` Contract (`training/model.py`)
* `RecurrentState`:
  * `gru_h`: `(1, B, 256)` `float32` — GRU hidden state.
  * `obj_slots`: `(B, 4, 8)` `float32` — Learned objective memory slots (`0: key_state`, `1: locked_door_mem`, `2: exit_mem`, `3: combat_breadcrumb`), warped by `(Δx, Δy, Δθ)` + gated GRU write.
* `HybridDoomNet(model_config: dict)`:
  * `forward(batch: ModelInputs, h: RecurrentState, return_aux: bool = False) -> tuple[torch.Tensor, torch.Tensor, Optional[dict], RecurrentState]`
    * Returns `(logits_9, value, aux_preds, next_h)` where `aux_preds` is `None` when `return_aux=False` and a dict when `return_aux=True`:
      * `"next_rays"`: `(B, 64)` `float32` — predicted next-step 360° ray distances.
      * `"value_compass"`: `(B, 8)` `float32` — **Ego-Directional Value / Advantage Compass** predicting expected short/medium-horizon reward and novelty ("plus points") across 8 ego-centric angular sectors (`0` = ahead, CCW `45°`/sector), enabling directional credit assignment on arbitrary random maps without fixed world coordinates.
      * `"door_prox"`: `(B, 3)`, `"key_prox"`: `(B, 3)`, `"exit_prox"`: `(B, 3)` (or `"obj_slots"`: `(B, 4, 8)`) — ego-relative presence + `(sin, cos)` bearing predictions.
    * **Zero-entity safety:** When `ent_mask.sum(dim=-1) == 0` (no visible entities in frame), masked cross-attention / pooling safely produces zeroed entity embeddings without `NaN` (`softmax` over all-masked keys avoided via explicit zero-mask guard).

### 2.4 `RewardShaper`, `MapGen` & Offline `llm_coach.py` Contracts (`training/rewards.py`, `training/mapgen/tiers.py`, `training/llm_coach.py`)
* `RewardShaper`:
  * `reset(sector_polygons: list | None = None) -> None` — initializes per-episode visited-sector geometry and previous stats (strictly relative deltas, zero fixed world XYZ assumptions).
  * `step(priv_prev: dict, priv_now: dict, action: np.ndarray, info: dict | None = None) -> float` — computes dense training-only reward from `privileged_state()` deltas (exit, unique sector coverage, keycard pickup, door opened/unlocked, kills, damage balance, anti-spin/anti-`USE`-bounce).
* `MapGen`:
  * `sample(tier: int, knobs: dict | None = None, seed: int = 0, scaffolding_alpha: float = 0.0) -> tuple[Path, str]` — generates a node-built, BFS-verified solvable `.wad` (`< 2048` sectors) with **wide domain randomization & extreme scenarios** (tiny claustrophobic ambushes, large open arenas, multi-branch corridor mazes, scarce ammo survival, dense monster waves, randomized geometry scales), training scaffolding scaled by `scaffolding_alpha ∈ [0.0, 1.0]` (`0.0` = strictly vanilla rules), and returns `(wad_path, map_name)` (e.g. `(Path(...), "MAP01")`).
* `llm_coach.py` *(Training / Offline Tooling Only — NEVER imported by `submission/`)*:
  * Reads `CREO_BASE_URL` (`https://creo.creoai.workers.dev/v1`), `CREO_MODEL` (`@cf/openai/gpt-oss-120b`), and `CREO_API_KEY` from environment / gitignored `.env` (`.env.example` provides placeholders only). Falls back deterministically when offline or `CREO_API_KEY` is unset.
  * Exposes `advise_curriculum(telemetry: dict, current_knobs: dict) -> dict` and `summarize_evaluation(scorecard: dict) -> str` for PLR knob tuning and 2-Page Brief scorecard analysis.

---

## 3. Optional / Synthetic Trajectory Schema (`data/demos/<source>/<ep_id>.npz`)

*(Optional — the primary RL pipeline in `training/ppo.py` is **100% self-evolving without requiring human demos**; `training/bc/train_bc.py` remains compatible with optional `.npz` or synthetic self-play trajectories.)*

Compressed NumPy archive (`np.savez_compressed`), one file per recorded episode (gitignored from repo):
* `rgb`: `(T, 48, 64, 3)` `uint8` *(already resized to save disk & speed up BC loading!)*
* `depth`: `(T, 48, 64)` `uint8`
* `vitals`: `(T, 17)` `float32`
* `rays_distance`: `(T, 64)` `float32`
* `rays_hit_type`: `(T, 64)` `uint8`
* `ent_features`: `(T, 32, 5)` `float32`
* `ent_type`: `(T, 32)` `uint8`
* `ent_mask`: `(T, 32)` `uint8`
* `actions`: `(T, 9)` `uint8`
* `rewards`: `(T,)` `float32`
* `dones`: `(T,)` `bool` (`True` on final step `T-1`)
* `meta_json`: scalar UTF-8 JSON string:
  ```json
  {"wad": "tier3_s104.wad", "map": "MAP01", "tier": 3, "source": "synthetic", "exit_reached": true, "ticks": 1480, "contract_version": "1.0.0-draft"}
  ```

---

## 4. Checkpoint Schema (`.pt`)

Saved via `torch.save(payload, path)` and loaded with `weights_only=False` (or safetensors + JSON config). Both PPO checkpoints and `bc_warmstart.pt` emitted by `training/bc/train_bc.py` use this exact dictionary schema:
```python
{
    "contract_version": "1.0.0-draft",
    "git_sha": "a1b2c3d",
    "global_step": 18_500_000,
    "update_idx": 903,
    "model_config": {
        "d_model": 256,
        "gru_dim": 256,
        "num_slots": 4,
        "slot_dim": 8,
        "use_rgb": True,
        "use_grid": True,
    },
    "model_state_dict": model.state_dict(),
    "scorer_summary": {          # Populated by LOQ judge upon promotion
        "total_score": 74.8,
        "mean_act_ms": 14.2,
        "p99_act_ms": 23.9,
    },
}
```

---

## 5. Map Manifest, PLR & Scorer JSON Schemas

### 5.1 Map Manifest (`training/scorer/eval_suite.json`)
```json
[
  {
    "id": "heldout_t1_arena_01",
    "wad": "maps/heldout/t1_arena_01.wad",
    "map_name": "MAP01",
    "tier": 1,
    "episode_ticks": 2100,
    "seed": 1001,
    "source": "procedural_heldout",
    "split": "heldout"
  }
]
```
*(Rule: Training procedural seeds use `seed ∈ [100_000, 999_999]`; held-out procedural seeds use `seed ∈ [1_000, 9_999]`.)*

### 5.2 PLR Buffer Entry (In-Memory / Checkpoint Dict)
```json
{
  "level_id": "proc_t3_s240912",
  "tier": 3,
  "knobs": {"num_rooms": 5, "num_keys": 1, "enemies_per_room": 3, "ammo_mult": 1.0},
  "seed": 240912,
  "score_gae": 0.412,
  "last_seen_update": 310,
  "visits": 4
}
```

### 5.3 Scorer Output (`runs/eval/<ckpt_name>_score.json`)
```json
{
  "checkpoint": "ckpt_step_018500000.pt",
  "git_sha": "a1b2c3d",
  "total_score": 74.82,
  "tier_scores": {"1": 88.5, "2": 79.1, "3": 68.4, "4": 63.2},
  "exit_rate": 0.625,
  "progression_rate": 0.71,
  "avg_sector_coverage": 0.78,
  "avg_kills": 6.4,
  "avg_damage_taken": 38.0,
  "mean_act_ms": 14.2,
  "p99_act_ms": 23.9,
  "peak_ram_mb": 1120.0,
  "peak_vram_mb": 420.0,
  "passed_limits": true
}
```

---

## 6. W&B Metric Naming Convention

* **Throughput & Perf:** `perf/sps`, `perf/update_time_s`, `perf/act_ms_mean`, `perf/act_ms_p99`
* **Rollout Stats (Train Envs):** `rollout/ep_return`, `rollout/exit_rate`, `rollout/sector_coverage`, `rollout/kills`, `rollout/keys_picked`, `rollout/doors_unlocked`
* **Losses:** `train/policy_loss`, `train/value_loss`, `train/entropy`, `train/bc_kl_loss`, `train/rnd_loss`, `train/aux_loss`, `train/approx_kl`, `train/grad_norm`
* **Curriculum:** `plr/tier_mix_1..4`, `plr/scaffolding_alpha`, `plr/bc_kl_beta`
* **Held-Out Judge (LOQ):** `eval/total_score`, `eval/exit_rate`, `eval/sector_coverage`, `eval/tier1_score`, `eval/tier2_score`, `eval/tier3_score`, `eval/tier4_score`

---

## 7. Default Hyperparameter Configs (`training/configs/`)

```yaml
# training/configs/ppo_curriculum.yaml
seed: 42
num_envs: 16                 # Tune to (OMEN physical/logical cores - 2)
rollout_steps: 128           # 16 * 128 = 2,048 transitions per PPO update
bptt_len: 32                 # 4 chunks per env per rollout; 64 chunks total
ppo_epochs: 4
minibatch_chunks: 16         # 16 * 32 = 512 steps per minibatch
gamma: 0.99
gae_lambda: 0.95
clip_coef: 0.2
lr: 2.5e-4
max_grad_norm: 0.5
vf_coef: 0.5
ent_coef: 0.01               # Anneal 0.015 -> 0.005
rnd_coef: 0.05
grid_novelty_coef: 0.03
aux_coef: 0.10
bc_kl_beta_init: 0.0         # Self-evolving RL default (0.0 without BC warmstart; 0.20 if optional BC used)
compass_coef: 0.15           # 8-sector ego-directional value compass auxiliary loss weight
rgb_dropout_p: 0.25          # Zero RGB+depth tensor per sequence chunk with p=0.25
scaffolding_end_step: 12000000
tier_weights: [1.0, 1.5, 2.5, 3.0]
worker_max_episodes: 100     # Recycle subprocess worker every N episodes to prevent ViZDoom RAM creep
worker_timeout_s: 15.0       # Restart single hung worker if pipe call exceeds 15s
render_buffers_ratio: 0.25   # Fraction of workers with render_buffers=True (e.g. 4/16)
```

```yaml
# training/configs/bc.yaml
seed: 42
demo_dir: "data/demos"
output_checkpoint: "runs/bc/bc_warmstart.pt"
seq_len: 32                  # BPTT chunk length T=32 with done boundary masking
batch_seqs: 16
epochs: 30
lr: 3.0e-4
weight_decay: 1.0e-4
max_grad_norm: 0.5
pos_weight_use: 2.5          # Sparse USE (button 7) BCE positive weight
pos_weight_attack: 2.5       # Sparse ATTACK (button 6) BCE positive weight
rgb_dropout_p: 0.15
val_split: 0.10
device: "auto"
wandb_project: "devforge-doom"
wandb_enabled: false
model_config:
  d_model: 256
  gru_dim: 256
  num_slots: 4
  slot_dim: 8
  use_rgb: true
  use_grid: true
```

```yaml
# training/configs/eval.yaml
suite_path: "training/scorer/eval_suite.json"
output_dir: "runs/eval"
device: "cpu"
cpu_threads: 1               # Enforces torch.set_num_threads(1) & PR #2 THREAD_ENV
max_vram_mb: 2048            # PR #2 locked limit
max_ram_mb: 8192             # PR #2 locked limit
agent_ms_per_step: 40.0
latency_target_ms: 25.0      # Promotion gate (< 25 ms/step mean)
tier_weights: [1.0, 1.5, 2.5, 3.0]
deterministic_seed: 0
run_docker_validate: true
```
