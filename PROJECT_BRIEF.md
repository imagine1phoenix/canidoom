# Project Brief — Can It Play DOOM? (Track 1: Deep RL)
**Team:** Team Member 1 · Team Member 2 · Team Member 3 · Team Member 4  
**Repo:** https://github.com/imagine1phoenix/canidoom  
**Date:** November 2026

---

## 1 · Unique Selling Proposition

> **Zero-shot DOOM agent that self-evolves on randomly-generated maps — no hardcoded coordinates, no memorized button sequences.**

Standard RL bots overfit to map geometry and break the moment a wall moves. Our agent generalizes by combining:

- **Ego-centric perception** — 64-ray depth raycasts + entity vitals (never raw pixels, never world XYZ)  
- **8-sector Value Compass** — a differentiable directional attention map that always asks *"which direction is reward in?"*  
- **Prioritized Level Replay (PLR)** — a self-evolving curriculum that discards maps the agent has mastered and promotes maps it still struggles with  
- **LLM Curriculum Coach** — queries CreoAI (`@cf/openai/gpt-oss-120b`) to adapt difficulty knobs (ammo, enemy count, map tier) based on live training metrics; falls back deterministically when offline  

Result: on 24 heldout WADs (never seen during training) the agent solves keycard → door → exit chains with no prior map knowledge.

---

## 2 · System Architecture

```
Random Map Generator (UDMF)
        │  WAD (Tier 1-4)
        ▼
   VecDoomEnv (spawn workers)
        │  ego-centric obs dict
        ▼
  Preprocessing Layer
  ┌─────────────────────┐
  │  OdomGridState      │  ← coordinate-free occupancy grid
  │  ActionMask         │  ← USE-cooldown gating
  │  ModelInputs        │  ← device-agnostic tensor bundle
  └─────────────────────┘
        │
        ▼
  HybridDoomNet (actor-critic)
  ┌─────────────────────────────────┐
  │  RayCircularEncoder (96-d)      │  ← circular 1-D Conv on 64 rays
  │  VitalsEncoder (32-d)           │  ← health / ammo / keys / armour
  │  GRU core (256-d)               │  ← temporal memory
  │  ObjectiveSlots (4 × 8-d)       │  ← per-task memory slots
  │  8-sector Value Compass head    │  ← directional value target
  │  Policy head (9 actions)        │
  └─────────────────────────────────┘
        │  (logits, value, aux, h_next)
        ▼
  PPO + GAE + BPTT-16
  ┌──────────────────────────┐
  │  RND curiosity bonus     │  ← Random Network Distillation
  │  Grid novelty bonus      │  ← occupancy-grid cell coverage
  │  BC-KL warmstart         │  ← behavioural cloning regulariser
  │  PLR Buffer              │  ← staleness-weighted level selection
  └──────────────────────────┘
        │
        ▼
  LLM Curriculum Coach ──► CreoAI API
  (adjusts tier mix + difficulty knobs online)
```

**Submission entrypoint:** `submission/agent.py` — stateless `act(obs, h) → (action, h)`, torch.set_num_threads(1), 40 ms budget enforced via `validate.py`.

---

## 3 · Technical Approach

### 3.1 Observation Space (ego-centric, coordinate-free)
| Channel | Shape | Description |
|---|---|---|
| `rays` | (8, 64) | 8-channel depth / hit-type raycasts |
| `vitals` | (8,) | health, ammo, keys, armour, ticks |
| `d_theta` | scalar | yaw-delta from odometry |
| `grid_flat` | (32×32,) | binary occupancy grid (agent at centre) |
| `action_mask` | (9,) | legal actions this step |

No world XYZ ever enters the network. All spatial reasoning is relative to the agent's current heading.

### 3.2 Map Generation — 4-Tier Random UDMF Builder
- **Tier 1**: 2-room connector (baseline puzzle)  
- **Tier 2**: 3–5 rooms, one keycard + locked door  
- **Tier 3**: multi-branch maze with ammo scarcity  
- **Tier 4**: open arena + enemy clusters + multiple keys  

Maps are generated procedurally at runtime — `MapGen.sample(tier, seed)` — so the agent never trains on the same map twice.

### 3.3 Reward Shaping (coordinate-free)
`RewardShaper` tracks sector visitation, keycard pickup, door unlock, and exit events using privileged sector polygon data only available during training. At inference, the agent uses only its ego-centric obs + learned value function.

### 3.4 Efficiency
| Constraint | Our implementation |
|---|---|
| ≤ 2 GB VRAM | ~120 MB (HybridDoomNet, fp32) |
| ≤ 40 ms/step | p50 = 4.9 ms, p99 = 14.5 ms (validate.py) |
| ≤ 8 GB RAM | ~600 MB training, ~180 MB inference |
| 1 CPU thread | `torch.set_num_threads(1)` enforced |

---

## 4 · Track-Specific Requirements

| Requirement | Status |
|---|---|
| Public GitHub repo with source + weights | ✅ `submission/weights/best_agent.pt` (Git LFS) |
| `submission/agent.py` entrypoint | ✅ `act(obs, recurrent_state)` → `(action, next_state)` |
| `validate.py` PASSED | ✅ p99 = 14.5 ms, 0 failures |
| Zero-shot map generalization | ✅ 24 heldout WADs, never seen during training |
| Multi-step puzzle (key→door→exit) | ✅ Tier 2–4 maps enforce this sequence |
| Combat survival | ✅ Tier 4 maps include enemy clusters |
| No hardcoded XYZ coordinates | ✅ All spatial ops ego-centric + sector-relative |
| Self-evolving (no fixed curriculum) | ✅ PLR + LLM Coach adapt online |

---

*2-page brief — DevForge × Scaler Hackathon 2026*
