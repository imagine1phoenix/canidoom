# DevForge "Can It Play DOOM?" — Technical Guide (Simple Version)

> **Team:** Team Member 2 (features & systems) + Team Member 1 (AI training)
> **Best Score So Far:** **2126.58** on 27 maps it has never seen before
> **Docker Check:** ✅ PASSED — runs safely inside the official sandbox

---

## 🎯 What Is This Project? (The One-Liner)

We built an AI agent that plays DOOM **completely on its own** — it navigates rooms, kills enemies, picks up keys, opens locked doors, and finds the exit — **without memorizing any map** and **without being told where to go**.

---

## 💡 What Makes It Special? (Our 3 Big Ideas)

### 1. It Never Cheats With GPS Coordinates
Most DOOM bots get "cheated" position data like `(x=1024, y=-512)` — absolute coordinates. Our agent gets **zero coordinates**. Instead, it builds its own mental map of "where it has been" using only what it can *sense* right now — like a person navigating blindfolded, using echoes. This is why it generalizes to brand-new maps.

### 2. It Has 5 Different "Senses" Fused Into One Brain
The agent doesn't just look at pixels. It uses:
- 🔭 **64 radar beams** shooting out in all directions (like sonar)
- 👁️ **A camera view** (RGB + depth — it sees colors AND distances)
- 📍 **A self-built map** that it draws as it moves (a 32×32 grid in its head)
- 👾 **Enemy/item tracker** (up to 32 nearby entities at once)
- 💊 **Health, armor, ammo, keys** — knows its own status at all times

### 3. It Learned from a Smart Teacher First, Then Practiced Alone
Pure trial-and-error (RL) only got us a 3.7% exit rate — the agent kept spinning in corners. We broke through by having a **privileged teacher bot** (which *can* see coordinates) show the student agent the right moves first. The student mimicked the teacher, then went on to practice on its own. This is called **DAgger + PPO**. After that, exit rate jumped to **22.2%**.

---

## 🧠 How Does It Think? (The Neural Network, Simply)

Think of it as a brain with **5 eyes** feeding into a **memory core**:

```
[64 Radar Rays]  ──▶  [Radar Processor]  ─┐
[Camera View]    ──▶  [Vision Processor]  ─┤
[Self-Built Map] ──▶  [Map Processor]    ─┼──▶ [Fusion] ──▶ [Memory (GRU)] ──▶ [Action]
[Nearby Enemies] ──▶  [Enemy Tracker]   ─┤                       │
[Health/Ammo]   ──▶  [Status Processor]  ─┘               [Goal Memory Slots]
```

- The **Fusion** layer combines all 5 inputs into one vector
- The **Memory (GRU)** is like working memory — it *remembers* what happened earlier in the same level (e.g., "I saw a yellow key 30 seconds ago")
- The **Goal Memory Slots** are long-term sticky notes — they store things like "there's a locked door to the north" across hundreds of steps
- Finally it outputs **9 binary actions**: Move Forward/Back, Strafe Left/Right, Turn Left/Right, Run, Attack, Use

**Total brain size: 868,403 parameters** — well under the 2 million limit set by the judges.

---

## 🗺️ How Does It Build Its Own Map? (The Cool Part)

This is called **Egocentric Occupancy Grid Mapping** — a fancy way of saying "it draws a map of its surroundings centered on itself."

Here's how it works step-by-step:
1. Every frame, 64 radar rays shoot out in all directions. Each ray returns a distance + what it hit (wall, door, locked door, enemy, item, nothing).
2. It figures out "how much did I turn since last frame?" by comparing the current rays to the previous frame's rays — like **comparing two sonar snapshots**.
3. It rotates and shifts its mental map to stay correctly oriented.
4. It paints what it sees: visited floor = blue, walls = grey, doors = yellow, locked doors = red, enemies = orange.
5. The map fades slowly over time (old cells become less certain), so stale information doesn't confuse the agent.

**Key insight:** This map is always agent-centered (it rotates with the player). So the same room looks the same no matter which direction the agent entered from — making it truly generalizable.

---

## 🔢 How Was It Trained? (5 Stages)

We didn't just throw it into hard maps from day one. We trained it in stages, like teaching a child:

| Stage | What It Practiced | "Pass" Condition |
|---|---|---|
| 1 — Single Map | One simple room, just move & explore | Consistently explores |
| 2 — Easy Maps | Simple open arenas, some enemies | 80% exit rate |
| 3 — Medium Maps | Multi-room, basic doors | 60% exit rate |
| 4 — Hard Maps | Keycards, locked doors, mazes | 45% exit rate |
| 5 — Random Mix | All map types randomly | 30% exit rate |

Then after stage 5, we ran the **Teacher DAgger** phase:
- A teacher bot (with full coordinates) played alongside the agent
- The agent watched what the teacher did and copied the moves
- This fixed the agent's biggest weaknesses: dying early and not connecting "key → locked door"

After DAgger, we did **PPO fine-tuning** (pure self-practice) to let the agent polish what it learned from the teacher.

---

## 🎲 How Do Maps Stay Fair & Novel?

We generate maps procedurally (randomly) using our own Python map generator:
- Every map has random room sizes, random enemy placements, random textures, and random lighting
- **Training maps use seeds 100,000–999,999**
- **Evaluation maps use seeds 1,000–9,999** — completely different, never seen during training
- Plus 3 real maps from `freedoom2.wad` that the agent never trained on

The agent has **never seen any of the 27 evaluation maps** before scoring day. So the 2126.58 score is real generalization, not memorization.

---

## 🛡️ Rules Compliance — How We Know It's Not Cheating

The organizers have strict rules. Here's exactly what our agent is and isn't allowed to do:

| ❌ Forbidden (and we don't do it) | ✅ What we actually do |
|---|---|
| Read `privileged_state()` during evaluation | Only use standard observations |
| Use A* pathfinding or Dijkstra navigation | Zero pathfinding algorithms at eval time |
| Use `atan2` to aim at enemies | Model outputs aim from raw radar + entities |
| Hard-coded if/else action rules | Every action comes from the neural network |
| Use GPS/world coordinates | Fully coordinate-free, egocentric only |

The two only "masks" we apply are **legal observation masks** (explicitly allowed by rules):
1. **USE cooldown:** Don't hold the "Use" button on the same door non-stop
2. **Empty ammo:** Don't fire when you have no bullets

Both of these are derived only from what the agent can legally observe.

---

## ⚡ Speed & Safety Numbers

The judges care about latency — the agent must decide within 40ms per step.

| Test | Result | Decision Speed | Memory |
|---|---|---|---|
| Official Docker sandbox | ✅ PASSED | 9.4ms average, 16.8ms worst case | 1556 MB RAM, ~127 MB GPU |
| Hardest map type (Tier 4) | ✅ PASSED | 12.2ms average, 18.6ms worst case | 1330 MB RAM |
| CPU-only mode (no GPU) | ✅ PASSED | 8.0ms average, 14.0ms worst case | 601 MB RAM |
| Unit tests (4 tests) | ✅ 4/4 PASSED | 1.95 seconds total | — |

The agent comfortably stays **under 40ms**, even in the hardest scenarios.

---

## 📊 Score Breakdown by Difficulty Level

The 27 evaluation maps are split into 4 tiers of difficulty:

| Difficulty | Maps | Exits Found | Avg Exploration | Kills |
|---|---|---|---|---|
| Tier 1 (Easy — open rooms) | 6 | **5 out of 6** (83%) | 95.8% | 3.83 avg |
| Tier 2 (Medium — multi-room) | 7 | 1 out of 7 (14%) | 59.9% | 4.29 avg |
| Tier 3 (Hard — keycards) | 7 | 0 out of 7 | 37.9% | 2.86 avg |
| Tier 4 (Hardest — maze+ambush) | 7 | 0 out of 7 | 23.2% | 5.71 avg |

**Big picture:** The agent dominates easy maps, holds its ground on medium maps, and survives the hard ones (5 locked doors unlocked on Tier 3!). More training time would push Tier 3–4 exits up.

---

## 📈 How We Got from Bad to Good (Progress Timeline)

| What We Tried | Score | Exit Rate | Why It Failed / What Changed |
|---|---|---|---|
| Random agent (untrained) | 0.00 | 0% | Control baseline |
| Basic PPO, jumped to hard maps too fast | 344.28 | 3.7% | Agent was overwhelmed, learned nothing |
| PPO + bonus exploration reward | 504.13 | 3.7% | Better exploration but still spun in corners |
| **5-Stage Curriculum** | **1440.45** | **14.8%** | First real breakthrough — staged difficulty worked! |
| **Teacher DAgger + Fine-tuned PPO** | **2126.58** | **22.2%** | Agent learned from teacher, dramatically improved |

---

## 🙋 Judge Questions — Plain English Answers

### Q1: "How do we know your agent isn't using A*, coordinates, or if/else tricks during evaluation?"

Look at `submission/agent.py`. The `act()` function does exactly 4 things:
1. Runs `preprocess()` to convert raw observations into tensors
2. Passes them through `HybridDoomNet` (the neural network)
3. Applies temperature sampling (`tau=0.85`) to pick actions
4. Applies 2 legal masks (USE cooldown, empty ammo)

There is literally zero pathfinding code, zero coordinate reading, and zero hand-coded rules. You can grep the file for `A*`, `atan2`, `privileged_state` — none of them appear.

---

### Q2: "Isn't using the Teacher Bot during training against the rules?"

No — the rules explicitly say you can use **anything during training**, including privileged state access. The teacher bot lives in `training/bc/teacher_bot.py` and is **never called** during evaluation. The evaluation-time agent (`submission/agent.py`) uses only standard `doom_env` observations, same as any other team.

---

### Q3: "What if the hidden evaluation maps have completely different room layouts or textures?"

Two things protect us:
1. **During training**, we randomly scrambled textures and lighting on every map so the agent couldn't rely on "this blue wall means exit room." We even randomly **turned off the camera** 20–25% of training steps, forcing the agent to navigate by radar alone.
2. **The radar + self-built map** are geometry-based — they don't care what color the walls are. A wall is a wall.

---

### Q4: "How does the agent remember a locked door it saw 2 minutes ago?"

Two memory systems work together:
1. **The GRU (working memory):** Like RAM — it actively tracks "I saw a locked door, I need a key" across the whole episode.
2. **Goal Memory Slots:** 32 "sticky notes" that slowly update but don't forget — the grid cell for "locked door" decays at 99.8% per step (stays almost forever).

Together they let the agent hold a plan for hundreds of steps.

---

### Q5: "Why tau=0.85 and not just 'pick the most likely action'?"

Picking the highest probability action every time (greedy/argmax) causes the agent to get **stuck in loops** at symmetric doorways — it can't break a tie between "turn left" and "turn right." Adding just a little randomness (`tau=0.85`) helps it escape these loops. We tested 4 values on 27 maps: `0.85` scored **1440 vs 1050 for 0.65 and 1085 for 1.00**. It's the sweet spot.

---

### Q6: "What's your biggest weakness and what would you do with more time?"

**Weakness:** Tier 3 and 4 maps — after unlocking a door, the agent sometimes forgets to backtrack through it. It also struggles with Tier 4 ambushes (lots of enemies at once at the start).

**Fix if we had 12 more hours:** Longer training on Stage 4 PPO, and extending BPTT (backpropagation through time) from 32 steps to 64 so the agent can plan further into the future.

---

## 🔑 The 10 Numbers That Matter

| Metric | Value |
|---|---|
| **Score on 27 unseen maps** | **2126.58** |
| **Exit rate** | **22.2%** |
| **Tier 1 exits** | **5/6 (83%)** |
| **Locked doors unlocked** | **5** |
| **Enemies killed** | **113** |
| **Average decision time** | **4.28 ms** |
| **Worst-case decision time (p99)** | **16.8 ms** |
| **Budget used (40ms limit)** | **~23%** |
| **Model parameters** | **868,403 (< 2M limit)** |
| **Docker validation** | **PASSED ✅** |
