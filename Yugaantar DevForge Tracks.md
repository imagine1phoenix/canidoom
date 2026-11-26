# **Yugaantar DevForge Problem Statements**

Welcome to the Yugaantar DevForge hackathon\! Below are the four problem tracks available. Choose one track to build and demo during the event.

## **Universal Submission & Evaluation Structure**

### **Team Submission (All Tracks)**

* **Public GitHub Repository URL:** Created during the hackathon, containing your complete source code, trained weights/artifacts, and track-specific entrypoints.  
* **2-Page Project Brief (PDF, maximum 2 pages):** Covering your Unique Selling Proposition (USP), system architecture, technical approach, and any track-specific brief requirements.

### **Evaluation & Shortlisting Funnel (Round 2 → Round 3 Finals, All Tracks):**

* **Stage 1 — Automated Evaluation & Verification:** Every valid submission is evaluated by an automated test suite and code/artifact check. The **Top 10 teams** in each track advance to the desk review.  
* **Stage 2 — Desk Review & Live Demo:** Organizing team members review a live demo and ask technical questions to verify your implementation and understanding, shortlisting the **Top 2 teams** per track.  
* **Round 3 — Final Judging:** The **Top 2 finalists** from each track (8 teams total) present their projects live in front of the official judging panel.

# **Track 1: "Can It Play DOOM?"**

This track focuses on Deep Reinforcement Learning, Spatial Intelligence, and Autonomous Control within the DOOM video game environment via the ViZDoom engine. Autonomous agents must learn zero-shot navigation, combat, and long-horizon multi-room progression directly through engine interactions using spatial sensory data rather than memorized actions.  
*Note: Full details for this track can be viewed in its respective tab.*

# **Track 2: "Multiverse of Monsoon"**

Centered around Earth-System Modelling, Spatiotemporal Deep Learning, and Climate Simulation, this track utilizes a daily Indian climate dataset and an offline inference server. Participants must construct a fast, physics-aware Climate Digital Twin capable of performing stable multi-day rollouts and simulating localized atmospheric shocks to model alternate weather outcomes.  
*Note: Full details for this track can be viewed in its respective tab.*

# **Track 3: "Secret Wards"**

This track covers Privacy Engineering, Multilingual Information Extraction, Record Linkage, and Data Security using a synthetic multi-hospital record dump in English, Hindi, and Hinglish. The goal is to build an offline, LLM-free privacy firewall that strips personal identifier information, protects against linkage attacks, and maintains medical data fidelity across fragmented patient records.  
*Note: Full details for this track can be viewed in its respective tab.*

# **Track 4: "Key Identity Guard"**

Focusing on Post-Quantum Cryptography and Zero-Trust Identity Controls, this track operates in a sandbox environment with mock services and synthetic identities. Participants must build a working prototype to secure sensitive workflows, enforce dynamic access changes, and provide verifiable security telemetry when identities or keys are compromised.  
*Note: Full details for this track can be viewed in its respective tab.*

# **Track 1: "Can It Play DOOM?"**

**Domain:** Deep Reinforcement Learning, Spatial Intelligence & Autonomous Control

**Environment:** DOOM (via ViZDoom Engine)

# **1\. Mission Brief**

For decades, "Can it run DOOM?" has been the ultimate test for hardware. In this hackathon, we flip the question to Machine Learning: **"Can your neural network actually play the classic video game DOOM?"**

Standard RL bots fail the moment a wall shifts or an enemy spawns in a new corner because they simply memorize fixed button sequences. Your challenge is to design and train an autonomous agent capable of playing through DOOM levels zero-shot—surviving demon ambushes, navigating complex multi-room architectures, handling doors and keycards, and reaching the level exit.

**No static dataset is provided.** Your agent must learn by interacting directly with the game engine.**Compute & Sensory Guidance:** Training solely on raw RGB screen pixels requires heavy GPU compute and spends 90% of training time just learning basic edge detection. Instead, we strongly recommend utilizing ego-centric spatial data. The official doom\_env wrapper (in the starter template) already provides it on every step: 64 ego-centric **raycasts** with hit types (wall, door, locked door, enemy, item), a **depth** buffer, a **vitals** vector (health, armor, ammo, keycards, velocity), and a list of visible entities in relative coordinates, alongside a low-resolution RGB frame. It runs at thousands of ticks per second on a standard CPU. You are completely free to use any subset of these signals, or combine pixels with spatial telemetry in a hybrid model.

# **2\. Core Challenge Objectives**

Your model must be **generalizable across different maps and unseen layouts** rather than overfitted to a single scenario. A successful agent must contend with:

* **Partial Observability:** Navigating 3D spaces without a god-view radar, relying only on what is currently within the player's field of view and local sensory range.  
* **Long-Horizon Progression:** Solving multi-step room dependencies—such as exploring branches to find a Keycard, remembering where a Locked Door was sighted earlier, and backtracking to unlock the exit.  
* **Tactical Combat & Survival:** Managing health, ammunition, and movement momentum while dodging incoming projectiles and prioritizing multiple threats.

# **3\. Deliverables & System Architecture**

Your final submission must include your trained agent, training pipeline, and inference setup covering the following core areas:

* **Sensory & State Representation:** Your encoders and preprocessing on top of the official observation dictionary (ego-centric rays, entity vectors, depth buffers, vitals, or hybrid vision-spatial inputs). The observation format is fixed by the official wrapper and identical in training and evaluation. It contains no global map coordinates, so the agent cannot simply memorize positions.  
* **Model Architecture & Memory:** Your custom neural network topology, specifically addressing how the agent maintains spatial memory across rooms and handles variable numbers of enemies.  
* **Training & Exploration Pipeline:** Your reinforcement learning implementation, custom reward shaping, and exploration mechanisms that drive the agent to navigate unfamiliar rooms rather than getting stuck in corners.

## **Submission Format**

* **Repository:** A **public GitHub repository** created from the official starter template. Total size, including Git LFS objects, must be **≤ 1 GB**. Model weights go through **Git LFS**.  
* **Agent Entry Point:** `submission/agent.py` defining a **stateful** class `DoomAgent` with `__init__(self, observation_space, action_space, device)` (load weights once), `reset(self)` (clear memory before every map), and `act(self, observation) -> action` (one decision every 4 game ticks).  
* **Dependencies:** Do **not** submit a Dockerfile. Agents run in the organizers' official base image (Python 3.13, `vizdoom`, `gymnasium`, `torch`, `jax`/`flax`, `numpy`, `scipy`, `opencv`, and common ML utilities). You may add a small `submission/requirements.txt` with pinned versions of extra packages.  
* **Local Validator:** Run `python validate.py` from the template before submitting. It checks the repo size, the `DoomAgent` interface, a full dry-run episode, and step latency against the official timeouts. A repository that fails the validator will most likely fail automated evaluation.  
* **2-Page Project Brief (PDF):** Summarize your USP, neural architecture, spatial memory design, and training/exploration pipeline.

**Note on Innovation:** **Innovative, custom-engineered architectures are heavily rewarded.** Simply importing an off-the-shelf, plug-and-play `Stable-Baselines3` PPO script with default hyperparameters will score poorly. We are looking for bespoke neural modules, custom spatial memory systems, intrinsic curiosity mechanisms, or hierarchical control policies.

# **4\. Prerequisites & Recommended Libraries**

## **Core Libraries**

* **Starter template** ([devforge-doom-template](https://github.com/Reinforce-SST/devforge-doom-template)): The official `doom_env` Gymnasium wrapper (the exact observation/action contract used in evaluation), the `DoomAgent` skeleton, the local validator, the base image `Dockerfile`, and one small test map. Read `docs/CONTRACT.md` first.  
* **`vizdoom`:** The official Python API binding for the DOOM engine (pinned to `vizdoom==1.3.3`), supporting both raw screen buffers and internal object/sector state extraction. Evaluation uses the bundled **Freedoom 2** IWAD at skill 3\.  
* **`gymnasium`:** Standard interface for wrapping the environment step, action, and observation loops.  
* **`torch` / `jax` \+ `flax`:** For building custom spatial encoders, attention heads, recurrent memory cells, and custom RL loss functions from scratch.

## **Key Concepts to Research**

* **Ego-Centric Spatial Encoding:** Raycasting, local coordinate frame transformations, and Line-of-Sight filtering.  
* **Spatial Memory & Exploration:** Recurrent policies (LSTM/GRU), differentiable occupancy grids, and intrinsic curiosity rewards.  
* **Generalization in RL:** Curriculum learning, domain randomization, and hierarchical sub-goal policies.  
* **Training Maps:** Only one test map is provided, and it is **not** an evaluation map. Sourcing training maps is part of the challenge: classic DOOM/Freedoom levels, community WADs, any procedural level generator, or your own. The wrapper loads any Doom-format or UDMF WAD.

# **5\. Evaluation & Scoring**

* **Hidden Map Suite (Automated Evaluation):** Every agent plays the same set of **unseen** maps across four tiers of increasing difficulty and weight: **Tier 1** Combat & Survival, **Tier 2** Multi-Room Exploration, **Tier 3** Long-Horizon Progression (keycard → locked door → exit), and **Tier 4** Full Gauntlet. Each map has a fixed tick cap (about 2,100–4,200 ticks) and a fixed random seed.  
* **Non-Binary Scoring:** Partial progress always counts. Each map awards points for reaching the **exit** (the largest reward, plus a small bonus for remaining health and time), **progression** (keycards picked up, doors opened, locked doors unlocked), **exploration** (unique areas visited, tracked privately by the scorer), and **combat** (enemies killed, minus small penalties for damage taken and dying early). Your automated leaderboard score is the tier-weighted sum over all maps.  
* **Hardware & Sandbox:** Evaluated offline on an NVIDIA RTX 3090 sandbox. Each agent gets up to 2 GB of VRAM, 8 GB of RAM, and 1 CPU thread, with **no network access** and its repository mounted read-only.  
* **Timeouts:** 60 s to initialize (load weights), 10 s per `reset()`, 10 s for the first `act()` of each map (JIT warm-up), 5 s for any other `act()`, and a compute budget of about **40 ms per decision** on average per map. If your agent crashes, times out, or returns an invalid action, your player stands still for the rest of that map: you keep the points earned so far, and penalties such as dying still apply. The exact values ship in the template (`env/doom_env/limits.py`) and are enforced identically by the local validator.  
* **Advancement & Final Judging:** The automated scorer ranks all teams and advances the **Top 10** to the desk review after code verification for rule compliance. Following the live demo and technical Q\&A during the desk review, the **Top 2 finalists** advance to Round 3 final judging, where innovative, custom-engineered architectures are rewarded.

# **6\. Ground Rules**

* **No Hardcoded Scripts:** Final agents must make decisions via trained neural parameters. Hardcoded `if/else` rule trees, classical pathfinders (`A*`, `Dijkstra`), or trigonometry aimbots (`atan2`) are discouraged.  
* **Engine Modifications & Curriculum Training:** You are completely free to modify the DOOM engine, map scripts (ACS), or game variables during **training** for focused pretraining or curriculum learning (e.g., making doors open automatically, granting infinite ammo, or isolating combat/navigation phases). However, **final evaluation will run strictly on the standard, unmodified DOOM engine and vanilla game rules—no mods allowed.**  
* **Original Work:** Your repository must be created during the hackathon (from the official starter template). Pre-existing open-source libraries are permitted, provided your core agent architecture and training pipeline are built during the event. Keep your training code in the repository: reviewers and judges read it.  
* **Fair Play:** Your agent may only use the observations it is given. Attempting to access the game engine, the evaluation maps, the network, or the host, or deliberately exhausting shared resources (GPU memory, CPU, disk), leads to disqualification.

# **Track 2: "Multiverse of Monsoon"**

**Domain:** Earth-System Modelling, Spatiotemporal Deep Learning & Climate Simulation

**Environment:** Provided Indian climate dataset (`india_climate_train.nc`, 2005–2021 daily, `32 × 36` grid) \+ an offline inference server (`serve.py`)

**Team Kit:** [yug-monsoon-kit](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit) (data, API contract, validator and starter server on its [Releases page](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0))

**Inspired by:** ISRO's Bharatiya Antariksh Hackathon PS-5 (*AI-Powered Digital Twin of India's Climate*)

# **1\. Mission Brief**

Weather apps answer *"Will it rain tomorrow?"* A **Climate Digital Twin** answers: **"How does India's atmosphere evolve, and what happens in an alternate universe if we change the starting conditions?"**

Build a fast, physics-aware Digital Twin of India's weather that can:

1. **Simulate (Timeline Prime):** from the last 3 days of India's state (temperature, rain, wind, pressure, moisture), roll the weather forward day by day without exploding or blurring into a flat average.  
2. **Branch (Multiverse):** accept a localized shock, such as a hotter Bay of Bengal or a weaker monsoon wind, and show how it ripples across India.

**No starter model or baseline weights are provided.** The kit gives you the data, the API contract, a validator and a starter `serve.py` that only handles the API (its placeholder model repeats the last day and scores zero by design).Beyond the Model: Passing the automated benchmark only gets you to the desk review. There you demo your full system: your engine, an interactive map dashboard, and a scientific or social-impact extension of your choice (e.g. crop stress under a stalled monsoon, solar/wind generation, heatwave or flood risk, ensemble uncertainty, or a policy co-pilot).

# **2\. What We Look For**

* **Stable rollouts:** 10 steps or more with no NaN/Inf, rain never negative, no collapse into blurry averages.  
* **Real weather skill:** beat "repeat yesterday" and "climatology", including heavy rain.  
* **Causal shocks:** a shock stays near where it was applied at first, then spreads outward. A zero shock changes nothing.  
* **Real impact:** a dashboard and impact module driven live by your own simulation.

# **3\. Submission**

**Public GitHub repo** with:

* training code and `serve.py` at the repo root  
* model weights **inside the repo** (Git LFS is fine, keep it under 2 GB; nothing is downloaded at evaluation, so a Hugging Face link alone is not enough)  
* dashboard code, separate from the server code  
* a README on how to train and run

**2-page brief (PDF):** page 1 is model design, how you prevent drift and blur, and failure modes. Page 2 is your visualization and the impact use case.

## **Offline run**

Organizers run your repo offline in a sandbox. No internet, no public URL needed:

`python serve.py --port 8000`

* Listen on `0.0.0.0`. Limits: **6 GB RAM, 4 CPU cores, no GPU**. The repo is read-only (write only to `/tmp`).  
* Libraries available: torch (CPU), numpy, scipy, pandas, xarray, scikit-learn, lightgbm, onnxruntime, fastapi, flask, uvicorn. Anything else must be in your repo.

## **API (plain nested JSON lists, raw physical units)**

* `POST /health` returns `{"status": "ready", "model_name": "..."}` once your model is loaded.  
* `POST /simulate`:

```
{ "init_states": "[3, 6, 32, 36]", "doy_sincos": [0.42, -0.91], "steps": 10, "perturbation": "[6, 32, 36] or null" }
```

returns

```
{ "rollout": "[steps, 6, 32, 36]", "inference_ms": 142.5 }
```

* **Channels, in order:** `t2m` (K), `tp` (mm/day), `u10` (m/s), `v10` (m/s), `msl` (Pa), `tcwv` (kg/m²).  
* **Units:** always physical. `stats.json` is only for use inside your model. The perturbation is a physical delta added to `init_states[-1]`.  
* **`doy_sincos`** is for the last input day. Calendar years are never sent.  
* **Timing:** the grader waits up to 120 s for `/health`, then sends each request once. Each 10-step answer must come back in **under 5 s**.  
* Full details: [API\_CONTRACT.md](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/blob/main/docs/R1_CONTRACT.md)

# **4\. Provided Assets & Quick Start**

From the [team kit](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit):

* `india_climate_train.nc` \+ `stats.json`: 2005–2021 daily data, 6 dynamic channels and 2 static channels (`land_sea_mask`, `orography_m`, bundle them with your model).  
* `API_CONTRACT.md`, a **schema validator** (`python -m monsoon_r1.schema_validator http://localhost:8000`), and a starter `serve.py`.

**Quick start:**

1. Install the kit and download the data from the Releases page.  
2. Copy the starter `serve.py` into your repo root and replace `simulate()` with your model.  
3. Run `python serve.py --port 8000` and check it with the validator.  
4. Add weights, dashboard and brief, then submit before the deadline. The repo is frozen after that.

# **5\. Evaluation**

## **Stage 1: automatic (100 points)**

The grader sends **8 hidden requests**: 5 real weather windows from 2023–24 (monsoon onset, heavy rain, heatwave, cyclone, winter disturbance) and 3 shocks (warm Bay of Bengal, weaker Arabian Sea wind, and a zero shock that must change nothing).

| Part | Points | What it checks |
| :---- | :---- | :---- |
| Forecast skill | 45 | Error (30) and heavy-rain hits (15), at days 1, 4 and 8, against the better of "repeat yesterday" and climatology |
| Dynamics | 25 | No blur (10), shocks stay local then spread outward and stay stable (15) |
| Repo check | 15 | No weather APIs, no hardcoded dates or data archives, real training code, weights included |
| Brief | 15 | Read by a person |

**To qualify:** valid outputs on all 8 requests, beat "repeat yesterday" at day 4, and react sensibly to at least one shock. The **Top 10** qualifying teams by score go through.

## **Stage 2: desk review and Round 3 finals**

Top 10 do a live demo: judges apply a new shock on the spot, review your impact extension, and ask about your model's design and limits. The **Top 2** go to the finals. This stage is judged by people.

# **6\. Ground Rules**

* **Offline only.** No external weather APIs or internet at inference. Everything must come from your model and the provided inputs.  
* **No lookup tricks.** Hardcoded dates, lookup tables or shipped archives of past weather fail the shock tests and the repo scan, and lead to disqualification.  
* **Real coupling.** Your dashboard must run on live output from your own `/simulate`, not pre-rendered results.  
* **Frozen at the deadline.** Changes after the deadline are ignored.  
* **Original work.** Repo, training, weights and dashboard are built during the hackathon. Open-source libraries are fine.

# **Track 3: "Secret Wards"**

**Domain:** Privacy Engineering, Multilingual Information Extraction, Record Linkage & Data Security

**Environment:** A synthetic Indian hospital records dump (English, Hindi & Hinglish) from 4 hospital systems, provided by the organizers, plus a hidden attacker that hunts for patients in your output. No real patient data is used anywhere in this track.

What happens in the ward stays in the ward.

# **1\. Mission Brief**

Hospitals in Yugaantar Pradesh are merging years of patient records into one research database for public-health studies. What arrives is chaos: registration exports with shifted columns, legacy CSVs in old encodings, nested JSON, FHIR bundles, discharge summaries, prescriptions and nursing notes that jump between English, Devanagari Hindi and Hinglish in the same sentence.

The same patient shows up as "Mohammad Arif" in a registration file, "मोहम्मद आरिफ" in a Hindi prescription, "Md. Arif" in a nursing note and "Arif sahab ka BP 150/90" in an OPD record. Each copy has a different phone format, a half-written address and a different hospital ID. Names also hide inside JSON keys, file names and file attachments.

Your challenge is to build an **offline, CPU-only, LLM-free privacy firewall** that:

* removes every piece of personal information,  
* keeps every medically useful value exactly intact,  
* recognizes the same patient across languages, files and hospitals without merging different people, and  
* leaks nothing: not in outputs, logs, temp files or error messages.

Removing names is the easy part. The real test is whether an attacker holding a voter list who knows "a 47-year-old woman from a small village was admitted for 9 days with dengue" can still find her in your output. After you submit, our automated attacker tries exactly that. Your job is to make it fail.

A practice dataset with full answers is released at the start.

# **2\. Core Challenge Objectives**

* **The Language Trap:** Detect patients across English, Hindi and Hinglish: names in any order (surname first, initials, honorifics like *ji*, *sahab*, *didi*), transliteration variants (Mohammad / Mohd / Md. / मोहम्मद), relatives via S/o, D/o, W/o, and Indian IDs (Aadhaar-like, ABHA, UHID / MRN, insurance IDs, PAN) and phone numbers in every format.  
* **The Meaning Trap:** "C/O Ramesh" (care of a relative) is not "C/O fever" (complaint of). "Dr. Ramesh" stays; relative "Ramesh" goes. "Ramipril 5 mg", "Dolo 650 BD x 5 days" and "1-0-1" are doses, not phone numbers or IDs. "Parkinson's disease" is not a person.  
* **The Fragmented Identity Trap:** Give every real patient one stable pseudonym across all records, languages and hospitals, even when a note only says "Meera, W/o Rajan". Never merge two different people who share a name. When evidence is too weak, don't link.  
* **The Re-identification Trap:** Quasi-identifiers (age, village, PIN code, exact dates, length of stay, rare diagnoses) can still single someone out. Decide what to generalize, shift or suppress, and justify the privacy-vs-utility trade-off.  
* **The Clinical Integrity Trap:** Medicines, doses, frequencies, vitals, lab values with units, diagnoses and procedures must survive exactly. If you shift dates, a 9-day stay must still be 9 days. Age bands must contain the real age.  
* **The Hidden Leakage Trap:** Clean JSON keys, CSV headers, file names, metadata and attachments. Quarantine any file you cannot safely process instead of passing it through half-cleaned.

# **3\. Deliverables & System Architecture**

Your submission is a **public GitHub repository** with a runnable pipeline, a `README.md`, and the **2-page Project Brief PDF** covering:

* **Ingestion:** Reads CSV, TSV, JSON, FHIR R4 and text notes in different encodings, including broken, truncated and unsupported files.  
* **Detection Engine:** How personal data is found in structured columns and free text across three scripts: rules, gazetteers, transliteration, classical ML, and how they are combined.  
* **Linkage & Pseudonymization:** How records of the same patient are clustered (blocking, fuzzy and phonetic matching, cross-script comparison) and how stable, **non-reversible** pseudonyms are issued (keyed HMAC, not plain hashing).  
* **Privacy Layer:** How quasi-identifiers are handled (per-patient date shifting, age banding, location generalization, rare-value suppression) and your measured re-identification risk.  
* **Security Design:** Encryption of raw and intermediate data at rest, keys kept away from data, PII-free logging and a quarantine design. Assume your server will be breached one day.  
* **README & 2-Page Project Brief (PDF):** Setup steps, architecture diagram, USP, privacy-utility trade-off analysis, and known limitations.

**Note on Innovation:** Wrapping an off-the-shelf PII tool with default settings will score poorly. These tools miss Hindi and Hinglish, break on Indian name structures, and do nothing against linkage attacks. Deleting everything scores about 20/100, and copying data unchanged scores below zero. We reward cross-script identity matching, smart quasi-identifier handling, measurable privacy and systems that still work on formats they have never seen.

## **Submission Format**

Step 1: Add a raksha.toml file at the root of your repo. It tells us how to run your system:

```
[submission]
name  = "your-team-name"
setup = "pip install -r requirements.txt"                    # run once, internet allowed
run   = "python main.py --input {input} --output {output}"   # run with NO internet
```

```

We replace `{input}` with the hidden data folder and `{output}` with an empty folder. Your secret key arrives in the environment variable `RAKSHA_SECRET`. Use it for pseudonyms and date shifts, and never write it anywhere.

**Step 2: Your system must create these inside `{output}`:**
```

output/  
├── sanitized/       ← cleaned files: same folders, names and formats as the input  
├── report.json      ← what you processed  
└── records.jsonl    ← which record belongs to which fake patient ID

````

**① `sanitized/`**

* `input/registration/hospital.csv` becomes `output/sanitized/registration/hospital.csv`, with the same columns.
* CSV stays CSV, JSON stays JSON, FHIR stays valid FHIR R4. Save everything as UTF-8.
* If a file name itself contains personal data, rename it and record the new name in the report.
* Never copy a file you could not process. Mark it `quarantined` instead.

**② `report.json`:** every input file (identified by its SHA-256 hash, never its name), plus totals:

```json
{
  "files": [
    {"input_sha256": "9f2c…", "output": "sanitized/discharge/note1.txt", "status": "sanitized"},
    {"input_sha256": "51aa…", "output": null, "status": "quarantined", "reason": "unsupported format"}
  ],
  "counts": {"files_total": 817, "files_sanitized": 809, "files_quarantined": 8,
             "records_processed": 5756, "identifiers_detected": 21950, "records_linked": 5690,
             "records_generalized": 2100, "records_suppressed": 3, "records_quarantined": 0},
  "timing": {"seconds": 41.2},
  "resources": {"peak_rss_mb": 380}
}
````

The SHA-256 is one line of Python: hashlib.sha256(open(path, "rb").read()).hexdigest(). Report numbers must be true; we cross-check them.

③ records.jsonl: one line per patient record, with the fake ID you assigned:

```
{"file": "sanitized/registration/hospital.csv", "loc": "row:12", "patient": "PSN-7Q2K9D"}
{"file": "sanitized/opd/visits.json", "loc": "/visits/3", "patient": "PSN-7Q2K9D"}
{"file": "sanitized/discharge/note1.txt", "loc": "doc", "patient": "PSN-7Q2K9D"}
```

| File type | What to write in loc |
| :---- | :---- |
| CSV / TSV row | `row:N` (first data row after the header is `row:0`) |
| JSON record | its path, e.g. `/visits/3` |
| Text note / FHIR file | `doc` (the whole file) |

* The same patient gets the **same ID in every file**, and that ID must also appear inside the cleaned record.  
* Not sure who the patient is? Write `"patient": null`. A wrong link costs more than no link.

**Step 3: Check your format before submitting**, using the checker from the team kit:

`python3 validate_output.py --input dev/input --output my_output/`It must print CONTRACT OK.

Step 4: Submit your GitHub repo link and 2-page Project Brief PDF before the deadline. We record your last commit at the deadline; later commits are ignored.

**Minimal repo layout**

```
your-repo/
├── raksha.toml
├── requirements.txt
├── README.md
├── config/policy.yaml    ← your privacy rules
└── main.py               ← plus the rest of your code
```

# **4\. Prerequisites & Recommended Libraries**

## **Core Libraries**

* `regex`, `phonenumbers`, `dateparser`: Indian phones, dates and ID patterns  
* `indic-transliteration` / `indic-nlp-library`: Devanagari ↔ Latin transliteration and normalization  
* `rapidfuzz`, `jellyfish`: fuzzy and phonetic matching  
* `sklearn-crfsuite` / `spaCy` (rule-based Matcher, small non-LLM pipelines): classical NER  
* `splink` / `recordlinkage`: probabilistic record linkage (Fellegi–Sunter)  
* `fhir.resources`, `pydantic`, `jsonschema`: valid FHIR R4 and JSON output  
* `cryptography`: AES-GCM / Fernet encryption, HMAC keyed pseudonyms  
* `anonymeter`, `pycanon`: measuring re-identification risk, k-anonymity and l-diversity  
* `Faker` (`en_IN`, `hi_IN`): generating extra test data of your own

## **Key Concepts to Research**

* **Indian Name Structures:** patronymics, initials, honorifics, cross-script spellings and nicknames  
* **Quasi-identifiers & Linkage Attacks:** k-anonymity, l-diversity, date shifting, generalization hierarchies  
* **Record Linkage:** blocking, Fellegi–Sunter, clustering without transitive over-merging  
* **De-identification Standards:** HIPAA Safe Harbor's 18 identifiers, India's DPDP Act 2023, ABDM health data guidelines  
* **Secure Data Handling:** envelope encryption, HMAC vs plain hashing, canary tokens, log hygiene

# **5\. Evaluation & Scoring**

**Automated Scoring:** Your system runs on 3 hidden datasets, then our automated attacker runs on your output. The **Top 10 teams** advance to the desk review and live demo to select the **Top 2 finalists** for Round 3 final judging.

| \# | Category | What we measure |
| :---- | :---- | :---- |
| 1 | **Privacy Against Attacks** | How few patients the attacker can re-identify using a voter-list-style register and known facts (visit dates, rare diseases, length of stay) |
| 2 | **Personal Data Detection** | Personal data removed across English, Hindi and Hinglish, in content and hidden places; doctors, doses and medical terms kept |
| 3 | **Medical Data Preservation** | Medicines, doses, vitals, lab values, diagnoses, exact hospital-stay lengths, truthful age bands |
| 4 | **Patient Identity Consistency** | Same patient linked across files and languages; different people never merged |
| 5 | **Output Quality & Validity** | Valid CSV / JSON / FHIR, honest report, no leaks in logs or temp files, proper quarantine |
| 6 | **Speed & Reliability** | Speed, memory, no crashes, identical output on repeat runs |

* **Penalties:** Points are deducted for every real ID or canary leaked, raw files copied out, files skipped silently, and modifying the input.  
* **Ties:** A team ranks above another only if it scores more than 1 point higher on every hidden dataset.

**Evaluation Principle:** Build a system that protects patient privacy without compromising the accuracy, integrity or medical usefulness of healthcare data.

# **6\. Ground Rules**

* **No LLMs:** No GPT, Llama, Gemini, Claude, Mistral or any other large language model, whether hosted, local or distilled. No generative models, and no pretrained transformers above 50M parameters. Rules, gazetteers, CRFs, classical ML and small models you train yourself are allowed; declare any model in your README.  
* **Fully Offline, CPU-only:** Evaluation runs on organizer machines with no internet, 2 CPUs, 4 GB RAM and 15 minutes per run. Any network call during the run is blocked and is grounds for disqualification.  
* **Synthetic Data Only:** All data comes from the organizers or is generated by your team. Real patient records are strictly prohibited.  
* **No Hardcoding:** Hidden data uses new patients, and at least one hidden dataset has names, labels and formats never seen in practice data. Memorized names or files will fail.  
* **Zero Leaks:** Hidden data contains planted canary PII. A canary found anywhere (outputs, logs, temp files, error messages or reports) counts as a leak and is heavily penalized.  
* **Fail Safe:** Never silently skip files or export raw files as "cleaned". Exiting successfully with missing files is penalized.  
* **Reproducible:** The same input and the same `RAKSHA_SECRET` must give exactly the same output. A different secret must give different pseudonyms and date shifts.  
* **Original Work:** Your repository must be created during the hackathon. Open-source libraries are allowed, but your core pipeline, linkage logic and privacy layer must be built during the event.

# **Track 4: "Key Identity Guard"**

**Domain:** Post-quantum cryptography, identity and access control

**Environment:** Organizer sandbox with mock services, synthetic data and test identities

# **1\. Mission Brief**

A user has access to a sensitive file. Their device is later reported lost. Can your prototype protect the file, enforce the change in access and show an operator what happened?

Build a working prototype that combines post-quantum cryptography with zero-trust identity controls. Describe the workflow you protect and the security claim judges can test. Examples include rotating a service key, detecting an unexpected privilege change or revoking a device used for encrypted messaging. All entries use one Track 4 rubric.

# **2\. Core Challenge Objectives**

* **Normal Operation:** Demonstrate an authorized user completing the protected workflow.  
* **Security Control:** Show what happens when an identity, key or access condition changes. If your prototype raises an alert, explain which rule or sequence of events triggered it.  
* **Reliability:** Show how the workflow handles a relevant interruption or change in input.  
* **Verifiable Outcome:** Give judges evidence that connects the test input to the observed decision, alert or delivery result.

# **3\. Deliverables & System Architecture**

Your final submission must cover the following components:

1. **Working Prototype:** Demonstrate the protected workflow in the organizer sandbox.  
2. **Source & Setup (Public GitHub Repository):** Submit the repository, dependency list, and a `README.md` with clean-machine setup steps.  
3. **2-Page Project Brief (PDF):** Include your USP and system architecture alongside:  
   * **One-Page Threat Model:** Identify protected assets, possible attackers, assumptions, and limits.  
   * **5-Minute Demo Video Link:** Embed a publicly accessible link (e.g., unlisted YouTube or Google Drive) to a 5-minute walkthrough recording inside the PDF as a fallback if the live demo fails.

## **Submission Format**

* **Public GitHub Repository URL:** Created during the hackathon with full source code and clean-machine setup instructions.  
* **2-Page Project Brief (PDF):** Contains your USP/architecture overview, your 1-page threat model, and the link to your 5-minute demo video.

# **4\. Prerequisites & Recommended Libraries**

* **Post-Quantum Cryptography:** Use a vetted library that implements the standardized algorithm your design needs. `ML-KEM` establishes a shared secret. `ML-DSA` and `SLH-DSA` provide digital signatures. A hybrid design can combine classical and post-quantum algorithms.  
* **Identity Controls:** Be ready to explain how your prototype checks an identity, applies permissions and records its decision.  
* **Detection:** Use explainable rules, statistics or graph-based methods. Document the dependencies used by your prototype.

# **5\. Evaluation & Scoring**

* **Stage 1 — Automated Build & Assessment (100 Points):** Organizers build the submitted repository in a clean sandbox and run automated functional and security verification tests to advance the **Top 10 teams** to the desk review:  
  * **Security Correctness:** 25 points  
  * **Live / Automated Security Assessment:** 25 points *(Starts at 25 points. Each confirmed critical, high, medium, or low finding deducts 10, 6, 3, or 1 point respectively. Detecting and responding during the assessment restores up to half the deduction; minimum score is 0.)*  
  * **Functionality & Reliability:** 20 points *(5 points for each of the four core objectives)*  
  * **Threat Model & Honesty:** 10 points  
  * **Performance & Overhead:** 10 points  
  * **Usability & Documentation:** 10 points  
* **Stage 2 — Desk Review & Round 3 Finals:** The Top 10 teams undergo a live desk review featuring a 5-minute demo, 5 minutes of technical questions, and a live security assessment in the sandbox to select the **Top 2 finalists** for Round 3 final judging.

# **6\. Ground Rules**

* **Scope:** Use only the assigned sandbox, mock services and test identities. Attacks on other teams or out-of-scope systems are grounds for disqualification. Denial-of-service against shared services are prohibited.  
* **Submission Integrity:** Hard-coded secrets, AI/ML components, custom cryptographic primitives presented as secure, and plagiarism without attribution are grounds for disqualification.  
* **Original Work:** Your repository must be created during the hackathon. Pre-existing open-source libraries are permitted, provided your core prototype and security controls are built during the event.  
* 