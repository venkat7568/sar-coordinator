# SAR Coordinator Environment

**A Search & Rescue operational RL environment built on the OpenEnv spec.**

An AI agent acts as a SAR field coordinator stranded with a team in hostile terrain. Every decision matters — deploy water before the team dehydrates, establish shelter before hypothermia sets in, signal extraction before the window closes. Wrong prioritization cascades forward. There are no second chances.

> This is not a game. It is the most compact, measurable abstraction of the decision problem that governs real search and rescue operations, autonomous field robotics, and humanitarian crisis response — all domains with no shared training benchmark. SAR Coordinator is that benchmark.

---

## Why This Environment

Every year, SAR teams, NASA rover engineers, and humanitarian field coordinators solve the same underlying problem: *sequential decisions under resource scarcity, partial observability, and irreversibility.* Each team builds its own simulator from scratch, in isolation.

SAR Coordinator gives all of them a single, standardized, deployable environment. Plug in your agent. Compare results. No custom simulation code required.

---

## Environment Overview

| Property | Value |
|---|---|
| Spec | OpenEnv v1 |
| Runtime | FastAPI + WebSocket |
| Port | 8000 |
| Environment name | `sar-coordinator` |
| Tasks | 3 (Easy / Medium / Hard) |
| Reward | Dense, 6-component partial-credit |
| Requires GPU | No — pure Python, 2 vCPU / 8GB RAM |

---

## Action Space

All actions use a single flat model (`SARAction`) with an `action_type` discriminator.

| `action_type` | Key Parameters | What it does |
|---|---|---|
| `deploy` | `resource_type`: water/food/equipment/medical | Deploy resource acquisition team |
| `establish` | `structure_type`: field_shelter/signal_fire/extraction_point | Establish field infrastructure |
| `relocate` | `direction`: N/S/E/W, `distance`: 1–5 km | Move operations to new sector |
| `standby` | `duration`: 1–8 hours | Hold position, recover team capacity |
| `triage` | `condition`: trauma/hypothermia/dehydration | Administer medical response |
| `extract` | `signal_method`: flare/mirror/radio/beacon | Request extraction from command |
| `allocate` | `item`: food/water/medicine, `quantity` | Distribute inventory to personnel |
| `assess` | `target`: terrain/weather/personnel/resources | Gather situational intelligence |

**Example action:**
```json
{
  "action_type": "deploy",
  "resource_type": "water",
  "reasoning": "hydration_deficit at 0.72 — critical threshold is 0.80, water acquisition is priority one."
}
```

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `hydration_deficit` | float 0–1 | 0 = hydrated, 1 = critically dehydrated |
| `nutrition_deficit` | float 0–1 | 0 = adequate, 1 = critical |
| `team_capacity` | float 0–1 | 1 = full operational, 0 = non-operational |
| `core_temperature` | float (°C) | Hypothermia onset below 35°C |
| `mission_viability` | float 0–1 | Primary reward signal — 0 = mission failure |
| `base_camp_status` | none/established/fortified | Field infrastructure level |
| `weather_conditions` | clear/rain/storm/snow | Affects all physiological decay rates |
| `weather_severity` | float 0–1 | Severity of current weather system |
| `sector_map` | 5×5 grid | Partial observability — unknown sectors masked |
| `resource_inventory` | dict | water, food, equipment, medical quantities |
| `mission_elapsed_hours` | int (hours) | Operational clock |
| `extraction_requested` | bool | Whether extraction has been signalled |
| `distance_to_extraction` | float (km) | Proximity to nearest extraction point |
| `sitrep` | string | Situation report — outcome of last action |
| `active_incidents` | list[str] | Active field incidents this step |

---

## Reward Function

Dense partial-credit — the agent gets a learning signal on every step, not just win/loss.

| Component | Weight | Signal |
|---|---|---|
| Mission viability maintenance | 0.30 | Continuous — `mission_viability` value per step |
| Resource efficiency | 0.20 | High when deployed resource was urgently needed |
| Base camp & signal fire | 0.15 | Partial credit per camp level + fire established |
| Extraction signal quality | 0.15 | Quality × timing multiplier |
| Sector coverage | 0.10 | Novel sectors reconnoitred / 25 |
| Decision optimality | 0.10 | Penalises clearly counterproductive orders |

*A sparse reward (survive or fail) gives zero gradient for thousands of steps. SAR Coordinator's multi-component reward means the agent learns on every action — identical to how SAR probability models and rover energy budgets actually work.*

---

## The Three Tasks

### Task 1 — Resource Triage (Easy, 5 steps)
**Scenario:** Forest terrain, clear weather, Day 1. Three resource caches visible nearby.

**Objective:** Prioritise correctly — water > shelter > food > extraction signal.

**Grader:**
- Score 1.0: Water deployed first, field shelter established before food
- Score 0.5: Water first, suboptimal subsequent priority
- Score 0.0: Water not the first action

**Real-world analog:** SAR first-responder arriving at a stranded team — immediate resource triage.

---

### Task 2 — 24-Hour Operational Arc (Medium, 24 steps)
**Scenario:** Jungle terrain. Rain forecast at hour 8. Food scarce, water nearby. Day 1–2.

**Objective:** Maintain team operational status for 24 hours across all four survival milestones.

**Grader (4 × 0.25):**
- +0.25 Field shelter established before hour 8 (before rain arrives)
- +0.25 Water deployed within first 6 hours
- +0.25 Food allocated at least twice
- +0.25 Extraction signal sent at any point

**Real-world analog:** Multi-sol rover mission planning / humanitarian camp setup in hostile conditions.

---

### Task 3 — Multi-Day Rescue Arc (Hard, 120 steps = 5 days)
**Scenario:** Arctic terrain. Team borderline hypothermic from step 1. Random events: trauma on day 2, storm on day 4. Rescue window opens from day 3.

**Objective:** Maintain mission viability for 5 days, respond to dynamic incidents, execute extraction.

**Grader (4 components):**
- 0.0–0.40: Mission duration (steps survived / 120)
- 0.0–0.20: Trauma treated within 12 hours of day 2 incident
- 0.0–0.20: Base camp established before storm at hour 72
- 0.0–0.20: Extraction signalled after rescue window opens (hour 48)

**Real-world analog:** Long-duration SAR operation, multi-sol planetary mission with anomaly response.

---

## Setup & Usage

### Local (no Docker)

```bash
# Install dependencies
pip install -r requirements.txt

# Start the environment server
uvicorn myenv.server.app:app --host 0.0.0.0 --port 8000 --reload

# In a second terminal — run the inference script
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token_here"
export ENV_BASE_URL="http://localhost:8000"

python inference.py
```

### Docker

```bash
cd myenv

# Build
docker build -t sar-coordinator .

# Run
docker run -p 8000:8000 \
  -e HF_TOKEN=your_token \
  sar-coordinator
```

### Live Dashboard

Navigate to `http://localhost:8000` to open the SAR Operations Center dashboard — a real-time dark-themed visualization showing mission viability, personnel status, resource inventory, and active incidents. Auto-refreshes every 2 seconds.

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/reset` | POST | Start new episode. Body: `{"task_id": 1}` |
| `/step` | POST | Execute operational order. Body: `SARAction` JSON |
| `/state` | GET | Current episode state |
| `/schema` | GET | Action / observation schemas |
| `/render` | GET | ASCII ops dashboard (terminal) |
| `/last_obs` | GET | Last cached observation (used by dashboard) |
| `/docs` | GET | Interactive FastAPI docs |

---

## Inference Script

`inference.py` at the project root runs all 3 tasks and emits structured logs:

```
[START] task=resource-triage env=sar-coordinator model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action={"action_type":"deploy","resource_type":"water"} reward=0.31 done=false error=null
[END]   success=true steps=5 rewards=0.31,0.28,0.33,0.30,0.29
```

**Required environment variables:**

| Variable | Description |
|---|---|
| `API_BASE_URL` | LLM inference endpoint (OpenAI-compatible) |
| `MODEL_NAME` | Model identifier |
| `HF_TOKEN` | HuggingFace token |
| `ENV_BASE_URL` | SAR Coordinator server URL (default: `http://localhost:8000`) |

---

## Physics Summary

| Vital | Base decay/hr | Critical threshold |
|---|---|---|
| Hydration deficit | +0.06/hr | > 0.80 → viability -0.05/hr |
| Nutrition deficit | +0.04/hr | > 0.85 → viability -0.03/hr |
| Team capacity | -0.05/hr | No direct failure, impairs all actions |
| Core temperature | Drifts to ambient | < 34°C or > 40°C → viability -0.04/hr |

Weather and base camp status modify all rates. Signal fire counteracts cold.

---

## Team

Built by **Venkat Dhanikonda** and **Chetana Varahachalam** for the Meta × HuggingFace OpenEnv Hackathon.

OpenEnv Track | Round 1 | April 2026
