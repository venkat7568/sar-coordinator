#!/usr/bin/env python3
"""
SAR Coordinator Environment — Inference Script
===============================================
Runs an LLM agent through all 3 operational tasks as a SAR Coordinator.

Required environment variables:
    API_BASE_URL  — LLM inference endpoint (OpenAI-compatible)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — HuggingFace token (also reads TOKEN from .env as fallback)
    ENV_BASE_URL  — SAR Coordinator server URL (default: https://venkat7568-sar-coordinator.hf.space)

MANDATORY stdout format (judges parse this exactly):
    [START] task=<task_name> env=sar-coordinator model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

import os
import sys
import json
import time
from typing import Optional

# ── Load .env if present (for local dev) ─────────────────────────────────────
def _load_dotenv():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip()
                    if v:  # skip empty values — let code defaults take effect
                        os.environ.setdefault(k.strip(), v)

_load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "https://venkat7568-sar-coordinator.hf.space")

HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("[WARN] No HF_TOKEN or API_KEY found. LLM calls may fail.", file=sys.stderr)

# Task config — matches SAREnvironment task definitions
TASKS = {
    1: {"name": "resource-triage",         "max_steps": 5,   "threshold": 0.5},
    2: {"name": "24hr-operational-arc",    "max_steps": 24,  "threshold": 0.5},
    3: {"name": "multiday-rescue-arc",     "max_steps": 120, "threshold": 0.4},
}

# ── Heuristic Fallback Sequences (used when LLM is unavailable) ──────────────
FALLBACK_SEQUENCES = {
    1: [  # Resource Triage — water first, shelter before food
        {"action_type": "deploy",    "resource_type": "water"},
        {"action_type": "deploy",    "resource_type": "equipment"},
        {"action_type": "establish", "structure_type": "field_shelter"},
        {"action_type": "deploy",    "resource_type": "food"},
        {"action_type": "allocate",  "item": "food", "quantity": 1.0},
        {"action_type": "allocate",  "item": "food", "quantity": 1.0},
        {"action_type": "extract",   "signal_method": "radio"},
    ],
    2: [  # 24-Hour Arc — same doctrine, extract after hr20
        {"action_type": "deploy",    "resource_type": "water"},
        {"action_type": "deploy",    "resource_type": "equipment"},
        {"action_type": "establish", "structure_type": "field_shelter"},
        {"action_type": "deploy",    "resource_type": "food"},
        {"action_type": "allocate",  "item": "food", "quantity": 1.0},
        {"action_type": "allocate",  "item": "food", "quantity": 1.0},
        {"action_type": "extract",   "signal_method": "radio"},
    ],
    3: [  # Multi-Day Rescue — signal_fire + shelter first (3.0 equipment provided at start)
        {"action_type": "establish", "structure_type": "signal_fire"},   # step 1
        {"action_type": "establish", "structure_type": "field_shelter"}, # step 2
        # Survival cycle (steps 3-118): water/food every 2-3 steps to prevent dehydration
        {"action_type": "deploy",    "resource_type": "water"},
        {"action_type": "allocate",  "item": "water", "quantity": 1.0},
        {"action_type": "deploy",    "resource_type": "food"},
        {"action_type": "allocate",  "item": "food",  "quantity": 1.0},
        {"action_type": "allocate",  "item": "water", "quantity": 1.0},
        {"action_type": "allocate",  "item": "food",  "quantity": 1.0},
        {"action_type": "deploy",    "resource_type": "water"},
        {"action_type": "allocate",  "item": "water", "quantity": 1.0},
        {"action_type": "triage",    "condition": "trauma"},
        {"action_type": "allocate",  "item": "water", "quantity": 1.0},
        {"action_type": "extract",   "signal_method": "radio"},          # step 119+
    ],
}

def get_fallback_action(step: int, task_id: int = 1) -> dict:
    """Return a safe heuristic action when LLM is unavailable."""
    seq = FALLBACK_SEQUENCES.get(task_id, FALLBACK_SEQUENCES[1])
    if task_id == 3:
        # Task 3 (120 steps): setup first, then cycle survival, extract only at step 119+
        # Strategy: survive all 120 steps for max survival score, extract late for bonus
        if step <= 2:
            return seq[step - 1]   # step 1: signal_fire, step 2: field_shelter
        if step >= 119:
            return {"action_type": "extract", "signal_method": "radio"}
        # Steps 3–118: cycle water/food/allocate/triage without extract
        cycle = seq[2:-1]  # excludes extract (last item)
        return cycle[(step - 3) % len(cycle)]
    idx = min(step - 1, len(seq) - 1)
    return seq[idx]


# ── OpenAI Client ─────────────────────────────────────────────────────────────
from openai import OpenAI
import requests

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

# ── Rich for visual dashboard (stderr only — stdout reserved for log lines) ───
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    RICH = True
    console = Console(stderr=True)
except ImportError:
    RICH = False
    console = None


# ── Mandatory Log Functions ───────────────────────────────────────────────────

def log_start(task_name: str, model: str) -> None:
    print(f"[START] task={task_name} env=sar-coordinator model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: list) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ── Batch planning chunk sizes ────────────────────────────────────────────────
# Task 1: 1 LLM call for all 5 steps
# Task 2: 3 calls of 8 steps each
# Task 3: 6 calls of 20 steps each  (was 120 individual calls)
CHUNK_SIZES = {1: 5, 2: 8, 3: 20}

# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI decision-support system acting as a SAR (Search & Rescue) field coordinator.
You receive the current operational state as JSON and must respond with a JSON ARRAY of sequential actions.
Return ONLY the JSON array — no explanation, no markdown, no extra text.

BATCH PLANNING: When asked to plan N sequential actions, respond with a JSON ARRAY of N action objects.
Example 3-action response: [{"action_type":"deploy","resource_type":"water"},{"action_type":"establish","structure_type":"field_shelter"},{"action_type":"extract","signal_method":"radio"}]

TASK-SPECIFIC STEP 1 RULES:
  TASKS 1 & 2: Step 1 is ALWAYS {"action_type": "deploy", "resource_type": "water"}
  TASK 3 ARCTIC: Step 1 is ALWAYS {"action_type": "establish", "structure_type": "signal_fire"}
                 (equipment: 3.0 provided at start — use it immediately)

SAR TRIAGE DOCTRINE — TASKS 1 AND 2:
  1. WATER FIRST — Step 1 always deploy water
  2. EQUIPMENT — deploy equipment to get materials for shelter
  3. SHELTER — establish field_shelter once resource_inventory.equipment >= 2.0
  4. NUTRITION — deploy food, then allocate food TWICE to distribute rations to team
  5. REST — standby (duration=2) when team_capacity < 0.25
  6. EXTRACTION — extract with radio after shelter established and mission > 60% complete

TASK 2 JUNGLE HEAT NOTE — core temperature starts HIGH (38.5°C), drifts DOWN toward 28°C ambient:
  Without shelter, team hits hypothermia by hour 8-10.
  Step 3: establish field_shelter — CRITICAL to slow temperature drop.
  After rain at hour 8: shelter slows cooling. Extract after hour 20.

TASK 3 ARCTIC — resource_inventory starts with equipment: 3.0, water: 1.0, medical: 1.0:
  Step 1: establish signal_fire   (costs 1.0 equipment → 2.0 remaining, +1°C/hr warming)
  Step 2: establish field_shelter (costs 2.0 equipment → 0.0 remaining, slows temp drift)
  Step 3: deploy water
  Step 4: deploy food
  Step 5: allocate food
  Step 6: allocate food
  Ongoing: deploy water/food as needed, triage trauma at hour 24, extract ONLY after hour 48
  CRITICAL: Do NOT use extract before hour 48 in Task 3 — it ends the mission early!

CORRECT FULL SEQUENCE FOR TASKS 1 AND 2:
  Step 1: {"action_type": "deploy", "resource_type": "water"}          ← MANDATORY FIRST
  Step 2: {"action_type": "deploy", "resource_type": "equipment"}
  Step 3: {"action_type": "establish", "structure_type": "field_shelter"}  ← needs equipment >= 2.0
  Step 4: {"action_type": "deploy", "resource_type": "food"}
  Step 5: {"action_type": "allocate", "item": "food", "quantity": 1.0}    ← distribute rations
  Step 6: {"action_type": "allocate", "item": "food", "quantity": 1.0}    ← do this TWICE
  Step 7: {"action_type": "standby", "duration": 2}                       ← if capacity < 0.25
  Step 8+: {"action_type": "extract", "signal_method": "radio"}           ← request rescue

CRITICAL RULES:
  - "deploy" acquires resources INTO inventory (water/food/equipment/medical)
  - "establish field_shelter" requires resource_inventory.equipment >= 2.0
  - "allocate" distributes FROM inventory TO personnel — item: "food"|"water"|"medicine" ONLY
  - "allocate" does NOT accept "equipment" — equipment is ONLY used by "establish"
  - "extract" is the rescue signal action — use it to complete the mission
  - "standby" recovers team_capacity — use it when capacity drops below 0.25

ACTION SCHEMA:
{
  "action_type": "<deploy|establish|relocate|standby|triage|extract|allocate|assess>"
}

PARAMS:
  deploy:    "resource_type": "water"|"food"|"equipment"|"medical"
  establish: "structure_type": "field_shelter"|"signal_fire"|"extraction_point"
  relocate:  "direction": "N"|"S"|"E"|"W",  "distance": 1.0-5.0
  standby:   "duration": 1-8
  triage:    "condition": "trauma"|"hypothermia"|"dehydration"
  extract:   "signal_method": "flare"|"mirror"|"radio"|"beacon"
  allocate:  "item": "food"|"water"|"medicine",  "quantity": 0.5-2.0  (NOT equipment)
  assess:    "target": "terrain"|"weather"|"personnel"|"resources"

KEY FIELD MEANINGS:
  mission_viability: 1.0=optimal, 0.0=mission failure
  hydration_deficit: 0.0=hydrated, 1.0=critically dehydrated — CRITICAL above 0.80
  nutrition_deficit: 0.0=adequate, 1.0=critical — CRITICAL above 0.85
  team_capacity:     1.0=full operational, 0.0=non-operational — use standby to recover
  core_temperature:  hypothermia onset below 35°C — use triage hypothermia or signal_fire
  resource_inventory.equipment: must be >= 2.0 before establish field_shelter

Respond ONLY with valid JSON. No markdown, no extra text."""


# ── LLM Calls ─────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Raised when the LLM is unavailable and no fallback should be attempted."""
    pass


VALID_ACTION_TYPES = {"deploy", "establish", "relocate", "standby", "triage", "extract", "allocate", "assess"}


def _parse_action_list(text: str) -> list:
    """Parse LLM response into a list of action dicts. Handles arrays and single objects."""
    # Strip markdown fences
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else parts[0]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    parsed = json.loads(text)
    if isinstance(parsed, list):
        return [a for a in parsed if isinstance(a, dict) and a.get("action_type") in VALID_ACTION_TYPES]
    if isinstance(parsed, dict) and parsed.get("action_type") in VALID_ACTION_TYPES:
        return [parsed]
    return []


def get_action_plan(obs: dict, task_id: int, step: int, n: int) -> list:
    """
    Request N sequential actions from the LLM in a single call.
    Returns a list of up to N action dicts.
    Raises LLMError on 402 or 3 consecutive failures.
    """
    max_hr = obs.get("max_steps", 5)
    elapsed = obs.get("mission_elapsed_hours", 0)

    # Task 3 guard: tell LLM exactly when extract is allowed
    extract_note = ""
    if task_id == 3:
        extract_note = (
            f"\nWARNING: Do NOT include extract until hour 48+ "
            f"(currently hour {elapsed}). Plan {n} survival actions only if hour < 48."
        )

    prompt = (
        f"TASK {task_id} | Step {step} | Hour {elapsed}/{max_hr}\n"
        f"CURRENT STATE:\n{json.dumps(obs, indent=2)}\n\n"
        f"Plan the NEXT {n} sequential actions starting from step {step}."
        f"{extract_note}\n"
        f"Return ONLY a JSON array of {n} action objects. No explanation."
    )

    last_error = None
    for attempt in range(3):
        try:
            response = llm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.2,
                max_tokens=max(300, n * 60 + 50),
                stream=False,
            )
            text = (response.choices[0].message.content or "").strip()
            actions = _parse_action_list(text)
            if actions:
                return actions[:n]
            raise ValueError(f"LLM returned empty or invalid action list: {text[:200]}")

        except Exception as exc:
            last_error = str(exc)
            if "402" in last_error:
                raise LLMError(
                    f"LLM_UNAVAILABLE: HF Inference credits depleted (402). "
                    f"Set HF_TOKEN with active credits. Error: {last_error}"
                )
            if attempt < 2:
                time.sleep(1)

    raise LLMError(f"LLM_UNAVAILABLE: 3 consecutive failures. Last: {last_error}")


# ── Graders ───────────────────────────────────────────────────────────────────

def _clamp(score: float) -> float:
    """Clamp score to strictly open interval (0, 1) as required by Phase 2 validation."""
    return round(max(0.001, min(0.999, score)), 4)


def grade_task1(step_log: list) -> float:
    """
    Task 1: Resource Triage (5 steps).
    1.0 = deploy water first + establish shelter before food
    0.5 = deploy water first only
    0.0 = water not deployed first
    """
    if not step_log:
        return _clamp(0.0)

    first = step_log[0]["action"]
    water_first = (first.get("action_type") == "deploy" and first.get("resource_type") == "water")
    if not water_first:
        return _clamp(0.0)

    actions = [s["action"] for s in step_log[1:]]
    shelter_idx = next(
        (i for i, a in enumerate(actions)
         if a.get("action_type") == "establish" and a.get("structure_type") == "field_shelter"),
        None,
    )
    food_idx = next(
        (i for i, a in enumerate(actions)
         if a.get("action_type") == "deploy" and a.get("resource_type") == "food"),
        None,
    )

    if shelter_idx is not None and food_idx is not None:
        return _clamp(1.0) if shelter_idx < food_idx else _clamp(0.5)
    return _clamp(0.5)


def grade_task2(step_log: list, final_obs: dict) -> float:
    """
    Task 2: 24-Hour Operational Arc. 4 × 0.25:
      shelter established < hr8 | water deployed ≤ hr6 | food allocated ≥ 2x | extraction requested
    """
    score = 0.0
    shelter_at  = None
    water_at    = None
    food_count  = 0
    signalled   = False

    for s in step_log:
        a = s["action"]
        t = s["observation"].get("mission_elapsed_hours", 0)

        if (a.get("action_type") == "establish"
                and a.get("structure_type") == "field_shelter"
                and shelter_at is None):
            shelter_at = t

        if (a.get("action_type") == "deploy"
                and a.get("resource_type") == "water"
                and water_at is None):
            water_at = t

        if a.get("action_type") == "allocate" and a.get("item") == "food":
            food_count += 1

        if a.get("action_type") == "extract":
            signalled = True

    if shelter_at is not None and shelter_at < 8:
        score += 0.25
    if water_at is not None and water_at <= 6:
        score += 0.25
    if food_count >= 2:
        score += 0.25
    if signalled:
        score += 0.25

    return _clamp(score)


def grade_task3(step_log: list, final_obs: dict) -> float:
    """
    Task 3: Multi-Day Rescue Arc.
    0–0.40: survival_steps/120 × 0.4
    +0.20:  trauma treated within hr36
    +0.20:  base camp fortified before hr72
    +0.20:  extraction signalled after hr48
    """
    score = 0.4 * (len(step_log) / 120.0)

    # Injury treated within 36 hours
    for s in step_log:
        a = s["action"]
        t = s["observation"].get("mission_elapsed_hours", 0)
        if t <= 36:
            if (a.get("action_type") == "triage" and a.get("condition") == "trauma") or \
               (a.get("action_type") == "allocate" and a.get("item") == "medicine"):
                score += 0.2
                break

    # Base camp established before the storm (hr72)
    for s in step_log:
        if s["observation"].get("mission_elapsed_hours", 0) >= 72:
            break
        if s["observation"].get("base_camp_status") in ("established", "fortified"):
            score += 0.2
            break

    # Extraction signalled after rescue window opens (hr48)
    for s in step_log:
        if (s["action"].get("action_type") == "extract"
                and s["observation"].get("mission_elapsed_hours", 0) >= 48):
            score += 0.2
            break

    return _clamp(score)


# ── Dashboard (stderr only — stdout reserved for judge log lines) ─────────────

def _bar(v: float, w: int = 10, inv: bool = False) -> str:
    pct = (1.0 - v) if inv else v
    f = int(pct * w)
    return "█" * f + "░" * (w - f)


def render_dashboard(obs: dict, task_id: int, step: int, last_reward: float):
    if not RICH:
        return

    viability  = obs.get("mission_viability", 1.0)
    hydration  = obs.get("hydration_deficit", 0.0)
    nutrition  = obs.get("nutrition_deficit", 0.0)
    capacity   = obs.get("team_capacity", 1.0)
    core_temp  = obs.get("core_temperature", 37.0)
    camp       = obs.get("base_camp_status", "none")
    weather    = obs.get("weather_conditions", "clear")
    inv        = obs.get("resource_inventory", {})
    elapsed    = obs.get("mission_elapsed_hours", 0)
    max_steps  = obs.get("max_steps", 5)
    sitrep     = obs.get("sitrep", "")
    incidents  = obs.get("active_incidents", [])

    weather_icon = {"clear": "CLEAR", "rain": "RAIN", "storm": "STORM", "snow": "SNOW"}.get(weather, weather.upper())

    vitals = Table.grid(padding=(0, 1))
    vitals.add_column(min_width=18)
    vitals.add_column(min_width=12)
    vitals.add_column(min_width=6)

    vc = "red" if viability < 0.3 else ("yellow" if viability < 0.6 else "green")
    vitals.add_row(f"[{vc}]MISSION VIABILITY[/]", f"[{vc}]{_bar(viability)}[/]", f"[{vc}]{viability:.0%}[/]")

    hc = "red" if hydration > 0.7 else ("yellow" if hydration > 0.4 else "cyan")
    vitals.add_row(f"[{hc}]HYDRATION DEFICIT[/]", f"[{hc}]{_bar(hydration, inv=True)}[/]", f"[{hc}]{hydration:.0%}[/]")

    nc = "red" if nutrition > 0.7 else ("yellow" if nutrition > 0.4 else "white")
    vitals.add_row(f"[{nc}]NUTRITION DEFICIT[/]", f"[{nc}]{_bar(nutrition, inv=True)}[/]", f"[{nc}]{nutrition:.0%}[/]")

    cc = "red" if capacity < 0.3 else ("yellow" if capacity < 0.6 else "green")
    vitals.add_row(f"[{cc}]TEAM CAPACITY[/]", f"[{cc}]{_bar(capacity)}[/]", f"[{cc}]{capacity:.0%}[/]")

    tc = "cyan" if core_temp < 35 else ("red" if core_temp > 39 else "white")
    vitals.add_row(f"[{tc}]CORE TEMPERATURE[/]", "", f"[{tc}]{core_temp:.1f}°C[/]")

    camp_c = "green" if camp == "fortified" else ("yellow" if camp == "established" else "red")
    status_line = f"[{camp_c}]BASE CAMP: {camp.upper()}[/]  [{('yellow' if weather != 'clear' else 'white')}]{weather_icon}[/]"
    if incidents:
        status_line += f"  [bold red]INCIDENTS: {', '.join(incidents).upper()}[/]"

    inv_line = (
        f"Water {inv.get('water', 0):.1f}L  "
        f"Food {inv.get('food', 0):.1f}  "
        f"Equipment {inv.get('equipment', 0):.1f}  "
        f"Medical {inv.get('medical', 0):.1f}"
    )

    rw_c = "green" if last_reward > 0.3 else ("yellow" if last_reward > 0.1 else "red")

    console.rule(f"[bold cyan]TASK {task_id}  STEP {step}/{max_steps}  HOUR {elapsed}[/]")
    console.print(Panel(vitals, title="OPERATIONAL STATUS", border_style="blue"))
    console.print(status_line)
    console.print(f"[dim]{inv_line}[/]")
    if sitrep:
        console.print(f"[italic dim]{sitrep}[/]")
    console.print(f"[{rw_c}]Step Reward: +{last_reward:.4f}[/]")


# ── Episode Runner ────────────────────────────────────────────────────────────

def run_task(task_id: int) -> tuple:
    """Run one operational task. Returns (final_score, success)."""
    cfg = TASKS[task_id]
    task_name = cfg["name"]
    threshold = cfg["threshold"]

    log_start(task_name, MODEL_NAME)

    step_log: list = []
    rewards:  list = []
    steps_taken = 0
    score = 0.001
    success = False

    try:
        # Use a session so cookies persist — keeps reset and step on the SAME server session
        http = requests.Session()

        # Reset episode
        resp = http.post(f"{ENV_BASE_URL}/reset", json={"task_id": task_id}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        obs  = data.get("observation", data)
        done = data.get("done", False)

        render_dashboard(obs, task_id, step=0, last_reward=0.0)

        step = 0
        llm_error: Optional[str] = None
        action_queue: list = []
        chunk_size = CHUNK_SIZES.get(task_id, 5)

        while not done and step < cfg["max_steps"]:
            step += 1

            # Refill action queue with a batch LLM call when empty
            if not action_queue:
                remaining = cfg["max_steps"] - step + 1
                n = min(chunk_size, remaining)
                try:
                    action_queue = get_action_plan(obs, task_id, step, n)
                    print(
                        f"[INFO] LLM planned {len(action_queue)} actions "
                        f"(step {step}–{step + len(action_queue) - 1})",
                        file=sys.stderr,
                    )
                except LLMError as llm_exc:
                    llm_error = str(llm_exc)
                    print(f"[WARN] LLM unavailable (step {step}), using fallback: {llm_error}", file=sys.stderr)
                    action_queue = [get_fallback_action(step + i, task_id) for i in range(n)]

            action_dict = action_queue.pop(0) if action_queue else get_fallback_action(step, task_id)

            # Task 3 safety: never extract before hour 48 (would end mission early → low score)
            if task_id == 3 and action_dict.get("action_type") == "extract":
                elapsed_hrs = obs.get("mission_elapsed_hours", 0)
                if elapsed_hrs < 48:
                    action_dict = get_fallback_action(step, task_id)

            action_str = json.dumps(action_dict, separators=(",", ":"))
            step_error: Optional[str] = None

            try:
                step_resp = http.post(
                    f"{ENV_BASE_URL}/step",
                    json={"action": action_dict},
                    timeout=30,
                )
                step_resp.raise_for_status()
                step_data = step_resp.json()
                new_obs   = step_data.get("observation", step_data)
                reward    = float(step_data.get("reward", 0.0) or 0.0)
                done      = bool(step_data.get("done", False))
            except Exception as exc:
                new_obs    = obs
                reward     = 0.0
                step_error = str(exc)
                # 422 = invalid action schema — don't end episode, skip step
                if "422" not in step_error:
                    done = True

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=step_error)

            step_log.append({
                "step":        step,
                "action":      action_dict,
                "observation": new_obs,
                "reward":      reward,
                "done":        done,
            })

            render_dashboard(new_obs, task_id, step=step, last_reward=reward)
            obs = new_obs

        # Grade episode
        final_obs = obs
        if task_id == 1:
            score = grade_task1(step_log)
        elif task_id == 2:
            score = grade_task2(step_log, final_obs)
        else:
            score = grade_task3(step_log, final_obs)

        success = score >= threshold

    except Exception as exc:
        print(f"[ERR] Task {task_id} failed: {exc}", file=sys.stderr)

    log_end(success=success, steps=steps_taken, rewards=rewards)

    if RICH:
        c = "green" if success else "red"
        console.print(
            Panel(
                f"[bold {c}]Task {task_id} ({task_name})[/]  Score: {score:.4f}  {'PASS' if success else 'FAIL'}",
                border_style=c,
            )
        )

    return score, success


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if RICH:
        console.rule("[bold blue]SAR Coordinator Environment — Inference Run[/]")
        console.print(f"Model:  [cyan]{MODEL_NAME}[/]")
        console.print(f"Server: [cyan]{ENV_BASE_URL}[/]")

    all_scores = []
    for task_id in [1, 2, 3]:
        score, _ = run_task(task_id)
        all_scores.append(score)
        print()

    avg = sum(all_scores) / 3

    # Final summary to stdout
    print(
        f"# Task 1: {all_scores[0]:.4f}  "
        f"Task 2: {all_scores[1]:.4f}  "
        f"Task 3: {all_scores[2]:.4f}  "
        f"Avg: {avg:.4f}"
    )

    if RICH:
        t = Table(title="Final Operational Scores")
        t.add_column("Task")
        t.add_column("Difficulty")
        t.add_column("Score", justify="right")
        t.add_column("Result")
        threshold = [0.5, 0.5, 0.4]
        labels = ["Resource Triage", "24hr Arc", "Multi-Day Rescue"]
        for i, (s, th, lbl) in enumerate(zip(all_scores, threshold, labels), 1):
            result = "[green]PASS[/]" if s >= th else "[red]FAIL[/]"
            t.add_row(str(i), lbl, f"{s:.4f}", result)
        t.add_row("Avg", "", f"[bold]{avg:.4f}[/]", "")
        console.print(t)
