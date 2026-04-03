#!/usr/bin/env python3
"""
SAR Coordinator Environment — Inference Script
===============================================
Runs an LLM agent through all 3 operational tasks as a SAR Coordinator.

Required environment variables:
    API_BASE_URL  — LLM inference endpoint (OpenAI-compatible)
    MODEL_NAME    — Model identifier
    HF_TOKEN      — HuggingFace token (also reads TOKEN from .env as fallback)
    ENV_BASE_URL  — SAR Coordinator server URL (default: http://localhost:8000)

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
                    os.environ.setdefault(k.strip(), v.strip())

_load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
ENV_BASE_URL = os.getenv("ENV_BASE_URL",  "http://localhost:8000")

# Accept both HF_TOKEN (hackathon standard) and TOKEN (our .env variable)
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("TOKEN", "")

if not HF_TOKEN:
    print("[WARN] No HF_TOKEN or TOKEN found. LLM calls may fail.", file=sys.stderr)

# Task config — matches SAREnvironment task definitions
TASKS = {
    1: {"name": "resource-triage",         "max_steps": 5,   "threshold": 0.5},
    2: {"name": "24hr-operational-arc",    "max_steps": 24,  "threshold": 0.5},
    3: {"name": "multiday-rescue-arc",     "max_steps": 120, "threshold": 0.4},
}

# ── OpenAI Client ─────────────────────────────────────────────────────────────
from openai import OpenAI
import requests

llm_client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN or "dummy",
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


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI decision-support system acting as a SAR (Search & Rescue) field coordinator.
You receive the current operational state as JSON and must respond with ONE action JSON object.
No explanation outside the JSON — just the object.

SAR TRIAGE DOCTRINE — STRICT PRIORITY ORDER:
  1. WATER FIRST — ALWAYS deploy water as the very first action of any mission (SAR doctrine: hydration saves lives)
  2. SHELTER — establish field_shelter once equipment >= 2.0 in inventory
  3. NUTRITION — deploy food when nutrition_deficit > 0.40, allocate food to team
  4. REST — use standby (duration=2) when team_capacity < 0.30 to recover personnel
  5. EXTRACTION — extract with radio/flare after shelter established and team stable

TASK 3 ARCTIC EXCEPTION — if core_temperature < 36.0 on step 1:
  Step 1: deploy equipment   (need materials for signal_fire)
  Step 2: establish signal_fire   (CRITICAL — stops hypothermia, +1°C/hr)
  Step 3: deploy water   (then follow normal doctrine)

ALL OTHER TASKS — STEP 1: {"action_type": "deploy", "resource_type": "water"}
This is mandatory SAR doctrine. Water is always the first priority.

CRITICAL RULES:
  - "deploy" acquires resources INTO inventory (water/food/equipment/medical)
  - "establish field_shelter" requires resource_inventory.equipment >= 2.0
  - "allocate" distributes FROM inventory TO personnel — item: "food"|"water"|"medicine" ONLY
  - "allocate" does NOT accept "equipment" — equipment is ONLY used by "establish"
  - "standby" recovers team_capacity — use it when capacity drops below 0.25
  - Check resource_inventory values before establish or allocate

CORRECT FULL SEQUENCE:
  Step 1: deploy water        (SAR doctrine: water first)
  Step 2: deploy equipment    (get materials)
  Step 3: establish field_shelter  (uses 2.0 equipment)
  Step 4: deploy food / allocate water to recover team
  Step 5: standby duration=2  (if capacity < 0.30)
  Step 6+: extract / triage as needed

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


# ── LLM Call ──────────────────────────────────────────────────────────────────

def get_action(obs: dict, task_id: int, step: int) -> tuple:
    """Call LLM for next operational order. Returns (action_dict, error_string_or_None)."""
    prompt = (
        f"TASK {task_id} | Step {step} | Hour {obs.get('mission_elapsed_hours', 0)}/{obs.get('max_steps', 5)}\n"
        f"OPERATIONAL STATE:\n{json.dumps(obs, indent=2)}\n\nIssue next operational order:"
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
                temperature=0.3,
                max_tokens=300,
                stream=False,
            )
            text = (response.choices[0].message.content or "").strip()

            # Strip markdown fences if model adds them
            if "```" in text:
                parts = text.split("```")
                text = parts[1] if len(parts) > 1 else parts[0]
                if text.startswith("json"):
                    text = text[4:]

            action = json.loads(text.strip())
            return action, None

        except Exception as exc:
            last_error = str(exc)
            if attempt < 2:
                time.sleep(1)

    # Safe fallback after 3 failed attempts
    return {"action_type": "assess", "target": "personnel"}, last_error


# ── Graders ───────────────────────────────────────────────────────────────────

def grade_task1(step_log: list) -> float:
    """
    Task 1: Resource Triage (5 steps).
    1.0 = deploy water first + establish shelter before food
    0.5 = deploy water first only
    0.0 = water not deployed first
    """
    if not step_log:
        return 0.0

    first = step_log[0]["action"]
    water_first = (first.get("action_type") == "deploy" and first.get("resource_type") == "water")
    if not water_first:
        return 0.0

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
        return 1.0 if shelter_idx < food_idx else 0.5
    return 0.5


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

    return round(score, 4)


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

    return round(score, 4)


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
    score = 0.0
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
        while not done and step < cfg["max_steps"]:
            step += 1
            action_dict, err = get_action(obs, task_id, step)

            action_str = json.dumps(action_dict, separators=(",", ":"))

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
                step_error = err
            except Exception as exc:
                new_obs    = obs
                reward     = 0.0
                # 422 = invalid action from LLM — don't end episode, just skip this step
                step_error = str(exc)
                if "422" in step_error:
                    done = False
                else:
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
