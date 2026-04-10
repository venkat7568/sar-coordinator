"""
SAR Coordinator Environment — Core Simulation

Models the decision problem faced by a Search & Rescue coordinator in the field.
An isolated team must maintain operational capacity long enough to execute extraction.

Physics are grounded in real SAR science:
  - Hydration deficit escalates faster than nutrition deficit (0.06/hr vs 0.04/hr)
  - Hypothermia onset below 35°C core temperature — primary killer in cold terrain
  - Field shelter reduces both temperature drift and team energy expenditure
  - Dynamic incidents (trauma events, storms) test adaptive decision-making
  - Extraction requires both signal quality and being within the rescue window

Three tasks mirror real operational complexity:
  Task 1 — Field resource prioritization (SAR first-responder triage)
  Task 2 — 24-hour operational arc (multi-phase mission planning)
  Task 3 — Extended field deployment with dynamic incident response
"""

import random
from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SARAction, SARObservation, BaseStatus, WeatherCondition
except ImportError:
    from models import SARAction, SARObservation, BaseStatus, WeatherCondition


# ── Operational Physics Constants ─────────────────────────────────────────────

# Deficit accumulation per operational hour (1 step = 1 hour)
NUTRITION_DECAY  = 0.04   # per hour without food resupply
HYDRATION_DECAY  = 0.06   # per hour without water — higher rate, mirrors physiology
CAPACITY_DECAY   = 0.05   # per hour of sustained operations

# Core temperature drift toward ambient per hour (without shelter)
TEMP_DRIFT_RATE  = 0.5    # °C/hr

# Weather impact multipliers on operational parameters
WEATHER_MODIFIERS = {
    "clear": {"nutrition": 1.0, "hydration": 1.2, "capacity": 1.0, "temp_drift": 1.0},
    "rain":  {"nutrition": 1.0, "hydration": 0.8, "capacity": 1.2, "temp_drift": 1.5},
    "storm": {"nutrition": 1.0, "hydration": 0.8, "capacity": 1.5, "temp_drift": 2.5},
    "snow":  {"nutrition": 1.2, "hydration": 1.1, "capacity": 1.3, "temp_drift": 3.0},
}

# Base camp reduces temperature drift and team capacity drain
SHELTER_MODIFIERS = {
    "none":        {"temp_drift": 1.0, "capacity": 1.0},
    "established": {"temp_drift": 0.5, "capacity": 0.8},
    "fortified":   {"temp_drift": 0.2, "capacity": 0.6},
}

# Terrain ambient temperatures (°C)
TERRAIN_AMBIENT_TEMP = {
    "jungle":  28.0,
    "arctic": -15.0,
    "forest":  15.0,
}

# Critical thresholds — triggers mission viability damage
CRITICAL_NUTRITION  = 0.85
CRITICAL_HYDRATION  = 0.80
CRITICAL_TEMP_LOW   = 34.0   # hypothermia
CRITICAL_TEMP_HIGH  = 40.0   # hyperthermia

# Viability damage rates when critical
DAMAGE_NUTRITION    = 0.03   # per hour
DAMAGE_HYDRATION    = 0.05
DAMAGE_TEMP         = 0.04
DAMAGE_TRAUMA       = 0.02   # per hour if trauma is untreated

# Resource acquisition yields per deploy action
DEPLOY_YIELDS = {
    "water":     {"base": 2.0, "storm_penalty": 0.3},
    "food":      {"base": 1.5, "storm_penalty": 0.5},
    "equipment": {"base": 3.0, "storm_penalty": 0.4},
    "medical":   {"base": 1.0, "storm_penalty": 0.0},
}

# Effects of allocating resources to personnel
ALLOCATE_EFFECTS = {
    "food":     {"nutrition_deficit": -0.25, "team_capacity": 0.10},
    "water":    {"hydration_deficit": -0.30},
    "medicine": {"mission_viability":  0.20},
}


# ── Situation Reports (SITREP) ────────────────────────────────────────────────
# Professional field communications — not game flavor text.

DEPLOY_SITREP = {
    "water": {
        "forest": "Team reports water extraction complete. Source confirmed potable. Supply secured.",
        "jungle": "Water resupply successful. Natural reservoir located. Purification protocol followed.",
        "arctic": "Snow melt operation complete. Fuel reserves used. Water supply replenished.",
    },
    "food": {
        "forest": "Foraging team returned with field rations. Caloric deficit partially offset.",
        "jungle": "Edible vegetation identified and harvested. Nutritional assessment positive.",
        "arctic": "Limited food source located under snowpack. Caloric contribution marginal.",
    },
    "equipment": {
        "forest": "Field materials acquired. Construction supplies adequate for infrastructure build.",
        "jungle": "Dense vegetation yields structural material. Equipment cache established.",
        "arctic": "Frozen timber recovered. Equipment degraded but serviceable for construction.",
    },
    "medical": {
        "forest": "Medical supplies sourced from field kit. Inventory partially restocked.",
        "jungle": "Local plant compounds identified with antiseptic properties. Medical cache augmented.",
        "arctic": "Emergency medical pack located in gear. Supplies accounted for.",
    },
}

ESTABLISH_SITREP = {
    "field_shelter": {
        "established": "Field shelter established. Wind and precipitation exposure reduced. Team status improving.",
        "fortified":   "Base camp reinforced to fortified status. Thermal protection now optimal.",
        "already":     "Base camp already at maximum fortification. No further action required.",
        "insufficient":"Insufficient equipment. Minimum 2 units required for shelter construction.",
    },
    "signal_fire":        "Signal fire established. Smoke column visible at elevation. Extraction teams alerted.",
    "extraction_point":   "Extraction point marked and signalled. Coordinates transmitted to command.",
    "insufficient_equip": "Insufficient equipment for construction operation.",
}

RELOCATE_SITREP = {
    "N": "Team advanced north {d}km. New sector acquired. Terrain intelligence updated.",
    "S": "Team advanced south {d}km. Sector repositioned. Perimeter assessment underway.",
    "E": "Team moved east {d}km. Forward position established. Reconnaissance complete.",
    "W": "Team moved west {d}km. Flanking position taken. Sector map updated.",
}

STANDBY_SITREP = [
    "Team holding position. Operational capacity recovering. Monitoring situation.",
    "Standby order issued. Personnel resting in place. Status checks continuing.",
    "Position held for {d} hours. Team recovery confirmed. Ready to resume operations.",
]

TRIAGE_SITREP = {
    "trauma":      "Trauma protocol administered. Wound stabilised. Personnel returned to operational status.",
    "hypothermia": "Hypothermia intervention complete. Core temperature recovering. Monitoring vital signs.",
    "dehydration": "Oral rehydration administered. Hydration levels improving. Team capacity restored.",
}

EXTRACT_SITREP = {
    "flare":   "Flare deployed. Visual signal confirmed. Extraction team alerted.",
    "mirror":  "Heliograph signal transmitted. Visibility {v}. Signal reception {r}.",
    "radio":   "Radio contact established. Position coordinates transmitted. ETA requested.",
    "beacon":  "Distress beacon activated. Signal broadcasting on emergency frequency.",
}

ASSESS_SITREP = {
    "terrain":   "Sector reconnaissance complete. Terrain intelligence updated across all visible quadrants.",
    "weather":   "Meteorological assessment: {w} conditions, severity {s:.0%}. Ambient {a}°C. Core temp {t:.1f}°C.",
    "personnel": "Personnel status: Viability {v:.0%} | Hydration deficit {h:.0%} | Nutrition deficit {n:.0%} | Capacity {c:.0%} | Temp {t:.1f}°C{inj}",
    "resources": "Resource inventory: Food {f:.1f}u | Water {w:.1f}L | Equipment {e:.1f}u | Medical {m:.1f} kits",
}


# ── Environment ───────────────────────────────────────────────────────────────

class SAREnvironment(Environment):
    """
    Search & Rescue Coordinator RL Environment.

    Trains AI agents to make optimal operational decisions under resource
    constraints, partial terrain intelligence, and dynamic field incidents.

    Compatible with standard RL training frameworks via OpenEnv spec.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = False

    # Class-level shared state — the HTTP server creates a fresh instance per request,
    # so we use class variables (persisted across instances) + properties to expose them
    # as self._world / self._rng without changing any downstream code.
    _shared_world: dict = {}
    _shared_rng: random.Random = random.Random()

    @property
    def _world(self) -> dict:
        return SAREnvironment._shared_world

    @_world.setter
    def _world(self, value: dict) -> None:
        SAREnvironment._shared_world = value

    @property
    def _rng(self) -> random.Random:
        return SAREnvironment._shared_rng

    @_rng.setter
    def _rng(self, value: random.Random) -> None:
        SAREnvironment._shared_rng = value

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)

    # ── Public API ────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: int = 1,
        **kwargs,
    ) -> SARObservation:
        """Initialise a new mission episode. task_id selects difficulty (1/2/3)."""
        self._rng = random.Random(seed)
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._world = self._init_world(task_id)
        return self._make_observation(
            incidents=[],
            sitrep="Mission initiated. Team deployed to field. Awaiting first operational order."
        )

    def step(self, action: SARAction, **kwargs) -> SARObservation:
        """Execute one operational decision (1 mission hour)."""
        if not self._world:
            self._world = self._init_world(1)
        self._state.step_count += 1

        incidents = self._apply_scheduled_incidents()
        sitrep    = self._dispatch_action(action)
        # Track whether action had any real effect (for negative reward)
        self._world["_last_action_failed"] = any(
            phrase in sitrep for phrase in [
                "Insufficient", "No ", "not specified", "not required",
                "already at maximum", "No active trauma", "within acceptable",
                "within nominal"
            ]
        )
        self._apply_physics()
        self._update_viability()

        reward = self._compute_reward(action)
        self._world["cumulative_reward"] += reward
        self._world["step_rewards"].append(reward)

        done = self._check_done()
        obs  = self._make_observation(incidents=incidents, sitrep=sitrep)
        obs.done   = done
        obs.reward = round(reward, 4)
        return obs

    @property
    def state(self) -> State:
        return self._state

    # ── World Initialization ──────────────────────────────────────────────────

    def _init_world(self, task_id: int) -> dict:
        base = {
            # Personnel
            "nutrition_deficit":  0.0,
            "hydration_deficit":  0.0,
            "team_capacity":      1.0,
            "mission_viability":  1.0,
            "core_temperature":   37.0,

            # Field
            "base_camp_status":   "none",
            "weather_conditions": "clear",
            "weather_severity":   0.0,
            "pos":                [2, 2],
            "visited_sectors":    set(),

            # Resources
            "resource_inventory": {"food": 0.0, "water": 0.0, "equipment": 0.0, "medical": 0.0},

            # Mission
            "extraction_requested":      False,
            "extraction_signal_quality": 0.0,
            "extraction_attempts":       0,
            "extraction_executed":       False,
            "distance_to_extraction":    10.0,

            # Incident tracking (grader accumulators)
            "shelter_established_at":    None,
            "water_secured_at":          None,
            "signal_fire_active":        False,
            "trauma_active":             False,
            "trauma_treated_at":         None,
            "storm_prep_done":           False,

            # Episode
            "task_id":        task_id,
            "event_schedule": {},
            "active_events":  [],

            # Reward tracking
            "cumulative_reward": 0.0,
            "step_rewards":      [],
        }

        if task_id == 1:
            # Easy — resource prioritization, forest terrain, 5 steps
            base.update({
                "max_steps":             5,
                "terrain":               "forest",
                "ambient_temp":          TERRAIN_AMBIENT_TEMP["forest"],
                "sector_grid":           self._make_forest_grid(),
                "extraction_window_from": 99,   # extraction not the objective in task 1
            })

        elif task_id == 2:
            # Medium — 24hr operational arc, jungle, storm forecast hour 8
            base.update({
                "max_steps":             24,
                "terrain":               "jungle",
                "ambient_temp":          TERRAIN_AMBIENT_TEMP["jungle"],
                "core_temperature":      38.5,   # heat stress risk — drifts toward 28°C jungle ambient
                "nutrition_deficit":     0.2,
                "resource_inventory":    {"food": 0.0, "water": 0.5, "equipment": 0.0, "medical": 0.0},
                "sector_grid":           self._make_jungle_grid(),
                "event_schedule":        {8: "rain_onset"},
                "extraction_window_from": 20,
                "distance_to_extraction": 8.0,
            })

        elif task_id == 3:
            # Hard — extended deployment, arctic, trauma day 2, storm day 4
            base.update({
                "max_steps":             120,
                "terrain":               "arctic",
                "ambient_temp":          TERRAIN_AMBIENT_TEMP["arctic"],
                "core_temperature":      35.5,   # borderline hypothermia from mission start
                "resource_inventory":    {"food": 0.0, "water": 1.0, "equipment": 3.0, "medical": 1.0},
                "sector_grid":           self._make_arctic_grid(),
                "event_schedule":        {
                    24: "trauma_event",    # day 2
                    72: "storm_onset",     # day 4
                    80: "storm_clearance",
                },
                "extraction_window_from": 48,    # day 3 onward
                "distance_to_extraction": 15.0,
            })

        return base

    def _make_forest_grid(self):
        g = [["forest"] * 5 for _ in range(5)]
        g[2][3] = "river"  # water source east
        g[1][2] = "plain"  # food source north
        return g

    def _make_jungle_grid(self):
        g = [["jungle"] * 5 for _ in range(5)]
        g[2][3] = "river"
        g[0][2] = "plain"
        return g

    def _make_arctic_grid(self):
        g = [["snow"] * 5 for _ in range(5)]
        g[3][2] = "mountain"  # equipment source south
        g[2][1] = "forest"    # timber west
        return g

    # ── Incident Schedule ─────────────────────────────────────────────────────

    def _apply_scheduled_incidents(self) -> list[str]:
        w = self._world
        t = w.get("mission_elapsed_hours", 0)
        incidents = []

        event = w.get("event_schedule", {}).get(t)
        if event:
            incidents.append(event)
            if event == "rain_onset":
                w["weather_conditions"] = "rain"
                w["weather_severity"]   = 0.6
            elif event == "storm_onset":
                w["weather_conditions"] = "storm"
                w["weather_severity"]   = 1.0
                if w["base_camp_status"] == "fortified":
                    w["storm_prep_done"] = True
            elif event == "storm_clearance":
                w["weather_conditions"] = "clear"
                w["weather_severity"]   = 0.0
            elif event == "trauma_event":
                w["trauma_active"] = True

        w["active_events"] = incidents
        return incidents

    # ── Action Dispatch ───────────────────────────────────────────────────────

    def _dispatch_action(self, action: SARAction) -> str:
        handlers = {
            "deploy":    self._action_deploy,
            "establish": self._action_establish,
            "relocate":  self._action_relocate,
            "standby":   self._action_standby,
            "triage":    self._action_triage,
            "extract":   self._action_extract,
            "allocate":  self._action_allocate,
            "assess":    self._action_assess,
        }
        handler = handlers.get(action.action_type)
        if handler is None:
            return "Invalid operational order. No action taken."
        return handler(action)

    def _action_deploy(self, action: SARAction) -> str:
        w = self._world
        rt = action.resource_type
        if not rt:
            return "Deploy order incomplete. Resource type not specified."

        base_yield = DEPLOY_YIELDS[rt]["base"]
        if w["weather_conditions"] == "storm":
            base_yield *= (1 - DEPLOY_YIELDS[rt]["storm_penalty"])
        actual = round(base_yield * self._rng.uniform(0.8, 1.2), 2)

        w["resource_inventory"][rt] = round(w["resource_inventory"].get(rt, 0) + actual, 2)
        w["team_capacity"]    = max(0.0, w["team_capacity"] - 0.10)
        w["nutrition_deficit"] = min(1.0, w["nutrition_deficit"] + 0.02)

        if rt == "water" and w["water_secured_at"] is None:
            w["water_secured_at"] = w.get("mission_elapsed_hours", 0)

        terrain = w.get("terrain", "forest")
        sitrep  = DEPLOY_SITREP.get(rt, {}).get(terrain, f"Resource acquired: {actual} units of {rt}.")
        return f"{sitrep} [+{actual} {rt}]"

    def _action_establish(self, action: SARAction) -> str:
        w = self._world
        st = action.structure_type
        if not st:
            return "Establish order incomplete. Structure type not specified."

        if st == "field_shelter":
            if w["resource_inventory"].get("equipment", 0) < 2.0:
                return ESTABLISH_SITREP["field_shelter"]["insufficient"]
            w["resource_inventory"]["equipment"] = round(w["resource_inventory"]["equipment"] - 2.0, 2)
            w["team_capacity"] = max(0.0, w["team_capacity"] - 0.15)

            current = w["base_camp_status"]
            if current == "none":
                w["base_camp_status"] = "established"
                if w["shelter_established_at"] is None:
                    w["shelter_established_at"] = w.get("mission_elapsed_hours", 0)
                return ESTABLISH_SITREP["field_shelter"]["established"]
            elif current == "established":
                w["base_camp_status"] = "fortified"
                return ESTABLISH_SITREP["field_shelter"]["fortified"]
            else:
                return ESTABLISH_SITREP["field_shelter"]["already"]

        elif st == "signal_fire":
            if w["resource_inventory"].get("equipment", 0) < 1.0:
                return ESTABLISH_SITREP["insufficient_equip"]
            w["resource_inventory"]["equipment"] = round(w["resource_inventory"]["equipment"] - 1.0, 2)
            w["signal_fire_active"] = True
            w["core_temperature"]   = min(37.0, w["core_temperature"] + 2.0)
            w["team_capacity"]      = max(0.0, w["team_capacity"] - 0.10)
            return ESTABLISH_SITREP["signal_fire"]

        elif st == "extraction_point":
            if w["resource_inventory"].get("equipment", 0) < 1.0:
                return ESTABLISH_SITREP["insufficient_equip"]
            w["resource_inventory"]["equipment"] = round(w["resource_inventory"]["equipment"] - 1.0, 2)
            w["extraction_signal_quality"] = min(1.0, w["extraction_signal_quality"] + 0.15)
            w["team_capacity"] = max(0.0, w["team_capacity"] - 0.10)
            return ESTABLISH_SITREP["extraction_point"]

        return "Unrecognised structure type."

    def _action_relocate(self, action: SARAction) -> str:
        w = self._world
        d = action.direction
        dist = action.distance or 1.0
        if not d:
            return "Relocate order incomplete. Direction not specified."

        vectors = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
        dr, dc = vectors[d]
        r, c = w["pos"]
        w["pos"] = [max(0, min(4, r + dr)), max(0, min(4, c + dc))]

        w["team_capacity"]    = max(0.0, w["team_capacity"] - 0.05 * dist)
        w["nutrition_deficit"] = min(1.0, w["nutrition_deficit"] + 0.01 * dist)
        w["hydration_deficit"] = min(1.0, w["hydration_deficit"] + 0.01 * dist)

        cell_key = tuple(w["pos"])
        w["visited_sectors"].add(cell_key)
        w["distance_to_extraction"] = max(0.0,
            w["distance_to_extraction"] - dist * self._rng.uniform(0.1, 0.3))

        terrain_at = w["sector_grid"][w["pos"][0]][w["pos"][1]]
        template = RELOCATE_SITREP.get(d, "Team relocated {d}{d}km.")
        return template.format(d=int(dist)) + f" Sector terrain: {terrain_at}."

    def _action_standby(self, action: SARAction) -> str:
        w = self._world
        dur = action.duration or 1
        w["team_capacity"]    = min(1.0, w["team_capacity"] + dur * 0.15)
        w["nutrition_deficit"] = min(1.0, w["nutrition_deficit"] + dur * 0.02)
        w["mission_elapsed_hours"] = w.get("mission_elapsed_hours", 0) + (dur - 1)

        template = self._rng.choice(STANDBY_SITREP)
        return template.format(d=dur)

    def _action_triage(self, action: SARAction) -> str:
        w = self._world
        cond = action.condition
        if not cond:
            return "Triage order incomplete. Condition not specified."

        w["team_capacity"] = max(0.0, w["team_capacity"] - 0.05)

        if cond == "trauma":
            if not w["trauma_active"]:
                return "No active trauma incident in current operational area."
            if w["resource_inventory"].get("medical", 0) < 0.5:
                return "Insufficient medical supplies for trauma intervention."
            w["resource_inventory"]["medical"] = round(w["resource_inventory"]["medical"] - 0.5, 2)
            w["trauma_active"]    = False
            w["trauma_treated_at"] = w.get("mission_elapsed_hours", 0)
            w["mission_viability"] = min(1.0, w["mission_viability"] + 0.10)
            return TRIAGE_SITREP["trauma"]

        elif cond == "hypothermia":
            if w["core_temperature"] >= 35.0:
                return "Core temperature within nominal range. No hypothermia intervention required."
            if not w["signal_fire_active"]:
                return "No heat source available. Establish signal_fire before hypothermia triage."
            w["core_temperature"] = min(37.0, w["core_temperature"] + 3.0)
            return TRIAGE_SITREP["hypothermia"]

        elif cond == "dehydration":
            if w["hydration_deficit"] < 0.7:
                return "Hydration levels within acceptable operational range."
            if w["resource_inventory"].get("water", 0) < 0.5:
                return "Insufficient water reserves for rehydration protocol."
            w["resource_inventory"]["water"] = round(w["resource_inventory"]["water"] - 0.5, 2)
            w["hydration_deficit"] = max(0.0, w["hydration_deficit"] - 0.20)
            return TRIAGE_SITREP["dehydration"]

        return "Unrecognised condition."

    def _action_extract(self, action: SARAction) -> str:
        w = self._world
        method = action.signal_method
        if not method:
            return "Extract order incomplete. Signal method not specified."

        quality_map = {
            "flare":  0.30,
            "mirror": 0.25 if w["weather_conditions"] == "clear" else 0.05,
            "radio":  0.40,
            "beacon": 0.20,
        }
        quality = quality_map.get(method, 0.10)

        w["extraction_requested"]      = True
        w["extraction_signal_quality"] = min(1.0, w["extraction_signal_quality"] + quality)
        w["extraction_attempts"]      += 1
        w["team_capacity"]             = max(0.0, w["team_capacity"] - 0.05)

        t = w.get("mission_elapsed_hours", 0)
        if t >= w.get("extraction_window_from", 99) and w["extraction_signal_quality"] >= 0.5:
            w["extraction_executed"] = True

        visibility = "good" if w["weather_conditions"] == "clear" else "reduced"
        reception  = "confirmed" if quality >= 0.25 else "uncertain"
        template   = EXTRACT_SITREP.get(method, "Extraction signal transmitted.")
        return template.format(v=visibility, r=reception)

    def _action_allocate(self, action: SARAction) -> str:
        w = self._world
        item = action.item
        qty  = action.quantity or 1.0
        if not item:
            return "Allocate order incomplete. Resource type not specified."

        inv_key  = item if item != "medicine" else "medical"
        available = w["resource_inventory"].get(inv_key, 0)
        if available <= 0:
            return f"No {item} in current inventory. Deploy team to acquire."

        consumed = min(qty, available)
        w["resource_inventory"][inv_key] = round(available - consumed, 2)

        effects = ALLOCATE_EFFECTS.get(item, {})
        for stat, delta in effects.items():
            if stat == "mission_viability":
                w["mission_viability"] = min(1.0, w["mission_viability"] + delta * consumed)
            elif stat in w:
                w[stat] = max(0.0, min(1.0, w[stat] + delta * consumed))

        if item == "medicine" and w["trauma_active"]:
            w["trauma_active"]    = False
            w["trauma_treated_at"] = w.get("mission_elapsed_hours", 0)

        unit = "L" if item == "water" else "unit(s)"
        return f"Resource allocation complete. {consumed:.1f} {unit} of {item} distributed to personnel."

    def _action_assess(self, action: SARAction) -> str:
        w = self._world
        target = action.target
        if not target:
            return "Assess order incomplete. Target not specified."

        w["team_capacity"] = max(0.0, w["team_capacity"] - 0.02)

        if target == "terrain":
            for r in range(5):
                for c in range(5):
                    w["visited_sectors"].add((r, c))
            return ASSESS_SITREP["terrain"]

        elif target == "weather":
            return ASSESS_SITREP["weather"].format(
                w=w["weather_conditions"],
                s=w["weather_severity"],
                a=w["ambient_temp"],
                t=w["core_temperature"],
            )

        elif target == "personnel":
            inj = " | TRAUMA ACTIVE" if w["trauma_active"] else ""
            return ASSESS_SITREP["personnel"].format(
                v=w["mission_viability"],
                h=w["hydration_deficit"],
                n=w["nutrition_deficit"],
                c=w["team_capacity"],
                t=w["core_temperature"],
                inj=inj,
            )

        elif target == "resources":
            inv = w["resource_inventory"]
            return ASSESS_SITREP["resources"].format(
                f=inv.get("food", 0),
                w=inv.get("water", 0),
                e=inv.get("equipment", 0),
                m=inv.get("medical", 0),
            )

        return "Assessment complete."

    # ── Physics ───────────────────────────────────────────────────────────────

    def _apply_physics(self):
        """One operational hour passes — parameters evolve per field conditions."""
        w  = self._world
        wm = WEATHER_MODIFIERS[w["weather_conditions"]]
        sm = SHELTER_MODIFIERS[w["base_camp_status"]]

        w["nutrition_deficit"]  = min(1.0, w["nutrition_deficit"]  + NUTRITION_DECAY * wm["nutrition"])
        w["hydration_deficit"]  = min(1.0, w["hydration_deficit"]  + HYDRATION_DECAY * wm["hydration"])
        w["team_capacity"]      = max(0.0, w["team_capacity"]      - CAPACITY_DECAY  * wm["capacity"] * sm["capacity"])

        # Core temperature drifts toward ambient
        ambient = w["ambient_temp"]
        drift   = TEMP_DRIFT_RATE * wm["temp_drift"] * sm["temp_drift"]
        if w["core_temperature"] > ambient:
            w["core_temperature"] = max(ambient, w["core_temperature"] - drift)
        else:
            w["core_temperature"] = min(ambient, w["core_temperature"] + drift)

        # Signal fire provides active rewarming
        if w["signal_fire_active"]:
            w["core_temperature"] = min(37.0, w["core_temperature"] + 1.0)

        w["mission_elapsed_hours"] = w.get("mission_elapsed_hours", 0) + 1

    def _update_viability(self):
        """Apply mission viability damage from critical parameter states."""
        w      = self._world
        damage = 0.0

        if w["nutrition_deficit"]  >= CRITICAL_NUTRITION:  damage += DAMAGE_NUTRITION
        if w["hydration_deficit"]  >= CRITICAL_HYDRATION:  damage += DAMAGE_HYDRATION
        if w["core_temperature"]   <  CRITICAL_TEMP_LOW:   damage += DAMAGE_TEMP
        if w["core_temperature"]   >  CRITICAL_TEMP_HIGH:  damage += DAMAGE_TEMP
        if w["trauma_active"]:                              damage += DAMAGE_TRAUMA

        w["mission_viability"] = max(0.0, w["mission_viability"] - damage)

    # ── Reward ────────────────────────────────────────────────────────────────

    def _compute_reward(self, action: SARAction) -> float:
        """
        6-component dense reward — signal at every step, not just episode end.
        Weights reflect operational priorities: viability > resources > shelter > extraction > intel > optimality.
        """
        w = self._world
        r = 0.0

        # 1. Mission viability (0.30)
        r += 0.30 * w["mission_viability"]

        # 2. Resource acquisition efficiency (0.20)
        if action.action_type == "deploy" and action.resource_type:
            r += 0.20 * self._resource_efficiency(action.resource_type)

        # 3. Base camp and signal fire (0.15)
        camp_score = 0.0
        if w["base_camp_status"] == "established": camp_score = 0.50
        elif w["base_camp_status"] == "fortified": camp_score = 0.85
        if w["signal_fire_active"]: camp_score = min(1.0, camp_score + 0.25)
        r += 0.15 * camp_score

        # 4. Extraction signal quality × timing (0.15)
        timing_mult = 0.0
        if w["extraction_attempts"] > 0:
            t    = w.get("mission_elapsed_hours", 0)
            from_hr = w.get("extraction_window_from", 99)
            if t < from_hr:
                timing_mult = 0.3   # too early — outside extraction window
            else:
                hours_in = t - from_hr
                timing_mult = max(0.5, 1.0 - hours_in * 0.02)
        r += 0.15 * w["extraction_signal_quality"] * timing_mult

        # 5. Sector intelligence coverage (0.10)
        r += 0.10 * (len(w["visited_sectors"]) / 25.0)

        # 6. Decision quality heuristic (0.10)
        r += 0.10 * self._decision_quality(action)

        # 7. Penalties — negative rewards for wasted/harmful actions
        # Failed action (insufficient resources, nothing to treat, etc.)
        if w.get("_last_action_failed"):
            r -= 0.15

        # Critical dehydration not being addressed
        if w["hydration_deficit"] >= CRITICAL_HYDRATION and action.action_type != "deploy" \
                and action.action_type != "allocate":
            r -= 0.10

        # Untreated trauma persisting
        if w["trauma_active"] and action.action_type not in ("triage", "allocate"):
            r -= 0.08

        # Hypothermia not being addressed
        if w["core_temperature"] < CRITICAL_TEMP_LOW and action.action_type not in (
                "establish", "triage", "standby"):
            r -= 0.08

        return round(max(-1.0, r), 4)

    def _resource_efficiency(self, resource_type: str) -> float:
        """Score how urgently this resource was needed."""
        w = self._world
        need = {
            "water":     w["hydration_deficit"],
            "food":      w["nutrition_deficit"],
            "equipment": max(0.0, (4.0 - w["resource_inventory"].get("equipment", 0)) / 4.0),
            "medical":   1.0 if w["trauma_active"] else 0.1,
        }
        score = need.get(resource_type, 0.0)
        # Hydration bonus — mirrors SAR Rule of 3 priority
        if resource_type == "water" and w["hydration_deficit"] >= w["nutrition_deficit"]:
            score = min(1.0, score * 1.2)
        return score

    def _decision_quality(self, action: SARAction) -> float:
        """Penalise operationally suboptimal decisions."""
        w = self._world
        score = 1.0

        # Requesting extraction before the window opens wastes resources
        if action.action_type == "extract":
            t = w.get("mission_elapsed_hours", 0)
            if t < w.get("extraction_window_from", 99):
                score = 0.3

        # Standing by when critically dehydrated or trauma active delays recovery
        if action.action_type == "standby":
            if w["hydration_deficit"] > CRITICAL_HYDRATION or w["trauma_active"]:
                score = 0.2

        # Reinforcing a fortified base camp wastes equipment
        if action.action_type == "establish" and action.structure_type == "field_shelter":
            if w["base_camp_status"] == "fortified":
                score = 0.1

        # Relocating when team capacity is critically low risks collapse
        if action.action_type == "relocate" and w["team_capacity"] < 0.2:
            score = 0.3

        return score

    # ── Done Check ────────────────────────────────────────────────────────────

    def _check_done(self) -> bool:
        w = self._world
        if w["mission_viability"] <= 0.0:   return True
        if w["extraction_executed"]:         return True
        if self._state.step_count >= w["max_steps"]: return True
        return False

    # ── Observation Builder ───────────────────────────────────────────────────

    def _make_observation(self, incidents: list[str], sitrep: str) -> SARObservation:
        w   = self._world
        pos = w.get("pos", [2, 2])

        return SARObservation(
            nutrition_deficit=round(w["nutrition_deficit"], 3),
            hydration_deficit=round(w["hydration_deficit"], 3),
            team_capacity=round(w["team_capacity"], 3),
            core_temperature=round(w["core_temperature"], 2),
            mission_viability=round(w["mission_viability"], 3),
            base_camp_status=BaseStatus(w["base_camp_status"]),
            weather_conditions=WeatherCondition(w["weather_conditions"]),
            weather_severity=round(w["weather_severity"], 2),
            sector_map=self._get_visible_map(w["sector_grid"], pos, w["visited_sectors"]),
            resource_inventory={k: round(v, 2) for k, v in w["resource_inventory"].items()},
            mission_elapsed_hours=w.get("mission_elapsed_hours", 0),
            extraction_requested=w["extraction_requested"],
            distance_to_extraction=round(w["distance_to_extraction"], 2),
            sitrep=sitrep,
            task_id=w["task_id"],
            max_steps=w["max_steps"],
            active_incidents=incidents,
            trauma_active=w.get("trauma_active", False),
            done=False,
            reward=None,
        )

    def _get_visible_map(self, grid, pos, visited) -> list[list[str]]:
        r, c = pos
        result = []
        for row in range(5):
            row_data = []
            for col in range(5):
                if abs(row - r) + abs(col - c) <= 2 or (row, col) in visited:
                    row_data.append(grid[row][col])
                else:
                    row_data.append("unknown")
            result.append(row_data)
        return result
