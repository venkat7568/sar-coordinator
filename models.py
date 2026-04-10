"""
SAR Coordinator Environment — Data Models

The agent acts as an AI decision-support system for Search & Rescue operations.
Every observation is a field intelligence report. Every action is an operational order.
The physics are grounded in real SAR science — victim deterioration rates, resource
depletion curves, extraction window timing.

This is not a game. It is a training environment for AI systems that will assist
real SAR coordinators, autonomous field robots, and humanitarian logistics teams.
"""

from enum import Enum
from typing import Optional, Literal

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


# ── Action Types ──────────────────────────────────────────────────────────────

ActionType = Literal[
    "deploy",     # deploy a resource acquisition team (water/food/equipment/medical)
    "establish",  # establish field infrastructure (shelter/signal_fire/extraction_point)
    "relocate",   # move operations to a new sector
    "standby",    # hold position and recover team capacity
    "triage",     # administer medical response (trauma/hypothermia/dehydration)
    "extract",    # request extraction / contact command (flare/mirror/radio/beacon)
    "allocate",   # allocate inventory resources to personnel
    "assess",     # gather situational intelligence (terrain/weather/personnel/resources)
]


class SARAction(Action):
    """
    A single operational decision issued by the SAR coordinator agent.

    Only fields relevant to the chosen action_type need to be populated.
    All other parameter fields remain None.

    Examples:
        SARAction(action_type="deploy", resource_type="water")
        SARAction(action_type="establish", structure_type="field_shelter")
        SARAction(action_type="extract", signal_method="radio")
        SARAction(action_type="triage", condition="trauma")
    """

    action_type: ActionType = Field(
        ..., description="Operational order type"
    )

    # deploy — resource acquisition
    resource_type: Optional[Literal["water", "food", "equipment", "medical"]] = Field(
        default=None, description="Resource to acquire (deploy only)"
    )

    # establish — field infrastructure
    structure_type: Optional[Literal["field_shelter", "signal_fire", "extraction_point"]] = Field(
        default=None, description="Infrastructure to establish (establish only)"
    )

    # relocate — sector movement
    direction: Optional[Literal["N", "S", "E", "W"]] = Field(
        default=None, description="Direction of movement (relocate only)"
    )
    distance: Optional[float] = Field(
        default=None, ge=1.0, le=5.0, description="Distance in km, 1–5 (relocate only)"
    )

    # standby — hold position
    duration: Optional[int] = Field(
        default=None, ge=1, le=8, description="Hours to hold position, 1–8 (standby only)"
    )

    # triage — medical response
    condition: Optional[Literal["trauma", "hypothermia", "dehydration"]] = Field(
        default=None, description="Medical condition to address (triage only)"
    )

    # extract — request extraction
    signal_method: Optional[Literal["flare", "mirror", "radio", "beacon"]] = Field(
        default=None, description="Extraction request method (extract only)"
    )

    # allocate — resource distribution
    item: Optional[Literal["food", "water", "medicine"]] = Field(
        default=None, description="Resource to allocate (allocate only)"
    )
    quantity: Optional[float] = Field(
        default=None, ge=0.1, description="Quantity to allocate (allocate only)"
    )

    # assess — intelligence gathering
    target: Optional[Literal["terrain", "weather", "personnel", "resources"]] = Field(
        default=None, description="Intelligence target (assess only)"
    )

    # optional chain-of-thought — used in decision quality scoring
    reasoning: Optional[str] = Field(
        default=None, description="Coordinator's operational reasoning (optional)"
    )


# ── Observation Types ─────────────────────────────────────────────────────────

class BaseStatus(str, Enum):
    none        = "none"
    established = "established"
    fortified   = "fortified"


class WeatherCondition(str, Enum):
    clear = "clear"
    rain  = "rain"
    storm = "storm"
    snow  = "snow"


class SARObservation(Observation):
    """
    Field intelligence report returned after each operational decision.

    Personnel metrics run 0.0 (nominal) → 1.0 (critical failure).
    Mission viability runs 1.0 (optimal) → 0.0 (mission failure).
    core_temperature in Celsius — hypothermia onset below 35°C.
    sector_map is a 5×5 tactical grid centered on current position (partial intel).
    """

    # ── Personnel Status ─────────────────────────────────────────────────────
    nutrition_deficit: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Personnel nutrition deficit — 0=adequate, 1=critical")
    hydration_deficit: float = Field(default=0.0, ge=0.0, le=1.0,
        description="Personnel hydration deficit — 0=adequate, 1=critical")
    team_capacity:     float = Field(default=1.0, ge=0.0, le=1.0,
        description="Team operational capacity — 1=full, 0=non-operational")
    core_temperature:  float = Field(default=37.0,
        description="Core body temperature in Celsius")
    mission_viability: float = Field(default=1.0, ge=0.0, le=1.0,
        description="Overall mission viability — primary reward signal")

    # ── Field Conditions ─────────────────────────────────────────────────────
    base_camp_status:   BaseStatus      = Field(default=BaseStatus.none)
    weather_conditions: WeatherCondition = Field(default=WeatherCondition.clear)
    weather_severity:   float           = Field(default=0.0, ge=0.0, le=1.0)

    # 5×5 tactical sector grid — unrecognised sectors marked "unknown"
    sector_map: list[list[str]] = Field(
        default_factory=lambda: [["unknown"] * 5 for _ in range(5)]
    )

    # ── Resource Inventory ───────────────────────────────────────────────────
    # food (ration units), water (litres), equipment (units), medical (kits)
    resource_inventory: dict[str, float] = Field(
        default_factory=lambda: {"food": 0.0, "water": 0.0, "equipment": 0.0, "medical": 0.0}
    )

    # ── Mission Progress ─────────────────────────────────────────────────────
    mission_elapsed_hours: int   = Field(default=0,    description="Hours elapsed since mission start")
    extraction_requested:  bool  = Field(default=False, description="Whether extraction has been requested")
    distance_to_extraction: float = Field(default=10.0, description="km to nearest extraction point")

    # ── Intelligence ─────────────────────────────────────────────────────────
    sitrep:           str       = Field(default="", description="Situation report — outcome of last action")
    task_id:          int       = Field(default=1,  description="Task difficulty (1=easy, 2=medium, 3=hard)")
    max_steps:        int       = Field(default=5,  description="Maximum steps for this task")
    active_incidents: list[str] = Field(default_factory=list,
        description="Active field incidents this step (e.g. trauma, storm_onset)")
    trauma_active:    bool      = Field(default=False,
        description="Whether an untreated trauma injury is active (causes -0.02/hr viability drain)")
