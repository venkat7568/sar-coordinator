"""
FastAPI application for the SAR Coordinator Environment.

Endpoints:
    POST /reset   — Initialise a new mission episode
    POST /step    — Execute an operational decision
    GET  /state   — Current episode state
    GET  /schema  — Action / observation schemas
    GET  /render  — Plain-text operational dashboard
    GET  /        — Live HTML operations center dashboard
    GET  /docs    — Interactive API documentation
    WS   /ws      — WebSocket for persistent agent sessions
"""

import json

from fastapi import Request
from fastapi.responses import HTMLResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError("openenv-core is required. pip install openenv-core[core]") from e

try:
    from ..models import SARAction, SARObservation
    from .myenv_environment import SAREnvironment
except ImportError:
    from models import SARAction, SARObservation
    from server.myenv_environment import SAREnvironment


# ── App ───────────────────────────────────────────────────────────────────────

app = create_app(
    SAREnvironment,
    SARAction,
    SARObservation,
    env_name="sar-coordinator",
    max_concurrent_envs=1,
)

app.title       = "SAR Coordinator Environment"
app.description = (
    "**Search & Rescue Coordinator RL Environment** — OpenEnv Specification\n\n"
    "An AI decision-support system for Search & Rescue field operations. "
    "The agent acts as an operational coordinator: deploying resource teams, "
    "establishing field infrastructure, managing personnel capacity, and executing "
    "extraction under time pressure and dynamic incident conditions.\n\n"
    "**Physics:** Grounded in real SAR science — hydration deficit escalates at 0.06/hr, "
    "hypothermia onset below 35°C, shelter reduces temperature drift by up to 80%.\n\n"
    "**Tasks:**\n"
    "- Task 1 (Easy): Field resource triage — 5 operational decisions\n"
    "- Task 2 (Medium): 24-hour operational arc — jungle terrain, storm at hour 8\n"
    "- Task 3 (Hard): Extended field deployment — arctic, trauma day 2, storm day 4\n\n"
    "**Real-world analog:** SAR coordination, autonomous field robotics, humanitarian logistics."
)
app.version = "1.0.0"


# ── Observation Cache Middleware ──────────────────────────────────────────────

_last_obs: dict = {}


class ObsCacheMiddleware(BaseHTTPMiddleware):
    """Caches the latest observation from /reset and /step for the dashboard."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        if request.url.path in ("/reset", "/step") and response.status_code == 200:
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            try:
                data = json.loads(body)
                obs  = data.get("observation", data)
                if isinstance(obs, dict) and "mission_viability" in obs:
                    global _last_obs
                    _last_obs = obs
            except Exception:
                pass
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
        return response


app.add_middleware(ObsCacheMiddleware)


# ── ASCII Render ──────────────────────────────────────────────────────────────

def _bar(value: float, width: int = 12, invert: bool = False) -> str:
    pct = max(0.0, min(1.0, (1.0 - value) if invert else value))
    filled = int(pct * width)
    return "█" * filled + "░" * (width - filled)


def _render_ops(obs: dict) -> str:
    if not obs:
        return "No active mission. POST /reset to initialise.\n"

    mv   = obs.get("mission_viability", 1.0)
    hd   = obs.get("hydration_deficit", 0.0)
    nd   = obs.get("nutrition_deficit", 0.0)
    tc   = obs.get("team_capacity", 1.0)
    temp = obs.get("core_temperature", 37.0)
    camp = obs.get("base_camp_status", "none")
    wx   = obs.get("weather_conditions", "clear")
    inv  = obs.get("resource_inventory", {})
    t    = obs.get("mission_elapsed_hours", 0)
    ms   = obs.get("max_steps", 5)
    tid  = obs.get("task_id", 1)
    dist = obs.get("distance_to_extraction", 10.0)
    ext  = obs.get("extraction_requested", False)
    inc  = obs.get("active_incidents", [])
    srep = obs.get("sitrep", "")
    smap = obs.get("sector_map", [])

    cell_sym = {
        "forest": " F ", "jungle": " J ", "river":  " ~ ",
        "snow":   " * ", "arctic": " A ", "mountain":" M ",
        "plain":  " . ", "unknown":" ? ",
    }
    grid_rows = []
    for ri, row in enumerate(smap):
        line = "|"
        for ci, cell in enumerate(row):
            sym = "[@]" if (ri == 2 and ci == 2) else cell_sym.get(cell, " ? ")
            line += sym
        line += "|"
        grid_rows.append(line)

    task_names = {1: "Resource Triage", 2: "24-Hour Arc", 3: "Extended Deployment"}

    lines = [
        "╔══════════════════════════════════════════════════════════╗",
        f"║  SAR COORDINATOR  │  {task_names.get(tid,'Task '+str(tid)):<18}  │  H{t:>3}/{ms:<3}  ║",
        "╠════════════════════╦═════════════════════════════════════╣",
        "║  SECTOR MAP        ║  OPERATIONAL STATUS                 ║",
        f"║  {grid_rows[0] if grid_rows else '':18} ║  Mission Viability  {_bar(mv)}  {mv:>4.0%}  ║",
        f"║  {grid_rows[1] if len(grid_rows)>1 else '':18} ║  Hydration Deficit  {_bar(hd,invert=True)}  {hd:>4.0%}  ║",
        f"║  {grid_rows[2] if len(grid_rows)>2 else '':18} ║  Nutrition Deficit  {_bar(nd,invert=True)}  {nd:>4.0%}  ║",
        f"║  {grid_rows[3] if len(grid_rows)>3 else '':18} ║  Team Capacity      {_bar(tc)}  {tc:>4.0%}  ║",
        f"║  {grid_rows[4] if len(grid_rows)>4 else '':18} ║  Core Temp          {temp:>5.1f}°C              ║",
        "╠════════════════════╩═════════════════════════════════════╣",
        f"║  Base Camp: {camp:<12}  Weather: {wx:<8}  Dist: {dist:.1f}km  ║",
        f"║  Food:{inv.get('food',0):>4.1f}u  Water:{inv.get('water',0):>4.1f}L  Equip:{inv.get('equipment',0):>4.1f}u  Med:{inv.get('medical',0):>3.1f}  ║",
        f"║  Extraction: {'REQUESTED ✓' if ext else 'not requested':<43} ║",
    ]
    if inc:
        lines.append(f"║  ⚠ INCIDENTS: {', '.join(inc):<43} ║")
    if srep:
        s = srep[:52] + "..." if len(srep) > 52 else srep
        lines.append(f"║  SITREP: {s:<47} ║")
    lines.append("╚══════════════════════════════════════════════════════════╝")
    return "\n".join(lines)


@app.get("/render", response_class=PlainTextResponse, tags=["Operations"])
def render_ops():
    """Plain-text operational status dashboard."""
    return _render_ops(_last_obs)


@app.get("/last_obs", tags=["Operations"])
def last_observation():
    """Latest cached observation — used by the live dashboard."""
    return _last_obs if _last_obs else {}


# ── HTML Operations Center Dashboard ─────────────────────────────────────────

OPS_DASHBOARD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SAR Coordinator — Operations Center</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0a0e17;
    --surface:   #0f1623;
    --surface2:  #151d2e;
    --border:    #1e2d45;
    --text:      #cdd9e5;
    --text-dim:  #4d6080;
    --accent:    #2f7df7;
    --green:     #2ea043;
    --yellow:    #c79f0a;
    --red:       #cf3c3c;
    --cyan:      #1f8fcf;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    font-size: 14px;
    min-height: 100vh;
    padding: 0;
  }

  /* ── Top bar ── */
  .topbar {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 12px 24px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: sticky;
    top: 0;
    z-index: 100;
  }
  .topbar-title {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--accent);
  }
  .topbar-sub {
    font-size: 0.75rem;
    color: var(--text-dim);
    letter-spacing: 0.04em;
  }
  .status-pill {
    margin-left: auto;
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 0.75rem;
    color: var(--text-dim);
  }
  .pulse {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--green);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.3} }

  .nav-links { display:flex; gap:16px; font-size:0.78rem; }
  .nav-links a { color:var(--text-dim); text-decoration:none; }
  .nav-links a:hover { color:var(--accent); }

  /* ── Layout ── */
  .main {
    padding: 20px 24px;
    display: grid;
    grid-template-columns: 280px 1fr 290px;
    grid-template-rows: auto auto 1fr;
    gap: 16px;
    max-width: 1600px;
    margin: 0 auto;
  }

  /* ── Cards ── */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 16px;
  }
  .card-title {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
  }

  /* ── Mission header strip ── */
  .mission-header {
    grid-column: 1 / -1;
    display: flex;
    gap: 16px;
    align-items: stretch;
  }
  .mission-stat {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 18px;
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 120px;
  }
  .mission-stat-label { font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.08em; }
  .mission-stat-value { font-size: 1.2rem; font-weight: 700; color: var(--text); }
  .mission-stat-value.good   { color: var(--green); }
  .mission-stat-value.warn   { color: var(--yellow); }
  .mission-stat-value.danger { color: var(--red); }

  /* ── Sector map ── */
  .sector-map {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 3px;
    font-size: 1.0rem;
  }
  .sector-cell {
    aspect-ratio: 1;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    border: 1px solid transparent;
    background: var(--surface2);
    transition: background 0.3s;
  }
  .cell-unknown  { color: var(--text-dim); background: #0d1220; border-color: #151d2e; }
  .cell-forest   { color: #3fb950; }
  .cell-jungle   { color: #26a641; }
  .cell-river    { color: #58a6ff; }
  .cell-snow     { color: #a8c4d4; }
  .cell-arctic   { color: #79c0ff; }
  .cell-mountain { color: #f0883e; }
  .cell-plain    { color: #56d364; background: #161e28; }
  .cell-agent    { background: #2f1d1d; border-color: var(--red) !important; color: var(--red); font-weight: 700; }

  /* ── Metric bars ── */
  .metric-row { margin-bottom: 10px; }
  .metric-header { display: flex; justify-content: space-between; margin-bottom: 4px; }
  .metric-label { font-size: 0.78rem; color: var(--text-dim); }
  .metric-value { font-size: 0.78rem; font-weight: 600; }
  .bar-track {
    height: 6px;
    background: #1a2235;
    border-radius: 3px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.5s ease, background-color 0.5s ease;
  }

  /* ── Resource table ── */
  .resource-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .resource-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 8px 12px;
    display: flex;
    flex-direction: column;
    gap: 2px;
  }
  .resource-label { font-size: 0.68rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.06em; }
  .resource-value { font-size: 1.05rem; font-weight: 700; color: var(--text); }

  /* ── Incident badges ── */
  .incidents { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 8px; }
  .incident-badge {
    background: #3d1515;
    border: 1px solid var(--red);
    color: #ff7b7b;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  /* ── SITREP ── */
  .sitrep-box {
    background: var(--surface2);
    border-left: 3px solid var(--accent);
    border-radius: 0 4px 4px 0;
    padding: 10px 14px;
    font-size: 0.82rem;
    color: var(--text);
    line-height: 1.5;
    font-style: italic;
    min-height: 48px;
  }

  /* ── Status tags ── */
  .tag {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .tag-green  { background:#122119; border:1px solid var(--green);  color:var(--green); }
  .tag-yellow { background:#1f1a08; border:1px solid var(--yellow); color:var(--yellow); }
  .tag-red    { background:#1f0d0d; border:1px solid var(--red);    color:var(--red); }
  .tag-blue   { background:#0d1a2e; border:1px solid var(--accent); color:var(--accent); }
  .tag-dim    { background:var(--surface2); border:1px solid var(--border); color:var(--text-dim); }

  .no-mission {
    grid-column: 1 / 3;
    text-align: center;
    padding: 60px 20px;
  }
  .no-mission h2 { color: var(--text-dim); font-size: 1rem; margin-bottom: 12px; }

  #left-panel  { grid-column: 1; grid-row: 2 / 4; }
  #right-panel { grid-column: 2; grid-row: 2 / 4; }
  #cmd-panel   { grid-column: 3; grid-row: 2 / 4; }

  /* ── Right panel grid ── */
  .right-panel {
    display: grid;
    grid-template-rows: auto auto 1fr;
    gap: 16px;
  }

  /* ── Command Center ── */
  .cmd-panel-inner {
    height: 100%;
    display: flex;
    flex-direction: column;
    gap: 0;
  }
  .cmd-section {
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid var(--border);
  }
  .cmd-section:last-child { border-bottom: none; margin-bottom: 0; flex: 1; display: flex; flex-direction: column; }
  .cmd-section-title {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-bottom: 8px;
  }
  .cmd-task-btns { display: flex; gap: 5px; }
  .cmd-task-btn {
    flex: 1;
    padding: 6px 4px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    text-align: center;
  }
  .cmd-task-btn:hover { border-color: var(--accent); background: #0d1a2e; color: var(--accent); }
  .cmd-select, .cmd-input {
    width: 100%;
    padding: 6px 8px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 0.78rem;
    margin-bottom: 5px;
    outline: none;
    font-family: inherit;
  }
  .cmd-select:focus, .cmd-input:focus { border-color: var(--accent); }
  .cmd-btn-issue {
    width: 100%;
    margin-top: 4px;
    padding: 8px;
    background: var(--accent);
    border: none;
    border-radius: 4px;
    color: #fff;
    font-size: 0.8rem;
    font-weight: 700;
    cursor: pointer;
    letter-spacing: 0.04em;
    transition: opacity 0.2s;
  }
  .cmd-btn-issue:hover { opacity: 0.85; }
  .cmd-btn-issue:disabled { opacity: 0.4; cursor: not-allowed; }
  .cmd-log {
    flex: 1;
    overflow-y: auto;
    display: flex;
    flex-direction: column;
    gap: 5px;
    max-height: 200px;
  }
  .cmd-log-entry {
    padding: 6px 10px;
    border-radius: 4px;
    font-size: 0.71rem;
    line-height: 1.4;
  }
  .log-reset { background: #0d1a2e; border-left: 3px solid var(--accent); }
  .log-step  { background: #0d1a15; border-left: 3px solid var(--green); }
  .log-error { background: #1f0d0d; border-left: 3px solid var(--red); }
  .log-time  { color: var(--text-dim); font-size: 0.65rem; }
  .log-act   { color: var(--text); font-weight: 600; margin: 1px 0; }
  .log-rew   { color: var(--green); }
</style>
</head>
<body>

<!-- Top bar -->
<div class="topbar">
  <div>
    <div class="topbar-title">SAR Coordinator — Operations Center</div>
    <div class="topbar-sub">OpenEnv RL Environment · Meta × HuggingFace Hackathon</div>
  </div>
  <div class="nav-links">
    <a href="/docs">API Docs</a>
    <a href="/schema">Schema</a>
    <a href="/render">ASCII</a>
  </div>
  <div class="status-pill">
    <div class="pulse"></div>
    <span id="live-label">Awaiting mission</span>
    &nbsp;·&nbsp; Refresh <strong>2s</strong>
  </div>
</div>

<!-- Main layout -->
<div class="main" id="main-layout">

  <!-- No mission placeholder -->
  <div class="no-mission" id="no-mission">
    <h2>No Active Mission</h2>
    <p style="color:var(--text-dim);font-size:0.85rem;margin-bottom:16px">
      Use the Command Center to initialise an episode
    </p>
    <div style="display:flex;gap:10px;justify-content:center">
      <div onclick="startMission(1)" class="cmd-task-btn" style="padding:10px 20px;font-size:0.8rem">ALPHA — Field Triage</div>
      <div onclick="startMission(2)" class="cmd-task-btn" style="padding:10px 20px;font-size:0.8rem">BRAVO — 24-Hour Arc</div>
      <div onclick="startMission(3)" class="cmd-task-btn" style="padding:10px 20px;font-size:0.8rem">CHARLIE — Extended</div>
    </div>
  </div>

  <!-- Mission header -->
  <div class="mission-header" id="mission-header" style="display:none">
    <div class="mission-stat">
      <div class="mission-stat-label">Task</div>
      <div class="mission-stat-value" id="hdr-task">—</div>
    </div>
    <div class="mission-stat">
      <div class="mission-stat-label">Mission Hour</div>
      <div class="mission-stat-value" id="hdr-hour">—</div>
    </div>
    <div class="mission-stat">
      <div class="mission-stat-label">Viability</div>
      <div class="mission-stat-value" id="hdr-viability">—</div>
    </div>
    <div class="mission-stat">
      <div class="mission-stat-label">Base Camp</div>
      <div class="mission-stat-value" id="hdr-camp">—</div>
    </div>
    <div class="mission-stat">
      <div class="mission-stat-label">Weather</div>
      <div class="mission-stat-value" id="hdr-weather">—</div>
    </div>
    <div class="mission-stat">
      <div class="mission-stat-label">Extraction</div>
      <div class="mission-stat-value" id="hdr-extract">—</div>
    </div>
  </div>

  <!-- Left panel — Sector map -->
  <div id="left-panel" style="display:none">
    <div class="card" style="height:100%">
      <div class="card-title">Tactical Sector Map</div>
      <div class="sector-map" id="sector-map"></div>
      <div style="margin-top:14px">
        <div class="card-title" style="margin-bottom:10px">Field Resources</div>
        <div class="resource-grid" id="resource-grid"></div>
      </div>
    </div>
  </div>

  <!-- Right panel -->
  <div class="right-panel" id="right-panel" style="display:none">

    <!-- Personnel status -->
    <div class="card">
      <div class="card-title">Personnel Status</div>
      <div id="metric-viability" class="metric-row"></div>
      <div id="metric-hydration" class="metric-row"></div>
      <div id="metric-nutrition" class="metric-row"></div>
      <div id="metric-capacity"  class="metric-row"></div>
      <div style="margin-top:10px;font-size:0.8rem;color:var(--text-dim)">
        Core Temp: <strong id="core-temp" style="color:var(--text)">—</strong>
      </div>
    </div>

    <!-- Situation report -->
    <div class="card">
      <div class="card-title">Situation Report (SITREP)</div>
      <div id="incidents" class="incidents"></div>
      <div class="sitrep-box" id="sitrep">Awaiting first operational order.</div>
    </div>

  </div>

  <!-- Command Center — always visible -->
  <div id="cmd-panel">
    <div class="card cmd-panel-inner">
      <div class="card-title">Command Center</div>

      <!-- Mission Init -->
      <div class="cmd-section">
        <div class="cmd-section-title">Initialise Mission</div>
        <div class="cmd-task-btns">
          <div class="cmd-task-btn" onclick="startMission(1)">ALPHA</div>
          <div class="cmd-task-btn" onclick="startMission(2)">BRAVO</div>
          <div class="cmd-task-btn" onclick="startMission(3)">CHARLIE</div>
        </div>
      </div>

      <!-- Action Builder -->
      <div class="cmd-section">
        <div class="cmd-section-title">Dispatch Operational Order</div>
        <select id="cmd-atype" class="cmd-select" onchange="updateParams()">
          <option value="deploy">deploy</option>
          <option value="establish">establish</option>
          <option value="relocate">relocate</option>
          <option value="standby">standby</option>
          <option value="triage">triage</option>
          <option value="extract">extract</option>
          <option value="allocate">allocate</option>
          <option value="assess">assess</option>
        </select>

        <div id="p-deploy">
          <select id="p-resource_type" class="cmd-select">
            <option value="water">water</option>
            <option value="food">food</option>
            <option value="equipment">equipment</option>
            <option value="medical">medical</option>
          </select>
        </div>
        <div id="p-establish" style="display:none">
          <select id="p-structure_type" class="cmd-select">
            <option value="field_shelter">field_shelter</option>
            <option value="signal_fire">signal_fire</option>
            <option value="extraction_point">extraction_point</option>
          </select>
        </div>
        <div id="p-relocate" style="display:none">
          <select id="p-direction" class="cmd-select">
            <option value="N">North</option>
            <option value="S">South</option>
            <option value="E">East</option>
            <option value="W">West</option>
          </select>
          <input id="p-distance" type="number" class="cmd-input" min="1" max="5" step="0.5" value="2" placeholder="Distance km">
        </div>
        <div id="p-standby" style="display:none">
          <input id="p-duration" type="number" class="cmd-input" min="1" max="8" value="2" placeholder="Duration hours">
        </div>
        <div id="p-triage" style="display:none">
          <select id="p-condition" class="cmd-select">
            <option value="trauma">trauma</option>
            <option value="hypothermia">hypothermia</option>
            <option value="dehydration">dehydration</option>
          </select>
        </div>
        <div id="p-extract" style="display:none">
          <select id="p-signal_method" class="cmd-select">
            <option value="flare">flare</option>
            <option value="mirror">mirror</option>
            <option value="radio">radio</option>
            <option value="beacon">beacon</option>
          </select>
        </div>
        <div id="p-allocate" style="display:none">
          <select id="p-item" class="cmd-select">
            <option value="food">food</option>
            <option value="water">water</option>
            <option value="medicine">medicine</option>
          </select>
          <input id="p-quantity" type="number" class="cmd-input" min="0.1" max="5" step="0.1" value="1.0" placeholder="Quantity">
        </div>
        <div id="p-assess" style="display:none">
          <select id="p-target" class="cmd-select">
            <option value="terrain">terrain</option>
            <option value="weather">weather</option>
            <option value="personnel">personnel</option>
            <option value="resources">resources</option>
          </select>
        </div>

        <button id="issue-btn" class="cmd-btn-issue" onclick="issueOrder()">Dispatch Order</button>
      </div>

      <!-- Command Log -->
      <div class="cmd-section">
        <div class="cmd-section-title">Command Log</div>
        <div id="cmd-log" class="cmd-log">
          <div class="cmd-log-entry log-reset">
            <div class="log-time">—</div>
            <div class="log-act">Awaiting first order</div>
          </div>
        </div>
      </div>

    </div>
  </div>

</div>

<script>
const TASK_NAMES = {
  1: "SCENARIO ALPHA — Field Resource Triage",
  2: "SCENARIO BRAVO — 24-Hour Operational Arc",
  3: "SCENARIO CHARLIE — Extended Field Deployment",
};

const WEATHER_ICON = { clear:"", rain:"", storm:"", snow:"" };
const CAMP_TAG = {
  none:        ['dim',    'NOT ESTABLISHED'],
  established: ['yellow', 'ESTABLISHED'],
  fortified:   ['green',  'FORTIFIED'],
};
const TERRAIN_CLASS = {
  forest:"cell-forest", jungle:"cell-jungle", river:"cell-river",
  snow:"cell-snow", arctic:"cell-arctic", mountain:"cell-mountain",
  plain:"cell-plain", unknown:"cell-unknown",
};
const TERRAIN_SYM = {
  forest:"F", jungle:"J", river:"~",
  snow:"*", arctic:"A", mountain:"^", plain:".", unknown:"?",
};

function mkMetric(label, value, invert, unit="") {
  const disp  = invert ? 1 - value : value;
  const pct   = (disp * 100).toFixed(0);
  const color = disp > 0.6 ? "#2ea043" : disp > 0.3 ? "#c79f0a" : "#cf3c3c";
  const valStr = invert ? `${(value*100).toFixed(0)}%` : `${pct}%`;
  return `
    <div class="metric-header">
      <span class="metric-label">${label}</span>
      <span class="metric-value" style="color:${color}">${valStr}${unit}</span>
    </div>
    <div class="bar-track">
      <div class="bar-fill" style="width:${pct}%;background:${color}"></div>
    </div>`;
}

function mkTag(type, text) {
  return `<span class="tag tag-${type}">${text}</span>`;
}

// ── Command Center ──────────────────────────────────────────────────────────

function updateParams() {
  const type = document.getElementById('cmd-atype').value;
  ['deploy','establish','relocate','standby','triage','extract','allocate','assess']
    .forEach(t => {
      const el = document.getElementById(`p-${t}`);
      if (el) el.style.display = t === type ? 'block' : 'none';
    });
}

async function startMission(taskId) {
  logCmd('reset', {task_id: taskId}, null, null);
  try {
    const r = await fetch('/reset', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({task_id: taskId}),
    });
    const data = await r.json();
    logCmd('reset', {task_id: taskId}, data, r.ok);
    if (r.ok) refresh();
  } catch(e) {
    logCmd('reset', {task_id: taskId}, {error: e.message}, false);
  }
}

async function issueOrder() {
  const type = document.getElementById('cmd-atype').value;
  const action = {action_type: type};

  if (type === 'deploy')     action.resource_type  = document.getElementById('p-resource_type').value;
  if (type === 'establish')  action.structure_type = document.getElementById('p-structure_type').value;
  if (type === 'relocate') {
    action.direction = document.getElementById('p-direction').value;
    action.distance  = parseFloat(document.getElementById('p-distance').value);
  }
  if (type === 'standby')  action.duration      = parseInt(document.getElementById('p-duration').value);
  if (type === 'triage')   action.condition     = document.getElementById('p-condition').value;
  if (type === 'extract')  action.signal_method = document.getElementById('p-signal_method').value;
  if (type === 'allocate') {
    action.item     = document.getElementById('p-item').value;
    action.quantity = parseFloat(document.getElementById('p-quantity').value);
  }
  if (type === 'assess')  action.target = document.getElementById('p-target').value;

  const btn = document.getElementById('issue-btn');
  btn.disabled = true;
  btn.textContent = 'Sending…';

  try {
    const r = await fetch('/step', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({action}),
    });
    const data = await r.json();
    logCmd('step', action, data, r.ok);
    if (r.ok) refresh();
  } catch(e) {
    logCmd('step', action, {error: e.message}, false);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Dispatch Order';
  }
}

function logCmd(type, action, data, ok) {
  const log = document.getElementById('cmd-log');
  if (!log) return;
  const now = new Date().toLocaleTimeString();

  // Remove placeholder
  const placeholder = log.querySelector('.log-reset');
  if (placeholder && placeholder.querySelector('.log-act')?.textContent === 'Awaiting first order') {
    placeholder.remove();
  }

  if (data === null) return; // pending — don't log yet

  const entry = document.createElement('div');

  if (type === 'reset') {
    entry.className = ok ? 'cmd-log-entry log-reset' : 'cmd-log-entry log-error';
    entry.innerHTML = `<div class="log-time">${now}</div>
      <div class="log-act">RESET — Task ${action.task_id}</div>
      <div class="${ok ? 'log-rew' : ''}" style="${ok ? '' : 'color:var(--red)'}">${ok ? 'Mission initialised ✓' : 'Failed'}</div>`;
  } else {
    const reward = data?.reward != null ? parseFloat(data.reward).toFixed(2) : null;
    const done   = data?.done ?? false;
    const params = Object.entries(action)
      .filter(([k]) => k !== 'action_type')
      .map(([k, v]) => `${k}=${v}`).join(' ');
    entry.className = ok ? 'cmd-log-entry log-step' : 'cmd-log-entry log-error';
    entry.innerHTML = `<div class="log-time">${now}</div>
      <div class="log-act">${action.action_type?.toUpperCase()}${params ? ' · ' + params : ''}</div>
      ${ok
        ? `<div><span class="log-rew">reward +${reward}</span>${done ? ' · <span style="color:var(--yellow)">DONE</span>' : ''}</div>`
        : `<div style="color:var(--red)">${data?.detail || data?.error || 'Error'}</div>`
      }`;
  }

  log.prepend(entry);
  // Keep log trimmed to last 20 entries
  while (log.children.length > 20) log.removeChild(log.lastChild);
}

async function refresh() {
  try {
    const r = await fetch("/last_obs");
    if (!r.ok) return;
    const obs = await r.json();
    if (!obs || obs.mission_viability === undefined) return;

    document.getElementById("no-mission").style.display      = "none";
    document.getElementById("mission-header").style.display  = "flex";
    document.getElementById("left-panel").style.display      = "block";
    document.getElementById("right-panel").style.display     = "grid";
    document.getElementById("live-label").textContent        = "Live";

    const mv   = obs.mission_viability;
    const tid  = obs.task_id;
    const camp = obs.base_camp_status;
    const wx   = obs.weather_conditions;

    // Header
    document.getElementById("hdr-task").textContent = TASK_NAMES[tid] || `Task ${tid}`;
    document.getElementById("hdr-hour").textContent = `H${obs.mission_elapsed_hours} / ${obs.max_steps}`;

    const vc = mv > 0.6 ? "good" : mv > 0.3 ? "warn" : "danger";
    document.getElementById("hdr-viability").className = `mission-stat-value ${vc}`;
    document.getElementById("hdr-viability").textContent = `${(mv*100).toFixed(0)}%`;

    const [campType, campText] = CAMP_TAG[camp] || ["dim", camp];
    document.getElementById("hdr-camp").innerHTML = mkTag(campType, campText);

    const wxIcon = WEATHER_ICON[wx] || "";
    const wxSev  = obs.weather_severity > 0.6 ? "red" : obs.weather_severity > 0.2 ? "yellow" : "green";
    document.getElementById("hdr-weather").innerHTML = mkTag(wxSev, `${wxIcon} ${wx}`);

    const extText = obs.extraction_requested ? "Requested ✓" : "Not Requested";
    const extType = obs.extraction_requested ? "green" : "dim";
    document.getElementById("hdr-extract").innerHTML = mkTag(extType, extText);

    // Sector map
    const mapEl = document.getElementById("sector-map");
    mapEl.innerHTML = "";
    const grid = obs.sector_map || [];
    grid.forEach((row, ri) => {
      row.forEach((cell, ci) => {
        const div = document.createElement("div");
        if (ri === 2 && ci === 2) {
          div.className = "sector-cell cell-agent";
          div.textContent = "◉";
          div.title = "Current position";
        } else {
          div.className = `sector-cell ${TERRAIN_CLASS[cell] || "cell-unknown"}`;
          div.textContent = TERRAIN_SYM[cell] || "?";
          div.title = cell;
        }
        mapEl.appendChild(div);
      });
    });

    // Resources
    const inv = obs.resource_inventory || {};
    document.getElementById("resource-grid").innerHTML = [
      ["🍖 Food",      inv.food      || 0, "units"],
      ["💧 Water",     inv.water     || 0, "L"],
      ["🔧 Equipment", inv.equipment || 0, "units"],
      ["💊 Medical",   inv.medical   || 0, "kits"],
    ].map(([l, v, u]) => `
      <div class="resource-item">
        <div class="resource-label">${l}</div>
        <div class="resource-value">${parseFloat(v).toFixed(1)} <span style="font-size:0.7rem;color:var(--text-dim)">${u}</span></div>
      </div>`).join("");

    // Personnel metrics
    document.getElementById("metric-viability").innerHTML =
      mkMetric("Mission Viability",   obs.mission_viability,  false);
    document.getElementById("metric-hydration").innerHTML =
      mkMetric("Hydration Deficit",   obs.hydration_deficit,  true);
    document.getElementById("metric-nutrition").innerHTML =
      mkMetric("Nutrition Deficit",   obs.nutrition_deficit,  true);
    document.getElementById("metric-capacity").innerHTML  =
      mkMetric("Team Capacity",       obs.team_capacity,      false);

    const t  = obs.core_temperature;
    const tc = t < 35 ? "#58a6ff" : t > 39 ? "#cf3c3c" : "#2ea043";
    document.getElementById("core-temp").textContent = `${t.toFixed(1)}°C`;
    document.getElementById("core-temp").style.color = tc;

    // Incidents
    const inc = obs.active_incidents || [];
    document.getElementById("incidents").innerHTML = inc.length
      ? inc.map(i => `<div class="incident-badge">⚠ ${i.replace(/_/g," ")}</div>`).join("")
      : "";

    // SITREP
    document.getElementById("sitrep").textContent =
      obs.sitrep || "No situation report.";

  } catch(e) {
    // Server not ready
  }
}

refresh();
setInterval(refresh, 2000);
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse, tags=["Operations"])
def operations_center():
    """
    Live HTML Operations Center dashboard.

    Real-time monitoring of the active SAR mission — auto-refreshes every 2 seconds.

    Displays:
    - Mission header: task, hour, viability, base camp, weather, extraction status
    - Tactical sector map (5×5 grid with partial intelligence)
    - Personnel status bars (viability, hydration deficit, nutrition deficit, team capacity)
    - Resource inventory (food, water, equipment, medical)
    - Active incident alerts
    - Latest situation report (SITREP)

    Start a mission first: POST /reset with {"task_id": 1}
    """
    return OPS_DASHBOARD


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
