"""
OpenEnv FastAPI Backend
Endpoints: /simulate, /train, /metrics, /stream, /state
"""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─────────────────────────────────────────────
# App state (in-memory for demo; use Redis in prod)
# ─────────────────────────────────────────────
_envs: Dict[str, Any] = {}
_train_jobs: Dict[str, Dict] = {}
_sim_states: Dict[str, List[Dict]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("OpenEnv API starting up")
    yield
    # Shutdown
    _envs.clear()
    _train_jobs.clear()
    print("OpenEnv API shut down")


app = FastAPI(
    title="OpenEnv API",
    description="Multi-Agent Emergency Response Simulator",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────
class SimConfig(BaseModel):
    num_ems: int = Field(2, ge=1, le=4)
    num_fire: int = Field(2, ge=1, le=4)
    num_police: int = Field(2, ge=1, le=4)
    num_dispatcher: int = Field(1, ge=0, le=2)
    grid_size: int = Field(32, ge=16, le=64)
    seed: int = Field(42)
    weather_dynamic: bool = True
    cascade_enabled: bool = True
    curriculum_level: int = Field(2, ge=1, le=4)
    max_steps: int = Field(500, ge=50, le=2000)
    city: str = Field("procedural", description="'procedural' or 'bangalore'")


class StepRequest(BaseModel):
    session_id: str
    actions: Optional[Dict[str, int]] = None  # agent_id -> action_idx; None = auto


class TrainRequest(BaseModel):
    total_episodes: int = Field(500, ge=10, le=10000)
    lr: float = Field(3e-4, gt=0)
    curriculum_thresholds: List[float] = [0.45, 0.60, 0.75]
    use_wandb: bool = False
    run_name: str = "openenv-run"


class SessionResponse(BaseModel):
    session_id: str
    status: str
    config: Dict
    initial_state: Dict


class StepResponse(BaseModel):
    session_id: str
    step: int
    rewards: Dict[str, float]
    info: Dict
    terminated: bool


class TrainJobResponse(BaseModel):
    job_id: str
    status: str
    message: str


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────
def _get_env(session_id: str):
    if session_id not in _envs:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    return _envs[session_id]["env"]


def _env_state_to_dict(env) -> Dict:
    """Serialize environment state for frontend."""
    active_events = [e for e in env.events.values() if e.active]
    return {
        "step": env.step_count,
        "sim_hour": env.sim_hour,
        "sim_day": env.sim_day,
        "grid_size": env.grid_size,
        "weather": {
            "condition": env.weather.condition.name,
            "wind_speed": round(env.weather.wind_speed, 2),
            "temperature": round(env.weather.temperature, 1),
            "visibility": round(env.weather.visibility, 2),
        },
        "agents": [
            {
                "id": a.id,
                "type": a.type.name,
                "x": a.x,
                "y": a.y,
                "busy": a.busy,
                "resources": a.resources,
                "lives_saved": a.lives_saved,
                "events_handled": a.events_handled,
                "task_event_id": a.task_event_id,
            }
            for a in env.agents
        ],
        "events": [
            {
                "id": e.id,
                "type": e.type.name,
                "x": e.x,
                "y": e.y,
                "severity": round(e.severity, 3),
                "victims": e.victims,
                "ttl": e.ttl,
                "max_ttl": e.max_ttl,
                "assigned": e.assigned_agents,
                "cascade_children": e.cascade_children,
            }
            for e in active_events
        ],
        "metrics": {
            "total_lives_saved": env.total_lives_saved,
            "total_lives_lost": env.total_lives_lost,
            "total_events_spawned": env.total_events_spawned,
            "total_events_resolved": env.total_events_resolved,
            "resolution_rate": round(
                env.total_events_resolved / max(1, env.total_events_spawned), 3
            ),
            "cascade_count": env.cascade_count,
            "avg_response_time": round(
                env.total_response_time / max(1, env.total_events_resolved), 2
            ),
        },
        "fire_grid": env.fire_intensity.tolist(),
        "infra_grid": env.infra_health.tolist(),
        "city_grid": env.grid.tolist(),
        "population_grid": (env.population / env.population.max()).tolist(),
    }


def _auto_actions(env) -> Dict[int, int]:
    """Greedy auto-pilot: each agent moves toward nearest compatible event."""
    from backend.core.environment import AGENT_EVENT_MAP
    actions = {}
    n = env.grid_size
    for agent in env.agents:
        # Find nearest active compatible event
        best_evt = None
        best_dist = float("inf")
        for evt in env.events.values():
            if not evt.active:
                continue
            if evt.type not in AGENT_EVENT_MAP.get(agent.type, []):
                continue
            dist = abs(evt.x - agent.x) + abs(evt.y - agent.y)
            if dist < best_dist:
                best_dist = dist
                best_evt = evt

        if best_evt:
            action = best_evt.x * n + best_evt.y
        else:
            action = n * n  # WAIT
        actions[agent.id] = action
    return actions


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "OpenEnv API",
        "health": "/health",
        "docs": "/docs",
        "endpoints": [
            "/health",
            "/cities",
            "/simulate/create",
            "/simulate/step",
            "/simulate/step/bulk",
            "/simulate/state/{session_id}",
            "/simulate/stream/{session_id}",
            "/simulate/{session_id}",
            "/simulate/metrics/{session_id}",
            "/train/start",
            "/train/status/{job_id}",
            "/train/list",
            "/metrics/global",
            "/datasets/info",
        ],
    }


@app.get("/health")
def health():
    return {"status": "ok", "timestamp": time.time(), "active_sessions": len(_envs)}


@app.get("/cities")
def list_cities():
    """List all available Indian cities."""
    from backend.core.indian_cities import list_available_cities, get_city_metadata
    cities = list_available_cities()
    return {
        "available_cities": cities,
        "metadata": {city: get_city_metadata(city) for city in cities}
    }


@app.post("/simulate/create", response_model=SessionResponse)
def create_simulation(config: SimConfig):
    from backend.core.environment import OpenEnv
    session_id = str(uuid.uuid4())[:8]
    env = OpenEnv(
        num_ems=config.num_ems,
        num_fire=config.num_fire,
        num_police=config.num_police,
        num_dispatcher=config.num_dispatcher,
        grid_size=config.grid_size,
        seed=config.seed,
        weather_dynamic=config.weather_dynamic,
        cascade_enabled=config.cascade_enabled,
        curriculum_level=config.curriculum_level,
        city=config.city,  # ← NEW: Pass city parameter
    )
    obs, info = env.reset()
    _envs[session_id] = {"env": env, "config": config.model_dump(), "created_at": time.time()}
    _sim_states[session_id] = []

    state = _env_state_to_dict(env)
    return SessionResponse(
        session_id=session_id,
        status="created",
        config=config.model_dump(),
        initial_state=state,
    )


@app.post("/simulate/step", response_model=StepResponse)
def step_simulation(req: StepRequest):
    env = _get_env(req.session_id)

    if req.actions is not None:
        actions = {int(k): v for k, v in req.actions.items()}
    else:
        actions = _auto_actions(env)

    obs, rewards, terminated, truncated, info = env.step(actions)

    state = _env_state_to_dict(env)
    _sim_states[req.session_id].append(state)
    # Keep last 200 states
    if len(_sim_states[req.session_id]) > 200:
        _sim_states[req.session_id].pop(0)

    done = all(truncated.values()) or all(terminated.values())
    return StepResponse(
        session_id=req.session_id,
        step=env.step_count,
        rewards={str(k): round(v, 4) for k, v in rewards.items()},
        info=state,
        terminated=done,
    )


@app.post("/simulate/step/bulk")
def step_bulk(req: StepRequest, n_steps: int = 10):
    """Run N steps at once, return all states."""
    env = _get_env(req.session_id)
    states = []
    for _ in range(n_steps):
        actions = _auto_actions(env)
        obs, rewards, terminated, truncated, info = env.step(actions)
        state = _env_state_to_dict(env)
        states.append(state)
        if all(truncated.values()):
            break
    return {"session_id": req.session_id, "states": states}


@app.get("/simulate/state/{session_id}")
def get_state(session_id: str):
    env = _get_env(session_id)
    return {"session_id": session_id, "state": _env_state_to_dict(env)}


@app.get("/simulate/stream/{session_id}")
async def stream_simulation(session_id: str, fps: int = 4):
    """SSE stream for real-time frontend updates."""
    if session_id not in _envs:
        raise HTTPException(status_code=404)

    async def generator():
        env = _envs[session_id]["env"]
        for _ in range(500):
            actions = _auto_actions(env)
            _, rewards, _, truncated, _ = env.step(actions)
            state = _env_state_to_dict(env)
            data = json.dumps({"step": env.step_count, "state": state})
            yield f"data: {data}\n\n"
            await asyncio.sleep(1 / fps)
            if all(truncated.values()):
                yield "data: {\"done\": true}\n\n"
                break

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.delete("/simulate/{session_id}")
def delete_simulation(session_id: str):
    if session_id in _envs:
        del _envs[session_id]
    if session_id in _sim_states:
        del _sim_states[session_id]
    return {"status": "deleted", "session_id": session_id}


@app.get("/simulate/metrics/{session_id}")
def get_metrics(session_id: str):
    env = _get_env(session_id)
    return {
        "session_id": session_id,
        "summary": env.get_metrics_summary(),
        "step_metrics": env.step_metrics[-100:],
        "reward_history": env.reward_history[-200:],
    }


@app.post("/train/start", response_model=TrainJobResponse)
def start_training(req: TrainRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    _train_jobs[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": req.total_episodes,
        "logs": [],
        "started_at": time.time(),
    }

    def run_training():
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from backend.core.trainer import OpenEnvTrainer, TrainConfig
            config = TrainConfig(
                total_episodes=req.total_episodes,
                lr=req.lr,
                use_wandb=req.use_wandb,
                run_name=req.run_name,
                curriculum_thresholds=tuple(req.curriculum_thresholds),
            )
            trainer = OpenEnvTrainer(config)
            _train_jobs[job_id]["status"] = "running"
            logs = trainer.train()
            _train_jobs[job_id]["status"] = "complete"
            _train_jobs[job_id]["progress"] = req.total_episodes
            _train_jobs[job_id]["logs"] = [
                {k: round(v, 4) if isinstance(v, float) else v
                 for k, v in log.items()}
                for log in logs[-50:]
            ]
        except Exception as e:
            _train_jobs[job_id]["status"] = "error"
            _train_jobs[job_id]["error"] = str(e)

    background_tasks.add_task(run_training)
    return TrainJobResponse(
        job_id=job_id,
        status="queued",
        message=f"Training job {job_id} queued for {req.total_episodes} episodes",
    )


@app.get("/train/status/{job_id}")
def training_status(job_id: str):
    if job_id not in _train_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **_train_jobs[job_id]}


@app.get("/train/list")
def list_jobs():
    return {"jobs": [{
        "job_id": jid,
        "status": j["status"],
        "progress": j["progress"],
        "total": j["total"],
        "started_at": j["started_at"],
    } for jid, j in _train_jobs.items()]}


@app.get("/metrics/global")
def global_metrics():
    """Aggregate metrics across all active sessions."""
    if not _envs:
        return {"sessions": 0, "data": {}}
    all_metrics = []
    for sid, data in _envs.items():
        env = data["env"]
        m = env.get_metrics_summary()
        m["session_id"] = sid
        all_metrics.append(m)
    return {
        "sessions": len(all_metrics),
        "aggregate": {
            "total_lives_saved": sum(m["total_lives_saved"] for m in all_metrics),
            "total_lives_lost": sum(m["total_lives_lost"] for m in all_metrics),
            "avg_resolution_rate": np.mean([m["resolution_rate"] for m in all_metrics]),
            "avg_survival_rate": np.mean([m["survival_rate"] for m in all_metrics]),
        },
        "per_session": all_metrics,
    }


@app.get("/datasets/info")
def dataset_info():
    """Return dataset integration status."""
    return {
        "datasets": [
            {"name": "NFIRS", "source": "FEMA", "use": "Fire incident distributions", "integrated": True},
            {"name": "NEISS", "source": "CPSC", "use": "Medical emergency rates", "integrated": True},
            {"name": "NIBRS", "source": "FBI", "use": "Crime pattern timing", "integrated": True},
            {"name": "NHTSA FARS", "source": "NHTSA", "use": "Accident rates vs weather", "integrated": True},
            {"name": "Boston EMS", "source": "Kaggle", "use": "Response time baseline", "integrated": True},
            {"name": "FEMA Disasters", "source": "FEMA", "use": "Cascade probability graph", "integrated": True},
            {"name": "EM-DAT", "source": "CRED", "use": "Multi-hazard sequences", "integrated": True},
            {"name": "US Wildfires", "source": "Kaggle", "use": "Fire spread model", "integrated": True},
            {"name": "Uber Movement", "source": "Uber", "use": "ETA lookup table", "integrated": True},
            {"name": "Open-Meteo", "source": "Open-Meteo", "use": "Real-time weather", "integrated": True},
            {"name": "WorldPop", "source": "WorldPop Hub", "use": "Population density", "integrated": True},
            {"name": "OSM", "source": "OpenStreetMap", "use": "Road network", "integrated": True},
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
