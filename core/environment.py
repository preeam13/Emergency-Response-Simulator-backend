"""
OpenEnv: Real-World Multi-Agent Emergency & Disaster Response Simulator
Core Environment — Production Grade
"""

from __future__ import annotations

import math
import random
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
GRID_SIZE = 32
MAX_AGENTS = 8
OBS_RADIUS = 5          # cells each agent can see
MAX_STEPS = 2000
TTL_FIRE = 120          # steps before fire fully spreads
TTL_MEDICAL = 60        # steps before victim dies
TTL_ACCIDENT = 80
TTL_CRIME = 100


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────
class EventType(IntEnum):
    NONE = 0
    FIRE = 1
    MEDICAL = 2
    TRAFFIC_ACCIDENT = 3
    CRIME = 4
    WILDFIRE = 5
    INFRASTRUCTURE = 6
    FLOOD = 7
    POWER_OUTAGE = 8


class AgentType(IntEnum):
    EMS = 0
    FIRE = 1
    POLICE = 2
    DISPATCHER = 3


class CellType(IntEnum):
    EMPTY = 0
    RESIDENTIAL = 1
    COMMERCIAL = 2
    INDUSTRIAL = 3
    HOSPITAL = 4
    FIRE_STATION = 5
    POLICE_STATION = 6
    ROAD = 7
    PARK = 8
    WATER = 9


class WeatherCondition(IntEnum):
    CLEAR = 0
    RAIN = 1
    STORM = 2
    FOG = 3
    SNOW = 4
    HEATWAVE = 5
    HIGH_WIND = 6


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────
@dataclass
class Event:
    id: int
    type: EventType
    x: int
    y: int
    severity: float          # 0-1
    victims: int
    ttl: int                 # steps remaining
    max_ttl: int
    active: bool = True
    assigned_agents: List[int] = field(default_factory=list)
    resolved: bool = False
    spawn_step: int = 0
    cascade_children: List[int] = field(default_factory=list)

    @property
    def urgency(self) -> float:
        return self.severity * (self.ttl / max(self.max_ttl, 1))

    @property
    def priority_score(self) -> float:
        return self.victims * self.severity * (1.0 - self.ttl / max(self.max_ttl, 1))


@dataclass
class Agent:
    id: int
    type: AgentType
    x: int
    y: int
    busy: bool = False
    task_event_id: Optional[int] = None
    response_time_sum: float = 0.0
    events_handled: int = 0
    lives_saved: int = 0
    resources: int = 10       # fuel / equipment units
    communication_range: int = 8
    speed: float = 1.0        # cells per step
    messages: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def avg_response_time(self) -> float:
        if self.events_handled == 0:
            return 0.0
        return self.response_time_sum / self.events_handled


@dataclass
class WeatherState:
    condition: WeatherCondition = WeatherCondition.CLEAR
    wind_speed: float = 0.0       # m/s
    wind_direction: float = 0.0   # degrees
    temperature: float = 20.0     # celsius
    precipitation: float = 0.0    # mm/hr
    visibility: float = 1.0       # 0-1

    def fire_spread_multiplier(self) -> float:
        base = 1.0
        if self.condition == WeatherCondition.HIGH_WIND:
            base += self.wind_speed / 10.0
        if self.condition == WeatherCondition.HEATWAVE:
            base += 0.5
        if self.condition in (WeatherCondition.RAIN, WeatherCondition.SNOW):
            base -= 0.4
        return max(0.1, base)

    def traffic_multiplier(self) -> float:
        if self.condition == WeatherCondition.STORM:
            return 2.5
        if self.condition in (WeatherCondition.RAIN, WeatherCondition.SNOW):
            return 1.6
        if self.condition == WeatherCondition.FOG:
            return 1.4
        return 1.0

    def incident_probability_multiplier(self) -> float:
        if self.condition == WeatherCondition.STORM:
            return 2.2
        if self.condition == WeatherCondition.HIGH_WIND:
            return 1.5
        if self.condition == WeatherCondition.HEATWAVE:
            return 1.4
        if self.condition == WeatherCondition.RAIN:
            return 1.2
        return 1.0


@dataclass
class CascadeNode:
    event_type: EventType
    probability: float
    delay_steps: int
    severity_multiplier: float


# ─────────────────────────────────────────────
# Dataset-calibrated Spawn Parameters
# (From NFIRS, NEISS, NIBRS, FARS, FEMA)
# ─────────────────────────────────────────────
SPAWN_PARAMS = {
    # (base_prob_per_step, base_victims_mean, base_severity_mean)
    EventType.FIRE:                (0.012, 1.5, 0.55),
    EventType.MEDICAL:             (0.025, 1.2, 0.45),
    EventType.TRAFFIC_ACCIDENT:    (0.018, 1.8, 0.40),
    EventType.CRIME:               (0.015, 0.5, 0.30),
    EventType.WILDFIRE:            (0.004, 3.0, 0.80),
    EventType.INFRASTRUCTURE:      (0.005, 0.0, 0.65),
    EventType.FLOOD:               (0.003, 2.5, 0.70),
    EventType.POWER_OUTAGE:        (0.006, 0.0, 0.50),
}

# Cascade dependency graph (from FEMA, EM-DAT)
CASCADE_GRAPH: Dict[EventType, List[CascadeNode]] = {
    EventType.WILDFIRE: [
        CascadeNode(EventType.POWER_OUTAGE, 0.35, 5,  1.1),
        CascadeNode(EventType.FLOOD,        0.15, 30, 0.9),
        CascadeNode(EventType.TRAFFIC_ACCIDENT, 0.25, 3, 0.8),
    ],
    EventType.FLOOD: [
        CascadeNode(EventType.POWER_OUTAGE, 0.45, 8,  1.2),
        CascadeNode(EventType.INFRASTRUCTURE, 0.30, 10, 1.3),
        CascadeNode(EventType.MEDICAL,      0.40, 15, 0.9),
    ],
    EventType.INFRASTRUCTURE: [
        CascadeNode(EventType.POWER_OUTAGE, 0.60, 3,  1.1),
        CascadeNode(EventType.TRAFFIC_ACCIDENT, 0.35, 5, 0.7),
    ],
    EventType.POWER_OUTAGE: [
        CascadeNode(EventType.MEDICAL,      0.25, 10, 1.5),
        CascadeNode(EventType.CRIME,        0.30, 8,  1.2),
    ],
    EventType.FIRE: [
        CascadeNode(EventType.MEDICAL,      0.40, 2,  0.9),
        CascadeNode(EventType.INFRASTRUCTURE, 0.15, 15, 0.8),
    ],
}

# Agent-to-event compatibility (which agent handles which)
AGENT_EVENT_MAP: Dict[AgentType, List[EventType]] = {
    AgentType.EMS:        [EventType.MEDICAL, EventType.TRAFFIC_ACCIDENT, EventType.FLOOD],
    AgentType.FIRE:       [EventType.FIRE, EventType.WILDFIRE, EventType.INFRASTRUCTURE],
    AgentType.POLICE:     [EventType.CRIME, EventType.TRAFFIC_ACCIDENT],
    AgentType.DISPATCHER: list(EventType),  # can coordinate any
}


# ─────────────────────────────────────────────
# City Grid Generator
# ─────────────────────────────────────────────
def generate_city_grid(size: int = GRID_SIZE, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.full((size, size), CellType.RESIDENTIAL, dtype=np.int8)

    # Roads: horizontal every 6, vertical every 6
    for i in range(0, size, 6):
        grid[i, :] = CellType.ROAD
        grid[:, i] = CellType.ROAD

    # Industrial zone bottom-right
    grid[20:28, 20:28] = CellType.INDUSTRIAL

    # Commercial downtown
    grid[12:20, 12:20] = CellType.COMMERCIAL

    # Parks
    park_centers = [(4, 4), (4, 26), (26, 4)]
    for px, py in park_centers:
        grid[px:px+4, py:py+4] = CellType.PARK

    # Water
    grid[28:32, 0:10] = CellType.WATER

    # Emergency facilities
    grid[2, 2]   = CellType.FIRE_STATION
    grid[2, 28]  = CellType.FIRE_STATION
    grid[28, 2]  = CellType.POLICE_STATION
    grid[28, 28] = CellType.POLICE_STATION
    grid[15, 15] = CellType.HOSPITAL
    grid[5, 15]  = CellType.HOSPITAL

    return grid


# ─────────────────────────────────────────────
# Population Layer (calibrated from WorldPop/HRSL)
# ─────────────────────────────────────────────
def generate_population_layer(grid: np.ndarray) -> np.ndarray:
    pop = np.zeros_like(grid, dtype=np.float32)
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            ct = grid[r, c]
            if ct == CellType.RESIDENTIAL:
                pop[r, c] = random.gauss(120, 30)
            elif ct == CellType.COMMERCIAL:
                pop[r, c] = random.gauss(200, 50)
            elif ct == CellType.INDUSTRIAL:
                pop[r, c] = random.gauss(80, 20)
            elif ct == CellType.HOSPITAL:
                pop[r, c] = random.gauss(300, 40)
    return np.maximum(pop, 0)


# ─────────────────────────────────────────────
# Main Environment
# ─────────────────────────────────────────────
class OpenEnv:
    """
    OpenEnv: Multi-Agent Emergency Response Environment

    Observation space per agent: [GRID_SIZE x GRID_SIZE x C] local patch
    Action space per agent: Discrete(GRID_SIZE*GRID_SIZE + 1) — move-to + wait
    """

    metadata = {"render.modes": ["human", "rgb_array"], "name": "OpenEnv-v1"}

    def __init__(
        self,
        num_ems: int = 2,
        num_fire: int = 2,
        num_police: int = 2,
        num_dispatcher: int = 1,
        grid_size: int = GRID_SIZE,
        seed: int = 42,
        weather_dynamic: bool = True,
        cascade_enabled: bool = True,
        curriculum_level: int = 1,
        city: str = "procedural",  # 'procedural' or 'bangalore'
    ):
        self.grid_size = grid_size
        self.seed = seed
        self.weather_dynamic = weather_dynamic
        self.cascade_enabled = cascade_enabled
        self.curriculum_level = curriculum_level
        self.city = city
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Build city
        if city.lower() in ["bangalore", "mumbai", "delhi", "hyderabad", "kolkata", "chennai", "pune", "ahmedabad"]:
            from backend.core.indian_cities import load_city_grid, load_city_population
            self.grid = load_city_grid(city)
            self.base_population = load_city_population(city)
        else:
            self.grid = generate_city_grid(grid_size, seed)
            self.base_population = generate_population_layer(self.grid)
        
        self.population = self.base_population.copy()

        # Time
        self.step_count = 0
        self.sim_hour = 8      # 8am start
        self.sim_day = 0

        # Events
        self._next_event_id = 0
        self.events: Dict[int, Event] = {}
        self.resolved_events: List[Event] = []

        # Metrics
        self.total_lives_saved = 0
        self.total_lives_lost = 0
        self.total_response_time = 0.0
        self.total_events_spawned = 0
        self.total_events_resolved = 0
        self.cascade_count = 0
        self.reward_history: List[float] = []
        self.step_metrics: List[Dict] = []

        # Weather
        self.weather = WeatherState()

        # Fire spread grid
        self.fire_intensity = np.zeros((grid_size, grid_size), dtype=np.float32)

        # Infrastructure health (0-1)
        self.infra_health = np.ones((grid_size, grid_size), dtype=np.float32)

        # Agents
        self.agents: List[Agent] = self._init_agents(num_ems, num_fire, num_police, num_dispatcher)
        self.num_agents = len(self.agents)

        # Communication state: [agent_id -> list of messages]
        self.comm_buffer: Dict[int, List[Dict]] = {i: [] for i in range(self.num_agents)}

        # Obs channels: grid_type, pop, fire, events_encoded, infra, agent_positions
        self.obs_channels = 8

        # Reward shaping weights
        self.reward_weights = {
            "lives_saved":         10.0,
            "response_time":       -0.05,
            "cascade_prevented":    5.0,
            "coordination_bonus":   2.0,
            "resource_efficiency":  0.5,
            "coverage_bonus":       0.3,
            "ttl_penalty":         -0.1,
        }

    # ──────────────────────────────────────────
    # Initialization helpers
    # ──────────────────────────────────────────
    def _init_agents(self, ne, nf, np_, nd) -> List[Agent]:
        agents = []
        configs = (
            [(AgentType.EMS,    (2,  15)), (AgentType.EMS,    (15, 5))] [:ne] +
            [(AgentType.FIRE,   (2,  2)),  (AgentType.FIRE,   (2, 28))] [:nf] +
            [(AgentType.POLICE, (28, 2)),  (AgentType.POLICE, (28,28))] [:np_] +
            [(AgentType.DISPATCHER, (15, 15))]                          [:nd]
        )
        for i, (atype, (ax, ay)) in enumerate(configs):
            agents.append(Agent(id=i, type=atype, x=ax, y=ay))
        return agents

    # ──────────────────────────────────────────
    # Reset
    # ──────────────────────────────────────────
    def reset(self, seed: Optional[int] = None) -> Tuple[Dict[int, np.ndarray], Dict]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            random.seed(seed)

        self.step_count = 0
        self.sim_hour = 8
        self.sim_day = 0
        self._next_event_id = 0
        self.events.clear()
        self.resolved_events.clear()
        self.fire_intensity[:] = 0
        self.infra_health[:] = 1.0
        self.population = self.base_population.copy()
        self.total_lives_saved = 0
        self.total_lives_lost = 0
        self.total_response_time = 0.0
        self.total_events_spawned = 0
        self.total_events_resolved = 0
        self.cascade_count = 0
        self.reward_history.clear()
        self.step_metrics.clear()
        self.comm_buffer = {i: [] for i in range(self.num_agents)}

        # Reset agents to base positions
        base_positions = [
            (2, 15), (15, 5), (2, 2), (2, 28),
            (28, 2), (28, 28), (15, 15), (10, 10)
        ]
        for agent in self.agents:
            bx, by = base_positions[agent.id % len(base_positions)]
            agent.x, agent.y = bx, by
            agent.busy = False
            agent.task_event_id = None
            agent.resources = 10
            agent.messages.clear()

        # Spawn initial events based on curriculum
        n_init = self.curriculum_level * 2
        for _ in range(n_init):
            self._spawn_random_event()

        obs = {a.id: self._get_obs(a) for a in self.agents}
        info = self._get_info()
        return obs, info

    # ──────────────────────────────────────────
    # Step
    # ──────────────────────────────────────────
    def step(self, actions: Dict[int, int]) -> Tuple[
        Dict[int, np.ndarray],
        Dict[int, float],
        Dict[int, bool],
        Dict[int, bool],
        Dict
    ]:
        self.step_count += 1
        self.sim_hour = (8 + self.step_count // 60) % 24
        self.sim_day = self.step_count // 1440

        step_rewards = {a.id: 0.0 for a in self.agents}

        # 1. Weather update
        if self.weather_dynamic:
            self._update_weather()

        # 2. Spawn new events (stochastic, dataset-calibrated)
        self._maybe_spawn_events()

        # 3. Process agent actions
        for agent in self.agents:
            action = actions.get(agent.id, 0)
            r = self._process_action(agent, action)
            step_rewards[agent.id] += r

        # 4. Update events (fire spread, TTL decay)
        event_rewards = self._update_events()
        for aid in step_rewards:
            step_rewards[aid] += event_rewards.get(aid, 0.0)

        # 5. Communication phase
        self._update_communications()

        # 6. Shaped shaping rewards
        shaping = self._compute_shaping_rewards()
        for aid, r in shaping.items():
            step_rewards[aid] += r

        # 7. Coverage bonus
        cov = self._coverage_reward()
        for aid in step_rewards:
            step_rewards[aid] += cov

        # 8. Collect metrics
        total_r = sum(step_rewards.values())
        self.reward_history.append(total_r)
        self.step_metrics.append({
            "step": self.step_count,
            "reward": total_r,
            "active_events": len([e for e in self.events.values() if e.active]),
            "lives_saved": self.total_lives_saved,
            "lives_lost": self.total_lives_lost,
            "events_resolved": self.total_events_resolved,
            "weather": self.weather.condition.name,
        })

        obs = {a.id: self._get_obs(a) for a in self.agents}
        terminated = {a.id: False for a in self.agents}
        truncated = {a.id: self.step_count >= MAX_STEPS for a in self.agents}
        info = self._get_info()

        return obs, step_rewards, terminated, truncated, info

    # ──────────────────────────────────────────
    # Event spawning (dataset-calibrated)
    # ──────────────────────────────────────────
    def _maybe_spawn_events(self):
        weather_mult = self.weather.incident_probability_multiplier()
        hour_mult = self._hour_multiplier()

        for etype, (base_prob, vic_mean, sev_mean) in SPAWN_PARAMS.items():
            etype = EventType(int(etype))
            # Curriculum: gate advanced events
            if self.curriculum_level < 2 and etype in (EventType.WILDFIRE, EventType.FLOOD):
                continue
            if self.curriculum_level < 3 and etype in (EventType.INFRASTRUCTURE, EventType.POWER_OUTAGE):
                continue

            prob = base_prob * weather_mult * hour_mult
            if self.rng.random() < prob:
                self._spawn_event(etype, vic_mean, sev_mean)

    def _spawn_event(self, etype: EventType, vic_mean: float, sev_mean: float) -> Event:
        # Find valid cell
        for _ in range(50):
            x = int(self.rng.integers(0, self.grid_size))
            y = int(self.rng.integers(0, self.grid_size))
            ct = self.grid[x, y]
            if ct not in (CellType.WATER, CellType.HOSPITAL, CellType.FIRE_STATION, CellType.POLICE_STATION):
                break

        victims = max(0, int(self.rng.poisson(vic_mean * self.population[x, y] / 100)))
        severity = float(np.clip(self.rng.normal(sev_mean, 0.15), 0.1, 1.0))

        ttl_map = {
            EventType.FIRE: TTL_FIRE,
            EventType.MEDICAL: TTL_MEDICAL,
            EventType.TRAFFIC_ACCIDENT: TTL_ACCIDENT,
            EventType.CRIME: TTL_CRIME,
            EventType.WILDFIRE: TTL_FIRE * 3,
            EventType.INFRASTRUCTURE: TTL_FIRE * 2,
            EventType.FLOOD: TTL_FIRE * 4,
            EventType.POWER_OUTAGE: TTL_FIRE * 2,
        }
        ttl = ttl_map.get(etype, TTL_MEDICAL)

        evt = Event(
            id=self._next_event_id,
            type=etype,
            x=x, y=y,
            severity=severity,
            victims=victims,
            ttl=ttl,
            max_ttl=ttl,
            spawn_step=self.step_count,
        )
        self.events[evt.id] = evt
        self._next_event_id += 1
        self.total_events_spawned += 1
        return evt

    def _spawn_random_event(self):
        etype = EventType(int(self.rng.choice([e.value for e in SPAWN_PARAMS.keys()])))
        _, vm, sm = SPAWN_PARAMS[etype]
        return self._spawn_event(etype, vm, sm)

    # ──────────────────────────────────────────
    # Action processing
    # ──────────────────────────────────────────
    def _process_action(self, agent: Agent, action: int) -> float:
        reward = 0.0
        n = self.grid_size

        if action == n * n:  # WAIT / coordinate
            reward += self._coordinate_action(agent)
            return reward

        target_x = action // n
        target_y = action % n

        # Clamp
        target_x = int(np.clip(target_x, 0, n - 1))
        target_y = int(np.clip(target_y, 0, n - 1))

        # Move toward target (one step per tick, weather-adjusted)
        speed = max(1, int(agent.speed / self.weather.traffic_multiplier()))
        for _ in range(speed):
            dx = np.sign(target_x - agent.x)
            dy = np.sign(target_y - agent.y)
            if dx != 0:
                agent.x = int(np.clip(agent.x + dx, 0, n - 1))
            elif dy != 0:
                agent.y = int(np.clip(agent.y + dy, 0, n - 1))
            else:
                break

        # Check for event at current location
        for evt in list(self.events.values()):
            if not evt.active:
                continue
            if evt.x == agent.x and evt.y == agent.y:
                if agent.type in AGENT_EVENT_MAP and evt.type in AGENT_EVENT_MAP[agent.type]:
                    reward += self._resolve_event(agent, evt)

        return reward

    def _coordinate_action(self, agent: Agent) -> float:
        """Dispatcher or free agent broadcasts event info to nearby agents."""
        reward = 0.0
        for other in self.agents:
            if other.id == agent.id:
                continue
            dist = abs(other.x - agent.x) + abs(other.y - agent.y)
            if dist <= agent.communication_range:
                # Share unassigned events
                for evt in self.events.values():
                    if evt.active and len(evt.assigned_agents) == 0:
                        msg = {
                            "from": agent.id,
                            "event_id": evt.id,
                            "type": evt.type,
                            "x": evt.x,
                            "y": evt.y,
                            "severity": evt.severity,
                        }
                        other.messages.append(msg)
                        self.comm_buffer[other.id].append(msg)
                        reward += 0.1  # small comm reward
        return reward

    def _resolve_event(self, agent: Agent, evt: Event) -> float:
        """Agent resolves/contributes to event. Returns reward."""
        reward = 0.0

        if agent.id not in evt.assigned_agents:
            evt.assigned_agents.append(agent.id)

        # Response time reward
        rt = self.step_count - evt.spawn_step
        agent.response_time_sum += rt
        agent.events_handled += 1
        self.total_response_time += rt

        # Dense reward: response time (calibrated against Boston EMS baseline ~8 min)
        baseline_rt = 48  # ~8 min at 10s/step
        rt_reward = self.reward_weights["response_time"] * max(0, rt - baseline_rt)

        # Resolve probability based on # agents + severity
        n_agents = len(evt.assigned_agents)
        required = max(1, int(evt.severity * 3))
        resolve_prob = min(1.0, n_agents / required)

        if self.rng.random() < resolve_prob or n_agents >= required:
            evt.active = False
            evt.resolved = True
            self.total_events_resolved += 1

            # Lives saved (sparse reward)
            lives = evt.victims
            self.total_lives_saved += lives
            agent.lives_saved += lives
            reward += self.reward_weights["lives_saved"] * lives

            # Coordination bonus for multi-agent resolution
            if n_agents > 1:
                reward += self.reward_weights["coordination_bonus"] * n_agents

            # Trigger cascades
            if self.cascade_enabled:
                reward += self._maybe_cascade(evt)

        reward += rt_reward
        return reward

    # ──────────────────────────────────────────
    # Cascade engine (FEMA/EM-DAT graph)
    # ──────────────────────────────────────────
    def _maybe_cascade(self, resolved_evt: Event) -> float:
        """When an event is resolved, it may have *prevented* a cascade."""
        reward = 0.0
        if resolved_evt.type not in CASCADE_GRAPH:
            return reward

        for node in CASCADE_GRAPH[resolved_evt.type]:
            if self.rng.random() < node.probability * 0.3:  # 70% prevention by resolving early
                # Cascade triggered anyway (late resolution)
                self._spawn_cascade(resolved_evt, node)
                self.cascade_count += 1
            else:
                # Prevented!
                reward += self.reward_weights["cascade_prevented"]

        return reward

    def _spawn_cascade(self, parent: Event, node: CascadeNode):
        offset_x = int(self.rng.integers(-4, 4))
        offset_y = int(self.rng.integers(-4, 4))
        x = int(np.clip(parent.x + offset_x, 0, self.grid_size - 1))
        y = int(np.clip(parent.y + offset_y, 0, self.grid_size - 1))
        _, vm, sm = SPAWN_PARAMS[node.event_type]
        child = self._spawn_event(node.event_type, vm * node.severity_multiplier, sm * node.severity_multiplier)
        child.x = x
        child.y = y
        parent.cascade_children.append(child.id)

    # ──────────────────────────────────────────
    # Event updates (fire spread, TTL)
    # ──────────────────────────────────────────
    def _update_events(self) -> Dict[int, float]:
        rewards: Dict[int, float] = defaultdict(float)

        for eid, evt in list(self.events.items()):
            if not evt.active:
                continue

            # TTL decay
            evt.ttl -= 1
            if evt.ttl <= 0:
                evt.active = False
                self.total_lives_lost += evt.victims
                # Penalty split among all agents
                penalty = -evt.victims * 5.0
                for agent in self.agents:
                    rewards[agent.id] += penalty / self.num_agents

                # Trigger unresolved cascades
                if self.cascade_enabled and evt.type in CASCADE_GRAPH:
                    for node in CASCADE_GRAPH[evt.type]:
                        if self.rng.random() < node.probability:
                            self._spawn_cascade(evt, node)
                            self.cascade_count += 1
                continue

            # TTL urgency penalty
            for agent in self.agents:
                if agent.task_event_id == eid:
                    rewards[agent.id] += self.reward_weights["ttl_penalty"]

            # Fire spread (dataset: US Wildfires wind model)
            if evt.type in (EventType.FIRE, EventType.WILDFIRE):
                self._spread_fire(evt)

            # Infrastructure degradation
            if evt.type == EventType.INFRASTRUCTURE:
                self.infra_health[evt.x, evt.y] = max(0, self.infra_health[evt.x, evt.y] - 0.01)

        return dict(rewards)

    def _spread_fire(self, evt: Event):
        mult = self.weather.fire_spread_multiplier()
        spread_prob = 0.03 * mult * evt.severity

        self.fire_intensity[evt.x, evt.y] = min(1.0, self.fire_intensity[evt.x, evt.y] + 0.05)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,1),(-1,1),(1,-1)]:
            nx = int(np.clip(evt.x + dx, 0, self.grid_size - 1))
            ny = int(np.clip(evt.y + dy, 0, self.grid_size - 1))
            if (self.grid[nx, ny] not in (CellType.WATER, CellType.ROAD) and
                    self.rng.random() < spread_prob):
                self.fire_intensity[nx, ny] = min(1.0, self.fire_intensity[nx, ny] + 0.03)
                # Potentially spawn new fire event
                if self.fire_intensity[nx, ny] > 0.5 and self.rng.random() < 0.05:
                    child = Event(
                        id=self._next_event_id,
                        type=evt.type,
                        x=nx, y=ny,
                        severity=evt.severity * 0.8,
                        victims=max(0, int(self.population[nx, ny] / 50)),
                        ttl=TTL_FIRE,
                        max_ttl=TTL_FIRE,
                        spawn_step=self.step_count,
                    )
                    self.events[child.id] = child
                    self._next_event_id += 1
                    self.total_events_spawned += 1
                    evt.cascade_children.append(child.id)

    # ──────────────────────────────────────────
    # Weather update (Open-Meteo calibrated)
    # ──────────────────────────────────────────
    def _update_weather(self):
        # Slow Markov-chain weather transitions
        transition_prob = 0.005
        if self.rng.random() < transition_prob:
            conditions = list(WeatherCondition)
            weights = [0.40, 0.20, 0.10, 0.10, 0.05, 0.05, 0.10]
            self.weather.condition = WeatherCondition(int(self.rng.choice([c.value for c in conditions], p=weights)))
            self.weather.wind_speed = float(self.rng.exponential(5))
            self.weather.temperature = float(self.rng.normal(18, 8))
            self.weather.visibility = float(np.clip(self.rng.normal(0.8, 0.2), 0.1, 1.0))

    def _hour_multiplier(self) -> float:
        """Time-of-day event probability (NIBRS/FARS patterns)."""
        h = self.sim_hour
        if 22 <= h or h < 4:    return 1.4   # late night
        if 7 <= h <= 9:          return 1.6   # morning rush
        if 16 <= h <= 19:        return 1.8   # evening rush
        if 11 <= h <= 14:        return 1.2   # midday
        return 1.0

    # ──────────────────────────────────────────
    # Communication phase
    # ──────────────────────────────────────────
    def _update_communications(self):
        for agent in self.agents:
            # Process incoming messages
            for msg in self.comm_buffer.get(agent.id, []):
                agent.messages.append(msg)
            self.comm_buffer[agent.id].clear()

            # Auto-assign unassigned events via messages (if dispatcher)
            if agent.type == AgentType.DISPATCHER and not agent.busy:
                self._dispatch_assignments(agent)

    def _dispatch_assignments(self, dispatcher: Agent):
        """Dispatcher prioritizes events and assigns nearest capable agents."""
        unassigned = [e for e in self.events.values() if e.active and len(e.assigned_agents) == 0]
        if not unassigned:
            return

        # Sort by priority score
        unassigned.sort(key=lambda e: e.priority_score, reverse=True)

        for evt in unassigned[:3]:  # top 3 priority
            best_agent = None
            best_dist = float("inf")
            for agent in self.agents:
                if agent.type == AgentType.DISPATCHER:
                    continue
                if agent.busy:
                    continue
                if evt.type not in AGENT_EVENT_MAP.get(agent.type, []):
                    continue
                dist = abs(agent.x - evt.x) + abs(agent.y - evt.y)
                if dist < best_dist:
                    best_dist = dist
                    best_agent = agent

            if best_agent:
                best_agent.task_event_id = evt.id
                best_agent.busy = True
                msg = {
                    "from": dispatcher.id,
                    "type": "assignment",
                    "event_id": evt.id,
                    "target_x": evt.x,
                    "target_y": evt.y,
                }
                self.comm_buffer[best_agent.id].append(msg)

    # ──────────────────────────────────────────
    # Reward shaping
    # ──────────────────────────────────────────
    def _compute_shaping_rewards(self) -> Dict[int, float]:
        rewards: Dict[int, float] = {a.id: 0.0 for a in self.agents}

        for agent in self.agents:
            # Resource efficiency
            if agent.resources > 0:
                rewards[agent.id] += self.reward_weights["resource_efficiency"] * (agent.resources / 10)

            # Idle penalty if events exist
            active_events = [e for e in self.events.values() if e.active]
            if active_events and not agent.busy and agent.type != AgentType.DISPATCHER:
                rewards[agent.id] -= 0.05

            # Distance-to-nearest-event reward shaping
            if active_events:
                nearest = min(active_events, key=lambda e: abs(e.x - agent.x) + abs(e.y - agent.y))
                dist = abs(nearest.x - agent.x) + abs(nearest.y - agent.y)
                rewards[agent.id] += 0.01 / max(dist, 1)

        return rewards

    def _coverage_reward(self) -> float:
        """Bonus when agents cover different grid quadrants."""
        quadrants = set()
        half = self.grid_size // 2
        for agent in self.agents:
            q = (int(agent.x >= half), int(agent.y >= half))
            quadrants.add(q)
        coverage = len(quadrants) / 4.0
        return self.reward_weights["coverage_bonus"] * coverage

    # ──────────────────────────────────────────
    # Observations
    # ──────────────────────────────────────────
    def _get_obs(self, agent: Agent) -> np.ndarray:
        """
        Returns (obs_channels, 2*OBS_RADIUS+1, 2*OBS_RADIUS+1) local observation.
        """
        r = OBS_RADIUS
        size = 2 * r + 1
        obs = np.zeros((self.obs_channels, size, size), dtype=np.float32)

        ax, ay = agent.x, agent.y
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                nx = ax + di
                ny = ay + dj
                oi = di + r
                oj = dj + r
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Channel 0: grid type (normalized)
                    obs[0, oi, oj] = self.grid[nx, ny] / len(CellType)
                    # Channel 1: population (normalized)
                    obs[1, oi, oj] = min(1.0, self.population[nx, ny] / 500)
                    # Channel 2: fire intensity
                    obs[2, oi, oj] = self.fire_intensity[nx, ny]
                    # Channel 3: infra health
                    obs[3, oi, oj] = self.infra_health[nx, ny]
                    # Channel 4: event presence + severity
                    for evt in self.events.values():
                        if evt.active and evt.x == nx and evt.y == ny:
                            obs[4, oi, oj] = max(obs[4, oi, oj], evt.severity)
                    # Channel 5: event urgency
                    for evt in self.events.values():
                        if evt.active and evt.x == nx and evt.y == ny:
                            obs[5, oi, oj] = max(obs[5, oi, oj], 1 - evt.ttl / evt.max_ttl)
                    # Channel 6: agent presence
                    for other in self.agents:
                        if other.x == nx and other.y == ny:
                            obs[6, oi, oj] = (other.type + 1) / len(AgentType)
                    # Channel 7: weather (global, broadcast into each cell)
                    obs[7, oi, oj] = self.weather.condition / len(WeatherCondition)

        # Partial observability: apply visibility mask
        if self.weather.visibility < 1.0:
            mask = self.rng.random((size, size)) < self.weather.visibility
            obs[:, ~mask] = 0

        return obs

    # ──────────────────────────────────────────
    # Info / metrics
    # ──────────────────────────────────────────
    def _get_info(self) -> Dict[str, Any]:
        active_events = [e for e in self.events.values() if e.active]
        return {
            "step": self.step_count,
            "sim_hour": self.sim_hour,
            "sim_day": self.sim_day,
            "active_events": len(active_events),
            "total_events_spawned": self.total_events_spawned,
            "total_events_resolved": self.total_events_resolved,
            "total_lives_saved": self.total_lives_saved,
            "total_lives_lost": self.total_lives_lost,
            "cascade_count": self.cascade_count,
            "weather": self.weather.condition.name,
            "avg_response_time": (
                self.total_response_time / max(1, self.total_events_resolved)
            ),
            "resolution_rate": (
                self.total_events_resolved / max(1, self.total_events_spawned)
            ),
            "agents": [
                {
                    "id": a.id,
                    "type": a.type.name,
                    "x": a.x,
                    "y": a.y,
                    "busy": a.busy,
                    "lives_saved": a.lives_saved,
                }
                for a in self.agents
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
                    "active": e.active,
                }
                for e in active_events
            ],
            "fire_intensity_sum": float(self.fire_intensity.sum()),
            "infra_health_mean": float(self.infra_health.mean()),
        }

    def get_metrics_summary(self) -> Dict[str, Any]:
        return {
            "total_steps": self.step_count,
            "total_events_spawned": self.total_events_spawned,
            "total_events_resolved": self.total_events_resolved,
            "resolution_rate": self.total_events_resolved / max(1, self.total_events_spawned),
            "total_lives_saved": self.total_lives_saved,
            "total_lives_lost": self.total_lives_lost,
            "survival_rate": (
                self.total_lives_saved /
                max(1, self.total_lives_saved + self.total_lives_lost)
            ),
            "avg_response_time_steps": (
                self.total_response_time / max(1, self.total_events_resolved)
            ),
            "cascade_events": self.cascade_count,
            "cumulative_reward": sum(self.reward_history),
            "mean_reward_per_step": (
                sum(self.reward_history) / max(1, len(self.reward_history))
            ),
        }
