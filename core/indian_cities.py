"""
Indian Cities Database — Real Infrastructure for Major Metro Areas
Includes: Bangalore, Mumbai, Delhi, Hyderabad, Kolkata, Chennai, Pune, Ahmedabad
"""

import numpy as np
from typing import Dict, Tuple, List

GRID_SIZE = 32

# City Bounding Boxes (lat/lon for georeferencing)
CITY_BOUNDS = {
    "bangalore": {"south": 12.863, "north": 13.112, "west": 77.451, "east": 77.768},
    "mumbai": {"south": 18.900, "north": 19.300, "west": 72.800, "east": 73.050},
    "delhi": {"south": 28.400, "north": 28.880, "west": 76.800, "east": 77.350},
    "hyderabad": {"south": 17.300, "north": 17.450, "west": 78.350, "east": 78.550},
    "kolkata": {"south": 22.500, "north": 22.640, "west": 88.300, "east": 88.480},
    "chennai": {"south": 12.800, "north": 13.100, "west": 80.150, "east": 80.300},
    "pune": {"south": 18.450, "north": 18.650, "west": 73.700, "east": 73.900},
    "ahmedabad": {"south": 23.000, "north": 23.250, "west": 72.500, "east": 72.700},
}

# Real Infrastructure Data — Major Emergency Services
CITY_INFRASTRUCTURE = {
    "bangalore": {
        "name": "Bangalore",
        "population_real": 8_500_000,
        "area_km2": 709,
        "fire_stations": [(2, 2), (8, 4), (4, 16), (20, 10), (12, 22), (26, 18)],
        "hospitals": [(6, 8), (10, 12), (15, 15), (18, 6), (12, 28)],
        "police_stations": [(4, 2), (14, 10), (20, 8), (8, 20)],
        "parks": [(3, 3), (10, 20), (24, 24)],
        "high_density_zones": [(10, 16), (8, 20), (15, 20)],  # Whitefield, Indiranagar, Marathahalli
        "industrial_zones": [(2, 20), (24, 8)],  # Peenya, Yeshwanthpur
    },
    
    "mumbai": {
        "name": "Mumbai",
        "population_real": 20_500_000,
        "area_km2": 603,
        "fire_stations": [(4, 4), (8, 8), (4, 24), (20, 12), (12, 28), (24, 20)],
        "hospitals": [(10, 10), (14, 14), (18, 8), (12, 22), (6, 28)],
        "police_stations": [(8, 4), (16, 10), (20, 6), (12, 24)],
        "parks": [(2, 2), (26, 6), (28, 26)],
        "high_density_zones": [(12, 12), (14, 10), (16, 16)],  # South Mumbai, Worli, BKC
        "industrial_zones": [(2, 24), (28, 4)],  # Docks, JNPT
    },
    
    "delhi": {
        "name": "Delhi",
        "population_real": 16_700_000,
        "area_km2": 1484,
        "fire_stations": [(4, 8), (10, 4), (6, 20), (22, 10), (14, 24), (26, 12)],
        "hospitals": [(8, 6), (12, 10), (16, 14), (12, 22), (20, 8)],
        "police_stations": [(6, 6), (14, 8), (18, 12), (10, 22)],
        "parks": [(4, 4), (20, 20), (26, 26)],
        "high_density_zones": [(10, 10), (14, 14), (18, 16)],  # Central Delhi, South Delhi
        "industrial_zones": [(2, 20), (24, 4)],  # DSIIDC, factories
    },
    
    "hyderabad": {
        "name": "Hyderabad",
        "population_real": 6_800_000,
        "area_km2": 625,
        "fire_stations": [(4, 6), (10, 10), (6, 22), (20, 14), (12, 24)],
        "hospitals": [(8, 8), (14, 12), (16, 18), (12, 20)],
        "police_stations": [(6, 4), (14, 10), (18, 14), (10, 20)],
        "parks": [(2, 2), (24, 8), (26, 24)],
        "high_density_zones": [(12, 10), (14, 14), (16, 16)],  # Kukatpally, Banjara Hills, HITEC
        "industrial_zones": [(2, 24), (26, 6)],  # MIDC, Export zones
    },
    
    "kolkata": {
        "name": "Kolkata",
        "population_real": 14_600_000,
        "area_km2": 1278,
        "fire_stations": [(4, 4), (10, 8), (6, 20), (20, 10), (12, 24)],
        "hospitals": [(8, 6), (12, 12), (14, 16), (18, 8), (10, 22)],
        "police_stations": [(6, 6), (14, 10), (16, 14), (8, 20)],
        "parks": [(2, 2), (26, 8), (24, 24)],
        "high_density_zones": [(10, 10), (12, 12), (14, 14)],  # Central, South Kolkata
        "industrial_zones": [(2, 20), (26, 4)],  # Port, Eastern industrial
    },
    
    "chennai": {
        "name": "Chennai",
        "population_real": 7_000_000,
        "area_km2": 426,
        "fire_stations": [(4, 6), (10, 10), (6, 20), (18, 14), (12, 24)],
        "hospitals": [(8, 8), (14, 12), (16, 16), (12, 20)],
        "police_stations": [(6, 4), (14, 10), (16, 14), (10, 20)],
        "parks": [(2, 2), (22, 8), (24, 22)],
        "high_density_zones": [(10, 10), (12, 12), (14, 14)],  # Central, Egmore
        "industrial_zones": [(2, 22), (24, 6)],  # Port, Ambattur
    },
    
    "pune": {
        "name": "Pune",
        "population_real": 6_500_000,
        "area_km2": 331,
        "fire_stations": [(4, 4), (10, 8), (6, 20), (18, 12), (12, 24)],
        "hospitals": [(8, 6), (12, 10), (14, 14), (16, 20)],
        "police_stations": [(6, 6), (12, 10), (16, 14), (10, 20)],
        "parks": [(2, 2), (22, 6), (24, 24)],
        "high_density_zones": [(10, 8), (12, 10), (14, 12)],  # Central, Camp
        "industrial_zones": [(2, 20), (24, 4)],  # MIDC zones
    },
    
    "ahmedabad": {
        "name": "Ahmedabad",
        "population_real": 8_500_000,
        "area_km2": 466,
        "fire_stations": [(4, 6), (10, 10), (6, 20), (18, 12), (12, 24)],
        "hospitals": [(8, 8), (12, 12), (14, 16), (16, 20)],
        "police_stations": [(6, 6), (12, 10), (16, 14), (10, 20)],
        "parks": [(2, 2), (22, 8), (24, 22)],
        "high_density_zones": [(10, 10), (12, 12), (14, 14)],  # Central, Paldi
        "industrial_zones": [(2, 22), (24, 4)],  # GIDC zones
    },
}


def load_city_grid(city: str = "bangalore") -> np.ndarray:
    """Load city grid for Indian metro with real infrastructure."""
    from backend.core.environment import CellType
    
    city_lower = city.lower()
    if city_lower not in CITY_INFRASTRUCTURE:
        raise ValueError(f"City '{city}' not found. Available: {list(CITY_INFRASTRUCTURE.keys())}")
    
    infra = CITY_INFRASTRUCTURE[city_lower]
    grid = np.full((GRID_SIZE, GRID_SIZE), CellType.RESIDENTIAL, dtype=np.int8)
    
    # ─────────────────────────────────────────
    # Roads (major thoroughfares)
    # ─────────────────────────────────────────
    grid[6, :] = CellType.ROAD
    grid[14, :] = CellType.ROAD
    grid[22, :] = CellType.ROAD
    grid[:, 8] = CellType.ROAD
    grid[:, 16] = CellType.ROAD
    grid[:, 24] = CellType.ROAD
    
    # ─────────────────────────────────────────
    # Commercial zones (CBD)
    # ─────────────────────────────────────────
    for hd in infra["high_density_zones"]:
        cx, cy = hd
        grid[max(0, cx-2):min(GRID_SIZE, cx+2), max(0, cy-2):min(GRID_SIZE, cy+2)] = CellType.COMMERCIAL
    
    # ─────────────────────────────────────────
    # Industrial zones
    # ─────────────────────────────────────────
    for iz in infra["industrial_zones"]:
        ix, iy = iz
        grid[max(0, ix-3):min(GRID_SIZE, ix+4), max(0, iy-3):min(GRID_SIZE, iy+4)] = CellType.INDUSTRIAL
    
    # ─────────────────────────────────────────
    # Parks
    # ─────────────────────────────────────────
    for px, py in infra["parks"]:
        grid[max(0, px-2):min(GRID_SIZE, px+3), max(0, py-2):min(GRID_SIZE, py+3)] = CellType.PARK
    
    # ─────────────────────────────────────────
    # Water bodies
    # ─────────────────────────────────────────
    grid[26:GRID_SIZE, 0:6] = CellType.WATER
    grid[0:4, 28:GRID_SIZE] = CellType.WATER
    
    # ─────────────────────────────────────────
    # Fire Stations
    # ─────────────────────────────────────────
    for fx, fy in infra["fire_stations"]:
        if 0 <= fx < GRID_SIZE and 0 <= fy < GRID_SIZE:
            grid[fx, fy] = CellType.FIRE_STATION
    
    # ─────────────────────────────────────────
    # Hospitals
    # ─────────────────────────────────────────
    for hx, hy in infra["hospitals"]:
        if 0 <= hx < GRID_SIZE and 0 <= hy < GRID_SIZE:
            grid[hx, hy] = CellType.HOSPITAL
    
    # ─────────────────────────────────────────
    # Police Stations
    # ─────────────────────────────────────────
    for px, py in infra["police_stations"]:
        if 0 <= px < GRID_SIZE and 0 <= py < GRID_SIZE:
            grid[px, py] = CellType.POLICE_STATION
    
    return grid


def load_city_population(city: str = "bangalore") -> np.ndarray:
    """Load population density for city."""
    city_lower = city.lower()
    if city_lower not in CITY_INFRASTRUCTURE:
        raise ValueError(f"City '{city}' not found")
    
    infra = CITY_INFRASTRUCTURE[city_lower]
    pop = np.ones((GRID_SIZE, GRID_SIZE), dtype=np.float32) * 300
    
    # High density zones
    for hd in infra["high_density_zones"]:
        cx, cy = hd
        pop[max(0, cx-2):min(GRID_SIZE, cx+2), max(0, cy-2):min(GRID_SIZE, cy+2)] = 1000
    
    # Medium density (residential)
    pop[4:8, 4:8] = 600
    pop[16:22, 14:20] = 600
    
    # Low density
    pop[0:4, :] = 100
    pop[26:GRID_SIZE, :] = 100
    
    return pop


def get_city_metadata(city: str = "bangalore") -> dict:
    """Metadata about city simulation."""
    city_lower = city.lower()
    if city_lower not in CITY_INFRASTRUCTURE:
        raise ValueError(f"City '{city}' not found")
    
    infra = CITY_INFRASTRUCTURE[city_lower]
    bounds = CITY_BOUNDS[city_lower]
    
    return {
        "city": infra["name"],
        "country": "India",
        "population_real": infra["population_real"],
        "area_km2": infra["area_km2"],
        "bounding_box": bounds,
        "grid_size": GRID_SIZE,
        "cell_size_km": round(infra["area_km2"] / (GRID_SIZE ** 2), 2),
        "fire_stations": len(infra["fire_stations"]),
        "hospitals": len(infra["hospitals"]),
        "police_stations": len(infra["police_stations"]),
        "description": f"Multi-agent emergency response simulation for {infra['name']}"
    }


def list_available_cities() -> List[str]:
    """Return list of available cities."""
    return list(CITY_INFRASTRUCTURE.keys())
