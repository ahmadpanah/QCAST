# src/models/data_models.py
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime

@dataclass
class Metrics:
    cpu_usage: float
    memory_usage: float
    network_io: float
    disk_io: float
    request_count: int
    response_time: float
    error_rate: float

@dataclass
class Container:
    id: str
    status: str
    resources: Dict[str, float]
    metrics: Metrics
    creation_time: float

@dataclass
class Node:
    id: str
    available_resources: Dict[str, float]
    containers: List[Container]

@dataclass
class DemandForecast:
    predicted_demand: int
    confidence_interval: Tuple[int, int]
    trend: float
    seasonality_factor: float
    forecast_horizon: List[int]