"""Fault injection: scheduled sensor/actuator failures with a ground-truth log.

A fault is `{type, t_start, duration, params}`; `duration: null` means permanent. The injector is
queried by the sensors and the vehicle model every tick, and it records exactly when each fault was
active, so the health monitor and the diagnostics agent can be scored against ground truth.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# type -> (affected component, short description). The closed set used by the eval fault tier.
FAULT_TYPES = {
    'camera_blackout': ('camera', 'front camera returns black frames'),
    'turbidity': ('camera', 'visibility drops (params: visibility_m)'),
    'dvl_dropout': ('dvl', 'DVL loses bottom lock: no valid velocity'),
    'imu_bias': ('imu', 'gyro z bias step (params: gyro_bias_z rad/s)'),
    'compass_disturbance': ('compass', 'heading offset from a magnetic disturbance (params: offset_deg)'),
    'pressure_stuck': ('pressure', 'depth sensor freezes at its last value'),
    'sonar_false_target': ('sonar', 'phantom sonar target (params: x, y, z)'),
    'sonar_dropout': ('sonar', 'no sonar pings'),
    'thruster_dead': ('thrusters', 'one thruster produces no thrust (params: name)'),
    'leak': ('hull', 'leak sensor reports water ingress'),
    'battery_drop': ('battery', 'sudden capacity loss (params: drop_pct)'),
}


@dataclass
class Fault:
    type: str
    t_start: float = 0.0
    duration: Optional[float] = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in FAULT_TYPES:
            raise ValueError(f'unknown fault type {self.type!r}; known: {sorted(FAULT_TYPES)}')

    def active(self, t: float) -> bool:
        if t < self.t_start:
            return False
        return self.duration is None or t < self.t_start + self.duration

    @property
    def component(self) -> str:
        return FAULT_TYPES[self.type][0]


class FaultInjector:
    def __init__(self, faults: list | None = None):
        self.faults: list[Fault] = [f if isinstance(f, Fault) else Fault(**f) for f in (faults or [])]
        self.log: list[dict] = []          # onset/clear events (ground truth)
        self._state: dict[int, bool] = {}

    def add(self, fault: Fault | dict):
        self.faults.append(fault if isinstance(fault, Fault) else Fault(**fault))

    def update(self, t: float):
        """Record onsets/clears; call once per tick."""
        for i, f in enumerate(self.faults):
            a = f.active(t)
            if a != self._state.get(i, False):
                self._state[i] = a
                self.log.append({'t': round(t, 2), 'event': 'onset' if a else 'clear', **asdict(f)})

    def get(self, ftype: str, t: float) -> Optional[Fault]:
        for f in self.faults:
            if f.type == ftype and f.active(t):
                return f
        return None

    def all(self, ftype: str, t: float) -> list[Fault]:
        return [f for f in self.faults if f.type == ftype and f.active(t)]

    def active_types(self, t: float) -> list[str]:
        return sorted({f.type for f in self.faults if f.active(t)})

    def ground_truth(self) -> list[dict]:
        return [asdict(f) for f in self.faults]
