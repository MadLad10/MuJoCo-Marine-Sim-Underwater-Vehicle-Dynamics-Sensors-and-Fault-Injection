"""Ground-truth referee (port of v1's referee node, no ROS): scores what really happened.

Events: gate_passed / gate_missed, buoy_touched, surfaced, boundary_contact, collision.
v2 additions: collisions (hull contacts), and localisation error of the onboard estimate against
truth (`record_estimate`). The eval scores missions only from `report()`.
"""
from __future__ import annotations

import math
from collections import deque

TOUCH_MARGIN = 0.35        # hull half-length + slack
SURFACE_WINDOW_S = 2.0     # a floating hull bobs: judge "surfaced" over a window
SURFACED_DEPTH_M = 0.3


def world_to_body(dx, dy, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


class Referee:
    def __init__(self, arena: dict):
        self.arena = arena
        self.events: list[dict] = []
        self.touched: set[str] = set()
        self.gate_side: dict[str, bool] = {}
        self.max_depth = 0.0
        self.was_deep = False
        self.at_boundary = False
        self.path_len = 0.0
        self.last_pos = None
        self.final = None
        self.t = 0.0
        self.recent: deque = deque()
        self.contacts: set[str] = set()
        self.loc_err: list[tuple[float, float, float]] = []   # (t, horizontal err, depth err)

    def emit(self, event, **data):
        self.events.append({'event': event, 't': round(self.t, 2), **data})

    def update(self, t: float, pos, contacts: list[str] | None = None):
        self.t = t
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        depth = -z
        if self.last_pos:
            self.path_len += math.dist(self.last_pos, (x, y, z))
        self.last_pos = (x, y, z)
        self.final = {'x': round(x, 2), 'y': round(y, 2), 'depth': round(depth, 2)}
        self.recent.append((t, depth))
        while self.recent and t - self.recent[0][0] > SURFACE_WINDOW_S:
            self.recent.popleft()
        self.max_depth = max(self.max_depth, depth)
        if depth > 0.5:
            self.was_deep = True
        if self.was_deep and depth < 0.2:
            self.was_deep = False
            self.emit('surfaced', x=round(x, 2), y=round(y, 2))
        for o in self.arena['objects']:
            if o['class'] == 'gate':
                self._gate(o, x, y, z)
            elif o['id'] not in self.touched:
                d = math.dist((x, y, z), (o['x'], o['y'], o['z']))
                if d < o.get('radius', 0.2) + TOUCH_MARGIN:
                    self.touched.add(o['id'])
                    self.emit('buoy_touched', buoy=o['id'], cls=o['class'], dist=round(d, 2))
        pool = self.arena['pool']
        x0 = pool.get('x_min', -5.0)
        near = (x - x0 < 0.4 or x0 + pool['length'] - x < 0.4 or pool['width'] / 2 - abs(y) < 0.4
                or pool['depth'] - depth < 0.3)
        if near and not self.at_boundary:
            self.emit('boundary_contact', x=round(x, 2), y=round(y, 2), depth=round(depth, 2))
        self.at_boundary = near
        now = set()
        for name in contacts or []:
            obj = name.split('__')[0]
            if obj.startswith('lane_'):
                continue
            now.add(obj)
        for obj in now - self.contacts:
            touched_buoy = obj in self.touched
            self.emit('collision', with_=obj, buoy=touched_buoy)
        self.contacts = now

    def record_estimate(self, t: float, est_xyz, true_xyz):
        eh = math.hypot(est_xyz[0] - true_xyz[0], est_xyz[1] - true_xyz[1])
        ez = abs(est_xyz[2] - true_xyz[2])
        self.loc_err.append((t, eh, ez))

    def _gate(self, o, x, y, z):
        gx, gy = world_to_body(x - o['x'], y - o['y'], o.get('yaw', 0.0))
        gz = z - o['z']
        side = gx > 0
        prev = self.gate_side.get(o['id'])
        self.gate_side[o['id']] = side
        if prev is None or prev == side:
            return
        inside = abs(gy) < o['width'] / 2 - 0.05 and abs(gz) < o['height'] / 2
        self.emit('gate_passed' if inside else 'gate_missed', gate=o['id'],
                  direction='forward' if side else 'backward', lateral=round(gy, 2), vertical=round(gz, 2))

    def surfaced_now(self) -> bool:
        if not self.recent:
            return False
        return sum(d for _, d in self.recent) / len(self.recent) < SURFACED_DEPTH_M

    def localisation(self) -> dict:
        if not self.loc_err:
            return {}
        eh = [e for _, e, _ in self.loc_err]
        ez = [e for _, _, e in self.loc_err]
        return {'rmse_xy_m': round(math.sqrt(sum(e * e for e in eh) / len(eh)), 3),
                'max_xy_m': round(max(eh), 3), 'final_xy_m': round(eh[-1], 3),
                'rmse_depth_m': round(math.sqrt(sum(e * e for e in ez) / len(ez)), 3), 'samples': len(eh)}

    def report(self) -> dict:
        ev = self.events
        return {
            'events': ev,
            'gate_passed': any(e['event'] == 'gate_passed' for e in ev),
            'buoys_touched': sorted(self.touched),
            'touch_order': [e['buoy'] for e in ev if e['event'] == 'buoy_touched'],
            'surfaced': any(e['event'] == 'surfaced' for e in ev),
            'surfaced_at_end': self.surfaced_now(),
            'max_depth_m': round(self.max_depth, 2),
            'boundary_contacts': sum(e['event'] == 'boundary_contact' for e in ev),
            'collisions': sum(e['event'] == 'collision' and not e.get('buoy') for e in ev),
            'path_length_m': round(self.path_len, 1),
            'final': self.final,
            'localisation': self.localisation(),
            'sim_time_s': round(self.t, 2),
        }
