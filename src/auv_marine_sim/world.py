"""Build the MuJoCo (MJCF) world from a v1 arena YAML and the sim config.

The arena YAML stays the single source of truth for geometry and ground truth (shared with v1).
Buoyancy and hydrodynamic drag are NOT MuJoCo features here: `MarineSim` applies them
as external wrenches every step, so the MJCF only holds rigid bodies, thrusters and sensors.
"""
from __future__ import annotations

import math
import os

import yaml

BUOY_RGBA = {
    'red_buoy': '0.9 0.05 0.05 1', 'green_buoy': '0.05 0.8 0.1 1', 'yellow_buoy': '0.95 0.85 0.05 1',
    'purple_buoy': '0.55 0.05 0.75 1',
}
GATE_RGBA = '1 0.45 0.0 1'
GATE_BAR_RGBA = '0.08 0.08 0.08 1'
POST_RADIUS = 0.05


def share_dir() -> str:
    """Use this installed package's resources, independent of ROS overlays."""
    from importlib.resources import files
    return str(files('auv_marine_sim'))


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def arena_path(name_or_path: str) -> str:
    if os.path.isfile(name_or_path):
        return name_or_path
    for base in (share_dir(), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')):
        p = os.path.join(base, 'config', 'arenas', f'{name_or_path}.yaml')
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f'arena not found: {name_or_path}')


def load_arena(name_or_path: str) -> dict:
    return load_yaml(arena_path(name_or_path))


def default_config_path() -> str:
    for base in (share_dir(), os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')):
        p = os.path.join(base, 'config', 'sim.yaml')
        if os.path.isfile(p):
            return p
    raise FileNotFoundError('sim.yaml not found')


def load_config(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = load_yaml(path or default_config_path())
    if overrides:
        deep_update(cfg, overrides)
    return cfg


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _f(*v):
    return ' '.join(f'{x:.4g}' for x in v)


def _pool(arena) -> str:
    p = arena['pool']
    L, W, D, x0 = p['length'], p['width'], p['depth'], p.get('x_min', -5.0)
    cx, t = x0 + L / 2, 0.1
    walls = [
        ('pool_floor', (cx, 0, -D - t), (L / 2, W / 2, t), '0.55 0.55 0.48 1'),
        ('wall_back', (x0 - t, 0, -D / 2), (t, W / 2, D / 2), '0.45 0.55 0.6 1'),
        ('wall_front', (x0 + L + t, 0, -D / 2), (t, W / 2, D / 2), '0.45 0.55 0.6 1'),
        ('wall_left', (cx, W / 2 + t, -D / 2), (L / 2, t, D / 2), '0.45 0.55 0.6 1'),
        ('wall_right', (cx, -W / 2 - t, -D / 2), (L / 2, t, D / 2), '0.45 0.55 0.6 1'),
    ]
    out = [f'<geom name="{n}" type="box" pos="{_f(*pos)}" size="{_f(*size)}" rgba="{rgba}" '
           f'class="structure"/>' for n, pos, size, rgba in walls]
    # lane markers on the floor (texture for the camera; no collision)
    for i, x in enumerate(range(int(x0) + 2, int(x0 + L) - 1, 3)):
        out.append(f'<geom name="lane_{i}" type="box" pos="{_f(x, 0, -D + 0.002)}" '
                   f'size="0.6 0.08 0.002" rgba="0.15 0.15 0.15 1" contype="0" conaffinity="0" group="2"/>')
    return '\n    '.join(out)


def _gate(o) -> str:
    w, h, r = o['width'], o['height'], POST_RADIUS
    yaw = o.get('yaw', 0.0)
    return f"""<body name="{o['id']}" pos="{_f(o['x'], o['y'], o['z'])}" euler="0 0 {yaw:.4g}">
      <geom name="{o['id']}__post_left" type="cylinder" pos="0 {w / 2:.4g} 0" size="{r} {h / 2:.4g}" rgba="{GATE_RGBA}" class="object"/>
      <geom name="{o['id']}__post_right" type="cylinder" pos="0 {-w / 2:.4g} 0" size="{r} {h / 2:.4g}" rgba="{GATE_RGBA}" class="object"/>
      <geom name="{o['id']}__bar" type="cylinder" pos="0 0 {h / 2:.4g}" euler="1.5708 0 0" size="{r} {w / 2 + r:.4g}" rgba="{GATE_BAR_RGBA}" class="object"/>
    </body>"""


def _buoy(o) -> str:
    # No contacts (as in v1): the referee judges "touched" from geometry, so the hull can reach the
    # centre. Sonar and camera still see the buoy (ray casts ignore contact flags).
    r = o.get('radius', 0.2)
    anchor_len = max(0.0, (o.get('pool_depth', 5.0) + o['z']) - r)
    tether = ''
    if anchor_len > 0.05:
        tether = (f'<geom name="{o["id"]}__tether" type="cylinder" pos="0 0 {-(r + anchor_len / 2):.4g}" '
                  f'size="0.006 {anchor_len / 2:.4g}" rgba="0.2 0.2 0.2 1" contype="0" conaffinity="0" group="2"/>')
    return f"""<body name="{o['id']}" pos="{_f(o['x'], o['y'], o['z'])}">
      <geom name="{o['id']}__ball" type="sphere" size="{r}" material="mat_{o['class'] if o['class'] in BUOY_RGBA else 'default'}" class="object" contype="0" conaffinity="0"/>
      {tether}
    </body>"""


def _auv(arena, cfg) -> str:
    v = cfg['vehicle']
    s = arena['start']
    hx, hy, hz = (d / 2 for d in v['hull'])
    m = v['mass']
    Lx, Ly, Lz = v['hull']
    inertia = _f(m / 12 * (Ly ** 2 + Lz ** 2), m / 12 * (Lx ** 2 + Lz ** 2), m / 12 * (Lx ** 2 + Ly ** 2))
    cam = cfg['sensors']['camera']
    fovy = math.degrees(2 * math.atan(math.tan(math.radians(cam['hfov_deg']) / 2) * cam['height'] / cam['width']))
    sites = []
    for name, (x, y, z, dx, dy, dz) in v['thrusters'].items():
        sites.append(f'<site name="thr_{name}" pos="{_f(x, y, z)}" size="0.02" rgba="0.1 0.1 0.1 1"/>')
    return f"""<body name="auv" pos="{_f(s['x'], s['y'], s['z'])}" euler="0 0 {s.get('yaw', 0.0):.4g}">
      <freejoint name="auv_free"/>
      <inertial pos="{_f(*v['com'])}" mass="{m}" diaginertia="{inertia}"/>
      <geom name="auv_hull" type="box" size="{_f(hx, hy, hz)}" rgba="0.92 0.92 0.92 1" mass="0" group="1"/>
      <geom name="auv_nose" type="box" pos="{hx + 0.005:.4g} 0 0" size="0.005 0.06 0.04" rgba="0.1 0.1 0.12 1" mass="0" contype="0" conaffinity="0" group="1"/>
      {chr(10).join('      ' + s_ for s_ in sites)}
      <site name="imu" pos="0 0 0" size="0.01"/>
      <site name="sonar" pos="{_f(*cfg['sensors']['sonar']['offset'])}" size="0.01"/>
      <camera name="front" pos="{_f(*cam['offset'])}" xyaxes="0 -1 0 0 0 1" fovy="{fovy:.4g}"/>
    </body>"""


def build_mjcf(arena: dict, cfg: dict) -> str:
    ph = cfg['physics']
    cam = cfg['sensors']['camera']
    pool_depth = arena['pool']['depth']
    objects = []
    for o in arena['objects']:
        o = dict(o, pool_depth=pool_depth)
        objects.append(_gate(o) if o['class'] == 'gate' else _buoy(o))
    actuators = []
    for name, (x, y, z, dx, dy, dz) in cfg['vehicle']['thrusters'].items():
        mt = cfg['vehicle']['max_thrust_n']
        actuators.append(f'<motor name="{name}" site="thr_{name}" gear="{_f(dx, dy, dz)} 0 0 0" '
                         f'ctrllimited="true" ctrlrange="{-mt} {mt}"/>')
    sx, sy, sz = cfg['render']['spectator']
    return f"""<mujoco model="{arena.get('world_name', 'pool')}">
  <compiler angle="radian" autolimits="true"/>
  <!-- extent=1 so that clip distances below are in metres -->
  <statistic extent="1" center="10 0 -2.5"/>
  <option timestep="{ph['timestep']}" gravity="0 0 {-ph['gravity']}" density="0" viscosity="0" integrator="implicitfast"/>
  <visual>
    <global offwidth="{max(800, cam['width'])}" offheight="{max(450, cam['height'])}"/>
    <headlight ambient="0.35 0.38 0.4" diffuse="0.5 0.5 0.5" specular="0.05 0.05 0.05"/>
    <map znear="0.03" zfar="80"/>
    <quality shadowsize="0"/>
  </visual>
  <asset>
    {''.join(f'<material name="mat_{c}" rgba="{rgba}" emission="0.5" specular="0.2"/>' for c, rgba in {**BUOY_RGBA, 'default': '0.8 0.8 0.8 1'}.items())}
  </asset>
  <default>
    <default class="structure"><geom contype="1" conaffinity="1" group="4" friction="0.5"/></default>
    <default class="object"><geom contype="1" conaffinity="1" group="0" friction="0.5"/></default>
  </default>
  <worldbody>
    <light name="sun" pos="5 0 8" dir="0 0 -1" directional="true" diffuse="0.55 0.65 0.7" specular="0 0 0" castshadow="false"/>
    <camera name="spectator" pos="{_f(sx, sy, sz)}" xyaxes="1 0 0 0 0.29 0.96" fovy="58"/>
    <camera name="topdown" pos="{_f(arena['pool'].get('x_min', -5.0) + arena['pool']['length'] / 2, 0, 22)}" xyaxes="1 0 0 0 1 0" fovy="60"/>
    <geom name="water_surface" type="box" pos="{_f(arena['pool'].get('x_min', -5.0) + arena['pool']['length'] / 2, 0, 0.005)}" size="{_f(arena['pool']['length'] / 2 + 1, arena['pool']['width'] / 2 + 1, 0.005)}" rgba="0.55 0.8 0.9 1" contype="0" conaffinity="0" group="3"/>
    {_pool(arena)}
    {chr(10).join('    ' + o for o in objects)}
    {_auv(arena, cfg)}
  </worldbody>
  <actuator>
    {chr(10).join('    ' + a for a in actuators)}
  </actuator>
  <sensor>
    <gyro name="gyro" site="imu"/>
    <accelerometer name="accel" site="imu"/>
    <framequat name="quat" objtype="site" objname="imu"/>
    <velocimeter name="vel_body" site="imu"/>
  </sensor>
</mujoco>
"""
