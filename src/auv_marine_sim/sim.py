"""MuJoCo marine simulator: vehicle hydrodynamics, thrusters, sensors with noise and faults.

Lockstep: `step()` advances one control tick (`physics.control_dt`, several MuJoCo steps) and
samples every sensor whose period has elapsed. Nothing here is visible to the LLM; the onboard
software reads `readings` (noisy, possibly failed) and the referee reads `truth()`.

Hydrodynamics are applied as external wrenches each MuJoCo step:
  buoyancy  rho*g*V*f at the centre of buoyancy, f = submerged fraction (so the hull bobs at the
            surface); the centre of mass sits below it, which gives a righting moment
  damping   linear + quadratic in the body frame, relative to the water current (v1 coefficients)
Thrusters: first-order lag, per-thruster health (fault injection), forces applied by MuJoCo motors.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco  # noqa: E402

from .faults import FaultInjector  # noqa: E402
from .referee import Referee  # noqa: E402
from .world import build_mjcf, load_arena, load_config  # noqa: E402


# geom groups (world.py): 0 objects, 1 vehicle, 2 decoration, 3 water surface, 4 pool structure
GROUP_STRUCTURE = [0, 0, 0, 0, 1, 0]
GROUP_SONAR = [1, 0, 0, 0, 1, 0]


def quat_to_rpy(q):
    w, x, y, z = q
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Truth:
    t: float
    pos: np.ndarray        # body origin (centre of buoyancy), world
    rpy: tuple
    vel_world: np.ndarray  # linear velocity of the body origin, world
    vel_body: np.ndarray   # linear velocity, body frame
    omega_body: np.ndarray

    @property
    def depth(self):
        return -float(self.pos[2])

    @property
    def yaw(self):
        return self.rpy[2]


class MarineSim:
    def __init__(self, arena='default', config: dict | str | None = None, faults=None, seed: int = 0,
                 overrides: dict | None = None):
        self.arena = arena if isinstance(arena, dict) else load_arena(arena)
        self.cfg = config if isinstance(config, dict) else load_config(config, overrides)
        if isinstance(config, dict) and overrides:
            from .world import deep_update
            deep_update(self.cfg, overrides)
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.arena, self.cfg))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)
        self.faults = faults if isinstance(faults, FaultInjector) else FaultInjector(faults)
        ph, v = self.cfg['physics'], self.cfg['vehicle']
        self.dt = ph['timestep']
        self.control_dt = ph['control_dt']
        self.substeps = max(1, int(round(self.control_dt / self.dt)))
        self.rho, self.g = ph['water_density'], ph['gravity']
        self.current = np.array(ph.get('current', [0, 0, 0]), float)
        self.hull = np.array(v['hull'], float)
        self.volume = float(np.prod(self.hull))
        self.Dl = np.array(v['damping_linear'], float)
        self.Dq = np.array(v['damping_quadratic'], float)
        self.tau = v['thruster_tau_s']
        self.max_thrust = v['max_thrust_n']
        self.thruster_names = list(v['thrusters'])
        self.thr_geom = np.array([v['thrusters'][n] for n in self.thruster_names], float)  # x y z dx dy dz
        self.body = self.model.body('auv').id
        self.hull_geom = self.model.geom('auv_hull').id
        self.act = np.array([self.model.actuator(n).id for n in self.thruster_names])
        self.cmd = np.zeros(len(self.thruster_names))      # requested thrust (N)
        self.thrust = np.zeros(len(self.thruster_names))   # actual thrust after lag + health
        self.t = 0.0
        self.readings: dict[str, dict] = {}
        self._next: dict[str, float] = {}
        self._gyro_bias = np.zeros(3)
        self._stuck_depth = None
        self._battery_drops_applied: set[int] = set()
        b = self.cfg['sensors']['battery']
        self.energy_wh = b['capacity_wh'] * b['initial_percent'] / 100.0
        self._render_pool = None           # EGL contexts are thread-bound: all rendering on one thread
        self._geom_class = self._geom_classes()
        self._geom_obj = self._geom_objects()
        self._sonar_imager = None
        self._sonar_dirs = self._make_sonar_dirs()
        self._sonar_targets = self._make_sonar_targets()
        self._dvl_dirs = self._make_dvl_dirs()
        self.referee = Referee(self.arena)
        mujoco.mj_forward(self.model, self.data)
        self._sample_sensors(force=True)

    # ------------------------------------------------------------------ setup helpers
    def _geom_objects(self):
        """geom id -> object id (arena id; 'structure' for the pool), tethers excluded."""
        ids = {o['id'] for o in self.arena['objects']}
        out = {}
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name or ''
            obj = name.split('__')[0]
            if name.endswith('__tether'):
                continue
            part = {'post_left': '#L', 'post_right': '#R', 'bar': '#bar'}.get(name.split('__')[-1], '')
            if obj in ids:
                out[g] = obj + part            # gate posts are separate sonar targets
            elif name.startswith(('pool_', 'wall_')):
                out[g] = 'structure'
        return out

    def _geom_classes(self):
        """geom id -> object class (gate / red_buoy / ...) or 'structure' / None."""
        by_id = {o['id']: o['class'] for o in self.arena['objects']}
        out = {}
        for g in range(self.model.ngeom):
            name = self.model.geom(g).name or ''
            obj = name.split('__')[0]
            if name.endswith('__tether'):          # mooring line: not part of the object's label
                continue
            if obj in by_id:
                out[g] = by_id[obj]
            elif name.startswith(('pool_', 'wall_')):
                out[g] = 'structure'
        return out

    def _make_sonar_dirs(self):
        s = self.cfg['sensors']['sonar']
        az = np.radians(np.linspace(-s['hfov_deg'] / 2, s['hfov_deg'] / 2, s['beams']))
        el = np.radians(np.array(s['elevations_deg'], float))
        A, E = np.meshgrid(az, el, indexing='ij')          # beams x elevations
        dirs = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1)
        return az, dirs.reshape(-1, 3)

    def _make_sonar_targets(self):
        """Thin course objects as vertical segments (object id, top, bottom) in world coordinates."""
        out = []
        for o in self.arena['objects']:
            c = np.array([o['x'], o['y'], o['z']], float)
            if o['class'] == 'gate':
                w, h, yaw = o['width'], o['height'], o.get('yaw', 0.0)
                pts = []
                for side, tag in ((1, 'L'), (-1, 'R')):
                    off = np.array([-math.sin(yaw), math.cos(yaw), 0.0]) * side * w / 2
                    pts.append((f'{o["id"]}#{tag}', c + off + [0, 0, h / 2], c + off - [0, 0, h / 2]))
                out.append(('gate', pts))
            else:
                r = o.get('radius', 0.2)
                out.append((o['class'], [(o['id'], c + [0, 0, r], c - [0, 0, r])]))
        return out

    def _make_dvl_dirs(self):
        a = math.radians(self.cfg['sensors']['dvl']['beam_angle_deg'])
        s, c = math.sin(a), math.cos(a)
        return np.array([[s, 0, -c], [-s, 0, -c], [0, s, -c], [0, -s, -c]], float)

    # ------------------------------------------------------------------ state
    def R(self) -> np.ndarray:
        return self.data.xmat[self.body].reshape(3, 3).copy()

    def truth(self) -> Truth:
        d = self.data
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self.body, vel, 1)  # local
        R = self.R()
        q = d.xquat[self.body]
        return Truth(self.t, d.xpos[self.body].copy(), quat_to_rpy(q), R @ vel[3:], vel[3:].copy(), vel[:3].copy())

    def health_mask(self) -> np.ndarray:
        mask = np.ones(len(self.thruster_names))
        for f in self.faults.all('thruster_dead', self.t):
            name = f.params.get('name')
            if name in self.thruster_names:
                mask[self.thruster_names.index(name)] = 0.0
        return mask

    # ------------------------------------------------------------------ actuation
    def set_thrust(self, cmd):
        """Requested thrust per thruster (N), dict {name: N} or array in `thruster_names` order.
        Only the onboard controller calls this; agents/tools have no path to it."""
        if isinstance(cmd, dict):
            cmd = [cmd.get(n, 0.0) for n in self.thruster_names]
        self.cmd = np.clip(np.asarray(cmd, float), -self.max_thrust, self.max_thrust)

    def _apply_hydro(self):
        d = self.data
        R = self.R()
        pos = d.xpos[self.body]
        com = d.xipos[self.body]
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, d, mujoco.mjtObj.mjOBJ_XBODY, self.body, vel, 1)
        w_b, v_b = vel[:3], vel[3:] - R.T @ self.current
        f_b = -(self.Dl[:3] * v_b + self.Dq[:3] * np.abs(v_b) * v_b)
        t_b = -(self.Dl[3:] * w_b + self.Dq[3:] * np.abs(w_b) * w_b)
        h = self.hull[2]
        frac = float(np.clip(0.5 - pos[2] / h, 0.0, 1.0))         # submerged fraction (upright approx)
        f_buoy = np.array([0.0, 0.0, self.rho * self.g * self.volume * frac])
        f_w = R @ f_b * max(frac, 0.2) + f_buoy                   # little drag in air
        arm = pos - com                                            # forces act at the body origin
        d.xfrc_applied[self.body, :3] = f_w
        d.xfrc_applied[self.body, 3:] = R @ t_b + np.cross(arm, f_w)

    def step(self, n: int = 1):
        for _ in range(n):
            mask = self.health_mask()
            for _k in range(self.substeps):
                self.thrust += (self.cmd * mask - self.thrust) * min(1.0, self.dt / self.tau)
                self.data.ctrl[self.act] = self.thrust
                self._apply_hydro()
                mujoco.mj_step(self.model, self.data)
            self.t = round(self.t + self.control_dt, 6)
            self.faults.update(self.t)
            self._drain_battery()
            self._sample_sensors()
            tr = self.truth()
            self.referee.update(self.t, tr.pos, self._hull_contacts())

    def run_for(self, seconds: float, controller=None):
        for _ in range(int(round(seconds / self.control_dt))):
            if controller is not None:
                controller(self)
            self.step()

    def _hull_contacts(self) -> list[str]:
        out = []
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if self.hull_geom in (c.geom1, c.geom2):
                other = c.geom2 if c.geom1 == self.hull_geom else c.geom1
                out.append(self.model.geom(other).name)
        return out

    # ------------------------------------------------------------------ battery
    def _drain_battery(self):
        b = self.cfg['sensors']['battery']
        p = b['hotel_w'] + b['w_per_newton'] * float(np.abs(self.thrust).sum())
        self.energy_wh = max(0.0, self.energy_wh - p * self.control_dt / 3600.0)
        for i, f in enumerate(self.faults.faults):
            if f.type == 'battery_drop' and f.active(self.t) and i not in self._battery_drops_applied:
                self._battery_drops_applied.add(i)
                self.energy_wh = max(0.0, self.energy_wh - b['capacity_wh'] * f.params.get('drop_pct', 30.0) / 100)

    def battery_percent(self) -> float:
        return 100.0 * self.energy_wh / self.cfg['sensors']['battery']['capacity_wh']

    # ------------------------------------------------------------------ sensors
    def _due(self, name, rate_hz, force=False) -> bool:
        if force or self.t + 1e-9 >= self._next.get(name, 0.0):
            self._next[name] = self.t + 1.0 / rate_hz
            return True
        return False

    def _sample_sensors(self, force=False):
        s = self.cfg['sensors']
        tr = self.truth()
        if self._due('imu', s['imu']['rate_hz'], force):
            self.readings['imu'] = self._imu(tr)
        if self._due('compass', s['compass']['rate_hz'], force):
            self.readings['compass'] = self._compass(tr)
        if self._due('pressure', s['pressure']['rate_hz'], force):
            self.readings['pressure'] = self._pressure(tr)
        if self._due('dvl', s['dvl']['rate_hz'], force):
            self.readings['dvl'] = self._dvl(tr)
        if self._due('sonar', s['sonar']['rate_hz'], force):
            self.readings['sonar'] = self._sonar(tr)
        if self._due('battery', 1.0, force):
            pct = self.battery_percent()
            self.readings['battery'] = {'t': self.t, 'percent': round(pct, 2),
                                        'voltage': round(14.0 + 2.8 * pct / 100.0, 3),
                                        'power_w': round(s['battery']['hotel_w'] + s['battery']['w_per_newton']
                                                         * float(np.abs(self.thrust).sum()), 2)}
        if self._due('thrusters', s['thruster_telemetry']['rate_hz'], force):
            k = self.cfg['vehicle']['current_coeff']
            self.readings['thrusters'] = {
                't': self.t,
                'names': self.thruster_names,
                'cmd_n': [round(float(c), 2) for c in self.cmd],
                'thrust_n': [round(float(x), 2) for x in self.thrust],
                'current_a': [round(float(k * abs(x) ** 1.5 + self.rng.normal(0, 0.05)), 3) for x in self.thrust],
            }
        if self._due('leak', s['leak']['rate_hz'], force):
            self.readings['leak'] = {'t': self.t, 'leak': self.faults.get('leak', self.t) is not None}

    def _imu(self, tr: Truth) -> dict:
        c = self.cfg['sensors']['imu']
        dt = 1.0 / c['rate_hz']
        self._gyro_bias += self.rng.normal(0, c['gyro_bias_walk'] * math.sqrt(dt), 3)
        bias = self._gyro_bias.copy()
        f = self.faults.get('imu_bias', self.t)
        if f:
            bias[2] += f.params.get('gyro_bias_z', 0.02)
        sd = self.data.sensordata
        gyro = sd[0:3] + bias + self.rng.normal(0, c['gyro_noise'], 3)
        accel = sd[3:6] + self.rng.normal(0, c['accel_noise'], 3)
        an = math.radians(c['attitude_noise_deg'])
        return {'t': self.t, 'gyro': gyro.tolist(), 'accel': accel.tolist(),
                'roll': tr.rpy[0] + self.rng.normal(0, an), 'pitch': tr.rpy[1] + self.rng.normal(0, an)}

    def _compass(self, tr: Truth) -> dict:
        c = self.cfg['sensors']['compass']
        off = 0.0
        f = self.faults.get('compass_disturbance', self.t)
        if f:
            off = math.radians(f.params.get('offset_deg', 25.0))
        return {'t': self.t, 'yaw': wrap(tr.yaw + off + self.rng.normal(0, math.radians(c['noise_deg'])))}

    def _pressure(self, tr: Truth) -> dict:
        c = self.cfg['sensors']['pressure']
        depth = tr.depth + self.rng.normal(0, c['noise_m'])
        if self.faults.get('pressure_stuck', self.t):
            if self._stuck_depth is None:
                self._stuck_depth = self.readings.get('pressure', {}).get('depth', depth)
            depth = self._stuck_depth
        else:
            self._stuck_depth = None
        return {'t': self.t, 'depth': round(float(depth), 4)}

    def _cast(self, origin, dirs_world, geomgroup, cutoff):
        n = len(dirs_world)
        gid = np.full(n, -1, np.int32)
        dist = np.full(n, -1.0)
        mujoco.mj_multiRay(self.model, self.data, np.asarray(origin, float), dirs_world.reshape(-1).copy(),
                           np.asarray(geomgroup, np.uint8), 1, self.body, gid, dist, None, n, cutoff)
        return gid, dist

    def _dvl(self, tr: Truth) -> dict:
        c = self.cfg['sensors']['dvl']
        R = self.R()
        gid, dist = self._cast(tr.pos, self._dvl_dirs @ R.T, GROUP_STRUCTURE, c['max_range_m'])  # bottom track
        floor_hits = [d for g, d in zip(gid, dist) if g >= 0 and 0 < d <= c['max_range_m']
                      and self._geom_class.get(int(g)) is not None]
        valid = len(floor_hits) == 4 and self.faults.get('dvl_dropout', self.t) is None
        alt = float(np.mean(floor_hits) * math.cos(math.radians(c['beam_angle_deg']))) if floor_hits else -1.0
        valid = valid and alt >= c['min_altitude_m']
        if not valid:
            return {'t': self.t, 'valid': False, 'vel': None, 'altitude': round(alt, 3)}
        v_ground_body = R.T @ tr.vel_world
        vel = v_ground_body * (1 + c['scale_error']) + self.rng.normal(0, c['vel_noise_mps'], 3)
        return {'t': self.t, 'valid': True, 'vel': vel.tolist(), 'altitude': round(alt, 3)}

    def _sonar(self, tr: Truth) -> dict:
        c = self.cfg['sensors']['sonar']
        az, dirs = self._sonar_dirs
        if self.faults.get('sonar_dropout', self.t):
            return {'t': self.t, 'valid': False, 'bearings': az.tolist(), 'ranges': [], 'intensity': []}
        R = self.R()
        origin = tr.pos + R @ np.array(c['offset'], float)
        gid, dist = self._cast(origin, dirs @ R.T, GROUP_SONAR, c['max_range_m'])
        nb, ne = len(az), len(dirs) // len(az)
        gid, dist = gid.reshape(nb, ne), dist.reshape(nb, ne)
        ranges = np.full(nb, -1.0)
        inten = np.zeros(nb)
        truth = [None] * nb
        echoes = []                      # (beam, range, strength, class, object id): for sonar images
        for i in range(nb):
            hits = [(d, g) for d, g in zip(dist[i], gid[i]) if g >= 0 and 0 < d <= c['max_range_m']]
            for d, g in hits:
                cl = self._geom_class.get(int(g))
                ts = 0.45 if cl in (None, 'structure') else 1.0            # target strength
                echoes.append((i, d + self.rng.normal(0, c['range_noise_m']), ts / (1 + 0.05 * d), cl,
                               self._geom_obj.get(int(g))))
            if not hits or self.rng.random() < c['p_miss']:
                continue
            d, g = min(hits)
            cls = self._geom_class.get(int(g))
            ranges[i] = d + self.rng.normal(0, c['range_noise_m'])
            inten[i] = (1.0 if cls not in (None, 'structure') else 0.6) / (1 + 0.05 * d) + abs(self.rng.normal(0, 0.05))
            truth[i] = cls
        # Course objects are thin (10 cm gate posts) and can fall between ray-cast beams; a real beam
        # is ~1 deg wide, so each object also returns on its nearest beam (analytic geometry, occlusion
        # by structure checked with a ray).
        bw = abs(az[1] - az[0]) if nb > 1 else 0.02
        el_lim = math.radians(max(abs(e) for e in c['elevations_deg'])) + math.radians(5)
        for cls, pts in self._sonar_targets:
            for oid, top, bot in pts:
                rel_t, rel_b = R.T @ (top - origin), R.T @ (bot - origin)
                mid = (rel_t + rel_b) / 2
                rng_ = float(np.linalg.norm(mid))
                b = math.atan2(mid[1], mid[0])
                e_hi = math.atan2(rel_t[2], math.hypot(rel_t[0], rel_t[1]))
                e_lo = math.atan2(rel_b[2], math.hypot(rel_b[0], rel_b[1]))
                if mid[0] <= 0 or rng_ > c['max_range_m'] or abs(b) > az[-1] + bw / 2:
                    continue
                if e_lo > el_lim or e_hi < -el_lim or self.rng.random() < c['p_miss']:
                    continue
                i = int(np.argmin(np.abs(az - b)))
                if 0 < ranges[i] < rng_ - 0.3:          # something closer already answers on that beam
                    continue
                ranges[i] = rng_ + self.rng.normal(0, c['range_noise_m'])
                inten[i] = 1.0 / (1 + 0.05 * rng_) + abs(self.rng.normal(0, 0.05))
                truth[i] = cls
                echoes.append((i, ranges[i], 1.0 / (1 + 0.05 * rng_), cls, oid))
        for f in self.faults.all('sonar_false_target', self.t):
            p = np.array([f.params.get('x', 0.0), f.params.get('y', 0.0), f.params.get('z', -1.5)])
            rel = R.T @ (p - origin)
            rng_ = float(np.linalg.norm(rel))
            b = math.atan2(rel[1], rel[0])
            e = math.asin(rel[2] / max(rng_, 1e-6))
            if rng_ < c['max_range_m'] and abs(b) <= az[-1] and abs(e) <= math.radians(max(c['elevations_deg'])):
                i = int(np.argmin(np.abs(az - b)))
                for j in (i - 1, i, i + 1):
                    if 0 <= j < nb and (ranges[j] < 0 or ranges[j] > rng_):
                        ranges[j] = rng_ + self.rng.normal(0, c['range_noise_m'])
                        inten[j] = 0.9
                        truth[j] = 'false_target'
                        echoes.append((j, ranges[j], 0.9 / (1 + 0.05 * rng_), 'false_target', None))
        return {'t': self.t, 'valid': True, 'bearings': az.tolist(), 'ranges': ranges.round(3).tolist(),
                'intensity': inten.round(3).tolist(), '_truth': truth, '_echoes': echoes}

    def sonar_image(self, scan: dict | None = None):
        """FLS fan image (uint8 HxW) of the latest ping + ground-truth boxes (see sonar_image.py).
        Returns (None, []) when the sonar has no valid ping (dropout)."""
        from .sonar_image import SonarImager
        scan = scan or self.readings.get('sonar')
        if not scan or not scan.get('valid'):
            return None, []
        if self._sonar_imager is None:
            self._sonar_imager = SonarImager(self.cfg, np.asarray(scan['bearings']))
        return self._sonar_imager.render(scan.get('_echoes', []), self.rng)

    # ------------------------------------------------------------------ camera
    def _get_renderer(self, kind='rgb'):
        cam = self.cfg['sensors']['camera']
        attr = f'_renderer_{kind}'
        r = getattr(self, attr, None)
        if r is None:
            r = mujoco.Renderer(self.model, cam['height'], cam['width'])
            if kind == 'seg':
                r.enable_segmentation_rendering()
            elif kind == 'depth':
                r.enable_depth_rendering()
            setattr(self, attr, r)
        return r

    def _underwater_opt(self):
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 1            # show the water surface from below
        opt.geomgroup[4] = 1            # pool structure
        return opt

    def _render(self, fn, *a, **kw):
        if self._render_pool is None:
            self._render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='mujoco-render')
        return self._render_pool.submit(fn, *a, **kw).result()

    def camera_image(self, camera='front', noise=True) -> Optional[np.ndarray]:
        return self._render(self._camera_image, camera, noise)

    def render_view(self, camera='spectator', width=800, height=450) -> np.ndarray:
        return self._render(self._render_view, camera, width, height)

    def segmentation(self, camera='front') -> np.ndarray:
        return self._render(self._segmentation, camera)

    def _camera_image(self, camera='front', noise=True) -> Optional[np.ndarray]:
        """Front camera RGB (H, W, 3) uint8 through water; faults: blackout, turbidity.

        Water model (Beer-Lambert on the rendered depth): I = J*T + B*(1-T), T = exp(-c*d) per
        channel, c = k/visibility * channel_factor (red is absorbed fastest). Turbidity lowers the
        visibility. Everything beyond the far plane (no geometry) is pure backscatter colour B.
        """
        cam = self.cfg['sensors']['camera']
        if camera == 'front' and self.faults.get('camera_blackout', self.t):
            return np.zeros((cam['height'], cam['width'], 3), np.uint8)
        opt = self._underwater_opt()
        r = self._get_renderer('rgb')
        r.update_scene(self.data, camera=camera, scene_option=opt)
        rgb = r.render().astype(np.float32)
        rd = self._get_renderer('depth')
        rd.update_scene(self.data, camera=camera, scene_option=opt)
        depth = rd.render()
        vis = self.visible_range()
        c = cam.get('extinction_k', 1.8) / max(vis, 0.1) * np.array(cam.get('channel_attenuation', [1.3, 1.0, 0.9]), np.float32)
        T = np.exp(-depth[..., None] * c)
        B = np.array(cam['water_rgb'], np.float32) * 255.0
        img = rgb * T + B * (1.0 - T)
        if noise and cam['noise_std'] > 0:
            img = img + self.rng.normal(0, cam['noise_std'], img.shape)
        return np.clip(img, 0, 255).astype(np.uint8)

    def _render_view(self, camera='spectator', width=800, height=450) -> np.ndarray:
        """Third-person / top-down views for media (no water effect, no noise)."""
        key = f'_view_{width}x{height}'
        r = getattr(self, key, None)
        if r is None:
            r = mujoco.Renderer(self.model, height, width)
            setattr(self, key, r)
        opt = mujoco.MjvOption()
        opt.geomgroup[3] = 0            # hide the water surface for views from above the water
        opt.geomgroup[4] = 1            # pool structure
        r.update_scene(self.data, camera=camera, scene_option=opt)
        return r.render()

    def _segmentation(self, camera='front') -> np.ndarray:
        """Per-pixel geom id (-1 = background), same camera as `camera_image`."""
        r = self._get_renderer('seg')
        r.update_scene(self.data, camera=camera, scene_option=self._underwater_opt())
        seg = r.render()
        ids = seg[..., 0].astype(np.int32)
        ids[seg[..., 1] != int(mujoco.mjtObj.mjOBJ_GEOM)] = -1
        return ids

    def geom_class(self, gid: int) -> Optional[str]:
        return self._geom_class.get(int(gid))

    def visible_range(self) -> float:
        """Current camera visibility (m): clear-water visibility, reduced by turbidity faults."""
        f = self.faults.get('turbidity', self.t)
        return f.params.get('visibility_m', 2.0) if f else self.cfg['sensors']['camera']['visibility_m']

    def close(self):
        if self._render_pool is not None:
            self._render_pool.submit(self._close_renderers).result()
            self._render_pool.shutdown(wait=True)
            self._render_pool = None

    def _close_renderers(self):
        for k in list(vars(self)):
            if k.startswith(('_renderer_', '_view_')):
                r = getattr(self, k)
                if r is not None:
                    try:
                        r.close()
                    except Exception:
                        pass
                setattr(self, k, None)
