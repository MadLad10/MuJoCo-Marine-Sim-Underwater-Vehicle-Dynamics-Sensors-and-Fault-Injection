"""MuJoCo marine sim: physics sanity, every sensor, every fault type, referee."""
import math

import mujoco
import numpy as np
import pytest

from auv_marine_sim.faults import FAULT_TYPES, Fault, FaultInjector
from auv_marine_sim.referee import Referee
from auv_marine_sim.sim import MarineSim
from auv_marine_sim.world import build_mjcf, load_arena, load_config


def place(sim, x, y, z, yaw=0.0):
    sim.data.qpos[:3] = [x, y, z]
    sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)


@pytest.fixture
def sim():
    s = MarineSim('default', seed=3)
    yield s
    s.close()


def test_mjcf_has_world_and_vehicle():
    xml = build_mjcf(load_arena('default'), load_config())
    m = mujoco.MjModel.from_xml_string(xml)
    for name in ('gate__post_left', 'red_buoy__ball', 'pool_floor', 'auv_hull'):
        assert m.geom(name) is not None
    assert m.nu == 6


def test_idle_vehicle_floats_level_at_surface(sim):
    place(sim, 0, 0, -2.0)
    sim.run_for(40.0)
    tr = sim.truth()
    assert tr.depth < 0.2                         # slightly positive buoyancy
    assert abs(tr.rpy[0]) < 0.05 and abs(tr.rpy[1]) < 0.05


def test_surge_thrust_moves_forward(sim):
    place(sim, 0, 0, -1.5)
    sim.set_thrust({'surge_left': 20, 'surge_right': 20, 'heave_front': -1.5, 'heave_back': -1.5})
    sim.run_for(10.0)
    tr = sim.truth()
    assert 0.6 < tr.vel_body[0] < 1.3
    assert tr.pos[0] > 5.0


def test_all_sensors_publish(sim):
    sim.run_for(1.0)
    r = sim.readings
    for k in ('imu', 'compass', 'pressure', 'dvl', 'sonar', 'battery', 'thrusters', 'leak'):
        assert k in r, k
    assert r['dvl']['valid'] and len(r['dvl']['vel']) == 3
    assert r['imu']['accel'][2] == pytest.approx(9.81, abs=0.3)   # specific force at rest
    assert 0 < r['battery']['percent'] <= 100
    assert r['leak']['leak'] is False
    assert len(r['sonar']['ranges']) == sim.cfg['sensors']['sonar']['beams']


def test_sonar_sees_gate_and_buoys():
    s = MarineSim('default', seed=1)
    place(s, 3.0, 0.0, -1.5)
    s.step()
    s._next.clear()
    s.step()
    classes = {c for c in s.readings['sonar']['_truth'] if c}
    assert 'gate' in classes
    s.close()


def test_camera_and_segmentation(sim):
    place(sim, 3.0, 0.0, -1.5)
    img = sim.camera_image()
    assert img.shape == (480, 640, 3) and img.dtype == np.uint8
    seg = sim.segmentation()
    classes = {sim.geom_class(g) for g in np.unique(seg) if g >= 0}
    assert 'gate' in classes


def run_fault(ftype, params=None, x=3.0, z=-1.5, seconds=3.0):
    s = MarineSim('default', faults=[{'type': ftype, 't_start': 0.5, 'params': params or {}}], seed=2)
    place(s, x, 0.0, z)
    s.run_for(seconds)
    return s


def test_fault_types_are_validated():
    with pytest.raises(ValueError):
        Fault('not_a_fault')
    assert len(FAULT_TYPES) == 11


def test_dvl_dropout():
    s = run_fault('dvl_dropout')
    assert s.readings['dvl']['valid'] is False
    assert any(e['event'] == 'onset' and e['type'] == 'dvl_dropout' for e in s.faults.log)
    s.close()


def test_pressure_stuck():
    s = run_fault('pressure_stuck')
    d0 = s.readings['pressure']['depth']
    s.run_for(1.0)
    assert s.readings['pressure']['depth'] == d0
    s.close()


def test_camera_blackout_and_turbidity():
    s = run_fault('camera_blackout')
    assert s.camera_image().max() == 0
    s.close()
    clear = MarineSim('default', seed=2)
    place(clear, 3.0, 0.0, -1.5)
    water = np.array(clear.cfg['sensors']['camera']['water_rgb']) * 255
    dev_clear = np.abs(clear.camera_image().astype(float) - water).mean()
    clear.close()
    s = run_fault('turbidity', {'visibility_m': 2.0})
    place(s, 3.0, 0.0, -1.5)
    dev_turbid = np.abs(s.camera_image().astype(float) - water).mean()
    assert s.visible_range() == 2.0
    assert dev_turbid < 0.6 * dev_clear
    s.close()


def test_thruster_dead_produces_no_thrust():
    s = run_fault('thruster_dead', {'name': 'surge_left'})
    s.set_thrust({'surge_left': 20, 'surge_right': 20})
    s.run_for(1.0)
    th = s.readings['thrusters']
    i = th['names'].index('surge_left')
    assert th['cmd_n'][i] == 20.0 and abs(th['thrust_n'][i]) < 1e-6 and th['current_a'][i] < 0.2
    assert th['thrust_n'][th['names'].index('surge_right')] > 19.0
    s.close()


def test_leak_battery_drop_compass_imu_bias():
    s = run_fault('leak')
    assert s.readings['leak']['leak'] is True
    s.close()
    s = run_fault('battery_drop', {'drop_pct': 40})
    assert s.readings['battery']['percent'] < 61.0
    s.close()
    s = run_fault('compass_disturbance', {'offset_deg': 30})
    assert math.degrees(s.readings['compass']['yaw']) == pytest.approx(30, abs=6)
    s.close()
    s = run_fault('imu_bias', {'gyro_bias_z': 0.05})
    gz = []
    for _ in range(20):
        s.step()
        gz.append(s.readings['imu']['gyro'][2])
    assert np.mean(gz) == pytest.approx(0.05, abs=0.01)
    s.close()


def test_sonar_false_target_and_dropout():
    s = run_fault('sonar_false_target', {'x': 7.0, 'y': 0.5, 'z': -1.5}, x=0.0)
    assert 'false_target' in s.readings['sonar']['_truth']
    s.close()
    s = run_fault('sonar_dropout')
    assert s.readings['sonar']['valid'] is False
    s.close()


def test_fault_injector_window():
    fi = FaultInjector([{'type': 'leak', 't_start': 1.0, 'duration': 2.0}])
    for t in (0.5, 1.5, 3.5):
        fi.update(t)
    assert fi.get('leak', 1.5) and not fi.get('leak', 3.5)
    assert [e['event'] for e in fi.log] == ['onset', 'clear']


def test_referee_gate_buoy_surface_collision():
    ref = Referee(load_arena('default'))
    t = 0.0
    for x in np.linspace(6.0, 10.0, 40):           # straight through the gate at its depth
        t += 0.1
        ref.update(t, (x, 0.0, -1.5))
    for p in [(13.6, -2.0, -1.5), (14.0, -2.0, -1.5)]:
        t += 0.1
        ref.update(t, p, contacts=['gate__post_left'])
    for _ in range(30):
        t += 0.1
        ref.update(t, (14.0, -2.0, -0.05))
    r = ref.report()
    assert r['gate_passed'] and r['buoys_touched'] == ['red_buoy'] and r['surfaced_at_end']
    assert r['collisions'] == 1
    ref.record_estimate(t, (14.3, -2.0, -0.05), (14.0, -2.0, -0.05))
    assert ref.localisation()['rmse_xy_m'] == pytest.approx(0.3, abs=1e-6)


def test_sonar_image_and_autolabels():
    s = MarineSim('default', seed=0)
    try:
        place(s, 3.0, 0.0, -1.5)
        scan = s._sonar(s.truth())
        img, boxes = s.sonar_image(scan)
        assert img.shape == (256, 448) and img.dtype == np.uint8
        objs = {b['obj'] for b in boxes}
        assert {'gate#L', 'gate#R', 'red_buoy', 'green_buoy'} <= objs
        assert all(b['class'] in ('buoy', 'gate_post') for b in boxes)
        # a labelled echo is brighter than the speckle background around it
        b = next(b for b in boxes if b['obj'] == 'red_buoy')
        x0, y0, x1, y1 = (int(round(v)) for v in b['box'])
        assert img[y0:y1 + 1, x0:x1 + 1].max() > np.percentile(img, 95)
    finally:
        s.close()
    s = run_fault('sonar_dropout')
    assert s.sonar_image() == (None, [])
    s.close()
