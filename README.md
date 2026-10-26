# MuJoCo Marine Sim

A standalone underwater vehicle simulator with configurable buoyancy and drag, noisy sensors, scheduled failures, camera/sonar annotations and a ground-truth referee.

The simulator was extracted from the Agentic AUV engineering project. It can run as a Python package without ROS, an LLM service, navigation models or detector weights.

## Sensor and fault demos

![Nominal camera, sonar and spectator views](docs/media/nominal.gif)

![Camera blackout with continuing sonar](docs/media/camera_blackout.gif)

These recordings show passive buoyancy and sensor faults. The vehicle receives zero motor commands. The first two panels are the noisy front camera and synthetic imaging sonar; the third is a noiseless spectator view for visualization. [All scenarios and recording details](docs/DEMOS.md).

## What is included

- A configurable rigid-body AUV with six lagged thrusters, buoyancy, body-frame linear/quadratic drag and water current.
- Front camera, imaging sonar, DVL, IMU, compass, pressure, battery, thruster telemetry and leak indication.
- Eleven scheduled fault types, with onset/clear events and optional durations.
- YAML arenas and sensor/vehicle configuration, loaded from the installed package or an explicit file path.
- Camera segmentation and YOLO-format automatic labels; sonar echo-based boxes.
- A referee for gate crossings, buoy touches, surfacing, collisions, boundaries and optional localization error.
- A CLI, a small Python API example, regression tests and a wheel build/install check.

## Installation

Python 3.10 or newer. A rendering backend is needed for camera images and the visual demos. The package was exercised on Linux; see [rendering](#rendering) for backend choices.

Using uv:

```bash
uv venv --python python3 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

Or, with Python's venv/pip where available:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

[Exact versions used for local verification](docs/verification.json) are recorded separately from the supported dependency ranges in `pyproject.toml`.

## Quick start

List the available faults and inspect the default configuration:

```bash
marine-sim faults
marine-sim config
```

Run a short physics/sensor rollout:

```bash
marine-sim run --seconds 3 --seed 0 --output runs/nominal
marine-sim run --seconds 3 --seed 0 \
  --faults examples/fault_schedule.yaml --output runs/faults
```

Each run saves `sensors.jsonl`, `sim_config.json`, `arena.json` and `summary.json`. Sensor records exclude private annotation fields; the `truth` field and referee summary are explicitly ground truth. These commands let the buoyant vehicle drift with zero motor commands. Choose an empty output directory for each run.

Record the four visual scenarios:

```bash
MUJOCO_GL=egl marine-sim demo --seconds 2 --fps 10 --output runs/demo
```

This saves nominal, turbidity, camera-blackout and DVL-dropout GIFs, raw initial/final camera PNGs, a final sonar PNG and per-scenario JSON summaries. The seed, fault schedule and actual simulation timestamps are saved in `manifest.json`.

## Python API

```python
import os
os.environ.setdefault("MUJOCO_GL", "disable")  # physics/sensors, no camera rendering

from auv_marine_sim import MarineSim

sim = MarineSim("default", seed=7, faults=[
    {"type": "dvl_dropout", "t_start": 0.5},
])
try:
    sim.run_for(1.0)
    print(sim.readings["dvl"]["valid"])
    print(sim.referee.report())
finally:
    sim.close()
```

The library exposes the underlying MuJoCo model/data for simulation experiments. In the integrated AUV application, only the deterministic onboard controller issues motor commands. No real hardware driver or command publisher is included here.

## Automatic camera labels

The existing generator randomizes scene layout, vehicle pose and water visibility. Train/validation splits are assigned by scene.

```bash
MUJOCO_GL=egl python -m auv_marine_sim.autolabel \
  --out runs/tiny_dataset --scenes 5 --frames 4 --seed 0
```

The generated images, labels, `data.yaml` and `stats.json` remain local under `runs/`. Visibility/transmission thresholds filter the labels; rendered segmentation is an annotation source, not a deployable detector.

## Rendering

- `MUJOCO_GL=egl`: offscreen rendering with a working EGL/OpenGL driver; used for the local visual checks.
- `MUJOCO_GL=osmesa` and `PYOPENGL_PLATFORM=osmesa`: software rendering when the operating system provides OSMesa; configured for CI.
- Physics-only CLI runs default to `MUJOCO_GL=disable` when no backend is already selected.

Set the backend before importing MuJoCo. This repository does not install system graphics drivers. The CI workflow provisions its own container graphics libraries.

## Tests and wheel verification

```bash
MUJOCO_GL=egl python -m pytest -q
uv build
```

Tests cover dynamics, sensor payloads, all fault types, referee events, sonar annotations, deterministic seeded rollouts, CLI behavior and packaged configuration lookup. The installation verification also builds a wheel, installs it in a separate environment and runs it outside the source directory. [Local verification receipt](docs/verification.json).

## Repository layout

```text
src/auv_marine_sim/     simulator, sensor/fault models and packaged YAML
examples/              minimal API example and fault schedule
tests/                 simulator and standalone CLI tests
docs/                  verification, extraction provenance and real demo media
.github/workflows/     package/test workflow
```

## Model limits

Hydrodynamics use a simplified buoyancy and damping model, without added-mass coupling or CFD. Sonar uses geometric ray/segment echoes and a synthetic intensity/speckle model, without full acoustic multipath. Camera appearance uses range-dependent attenuation and water-color blending rather than a calibrated underwater optical renderer. Sensor parameters have not been fitted to physical hardware.

The passive demos establish simulator behavior and fault injection; they do not evaluate autonomous navigation, mission success or transfer to a real vehicle.

## License

MIT. External dependencies retain their respective licenses.
