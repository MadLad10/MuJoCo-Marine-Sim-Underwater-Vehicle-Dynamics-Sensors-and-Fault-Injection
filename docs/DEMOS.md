# Simulator demos

Each recording starts the same configured vehicle at x=3 m, y=0 m, z=−1.5 m, heading along the pool. The vehicle receives zero motor commands and rises under buoyancy. Every panel comes from the simulator. The front-camera and sonar panels retain their configured noise; the spectator panel is a noiseless explanatory view.

The default recording covers simulated t=0–2 seconds inclusive at 10 frames/s (21 frames). Faults begin at t=0.5 seconds. These are passive sensor/physics demonstrations, not autonomous mission trials or paired statistical comparisons.

| Scenario | Recording | Final three-panel view | Raw camera | Raw sonar |
|---|---|---|---|---|
| Nominal | [GIF](media/nominal.gif) | [PNG](media/nominal.png) | [PNG](media/nominal_front.png) | [PNG](media/nominal_sonar.png) |
| Turbidity, 2 m visibility | [GIF](media/turbidity.gif) | [PNG](media/turbidity.png) | [PNG](media/turbidity_front.png) | [PNG](media/turbidity_sonar.png) |
| Camera blackout | [GIF](media/camera_blackout.gif) | [PNG](media/camera_blackout.png) | [PNG](media/camera_blackout_front.png) | [PNG](media/camera_blackout_sonar.png) |
| DVL dropout | [GIF](media/dvl_dropout.gif) | [PNG](media/dvl_dropout.png) | [PNG](media/dvl_dropout_front.png) | [PNG](media/dvl_dropout_sonar.png) |

Recreate the recordings after installation:

```bash
MUJOCO_GL=egl marine-sim demo --seconds 2 --fps 10 --seed 0 --output runs/demo
```

The output includes per-scenario `summary.json` files and a top-level `manifest.json` with exact frame timestamps, fault schedules, seeds and referee reports. [Checksums and decoding checks for the included media](demo_verification.json).
