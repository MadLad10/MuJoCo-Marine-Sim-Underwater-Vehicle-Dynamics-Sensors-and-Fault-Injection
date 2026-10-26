"""Standalone simulator commands; no ROS, mission agents or external services."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
from importlib.metadata import version

from . import __version__
from .faults import FAULT_TYPES, FaultInjector
from .world import load_arena, load_config


def json_value(value, public=False):
    """Sensor logs omit private annotation fields; truth has its own record."""
    import numpy as np
    if isinstance(value, dict):
        return {str(k): json_value(v, public) for k, v in value.items()
                if not public or not str(k).startswith('_')}
    if isinstance(value, (list, tuple)):
        return [json_value(v, public) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def positive(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError('must be a finite positive number')
    return result


def seed_value(value):
    seed = int(value)
    if seed < 0:
        raise argparse.ArgumentTypeError('seed must be nonnegative')
    return seed


def new_output(path):
    output = Path(path).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output is not an empty directory: {output}')
    output.mkdir(parents=True, exist_ok=True)
    return output


def save_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + '\n')


def run(args):
    # A physics/sensor rollout requires no graphics context. Set before importing
    # MuJoCo; an explicit caller-selected backend is always respected.
    os.environ.setdefault('MUJOCO_GL', 'disable')
    from .sim import MarineSim
    from .world import load_yaml
    faults = []
    if args.faults:
        payload = load_yaml(args.faults)
        faults = payload.get('faults', []) if isinstance(payload, dict) else payload
        if not isinstance(faults, list):
            raise ValueError('Fault schedule must be a list or a mapping with faults: [...]')
    injector = FaultInjector(faults)
    output = new_output(args.output)
    sim = MarineSim(args.arena, config=args.config, faults=injector, seed=args.seed)
    try:
        save_json(output / 'sim_config.json', sim.cfg)
        save_json(output / 'arena.json', sim.arena)
        with (output / 'sensors.jsonl').open('w') as stream:
            for _ in range(math.ceil(args.seconds / sim.control_dt)):
                sim.step()
                tr = sim.truth()
                record = {'time_s': sim.t,
                          'sensors': json_value(sim.readings, public=True),
                          'truth': {'position_m': tr.pos, 'rpy_rad': tr.rpy,
                                    'velocity_body_mps': tr.vel_body}}
                stream.write(json.dumps(json_value(record), allow_nan=False) + '\n')
        summary = {'package_version': __version__, 'seed': args.seed,
                   'mujoco_version': version('mujoco'), 'requested_seconds': args.seconds,
                   'simulated_seconds': sim.t, 'control_dt_s': sim.control_dt,
                   'mode': 'passive physics and sensor rollout; zero motor commands',
                   'fault_schedule': injector.ground_truth(), 'fault_events': injector.log,
                   'referee': sim.referee.report()}
        save_json(output / 'summary.json', summary)
    finally:
        sim.close()
    print(json.dumps({'output': str(output), 'simulated_seconds': summary['simulated_seconds'],
                      'fault_events': len(summary['fault_events'])}))


def _panel(image, size):
    from PIL import Image, ImageOps
    return ImageOps.pad(Image.fromarray(image).convert('RGB'), size,
                        method=Image.Resampling.BILINEAR, color=(8, 16, 24))


def demo(args):
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import numpy as np
    from PIL import Image, ImageDraw
    from .sim import MarineSim
    output = new_output(args.output)
    arena = copy.deepcopy(load_arena(args.arena))
    # This is an explicitly selected demo initial condition, not a mission or
    # successful controller. There are no motor commands during the recording.
    arena['start'].update(x=3.0, y=0.0, z=-1.5, yaw=0.0)
    cases = [
        ('nominal', []),
        ('turbidity', [{'type': 'turbidity', 't_start': .5,
                        'params': {'visibility_m': 2.0}}]),
        ('camera_blackout', [{'type': 'camera_blackout', 't_start': .5}]),
        ('dvl_dropout', [{'type': 'dvl_dropout', 't_start': .5}]),
    ]
    manifest = []
    for name, faults in cases:
        directory = output / name
        directory.mkdir()
        sim = MarineSim(arena, config=args.config, seed=args.seed, faults=faults)
        if args.fps > 1 / sim.control_dt:
            sim.close()
            raise ValueError('Recording fps must not exceed the simulator control rate')
        frames = []
        timestamps = []
        next_frame = 0.0
        try:
            ticks = math.ceil(args.seconds / sim.control_dt)
            for tick in range(ticks + 1):
                if tick:
                    sim.step()
                if sim.t + 1e-9 < next_frame:
                    continue
                front = sim.camera_image()
                sonar, _ = sim.sonar_image()
                if sonar is None:
                    sonar = np.zeros((256, 448), np.uint8)
                spectator = sim.render_view(width=640, height=360)
                canvas = Image.new('RGB', (960, 276), (8, 16, 24))
                canvas.paste(_panel(front, (320, 240)), (0, 30))
                canvas.paste(_panel(sonar, (320, 240)), (320, 30))
                canvas.paste(_panel(spectator, (320, 240)), (640, 30))
                draw = ImageDraw.Draw(canvas)
                labels = ['Noisy front camera', 'Synthetic imaging sonar',
                          'Spectator view (ground truth)']
                for i, text in enumerate(labels):
                    draw.text((8 + i * 320, 7), text, fill='white')
                draw.text((8, 260), f'{name}: t={sim.t:.2f}s; DVL valid={sim.readings["dvl"]["valid"]}', fill='white')
                frames.append(canvas)
                timestamps.append(float(sim.t))
                # Raw sensor pixels are stored without labels or resizing.
                if len(frames) == 1:
                    Image.fromarray(front).save(directory / 'front_initial.png')
                next_frame += 1 / args.fps
            Image.fromarray(front).save(directory / 'front_final.png')
            Image.fromarray(sonar).save(directory / 'sonar_final.png')
            frames[-1].save(directory / 'preview.png')
            frames[0].save(directory / 'demo.gif', save_all=True,
                           append_images=frames[1:], duration=round(1000 / args.fps), loop=0)
            entry = {'scenario': name, 'seed': args.seed, 'frame_times_s': timestamps,
                     'frames': len(frames), 'gif_fps': args.fps,
                     'fault_schedule': sim.faults.ground_truth(), 'fault_events': sim.faults.log,
                     'final_dvl_valid': bool(sim.readings['dvl']['valid']),
                     'front_final_max': int(front.max()),
                     'final_truth_position_m': sim.truth().pos.tolist(),
                     'referee': sim.referee.report()}
            save_json(directory / 'summary.json', entry)
            manifest.append(entry)
        finally:
            sim.close()
    save_json(output / 'manifest.json', {'mode': 'passive buoyancy and fault visualization',
              'independent_mission_trials': 0, 'mujoco_version': version('mujoco'),
              'seed': args.seed, 'scenarios': manifest})
    print(json.dumps({'output': str(output), 'scenarios': len(manifest),
                      'frames': sum(row['frames'] for row in manifest)}))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', action='version', version=__version__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('faults', help='List supported fault types')
    commands.add_parser('config', help='Print the packaged default configuration')
    for command, function in [('run', run), ('demo', demo)]:
        sub = commands.add_parser(command, help=function.__doc__)
        sub.set_defaults(function=function)
        sub.add_argument('--arena', default='default', help='Packaged arena name or YAML path')
        sub.add_argument('--config', help='Custom simulator YAML path')
        sub.add_argument('--seed', type=seed_value, default=0)
        sub.add_argument('--seconds', type=positive, default=2.0)
        sub.add_argument('--output', default='runs/' + command)
        if command == 'run':
            sub.add_argument('--faults', help='YAML fault schedule')
        else:
            sub.add_argument('--fps', type=positive, default=10.0)
    args = parser.parse_args(argv)
    if args.command == 'faults':
        print(json.dumps({k: {'component': v[0], 'description': v[1]}
                          for k, v in FAULT_TYPES.items()}, indent=2))
    elif args.command == 'config':
        print(json.dumps(load_config(), indent=2))
    else:
        try:
            args.function(args)
        except (ValueError, FileNotFoundError) as error:
            parser.error(str(error))
    return 0
