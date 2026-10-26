"""Synthetic dataset generator: simulator-rendered frames auto-labelled from segmentation masks.

Domain randomisation per scene: object layout (gate and buoys moved around the pool), water
visibility and colour, camera noise; per frame: vehicle pose (biased towards looking at objects).
Labels (YOLO format) come from MuJoCo's segmentation render, so they are pixel-exact, and an object
is only labelled when it is actually visible through the water: enough pixels AND enough
transmission at its range (Beer-Lambert, same model as the camera), so the detector is never
trained to see through murky water.

  python3 -m auv_marine_sim.autolabel --out data/synthetic --scenes 40 --frames 50 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os

import numpy as np

from .sim import MarineSim
from .world import load_arena

CLASSES = ['gate', 'red_buoy', 'green_buoy', 'yellow_buoy']
MIN_PIXELS = 30
MIN_BOX = 5
MIN_TRANSMISSION = 0.12


def random_arena(base: dict, rng: np.random.Generator) -> dict:
    a = copy.deepcopy(base)
    p = a['pool']
    x0, x1 = p.get('x_min', -5.0) + 2.0, p.get('x_min', -5.0) + p['length'] - 2.0
    yh = p['width'] / 2 - 2.0
    for o in a['objects']:
        if o['class'] == 'gate':
            o['x'] = float(rng.uniform(4.0, 14.0))
            o['y'] = float(rng.uniform(-4.0, 4.0))
            o['z'] = float(rng.uniform(-2.2, -1.0))
            o['yaw'] = float(rng.uniform(-0.6, 0.6))
        else:
            o['x'] = float(rng.uniform(x0 + 4, x1))
            o['y'] = float(rng.uniform(-yh, yh))
            o['z'] = float(rng.uniform(-3.0, -0.8))
    return a


def sample_pose(arena: dict, rng: np.random.Generator):
    """Camera pose: 70% aimed (with jitter) at a random object 1-12 m away, 30% random."""
    p = arena['pool']
    x0 = p.get('x_min', -5.0)
    if rng.random() < 0.7:
        o = arena['objects'][rng.integers(len(arena['objects']))]
        d = rng.uniform(1.0, 12.0)
        a = rng.uniform(-math.pi, math.pi)
        x, y = o['x'] - d * math.cos(a), o['y'] - d * math.sin(a)
        z = float(np.clip(o['z'] + rng.normal(0, 0.5), -4.0, -0.2))
        yaw = math.atan2(o['y'] - y, o['x'] - x) + rng.normal(0, 0.35)
    else:
        x, y = rng.uniform(x0 + 1, x0 + p['length'] - 1), rng.uniform(-p['width'] / 2 + 1, p['width'] / 2 - 1)
        z, yaw = rng.uniform(-4.0, -0.2), rng.uniform(-math.pi, math.pi)
    x = float(np.clip(x, x0 + 0.6, x0 + p['length'] - 0.6))
    y = float(np.clip(y, -p['width'] / 2 + 0.6, p['width'] / 2 - 0.6))
    return x, y, z, yaw


def labels_for(sim: MarineSim, seg: np.ndarray, cam_pos: np.ndarray, vis: float, k: float) -> list:
    H, W = seg.shape
    out = []
    by_obj: dict[str, list] = {}
    for g in np.unique(seg):
        if g < 0:
            continue
        cls = sim.geom_class(int(g))
        if cls not in CLASSES:
            continue
        obj = (sim.model.geom(int(g)).name or '').split('__')[0]
        by_obj.setdefault(obj, [cls, []])[1].append(int(g))
    centres = {o['id']: np.array([o['x'], o['y'], o['z']]) for o in sim.arena['objects']}
    for obj, (cls, gids) in by_obj.items():
        mask = np.isin(seg, gids)
        n = int(mask.sum())
        if n < MIN_PIXELS:
            continue
        rng_m = float(np.linalg.norm(centres[obj] - cam_pos))
        if math.exp(-k * rng_m / vis) < MIN_TRANSMISSION:
            continue
        ys, xs = np.nonzero(mask)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if x1 - x0 < MIN_BOX or y1 - y0 < MIN_BOX:
            continue
        out.append((CLASSES.index(cls), (x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return out


def generate(out_dir: str, scenes: int, frames: int, seed: int = 0, val_frac: float = 0.2,
             base_arena: str = 'default', vis_range=(4.0, 12.0), fixed_vis: float | None = None) -> dict:
    import mujoco
    from PIL import Image
    rng = np.random.default_rng(seed)
    base = load_arena(base_arena)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
    stats = {'images': 0, 'boxes': {c: 0 for c in CLASSES}, 'empty': 0, 'splits': {'train': 0, 'val': 0}}
    for sc in range(scenes):
        arena = random_arena(base, rng)
        vis = fixed_vis if fixed_vis is not None else float(rng.uniform(*vis_range))
        water = (np.array([0.05, 0.22, 0.30]) * rng.uniform(0.7, 1.3, 3)).clip(0, 1).tolist()
        sim = MarineSim(arena, seed=int(rng.integers(1 << 30)),
                        overrides={'sensors': {'camera': {'visibility_m': vis, 'water_rgb': water,
                                                          'noise_std': float(rng.uniform(2, 8))}}})
        split = 'val' if sc >= scenes * (1 - val_frac) else 'train'   # split by scene: no layout leakage
        k = sim.cfg['sensors']['camera']['extinction_k']
        cam_off = np.array(sim.cfg['sensors']['camera']['offset'])
        for fr in range(frames):
            x, y, z, yaw = sample_pose(arena, rng)
            sim.data.qpos[:3] = [x, y, z]
            sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(sim.model, sim.data)
            img = sim.camera_image()
            seg = sim.segmentation()
            cam_pos = sim.data.xpos[sim.body] + sim.R() @ cam_off
            labs = labels_for(sim, seg, cam_pos, vis, k)
            name = f's{sc:03d}_f{fr:03d}'
            Image.fromarray(img).save(os.path.join(out_dir, 'images', split, name + '.jpg'), quality=92)
            with open(os.path.join(out_dir, 'labels', split, name + '.txt'), 'w') as f:
                for c, cx, cy, w, h in labs:
                    f.write(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n')
                    stats['boxes'][CLASSES[c]] += 1
            stats['images'] += 1
            stats['splits'][split] += 1
            stats['empty'] += not labs
        sim.close()
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n'
                f'names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(CLASSES)))
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='data/synthetic')
    ap.add_argument('--scenes', type=int, default=40)
    ap.add_argument('--frames', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--vis-min', type=float, default=4.0)
    ap.add_argument('--vis-max', type=float, default=12.0)
    ap.add_argument('--fixed-vis', type=float, default=None, help='every scene at this visibility (test sets)')
    ap.add_argument('--val-frac', type=float, default=0.2)
    ap.add_argument('--sonar', action='store_true', help='forward-looking sonar images instead of camera')
    a = ap.parse_args()
    if a.sonar:
        print(json.dumps(generate_sonar(a.out, a.scenes, a.frames, a.seed, a.val_frac), indent=1))
        return
    print(json.dumps(generate(a.out, a.scenes, a.frames, a.seed, a.val_frac, vis_range=(a.vis_min, a.vis_max),
                              fixed_vis=a.fixed_vis), indent=1))



# ============================================================== sonar (forward-looking imaging sonar)
def generate_sonar(out_dir: str, scenes: int, frames: int, seed: int = 0, val_frac: float = 0.2,
                   base_arena: str = 'default') -> dict:
    """FLS fan images auto-labelled from the echo ground truth (classes: buoy, gate_post).
    Domain randomisation: object layout, vehicle pose, speckle strength, noise floor and gain."""
    import mujoco
    from PIL import Image
    from .sonar_image import SONAR_CLASSES
    rng = np.random.default_rng(seed)
    base = load_arena(base_arena)
    for split in ('train', 'val'):
        os.makedirs(os.path.join(out_dir, 'images', split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'labels', split), exist_ok=True)
    stats = {'images': 0, 'boxes': {c: 0 for c in SONAR_CLASSES}, 'empty': 0}
    for sc in range(scenes):
        arena = random_arena(base, rng)
        im = {'speckle': float(rng.uniform(0.7, 1.3)), 'noise_floor': float(rng.uniform(0.015, 0.06)),
              'gain': float(rng.uniform(1.2, 2.2))}
        sim = MarineSim(arena, seed=int(rng.integers(1 << 30)), overrides={'sensors': {'sonar': {'image': im}}})
        split = 'val' if sc >= scenes * (1 - val_frac) else 'train'
        for fr in range(frames):
            x, y, z, yaw = sample_pose(arena, rng)
            sim.data.qpos[:3] = [x, y, z]
            sim.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            mujoco.mj_forward(sim.model, sim.data)
            scan = sim._sonar(sim.truth())
            img, boxes = sim.sonar_image(scan)
            H, W = img.shape
            name = f's{sc:03d}_f{fr:03d}'
            Image.fromarray(img).save(os.path.join(out_dir, 'images', split, name + '.png'))
            with open(os.path.join(out_dir, 'labels', split, name + '.txt'), 'w') as f:
                for b in boxes:
                    x0, y0, x1, y1 = b['box']
                    f.write(f'{SONAR_CLASSES.index(b["class"])} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} '
                            f'{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n')
                    stats['boxes'][b['class']] += 1
            stats['images'] += 1
            stats['empty'] += not boxes
        sim.close()
    with open(os.path.join(out_dir, 'data.yaml'), 'w') as f:
        f.write(f'path: {os.path.abspath(out_dir)}\ntrain: images/train\nval: images/val\n'
                f'names:\n' + ''.join(f'  {i}: {c}\n' for i, c in enumerate(SONAR_CLASSES)))
    with open(os.path.join(out_dir, 'stats.json'), 'w') as f:
        json.dump(stats, f, indent=1)
    return stats


if __name__ == '__main__':
    main()
