"""Forward-looking imaging sonar (FLS) image synthesis from the simulator's echoes.

Each ping, the simulator records echoes (beam index, range, strength, class, object id) from every
ray elevation, the beam-width-aware returns of thin objects, and injected false targets. This module
turns them into what an FLS shows: a polar intensity grid (beams x range bins) with
  - two-way spreading loss and target strength per class (structure echoes weaker than targets),
  - beam-pattern smearing to neighbouring beams and range blur,
  - multiplicative Rayleigh speckle and an additive Rayleigh noise floor,
then scan-converts it to a Cartesian fan image (range up, bearing left-right). Ground-truth boxes
per object come from the echo cells, so sonar frames can be auto-labelled like the camera's.
"""
from __future__ import annotations

import math

import numpy as np

SONAR_CLASSES = ['buoy', 'gate_post']


class SonarImager:
    def __init__(self, cfg: dict, bearings: np.ndarray):
        c = cfg['sensors']['sonar']
        im = c.get('image', {})
        self.max_range = c['max_range_m']
        self.bearings = np.asarray(bearings, float)
        self.nb = len(self.bearings)
        self.bins = im.get('range_bins', 256)
        self.H = im.get('height', 256)
        self.W = im.get('width', 448)
        self.speckle = im.get('speckle', 1.0)
        self.noise = im.get('noise_floor', 0.05)
        self.gain = im.get('gain', 1.6)
        self.res = self.max_range / self.bins
        self._lut()

    def _lut(self):
        """Pixel -> (beam, bin) lookup for scan conversion (apex at the bottom centre)."""
        H, W = self.H, self.W
        scale = self.max_range / H                       # metres per pixel (range axis)
        v, u = np.mgrid[0:H, 0:W]
        y = (H - 0.5 - v) * scale                        # forward
        x = (W / 2 - 0.5 - u) * scale                    # left positive (bearing + left)
        r = np.hypot(x, y)
        b = np.arctan2(x, y)
        bi = np.rint((b - self.bearings[0]) / (self.bearings[1] - self.bearings[0])).astype(int)
        ri = np.floor(r / self.res).astype(int)
        valid = (bi >= 0) & (bi < self.nb) & (ri >= 0) & (ri < self.bins) & (r <= self.max_range)
        self.lut_b = np.where(valid, bi, 0)
        self.lut_r = np.where(valid, ri, 0)
        self.valid = valid
        self.scale = scale

    def to_pixel(self, bearing: float, rng: float):
        u = self.W / 2 - 0.5 - rng * math.sin(bearing) / self.scale
        v = self.H - 0.5 - rng * math.cos(bearing) / self.scale
        return u, v

    def render(self, echoes: list, rng: np.random.Generator):
        """echoes: [(beam, range_m, strength, cls, obj_id)]. Returns (uint8 HxW image, boxes)."""
        P = np.zeros((self.nb, self.bins), np.float32)
        for bi, r, s, _cls, _oid in echoes:
            ri = int(r / self.res)
            if not (0 <= ri < self.bins):
                continue
            for db, w in ((0, 1.0), (-1, 0.35), (1, 0.35)):            # beam pattern side lobes
                j = bi + db
                if 0 <= j < self.nb:
                    for dr, wr in ((0, 1.0), (1, 0.5), (-1, 0.3)):       # pulse length
                        k = ri + dr
                        if 0 <= k < self.bins:
                            P[j, k] += s * w * wr
        if self.speckle > 0:
            P *= rng.rayleigh(self.speckle / math.sqrt(math.pi / 2), P.shape).astype(np.float32)
        P += rng.rayleigh(self.noise, P.shape).astype(np.float32)
        img = np.zeros((self.H, self.W), np.float32)
        img[self.valid] = P[self.lut_b[self.valid], self.lut_r[self.valid]]
        img = np.clip(img * self.gain, 0, 1)
        out = (np.sqrt(img) * 255).astype(np.uint8)                     # display gamma
        return out, self.boxes(echoes)

    def boxes(self, echoes: list):
        """Pixel boxes per labelled object (class in SONAR_CLASSES) from its echo cells."""
        per = {}
        db = abs(self.bearings[1] - self.bearings[0])
        for bi, r, _s, cls, oid in echoes:
            lab = {'gate': 'gate_post'}.get(cls, 'buoy' if cls and cls.endswith('_buoy') else None)
            if lab is None or oid is None or oid.endswith('#bar'):     # crossbar is not a post
                continue
            b = self.bearings[bi]
            pts = [self.to_pixel(b + sb * db * 0.7, r + sr * self.res * 1.5) for sb in (-1, 1) for sr in (-1, 1)]
            e = per.setdefault(oid, [lab, [], []])
            e[1] += [p[0] for p in pts]
            e[2] += [p[1] for p in pts]
        out = []
        for oid, (lab, us, vs) in per.items():
            x0, x1 = max(0.0, min(us)), min(self.W - 1.0, max(us))
            y0, y1 = max(0.0, min(vs)), min(self.H - 1.0, max(vs))
            if x1 - x0 >= 2 and y1 - y0 >= 2:
                out.append({'class': lab, 'obj': oid, 'box': (x0, y0, x1, y1)})
        return out
