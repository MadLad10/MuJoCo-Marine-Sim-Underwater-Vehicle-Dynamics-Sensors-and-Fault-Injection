"""Tiny end-to-end dataset export using only the installed simulator package."""
from pathlib import Path
from PIL import Image
import yaml
from auv_marine_sim.autolabel import generate


def test_camera_export_splits_by_scene_and_emits_valid_boxes(tmp_path):
    output = tmp_path / 'dataset'
    stats = generate(str(output), scenes=2, frames=2, seed=11, val_frac=.5)
    assert stats['images'] == 4
    assert stats['splits'] == {'train': 2, 'val': 2}
    scene_ids = {}
    for split in ('train', 'val'):
        images = sorted((output / 'images' / split).glob('*.jpg'))
        assert len(images) == 2
        scene_ids[split] = {p.stem.split('_')[0] for p in images}
        for p in images:
            with Image.open(p) as image:
                assert image.size == (640, 480)
            for line in (output / 'labels' / split / (p.stem + '.txt')).read_text().splitlines():
                label, cx, cy, w, h = map(float, line.split())
                assert int(label) == label and 0 <= label < 4
                assert 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
    assert scene_ids['train'].isdisjoint(scene_ids['val'])
    metadata = yaml.safe_load((output / 'data.yaml').read_text())
    assert Path(metadata['path']) == output and len(metadata['names']) == 4
