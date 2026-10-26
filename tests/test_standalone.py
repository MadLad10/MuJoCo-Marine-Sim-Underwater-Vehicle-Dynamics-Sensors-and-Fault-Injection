"""Installed-resource and CLI behavior outside the original AUV workspace."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
from auv_marine_sim.world import default_config_path, load_arena, load_config


def cli(*args, cwd=None):
    env = os.environ.copy()
    env.pop('PYTHONPATH', None)
    return subprocess.run([sys.executable, '-m', 'auv_marine_sim', *map(str, args)],
                          cwd=cwd, env=env, capture_output=True, text=True)


def test_bundled_config_precedes_any_ros_workspace():
    path = Path(default_config_path()).resolve()
    assert path.name == 'sim.yaml'
    assert path.parent.parent.name == 'auv_marine_sim'
    assert path.parent.name == 'config'
    assert load_arena('default')['world_name'] == 'pool'
    assert len(load_config()['vehicle']['thrusters']) == 6


def test_cli_help_and_fault_catalog(tmp_path):
    result = cli('--help', cwd=tmp_path)
    assert result.returncode == 0 and 'demo' in result.stdout
    result = cli('faults', cwd=tmp_path)
    assert result.returncode == 0
    faults = json.loads(result.stdout)
    assert len(faults) == 11 and faults['dvl_dropout']['component'] == 'dvl'


def test_cli_fault_rollout_and_public_sensor_log(tmp_path):
    schedule = tmp_path / 'faults.yaml'
    schedule.write_text('faults:\n  - type: dvl_dropout\n    t_start: 0.1\n')
    output = tmp_path / 'run'
    result = cli('run', '--seconds', '.3', '--seed', '4', '--faults', schedule,
                 '--output', output, cwd=tmp_path)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in (output / 'sensors.jsonl').read_text().splitlines()]
    assert rows and rows[-1]['sensors']['dvl']['valid'] is False
    assert 'truth' in rows[-1]
    def public(value):
        if isinstance(value, dict):
            assert not any(k.startswith('_') for k in value)
            for item in value.values():
                public(item)
        elif isinstance(value, list):
            for item in value:
                public(item)
    public(rows[-1]['sensors'])
    summary = json.loads((output / 'summary.json').read_text())
    assert summary['fault_events'][0]['type'] == 'dvl_dropout'
    assert summary['simulated_seconds'] == pytest.approx(.3)


def test_same_seed_reproduces_public_sensor_trace(tmp_path):
    for name in ('one', 'two'):
        result = cli('run', '--seconds', '.2', '--seed', '7', '--output', tmp_path / name,
                     cwd=tmp_path)
        assert result.returncode == 0, result.stderr
    assert (tmp_path / 'one/sensors.jsonl').read_bytes() == (tmp_path / 'two/sensors.jsonl').read_bytes()


@pytest.mark.parametrize('duration', ['0', '-1', 'nan', 'inf'])
def test_invalid_duration_rejected_before_output(duration, tmp_path):
    output = tmp_path / 'never'
    result = cli('run', '--seconds', duration, '--output', output, cwd=tmp_path)
    assert result.returncode == 2 and not output.exists()


def test_existing_run_is_not_overwritten(tmp_path):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep.txt'
    marker.write_text('preserve')
    result = cli('run', '--seconds', '.1', '--output', output, cwd=tmp_path)
    assert result.returncode == 2 and marker.read_text() == 'preserve'


def test_custom_arena_does_not_mutate_the_packaged_one(tmp_path):
    import yaml
    original = load_arena('default')
    custom = load_arena('default')
    custom['start']['x'] = 2.5
    path = tmp_path / 'arena.yaml'
    path.write_text(yaml.safe_dump(custom))
    assert load_arena(str(path))['start']['x'] == 2.5
    assert load_arena('default') == original
