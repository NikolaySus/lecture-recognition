"""CPU-only R005 diagnostics for lower channel-curve thresholds; no ASR or PCM generation."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lecture_recognition.audio import RATE  # noqa: E402
from lecture_recognition.channel_curve_pilot import BASES, GRIDS  # noqa: E402
from lecture_recognition.dynamic_channels import sharpen_weights  # noqa: E402


def statistics(weights):
    return {'left_mean_percent': float(weights.mean() * 100),
            'left_min_percent': float(weights.min() * 100),
            'left_max_percent': float(weights.max() * 100),
            'left_above80_percent': float(np.mean(weights > .8) * 100),
            'left_above90_percent': float(np.mean(weights > .9) * 100),
            'right_above80_percent': float(np.mean(weights < .2) * 100),
            'right_above90_percent': float(np.mean(weights < .1) * 100)}


def analyze(source, output, grid='lower'):
    thresholds = GRIDS[grid]['thresholds']
    steepness_values = GRIDS[grid]['steepness']
    reference_path = source / 'reference.json'
    reference = json.loads(reference_path.read_text())
    window = next(g['cards']['R005']['window'] for g in reference['groups'] if 'R005' in g['cards'])
    a, b = (round(t * RATE) for t in window)
    times = np.arange(a, b) / RATE
    rows, bases = [], []
    paths = [reference_path, Path(__file__), ROOT / 'src/lecture_recognition/dynamic_channels.py',
             ROOT / 'src/lecture_recognition/channel_curve_pilot.py']
    for method, speed in GRIDS[grid].get('bases', BASES):
        path = source / 'audio' / f'dynamic_{method}_{speed}.json'
        paths.append(path)
        diagnostic = json.loads(path.read_text())
        before = np.interp(times, diagnostic['times'], diagnostic['weights_left'])
        bases.append({'method': method, 'speed': speed, **statistics(before)})
        for threshold in thresholds:
            for steepness in steepness_values:
                after = sharpen_weights(before, threshold / 100, steepness)
                assert np.allclose(after + sharpen_weights(1 - before, threshold / 100, steepness),
                                   1, atol=1e-12)
                unchanged = np.maximum(before, 1 - before) <= threshold / 100
                assert np.array_equal(after[unchanged], before[unchanged])
                rows.append({'method': method, 'speed': speed, 'threshold_percent': threshold,
                             'steepness': steepness, **statistics(after),
                             'changed_time_percent': float(np.mean(np.abs(after - before) > 1e-12) * 100),
                             'right_amplified_time_percent': float(np.mean(after < before - 1e-12) * 100)})
    result = {'stage': 'weights_only', 'asr_started': False, 'source': str(source.resolve()),
              'card': 'R005', 'window': window, 'samples': len(times), 'grid': grid, 'thresholds_percent': thresholds,
              'steepness': steepness_values, 'curve_position': 'after_smoothing_and_sample_interpolation',
              'base_weights': bases, 'mixtures': len(rows), 'future_model_configurations': 2 * len(rows),
              'rows': rows, 'source_hashes': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest()
                                            for p in paths}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    with output.with_suffix('.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['threshold_percent'] == 52 and row['steepness'] in (8, 16, 32, 64):
            print(f"{row['method']} {row['speed']} k{row['steepness']}: left mean {row['left_mean_percent']:.2f}%, "
                  f"max {row['left_max_percent']:.2f}%, time >80% {row['left_above80_percent']:.2f}%, "
                  f"right >80% {row['right_above80_percent']:.2f}%")
    print(f'Wrote {output}; {len(rows)} mixtures; no ASR started.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=tuple(GRIDS), default='lower')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/channel-curves-r005-weights.json')
    args = parser.parse_args()
    source = args.source or Path(json.loads((ROOT / '.lecture-cache/gigaam-dynamic-channels/latest.json')
                                           .read_text())['run'])
    analyze(source, args.output, args.grid)
