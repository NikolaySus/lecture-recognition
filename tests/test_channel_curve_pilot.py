import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from lecture_recognition.audio import RATE
from lecture_recognition.channel_curve_pilot import (
    BASES,
    GRIDS,
    Pilot,
    choose_shortlist,
    progress,
    quality_ties,
    r005_score,
)
from lecture_recognition.dynamic_channels import sharpen_weights
from lecture_recognition.model_benchmark import read, write


@pytest.mark.parametrize('k', [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize('threshold', [.51, .53, .52, .54, .56, .58, .60, .61, .62, .63, .64, .65, .66, .67])
def test_curve_identity_symmetry_monotonicity_and_strength(threshold, k):
    x = np.linspace(0, 1, 10001)
    y = sharpen_weights(x, threshold, k)
    assert np.array_equal(y[(x >= 1 - threshold) & (x <= threshold)],
                          x[(x >= 1 - threshold) & (x <= threshold)])
    assert np.allclose(y + sharpen_weights(1 - x, threshold, k), 1, atol=1e-14)
    assert np.all(np.diff(y) >= 0)
    assert y[0] == 0 and y[-1] == 1
    assert np.allclose(sharpen_weights(x, threshold, 0), x)
    assert np.all(sharpen_weights(x[x > threshold], threshold, k * 2) >= y[x > threshold])
    step = 1e-8  # Resolve the steeper curvature near the threshold at k=64.
    derivative = (sharpen_weights(threshold + step, threshold, k) - threshold) / step
    assert derivative == pytest.approx(1, abs=1e-5)


@pytest.mark.parametrize('threshold,k', [(.5, 4), (1, 4), (.6, -1), (.6, float('nan'))])
def test_invalid_curve_parameters(threshold, k):
    with pytest.raises(ValueError):
        sharpen_weights([.6], threshold, k)


def test_sample_curve_follows_interpolation_and_preserves_middle():
    weights = np.interp(np.linspace(0, 1, 11), [0, 1], [.55, .75])
    result = sharpen_weights(weights, .65, 4)
    assert np.array_equal(result[weights <= .65], weights[weights <= .65])
    assert result[-1] > weights[-1]


def test_r005_projection_keeps_word_past_time_boundary():
    reference = {'groups': [{'name': 'group', 'window': [0, 3], 'text': 'этого сегмента далее',
                  'cards': {'R005': {'text': 'этого сегмента', 'window': [0, .9], 'span': [0, 2]}}}]}
    words = [{'text': 'этого', 'start': .1, 'end': .4},
             {'text': 'сегмента', 'start': 1, 'end': 1.3}, {'text': 'далее', 'start': 2, 'end': 2.3}]
    score = r005_score(words, reference)
    assert score['errors'] == 0
    assert score['hypothesis'] == 'этого сегмента'


def candidate(i, errors):
    return {'id': str(i), 'label': str(i), 'status': 'ok', 'threshold_percent': 60,
            'steepness': 4, 'score': {'errors': errors, 'cer': errors / 100}}


def test_shortlist_extends_top10_and_excludes_errors():
    cases = [candidate(i, 2) for i in range(14)] + [candidate(20, 3), {'id': 'bad', 'status': 'error'}]
    result = choose_shortlist(cases, 2)
    assert len(result) == 14
    assert sum(c['in_top10'] for c in result) == 10
    assert all(c['reached_rnnt_left'] for c in result)
    assert choose_shortlist(list(reversed(cases)), 2) == result
    assert len(choose_shortlist([candidate(i, 3) for i in range(20)], 2)) == 10


def test_shared_chunk_cache_and_resume_keep_distinct_configurations(tmp_path, monkeypatch):
    pilot = Pilot.__new__(Pilot)
    pilot.root = tmp_path
    pilot.args = SimpleNamespace(retry_failed=False)
    pilot.expected_cases = 2
    pcm = tmp_path / 'audio.f32'
    np.zeros(RATE * 12, dtype='<f4').tofile(pcm)
    pilot.prepared = {'audio': str(pcm)}
    texts = [' '.join(['раз'] * 14), ' '.join(['раз'] * 13)]
    reference_text = ' '.join(texts)
    pilot.reference = {'groups': [{'name': 'context', 'text': reference_text, 'window': [10, 12],
                                  'cards': {'R005': {'text': reference_text, 'span': [0, 27]}}}]}
    chunks = [{'start': i, 'end': i + 1, 'core_start': i, 'core_end': i + 1} for i in range(12)]
    pilot.chunks = {'gigaam-rnnt': chunks[10:]}
    pilot.parent_aligned = {'gigaam-rnnt': [{'chunk': c, 'words': []} for c in chunks]}
    pilot.fixed = {'gigaam-rnnt left': {'config': {'model': 'gigaam-rnnt', 'backend': 'gigaam'}}}
    pilot.controls = {'gigaam-rnnt left': {'score': {'errors': 0, 'words': 27, 'wer': 0}}}
    pilot.cases = [{**candidate(i, 0), 'status': 'pending', 'model': 'gigaam-rnnt', 'method': 'energy',
                    'speed': 'fast', 'threshold_percent': 60, 'steepness': k} for i, k in enumerate([2, 4])]
    monkeypatch.setattr(pilot, 'audio', lambda case: pcm)
    calls = []
    def worker(request, directory, backend='qwen', retry=False):
        if (directory / 'result.json').exists():
            return read(directory / 'result.json')
        calls.append(request['operation'])
        write(directory / 'request.json', request)
        if request['operation'] == 'asr':
            chunk = request['chunks'][0]
            result = {'status': 'ok', 'transcripts': [{'chunk': chunk, 'text': texts[chunk['start'] - 10]}]}
        else:
            tr = request['transcripts'][0]
            key = hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest()
            write(directory.parent / 'alignment-cache/alignment' / (key + '.json'),
                  {'chunk': tr['chunk'], 'words': [{'text': tr['text'], 'start_time': .1, 'end_time': .5}]})
            result = {'status': 'ok'}
        write(directory / 'result.json', result)
        return result
    monkeypatch.setattr('lecture_recognition.channel_curve_pilot.worker', worker)
    for case in pilot.cases:
        pilot.execute(case)
    assert all(c['status'] == 'ok' and c['score']['errors'] == 0 for c in pilot.cases)
    assert calls.count('asr') == 2 and calls.count('align') == 2
    assert pilot.cases[0]['inference_cache_keys'] == pilot.cases[1]['inference_cache_keys']
    before = list(calls)
    for case in pilot.cases:
        pilot.execute(case)
    assert calls == before


def test_grid_sizes_and_ties():
    assert 2 * len(BASES) * len(GRIDS['lower']['thresholds']) * len(GRIDS['lower']['steepness']) == 160
    assert 2 * len(BASES) * len(GRIDS['legacy']['thresholds']) * len(GRIDS['legacy']['steepness']) == 192
    assert 50 not in GRIDS['lower']['thresholds']
    cases = [candidate(1, 3), candidate(2, 3), candidate(3, 4)]
    assert quality_ties(cases) == [{'errors': 3, 'cer': .03, 'case_ids': ['1', '2']}]


@pytest.mark.parametrize('count,total', [(18, 72), (160, 640), (192, 768)])
def test_progress_uses_metadata_and_counts_shared_work_per_case(tmp_path, count, total):
    write(tmp_path / 'metadata.json', {'expected_cases': count, 'chunk_indices': [10, 11]})
    write(tmp_path / 'execution.json', {'status': 'incomplete'})
    for i in range(2):
        write(tmp_path / 'cases' / str(i) / 'case.json', {'inference_cache_keys': ['shared']})
    write(tmp_path / 'inference-cache/shared/asr/result.json', {'status': 'ok'})
    write(tmp_path / 'inference-cache/shared/alignment/result.json', {'status': 'error'})
    result = progress(tmp_path)
    assert result['total_logical_steps'] == total
    assert result['planned_case_chunks'] == count * 2
    assert result['unique_asr_chunks'] == 1
    assert result['completed_logical_steps'] == 2
    assert result['percent'] == round(200 / total, 1)


def test_energy_focus_excludes_other_methods_and_speeds():
    grid = GRIDS['energy-focus']
    assert grid['bases'] == (('energy', 'fast'),)
    assert grid['thresholds'] == (51, 52, 53)
    assert grid['steepness'] == (16, 32, 64)
    assert 2 * len(grid['bases']) * len(grid['thresholds']) * len(grid['steepness']) == 18
