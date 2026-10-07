import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from lecture_recognition.channel_curve_full import FullStudy, alignment_key, chunk_key, progress
from lecture_recognition.model_benchmark import ROOT, read, write


def test_chunk_cache_identity_includes_samples_decoder_and_bounds():
    pcm = np.array([.1, .2], dtype='<f4')
    chunk = {'start': 1, 'end': 2}
    config = {'model': 'gigaam-ctc', 'beam': 32}
    key = chunk_key(pcm, config, chunk)
    assert key == chunk_key(pcm.copy(), dict(config), dict(chunk))
    assert key != chunk_key(pcm * 2, config, chunk)
    assert key != chunk_key(pcm, {**config, 'beam': 64}, chunk)
    assert key != chunk_key(pcm, config, {**chunk, 'start': 0})


def test_full_run_reuses_cached_chunks_and_resumes_failed_chunk(tmp_path, monkeypatch):
    study = FullStudy.__new__(FullStudy)
    study.root = tmp_path
    study.args = SimpleNamespace(retry_failed=False)
    study.prepared = {'audio': 'alignment.f32'}
    study.reference = {'groups': [{'cards': {'R004': {'window': [0, 1]}, 'R005': {'window': [10, 12]}}}]}
    pilot_score = {'errors': 3, 'words': 27}
    study.pilot_cases = {'gigaam-rnnt': {'score': pilot_score}}
    chunks = [{'start': i, 'end': i + 1} for i in range(22)]
    case = {'id': 'new', 'stage': 'dynamic', 'status': 'pending', 'label': 'test', 'audio': 'mix.f32',
            'config': {'model': 'gigaam-rnnt'}, 'chunks': chunks,
            'inference_cache_keys': [hashlib.sha256(str(i).encode()).hexdigest() for i in range(22)]}
    study.cases = {case['id']: case}
    calls = []
    fail = True

    def cached_chunk(i):
        transcript = {'chunk': chunks[i], 'text': f'word{i}'}
        shared = tmp_path / 'inference-cache' / case['inference_cache_keys'][i]
        write(shared / 'asr/result.json', {'status': 'ok', 'transcripts': [transcript]})
        write(shared / 'alignment/result.json', {'status': 'ok'})
        write(shared / 'alignment-cache/alignment' / (alignment_key(transcript) + '.json'),
              {'chunk': chunks[i], 'words': []})

    # The two pilot chunks must not launch another worker.
    for index in (10, 11):
        cached_chunk(index)

    def fake_worker(request, directory, backend='qwen', retry=False):
        index = (request['chunks'][0] if request['operation'] == 'asr'
                 else request['transcripts'][0]['chunk'])['start']
        calls.append((request['operation'], index, retry))
        if request['operation'] == 'asr' and index == 2 and fail:
            result = {'status': 'error', 'error': 'temporary network error'}
        elif request['operation'] == 'asr':
            result = {'status': 'ok', 'transcripts': [{'chunk': chunks[index], 'text': f'word{index}'}]}
        else:
            transcript = request['transcripts'][0]
            write(directory.parent / 'alignment-cache/alignment' / (alignment_key(transcript) + '.json'),
                  {'chunk': chunks[index], 'words': []})
            result = {'status': 'ok'}
        write(directory / 'result.json', result)
        return result

    monkeypatch.setattr('lecture_recognition.channel_curve_full.worker', fake_worker)
    finished = []

    def finish(record, aligned, directory):
        finished.extend(a['chunk']['start'] for a in aligned)
        return {**record, 'status': 'ok', 'score': {'cards': {'R004': {}, 'R005': pilot_score}}}

    monkeypatch.setattr(study, 'finish', finish)
    study.execute(case)
    assert case['status'] == 'error'
    assert progress(tmp_path)['completed_logical_steps'] == 8
    assert progress(tmp_path)['total_logical_steps'] == 44
    previous_calls = list(calls)
    study.execute(case)
    assert calls == previous_calls
    fail = False
    study.args.retry_failed = True
    study.execute(case)
    assert case['status'] == 'ok'
    assert finished == list(range(22))  # Whole timeline, without frozen context.
    assert all(index not in (10, 11) for _, index, _ in calls)
    assert sum(operation == 'asr' and index == 0 for operation, index, _ in calls) == 1
    assert read(tmp_path / 'cases/new/pilot-r005-check.json')['matches']
    assert progress(tmp_path)['percent'] == 100
    calls_before = list(calls)
    study.execute(case)
    assert calls == calls_before


def test_prepare_rejects_pcm_changed_since_pilot(tmp_path):
    study = FullStudy.__new__(FullStudy)
    study.root = tmp_path
    path = tmp_path / 'audio/dynamic_energy_fast_t52_k32.f32'
    path.parent.mkdir()
    np.array([.1, .2], dtype='<f4').tofile(path)
    write(path.with_suffix('.json'), {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    pilot_path = tmp_path / 'pilot.f32'
    np.array([.1, .3], dtype='<f4').tofile(pilot_path)
    study.cases = {'new': {'stage': 'dynamic', 'config': {'model': 'gigaam-rnnt'}}}
    study.pilot_cases = {'gigaam-rnnt': {'audio': str(pilot_path)}}
    with pytest.raises(ValueError, match='PCM differs'):
        study.prepare()


def test_report_retains_seven_variants_and_all_equal_winners(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    from create_gigaam_report import build, collect
    labels = [('gigaam-ctc', 'baseline'), ('gigaam-rnnt', 'baseline'),
              ('gigaam-rnnt', 'control'), ('gigaam-ctc', 'control'),
              ('gigaam-ctc', 'control'), ('gigaam-ctc', 'dynamic'), ('gigaam-rnnt', 'dynamic')]
    reference = {'groups': [{'cards': {f'R{i:03d}': {'text': 'эталон', 'window': [i, i + 1]}
                                     for i in range(1, 9)}}]}
    write(tmp_path / 'reference.json', reference)
    write(tmp_path / 'metadata.json', {'series': 'channel-curve-full-v1',
                                     'report_selection': 'all', 'expected_cases': 7})
    for i, (model, stage) in enumerate(labels):
        score = {'total': {'wer': .1 * i, 'number_errors': 0, 'negation_errors': 0, 'cer': 0},
                 'cards': {f'R{j:03d}': {'errors': 0, 'words': 1, 'wer': 0, 'hypothesis': 'эталон'}
                           for j in range(1, 9)}}
        write(tmp_path / 'cases' / str(i) / 'case.json', {'id': str(i), 'label': str(i), 'stage': stage,
              'status': 'ok', 'config': {'model': model}, 'score': score})
    rows, selected, cases, _ = collect(tmp_path)
    assert len(selected) == 7
    assert [c['score']['total']['wer'] for c in selected] == sorted(
        [c['score']['total']['wer'] for c in cases], reverse=True)
    assert len(rows) == 8
    assert all(len(row['models']) == 7 and all(c['best'] for c in row['models']) for row in rows)
    failed = read(tmp_path / 'cases/6/case.json')
    failed['status'] = 'error'
    write(tmp_path / 'cases/6/case.json', failed)
    with pytest.raises(ValueError, match='seven successful'):
        build(tmp_path, tmp_path / 'incomplete.pdf')
