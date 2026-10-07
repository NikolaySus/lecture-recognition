import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from lecture_recognition.audio import RATE, digest
from lecture_recognition.dynamic_channels import METHODS, SPEEDS, features, mix
from lecture_recognition.dynamic_study import FIXED, DynamicStudy
from lecture_recognition.evaluation import score_case
from lecture_recognition.model_benchmark import MODELS, ROOT, extra_metrics, read, write


def speech(seconds=5):
    t = np.arange(round(seconds * RATE)) / RATE
    return .2 * np.sin(2 * np.pi * 220 * t) * (.55 + .45 * np.sin(2 * np.pi * 4 * t))


@pytest.mark.parametrize('method', METHODS)
@pytest.mark.parametrize('speed', SPEEDS)
def test_identity_swap_mask_and_bounds(method, speed):
    x = speech()
    stereo = np.stack([x, x * .3], axis=1)
    wave, d = mix(stereo, method, speed, [(1, 2)])
    mirrored, md = mix(stereo[:, ::-1], method, speed, [(1, 2)])
    assert len(wave) == len(x)
    assert np.isfinite(wave).all()
    assert np.max(np.abs(wave)) <= np.max(np.abs(stereo))
    assert np.all(wave[RATE:2 * RATE] == 0)
    assert np.allclose(wave, mirrored, atol=1e-7)
    assert np.allclose(np.array(d['weights_left']) + md['weights_left'], 1)
    same, _ = mix(np.stack([x, x], axis=1), method, speed)
    assert np.allclose(same, x, atol=1e-7)
    silent, sd = mix(np.zeros_like(stereo), method, speed)
    assert not silent.any()
    assert set(sd['weights_left']) == {.5}
    absent, ad = mix(np.stack([x, np.zeros_like(x)], axis=1), method, speed)
    assert ad['weights_left'][-1] > .98
    assert np.isfinite(absent).all()


def test_energy_tracks_side_changes_and_speed():
    x = speech(12)
    stereo = np.stack([x.copy(), x.copy()], axis=1)
    stereo[:6 * RATE, 1] *= .2
    stereo[6 * RATE:, 0] *= .2
    _, fast = mix(stereo, 'energy', 'fast')
    _, slow = mix(stereo, 'energy', 'slow')
    assert fast['weights_left'][50] > .9
    assert fast['weights_left'][90] < .1
    assert fast['weights_left'][75] < slow['weights_left'][75]


def test_snr_rejects_steady_noise_and_ev_ignores_gain():
    rng = np.random.default_rng(17)
    x = speech(8)
    noise = .2 * rng.normal(size=len(x))
    stereo = np.stack([x, x + noise], axis=1)
    _, snr = mix(stereo, 'snr-proxy', 'fast')
    assert snr['weights_left'][-1] > .7
    _, ev = mix(np.stack([x, x * .2], axis=1), 'ev-normalized', 'fast')
    assert np.allclose(ev['weights_left'], .5, atol=1e-7)


def test_polarity_diagnostic_and_invalid_input():
    x = speech(2)
    wave, d = mix(np.stack([x, -x], axis=1), 'energy', 'fast')
    assert not wave.any()
    assert d['suppressed_intervals']
    with pytest.raises(ValueError):
        features(np.array([[np.nan, 0]]))
    with pytest.raises(ValueError):
        mix(np.stack([x, x], axis=1), 'unknown', 'fast')


def historical_fixture(tmp_path):
    source = tmp_path / 'historical'
    prepared = tmp_path / 'benchmark'
    audio, review = tmp_path / 'source.m4a', tmp_path / 'review.md'
    audio.write_bytes(b'fixture')
    review.write_text('fixture')
    pcm = tmp_path / 'masked.f32'
    np.zeros(RATE * 2, dtype='<f4').tofile(pcm)
    write(prepared / 'prepared.json', {'audio': str(pcm), 'prefix_end': 2, 'duration': 2,
                                     'masked_sha256': digest(pcm), 'regions': [[0, 2]], 'excluded': []})
    reference = {'groups': [{'name': 'all', 'window': [0, 2], 'text': 'текст',
                             'cards': {f'R{i:03d}': {'text': 'текст', 'window': [0, 2], 'span': [0, 1]}
                                       for i in range(1, 9)}}], 'target_window': [0, 2]}
    words = [{'start': .2, 'end': .6, 'text': 'текст'}]
    write(source / 'reference.json', reference)
    write(source / 'metadata.json', {'source': str(prepared), 'review_sha256': digest(review),
                                    'source_audio_sha256': digest(audio), 'postprocessing_code': {
                                        'src/lecture_recognition/' + n: digest(ROOT / 'src/lecture_recognition' / n)
                                        for n in ('timeline.py', 'evaluation.py', 'model_benchmark.py')}})
    for i, label in enumerate(FIXED):
        model = label.split()[0]
        cfg = {**MODELS[model], 'model': model, 'dictionary': None, 'decoding': 'greedy', 'maximum': 20,
               'context': 1, 'audio_variant': 'original', 'precision': 'default',
               'diagnostic_greedy': model == 'gigaam-rnnt'}
        case = {'id': str(i), 'label': label, 'config': cfg, 'chunks': [
            {'start': 0, 'end': 2, 'core_start': 0, 'core_end': 2}],
            'stage': 'baseline' if label.endswith('original') else 'input', 'status': 'ok',
            'score': extra_metrics(score_case(words, reference, 2)),
            'legacy_score': extra_metrics(score_case(words, reference, 1))}
        directory = source / 'cases' / str(i)
        write(directory / 'case.json', case)
        write(directory / 'alignment/result.json', {'words': words})
        write(directory / 'seam-audit.json', [])
        (directory / 'transcript.srt').write_text('fixture')
    return SimpleNamespace(source=source, output=tmp_path / 'output', review=review, audio=audio,
                           retry_failed=False)


def test_import_resume_failure_retry_and_pending_cases(tmp_path, monkeypatch):
    args = historical_fixture(tmp_path)
    def audio(self, variant):
        return Path(self.prepared['audio'])
    from pathlib import Path
    monkeypatch.setattr(DynamicStudy, 'audio', audio)
    study = DynamicStudy(args)
    study.prepare_cases()
    assert len(study.cases) == 17
    resumed = DynamicStudy(args)
    resumed.prepare_cases()
    assert resumed.root == study.root
    assert len(resumed.cases) == 17
    calls = []
    def worker(request, *args, **kwargs):
        calls.append(request['operation'])
        if request['operation'] == 'asr':
            return {'status': 'error', 'error': 'test failure'}
        raise AssertionError('Failed ASR must not start alignment')
    monkeypatch.setattr('lecture_recognition.gigaam_tuning.worker', worker)
    cfg = {k: v for k, v in study.fixed['gigaam-rnnt left']['config'].items() if k != 'model'}
    cfg['audio_variant'] = 'dynamic_energy_fast'
    label = 'gigaam-rnnt dynamic energy fast'
    case = resumed.case('gigaam-rnnt', label, 'dynamic', **cfg)
    assert case['status'] == 'error'
    assert len(resumed.cases) == 17
    resumed.case('gigaam-rnnt', label, 'dynamic', **cfg)
    assert calls == ['asr']
    args.retry_failed = True
    resumed.case('gigaam-rnnt', label, 'dynamic', **cfg)
    assert calls == ['asr', 'asr']
    def success(request, *args, **kwargs):
        calls.append(request['operation'])
        if request['operation'] == 'asr':
            return {'status': 'ok', 'transcripts': []}
        return {'status': 'ok', 'words': [{'start': .2, 'end': .6, 'text': 'текст'}]}
    monkeypatch.setattr('lecture_recognition.gigaam_tuning.worker', success)
    completed = resumed.case('gigaam-rnnt', label, 'dynamic', **cfg)
    assert completed['status'] == 'ok'
    before = len(calls)
    resumed.case('gigaam-rnnt', label, 'dynamic', **cfg)
    assert len(calls) == before
    assert len(resumed.cases) == 17
    args.review.write_text('changed')
    with pytest.raises(ValueError, match='Review changed'):
        DynamicStudy(args)


def test_report_keeps_all_dynamic_even_when_dominated(tmp_path, monkeypatch):
    args = historical_fixture(tmp_path)
    from pathlib import Path
    monkeypatch.setattr(DynamicStudy, 'audio', lambda self, variant: Path(self.prepared['audio']))
    study = DynamicStudy(args)
    study.prepare_cases()
    template = study.fixed['gigaam-ctc original']
    for case in study.cases.values():
        if case['stage'] == 'dynamic':
            case.update(status='ok', score=read(study.root / 'cases' / template['id'] / 'case.json')['score'])
            # Deliberately dominated synthetic results: all must still be included.
            for card in case['score']['cards'].values():
                card.update(errors=1, wer=1.0)
            case['score']['total'].update(errors=1, wer=1.0)
            write(study.root / 'cases' / case['id'] / 'case.json', case)
    command = "import sys; from pathlib import Path; sys.path.insert(0, 'scripts'); " \
              "from create_gigaam_report import collect; " \
              "rows, selected, cases, bases = collect(Path(sys.argv[1])); " \
              "assert len(selected) == 17; " \
              "assert len(rows) == 8; " \
              "assert all(len(row['models']) == 17 for row in rows); " \
              "wers = [c['score']['total']['wer'] for c in selected]; " \
              "assert wers == sorted(wers, reverse=True)"
    subprocess.run(['python3', '-c', command, str(study.root)], cwd=ROOT, check=True)
