import itertools
import math

import numpy as np
import pytest

from lecture_recognition.evaluation import edit_score, normalize
from lecture_recognition.gigaam_decoding import ContextBias, ctc_beam, rnnt_beam


@pytest.mark.parametrize('a,b', [
    ('20-30 отсчётов', 'двадцати тридцати отсчётов'),
    ('95–98 процентов', 'девяносто пять тире девяносто восемь процентов'),
    ('1024 отсчёта', 'тысяча двадцать четыре отсчёта'),
    ('250 отсчетов', 'двухсот пятидесяти отсчетов'),
])
def test_numbers(a, b):
    # An unspoken dash between adjacent numbers is punctuation, not lexical content.
    assert edit_score(a, b, version=2)['errors'] == 0


def test_numbers_preserve_meaning_and_words():
    assert normalize('124', 2) != normalize('1024', 2)
    assert normalize('95 или 98', 2) != normalize('95-98', 2)
    assert normalize('нестационарный не', 2) == ['нестационарный', 'не']
    assert normalize('сорока прилетела', 2) == ['сорока', 'прилетела']
    assert normalize('одной достаточно', 2) == ['одной', 'достаточно']
    assert normalize('двадцати', 1) == ['двадцати']


def test_ctc_matches_path_enumeration():
    p = np.array([[.4, .1, .5], [.5, .2, .3], [.3, .4, .3]])
    mass = {}
    for path in itertools.product(range(3), repeat=3):
        collapsed = tuple(k for i, k in enumerate(path) if k != 2 and (i == 0 or path[i - 1] != k))
        mass[collapsed] = mass.get(collapsed, 0) + math.prod(p[i, k] for i, k in enumerate(path))
    assert ctc_beam(np.log(p), 2, 100) == max(mass, key=mass.get)
    assert ctc_beam(np.log([[.99, .01], [.01, .99], [.99, .01]]), 1, 20) == (0, 0)


def test_rnnt_matches_bounded_path_enumeration():
    def probs(t, prefix):
        return np.array([.2 + .1 * t, .25 if not prefix else .15, .55 - .1 * t if not prefix else .65 - .1 * t])
    masses = {}
    def walk(t, prefix, depth, probability):
        if t == 2:
            masses[prefix] = masses.get(prefix, 0) + probability
            return
        p = probs(t, prefix)
        walk(t + 1, prefix, 0, probability * p[2])
        if depth < 2:
            for k in (0, 1):
                walk(t, prefix + (k,), depth + 1, probability * p[k])
    walk(0, (), 0, 1)
    result, _ = rnnt_beam(lambda t, p: np.log(probs(t, p)), 2, 2, 100, 2)
    assert result == max(masses, key=masses.get)


def test_bias_refunds_and_boundaries():
    def decode(ids):
        return ''.join(chr(i) for i in ids)
    b = ContextBias(decode, ['кот', 'котик'], 2)
    def p(s):
        return tuple(map(ord, s))
    assert b(p('ко')) > 0
    assert b(p('ко'), final=True) == 0
    assert b(p('кожа')) == 0
    assert b(p('скот')) == 0
    assert b(p('котик ')) == 2
    assert b(p('кот кот'), final=True) == 4
    assert ContextBias(decode, ['кот'], 0)(p('кот')) == 0


def test_zero_bias_decoders():
    p = np.log([[.3, .7], [.8, .2]])
    zero = ContextBias(lambda ids: 'a' * len(ids), ['a'], 0)
    assert ctc_beam(p, 1, 8) == ctc_beam(p, 1, 8, zero)
    def step(t, prefix):
        return p[t]
    assert rnnt_beam(step, 2, 1, 4, 2) == rnnt_beam(step, 2, 1, 4, 2, zero)


def test_selection_all_worse_mixed_equal_and_incomplete():
    from lecture_recognition.gigaam_tuning import eligible

    def case(values):
        return {'status': 'ok', 'score': {'cards': {f'R{i:03d}': {'wer': x} for i, x in enumerate(values, 1)}}}
    base = case([.2]*8)
    assert not eligible(case([.3]*8), base)
    assert eligible(case([.1]+[.3]*7), base)
    assert eligible(case([.2]+[.3]*7), base)
    assert eligible(base, base)
    assert not eligible(case([.1]*7), base)
    assert not eligible({'status': 'error'}, base)


@pytest.mark.parametrize('maximum,context', [(12, 1), (20, 1), (25, 1), (20, 2)])
@pytest.mark.parametrize('shift', [-5, 0, 5])
def test_tuning_layout_limits(maximum, context, shift):
    from lecture_recognition.gigaam_tuning import make_layout
    duration = 100.3
    audio = np.zeros(round(duration*16000), dtype=np.float32)
    chunks = make_layout(audio, duration, maximum, context, shift)
    assert chunks[0]['core_start'] == 0
    assert chunks[-1]['core_end'] == duration
    assert all(0 <= c['start'] <= c['core_start'] < c['core_end'] <= c['end'] <= duration for c in chunks)
    assert all(c['end']-c['start'] <= maximum+1e-6 for c in chunks)
    assert all(a['core_end'] == b['core_start'] for a, b in zip(chunks, chunks[1:]))


def test_tuning_layout_shift_both_directions():
    from lecture_recognition.gigaam_tuning import make_layout
    audio = np.zeros(100*16000, dtype=np.float32)
    cuts = [make_layout(audio, 100, shift=s)[0]['core_end'] for s in (-5, 0, 5)]
    assert cuts[0] < cuts[1] < cuts[2]


def test_numbers_do_not_hide_family():
    assert normalize('он любит семью', 2) == ['он', 'любит', 'семью']
    assert normalize('семью отсчетами', 2) == ['7', 'отсчетами']


def test_complete_matrix_orchestration_without_gpu(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lecture_recognition.gigaam_tuning as tuning

    study = object.__new__(tuning.Study)
    study.root = tmp_path
    study.source = tmp_path
    study.cases = {}
    monkeypatch.setattr(tuning.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(tuning, 'read', lambda p: {'transcripts': [{'text': 'test'}]})

    def case(model, label, stage, **options):
        cfg = dict(model=model, maximum=20, context=1, audio_variant='original', precision='default', **{})
        cfg.update(options)
        # Ensure both input and decoder winners, a combined candidate and conditional limit trials.
        wer = {'baseline': .3, 'input': .2, 'decoder': .15, 'dictionary': .1,
               'control': .4, 'combination': .05, 'limit': .09, 'validation': .05}[stage]
        total = dict(wer=wer, number_errors=0, negation_errors=0, cer=wer)
        result = {'id': str(len(study.cases)), 'label': label, 'stage': stage, 'status': 'ok',
                  'config': cfg, 'limit_hits': 1, 'asr_seconds': 1,
                  'score': {'total': total, 'cards': {f'R{i:03d}': {'wer': wer} for i in range(1, 9)}}}
        study.cases[result['id']] = result
        return result
    study.case = case
    study.run('all')
    assert len(study.cases) == 46
    selection = __import__('json').loads((tmp_path/'selection.json').read_text())
    assert len(selection) == 2
    assert all(len(item['checks']) == 6 for item in selection.values())
    assert all(study.cases[item['candidate']]['stage'] == 'combination' for item in selection.values())


def test_cuda_preflight_leaves_no_fake_results(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lecture_recognition.gigaam_tuning as tuning

    study = object.__new__(tuning.Study)
    study.root = tmp_path
    monkeypatch.setattr(tuning.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=1, stderr='no CUDA'))
    with pytest.raises(RuntimeError, match='CUDA is unavailable'):
        study.run('all')
    assert not (tmp_path/'cases').exists()
    assert 'cuda_unavailable' in (tmp_path/'execution.json').read_text()


def test_identical_audio_and_decoder_reuse_without_worker(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import lecture_recognition.gigaam_tuning as tuning

    study = object.__new__(tuning.Study)
    study.root = tmp_path
    study.args = SimpleNamespace(retry_failed=False)
    study.audio_hashes = {'original': 'same', 'left': 'same'}
    chunks = [dict(start=0, end=1, core_start=0, core_end=1)]
    study.originals = {'gigaam-ctc': {'chunks': chunks}}
    cfg = {**tuning.MODELS['gigaam-ctc'], 'model': 'gigaam-ctc', 'dictionary': None,
           'decoding': 'greedy', 'precision': 'default', 'diagnostic_greedy': False}
    cfg.pop('maximum')
    signature = tuning.identity({'config': cfg, 'chunks': chunks, 'audio': 'same'})
    previous = {'id': 'old', 'status': 'ok', 'signature': signature}
    study.cases = {'old': previous}
    study.audio = lambda variant: tmp_path/'audio.f32'
    monkeypatch.setattr(tuning, 'worker', lambda *a, **k: pytest.fail('Duplicate must not run ASR'))
    assert study.case('gigaam-ctc', 'left', 'input', audio_variant='left') is previous
    assert 'identical_input_and_decoder' in (tmp_path/'aliases.json').read_text()


def test_terminal_ctc_pruning_refunds_incomplete_phrase():
    bias = ContextBias(lambda ids: ''.join('ab'[i] for i in ids), ['ab'], 2)
    assert ctc_beam(np.log([[.44, .55, .01]]), 2, 1, bias) == (1,)
