import itertools

import numpy as np
import pytest

from lecture_recognition.channel_utility import (
    acoustic_statistics,
    assemble,
    ctc_log_likelihood,
    entropy_confidence,
    likelihood_margin,
    make_layout,
    select_channels,
    validation_windows,
    word_confidence,
)


def test_ctc_forward_matches_all_paths():
    p = np.array([[.2, .3, .5], [.4, .1, .5], [.3, .4, .3]])
    for target in ([], [0], [1], [0, 0], [0, 1], [1, 0]):
        mass = 0.
        for path in itertools.product(range(3), repeat=3):
            collapsed = [x for i, x in enumerate(path) if x != 2 and (i == 0 or x != path[i - 1])]
            if collapsed == target:
                mass += np.prod([p[i, x] for i, x in enumerate(path)])
        assert np.exp(ctc_log_likelihood(np.log(p), target, 2)) == pytest.approx(mass)
    assert ctc_log_likelihood(np.log(p), [0, 0, 0], 2) == -np.inf


def test_entropy_and_blank_aggregation():
    for kind in ('gibbs', 'tsallis'):
        assert entropy_confidence([[1, 0], [.5, .5]], kind) == pytest.approx([1, 0])
    p = np.array([[.9, .1], [.8, .2], [.01, .99], [.9, .1]])
    score = word_confidence(p, [0, 0, 1, 0], [0, 0, 0, 1], 1, ctc=True)
    conf = entropy_confidence(p)
    assert score['gibbs'] == pytest.approx((min(conf[:2]) + conf[3]) / 2)
    assert word_confidence([[.1, .9]], [1], [0], 1)['gibbs'] is None


@pytest.mark.parametrize('mode', ['regular', 'vad'])
@pytest.mark.parametrize('shift', [-2, 0, 2])
def test_layout_covers_without_holes_or_oversize(mode, shift):
    wave = np.zeros(100 * 16000)
    chunks = make_layout(wave, 100, 30, 2, mode, [(0, 35), (36, 100)], shift)
    assert chunks[0]['core_start'] == 0 and chunks[-1]['core_end'] == 100
    for c in chunks:
        assert c['end'] - c['start'] <= 30 + 1e-6
        assert c['start'] <= c['core_start'] < c['core_end'] <= c['end']
    assert all(a['core_end'] == b['core_start'] for a, b in zip(chunks, chunks[1:]))


def test_overlap_retains_real_repetition():
    a = {'chunk': {'start': 0, 'end': 12, 'core_start': 0, 'core_end': 10},
         'words': [{'text': 'для', 'start_time': 9, 'end_time': 9.3},
                   {'text': 'для', 'start_time': 10.3, 'end_time': 10.6}]}
    b = {'chunk': {'start': 8, 'end': 20, 'core_start': 10, 'core_end': 20},
         'words': [{'text': 'для', 'start_time': 1, 'end_time': 1.3},
                   {'text': 'для', 'start_time': 2.3, 'end_time': 2.6}]}
    assert [x['text'] for x in assemble([a, b], 'overlap')] == ['для', 'для']
    assert len(assemble([a, b], 'core')) == 2


def test_selection_hysteresis_fallback_and_symmetry():
    def row(v):
        return {'left_text': 'a', 'right_text': 'b', 'left': {}, 'right': {}, 'margin': v}
    decisions = select_channels([row(-.2), row(.01), row(None), row(.2)], 'margin', .05)
    assert [d['channel'] for d in decisions] == ['right', 'right', 'left', 'left']
    p, q = np.log([[.8, .1, .1], [.7, .1, .2]]), np.log([[.1, .8, .1], [.1, .7, .2]])
    margin = likelihood_margin(p, q, [0], [1], 2)
    assert margin == pytest.approx(-likelihood_margin(q, p, [1], [0], 2))
    assert likelihood_margin(p, q, [0], [0], 2) == 0


def test_acoustic_minimum_speech_and_windows():
    times = np.arange(0, 3, .1)
    values = acoustic_statistics(times, np.ones(30), np.arange(30), np.arange(30), [1, 2.5])
    assert values['snr'] == 17
    assert acoustic_statistics(times, np.zeros(30), times, times, [0, 3])['snr'] is None
    windows = validation_windows(3000, [(0, 3000)], [(350, 351)])
    assert len(windows) == 12
    assert all(a['input_window'][1] <= b['input_window'][0] for a, b in zip(windows, windows[1:]))


@pytest.mark.parametrize('mode', ['regular', 'vad'])
@pytest.mark.parametrize('shift', [-2, 2])
def test_boundary_shift_actually_translates_interior_grid(mode, shift):
    wave = np.zeros(100 * 16000)
    base = make_layout(wave, 100, 20, 2, mode, [(0, 100)])
    changed = make_layout(wave, 100, 20, 2, mode, [(0, 100)], shift)
    before = [c['core_end'] for c in base[:-1]]
    after = [c['core_end'] for c in changed[:-1]]
    assert before != after
    assert all(cut + shift in after for cut in before if 0 < cut + shift < 100)
    assert changed[0]['core_start'] == 0 and changed[-1]['core_end'] == 100
    assert all(c['end'] - c['start'] <= 20 + 1e-6 for c in changed)
    assert all(a['core_end'] == b['core_start'] for a, b in zip(changed, changed[1:]))


@pytest.mark.parametrize('shift', [-2, 2])
def test_historical_grid_shift_moves_inputs_and_keeps_limits(shift):
    from lecture_recognition.channel_utility import shift_layout

    old = [{'start': 0, 'end': 16, 'core_start': 0, 'core_end': 15},
           {'start': 14, 'end': 34, 'core_start': 15, 'core_end': 33},
           {'start': 32, 'end': 40, 'core_start': 33, 'core_end': 40}]
    new = shift_layout(old, 40, 20, 1, shift)
    cuts = [c['core_end'] for c in new[:-1]]
    assert 15 + shift in cuts and 33 + shift in cuts
    assert any(c['start'] not in {x['start'] for x in old} for c in new)
    assert all(c['end'] - c['start'] <= 20 for c in new)
    assert new[0]['core_start'] == 0 and new[-1]['core_end'] == 40
