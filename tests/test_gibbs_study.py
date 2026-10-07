import numpy as np
import pytest

from lecture_recognition.channel_utility import word_confidence
from lecture_recognition.gibbs_study import decisions, disagreement_words, local_confidence


def trace():
    return {'log_probs': np.log([[.8, .1, .1], [.7, .2, .1], [.1, .1, .8],
                                 [.1, .8, .1], [.8, .1, .1], [.1, .8, .1]]),
            'tokens': np.array([0, 0, 2, 1, 0, 1]), 'token_words': np.array([0, 0, 0, 1, 2, 3]),
            'times': np.array([0., 1., 2., 3., 4., 5.]), 'blank': 2}


def test_full_confidence_reproduces_old_aggregation():
    t = trace()
    assert local_confidence(t, {}, 'full') == word_confidence(
        np.exp(t['log_probs']), t['tokens'], t['token_words'], 2, ctc=True)['gibbs']


def test_core_ignores_blank_and_context_even_if_they_are_certain():
    t = trace()
    chunk = {'core_start': 3., 'core_end': 6.}
    first = local_confidence(t, chunk, 'core')
    t['log_probs'][:3] = np.log([[.99, .005, .005]] * 3)
    assert local_confidence(t, chunk, 'core') == first
    assert local_confidence(t, {'core_start': 2., 'core_end': 5.}, 'core') is None


def test_disagreement_restricts_words_and_requires_three_supported_words():
    t = trace()
    chunk = {'core_start': 0., 'core_end': 6.}
    assert local_confidence(t, chunk, 'disagreement', {1, 2}) is None
    assert local_confidence(t, chunk, 'disagreement', {1, 2, 3}) is not None


@pytest.mark.parametrize(('left', 'right', 'expected'), [
    ('а б в г д', 'а б ж г д', ({1, 2, 3}, {1, 2, 3})),
    ('а б в г', 'а б ж в г', ({1, 2}, {1, 2, 3})),
    ('а б ж в г', 'а б в г', ({1, 2, 3}, {1, 2})),
    ('а а б а', 'а а б а', (set(), set())),
    ('а б а б в', 'а б а г в', ({2, 3, 4}, {2, 3, 4})),
])
def test_disagreement_spans_and_neighbours(left, right, expected):
    assert disagreement_words(left, right) == list(expected)


def test_independent_rule_does_not_hold_an_unsupported_right_channel():
    rows = [{'same_text': False, 'core': [.2, .8]}, {'same_text': False, 'core': [.5, .51]}]
    assert [d['channel'] for d in decisions(rows, 'core', 'independent', .02)] == ['right', 'left']
    assert [d['channel'] for d in decisions(rows, 'core', 'hold', .02)] == ['right', 'right']


def test_equal_text_or_insufficient_support_resets_hold_to_left():
    rows = [{'same_text': False, 'core': [.2, .8]}, {'same_text': True, 'core': [.2, .8]},
            {'same_text': False, 'core': [.2, .8]}, {'same_text': False, 'core': [None, .8]}]
    assert [d['channel'] for d in decisions(rows, 'core', 'hold', .02)] == ['right', 'left', 'right', 'left']


def test_threshold_is_strict_and_equal_confidence_prefers_left():
    rows = [{'same_text': False, 'full': [0., .5]}, {'same_text': False, 'full': [.5, .5]}]
    assert [d['channel'] for d in decisions(rows, 'full', 'independent', .5)] == ['left', 'left']
