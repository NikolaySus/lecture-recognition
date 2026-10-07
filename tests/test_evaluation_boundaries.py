"""Card WER must not depend on internal forced-alignment boundaries."""
import pytest

from lecture_recognition.evaluation import combine, normalize, score_case


def reference(texts):
    words, spans = combine(texts, texts)
    return {'groups': [{'name': '+'.join(texts), 'text': ' '.join(words), 'window': [0, 20],
                        'cards': {k: {'text': v, 'span': spans[k], 'window': [0, 5] if k == 'A' else [4, 20]}
                                  for k, v in texts.items()}}], 'target_window': [0, 20]}


def test_internal_timestamps_and_word_units_do_not_change_card_scores():
    ref = reference({'A': 'это первый вопрос', 'B': 'первый вопрос и ответ'})
    a = [{'text': t, 'start': i, 'end': i + .2}
         for i, t in enumerate('это первый вопрос и ответ'.split(), 1)]
    b = [dict(w) for w in a]
    b[2].update(start=5.1, end=5.3)
    # Preserve order while moving words across the internal card boundary.
    b[3].update(start=6, end=6.2)
    b[4].update(start=7, end=7.2)
    grouped = [{'text': 'это первый вопрос и ответ', 'start': 1, 'end': 10}]
    scores = [score_case(w, ref, 2) for w in (a, b, grouped)]
    assert scores[0] == scores[1] == scores[2]
    assert all(c['errors'] == 0 for c in scores[0]['cards'].values())
    assert score_case(a, ref, 2, card_method='time-window-v1')['cards'] != score_case(
        b, ref, 2, card_method='time-window-v1')['cards']


@pytest.mark.parametrize('text,errors', [
    ('начало общий неверно конец', 1),
    ('начало общий конец', 1),
    ('начало общий лишнее текст конец', 1),
    ('начало общий текст общий текст конец', 2),
    ('', 4),
])
def test_real_errors_and_repetitions_are_not_hidden(text, errors):
    ref = reference({'A': 'начало общий текст', 'B': 'общий текст конец'})
    score = score_case([{'text': text, 'start': 1, 'end': 19}], ref, 2)
    assert score['total']['errors'] == errors
    assert any(c['errors'] for c in score['cards'].values())


def test_insertions_at_card_edges_have_explicit_ownership():
    ref = reference({'A': 'начало общий текст', 'B': 'общий текст конец'})
    score = score_case([{'text': 'до начало лишнее общий текст конец после', 'start': 1, 'end': 19}], ref, 2)
    assert score['cards']['A']['hypothesis'] == 'до начало лишнее общий текст'
    assert score['cards']['B']['hypothesis'] == 'лишнее общий текст конец после'
    assert score['total']['I'] == 3


def test_spoken_numbers_preserve_original_text_and_card_score():
    ref = reference({'A': 'объем 1024 отсчета', 'B': '1024 отсчета диапазон 95-98 процентов'})
    text = 'объем тысяча двадцать четыре отсчета диапазон девяносто пять тире девяносто восемь процентов'
    score = score_case([{'text': text, 'start': 1, 'end': 19}], ref, 2)
    assert score['total']['errors'] == 0
    assert all(c['errors'] == 0 for c in score['cards'].values())
    assert score['cards']['A']['hypothesis'] == 'объем тысяча двадцать четыре отсчета'
    assert 'тире' in score['cards']['B']['hypothesis']


@pytest.mark.parametrize('text', ['тысяча двадцать четыре', '95-98%', 'девяноста пяти тире девяноста восьми', 'обычный текст'])
def test_normalized_provenance_preserves_token_values(text):
    for version in (1, 2):
        tokens, spans = normalize(text, version, with_spans=True)
        assert tokens == normalize(text, version)
        assert len(spans) == len(tokens)
        assert all(a < b for a, b in spans)
