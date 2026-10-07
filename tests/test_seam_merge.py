import copy

import pytest

from lecture_recognition.seam_merge import merge_overlap_words
from lecture_recognition.timeline import Chunk


def item(chunk, units):
    return {'chunk': chunk.dict(), 'words': [
        {'text': text, 'start_time': start - chunk.start, 'end_time': end - chunk.start}
        for text, start, end in units]}


@pytest.mark.parametrize('ending', ['пер', 'первое'])
def test_full_overlap_match_handles_interior_phrase_and_terminal_disagreement(ending):
    aligned = [item(Chunk(0, 20, 0, 18), [('существует', 15.68, 16.32), ('ли', 16.32, 16.48),
               ('закономерность', 16.48, 17.36), ('в принципе', 17.36, 17.68), (f'это {ending}', 17.92, 18)]),
               item(Chunk(16, 36, 18, 34), [('закономерность в принципе это первый', 18.08, 20.08),
                                          ('вопрос', 20.08, 20.8), ('и второй вопрос', 22, 23)])]
    original = copy.deepcopy(aligned)
    audit = []
    result = merge_overlap_words(aligned, seam_audit=audit)
    assert ' '.join(w['text'] for w in result) == 'существует ли закономерность в принципе это первый вопрос и второй вопрос'
    assert aligned == original
    compound = next(w for w in result if w['text'].startswith('закономерность'))
    assert (compound['start'], compound['end']) == (18.08, 20.08)
    assert any(a['reason'] == 'contextual_overlap_copy' for a in audit)


def test_disjoint_phrase_copies_have_independent_context_anchors():
    aligned = [item(Chunk(30, 50, 32, 48), [('важная', 46, 46.4), ('две', 46.48, 46.64),
               ('статистики', 46.64, 47.52), ('а именно', 47.52, 47.6), ('статистика', 49.52, 50)]),
               item(Chunk(46, 66, 48, 64), [('важные', 46, 46.4), ('две', 46.4, 46.64),
               ('статистики', 46.64, 47.12), ('а именно', 48.8, 49.04),
               ('статистики', 49.52, 50.32), ('посвященные', 51.68, 52.48)])]
    text = ' '.join(w['text'] for w in merge_overlap_words(aligned))
    assert text == 'важная две статистики а именно статистики посвященные'


@pytest.mark.parametrize('phrase', ['очень важная', 'для того чтобы', 'закономерность в принципе это первый'])
def test_disjoint_real_repetition_is_not_removed_on_lexical_similarity_alone(phrase):
    aligned = [item(Chunk(0, 12, 0, 10), [(phrase, 8.2, 9.7)]),
               item(Chunk(8, 20, 10, 20), [(phrase, 10.1, 11.7)])]
    result = merge_overlap_words(aligned)
    assert ' '.join(w['text'] for w in result) == phrase + ' ' + phrase


@pytest.mark.parametrize('left,right', [('семь', 'семнадцать'), ('нестационарных', 'нестационарности')])
def test_disagreeing_numbers_and_negations_are_not_rewritten_as_clipped_tails(left, right):
    aligned = [item(Chunk(0, 12, 0, 10), [('в данном случае', 8.2, 9.1), (left, 9.2, 9.7)]),
               item(Chunk(8, 20, 10, 20), [(f'в данном случае {right}', 10.1, 11.7), ('данных', 12, 12.5)])]
    result = ' '.join(w['text'] for w in merge_overlap_words(aligned))
    assert left in result and right in result


def test_repeated_phrase_twice_in_both_inputs_preserves_both_occurrences():
    units = [('очень', 8.2, 8.5), ('важная', 8.5, 9), ('очень', 10.2, 10.5), ('важная', 10.5, 11)]
    aligned = [item(Chunk(0, 12, 0, 10), units), item(Chunk(8, 20, 10, 20), units)]
    assert ' '.join(w['text'] for w in merge_overlap_words(aligned)) == 'очень важная очень важная'


def test_no_audio_overlap_cannot_trigger_lexical_reconciliation():
    phrase = 'для того чтобы'
    aligned = [item(Chunk(0, 10, 0, 10), [(phrase, 8, 9)]),
               item(Chunk(10, 20, 10, 20), [(phrase, 11, 12)])]
    assert ' '.join(w['text'] for w in merge_overlap_words(aligned)) == phrase + ' ' + phrase


def test_conflicting_complete_endings_are_not_erased_without_anchors():
    aligned = [item(Chunk(0, 20, 0, 18), [('закономерность', 16.4, 17),
                ('в принципе', 17, 17.3), ('это первое', 17.4, 18)]),
               item(Chunk(16, 36, 18, 34), [('закономерность в принципе это первый', 18.1, 20.1),
                                          ('вопрос', 20.2, 20.8)])]
    result = ' '.join(w['text'] for w in merge_overlap_words(aligned))
    assert 'это первое' in result and 'это первый' in result


def test_valid_short_function_word_is_not_treated_as_a_clipped_long_word():
    aligned = [item(Chunk(0, 20, 0, 18), [('в данном', 16.4, 17), ('случае', 17, 17.3), ('это про', 17.92, 18)]),
               item(Chunk(16, 36, 18, 34), [('в данном случае это прогноз', 18.1, 20.1), ('данных', 20.2, 20.8)])]
    result = ' '.join(w['text'] for w in merge_overlap_words(aligned))
    assert 'это про ' in result and 'это прогноз' in result
