import pytest

from lecture_recognition.gibbs_completeness import completeness_decisions, length


def select(left, right, policy='symmetric_guard', metric='letters', allowance=0., score=(.2, .8)):
    rows = [{'same_text': left == right, 'core': list(score)}]
    return completeness_decisions(rows, [{'text': left}], [{'text': right}], policy, metric, allowance)[0]


def test_letter_count_ignores_punctuation_spaces_digits_and_normalizes_unicode():
    assert length('Эйлеру, 123! Euler.', 'letters') == 11
    assert length('е\u0308', 'letters') == 1
    assert length('', 'letters') == 0


def test_euler_omission_is_vetoed_despite_higher_gibbs_confidence():
    assert select('математику элеру где', 'математику где')['channel'] == 'left'


def test_letter_guard_keeps_nonstationary_recovery_but_word_guard_can_block_it():
    left, right = 'из стационарных данных', 'нестационарных данных'
    assert length(left, 'letters') == length(right, 'letters')
    assert select(left, right)['channel'] == 'right'
    assert select(left, right, metric='words')['channel'] == 'left'


def test_allowance_boundary_is_inclusive():
    assert select('а' * 100, 'а' * 95, allowance=.05)['channel'] == 'right'
    assert select('а' * 100, 'а' * 94, allowance=.05)['channel'] == 'left'


def test_guard_works_symmetrically_when_gibbs_prefers_shorter_left():
    selected = select('слово', 'слово повтор', score=(.8, .2))
    assert selected['channel'] == 'right'
    assert selected['gibbs_channel'] == 'left'
    assert selected['reason'] == 'shorter_candidate_veto'


def test_missing_confidence_retains_safe_left_but_pure_length_is_a_separate_control():
    assert select('слово', 'слово повтор', score=(None, .8))['channel'] == 'left'
    assert select('', 'слово', policy='pure_longer', score=(None, .8))['channel'] == 'right'


def test_equal_text_and_length_ties_do_not_force_a_different_channel():
    assert select('текст', 'текст')['channel'] == 'left'
    assert select('кот', 'кит')['channel'] == 'right'
    assert select('кот', 'кит', policy='pure_longer')['channel'] == 'left'


@pytest.mark.parametrize('allowance', [-.1, 1., float('nan')])
def test_invalid_allowance_rejected(allowance):
    with pytest.raises(ValueError):
        select('а', 'б', allowance=allowance)
