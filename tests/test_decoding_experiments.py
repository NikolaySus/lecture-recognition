import pytest

from lecture_recognition.decoding_experiments import (
    combine,
    config,
    edit_score,
    generation_kwargs,
    initial_matrix,
    normalize,
    regression,
    score_case,
)
from lecture_recognition.experiments import identity


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("1024 отсчёта", "тысяча двадцать четыре отсчета"),
        ("95–98%", "девяносто пять девяносто восемь процентов"),
        ("20-30 отсчётов", "двадцать тридцать отсчетов"),
        ("5 лет", "пять лет"),
    ],
)
def test_numbers(left, right):
    assert normalize(left) == normalize(right)


def test_do_not_normalize_away_errors():
    assert normalize("95 или 96") != normalize("95–98")
    assert normalize("124") != normalize("1024")
    assert normalize("нестационарных") != normalize("стационарных")
    assert normalize("не было") != normalize("было")
    assert normalize("квази стационарности") != normalize("квазистационарности")
    assert normalize("очень очень э") == ["очень", "очень", "э"]


def test_overlap_preserves_one_copy_and_card_spans():
    words, spans = combine({"a": "Первый вопрос. Это важно", "b": "Это важно. Дальше"}, ["a", "b"])
    assert words == ["Первый", "вопрос.", "Это", "важно", "Дальше"]
    assert spans == {"a": [0, 4], "b": [2, 5]}
    with pytest.raises(ValueError, match="overlap"):
        combine({"a": "Один текст", "b": "Совсем другой"}, ["a", "b"])


def test_editorial_punctuation_does_not_shift_reference_word_spans():
    words, spans = combine({"a": "Рекомендую записать . Это важно", "b": "Это важно. Дальше"}, ["a", "b"])
    assert words == ["Рекомендую", "записать", "Это", "важно", "Дальше"]
    assert spans["b"] == [2, 5]


def test_edit_counts_and_protected_positions():
    score = edit_score("не было 1024 отсчета", "было 124 отсчета вчера")
    assert (score["S"], score["D"], score["I"]) == (1, 1, 1)
    assert score["protected"] == [0, 2]
    assert not score["protected_correct"]


def test_fixed_windows_and_false_terms():
    reference = {
        "groups": [
            {
                "name": "one",
                "window": [1, 5],
                "text": "обычная речь",
                "cards": {"R001": {"window": [1, 5], "text": "обычная речь"}},
            }
        ],
        "target_window": [8, 12],
    }
    words = [
        {"start": 2, "end": 3, "text": "обычная речь квазистационарности"},
        {"start": 9, "end": 10, "text": "квази стационарности"},
        {"start": 145, "end": 147, "text": "квазистационарности"},
    ]
    score = score_case(words, reference)
    assert score["total"]["false_quasi"] == 1
    assert score["total"]["I"] == 1
    assert score["target"]["first"]["exact"]
    assert score["target"]["second"]["split"]
    assert not score["target"]["second"]["exact"]


def test_regression_rejects_lost_correct_number():
    def score(text):
        return {"total": {"false_quasi": 0}, "groups": {"x": edit_score("1024 нестационарных", text)}}

    assert regression(score("124 нестационарных"), score("1024 нестационарных"))
    assert not regression(score("1024 нестационарных"), score("124 нестационарных"))


def test_generation_parameters_and_unique_bias():
    class Tokenizer:
        def encode(self, text, add_special_tokens):
            assert not add_special_tokens
            # Duplicate prefix variants must not accumulate bias.
            return list(text.strip().encode("utf-8"))

    kwargs = generation_kwargs(config("dictionary", 4, 0.8, 0.5), Tokenizer())
    assert kwargs["num_beams"] == 4
    assert kwargs["length_penalty"] == 0.8
    assert kwargs["do_sample"] is False
    assert set(kwargs["sequence_bias"].values()) == {0.5}
    assert "length_penalty" not in generation_kwargs(config(), Tokenizer())


def test_matrix_and_cache_separation():
    assert len(initial_matrix()) == 18
    configs = initial_matrix() + [config(bias=0.5), config(batch=2), config(layout="90s"), config(repeat=1)]
    assert len({identity(c) for c in configs}) == len(configs)
