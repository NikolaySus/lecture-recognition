import numpy as np
import pytest

from lecture_recognition.timeline import Chunk, lecturer_regions, make_chunks, merge_words, stamp, to_srt


def test_global_speaker_union_and_overlap():
    segments = [
        {"Start": 0, "End": 2, "Speaker": 0},
        {"Start": 1, "End": 8, "Speaker": 1},
        {"Start": 2, "End": 4, "Speaker": 1},
        {"Start": 8, "End": 9, "Speaker": 0},
    ]
    speaker, regions, totals = lecturer_regions(segments, 10)
    assert speaker == 1
    assert totals[1] == 7
    assert regions == [(1, 8)]  # overlaps kept, student-only padding prohibited


def test_silence_and_short_turn():
    assert lecturer_regions([], 5)[0] is None
    _, regions, _ = lecturer_regions([{"Start": 1, "End": 1.05, "Speaker": 0}], 2)
    assert regions == [(0.8, 1.25)]


def test_no_bridging_student_turn():
    segments = [
        {"Start": 0, "End": 3, "Speaker": 0},
        {"Start": 3.1, "End": 3.2, "Speaker": 1},
        {"Start": 3.3, "End": 5, "Speaker": 0},
    ]
    _, regions, _ = lecturer_regions(segments, 5)
    assert regions == [(0, 3.1), (3.2, 5)]
    assert lecturer_regions(segments, 5, max_gap=120)[1] == regions


def test_long_chunks_join_pauses_but_stop_at_student():
    segments = [
        {"Start": 0, "End": 10, "Speaker": 0},
        {"Start": 15, "End": 100, "Speaker": 0},
        {"Start": 105, "End": 240, "Speaker": 0},
        {"Start": 241, "End": 244, "Speaker": 1},
        {"Start": 245, "End": 280, "Speaker": 0},
    ]
    _, regions, _ = lecturer_regions(segments, 280, max_gap=120)
    assert regions == [(0, 240.2), (244.8, 280)]
    audio = np.ones(280 * 16000, dtype=np.float32)
    audio[115 * 16000 : 116 * 16000] = 0
    chunks = make_chunks(audio, regions, max_core=118)
    assert 116 <= chunks[0].end <= 117
    assert all(c.end - c.start <= 120 for c in chunks)
    assert all(c.end <= 240.2 or c.start >= 244.8 for c in chunks)
    assert chunks[-1].end == 280


@pytest.mark.parametrize("seconds", [0.2, 28, 28.01, 60, 89.37])
def test_chunk_coverage_tail_and_context(seconds):
    audio = np.ones(round(seconds * 16000), dtype=np.float32)
    chunks = make_chunks(audio, [(0, seconds)])
    assert chunks[0].core_start == 0
    assert chunks[-1].core_end == seconds
    assert all(c.end - c.start <= 30 for c in chunks)
    assert all(a.core_end == b.core_start for a, b in zip(chunks, chunks[1:]))


def test_cut_prefers_relative_silence():
    audio = np.ones(60 * 16000, dtype=np.float32)
    audio[24 * 16000 : 25 * 16000] = 0.001
    chunks = make_chunks(audio, [(0, 60)])
    assert 24 <= chunks[0].core_end <= 25


def test_repetition_preserved_and_overlap_deduplicated():
    c = Chunk(0, 2.5, 0, 1.5).dict()
    items = [
        {
            "chunk": c,
            "words": [
                {"text": "да", "start_time": 1, "end_time": 1.3},
                {"text": "да", "start_time": 1.39, "end_time": 1.59},
            ],
        }
    ]
    items.append(
        {
            "chunk": Chunk(0.5, 4, 1.5, 4).dict(),
            "words": [{"text": "да", "start_time": 0.92, "end_time": 1.12}],
        }
    )
    assert len(merge_words(items)) == 2


def test_original_timeline_and_srt_gap():
    words = [{"start": 50, "end": 51, "text": "Первая"}, {"start": 60, "end": 61, "text": "вторая."}]
    srt = to_srt(words, [(49, 52), (59, 62)])
    assert "00:00:50,000 --> 00:00:51,000" in srt
    assert "00:01:00,000 --> 00:01:01,000" in srt
    assert stamp(3600.001) == "01:00:00,001"


def test_boundary_alignment_jitter_does_not_drop_word():
    items = [
        {"chunk": Chunk(0, 3, 0, 2).dict(), "words": [{"text": "слово", "start_time": 1.8, "end_time": 2.3}]},
        {"chunk": Chunk(1, 4, 2, 4).dict(), "words": [{"text": "слово", "start_time": 0.7, "end_time": 1.2}]},
    ]
    assert len(merge_words(items)) == 1


def test_srt_layout_and_duration():
    words = [{"text": "лекция", "start": i * 0.4, "end": i * 0.4 + 0.3} for i in range(50)]
    srt = to_srt(words, [(0, 30)])
    for block in srt.strip().split("\n\n"):
        lines = block.splitlines()
        assert len(lines) <= 4
        assert all(len(line) <= 42 for line in lines[2:])


def test_invalid_alignment_rejected():
    with pytest.raises(RuntimeError, match="invalid"):
        merge_words(
            [{"chunk": Chunk(0, 1, 0, 1).dict(), "words": [{"text": "test", "start_time": 2, "end_time": 3}]}]
        )


def seam_fixture(right_start=171.25):
    def item(chunk, words):
        return {'chunk': chunk.dict(), 'words': [
            {'text': text, 'start_time': start - chunk.start, 'end_time': end - chunk.start}
            for text, start, end in words]}
    return [
        item(Chunk(153.55, 171.01, 154.55, 170.01),
             [('серьезный', 168, 168.8), ('объем', 168.91, 169.31), ('данных', 169.31, 169.79)]),
        item(Chunk(169.01, 186.64, 170.01, 185.64),
             [('объем', right_start, right_start + .4), ('данных', right_start + .4, right_start + .72),
              ('объем', 174, 174.3), ('выборки', 174.3, 174.7)])]


def test_shifted_overlap_phrase_repaired_without_mutating_input():
    import copy
    items = seam_fixture()
    original = copy.deepcopy(items)
    audit = []
    result = merge_words(items, seam_audit=audit)
    assert ' '.join(w['text'] for w in result) == 'серьезный объем данных объем выборки'
    assert len(audit) == 1
    assert audit[0]['kept_chunk'] == 0
    assert result[1]['start'] == pytest.approx(168.91)
    assert items == original
    assert merge_words(items) == result


@pytest.mark.parametrize('variant', ['distant', 'both_anchored', 'gap', 'single'])
def test_ambiguous_seams_not_lexically_removed(variant):
    items = seam_fixture(175 if variant == 'distant' else 170.1 if variant == 'both_anchored' else 171.25)
    if variant == 'gap':
        items[1]['chunk']['core_start'] += 1
    elif variant == 'single':
        items[0]['words'].pop()
        items[1]['words'].pop(1)
    audit = []
    merge_words(items, seam_audit=audit)
    assert audit == []


def test_shifted_compound_suffix_keeps_right_hand_alignment():
    items = [
        {'chunk': Chunk(0, 23.61, 0, 22.61).dict(), 'words': [
            {'text': 'первый вопрос', 'start_time': 19, 'end_time': 20},
            {'text': 'и второй вопрос', 'start_time': 20.8, 'end_time': 21.44}]},
        {'chunk': Chunk(21.61, 40, 22.61, 40).dict(), 'words': [
            {'text': 'и', 'start_time': 1.04, 'end_time': 1.36},
            {'text': 'второй', 'start_time': 1.44, 'end_time': 1.76},
            {'text': 'вопрос', 'start_time': 1.76, 'end_time': 2.24}]}]
    audit = []
    result = merge_words(items, seam_audit=audit)
    assert ' '.join(w['text'] for w in result) == 'первый вопрос и второй вопрос'
    assert audit[0]['kept_chunk'] == 1


def test_partial_compound_duplicate_retains_unmatched_words_and_interval():
    items = seam_fixture()
    items[1]['words'][1]['text'] = 'данных объем'
    items[1]['words'].pop(2)
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result) == 'серьезный объем данных объем выборки'
    remaining = [w for w in result if w['text'] == 'объем'][-1]
    assert remaining['start'] == pytest.approx(171.65)
    assert remaining['end'] == pytest.approx(171.97)


@pytest.mark.parametrize('compound_left', [False, True])
def test_r006_shifted_phrase_with_compound_and_unowned_leading_fragment(compound_left):
    left = ([{'text': 'большим для того чтобы', 'start_time': 16.08, 'end_time': 16.32}]
            if compound_left else [
                {'text': 'большим', 'start_time': 15.76, 'end_time': 15.79},
                {'text': 'для', 'start_time': 15.82, 'end_time': 15.86},
                {'text': 'того', 'start_time': 15.89, 'end_time': 15.92},
                {'text': 'чтобы', 'start_time': 15.92, 'end_time': 16.48}])
    items = [
        {'chunk': Chunk(169.01, 186.64, 170.01, 185.64).dict(), 'words': left},
        {'chunk': Chunk(184.64, 202.92, 185.64, 201.92).dict(), 'words': [
            {'text': 'ши', 'start_time': .32, 'end_time': .72},
            {'text': 'для', 'start_time': .72, 'end_time': 1.6},
            {'text': 'того', 'start_time': 1.6, 'end_time': 2.24},
            {'text': 'чтобы отследить', 'start_time': 2.24, 'end_time': 2.8},
            {'text': 'изменение', 'start_time': 3, 'end_time': 3.5}]}]
    audit = []
    result = merge_words(items, seam_audit=audit)
    assert ' '.join(w['text'] for w in result) == 'большим для того чтобы отследить изменение'
    assert audit[0]['phrase'] == 'для того чтобы'
    word = next(w for w in result if w['text'] == 'отследить')
    assert (word['start'], word['end']) == pytest.approx((186.88, 187.44))


def test_coincident_compound_phrase_removed_once_across_seam():
    items = [
        {'chunk': Chunk(0, 5, 0, 4).dict(), 'words': [
            {'text': 'методы и алгоритмы', 'start_time': 3.7, 'end_time': 4.3}]},
        {'chunk': Chunk(3, 8, 4, 8).dict(), 'words': [
            {'text': 'и', 'start_time': .8, 'end_time': 1},
            {'text': 'алгоритмы', 'start_time': 1, 'end_time': 1.3},
            {'text': 'анализа', 'start_time': 1.4, 'end_time': 1.8}]}]
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result) == 'методы и алгоритмы анализа'


def test_earlier_real_phrase_repeat_survives_local_seam_repair():
    items = seam_fixture()
    items[0]['words'][:0] = [
        {'text': 'объем', 'start_time': 10, 'end_time': 10.3},
        {'text': 'данных', 'start_time': 10.3, 'end_time': 10.6}]
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result).count('объем данных') == 2


def test_repeated_phrase_within_shared_audio_is_not_collapsed():
    items = seam_fixture()
    items[0]['words'][:0] = [
        {'text': 'объем', 'start_time': 15.5, 'end_time': 15.6},
        {'text': 'данных', 'start_time': 15.6, 'end_time': 15.7}]
    audit = []
    merge_words(items, seam_audit=audit)
    assert not audit


def test_shifted_single_word_from_collapsed_compound_unit():
    items = [
        {'chunk': Chunk(93.15, 109.67, 94.15, 108.67).dict(), 'words': [
            {'text': 'временных собственно', 'start_time': 13.92, 'end_time': 14.24}]},
        {'chunk': Chunk(107.67, 125.25, 108.67, 124.25).dict(), 'words': [
            {'text': 'собственно', 'start_time': 1.6, 'end_time': 1.92},
            {'text': 'говоря', 'start_time': 1.92, 'end_time': 2.16}]}]
    assert ' '.join(w['text'] for w in merge_words(items)) == 'временных собственно говоря'


def test_stretched_word_in_overlap_does_not_duplicate_phrase():
    items = [
        {'chunk': Chunk(113.37, 132.81, 115.37, 130.81).dict(), 'words': [
            {'text': 'еще', 'start_time': 15.2, 'end_time': 15.44},
            {'text': 'раз', 'start_time': 15.92, 'end_time': 18.48}]},
        {'chunk': Chunk(128.81, 147.13, 130.81, 145.13).dict(), 'words': [
            {'text': 'еще', 'start_time': 2.56, 'end_time': 2.8},
            {'text': 'раз', 'start_time': 2.8, 'end_time': 3.2},
            {'text': 'повторюсь', 'start_time': 3.2, 'end_time': 3.7}]}]
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result) == 'еще раз повторюсь'
    assert result[0]['start'] == pytest.approx(131.37)


def test_matching_phrase_does_not_promote_unowned_conflicting_context():
    items = [
        {'chunk': Chunk(102.41, 113.69, 103.41, 112.69).dict(), 'words': [
            {'text': 'встраивание', 'start_time': 9.28, 'end_time': 10.08},
            {'text': 'если', 'start_time': 10.08, 'end_time': 10.4},
            {'text': 'угодно', 'start_time': 10.4, 'end_time': 10.88}]},
        {'chunk': Chunk(111.69, 122.14, 112.69, 121.14).dict(), 'words': [
            {'text': 'выстраивание если', 'start_time': .8, 'end_time': 1.12},
            {'text': 'угодно', 'start_time': 1.12, 'end_time': 1.6},
            {'text': 'подобного', 'start_time': 1.76, 'end_time': 2.56}]}]
    assert ' '.join(w['text'] for w in merge_words(items)) == 'встраивание если угодно подобного'


def test_complete_word_recovered_when_both_ownership_tests_reject_it():
    items = [
        {'chunk': Chunk(0, 16.65, 0, 15.65).dict(), 'words': [
            {'text': 'записать', 'start_time': 15.2, 'end_time': 15.52},
            {'text': 'существует', 'start_time': 15.68, 'end_time': 16.32},
            {'text': 'ли', 'start_time': 16.32, 'end_time': 16.4}]},
        {'chunk': Chunk(14.65, 31.68, 15.65, 30.68).dict(), 'words': [
            {'text': 'асуществует', 'start_time': 0, 'end_time': 1.68},
            {'text': 'ли', 'start_time': 1.68, 'end_time': 1.76},
            {'text': 'закономерность', 'start_time': 1.76, 'end_time': 2.56}]}]
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result) == 'записать существует ли закономерность'
    assert (result[1]['start'], result[1]['end']) == pytest.approx((15.68, 16.32))
    items[1]['words'][1]['text'] = 'другая'
    assert 'существует' not in ' '.join(w['text'] for w in merge_words(items))


def test_orphan_compound_keeps_order_despite_anchor_timestamp_jitter():
    items = [
        {'chunk': Chunk(215.27, 231.78, 216.27, 230.78).dict(), 'words': [
            {'text': 'временного', 'start_time': 14, 'end_time': 14.72},
            {'text': 'ряда а', 'start_time': 16.08, 'end_time': 16.32},
            {'text': 'поскол', 'start_time': 16.32, 'end_time': 16.48}]},
        {'chunk': Chunk(229.78, 249.55, 230.78, 248.55).dict(), 'words': [
            {'text': 'а', 'start_time': 1.52, 'end_time': 1.84},
            {'text': 'поскольку', 'start_time': 1.84, 'end_time': 2.24}]}]
    result = merge_words(items)
    assert ' '.join(w['text'] for w in result) == 'временного ряда а поскольку'
    assert (result[1]['start'], result[1]['end']) == pytest.approx((231.35, 231.59))


def test_clipped_word_fragment_requires_two_independent_overlap_anchors():
    items = [
        {'chunk': Chunk(184.64, 202.92, 185.64, 201.92).dict(), 'words': [
            {'text': 'выбирался', 'start_time': 16.16, 'end_time': 16.96},
            {'text': 'процесс', 'start_time': 17.6, 'end_time': 18},
            {'text': 'пер', 'start_time': 18.08, 'end_time': 18.24}]},
        {'chunk': Chunk(200.92, 217.27, 201.92, 216.27).dict(), 'words': [
            {'text': 'сяпроцент', 'start_time': 0, 'end_time': 1.76},
            {'text': 'перекрытия', 'start_time': 1.76, 'end_time': 2.64}]}]
    assert ' '.join(w['text'] for w in merge_words(items)) == 'выбирался процент перекрытия'
    items[0]['words'][-1]['text'] = 'другая'
    assert 'процент' not in ' '.join(w['text'] for w in merge_words(items))


def test_phrase_deduplication_preserves_unowned_compound_prefix():
    items = [
        {'chunk': Chunk(215.27, 231.78, 216.27, 230.78).dict(), 'words': [
            {'text': 'временного', 'start_time': 14, 'end_time': 14.72},
            {'text': 'ряда а', 'start_time': 16.08, 'end_time': 16.32},
            {'text': 'поскольку', 'start_time': 16.32, 'end_time': 16.48}]},
        {'chunk': Chunk(229.78, 249.55, 230.78, 248.55).dict(), 'words': [
            {'text': 'а', 'start_time': 1.52, 'end_time': 1.84},
            {'text': 'поскольку', 'start_time': 1.84, 'end_time': 2.24},
            {'text': 'мы', 'start_time': 2.24, 'end_time': 2.56}]}]
    audit = []
    result = merge_words(items, seam_audit=audit)
    assert ' '.join(w['text'] for w in result) == 'временного ряда а поскольку мы'
    assert (result[1]['start'], result[1]['end']) == pytest.approx((231.35, 231.59))
    assert audit[0]['preserved_prefix'] == 'ряда'


def test_complete_word_rescue_with_attached_prefix_and_two_anchors():
    items = [
        {'chunk': Chunk(184.64, 202.92, 185.64, 201.92).dict(), 'words': [
            {'text': 'выбирался', 'start_time': 16.16, 'end_time': 16.96},
            {'text': 'процент', 'start_time': 17.6, 'end_time': 18.08},
            {'text': 'пере', 'start_time': 18.08, 'end_time': 18.24}]},
        {'chunk': Chunk(200.92, 217.27, 201.92, 216.27).dict(), 'words': [
            {'text': 'спроцент', 'start_time': 0, 'end_time': 1.76},
            {'text': 'перекрытия', 'start_time': 1.76, 'end_time': 2.64}]}]
    audit = []
    assert ' '.join(w['text'] for w in merge_words(items, seam_audit=audit)) == 'выбирался процент перекрытия'
    assert audit[0]['reason'] == 'recovered_prefixed_copy'
    items[0]['words'][-1]['text'] = 'другая'
    assert 'процент' not in ' '.join(w['text'] for w in merge_words(items))
    items[0]['words'][-1]['text'] = 'пере'
    items[0]['words'][0].update(start_time=15, end_time=15.3)
    assert 'процент' not in ' '.join(w['text'] for w in merge_words(items))
