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
