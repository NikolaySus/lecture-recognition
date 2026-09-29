import numpy as np
import pytest

from lecture_recognition.audio import RATE, masked_audio
from lecture_recognition.timeline import (
    Chunk,
    make_packed_chunks,
    packing_size,
    split_chunk,
    student_only_regions,
    to_srt,
)


@pytest.mark.parametrize("seconds", [0.2, 59.99, 60, 118, 118.001, 119, 120, 121, 236, 237, 1000])
def test_strict_packing_and_tail(seconds):
    size, count, _, _ = packing_size(round(seconds * RATE))
    audio = np.zeros(size, dtype=np.float32)
    chunks = make_packed_chunks(audio, [(0, seconds)])
    assert len(chunks) == count
    assert chunks[0].core_start == 0
    assert chunks[-1].core_end == size / RATE
    assert all(60 - 1e-9 <= c.core_end - c.core_start <= 118 + 1e-9 for c in chunks)
    assert all(c.end - c.start <= 120 + 1e-9 for c in chunks)
    assert all(a.core_end == b.core_start for a, b in zip(chunks, chunks[1:]))


def test_student_mask_keeps_overlap_and_source(tmp_path):
    segments = [
        {"Start": 0, "End": 2, "Speaker": 0},
        {"Start": 1, "End": 4, "Speaker": 1},
        {"Start": 3, "End": 5, "Speaker": 0},
    ]
    excluded = student_only_regions(segments, 5, 0)
    assert excluded == [(2, 3)]
    original = np.ones(5 * RATE, dtype=np.float32)
    target = tmp_path / "masked.f32"
    result = masked_audio(original, target, excluded, 60 * RATE)
    assert np.all(original == 1)
    assert np.all(result[: 2 * RATE] == 1)
    assert np.all(result[2 * RATE : 3 * RATE] == 0)
    assert np.all(result[3 * RATE : 5 * RATE] == 1)
    assert np.all(result[5 * RATE :] == 0)
    assert not target.with_suffix(".tmp").exists()
    timestamp = target.stat().st_mtime_ns
    assert len(masked_audio(original, target, excluded, 60 * RATE)) == 60 * RATE
    assert target.stat().st_mtime_ns == timestamp


def test_silence_only_cores_skipped():
    chunks = make_packed_chunks(np.zeros(360 * RATE), [(0, 1), (359, 360)])
    assert len(chunks) == 2
    assert chunks[0].core_start == 0
    assert chunks[-1].core_end == 360


def test_srt_clips_excluded_speech_and_ignores_padding():
    words = [
        {"start": 1, "end": 2.1, "text": "before"},
        {"start": 2.3, "end": 2.8, "text": "student"},
        {"start": 2.9, "end": 4, "text": "after"},
        {"start": 7, "end": 8, "text": "padding"},
    ]
    srt = to_srt(words, [(0, 2), (3, 5)])
    assert "00:00:01,000 --> 00:00:02,000" in srt
    assert "00:00:03,000 --> 00:00:04,000" in srt
    assert "student" not in srt and "padding" not in srt


def test_emergency_split_does_not_break_minimum():
    audio = np.zeros(122 * RATE)
    with pytest.raises(RuntimeError, match="preserving 60s"):
        split_chunk(audio, Chunk(0, 118, 0, 118), min_core=60)
    children = split_chunk(audio, Chunk(0, 120, 0, 120), min_core=60)
    assert all(c.core_end - c.core_start == 60 for c in children)


@pytest.mark.parametrize("minimum,maximum", [(0, 118), (60, 59), (float("nan"), 118), (60, float("inf"))])
def test_invalid_limits(minimum, maximum):
    with pytest.raises(ValueError):
        packing_size(100, minimum, maximum)
