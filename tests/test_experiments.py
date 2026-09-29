import shutil

import numpy as np
import pytest

from lecture_recognition.experiments import FILTERS, evaluate, identity, preprocess


@pytest.mark.parametrize(
    "text,expected",
    [
        ("КВАЗИСТАЦИОНАРНОСТИ,", True),
        ("стационарности", False),
        ("квази стационарности", False),
        ("неквазистационарности", False),
    ],
)
def test_exact_target(text, expected):
    result = evaluate(text, [{"text": text, "start": 148, "end": 150}], "квазистационарности", [145, 155])
    assert result["pass"] == expected


@pytest.mark.parametrize(
    "start,end,expected", [(144, 146, True), (154, 156, True), (155, 157, False), (149, 149, False)]
)
def test_target_window(start, end, expected):
    word = "квазистационарности"
    assert evaluate(word, [{"text": word, "start": start, "end": end}], word, [145, 155])["pass"] == expected
    assert not evaluate("other", [{"text": word, "start": 148, "end": 150}], word, [145, 155])["pass"]


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
@pytest.mark.parametrize("name", list(FILTERS))
def test_filter_preserves_timeline_and_masks_after_processing(name):
    audio = (0.1 * np.sin(np.arange(32000) * 2 * np.pi * 440 / 16000)).astype(np.float32)
    original = audio.copy()
    result = preprocess(audio, name, [(0.5, 1)])
    assert len(result) == len(audio)
    assert np.isfinite(result).all() and np.max(np.abs(result)) < 1
    assert np.all(result[8000:16000] == 0)
    assert np.array_equal(audio, original)


def test_cache_identity_includes_settings():
    a = {"filter": "none", "window": [145, 155], "revision": "a"}
    assert identity(a) == identity(dict(reversed(list(a.items()))))
    assert identity(a) != identity(dict(a, filter="peak"))
    assert identity(a) != identity(dict(a, revision="b"))


def test_unmodified_aac_overload_is_preserved():
    wave = np.array([0, 1.07, -1.01, 0], dtype=np.float32)
    assert np.array_equal(preprocess(wave, "none", []), wave)
    assert np.max(np.abs(preprocess(wave, "peak", []))) < 1
