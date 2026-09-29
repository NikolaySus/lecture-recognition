from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lecture_recognition.cli import Cache
from lecture_recognition.timeline import Chunk


def test_alignment_context_respects_speaker_boundaries_and_core(monkeypatch):
    from lecture_recognition import models

    lengths = []
    words = [{"text": "word", "start_time": 1, "end_time": 2}]

    def fake_align(processor, model, wave, text, language):
        lengths.append(len(wave))
        if len(lengths) == 1:
            raise RuntimeError("Timestamp beyond audio")
        return words

    monkeypatch.setattr(models, "align_wave", fake_align)
    chunk = Chunk(10, 20, 11, 19)
    expanded, result = models.align_with_context(
        None, None, np.zeros(30 * 16000), chunk, "word", "ru", [(9.5, 21)]
    )
    assert expanded == Chunk(9.5, 21, 11, 19)
    assert lengths == [10 * 16000, int(11.5 * 16000)]
    assert result == words


def test_alignment_without_safe_context_still_fails(monkeypatch):
    from lecture_recognition import models

    def fail(*args):
        raise RuntimeError("Invalid timestamps")

    monkeypatch.setattr(models, "align_wave", fail)
    with pytest.raises(RuntimeError, match="Invalid timestamps"):
        models.align_with_context(None, None, np.zeros(16000), Chunk(0, 1, 0, 1), "word", "ru", [(0, 1)])


class Inputs(dict):
    def to(self, *args, **kwargs):
        return self


class Processor:
    def __init__(self):
        self.tokenizer = self

    def apply_transcription_request(self, audio, language):
        waves = audio if isinstance(audio, list) else [audio]
        return Inputs(
            input_ids=torch.zeros((len(waves), 1), dtype=torch.long), values=[int(w[0]) for w in waves]
        )

    def decode(self, ids, **kwargs):
        return " ".join(str(int(x)) for x in ids if int(x) not in {0, 2})


@pytest.mark.parametrize("oom", [False, True])
def test_batch_order_cache_and_oom_fallback(tmp_path, monkeypatch, oom):
    from lecture_recognition import models

    calls = []

    class Model:
        dtype = torch.bfloat16
        generation_config = SimpleNamespace(eos_token_id=2)

        def generate(self, input_ids, values, **kwargs):
            calls.append(len(values))
            if oom and len(values) > 1:
                raise torch.cuda.OutOfMemoryError("simulated batch OOM")
            return torch.tensor([[0, value, 2, 0] for value in values])

    monkeypatch.setattr(models, "load", lambda *_: (Processor(), Model()))
    monkeypatch.setattr(models, "release", lambda: None)
    cache = Cache(tmp_path)
    audio = np.repeat([11.0, 12.0, 13.0], 16000)
    chunks = [Chunk(i, i + 1, i, i + 1) for i in range(3)]
    # A partially completed batch must preserve cached rows and source order.
    cache.write("asr", chunks[1].dict(), [{"chunk": chunks[1].dict(), "text": "cached"}])
    result = models.recognize(audio, chunks, "ru", "revision", cache.read, cache.write, batch_size=3)
    assert [r["text"] for r in result] == ["11", "cached", "13"]
    assert calls == ([2, 1, 1] if oom else [2])
    monkeypatch.setattr(models, "load", lambda *_: pytest.fail("Completed stages must not load weights"))
    assert models.recognize(audio, chunks, "ru", "revision", cache.read, cache.write, batch_size=3) == result


@pytest.mark.parametrize("recover", [True, False])
def test_zero_duration_alignment_retries_without_inventing_times(recover):
    from lecture_recognition.models import align_wave

    lengths = []

    class AlignerProcessor:
        def prepare_forced_aligner_inputs(self, audio, **kwargs):
            lengths.append(len(audio))
            return Inputs(input_ids=torch.zeros((1, 1), dtype=torch.long)), [["Да"]]

        def decode_forced_alignment(self, **kwargs):
            if recover and len(lengths) == 2:
                return [[{"text": "Да", "start_time": 0.32, "end_time": 0.4}]]
            return [[{"text": "Да", "start_time": 0.16, "end_time": 0.16}]]

    class Aligner:
        dtype = torch.bfloat16
        config = SimpleNamespace(timestamp_token_id=1)

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=None)

    if recover:
        result = align_wave(AlignerProcessor(), Aligner(), np.zeros(11360), "Да.", "ru")
        assert result[0]["start_time"] == 0.07
        assert result[0]["end_time"] == 0.15
        assert lengths == [11360, 19360]
    else:
        with pytest.raises(RuntimeError, match="could not measure"):
            align_wave(AlignerProcessor(), Aligner(), np.zeros(11360), "Да.", "ru")
        assert len(lengths) == 4


@pytest.mark.parametrize("samples,start,end", [(1920000, 120.08, 120.16), (1919999, 120.0, 120.0)])
def test_outside_word_retries_instead_of_inverting_duration(samples, start, end):
    from lecture_recognition.models import align_wave

    attempts = []

    class AlignerProcessor:
        def prepare_forced_aligner_inputs(self, audio, **kwargs):
            attempts.append(len(audio))
            return Inputs(input_ids=torch.zeros((1, 1), dtype=torch.long)), [["word", "tail"]]

        def decode_forced_alignment(self, **kwargs):
            if len(attempts) == 1:
                return [
                    [
                        {"text": "word", "start_time": 1, "end_time": 2},
                        {"text": "tail", "start_time": start, "end_time": end},
                    ]
                ]
            return [
                [
                    {"text": "word", "start_time": 1.25, "end_time": 2.25},
                    {"text": "tail", "start_time": 119.25, "end_time": 119.75},
                ]
            ]

    class Aligner:
        dtype = torch.bfloat16
        config = SimpleNamespace(timestamp_token_id=1)

        def __call__(self, **kwargs):
            return SimpleNamespace(logits=None)

    result = align_wave(AlignerProcessor(), Aligner(), np.zeros(samples), "word tail", "ru")
    assert len(attempts) == 2
    assert result[-1]["start_time"] == 119
    assert result[-1]["end_time"] == 119.5


def test_only_invalid_alignment_cache_is_recomputed(tmp_path, monkeypatch):
    from lecture_recognition import models

    cache = Cache(tmp_path)
    items = [{"chunk": Chunk(i, i + 1, i, i + 1).dict(), "text": "word"} for i in range(2)]
    good = {"chunk": items[0]["chunk"], "words": [{"text": "word", "start_time": 0.1, "end_time": 0.9}]}
    bad = {"chunk": items[1]["chunk"], "words": [{"text": "word", "start_time": 1.08, "end_time": 1}]}
    cache.write("alignment", items[0], good)
    cache.write("alignment", items[1], bad)
    calls = []

    def recover(processor, model, audio, chunk, *args):
        calls.append(chunk.start)
        return chunk, [{"text": "word", "start_time": 0.1, "end_time": 0.9}]

    monkeypatch.setattr(models, "load", lambda *_: (None, None))
    monkeypatch.setattr(models, "release", lambda: None)
    monkeypatch.setattr(models, "align_with_context", recover)
    result = models.align(np.zeros(32000), items, "ru", "revision", cache.read, cache.write)
    assert calls == [1]
    assert result[0] == good
    assert cache.read("alignment", items[1]) == result[1]
    monkeypatch.setattr(models, "load", lambda *_: pytest.fail("Valid cache must avoid model loading"))
    assert models.align(np.zeros(32000), items, "ru", "revision", cache.read, cache.write) == result
