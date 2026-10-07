import json
from pathlib import Path

import numpy as np
import pytest

from lecture_recognition.evaluation import reference_cards, score_case
from lecture_recognition.model_benchmark import (
    Benchmark,
    extra_metrics,
    layout,
    safe,
    shifted,
    validate_srt,
    worker,
)


def test_selected_cards_do_not_accept_blank_or_unselected_answers():
    markdown = "### R009\n**Правильная транскрипция:**\n\n**Статус:**\n"
    with pytest.raises(ValueError, match="Missing reference"):
        reference_cards(markdown, ["R009"])
    assert reference_cards(markdown + "### R010\n**Правильная транскрипция:** Текст\n", ["R010"]) == {
        "R010": "Текст"
    }


@pytest.mark.parametrize("maximum", [20, 30, 60])
@pytest.mark.parametrize("duration", [20, 61.3, 345.92])
def test_layout_covers_timeline_and_preserves_bounds(maximum, duration):
    audio = np.zeros(round(duration * 16000), dtype=np.float32)
    for shift in (0, 5):
        chunks = layout(audio, duration, maximum, [(0, duration)], shift)
        assert chunks[0]["core_start"] == 0
        assert chunks[-1]["core_end"] == duration
        assert all(c["end"] - c["start"] <= maximum + 1e-6 for c in chunks)
        assert all(a["core_end"] == b["core_start"] for a, b in zip(chunks, chunks[1:]))


def test_shift_qwen_preserves_ownership_and_limits():
    chunks = [
        dict(start=0, end=116.37, core_start=0, core_end=115.37),
        dict(start=114.37, end=231.78, core_start=115.37, core_end=230.78),
        dict(start=229.78, end=345.92, core_start=230.78, core_end=344.92),
    ]
    result = shifted(chunks, 345.92, 120, 60)
    assert result[0]["core_start"] == 0
    assert result[-1]["core_end"] == 344.92
    assert result != chunks
    assert all(60 <= c["core_end"] - c["core_start"] <= 118 for c in result)
    assert all(c["end"] - c["start"] <= 120 for c in result)


def score(text):
    reference = {
        "groups": [{"name": "one", "window": [0, 10], "text": "не было 1024 отсчета", "cards": {}}],
        "target_window": [0, 10],
    }
    return extra_metrics(score_case([dict(start=1, end=2, text=text)], reference))


def test_safety_rejects_critical_regression_and_false_dictionary_terms():
    baseline = {"status": "ok", "score": score("не было 1024 отсчета вчера сегодня")}
    candidate = {"status": "ok", "score": score("было 124 отсчета")}
    assert not safe(candidate, baseline)
    candidate["score"] = score("не было 1024 отсчета квазистационарности")
    assert not safe(candidate, baseline)
    candidate["score"] = score("не было 1024 отсчета")
    assert safe(candidate, baseline)
    assert not safe({"status": "error"}, baseline)
    metrics = score("было 124 отсчета")["total"]
    assert metrics["number_errors"] == metrics["negation_errors"] == 1


def test_worker_caches_failure_but_retry_is_explicit(tmp_path, monkeypatch):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(command[-1]).write_text(json.dumps({"status": "error", "error": "example"}))
        return type("Result", (), {"returncode": 1})()

    monkeypatch.setattr("lecture_recognition.model_benchmark.subprocess.run", run)
    req = {"audio": "/audio", "chunks": []}
    assert worker(req, tmp_path)["status"] == "error"
    worker(req, tmp_path)
    assert len(calls) == 1
    worker(req, tmp_path, retry=True)
    assert len(calls) == 2
    with pytest.raises(ValueError, match="Request changed"):
        worker({"audio": "/other"}, tmp_path)


def test_failed_alignment_preserves_asr_and_has_no_score(tmp_path, monkeypatch):
    bench = Benchmark.__new__(Benchmark)
    bench.root = tmp_path
    bench.args = type("Args", (), {"retry_failed": False})()
    bench.config = {"models": {"qwen": {"backend": "qwen"}}}
    bench.prepared = {"audio": "/test.f32", "reference_sha256": "frozen"}
    bench.chunks = lambda *args: [dict(start=0, end=20, core_start=0, core_end=20)]
    requests = []

    def fake_worker(request, directory, *args, **kwargs):
        requests.append(request)
        if request["operation"] == "asr":
            return {"status": "ok", "transcripts": [dict(chunk=request["chunks"][0], text="raw text")]}
        return {"status": "error", "error": "invalid alignment"}

    monkeypatch.setattr("lecture_recognition.model_benchmark.worker", fake_worker)
    result = bench.case("qwen")
    assert result["phase"] == "alignment"
    assert result["status"] == "error"
    assert "score" not in result
    assert requests[1]["transcripts"][0]["text"] == "raw text"
    assert all("reference" not in key for key in requests[0])


def test_final_srt_validation_rejects_excluded_speech_and_padding():
    text = "1\n00:00:01,000 --> 00:00:03,000\nТекст\n"
    assert validate_srt(text, 10, [(4, 6)])["blocks"] == 1
    with pytest.raises(ValueError, match="excluded"):
        validate_srt(text, 10, [(2, 4)])
    with pytest.raises(ValueError, match="timeline"):
        validate_srt(text, 2, [])
    with pytest.raises(ValueError, match="timeline"):
        validate_srt(text + text, 10, [])


def test_whisper_decoder_prompt_is_not_scored_as_generated_text():
    import importlib.util
    from types import SimpleNamespace

    import torch

    path = Path(__file__).resolve().parents[1] / "scripts/asr_worker.py"
    spec = importlib.util.spec_from_file_location("asr_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = SimpleNamespace(sequences=torch.tensor([[50, 51, 52, 7, 8, 99]]), scores=(None,) * 3)
    assert module.whisper_generated_ids(output, 99).tolist() == [7, 8, 99]
    output.sequences = torch.tensor([[99, 50, 51, 7, 8, 9]])
    with pytest.raises(RuntimeError, match="limit"):
        module.whisper_generated_ids(output, 99)
