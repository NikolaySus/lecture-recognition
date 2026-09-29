from lecture_recognition.cli import Cache, atomic_text


def test_cache_keys_and_resume(tmp_path):
    cache = Cache(tmp_path)
    assert cache.read("asr", {"start": 1}) is None
    cache.write("asr", {"start": 1}, [{"text": "лекция"}])
    assert Cache(tmp_path).read("asr", {"start": 1}) == [{"text": "лекция"}]
    assert cache.read("asr", {"start": 2}) is None
    assert cache.read("alignment", {"start": 1}) is None


def test_atomic_output(tmp_path):
    path = tmp_path / "result.srt"
    atomic_text(path, "old")
    atomic_text(path, "new")
    assert path.read_text() == "new"
    assert not path.with_suffix(".srt.tmp").exists()
