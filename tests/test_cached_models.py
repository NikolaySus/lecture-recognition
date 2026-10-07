from lecture_recognition import models
from lecture_recognition.cuda_worker import ensure_gigaam_snapshot


def test_gigaam_complete_cache_never_contacts_network(tmp_path, monkeypatch):
    import huggingface_hub
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'pytorch_model.bin').write_bytes(b'cached')
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        assert kwargs['local_files_only'] is True
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    ensure_gigaam_snapshot({'repo': 'repo', 'revision': 'pinned'})
    assert len(calls) == 1


def test_incomplete_gigaam_cache_fetches_pinned_revision_once(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []
    def download(repo, **kwargs):
        calls.append(kwargs)
        return str(tmp_path)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    ensure_gigaam_snapshot({'repo': 'repo', 'revision': 'pinned'})
    assert len(calls) == 2
    assert all(c['revision'] == 'pinned' for c in calls)
    assert calls[0]['local_files_only'] is True


def test_diarization_prefers_cached_processor_and_model(monkeypatch):
    calls = []
    class Model:
        def to(self, device):
            assert device == 'cuda'
            return self
        def eval(self):
            return self
    class Loader:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(kwargs)
            assert kwargs['local_files_only'] is True
            return Model()
    monkeypatch.setattr(models, 'AutoProcessor', Loader)
    monkeypatch.setattr(models, 'AutoModelForAudioFrameClassification', Loader)
    models.load('diarization', 'pinned')
    assert len(calls) == 2
