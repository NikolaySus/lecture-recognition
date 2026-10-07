import json
from pathlib import Path

import pytest

from lecture_recognition.jobs import JobStore
from lecture_recognition.pipeline import comparison_segments, conflicts, historical_layout
from lecture_recognition.runtime import read, write
from lecture_recognition.service import TranscriptionService


def prepared_service(tmp_path):
    service = TranscriptionService(tmp_path / 'jobs')
    request = {'audio_path': '/unused.wav', 'context': 'Time series lecture', 'profile': {'id': 'time-series'}}
    job, _ = service.store.create(request)
    segments = [{'id': 'S000001', 'window': [0, 10], 'variants': {'A': {'text': 'нестационарных данных'},
                 'B': {'text': 'стационарных данных'}}, 'conflicts': []},
                {'id': 'S000002', 'window': [10, 20], 'variants': {'A': {'text': 'общий текст'},
                 'B': {'text': 'общий текст'}}, 'conflicts': []}]
    service.store.initialize_segments(job, segments)
    service.store.state(job, status='ready')
    return service, job


def test_job_identity_resume_and_path_traversal(tmp_path):
    store = JobStore(tmp_path)
    a, created = store.create({'audio_sha256': 'example', 'settings': 1})
    assert created
    b, created = store.create({'audio_sha256': 'example', 'settings': 1})
    assert a == b and not created
    c, _ = store.create({'audio_sha256': 'example', 'settings': 2})
    assert c != a
    with pytest.raises(ValueError):
        store.folder('../../audio')


def test_equal_sources_and_reversed_presentation_do_not_change_saved_choice(tmp_path):
    service, job = prepared_service(tmp_path)
    a = service.get_segments(job)
    b = service.get_segments(job, reverse_order=True)
    assert list(a['segments'][0]['variants']) == ['A', 'B']
    assert list(b['segments'][0]['variants']) == ['B', 'A']
    assert a['segments'][0]['variants'] == b['segments'][0]['variants']
    assert a['segments'][1]['decision']['text'] == 'общий текст'
    service.save_revision(job, [{'segment_id': 'S000001', 'choice': 'B', 'reason': 'User listened', 'origin': 'user'}], 0)
    assert service.get_segments(job)['segments'][0]['decision']['text'] == 'стационарных данных'


def test_revision_batch_is_atomic_and_stale_write_rejected(tmp_path):
    service, job = prepared_service(tmp_path)
    updates = [{'segment_id': 'S000001', 'choice': 'A', 'reason': 'Context'},
               {'segment_id': 'missing', 'choice': 'A', 'reason': 'Context'}]
    with pytest.raises(ValueError):
        service.save_revision(job, updates, 0)
    assert service.store.get(job)['revision'] == 0
    assert service.store.segments(job)[0]['decision']['status'] == 'pending'
    service.save_revision(job, updates[:1], 0)
    with pytest.raises(ValueError, match='Revision changed'):
        service.save_revision(job, updates[:1], 0)
    assert len(service.store.history(job)) == 1


def test_uncertainty_survives_reconnect_and_final_export_is_blocked(tmp_path):
    service, job = prepared_service(tmp_path)
    service.save_revision(job, [{'segment_id': 'S000001', 'choice': 'uncertain', 'reason': 'Negation changes meaning',
                                 'question': 'Стационарных или нестационарных?'}], 0)
    reconnected = TranscriptionService(tmp_path / 'jobs')
    assert reconnected.get_segments(job, unresolved_only=True)['segments'][0]['decision']['question']
    with pytest.raises(ValueError, match='Unresolved'):
        reconnected.export_transcript(job)
    draft = reconnected.export_transcript(job, allow_draft=True)
    assert draft['draft']
    assert 'Вариант B: стационарных' in Path(draft['paths']['markdown']).read_text()
    reconnected.save_revision(job, [{'segment_id': 'S000001', 'choice': 'edited', 'text': 'нестационарных данных',
                                     'reason': 'User confirmed', 'origin': 'user'}], 1)
    final = reconnected.export_transcript(job)
    assert not final['draft']
    audit = json.loads(Path(final['paths']['audit']).read_text())
    assert len(audit['history']) == 2
    assert audit['segments'][0]['decision']['origin'] == 'user'
    assert Path(draft['paths']['text']).exists()


def test_missing_reason_empty_edit_and_duplicate_ids_rejected(tmp_path):
    service, job = prepared_service(tmp_path)
    for changes in [[{'segment_id': 'S000001', 'choice': 'A'}],
                    [{'segment_id': 'S000001', 'choice': 'edited', 'reason': 'Correction', 'text': ''}],
                    [{'segment_id': 'S000001', 'choice': 'A', 'reason': 'a'}] * 2]:
        with pytest.raises(ValueError):
            service.save_revision(job, changes, 0)


def test_empty_asr_is_not_silent_agreement_and_explicit_empty_choice_has_no_srt_cue(tmp_path):
    service = TranscriptionService(tmp_path)
    job, _ = service.store.create({'audio_path': 'empty.wav', 'context': '', 'profile': {'id': 'general'}})
    service.store.initialize_segments(job, [{'id': 'S000001', 'window': [0, 10],
         'variants': {'A': {'text': ''}, 'B': {'text': ''}}}])
    service.store.state(job, status='ready')
    assert service.get_status(job)['unresolved_segments'] == 1
    service.save_revision(job, [{'segment_id': 'S000001', 'choice': 'A', 'reason': 'User confirms silence', 'origin': 'user'}], 0)
    result = service.export_transcript(job)
    assert Path(result['paths']['srt']).read_text() == ''


def test_mono_preserves_original_text_and_measured_unit_bounds():
    chunk = {'start': 0., 'end': 10., 'core_start': 0., 'core_end': 10.}
    source = {'A': {'transcripts': [{'text': 'сырой текст', 'confidence': {'gibbs': .5}}],
                    'aligned': [{'chunk': chunk, 'words': [{'text': 'сырой текст', 'start_time': 1., 'end_time': 2.}]}]}}
    rows = comparison_segments([chunk], source)
    assert list(rows[0]['variants']) == ['A']
    assert rows[0]['variants']['A']['input_text'] == 'сырой текст'
    assert rows[0]['variants']['A']['words'][0]['start'] == 1.
    assert rows[0]['variants']['A']['words'][0]['end'] == 2.


def test_historical_layout_uses_common_bounds_with_at_most_20_seconds():
    import numpy as np
    rows = historical_layout(np.ones(16000 * 70, dtype='<f4'), 70., [(0., 70.)])
    assert rows[0]['core_start'] == 0 and rows[-1]['core_end'] == 70
    assert all(r['end'] - r['start'] <= 20 for r in rows)
    assert all(a['core_end'] == b['core_start'] for a, b in zip(rows, rows[1:]))


def test_conflicts_show_missing_negation_and_word_in_other_channel():
    assert conflicts('математику элеру где', 'математику где') == [{'a': 'элеру', 'b': '', 'operation': 'delete'}]


def test_resume_worker_request_guard(tmp_path, monkeypatch):
    from lecture_recognition.runtime import cuda_worker
    request = {'operation': 'asr', 'audio': 'audio.f32'}
    write(tmp_path / 'request.json', request)
    write(tmp_path / 'result.json', {'status': 'ok', 'transcripts': []})
    assert cuda_worker(request, tmp_path)['status'] == 'ok'
    with pytest.raises(ValueError, match='request changed'):
        cuda_worker({**request, 'audio': 'other.f32'}, tmp_path)
    assert read(tmp_path / 'request.json') == request


def test_real_audio_probe_clip_and_idempotent_start(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf
    source = tmp_path / 'stereo.wav'
    sf.write(source, np.column_stack((np.ones(16000) * .2, np.ones(16000) * -.2)), 16000)
    service = TranscriptionService(tmp_path / 'jobs')
    launched = []
    monkeypatch.setattr(service, '_spawn', lambda job: launched.append(job))
    job = service.start_transcription(str(source))['job_id']
    assert service.start_transcription(str(source))['job_id'] == job
    assert launched == [job]
    request = service.store.get(job)['request']
    assert service.list_jobs(str(source))['jobs'][0]['job_id'] == job
    assert request['profile']['model_config']['beam'] == 32
    assert request['profile']['model_config']['bias_weight'] == 4
    assert request['audio_info']['channels'] == 2
    clip = service.get_audio_clip(job, 0., .5, 'B')
    data, rate = sf.read(clip['path'])
    assert rate == 16000 and data.mean() < 0 and len(data) == 8000
    with pytest.raises(ValueError):
        service.get_audio_clip(job, 0., 2.)
    sf.write(source, np.zeros((16000, 2)), 16000)
    with pytest.raises(ValueError, match='changed'):
        service.get_audio_clip(job, 0., .5)


def test_runtime_change_rejects_resume_before_model_work(tmp_path):
    from lecture_recognition.pipeline import run_pipeline
    store = JobStore(tmp_path)
    job, _ = store.create({'runtime_code': {'pipeline.py': 'wrong-digest'}})
    with pytest.raises(ValueError, match='Runtime changed'):
        run_pipeline(store, job)


def test_pipeline_validates_root_lock_and_completes_equal_channel_job(tmp_path, monkeypatch):
    import numpy as np
    import soundfile as sf

    from lecture_recognition import pipeline
    source = tmp_path / 'stereo.wav'
    sf.write(source, np.zeros((32000, 2)), 16000)
    service = TranscriptionService(tmp_path / 'jobs')
    monkeypatch.setattr(service, '_spawn', lambda job: None)
    job = service.start_transcription(str(source))['job_id']
    requests = []
    def model_worker(request, directory, gigaam_python=None):
        requests.append(request)
        assert not any(k in request for k in ('reference', 'ground_truth', 'review'))
        if request['operation'] == 'diarize':
            return {'segments': [{'Start': 0., 'End': 2., 'Speaker': 0}]}
        if request['operation'] == 'asr':
            return {'transcripts': [{'chunk': c, 'text': 'общий текст', 'confidence': {'gibbs': .7}}
                                    for c in request['chunks']]}
        return {'aligned': [{'chunk': t['chunk'], 'words': [{'text': t['text'], 'start_time': .1, 'end_time': 1.5}]}
                            for t in request['transcripts']]}
    monkeypatch.setattr(pipeline, 'cuda_worker', model_worker)
    pipeline.run_pipeline(service.store, job)
    assert service.get_status(job)['status'] == 'ready'
    assert service.get_status(job)['unresolved_segments'] == 0
    asr = [r for r in requests if r['operation'] == 'asr']
    assert len(asr) == 2 and asr[0]['config'] == asr[1]['config'] and asr[0]['chunks'] == asr[1]['chunks']
    assert not service.export_transcript(job)['draft']
