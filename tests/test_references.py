import json
import sqlite3
from pathlib import Path

import pytest

from lecture_recognition.jobs import JobStore
from lecture_recognition.service import TranscriptionService


def service_with_job(tmp_path, status='ready', limit_seconds=None):
    service = TranscriptionService(tmp_path / 'jobs')
    job, _ = service.store.create({'audio_path': '/unused.wav', 'audio_info': {'duration': 30},
                                   'limit_seconds': limit_seconds, 'context': '', 'profile': {'id': 'general'}})
    service.store.initialize_segments(job, [
        {'id': 'S000001', 'window': [0, 10], 'variants': {'A': {'text': 'Эйлера'}, 'B': {'text': 'Эйлера'}}},
        {'id': 'S000002', 'window': [10, 20], 'variants': {'A': {'text': 'слова'}, 'B': {'text': 'слова'}}},
    ])
    service.store.state(job, status=status)
    return service, job


def material(**changes):
    return {'title': 'Заметки пользователя', 'original_text': 'теорема Элера', 'text': 'теорема Эйлера',
            'usage': 'context', 'origin': 'agent', 'confirmed_by': 'none', 'reason': 'Исправлена опечатка', **changes}


def test_references_preserve_originals_history_and_asr_on_reconnect(tmp_path, monkeypatch):
    service, job = service_with_job(tmp_path, status='running')
    monkeypatch.setattr(service, '_spawn', lambda _: pytest.fail('Must not restart ASR'))
    before = service.store.snapshot(job)
    service.save_references(job, [material(window=[2, 12], usage='both')], 0)
    reconnected = TranscriptionService(tmp_path / 'jobs')
    page = reconnected.get_references(job)
    value = page['references'][0]
    assert value['confirmed_by'] == 'none'
    reconnected.save_references(job, [{**value, 'original_text': None, 'text': 'теорема Эйлера о рядах',
                                      'confirmed_by': 'agent', 'reason': 'Уточнение агентом'}], page['revision'])
    result = reconnected.get_references(job)
    assert result['revision'] == 2
    assert result['references'][0]['original_text'] == 'теорема Элера'
    assert result['references'][0]['confirmed_by'] == 'agent'
    assert result['history'][0]['changes'][0]['text'] == 'теорема Эйлера'
    assert result['history'][1]['changes'][0]['text'] == 'теорема Эйлера о рядах'
    after = reconnected.store.snapshot(job)
    assert before['request'] == after['request']
    assert before['segments'] == after['segments']
    assert after['state']['status'] == 'running'
    with pytest.raises(ValueError, match='immutable'):
        reconnected.save_references(job, [{**value, 'original_text': 'перезаписан'}], 2)


def test_overlaps_global_context_and_paginated_history(tmp_path):
    service, job = service_with_job(tmp_path)
    result = service.save_references(job, [material(), material(window=[8, 12], usage='both'),
                                          material(window=[10, 20], usage='ground_truth', text='иная версия')], 0)
    global_id, crossing, right = result['reference_ids']
    page = service.get_segments(job, limit=1)
    assert page['segments'][0]['reference_ids'] == [crossing]
    assert page['other_reference_count'] == 2
    assert page['reference_count'] == 3
    assert page['references'][0]['id'] == crossing
    assert 'text' not in page['references'][0]
    page = service.get_segments(job, offset=1)
    assert page['segments'][0]['reference_ids'] == [crossing, right]
    refs = service.get_references(job, limit=1)
    assert refs['references'][0]['id'] == global_id
    assert refs['total'] == 3 and refs['next_offset'] == 1
    assert len(refs['history'][0]['changes']) == 1
    assert service.get_references(job, offset=2)['next_offset'] is None


def test_shared_revision_atomicity_and_audit_export(tmp_path):
    service, job = service_with_job(tmp_path)
    with pytest.raises(ValueError):
        service.save_references(job, [material(), material(text='')], 0)
    assert service.get_references(job)['total'] == 0
    assert service.store.get(job)['revision'] == 0
    service.save_references(job, [material()], 0)
    decision = [{'segment_id': 'S000001', 'choice': 'edited', 'text': 'теорема Эйлера', 'reason': 'Reference'}]
    with pytest.raises(ValueError, match='Revision changed'):
        service.save_revision(job, decision, 0)
    ref = service.get_references(job)['references'][0]
    decision[0]['reason'] = 'Supported by ' + ref['id']
    service.save_revision(job, decision, 1)
    with pytest.raises(ValueError, match='Revision changed'):
        service.save_references(job, [material()], 1)
    service.save_references(job, [{**ref, 'origin': 'user', 'confirmed_by': 'user', 'reason': 'Пользователь подтвердил'}], 2)
    exported = service.export_transcript(job)
    audit = json.loads(Path(exported['paths']['audit']).read_text())
    assert audit['revision'] == 3
    assert audit['references'][0]['confirmed_by'] == 'user'
    assert [r['revision'] for r in audit['reference_history']] == [1, 3]
    assert audit['history'][0]['revision'] == 2
    assert ref['id'] in audit['segments'][0]['decision']['reason']
    assert Path(exported['paths']['text']).read_text() == 'теорема Эйлера\n\nслова\n'
    assert 'Заметки' not in Path(exported['paths']['srt']).read_text()
    with pytest.raises(ValueError, match='Duplicate'):
        service.save_references(job, [ref, ref], 3)
    assert service.store.get(job)['revision'] == 3


@pytest.mark.parametrize('changes', [
    {'original_text': ''}, {'text': ' '}, {'reason': ''}, {'title': '', 'source': ''},
    {'window': [0, 31]}, {'window': [10, 10]}, {'window': [-1, 3]}, {'window': [0, float('nan')]},
    {'window': [0, float('inf')]}, {'window': [0, 1, 2]}, {'window': [False, 3]},
    {'usage': 'ground_truth'}, {'usage': 'both'}, {'usage': 'wrong'}, {'origin': 'asr'},
    {'confirmed_by': 'yes'}, {'id': '../bad'}, {'id': 'a' * 32},
])
def test_invalid_materials_are_rejected_without_partial_updates(tmp_path, changes):
    service, job = service_with_job(tmp_path)
    with pytest.raises(ValueError):
        service.save_references(job, [material(), material(**changes)], 0)
    assert service.get_references(job)['references'] == []
    assert service.store.get(job)['revision'] == 0


def test_preview_interval_and_pagination_validation(tmp_path):
    service, job = service_with_job(tmp_path, limit_seconds=12)
    with pytest.raises(ValueError, match='timeline'):
        service.save_references(job, [material(window=[10, 15])], 0)
    service.save_references(job, [material(window=[10, 12], usage='both')], 0)
    for arguments in ({'offset': -1}, {'limit': 0}, {'limit': 21}):
        with pytest.raises(ValueError):
            service.get_references(job, **arguments)
    with pytest.raises(ValueError, match='Unknown job'):
        service.save_references('f' * 32, [material()], 0)


def test_existing_database_is_migrated_without_losing_jobs(tmp_path):
    root = tmp_path / 'jobs'
    root.mkdir()
    job = 'a' * 32
    with sqlite3.connect(root / 'jobs.sqlite') as db:
        db.execute('CREATE TABLE jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL, state TEXT NOT NULL, '
                   'revision INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL)')
        db.execute('INSERT INTO jobs VALUES(?,?,?,?,?)', (job, json.dumps({'legacy': True}),
                                                        json.dumps({'status': 'ready'}), 7, 0))
    store = JobStore(root)
    snapshot = store.snapshot(job)
    assert snapshot['revision'] == 7 and snapshot['request']['legacy']
    assert snapshot['references'] == [] and snapshot['reference_history'] == []
    store.save_references(job, [material(source='notes.txt')], 7)
    assert JobStore(root).snapshot(job)['revision'] == 8
