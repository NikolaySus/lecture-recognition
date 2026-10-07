"""Durable jobs and optimistic, atomic revision batches; no LLM calls inside MCP."""
import json
import math
import re
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from .runtime import identity


class JobStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        with self.db() as db:
            db.executescript('''
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, request TEXT NOT NULL,
                    state TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS segments (job TEXT NOT NULL, id TEXT NOT NULL, data TEXT NOT NULL,
                    decision TEXT NOT NULL, PRIMARY KEY(job,id));
                CREATE TABLE IF NOT EXISTS reference_materials (job TEXT NOT NULL, id TEXT NOT NULL,
                    data TEXT NOT NULL, PRIMARY KEY(job,id));
                CREATE TABLE IF NOT EXISTS reference_history (job TEXT NOT NULL, revision INTEGER NOT NULL,
                    changes TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(job,revision));
                CREATE TABLE IF NOT EXISTS revisions (job TEXT NOT NULL, revision INTEGER NOT NULL,
                    changes TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY(job,revision));
            ''')

    @contextmanager
    def db(self):
        connection = sqlite3.connect(self.root / 'jobs.sqlite', timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def folder(self, job_id):
        if not re.fullmatch(r'[0-9a-f]{32}', job_id):
            raise ValueError('Invalid job ID')
        return self.root / job_id

    def create(self, request):
        job_id = identity(request)[:32]
        with self.db() as db:
            cursor = db.execute('INSERT OR IGNORE INTO jobs(id,request,state,created) VALUES(?,?,?,?)',
                                (job_id, json.dumps(request, ensure_ascii=False),
                                 json.dumps({'status': 'queued', 'stage': 'queued'}), time.time()))
        self.folder(job_id).mkdir(exist_ok=True)
        return job_id, bool(cursor.rowcount)

    def get(self, job_id):
        self.folder(job_id)
        with self.db() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
        if row is None:
            raise ValueError('Unknown job')
        return {'job_id': job_id, 'request': json.loads(row['request']), 'state': json.loads(row['state']),
                'revision': row['revision'], 'created': row['created']}

    def list_ids(self, limit=10, audio_path=None):
        with self.db() as db:
            query, params = 'SELECT id FROM jobs', []
            if audio_path is not None:
                query += " WHERE json_extract(request,'$.audio_path')=?"
                params.append(audio_path)
            query += ' ORDER BY created DESC LIMIT ?'
            return [r['id'] for r in db.execute(query, [*params, limit])]

    def state(self, job_id, **changes):
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT state FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise ValueError('Unknown job')
            value = {**json.loads(row['state']), **changes, 'updated': time.time()}
            db.execute('UPDATE jobs SET state=? WHERE id=?', (json.dumps(value, ensure_ascii=False), job_id))

    def initialize_segments(self, job_id, segments):
        with self.db() as db:
            for segment in segments:
                variants = list(segment['variants'].values())
                same = len({v['text'] for v in variants}) == 1 and bool(variants[0]['text'].strip())
                decision = {'status': 'resolved' if same else 'pending', 'text': variants[0]['text'] if same else None,
                            'choice': 'agreement' if same else None, 'origin': 'asr',
                            'reason': 'identical transcripts' if same else '', 'question': ''}
                db.execute('INSERT OR IGNORE INTO segments VALUES(?,?,?,?)',
                           (job_id, segment['id'], json.dumps(segment, ensure_ascii=False), json.dumps(decision, ensure_ascii=False)))

    def segments(self, job_id):
        self.get(job_id)
        with self.db() as db:
            rows = db.execute('SELECT data,decision FROM segments WHERE job=? ORDER BY id', (job_id,)).fetchall()
        return [{**json.loads(r['data']), 'decision': json.loads(r['decision'])} for r in rows]

    def save(self, job_id, changes, expected_revision):
        if not changes or len(changes) > 50:
            raise ValueError('Supply 1-50 decisions')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT revision FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None or row['revision'] != expected_revision:
                raise ValueError('Revision changed; reread segments before saving')
            ids = [c['segment_id'] for c in changes]
            if len(set(ids)) != len(ids):
                raise ValueError('Duplicate segment decisions')
            for change in changes:
                stored = db.execute('SELECT data FROM segments WHERE job=? AND id=?', (job_id, change['segment_id'])).fetchone()
                if stored is None:
                    raise ValueError('Unknown segment')
                segment = json.loads(stored['data'])
                choice = change['choice']
                if choice not in (*segment['variants'], 'edited', 'uncertain'):
                    raise ValueError('Invalid choice')
                if not change.get('reason', '').strip():
                    raise ValueError('A reason is required for every decision')
                text = segment['variants'][choice]['text'] if choice in segment['variants'] else change.get('text')
                if choice == 'edited' and (not isinstance(text, str) or not text.strip()):
                    raise ValueError('Edited text must be nonempty')
                if choice == 'uncertain':
                    text = None
                    if not change.get('question', '').strip():
                        raise ValueError('Uncertain decisions require a question')
                origin = change.get('origin', 'agent')
                if origin not in ('agent', 'user'):
                    raise ValueError('Invalid decision origin')
                decision = {'status': 'uncertain' if choice == 'uncertain' else 'resolved', 'choice': choice,
                            'text': text, 'reason': change['reason'], 'question': change.get('question', ''), 'origin': origin}
                db.execute('UPDATE segments SET decision=? WHERE job=? AND id=?',
                           (json.dumps(decision, ensure_ascii=False), job_id, segment['id']))
            version = expected_revision + 1
            db.execute('UPDATE jobs SET revision=? WHERE id=?', (version, job_id))
            db.execute('INSERT INTO revisions VALUES(?,?,?,?)', (job_id, version, json.dumps(changes, ensure_ascii=False), time.time()))
        return version

    def save_references(self, job_id, references, expected_revision):
        """Save full material versions without touching ASR inputs or segment decisions."""
        self.folder(job_id)
        if not references or len(references) > 50:
            raise ValueError('Supply 1-50 reference materials')
        with self.db() as db:
            db.execute('BEGIN IMMEDIATE')
            job = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if job is None:
                raise ValueError('Unknown job')
            if job['revision'] != expected_revision:
                raise ValueError('Revision changed; reread references or segments before saving')
            request, state = json.loads(job['request']), json.loads(job['state'])
            duration = state.get('duration', request.get('audio_info', {}).get('duration'))
            if request.get('limit_seconds') is not None:
                duration = min(duration, request['limit_seconds']) if duration is not None else request['limit_seconds']
            saved, seen = [], set()
            for reference in references:
                value = dict(reference)
                ref_id = value.get('id')
                if ref_id is None:
                    ref_id = uuid.uuid4().hex
                    previous = None
                else:
                    if not isinstance(ref_id, str) or not re.fullmatch(r'[0-9a-f]{32}', ref_id):
                        raise ValueError('Invalid reference ID')
                    row = db.execute('SELECT data FROM reference_materials WHERE job=? AND id=?', (job_id, ref_id)).fetchone()
                    if row is None:
                        raise ValueError('Unknown reference; omit ID to create a material')
                    previous = json.loads(row['data'])
                if ref_id in seen:
                    raise ValueError('Duplicate reference updates')
                seen.add(ref_id)
                original = value.get('original_text')
                if previous:
                    if original is not None and original != previous['original_text']:
                        raise ValueError('Original text is immutable')
                    original = previous['original_text']
                for name, text in (('original_text', original), ('text', value.get('text')), ('reason', value.get('reason'))):
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f'{name} must be nonempty')
                title, source = value.get('title', ''), value.get('source', '')
                if not isinstance(title, str) or not isinstance(source, str) or not (title.strip() or source.strip()):
                    raise ValueError('Supply a title or source for the material')
                usage = value.get('usage', 'context')
                origin = value.get('origin', 'agent')
                confirmed_by = value.get('confirmed_by', 'none')
                if usage not in ('context', 'ground_truth', 'both'):
                    raise ValueError('Invalid reference usage')
                if origin not in ('agent', 'user') or confirmed_by not in ('none', 'agent', 'user'):
                    raise ValueError('Invalid reference provenance')
                window = value.get('window')
                if window is None and usage in ('ground_truth', 'both'):
                    raise ValueError('Ground truth requires a time interval')
                if window is not None:
                    if (not isinstance(window, (list, tuple)) or len(window) != 2
                            or any(type(v) not in (int, float) or not math.isfinite(v) for v in window)
                            or not 0 <= window[0] < window[1]):
                        raise ValueError('Invalid reference interval')
                    if duration is None or window[1] > duration:
                        raise ValueError('Reference interval must be within the job audio timeline')
                    window = list(window)
                material = {'id': ref_id, 'title': title, 'source': source, 'original_text': original,
                            'text': value['text'], 'window': window, 'usage': usage, 'origin': origin,
                            'confirmed_by': confirmed_by, 'reason': value['reason']}
                db.execute('INSERT INTO reference_materials VALUES(?,?,?) ON CONFLICT(job,id) DO UPDATE SET data=excluded.data',
                           (job_id, ref_id, json.dumps(material, ensure_ascii=False)))
                saved.append(material)
            version = expected_revision + 1
            db.execute('UPDATE jobs SET revision=? WHERE id=?', (version, job_id))
            db.execute('INSERT INTO reference_history VALUES(?,?,?,?)',
                       (job_id, version, json.dumps(saved, ensure_ascii=False), time.time()))
        return version, saved

    def history(self, job_id):
        with self.db() as db:
            return [{'revision': r['revision'], 'changes': json.loads(r['changes']), 'created': r['created']}
                    for r in db.execute('SELECT * FROM revisions WHERE job=? ORDER BY revision', (job_id,))]

    def snapshot(self, job_id):
        self.folder(job_id)
        with self.db() as db:
            db.execute('BEGIN')
            job = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if job is None:
                raise ValueError('Unknown job')
            segments = [{**json.loads(r['data']), 'decision': json.loads(r['decision'])}
                        for r in db.execute('SELECT data,decision FROM segments WHERE job=? ORDER BY id', (job_id,))]
            history = [{'revision': r['revision'], 'changes': json.loads(r['changes']), 'created': r['created']}
                       for r in db.execute('SELECT * FROM revisions WHERE job=? ORDER BY revision', (job_id,))]
            references = [json.loads(r['data']) for r in db.execute(
                'SELECT data FROM reference_materials WHERE job=? ORDER BY rowid', (job_id,))]
            reference_history = [{'revision': r['revision'], 'changes': json.loads(r['changes']), 'created': r['created']}
                                 for r in db.execute('SELECT * FROM reference_history WHERE job=? ORDER BY revision', (job_id,))]
            return {'job_id': job_id, 'request': json.loads(job['request']), 'state': json.loads(job['state']),
                    'revision': job['revision'], 'segments': segments, 'history': history,
                    'references': references, 'reference_history': reference_history}
