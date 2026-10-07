"""Durable jobs and optimistic, atomic revision batches; no LLM calls inside MCP."""
import json
import re
import sqlite3
import time
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
            return {'job_id': job_id, 'request': json.loads(job['request']), 'state': json.loads(job['state']),
                    'revision': job['revision'], 'segments': segments, 'history': history}
