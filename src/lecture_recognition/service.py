"""Application API behind MCP; GPU jobs outlive the client connection."""
import math
import os
import subprocess
import sys
from pathlib import Path

import psutil

from .audio import digest
from .jobs import JobStore
from .pipeline import probe
from .runtime import ROOT, identity, read, worker_env, write
from .timeline import stamp

PROFILES = Path(__file__).parent / 'profiles'


def alive(pid, birth):
    if not pid or birth is None:
        return False
    try:
        process = psutil.Process(pid)
        return abs(process.create_time() - birth) < .01 and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


class TranscriptionService:
    def __init__(self, jobs_dir, gigaam_python=None):
        self.store = JobStore(jobs_dir)
        self.gigaam_python = str(Path(gigaam_python).resolve()) if gigaam_python else None

    def get_profiles(self):
        return {'profiles': [read(p) for p in sorted(PROFILES.glob('*.json'))],
                'default': 'time-series', 'channels': 'A and B have identical decoding settings and no preferred order'}

    def list_jobs(self, audio_path=None, limit=10):
        if not 1 <= limit <= 50:
            raise ValueError('Limit must be 1-50')
        source = str(Path(audio_path).expanduser().resolve()) if audio_path else None
        jobs = []
        for job_id in self.store.list_ids(limit, source):
            job = self.store.get(job_id)
            jobs.append({**self.get_status(job_id), 'context': job['request'].get('context', ''), 'created': job['created']})
        return {'jobs': jobs}

    def start_transcription(self, audio_path, profile='time-series', terms=None, context='', speaker_id=None,
                            limit_seconds=None):
        source = Path(audio_path).expanduser().resolve()
        if not source.is_file():
            raise ValueError('Audio path must name an existing local file')
        if profile not in ('time-series', 'general'):
            raise ValueError('Unknown profile')
        settings = read(PROFILES / (profile + '.json'))
        if terms is not None:
            if len(terms) > 500 or any(not isinstance(t, str) or not t.strip() or len(t) > 150 for t in terms):
                raise ValueError('Terms must be 0-500 nonempty strings, at most 150 characters each')
            settings['model_config']['terms'] = sorted(set(terms))
        if speaker_id is not None and (type(speaker_id) is not int or not 0 <= speaker_id < 8):
            raise ValueError('Speaker ID must be 0-7')
        if limit_seconds is not None and (not math.isfinite(limit_seconds) or limit_seconds < 1):
            raise ValueError('Preview limit must be at least one second')
        if len(context) > 20000:
            raise ValueError('Context exceeds 20000 characters')
        modules = ('pipeline.py', 'runtime.py', 'cuda_worker.py', 'confidence.py', 'gigaam_decoding.py', 'models.py', 'timeline.py', 'audio.py')
        code = {name: digest(Path(__file__).parent / name) for name in modules}
        for path in (ROOT / 'uv.lock', ROOT / 'experiments/gigaam/uv.lock'):
            if path.exists():
                code[str(path.relative_to(ROOT))] = digest(path)
        request = {'audio_path': str(source), 'audio_sha256': digest(source), 'audio_info': probe(source),
                   'profile': settings, 'context': context, 'speaker_id': speaker_id,
                   'limit_seconds': limit_seconds, 'gigaam_python': self.gigaam_python,
                   'runtime_code': code, 'version': 1}
        job_id, created = self.store.create(request)
        if created:
            self._spawn(job_id)
        return self.get_status(job_id)

    def _spawn(self, job_id):
        try:
            with (self.store.folder(job_id) / 'job.log').open('a', encoding='utf-8') as log:
                process = subprocess.Popen([sys.executable, '-m', 'lecture_recognition.job_worker',
                                            '--store', str(self.store.root), '--job', job_id],
                                           stdout=log, stderr=subprocess.STDOUT, env=worker_env(),
                                           start_new_session=os.name != 'nt')
            try:
                birth = psutil.Process(process.pid).create_time()
            except psutil.NoSuchProcess:
                birth = None
            self.store.state(job_id, worker_pid=process.pid, worker_birth=birth)
        except Exception as exc:
            self.store.state(job_id, status='failed', error=str(exc))
            raise

    def get_status(self, job_id):
        job = self.store.get(job_id)
        state = job['state']
        if state['status'] in ('running', 'queued') and state.get('worker_pid') and not alive(state['worker_pid'], state.get('worker_birth')):
            self.store.state(job_id, status='interrupted', error='Worker exited; resume with retry_transcription')
            state = self.store.get(job_id)['state']
        folder = self.store.folder(job_id)
        completed = sum(len(list((folder / c / 'asr/raw-chunks').glob('*.json'))) for c in ('A', 'B'))
        unresolved = sum(s['decision']['status'] != 'resolved' for s in self.store.segments(job_id))
        return {'job_id': job_id, **state, 'audio_path': job['request']['audio_path'],
                'profile': job['request']['profile']['id'], 'revision': job['revision'],
                'completed_asr_chunks': completed, 'total_asr_chunks': state.get('chunks', 0) * state.get('channel_count', 0),
                'unresolved_segments': unresolved, 'log_path': str(folder / 'job.log')}

    def retry_transcription(self, job_id):
        state = self.get_status(job_id)
        if state['status'] not in ('failed', 'interrupted', 'cancelled'):
            raise ValueError('Only failed, interrupted or cancelled jobs can be resumed')
        if alive(state.get('worker_pid'), state.get('worker_birth')):
            raise ValueError('Previous worker is still alive; cancel it before retrying')
        self.store.state(job_id, status='queued', stage='queued', error=None)
        self._spawn(job_id)
        return self.get_status(job_id)

    def cancel_transcription(self, job_id):
        state = self.store.get(job_id)['state']
        if state['status'] not in ('queued', 'running', 'interrupted', 'failed'):
            raise ValueError('Job is not active')
        if alive(state.get('worker_pid'), state.get('worker_birth')):
            parent = psutil.Process(state['worker_pid'])
            processes = parent.children(recursive=True) + [parent]
            for process in reversed(processes):
                try:
                    process.terminate()
                except psutil.NoSuchProcess:
                    pass
            _, pending = psutil.wait_procs(processes, timeout=5)
            for process in pending:
                try:
                    process.kill()
                except psutil.NoSuchProcess:
                    pass
            psutil.wait_procs(pending, timeout=5)
        self.store.state(job_id, status='cancelled')
        return self.get_status(job_id)

    def get_segments(self, job_id, offset=0, limit=5, unresolved_only=False, reverse_order=False):
        if offset < 0 or not 1 <= limit <= 20:
            raise ValueError('Offset must be nonnegative; limit 1-20')
        snapshot = self.store.snapshot(job_id)
        if snapshot['state']['status'] != 'ready':
            raise ValueError('Transcription is not ready; check get_status')
        all_segments = snapshot['segments']
        selected = [s for s in all_segments if not unresolved_only or s['decision']['status'] != 'resolved']
        page = selected[offset:offset + limit]
        matched = set()
        for segment in page:
            a, b = segment['window']
            segment['reference_ids'] = [r['id'] for r in snapshot['references']
                                        if r['window'] is not None and r['window'][0] < b and a < r['window'][1]]
            matched.update(segment['reference_ids'])
            i = next(i for i, s in enumerate(all_segments) if s['id'] == segment['id'])
            segment['neighbours'] = [{'id': s['id'], 'variants': {c: v['text'] for c, v in s['variants'].items()}}
                                     for s in all_segments[max(0, i - 1):i + 2] if s['id'] != segment['id']]
            if reverse_order:
                segment['variants'] = dict(reversed(list(segment['variants'].items())))
        return {'job_id': job_id, 'revision': snapshot['revision'], 'context': snapshot['request']['context'],
                'references': [{k: r[k] for k in ('id', 'title', 'source', 'window', 'usage', 'confirmed_by')}
                               for r in snapshot['references'] if r['id'] in matched],
                'reference_count': len(snapshot['references']),
                'other_reference_count': len(snapshot['references']) - len(matched),
                'segments': page, 'total': len(selected), 'next_offset': offset + len(page) if offset + len(page) < len(selected) else None,
                'confidence_note': 'Greedy entropy proxy, not a calibrated probability of the beam text'}

    def save_references(self, job_id, references, expected_revision):
        version, saved = self.store.save_references(job_id, references, expected_revision)
        return {'job_id': job_id, 'revision': version, 'saved': len(saved), 'reference_ids': [r['id'] for r in saved]}

    def get_references(self, job_id, offset=0, limit=5):
        if offset < 0 or not 1 <= limit <= 20:
            raise ValueError('Offset must be nonnegative; limit 1-20')
        snapshot = self.store.snapshot(job_id)
        materials = snapshot['references']
        page = materials[offset:offset + limit]
        ids = {r['id'] for r in page}
        history = [{**event, 'changes': [r for r in event['changes'] if r['id'] in ids]}
                   for event in snapshot['reference_history'] if any(r['id'] in ids for r in event['changes'])]
        return {'job_id': job_id, 'revision': snapshot['revision'], 'references': page, 'history': history,
                'total': len(materials), 'next_offset': offset + len(page) if offset + len(page) < len(materials) else None}

    def get_raw_transcripts(self, job_id, channel='A', offset=0, limit=5):
        if channel not in ('A', 'B') or offset < 0 or not 1 <= limit <= 20:
            raise ValueError('Invalid channel or pagination')
        data = read(self.store.folder(job_id) / 'channels.json')
        if channel not in data:
            raise ValueError('Channel is unavailable (mono recordings have A only)')
        rows = data[channel]['transcripts']
        return {'channel': channel, 'total': len(rows), 'transcripts': rows[offset:offset + limit],
                'next_offset': offset + limit if offset + limit < len(rows) else None}

    def get_audio_clip(self, job_id, start, end, channel='both'):
        job = self.store.get(job_id)
        request = job['request']
        duration = job['state'].get('duration', min(request['audio_info']['duration'], request['limit_seconds'] or float('inf')))
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= duration or end - start > 120:
            raise ValueError('Clip must be within the job timeline and at most 120 seconds')
        if channel not in ('A', 'B', 'both') or channel == 'B' and request['audio_info']['channels'] == 1:
            raise ValueError('Channel is unavailable')
        source = Path(request['audio_path'])
        if digest(source) != request['audio_sha256']:
            raise ValueError('Source audio changed')
        path = self.store.folder(job_id) / 'clips' / (identity([start, end, channel])[:16] + '.wav')
        path.parent.mkdir(exist_ok=True)
        if not path.exists():
            temporary = path.with_suffix('.tmp.wav')
            command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-ss', str(start), '-i', str(source),
                       '-t', str(end - start), '-map', '0:a:0', '-vn', '-ar', '16000']
            if channel != 'both':
                command += ['-af', 'pan=mono|c0=c' + ('0' if channel == 'A' else '1')]
            subprocess.run(command + ['-c:a', 'pcm_s16le', str(temporary)], check=True)
            temporary.replace(path)
        return {'path': str(path), 'window': [start, end], 'channel': channel,
                'note': 'Original audio, not speaker-masked; open locally to listen'}

    def save_revision(self, job_id, changes, expected_revision):
        if self.store.get(job_id)['state']['status'] != 'ready':
            raise ValueError('Transcription is not ready')
        version = self.store.save(job_id, changes, expected_revision)
        return {'job_id': job_id, 'revision': version, 'saved': len(changes)}

    def export_transcript(self, job_id, allow_draft=False):
        job = self.store.snapshot(job_id)
        if job['state']['status'] != 'ready':
            raise ValueError('Transcription is not ready')
        unresolved = [s['id'] for s in job['segments'] if s['decision']['status'] != 'resolved']
        if unresolved and not allow_draft:
            raise ValueError('Unresolved segments remain; answer questions or export with allow_draft=true')
        folder = self.store.folder(job_id) / 'exports'
        folder.mkdir(exist_ok=True)
        base = folder / f'rev-{job["revision"]:06d}'
        markdown = ['# Транскрипция лекции', '', 'Черновик: есть неразрешённые места.' if unresolved else 'Все решения сохранены.', '']
        plain, subtitles = [], []
        for i, segment in enumerate(job['segments'], 1):
            text = segment['decision']['text'] if segment['decision']['status'] == 'resolved' else '[требует уточнения]'
            if not text.strip():
                continue
            a, b = segment['window']
            markdown += [f'## {stamp(a)} - {stamp(b)}', '', text, '']
            if segment['decision']['status'] != 'resolved':
                markdown += [f"Вариант {c}: {v['text']}" for c, v in segment['variants'].items()]
                markdown += ['Вопрос: ' + (segment['decision']['question'] or 'Какой вариант соответствует записи?'), '']
            plain.append(text)
            subtitles.append(f'{len(subtitles) + 1}\n{stamp(a)} --> {stamp(b)}\n{text}\n')
        from .cli import atomic_text
        paths = {'markdown': base.with_suffix('.md'), 'text': base.with_suffix('.txt'),
                 'srt': base.with_suffix('.srt'), 'audit': base.with_suffix('.json')}
        atomic_text(paths['markdown'], '\n'.join(markdown))
        atomic_text(paths['text'], '\n\n'.join(plain) + '\n')
        atomic_text(paths['srt'], '\n'.join(subtitles))
        write(paths['audit'], {**job, 'unresolved': unresolved, 'timing': 'segment bounds; edited words are not re-aligned'})
        return {'job_id': job_id, 'revision': job['revision'], 'draft': bool(unresolved),
                'paths': {k: str(v) for k, v in paths.items()}, 'timing': 'SRT uses segment bounds'}
