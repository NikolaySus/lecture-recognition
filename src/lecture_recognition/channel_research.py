"""Resumable, leakage-controlled channel research and annotation preparation."""

import argparse
import hashlib
import itertools
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .channel_curve_full import alignment_key
from .channel_utility import (
    MERGERS,
    THRESHOLDS,
    acoustic_statistics,
    aggregate_scores,
    assemble,
    channel_quality_metrics,
    likelihood_margin,
    make_layout,
    select_channels,
    shift_layout,
    validation_windows,
)
from .dynamic_study import PARENTS
from .evaluation import edit_score, score_case, select_text
from .experiments import identity
from .model_benchmark import ROOT, extra_metrics, read, worker, write
from .model_benchmark import layout as historical_layout

PLAN = ROOT / 'docs/channel-selection-research-plan.md'
REVIEW = ROOT / 'record/20260925_101716.channel-validation.review.md'


class CUDAUnavailable(RuntimeError):
    """A resumable environmental block, not a failed model configuration."""


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise CUDAUnavailable('CUDA unavailable in this environment; cached work preserved; resume on a CUDA-enabled host')


def decoder(config):
    return {k: v for k, v in config.items() if k not in ('audio_variant', 'maximum', 'context', 'offline')}


def rank(score, elapsed=0., seams=0):
    total = score['total']
    return (total['wer'], total['number_errors'] + total['negation_errors'], seams, total['cer'], elapsed)


class Research:
    def __init__(self, args):
        self.args = args
        self.source = args.source.resolve()
        self.metadata = read(self.source / 'metadata.json')
        dynamic = Path(self.metadata['source'])
        historical = read(Path(read(dynamic / 'metadata.json')['source']) / 'metadata.json')
        self.prepared = read(Path(historical['source']) / 'prepared.json')
        self.reference = read(self.source / 'reference.json')
        cases = [read(p) for p in (self.source / 'cases').glob('*/case.json')]
        self.parents = {model: next(c for c in cases if c['label'] == label and c['status'] == 'ok')
                        for model, label in PARENTS.items()}
        if digest(args.audio) != self.metadata['source_audio_sha256']:
            raise ValueError('Source audio changed')
        self.audio_source = args.audio.resolve()
        files = [Path(__file__), ROOT / 'src/lecture_recognition/channel_utility.py',
                 ROOT / 'src/lecture_recognition/channel_diagnostics.py', ROOT / 'src/lecture_recognition/seam_merge.py', ROOT / 'src/lecture_recognition/timeline.py',
                 ROOT / 'src/lecture_recognition/evaluation.py', ROOT / 'src/lecture_recognition/models.py', ROOT / 'scripts/asr_worker.py',
                 ROOT / 'src/lecture_recognition/gigaam_decoding.py', ROOT / 'experiments/gigaam/uv.lock',
                 ROOT / 'scripts/brouhaha_worker.py', ROOT / 'src/lecture_recognition/channel_transcribe.py', ROOT / 'experiments/brouhaha/uv.lock']
        meta = {'series': 'channel-research-v1', 'source': str(self.source),
                'audio_sha256': self.metadata['source_audio_sha256'],
                'reference_sha256': digest(self.source / 'reference.json'),
                'prepared_sha256': digest(Path(historical['source']) / 'prepared.json'),
                'code': {str(p.relative_to(ROOT)): digest(p) for p in files},
                'test': {'count': 12, 'seconds': 60, 'start': 345.92}, 'version': 1,
                'merger_override': getattr(args, 'merger', None)}
        if getattr(args, 'fixed_layout', None):
            meta['fixed_layout'] = args.fixed_layout
        pointer = args.output.resolve() / 'latest.json'
        previous_root = Path(read(pointer)['run']) if pointer.exists() else None
        self.root = args.output.resolve() / identity(meta)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        write(self.root / 'metadata.json', meta)
        write(self.root / 'reference.json', self.reference)
        write(args.output.resolve() / 'latest.json', {'run': str(self.root)})
        if previous_root and previous_root != self.root:
            previous_meta = read(previous_root / 'metadata.json')
            if previous_meta['audio_sha256'] == meta['audio_sha256'] and previous_meta['prepared_sha256'] == meta['prepared_sha256']:
                for name in ('reference-assistance.json', 'reference-review-check.20261006.json', 'validation-bans.json'):
                    prior = previous_root / name
                    if prior.exists() and not (self.root / name).exists():
                        write(self.root / name, read(prior))
        self.diagnostic_costs = {}
        self.status = read(self.root / 'status.json') if (self.root / 'status.json').exists() else {}
        print('RESEARCH', self.root, flush=True)
        approval = self.root / 'reference-review-check.20261006.json'
        if approval.exists() and REVIEW.exists():
            check = read(approval)
            if check.get('reference_confirmed_by') == 'user' and check.get('clean_sha256') == digest(REVIEW):
                self.state('reference', 'complete', result='12/12 references confirmed by user; provenance preserved')

    def state(self, stage, status, **details):
        print(stage, status, details.get('reason') or details.get('result', ''), flush=True)
        self.status[stage] = {'status': status, 'pid': os.getpid(), 'updated': time.strftime('%Y-%m-%d %H:%M:%S'), **details}
        write(self.root / 'status.json', self.status)
        if PLAN.exists():
            text = PLAN.read_text()
            start, end = '<!-- research-status:start -->', '<!-- research-status:end -->'
            rows = ['| Этап | Статус | Результат / причина |', '|---|---|---|']
            for name, item in self.status.items():
                description = str(item.get('reason', item.get('result', ''))).replace('|', '/')
                rows.append(f"| {name} | {item['status']} | {description} |")
            body = '\nЗапуск: `' + str(self.root.relative_to(ROOT)) + '`\n\n' + '\n'.join(rows) + '\n'
            checkpoints = {'P02': 'prepare', 'P03': 'audit', 'P04': 'chunk-study', 'P05': 'develop',
                           'P06': 'freeze', 'P07': 'validate', 'P08': 'export'}
            import re
            for checkpoint, name in checkpoints.items():
                if name in self.status:
                    status_value = self.status[name]['status']
                    if checkpoint == 'P06' and self.status.get('validate', {}).get('reason', '').startswith('manual_reference'):
                        status_value = 'blocked'
                    text = re.sub(r'(\| ' + checkpoint + r' \| [^|]+ \| )[^|]+', lambda m: m[1] + status_value + ' ', text)
            if start in text and end in text:
                text = text.split(start)[0] + start + body + end + text.split(end)[1]
                PLAN.write_text(text)

    def prepare(self):
        self.state('prepare', 'running')
        folder = self.root / 'audio'
        folder.mkdir(exist_ok=True)
        for index, channel in enumerate(('left', 'right')):
            path = folder / (channel + '.f32')
            info = path.with_suffix('.json')
            if not path.exists():
                for old in self.args.output.resolve().glob('*/metadata.json'):
                    prior = read(old)
                    previous = old.parent / 'audio' / path.name
                    previous_info = previous.with_suffix('.json')
                    if (prior['audio_sha256'] == self.metadata['source_audio_sha256']
                            and prior['prepared_sha256'] == read(self.root / 'metadata.json')['prepared_sha256']
                            and previous.exists() and previous_info.exists()):
                        os.link(previous, path)
                        write(info, read(previous_info))
                        break
            if path.exists() and info.exists():
                if digest(path) != read(info)['sha256']:
                    raise ValueError('Cached channel changed')
                continue
            temporary = path.with_suffix('.tmp')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.audio_source),
                            '-af', f'pan=mono|c0=c{index}', '-ar', str(RATE), '-f', 'f32le', str(temporary)], check=True)
            path_wave = np.memmap(temporary, dtype='<f4', mode='r+')
            if len(path_wave) != round(self.prepared['duration'] * RATE):
                raise ValueError('Channel length mismatch')
            for a, b in self.prepared['excluded']:
                path_wave[round(a * RATE):round(b * RATE)] = 0
            path_wave.flush()
            del path_wave
            temporary.replace(path)
            write(info, {'sha256': digest(path), 'samples': path.stat().st_size // 4})
        bans_path = self.root / 'validation-bans.json'
        bans = read(bans_path) if bans_path.exists() else []
        windows = validation_windows(self.prepared['duration'], self.prepared['regions'],
                                     self.prepared['excluded'] + [b['window'] for b in bans])
        if (self.root / 'validation-windows.json').exists() and read(self.root / 'validation-windows.json') != windows:
            raise ValueError('Validation windows changed')
        write(self.root / 'validation-windows.json', windows)
        clips = ROOT / 'output/channel-validation-clips'
        clips.mkdir(parents=True, exist_ok=True)
        for item in windows:
            path = clips / (item['id'] + '.wav')
            a, b = item['input_window']
            if not path.exists():
                subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(self.audio_source),
                                '-ss', str(a), '-t', str(b - a), '-c:a', 'pcm_s16le', str(path)], check=True)
            item['wav'] = str(path)
            item['wav_sha256'] = digest(path)
        write(self.root / 'validation-clips.json', windows)
        if not REVIEW.exists():
            text = '# Независимый эталон S001–S012\n\n'
            text += ('Транскрибируйте только центральные 60 секунд (02–62 с внутри WAV).\n'
                     'Сохраняйте повторы, отрицания и числа словами. Не используйте гипотезы ASR.\n'
                     'Для каждого фрагмента установите `valid: yes` только после проверки границ,\n'
                     'лектора и разборчивости. Непригодные места: `valid: no`, причина ниже.\n\n')
            for item in windows:
                a, b = item['window']
                text += (f"## {item['id']}\n\nВремя: {a:.2f}–{b:.2f} с\n\n"
                         f"Аудио: [WAV](../output/channel-validation-clips/{item['id']}.wav)\n\n"
                         'valid: pending\n\nПричина: \n\nТранскрипция:\n\n<!-- Введите ручной эталон -->\n\n')
            REVIEW.write_text(text)
        self.state('prepare', 'complete', result='12 × 60 с, WAV с контекстом, пустой ручной эталон')
        if not self.status.get('validate'):
            self.state('validate', 'blocked', reason='profiles_not_frozen; reference_ready'
                       if self.status.get('reference', {}).get('status') == 'complete' else 'manual_reference_pending')

    def replace_test(self):
        """Replace an unsuitable blind fragment before profiles/test windows freeze."""
        import re
        if (self.root / 'frozen.json').exists():
            raise ValueError('Test windows already frozen; start a new --output series before replacing')
        if not self.args.test_id or not self.args.reason:
            raise ValueError('--test-id Sxxx and --reason are required')
        old = read(self.root / 'validation-windows.json')
        target = next((w for w in old if w['id'] == self.args.test_id), None)
        if target is None:
            raise ValueError('Unknown test fragment')
        path = self.root / 'validation-bans.json'
        bans = read(path) if path.exists() else []
        bans.append({'id': target['id'], 'window': target['input_window'], 'reason': self.args.reason})
        windows = validation_windows(self.prepared['duration'], self.prepared['regions'],
                                     self.prepared['excluded'] + [b['window'] for b in bans])
        if any(a != b for a, b in zip(old, windows) if a['id'] != target['id']):
            raise ValueError('Replacement would change another fragment')
        replacement = next(w for w in windows if w['id'] == target['id'])
        section = re.search(r'^## ' + target['id'] + r'\s*\n(.*?)(?=^## |\Z)', REVIEW.read_text(), re.M | re.S)
        if not section:
            raise ValueError('Manual template section is missing')
        write(path, bans)
        write(self.root / 'validation-windows.json', windows)
        write(self.root / 'replaced-references' / (target['id'] + '-' + str(len(bans)) + '.json'),
              {'previous': target, 'replacement': replacement, 'reason': self.args.reason, 'previous_section': section[0]})
        a, b = replacement['window']
        content = (f"## {target['id']}\n\nВремя: {a:.2f}–{b:.2f} с\n\n"
                   f"Аудио: [WAV](../output/channel-validation-clips/{target['id']}.wav)\n\n"
                   'valid: pending\n\nПричина: \n\nТранскрипция:\n\n<!-- Введите ручной эталон -->\n\n')
        text = REVIEW.read_text()
        REVIEW.write_text(text[:section.start()] + content + text[section.end():])
        (ROOT / 'output/channel-validation-clips' / (target['id'] + '.wav')).unlink(missing_ok=True)
        self.prepare()
        self.state('replace-test', 'complete', result=target['id'] + ': ' + self.args.reason)

    def audio(self, channel):
        return self.root / 'audio' / (channel + '.f32')

    def historical(self, model):
        case = self.parents[model]
        raw = Path(case['evaluation_source']).parent
        transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in read(raw / 'alignment/request.json')['transcripts']]
        aligned = [read(raw / 'alignment-cache/alignment' / (alignment_key(t) + '.json')) for t in transcripts]
        return transcripts, aligned

    def seed_chunks(self, model, chunks, pcm_hashes, folder):
        """Reuse identical PCM independent of core ownership and layout labels."""
        config = decoder(self.parents[model]['config'])
        version = identity({'decoder': config, 'worker': digest(ROOT / 'scripts/asr_worker.py'),
                            'decoding': digest(ROOT / 'src/lecture_recognition/gigaam_decoding.py')})[:16]
        cache = self.args.output.resolve() / 'chunk-cache' / version
        cache.mkdir(parents=True, exist_ok=True)
        wanted = set(pcm_hashes)
        requests = list(self.args.output.resolve().glob('*/inference/*/asr/request.json'))
        if self.parents[model].get('evaluation_source'):
            requests.append(Path(self.parents[model]['evaluation_source']).parent / 'asr/request.json')
        for path in requests:
            request = read(path)
            if decoder(request['config']) != config:
                continue
            raw_wave = np.memmap(request['audio'], dtype='<f4', mode='r')
            for index, chunk in enumerate(request['chunks']):
                saved = path.parent / 'raw-chunks' / f'{index:05d}.json'
                if not saved.exists():
                    continue
                pcm = hashlib.sha256(raw_wave[round(chunk['start'] * RATE):round(chunk['end'] * RATE)].tobytes()).hexdigest()
                if pcm not in wanted or (cache / (pcm + '.json')).exists():
                    continue
                value = read(saved)
                write(cache / (pcm + '.json'), {'pcm_sha256': pcm, 'config': config, 'source': str(saved),
                      'text': value['text'], 'limit_hits': value.get('limit_hits', 0), 'native_words': value.get('native_words')})
        target = folder / 'asr/raw-chunks'
        target.mkdir(parents=True, exist_ok=True)
        seeded = []
        for index, (chunk, pcm) in enumerate(zip(chunks, pcm_hashes)):
            path = cache / (pcm + '.json')
            if not path.exists():
                continue
            value = read(path)
            item = {**value, 'chunk': chunk, 'seconds': 0.}
            saved = target / f'{index:05d}.json'
            if not saved.exists():
                write(saved, item)
            seeded.append(item)
        return seeded

    def infer(self, model, channel, chunks, split='dev'):
        config = decoder(self.parents[model]['config'])
        wave = np.memmap(self.audio(channel), dtype='<f4', mode='r')
        pcm_hashes = [hashlib.sha256(wave[round(c['start'] * RATE):round(c['end'] * RATE)].tobytes()).hexdigest()
                      for c in chunks]
        key = identity({'model': config, 'pcm': pcm_hashes, 'chunks': chunks})[:16]
        reusable_folder = None
        cache_code = ('scripts/asr_worker.py', 'src/lecture_recognition/gigaam_decoding.py', 'experiments/gigaam/uv.lock',
                      'src/lecture_recognition/models.py')
        # Raw inference is independent of the selector and report implementation.
        for other in self.args.output.resolve().glob('*/metadata.json'):
            if other.parent == self.root:
                continue
            previous = read(other)
            if previous['audio_sha256'] != self.metadata['source_audio_sha256'] or any(
                    previous['code'].get(p) != digest(ROOT / p) for p in cache_code):
                continue
            cached = other.parent / 'inference' / key / 'result.json'
            if cached.exists() and read(cached)['status'] == 'ok':
                return read(cached)
            if (cached.parent / 'asr/request.json').exists():
                reusable_folder = cached.parent
        folder = reusable_folder or self.root / 'inference' / key
        summary = folder / 'result.json'
        if summary.exists() and read(summary)['status'] == 'ok':
            return read(summary)
        folder.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            old_transcripts, old_aligned = self.historical(model)
            if channel == 'left' and chunks == [t['chunk'] for t in old_transcripts]:
                # Require exact channel samples against the old ASR source before reuse.
                raw = Path(self.parents[model]['evaluation_source']).parent
                old_audio = np.memmap(read(raw / 'asr/request.json')['audio'], dtype='<f4', mode='r')
                if all(wave[round(c['start'] * RATE):round(c['end'] * RATE)].tobytes()
                       == old_audio[round(c['start'] * RATE):round(c['end'] * RATE)].tobytes() for c in chunks):
                    result = {'status': 'ok', 'transcripts': old_transcripts, 'aligned': old_aligned,
                              'reused_historical': True, 'seconds': 0., 'split': split}
                    write(summary, result)
                    return result
            seeded = self.seed_chunks(model, chunks, pcm_hashes, folder)
            if len(seeded) == len(chunks):
                write(folder / 'asr/result.json', {'status': 'ok', 'transcripts': seeded, 'fully_reused_pcm': True})
            if len(seeded) != len(chunks):
                require_cuda()
            asr_audio = read(folder / 'asr/request.json')['audio'] if (folder / 'asr/request.json').exists() else str(self.audio(channel))
            asr = read(folder / 'asr/result.json') if len(seeded) == len(chunks) else worker({'operation': 'asr', 'audio': asr_audio,
                          'config': {**config, 'offline': True}, 'chunks': chunks}, folder / 'asr',
                         'gigaam', self.args.retry_failed)
            if asr['status'] != 'ok':
                raise RuntimeError(asr.get('error', 'ASR failed'))
            transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in asr['transcripts']]
            aligned_cache = folder / 'alignment/result.json'
            if not aligned_cache.exists() or (self.args.retry_failed and read(aligned_cache)['status'] != 'ok'):
                require_cuda()
            aligned_result = worker({'operation': 'align', 'audio': self.prepared['audio'],
                                     'transcripts': transcripts, 'cache': str(folder / 'alignment-cache')},
                                    folder / 'alignment', retry=self.args.retry_failed)
            if aligned_result['status'] != 'ok':
                raise RuntimeError(aligned_result.get('error', 'Alignment failed'))
            aligned = [read(folder / 'alignment-cache/alignment' / (alignment_key(t) + '.json')) for t in transcripts]
            result = {'status': 'ok', 'transcripts': transcripts, 'aligned': aligned,
                      'seconds': time.monotonic() - started, 'split': split}
        except CUDAUnavailable:
            raise
        except Exception as exc:
            result = {'status': 'error', 'error': str(exc), 'split': split}
        write(summary, result)
        return result

    def audit(self):
        self.state('audit', 'running')
        findings = []
        merger_results = []
        for model in PARENTS:
            transcripts, aligned = self.historical(model)
            for method in MERGERS:
                repairs = []
                words = assemble(aligned, method, repairs)
                score = extra_metrics(score_case(words, self.reference, 2))
                merger_results.append({'model': model, 'merger': method, 'score': score, 'repairs': repairs,
                                       'confirmation': 'automated_suspicions_only'})
                for i in range(1, len(aligned)):
                    left, right = aligned[i - 1:i + 1]
                    a, b = right['chunk']['start'], left['chunk']['end']
                    findings.append({'model': model, 'merger': method, 'index': i,
                                     'window': [a, b], 'left': transcripts[i - 1], 'right': transcripts[i],
                                     'aligned_left': left, 'aligned_right': right,
                                     'merged_overlap': [w for w in words if w['start'] < b and w['end'] > a],
                                     'status': 'unreviewed', 'suspicions': [r for r in repairs
                                         if r.get('window', r.get('overlap', [0, 0]))[0] < b
                                         and r.get('window', r.get('overlap', [0, 0]))[1] > a]})
            control = next(r for r in merger_results if r['model'] == model and r['merger'] == 'current')
            if control['score'] != self.parents[model]['score']:
                raise ValueError('Historical score cannot be reproduced: ' + model)
        write(self.root / 'audit.json', {'seams': findings, 'mergers': merger_results})
        lines = ['# Аудит исторических стыков', '', 'Подозрения не являются подтверждёнными ошибками. '
                 'Для подтверждения прослушайте исходный интервал и заполните seam-review.json.', '']
        for row in findings:
            a, b = row['window']
            lines += [f"## {row['model']} / {row['merger']} / стык {row['index']} ({a:.2f}-{b:.2f} с)",
                      '', 'Слева: ' + row['left']['text'], '', 'Справа: ' + row['right']['text'], '',
                      'После объединения в перекрытии: ' + ' '.join(w['text'] for w in row['merged_overlap']), '',
                      'Подозрения: ' + str(row['suspicions']), '']
        (self.root / 'audit.md').write_text('\n'.join(lines))
        manual = self.root / 'seam-review.json'
        if not manual.exists():
            write(manual, {'instructions': 'Confirm only after listening; annotate model/index/type and evidence.',
                           'confirmed': [], 'review_complete': False})
        self.state('audit', 'complete', result=f'{len(findings)} seam/method records; controls reproduced; manual review pending')

    def confirmed_seams(self, model, merger, layout='historical'):
        review = read(self.root / 'seam-review.json')
        return sum(c.get('model') == model and c.get('merger') == merger
                   and c.get('layout', 'historical') == layout and c.get('unresolved', True)
                   for c in review['confirmed'])

    def chunk_study(self):
        if not self.audio('left').exists() or not (self.root / 'validation-windows.json').exists():
            self.prepare()
        if not (self.root / 'audit.json').exists():
            self.audit()
        self.state('chunk-study', 'running')
        smokes = []
        for model in PARENTS:
            for seconds in (30, 45):
                chunks = [{'start': 10., 'end': 10. + seconds, 'core_start': 10., 'core_end': 10. + seconds}]
                result = self.infer(model, 'left', chunks)
                smokes.append({'model': model, 'seconds': seconds, 'status': result['status'], 'error': result.get('error')})
                write(self.root / 'smoke.json', smokes)
                if result['status'] != 'ok':
                    self.state('chunk-study', 'blocked', reason='30/45s smoke failed; no silent fallback')
                    return
        mergers = read(self.root / 'audit.json')['mergers']
        results, frozen = [], {}
        for model in PARENTS:
            choices = [r for r in mergers if r['model'] == model]
            selected_merger = getattr(self.args, 'merger', None) or min(choices, key=lambda r: (*rank(r['score'], seams=self.confirmed_seams(model, r['merger'])), MERGERS.index(r['merger'])))['merger']
            parent_transcripts, _ = self.historical(model)
            historic = [t['chunk'] for t in parent_transcripts]
            wave = np.memmap(self.audio('left'), dtype='<f4', mode='r')
            layouts = [('historical', historic)]
            for maximum, context, mode in itertools.product((20, 30, 45), (1, 2), ('regular', 'vad')):
                chunks = make_layout(wave, self.prepared['prefix_end'], maximum, context, mode,
                                     self.prepared['regions'])
                layouts.append((f'{mode}-m{maximum}-c{context}', chunks))
            for label, chunks in layouts:
                inference = self.infer(model, 'left', chunks)
                item = {'model': model, 'layout': label, 'chunks': chunks, 'merger': selected_merger,
                        'status': inference['status']}
                if inference['status'] == 'ok':
                    item.update(score=extra_metrics(score_case(assemble(inference['aligned'], selected_merger), self.reference, 2)),
                                seconds=inference['seconds'])
                else:
                    item['error'] = inference['error']
                results.append(item)
                write(self.root / 'chunk-study.json', {'complete': False, 'cases': results})
                self.state('chunk-study', 'running', result=f'{len(results)}/26 layouts evaluated')
            successful = [r for r in results if r['model'] == model and r['status'] == 'ok']
            winner = min(successful, key=lambda r: (*rank(r['score'], seams=self.confirmed_seams(model, r['merger'], r['layout'])), r['layout'] != 'historical', r['seconds']))
            shifts = []
            if winner['layout'] != 'historical':
                mode, maximum, context = winner['layout'].split('-')
                maximum, context = int(maximum[1:]), int(context[1:])
                for shift in (-2, 2):
                    chunks = make_layout(wave, self.prepared['prefix_end'], maximum, context, mode,
                                         self.prepared['regions'], shift=shift)
                    inference = self.infer(model, 'left', chunks)
                    shifts.append({'shift': shift, 'status': inference['status'],
                                   'score': extra_metrics(score_case(assemble(inference['aligned'], selected_merger), self.reference, 2))
                                   if inference['status'] == 'ok' else None})
            else:
                for shift in (-2, 2):
                    chunks = shift_layout(historic, self.prepared['prefix_end'], 20, 1, shift)
                    inference = self.infer(model, 'left', chunks)
                    shifts.append({'shift': shift, 'status': inference['status'],
                                   'score': extra_metrics(score_case(assemble(inference['aligned'], selected_merger), self.reference, 2))
                                   if inference['status'] == 'ok' else None,
                                   'note': 'input and ownership grid translated; short edge cores preserve coverage'})
            frozen[model] = {**winner, 'shifts': shifts}
        write(self.root / 'chunk-study.json', {'complete': all(r['status'] == 'ok' for r in results), 'cases': results})
        if any(r['status'] != 'ok' for r in results) or any(s['status'] != 'ok' for v in frozen.values() for s in v['shifts']):
            self.state('chunk-study', 'blocked', reason='failed_layouts; retry before freezing')
            return
        write(self.root / 'chunk-frozen.json', {'models': frozen, 'reference': digest(self.source / 'reference.json')})
        self.state('chunk-study', 'complete', result='26 layouts; finalists ±2s; frozen per model')

    def historical_baseline(self):
        """Lock the user-selected control without selecting layouts on test data."""
        if getattr(self.args, 'fixed_layout', None) != 'historical' or self.args.merger != 'current':
            raise ValueError('Historical baseline requires --fixed-layout historical --merger current')
        self.prepare()
        models = {}
        for model in PARENTS:
            transcripts, aligned = self.historical(model)
            score = extra_metrics(score_case(assemble(aligned, 'current'), self.reference, 2))
            if score != self.parents[model]['score']:
                raise ValueError('Historical baseline score changed: ' + model)
            models[model] = {'model': model, 'layout': 'historical', 'merger': 'current',
                             'chunks': [t['chunk'] for t in transcripts], 'score': score,
                             'status': 'ok', 'selection': 'user-fixed; no layout search'}
        value = {'models': models, 'reference': digest(self.source / 'reference.json'),
                 'selection': 'historical layout/current fixed by user after layout validation'}
        path = self.root / 'chunk-frozen.json'
        if path.exists() and read(path) != value:
            raise ValueError('Fixed baseline cannot overwrite a different frozen layout')
        write(path, value)
        self.state('historical-baseline', 'complete', result='historical controls reproduced; raw chunks retained')
        self.state('chunk-study', 'complete', result='user-fixed historical/current; no new layout search')

    def diagnostics(self, model, channel, chunks, competitors):
        inference = self.infer(model, channel, chunks)
        if inference['status'] != 'ok':
            raise RuntimeError(inference['error'])
        config = decoder(self.parents[model]['config'])
        settings = {'operation': 'channel-diagnostics', 'audio': str(self.audio(channel)), 'config': config,
                    'chunks': chunks, 'competitors': competitors}
        folder = self.root / 'diagnostics' / identity(settings)[:16]
        request = {**settings, 'trace_dir': str(folder / 'traces')}
        if not (folder / 'result.json').exists() or read(folder / 'result.json')['status'] != 'ok':
            require_cuda()
        result = worker(request, folder, 'gigaam', self.args.retry_failed)
        if result['status'] != 'ok':
            raise RuntimeError(result.get('error', 'Diagnostics failed'))
        self.diagnostic_costs[str(folder)] = {k: result.get(k) for k in ('seconds', 'peak_vram_gib', 'versions')}
        return result['items']

    def diagnose(self):
        if not (self.root / 'chunk-frozen.json').exists():
            self.state('diagnose', 'blocked', reason='chunk_study_not_frozen')
            return
        self.state('diagnose', 'running')
        frozen = read(self.root / 'chunk-frozen.json')['models']
        data = {}
        for model, layout in frozen.items():
            chunks = layout['chunks']
            left, right = [self.infer(model, channel, chunks) for channel in ('left', 'right')]
            if left['status'] != 'ok' or right['status'] != 'ok':
                self.state('diagnose', 'blocked', reason='channel_asr_failed')
                return
            competitors = {str(i): [lhs['text'], r['text']] for i, (lhs, r) in enumerate(zip(left['transcripts'], right['transcripts']))}
            trace_left = self.diagnostics(model, 'left', chunks, competitors)
            trace_right = self.diagnostics(model, 'right', chunks, competitors)
            # A CTC scorer uses the same bounds even when the RNNT layout differs.
            ctc_left = trace_left if model == 'gigaam-ctc' else self.diagnostics('gigaam-ctc', 'left', chunks, competitors)
            ctc_right = trace_right if model == 'gigaam-ctc' else self.diagnostics('gigaam-ctc', 'right', chunks, competitors)
            rows = []
            for i, (lhs, r) in enumerate(zip(left['transcripts'], right['transcripts'])):
                scores = [trace_left[i]['confidence'], trace_right[i]['confidence']]
                lp, rp = np.load(ctc_left[i]['path']), np.load(ctc_right[i]['path'])
                labels = ctc_left[i]['competitor_labels']
                margin = likelihood_margin(lp['log_probs'], rp['log_probs'], labels[lhs['text']], labels[r['text']], ctc_left[i]['blank'])
                rows.append({'chunk': chunks[i], 'left_text': lhs['text'], 'right_text': r['text'],
                             'left': scores[0], 'right': scores[1], 'margin': margin})
            data[model] = {'rows': rows, 'left': left, 'right': right, 'layout': layout}
        write(self.root / 'diagnose.json', {'models': data, 'brouhaha': 'pending', 'shared_diagnostic_costs': self.diagnostic_costs})
        self.brouhaha()
        self.state('diagnose', 'complete', result='GigaAM traces + CTC margins; Brouhaha status recorded separately')

    def brouhaha(self):
        interpreter = ROOT / 'experiments/brouhaha/.venv/bin/python'
        checkpoint = ROOT / 'experiments/brouhaha/model.json'
        if not interpreter.exists() or not checkpoint.exists():
            self.state('brouhaha', 'blocked', reason='environment_or_checkpoint_missing')
            return
        d = read(self.root / 'diagnose.json')
        try:
            for channel in ('left', 'right'):
                request = {'audio': str(self.audio(channel)), 'checkpoint': read(checkpoint),
                           'output': str(self.root / 'brouhaha' / (channel + '.json')),
                           'windows': [[0., self.prepared['prefix_end']]]}
                request_path = self.root / 'brouhaha' / (channel + '-request.json')
                write(request_path, request)
                subprocess.run([str(interpreter), str(ROOT / 'scripts/brouhaha_worker.py'), str(request_path)], check=True)
            for model in d['models'].values():
                for row in model['rows']:
                    for channel in ('left', 'right'):
                        values = read(self.root / 'brouhaha' / (channel + '.json'))
                        row[channel].update(acoustic_statistics(values['times'], values['speech'],
                            values['snr'], values['c50'], [row['chunk']['start'], row['chunk']['end']]))
            d['brouhaha'] = 'complete'
            write(self.root / 'diagnose.json', d)
            self.state('brouhaha', 'complete')
        except Exception as exc:
            self.state('brouhaha', 'blocked', reason=str(exc))

    def acoustic_trace(self, channel, windows, label):
        checkpoint = ROOT / 'experiments/brouhaha/model.json'
        folder = self.root / 'brouhaha' / label
        request = {'audio': str(self.audio(channel)), 'checkpoint': read(checkpoint),
                   'output': str(folder / (channel + '.json')), 'windows': windows}
        request_path = folder / (channel + '-request.json')
        write(request_path, request)
        subprocess.run([str(ROOT / 'experiments/brouhaha/.venv/bin/python'),
                        str(ROOT / 'scripts/brouhaha_worker.py'), str(request_path)], check=True)
        return read(Path(request['output']))

    def develop(self):
        if not (self.root / 'diagnose.json').exists():
            self.state('develop', 'blocked', reason='diagnostics_missing')
            return
        data = read(self.root / 'diagnose.json')
        results, selected = [], {}
        for model, values in data['models'].items():
            candidates = []
            for policy, thresholds in THRESHOLDS.items():
                if policy in ('snr', 'c50') and data['brouhaha'] != 'complete':
                    continue
                for threshold in thresholds:
                    decisions = select_channels(values['rows'], policy, threshold)
                    aligned = [values[d['channel']]['aligned'][i] for i, d in enumerate(decisions)]
                    score = extra_metrics(score_case(assemble(aligned, values['layout']['merger']), self.reference, 2))
                    candidates.append({'model': model, 'policy': policy, 'threshold': threshold,
                                       'decisions': decisions, 'score': score})
            if data['brouhaha'] == 'complete':
                best = {p: min((c for c in candidates if c['policy'] == p), key=lambda c: (*rank(c['score']), -c['threshold']))
                        for p in ('tsallis', 'c50', 'margin')}
                thresholds = {p: c['threshold'] for p, c in best.items()}
                decisions = select_channels(values['rows'], 'hybrid', hybrid_thresholds=thresholds)
                aligned = [values[d['channel']]['aligned'][i] for i, d in enumerate(decisions)]
                candidates.append({'model': model, 'policy': 'hybrid', 'threshold': 0., 'hybrid_thresholds': thresholds,
                     'decisions': decisions, 'score': extra_metrics(score_case(assemble(aligned, values['layout']['merger']), self.reference, 2))})
            channel_scores = {ch: score_case(assemble(values[ch]['aligned'], values['layout']['merger']), self.reference, 2)
                              for ch in ('left', 'right')}
            for candidate in candidates:
                units = [{'window': group['window'], 'chunks': [r['chunk'] for r in values['rows']],
                          'decisions': candidate['decisions'], 'scores': {
                              'new_left': channel_scores['left']['groups'][group['name']],
                              'new_right': channel_scores['right']['groups'][group['name']],
                              'selector': candidate['score']['groups'][group['name']]}} for group in self.reference['groups']]
                candidate['channel_quality'] = channel_quality_metrics(units)
            winner = min(candidates, key=lambda c: (*rank(c['score']), -c['threshold'], c['policy']))
            selected[model] = {**winner, 'layout': values['layout']}
            results.extend(candidates)
            for channel in ('left', 'right'):
                results.append({'model': model, 'policy': 'new_' + channel, 'threshold': 0.,
                                'score': extra_metrics(score_case(assemble(values[channel]['aligned'], values['layout']['merger']), self.reference, 2)),
                                'control': True})
        for path in (self.source / 'cases').glob('*/case.json'):
            case = read(path)
            if case['status'] == 'ok' and 'Energy fast' not in case['label']:
                results.append({'model': case['config']['model'], 'policy': case['label'], 'threshold': 0.,
                                'score': case['score'], 'control': True})
        write(self.root / 'develop.json', {'cases': results, 'selected': selected, 'brouhaha': data['brouhaha'],
              'complete': data['brouhaha'] == 'complete'})
        self.state('develop', 'complete' if data['brouhaha'] == 'complete' else 'blocked',
                   reason=None if data['brouhaha'] == 'complete' else 'Brouhaha missing; ASR-only results are provisional',
                   result=f'{len(results)} selector cases')

    def freeze(self):
        dev = read(self.root / 'develop.json')
        if not dev['complete']:
            self.state('freeze', 'blocked', reason='complete_development_required')
            return
        path = self.root / 'frozen.json'
        value = {'models': dev['selected'], 'development_sha256': digest(self.root / 'develop.json'),
                 'chunk_sha256': digest(self.root / 'chunk-frozen.json'),
                 'brouhaha_checkpoint': read(ROOT / 'experiments/brouhaha/model.json') if (ROOT / 'experiments/brouhaha/model.json').exists() else None,
                 'test_windows_sha256': digest(self.root / 'validation-windows.json'),
                 'code': read(self.root / 'metadata.json')['code']}
        if path.exists() and read(path) != value:
            raise ValueError('Frozen profile cannot be changed; start a new series')
        write(path, value)
        self.state('freeze', 'complete', result='profiles locked before reading test reference')

    def validation_reference(self):
        import re
        if not REVIEW.exists():
            raise ValueError('manual_reference_pending')
        text = REVIEW.read_text()
        result = {}
        for item in read(self.root / 'validation-windows.json'):
            match = re.search(r'^## ' + item['id'] + r'\s*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
            if not match or not re.search(r'^valid:\s*yes\s*$', match[1], re.M):
                raise ValueError('manual_reference_pending: ' + item['id'])
            if 'window' in item:
                bounds = re.search(r'Время:\s*([0-9.]+)[–-]([0-9.]+)\s*с', match[1])
                if not bounds or [float(bounds[1]), float(bounds[2])] != item['window']:
                    raise ValueError('manual_reference_window_mismatch: ' + item['id'])
            content = match[1].split('Транскрипция:', 1)[-1]
            content = re.sub(r'<!--.*?-->', '', content, flags=re.S).strip()
            if not content:
                raise ValueError('empty_manual_reference: ' + item['id'])
            result[item['id']] = content
        return result

    def validate(self):
        if not (self.root / 'frozen.json').exists():
            self.state('validate', 'blocked', reason='profiles_not_frozen')
            return
        try:
            references = self.validation_reference()
        except ValueError as exc:
            self.state('validate', 'blocked', reason=str(exc))
            return
        self.state('validate', 'running')
        frozen = read(self.root / 'frozen.json')
        if frozen.get('brouhaha_checkpoint') != read(ROOT / 'experiments/brouhaha/model.json'):
            raise ValueError('Frozen Brouhaha checkpoint changed')
        if frozen['test_windows_sha256'] != digest(self.root / 'validation-windows.json'):
            raise ValueError('Frozen test windows changed')
        # Test inference never receives the manual reference. Pipeline reuse is
        # intentionally explicit rather than invoking develop on the test set.
        result = self.validate_profiles(frozen, references)
        write(self.root / 'validation.json', {**result, 'reference_sha256': digest(REVIEW),
              'frozen_sha256': digest(self.root / 'frozen.json'), 'seam_review_sha256': digest(self.root / 'seam-review.json')})
        self.state('validate', 'complete', result='12 independent fragments; no test tuning; admission also requires seam review')

    def validate_profiles(self, frozen, references):
        outcomes = {}
        for model, profile in frozen['models'].items():
            fragments = []
            for item in read(self.root / 'validation-windows.json'):
                start, end = item['input_window']
                chosen = profile['layout']
                label = chosen['layout']
                if label == 'historical':
                    mode, maximum, context = 'vad', 20, 1
                else:
                    mode, maximum, context = label.split('-')
                    maximum, context = int(maximum[1:]), int(context[1:])
                wave = np.memmap(self.audio('left'), dtype='<f4', mode='r')
                layouts = {}
                for name, m, ctx, md in [('historical_left', 20, 1, 'vad'), ('new', maximum, context, mode)]:
                    regions = [(a - start, b - start) for a, b in self.prepared['regions']]
                    if name == 'historical_left' or label == 'historical':
                        mono = np.memmap(self.prepared['audio'], dtype='<f4', mode='r')
                        local = historical_layout(mono[round(start * RATE):round(end * RATE)], end - start, 20, regions)
                    else:
                        local = make_layout(wave[round(start * RATE):round(end * RATE)], end - start, m, ctx, md, regions)
                    layouts[name] = [{k: v + start for k, v in c.items()} for c in local]
                historical = self.infer(model, 'left', layouts['historical_left'], split='test')
                left, right = [self.infer(model, ch, layouts['new'], split='test') for ch in ('left', 'right')]
                if any(v['status'] != 'ok' for v in (historical, left, right)):
                    raise RuntimeError('Test inference failed')
                competitors = {str(i): [lhs['text'], r['text']] for i, (lhs, r) in enumerate(zip(left['transcripts'], right['transcripts']))}
                tl = self.diagnostics(model, 'left', layouts['new'], competitors)
                tr = self.diagnostics(model, 'right', layouts['new'], competitors)
                cl = tl if model == 'gigaam-ctc' else self.diagnostics('gigaam-ctc', 'left', layouts['new'], competitors)
                cr = tr if model == 'gigaam-ctc' else self.diagnostics('gigaam-ctc', 'right', layouts['new'], competitors)
                rows = []
                for i, (lhs, r) in enumerate(zip(left['transcripts'], right['transcripts'])):
                    labels = cl[i]['competitor_labels']
                    lp, rp = np.load(cl[i]['path']), np.load(cr[i]['path'])
                    row = {'left_text': lhs['text'], 'right_text': r['text'], 'left': tl[i]['confidence'], 'right': tr[i]['confidence'],
                           'margin': likelihood_margin(lp['log_probs'], rp['log_probs'], labels[lhs['text']], labels[r['text']], cl[i]['blank'])}
                    if profile['policy'] in ('snr', 'c50', 'hybrid'):
                        for channel in ('left', 'right'):
                            values = self.acoustic_trace(channel, [[start, end]], item['id'])
                            row[channel].update(acoustic_statistics(values['times'], values['speech'], values['snr'], values['c50'],
                                [layouts['new'][i]['start'], layouts['new'][i]['end']]))
                    rows.append(row)
                decisions = select_channels(rows, profile['policy'], profile['threshold'], profile.get('hybrid_thresholds'))
                selected = [({'left': left, 'right': right}[d['channel']])['aligned'][i] for i, d in enumerate(decisions)]
                hypotheses = {'historical_left': assemble(historical['aligned'], 'current'),
                              'new_left': assemble(left['aligned'], chosen['merger']),
                              'new_right': assemble(right['aligned'], chosen['merger']),
                              'selector': assemble(selected, chosen['merger'])}
                seam_id = model + '/' + item['id']
                seam_path = self.root / 'test-seams' / model / (item['id'] + '.json')
                write(seam_path, {'chunks': layouts, 'historical': historical['aligned'],
                                 'left': left['aligned'], 'right': right['aligned'], 'selected': selected})
                review = read(self.root / 'seam-review.json')
                if seam_id not in review.get('test_units', []):
                    review.setdefault('test_units', []).append(seam_id)
                    review['review_complete'] = False
                    write(self.root / 'seam-review.json', review)
                scores = {name: edit_score(references[item['id']], select_text(words, *item['window']), 2)
                          for name, words in hypotheses.items()}
                texts = {name: select_text(words, *item['window']) for name, words in hypotheses.items()}
                fragments.append({'id': item['id'], 'scores': scores, 'hypotheses': texts,
                                  'reference': references[item['id']], 'decisions': decisions, 'window': item['window'],
                                  'chunks': layouts['new'], 'seconds': sum(v.get('seconds', 0.) for v in (historical, left, right))})
            metrics = {name: aggregate_scores([f['scores'][name] for f in fragments]) for name in fragments[0]['scores']}
            total = {name: value['wer'] for name, value in metrics.items()}
            critical = {name: value['number_errors'] + value['negation_errors'] for name, value in metrics.items()}
            seam_review = read(self.root / 'seam-review.json')
            clean = seam_review['review_complete'] and not any(c.get('unresolved', True) for c in seam_review['confirmed'])
            outcomes[model] = {'fragments': fragments, 'wer': total, 'critical_errors': critical, 'metrics': metrics,
               'channel_quality': channel_quality_metrics(fragments), 'seconds': sum(f['seconds'] for f in fragments),
               'selector_passed': total['selector'] < min(total['historical_left'], total['new_left'])
                    and critical['selector'] <= critical['historical_left'] and clean,
               'left_passed': total['new_left'] < total['historical_left']
                    and critical['new_left'] <= critical['historical_left'] and clean}
        return {'models': outcomes}

    def export(self):
        validation = read(self.root / 'validation.json')
        if validation['reference_sha256'] != digest(REVIEW):
            raise ValueError('Manual reference changed; validate again before export')
        frozen = read(self.root / 'frozen.json')
        if (validation['frozen_sha256'] != digest(self.root / 'frozen.json')
                or validation['seam_review_sha256'] != digest(self.root / 'seam-review.json')):
            raise ValueError('Frozen profile or seam review changed; validate again before export')
        exports = []
        for model, result in validation['models'].items():
            if not result['selector_passed'] and not result['left_passed']:
                continue
            profile = {**frozen['models'][model], 'model_config': decoder(self.parents[model]['config']),
                       'validation': result, 'version': 1, 'code': frozen['code'],
                       'ctc_scorer_config': decoder(self.parents['gigaam-ctc']['config']),
                       'brouhaha_checkpoint': read(ROOT / 'experiments/brouhaha/model.json'),
                       'frozen_sha256': digest(self.root / 'frozen.json'),
                       'validation_sha256': digest(self.root / 'validation.json')}
            if not result['selector_passed']:
                profile['policy'] = 'left'
            path = ROOT / 'output/channel-profiles' / (model + '.json')
            write(path, profile)
            exports.append(str(path))
        self.state('export', 'complete', result=f'{len(exports)} admitted profiles')
        write(self.root / 'exports.json', exports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('replace-test', 'run-development', 'prepare', 'audit', 'chunk-study', 'historical-baseline', 'diagnose', 'develop', 'freeze',
                                        'validate', 'report', 'export', 'progress'))
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '.lecture-cache/channel-research')
    parser.add_argument('--audio', type=Path, default=ROOT / 'record/20260925_101716.m4a')
    parser.add_argument('--merger', choices=MERGERS, help='Fix an explicitly selected merger for the layout study')
    parser.add_argument('--fixed-layout', choices=('historical',), help='Use the user-selected historical layout without another search')
    parser.add_argument('--test-id')
    parser.add_argument('--reason')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.stage == 'progress':
        root = Path(read(args.output / 'latest.json')['run'])
        print(json.dumps(read(root / 'status.json'), ensure_ascii=False, indent=2))
        return
    if args.merger is None and (args.output / 'latest.json').exists():
        previous = Path(read(args.output / 'latest.json')['run'])
        args.merger = read(previous / 'metadata.json').get('merger_override')
    if args.fixed_layout is None and (args.output / 'latest.json').exists():
        previous = Path(read(args.output / 'latest.json')['run'])
        args.fixed_layout = read(previous / 'metadata.json').get('fixed_layout')
    if args.stage == 'historical-baseline':
        args.fixed_layout, args.merger = 'historical', 'current'
    if args.fixed_layout == 'historical' and args.merger != 'current':
        parser.error('The historical fixed layout requires --merger current')
    if args.fixed_layout and args.stage in ('chunk-study', 'run-development'):
        parser.error('Fixed-layout series uses historical-baseline then diagnose/develop; no layout search')
    args.source = args.source or Path(read(ROOT / '.lecture-cache/gigaam-channel-curve-full/latest.json')['run'])
    study = Research(args)
    if args.stage == 'run-development':
        for stage in ('prepare', 'audit', 'chunk-study', 'diagnose', 'develop', 'freeze'):
            try:
                getattr(study, stage.replace('-', '_'))()
            except CUDAUnavailable as exc:
                study.state(stage, 'blocked', reason=str(exc))
                raise SystemExit(2)
            except Exception as exc:
                study.state(stage, 'error', reason=str(exc))
                raise
            if study.status[stage]['status'] != 'complete':
                break
        from .channel_research_report import build
        build(study.root)
    elif args.stage == 'report':
        from .channel_research_report import build
        build(study.root)
    else:
        try:
            getattr(study, args.stage.replace('-', '_'))()
        except CUDAUnavailable as exc:
            study.state(args.stage, 'blocked', reason=str(exc))
            raise SystemExit(2)
        except Exception as exc:
            study.state(args.stage, 'error', reason=str(exc))
            raise
