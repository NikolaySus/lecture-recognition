"""Prepare editable, explicitly ASR-assisted references without approving them."""
import argparse
import hashlib
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .channel_curve_full import alignment_key
from .channel_research import REVIEW, decoder
from .channel_utility import assemble
from .evaluation import select_text
from .experiments import identity
from .model_benchmark import ROOT, read, worker, write
from .model_benchmark import layout as historical_layout

INTRO = '''# Проверочный эталон S001–S012

Начальная версия: GigaAM CTC combination (левый канал, beam32, bias4).
Это черновики ASR для ручного редактирования, не подтверждённый эталон.
Такой способ разметки может смещать оценку в пользу модели, создавшей черновик.
Новые фрагменты остаются вне подбора конфигураций, но разметка уже не слепая.

Прослушайте каждый WAV и исправьте **весь** текст, включая пропуски и повторы.
Транскрибируйте только центральные 60 секунд (02–62 с внутри WAV).
Сохраняйте настоящие повторы, отрицания и числа словами.
Для каждого фрагмента установите `valid: yes` только после полной проверки текста,
границ, лектора и разборчивости. Непригодные места: `valid: no`, причина ниже.
Время в обоих форматах относится к исходной записи, а не к отдельному WAV.
'''


def timestamp(seconds):
    value = int((Decimal(str(seconds)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    hours, value = divmod(value, 360000)
    minutes, value = divmod(value, 6000)
    sec, centiseconds = divmod(value, 100)
    return f'{hours:02d}:{minutes:02d}:{sec:02d}.{centiseconds:02d}'


def prepare_review(text):
    if not re.search(r'^## S001\s*$', text, re.M):
        raise ValueError('Review has no S001 section')
    text = INTRO.rstrip() + '\n\n' + text[text.index('## S001'):]
    text = re.sub(r'\n+ЧЧ:ММ:СС:[^\n]*\n*', '\n\n', text)
    return re.sub(r'^(Время:[ \t]*([\d.]+)[–-]([\d.]+) с)[ \t]*\n*',
                  lambda m: m[1] + f'\n\nЧЧ:ММ:СС: {timestamp(m[2])}–{timestamp(m[3])}\n\n', text, flags=re.M)


def insert_draft(text, name, hypothesis):
    if not hypothesis.strip():
        raise ValueError('Empty draft: ' + name)
    match = re.search(r'^## ' + re.escape(name) + r'\s*\n(.*?)(?=^## |\Z)', text, re.M | re.S)
    if not match or 'Транскрипция:' not in match[1]:
        raise ValueError('Missing transcription section: ' + name)
    section = match[0]
    prefix, body = section.split('Транскрипция:', 1)
    # Read the latest file at every insertion. Existing manual work always wins.
    if re.sub(r'<!--.*?-->', '', body, flags=re.S).strip() or re.search(r'^valid:\s*yes\s*$', prefix, re.M):
        return text, False
    origin = 'Черновик: GigaAM CTC combination; требует ручной проверки.\n\n'
    replacement = prefix + origin + 'Транскрипция:\n\n' + hypothesis.strip() + '\n\n'
    return text[:match.start()] + replacement + text[match.end():], True


def infer_fragment(run, prepared, config, item, retry):
    wave = np.memmap(run / 'audio/left.f32', dtype='<f4', mode='r')
    mono = np.memmap(prepared['audio'], dtype='<f4', mode='r')
    start, end = item['input_window']
    local = historical_layout(mono[round(start * RATE):round(end * RATE)], end - start, 20,
                              [(a - start, b - start) for a, b in prepared['regions']])
    chunks = [{k: v + start for k, v in c.items()} for c in local]
    pcm = [hashlib.sha256(wave[round(c['start'] * RATE):round(c['end'] * RATE)].tobytes()).hexdigest() for c in chunks]
    folder = run / 'inference' / identity({'model': config, 'pcm': pcm, 'chunks': chunks})[:16]
    summary = folder / 'result.json'
    if summary.exists() and read(summary)['status'] == 'ok':
        inference = read(summary)
    else:
        started = time.monotonic()
        asr = worker({'operation': 'asr', 'audio': str(run / 'audio/left.f32'),
                      'config': {**config, 'offline': True}, 'chunks': chunks}, folder / 'asr', 'gigaam', retry)
        if asr['status'] != 'ok':
            raise RuntimeError(item['id'] + ': ' + asr.get('error', 'ASR failed'))
        transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in asr['transcripts']]
        aligned = worker({'operation': 'align', 'audio': prepared['audio'], 'transcripts': transcripts,
                          'cache': str(folder / 'alignment-cache')}, folder / 'alignment', retry=retry)
        if aligned['status'] != 'ok':
            raise RuntimeError(item['id'] + ': ' + aligned.get('error', 'Alignment failed'))
        inference = {'status': 'ok', 'transcripts': transcripts, 'aligned': [
                     read(folder / 'alignment-cache/alignment' / (alignment_key(t) + '.json')) for t in transcripts],
                     'split': 'test', 'seconds': time.monotonic() - started, 'purpose': 'annotation_draft'}
        write(summary, inference)
    words = assemble(inference['aligned'], 'current')
    return {'id': item['id'], 'window': item['window'], 'input_window': item['input_window'],
            'config': config, 'layout': 'historical', 'merger': 'current', 'channel': 'left',
            'hypothesis': select_text(words, *item['window']), 'inference': str(folder), 'words': words}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--review', type=Path, default=REVIEW)
    parser.add_argument('--jobs', type=int, choices=(1, 2, 3), default=3)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    run = args.run or Path(read(ROOT / '.lecture-cache/channel-research/latest.json')['run'])
    metadata = read(run / 'metadata.json')
    for path in ('scripts/asr_worker.py', 'src/lecture_recognition/gigaam_decoding.py',
                 'src/lecture_recognition/models.py', 'experiments/gigaam/uv.lock'):
        if digest(ROOT / path) != metadata['code'][path]:
            raise ValueError('Research implementation changed: ' + path)
    source = Path(metadata['source'])
    parent = next(read(p) for p in (source / 'cases').glob('*/case.json')
                  if read(p)['label'] == 'gigaam-ctc combination' and read(p)['status'] == 'ok')
    dynamic = Path(read(source / 'metadata.json')['source'])
    historical = Path(read(dynamic / 'metadata.json')['source'])
    prepared = read(Path(read(historical / 'metadata.json')['source']) / 'prepared.json')
    windows = read(run / 'validation-windows.json')
    config = decoder(parent['config'])
    drafts = run / 'annotation-drafts/gigaam-ctc-combination'
    drafts.mkdir(parents=True, exist_ok=True)
    backup = drafts / 'review-before-drafts.md'
    if not backup.exists():
        backup.write_text(args.review.read_text())
    args.review.write_text(prepare_review(args.review.read_text()))
    write(run / 'reference-assistance.json', {'mode': 'asr_assisted', 'model': parent['label'],
          'config': config, 'review': str(args.review), 'drafts': str(drafts),
          'requires_manual_verification': True, 'not_blind_annotation': True})
    outcomes = {}
    def progress():
        write(drafts / 'status.json', {'pid': os.getpid(), 'status': 'complete' if len(outcomes) == len(windows)
                                       and all(r['status'] == 'ok' for r in outcomes.values()) else 'running',
                                       'completed': len(outcomes), 'total': len(windows), 'fragments': outcomes})
    progress()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(infer_fragment, run, prepared, config, item, args.retry_failed): item for item in windows}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
                write(drafts / (item['id'] + '.json'), result)
                updated, inserted = insert_draft(args.review.read_text(), item['id'], result['hypothesis'])
                if inserted:
                    args.review.write_text(updated)
                outcomes[item['id']] = {'status': 'ok', 'inserted': inserted, 'draft': str(drafts / (item['id'] + '.json'))}
                print(item['id'], 'inserted' if inserted else 'existing text preserved', flush=True)
            except Exception as exc:
                outcomes[item['id']] = {'status': 'error', 'error': str(exc)}
                print(item['id'], 'FAILED', exc, flush=True)
            progress()
    if any(r['status'] != 'ok' for r in outcomes.values()):
        write(drafts / 'status.json', {'status': 'error', 'fragments': outcomes})
        raise SystemExit(1)


if __name__ == '__main__':
    main()
