"""Validate fixed historical/new left-channel layouts on S001–S012."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from lecture_recognition.audio import RATE, digest
from lecture_recognition.channel_research import REVIEW, CUDAUnavailable, Research, decoder
from lecture_recognition.channel_utility import aggregate_scores, assemble, make_layout
from lecture_recognition.evaluation import edit_score, select_text
from lecture_recognition.experiments import identity
from lecture_recognition.model_benchmark import ROOT, read, write
from lecture_recognition.model_benchmark import layout as historical_layout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    source = args.research.resolve()
    meta = read(source / 'metadata.json')
    research = Research(SimpleNamespace(
        source=Path(meta['source']), output=source.parent,
        audio=ROOT / 'record/20260925_101716.m4a',
        merger=meta.get('merger_override'), retry_failed=args.retry_failed))
    if research.root != source:
        raise ValueError('Research code changed; finish the development study before validation')
    chosen = read(source / 'chunk-frozen.json')
    for model in ('gigaam-ctc', 'gigaam-rnnt'):
        if (chosen['models'][model]['layout'], chosen['models'][model]['merger']) != ('regular-m20-c2', 'contextual'):
            raise ValueError('This validation protocol requires regular-m20-c2/contextual')
    windows = read(source / 'validation-windows.json')
    if len(windows) != 12:
        raise ValueError('Expected 12 validation windows')
    mono = np.memmap(research.prepared['audio'], dtype='<f4', mode='r')
    left = np.memmap(research.audio('left'), dtype='<f4', mode='r')
    layouts = {}
    for item in windows:
        start, end = item['input_window']
        regions = [(a - start, b - start) for a, b in research.prepared['regions']]
        span = slice(round(start * RATE), round(end * RATE))
        local = {
            'historical': historical_layout(mono[span], end - start, 20, regions),
            'contextual': make_layout(left[span], end - start, 20, 2, 'regular', regions),
        }
        layouts[item['id']] = {name: [{k: v + start for k, v in c.items()} for c in chunks]
                               for name, chunks in local.items()}
    protocol = {
        'study': 'fixed-left-layout-validation-v1', 'research': str(source),
        'models': {m: decoder(research.parents[m]['config']) for m in chosen['models']},
        'mergers': {'historical': 'current', 'contextual': 'contextual'},
        'windows': windows, 'layouts': layouts, 'development_sha256': digest(source / 'chunk-frozen.json'),
        'reference_sha256': digest(REVIEW), 'code': meta['code'],
        'runner_sha256': digest(Path(__file__)), 'left_sha256': digest(research.audio('left')),
        'alignment_audio_sha256': digest(Path(research.prepared['audio'])),
        'reference_origin': 'ASR-assisted; manually edited and confirmed', 'test_tuning': False,
    }
    folder = ROOT / '.lecture-cache/chunk-validation' / identity(protocol)[:16]
    folder.mkdir(parents=True, exist_ok=True)
    write(folder / 'protocol.json', protocol)
    write(folder.parent / 'latest.json', {'run': str(folder)})
    print('VALIDATION', folder, flush=True)
    # Lock all configurations and layouts before loading reference text.
    if args.prepare_only:
        print('Protocol ready: 48 model/layout/fragment evaluations', flush=True)
        return 0
    references = research.validation_reference()
    outcomes = []
    try:
        for model in protocol['models']:
            for variant, merger in protocol['mergers'].items():
                for item in windows:
                    chunks = layouts[item['id']][variant]
                    print(f"INFER {model} {variant} {item['id']} ({len(outcomes)}/48)", flush=True)
                    inferred = research.infer(model, 'left', chunks, split='test')
                    if inferred['status'] != 'ok':
                        raise RuntimeError(inferred.get('error', 'Inference failed'))
                    audit = []
                    words = assemble(inferred['aligned'], merger, audit)
                    text = select_text(words, *item['window'])
                    result = {'model': model, 'variant': variant, **item,
                              'reference': references[item['id']], 'hypothesis': text,
                              'score': edit_score(references[item['id']], text, 2),
                              'chunks': chunks, 'words': words, 'seam_audit': audit,
                              'aligned': inferred['aligned'], 'transcripts': inferred['transcripts']}
                    write(folder / 'cases' / model / variant / (item['id'] + '.json'), result)
                    outcomes.append(result)
                    write(folder / 'status.json', {'status': 'running', 'completed': len(outcomes), 'total': 48})
        metrics = {model: {variant: aggregate_scores([r['score'] for r in outcomes
                   if r['model'] == model and r['variant'] == variant])
                   for variant in protocol['mergers']} for model in protocol['models']}
        write(folder / 'results.json', {'complete': True, 'metrics': metrics, 'cases': outcomes,
                                      'manual_seam_review_complete': False, 'test_tuning': False})
        write(folder / 'status.json', {'status': 'complete', 'completed': 48, 'total': 48})
        print('COMPLETE', folder / 'results.json', flush=True)
        return 0
    except CUDAUnavailable as exc:
        write(folder / 'status.json', {'status': 'blocked', 'reason': str(exc),
                                      'completed': len(outcomes), 'total': 48})
        print(str(exc), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
