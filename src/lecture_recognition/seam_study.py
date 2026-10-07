"""Re-score the completed segmentation cache without GPU or new ASR."""
import argparse
import hashlib
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .channel_research import decoder
from .channel_utility import make_layout, shift_layout
from .evaluation import score_case
from .experiments import identity
from .model_benchmark import ROOT, extra_metrics, read, write
from .seam_merge import merge_overlap_words
from .timeline import merge_words, to_srt


def find_inference(source, config, chunks):
    wave = np.memmap(source / 'audio/left.f32', dtype='<f4', mode='r')
    pcm = [hashlib.sha256(wave[round(c['start'] * RATE):round(c['end'] * RATE)].tobytes()).hexdigest() for c in chunks]
    key = identity({'model': config, 'pcm': pcm, 'chunks': chunks})[:16]
    candidates = [source] + sorted(p.parent for p in source.parent.glob('*/metadata.json') if p.parent != source)
    source_meta = read(source / 'metadata.json')
    for root in candidates:
        summary = root / 'inference' / key / 'result.json'
        if not summary.exists():
            continue
        metadata = read(root / 'metadata.json')
        if (metadata['audio_sha256'] != source_meta['audio_sha256']
                or metadata['prepared_sha256'] != source_meta['prepared_sha256']):
            continue
        result = read(summary)
        if result['status'] == 'ok' and [t['chunk'] for t in result['transcripts']] == chunks:
            return result, summary
    # The historical control was imported verbatim rather than saved as ASR.
    old = Path(source_meta['source'])
    label = 'gigaam-ctc combination' if config['model'] == 'gigaam-ctc' else 'gigaam-rnnt left'
    case = next(read(p) for p in (old / 'cases').glob('*/case.json') if read(p)['label'] == label)
    raw = Path(case['evaluation_source']).parent
    request = read(raw / 'alignment/request.json')
    if [t['chunk'] for t in request['transcripts']] == chunks:
        from .channel_curve_full import alignment_key
        aligned = [read(raw / 'alignment-cache/alignment' / (alignment_key(t) + '.json')) for t in request['transcripts']]
        return {'aligned': aligned}, raw / 'alignment/request.json'
    raise ValueError('Missing raw cache; no ASR will be started: ' + key)


def study(source, output):
    source = source.resolve()
    original = read(source / 'chunk-study.json')
    assert original['complete'] and len(original['cases']) == 26
    frozen = read(source / 'chunk-frozen.json')['models']
    metadata = read(source / 'metadata.json')
    old = Path(metadata['source'])
    dynamic = Path(read(old / 'metadata.json')['source'])
    historical = Path(read(dynamic / 'metadata.json')['source'])
    prepared = read(Path(read(historical / 'metadata.json')['source']) / 'prepared.json')
    configs = {}
    for p in (old / 'cases').glob('*/case.json'):
        c = read(p)
        if c['label'] in ('gigaam-ctc combination', 'gigaam-rnnt left'):
            configs[c['config']['model']] = decoder(c['config'])
    version = {'source': str(source), 'chunk_study_sha256': digest(source / 'chunk-study.json'),
               'frozen_sha256': digest(source / 'chunk-frozen.json'), 'reference_sha256': digest(source / 'reference.json'),
               'implementation': {name: digest(ROOT / name) for name in (
                   'src/lecture_recognition/seam_merge.py', 'src/lecture_recognition/timeline.py',
                   'src/lecture_recognition/evaluation.py', 'src/lecture_recognition/channel_utility.py',
                   'src/lecture_recognition/channel_research.py', 'src/lecture_recognition/seam_study.py')}, 'normalization': 2}
    root = output.resolve() / identity(version)[:16]
    write(root / 'metadata.json', version)
    write(output / 'latest.json', {'run': str(root)})
    reference = read(source / 'reference.json')
    records = list(original['cases'])
    wave = np.memmap(source / 'audio/left.f32', dtype='<f4', mode='r')
    for model, candidate in frozen.items():
        for shift in candidate['shifts']:
            mode, maximum, context = candidate['layout'].split('-') if candidate['layout'] != 'historical' else ('vad', 'm20', 'c1')
            chunks = (shift_layout(candidate['chunks'], prepared['prefix_end'], 20, 1, shift['shift']) if candidate['layout'] == 'historical'
                      else make_layout(wave, prepared['prefix_end'], int(maximum[1:]), int(context[1:]), mode, prepared['regions'], shift['shift']))
            records.append({'model': model, 'layout': candidate['layout'] + f" shift{shift['shift']:+d}",
                            'score': shift['score'], 'chunks': chunks, 'shift': shift['shift']})
    results = []
    for case in records:
        inference, path = find_inference(source, configs[case['model']], case['chunks'])
        legacy = extra_metrics(score_case(merge_words(inference['aligned']), reference, 2))
        assert legacy == case['score'], (case['model'], case['layout'], 'legacy score changed')
        audit = []
        words = merge_overlap_words(inference['aligned'], seam_audit=audit)
        score = extra_metrics(score_case(words, reference, 2))
        to_srt(words, prepared['regions'])
        item = {'model': case['model'], 'layout': case['layout'], 'baseline': legacy, 'score': score,
                'source': str(path), 'audit': audit, 'words': [{k: v for k, v in w.items() if k != 'chunks'} for w in words],
                'delta_errors': score['total']['errors'] - legacy['total']['errors'],
                'regressed_groups': [n for n,g in legacy['groups'].items() if score['groups'][n]['errors'] > g['errors']],
                'regressed_cards': [n for n,c in legacy['cards'].items() if score['cards'][n]['errors'] > c['errors']],
                'critical_regression': any(score['total'][k] > legacy['total'][k] for k in ('number_errors', 'negation_errors'))
                    or any(not set(g['protected_correct']).issubset(score['groups'][n]['protected_correct'])
                           for n, g in legacy['groups'].items())}
        results.append(item)
        print(item['model'],item['layout'],legacy['total']['errors'],'→',score['total']['errors'],
              'regressions',item['regressed_cards'],flush=True)
        write(root / 'results.json', {'complete': False, 'cases': results})
    passed = all(not c['regressed_groups'] and not c['regressed_cards'] and not c['critical_regression'] for c in results)
    write(root / 'results.json', {'complete': True, 'cases': results, 'non_regression_passed': passed,
          'new_asr_runs': 0, 'test_used': False, 'scope': '26 layouts + 4 phase shifts; same raw ASR/alignment'})
    write(root / 'reference.json', reference)
    print('RESULT', root, 'NON_REGRESSION', passed)
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '.lecture-cache/seam-merge')
    args = parser.parse_args()
    source = args.source or Path(read(ROOT / '.lecture-cache/channel-research/latest.json')['run'])
    if args.source is None and (args.output / 'latest.json').exists():
        previous = Path(read(args.output / 'latest.json')['run'])
        source = Path(read(previous / 'metadata.json')['source'])
    study(source, args.output)
