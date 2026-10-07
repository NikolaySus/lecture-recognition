"""Rebuild a completed study from cached alignments, without loading ASR models."""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lecture_recognition.evaluation import combine, normalize, reference_cards, score_case  # noqa: E402
from lecture_recognition.gigaam_tuning import ranking  # noqa: E402
from lecture_recognition.model_benchmark import extra_metrics, read, validate_srt, write  # noqa: E402
from lecture_recognition.timeline import merge_words, to_srt  # noqa: E402


def rebuild(source, scores_only=False, review=None):
    source = source.resolve()
    hashes = {}

    def tracked(path):
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return read(path)

    metadata = tracked(source / 'metadata.json')
    reference = tracked(source / 'reference.json')
    old_reference = copy.deepcopy(reference)
    review_bytes = (source / 'review-snapshot.md').read_bytes()
    if review is not None:
        review_bytes = review.read_bytes()
        hashes[str(review.resolve())] = hashlib.sha256(review_bytes).hexdigest()
        cards = reference_cards(review_bytes.decode('utf-8'))
        for group in reference['groups']:
            tokens, spans = combine(cards, group['cards'])
            text = ' '.join(tokens)
            if normalize(text, 2) != normalize(group['text'], 2):
                raise ValueError('This reference update changes normalized content; review separately.')
            group['text'] = text
            group.pop('words', None)
            for name, card in group['cards'].items():
                card.update(text=cards[name], span=spans[name])
        reference['method'] = 'historical-fixed-windows-current-review'

    prepared = tracked(Path(metadata['source']) / 'prepared.json')
    code = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT / 'src/lecture_recognition/timeline.py',
             ROOT / 'src/lecture_recognition/evaluation.py', ROOT / 'src/lecture_recognition/model_benchmark.py']}
    key = hashlib.sha256(json.dumps({'code': code, 'reference': reference,
                                    'review': hashlib.sha256(review_bytes).hexdigest()}, sort_keys=True).encode()).hexdigest()[:16]
    destination = source / ('rescored' if scores_only else 'remerged') / key
    cases, changes = {}, []
    for path in sorted((source / 'cases').glob('*/case.json')):
        case = tracked(path)
        if case['status'] != 'ok':
            raise ValueError(f'Incomplete source: {path}')
        old_score = case['score']
        if scores_only:
            words = tracked(path.parent / 'alignment/result.json')['words']
            audit = tracked(path.parent / 'seam-audit.json')
        else:
            request = tracked(path.parent / 'alignment/request.json')
            tracked(path.parent / 'asr/result.json')
            aligned = []
            for transcript in request['transcripts']:
                cache_key = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
                aligned.append(tracked(path.parent / 'alignment-cache/alignment' / (cache_key + '.json')))
            audit = []
            words = [{k: v for k, v in w.items() if k != 'chunks'}
                     for w in merge_words(aligned, seam_audit=audit)]
        case['score'] = extra_metrics(score_case(words, reference, version=2))
        case['legacy_score'] = extra_metrics(score_case(words, reference, version=1))
        case['previous_score'] = old_score
        if not scores_only:
            case['postprocessing'] = {'version': 'overlap-phrase-v4', 'source_case': str(path), 'repairs': len(audit)}
        case['evaluation_source'] = str(path)
        srt = to_srt(words, prepared['regions'])
        case['validation'] = validate_srt(srt, prepared['duration'], prepared['excluded'])
        target = destination / 'cases' / case['id']
        write(target / 'case.json', case)
        write(target / 'alignment/result.json', {'status': 'ok', 'words': words, 'source': str(path.parent)})
        write(target / 'seam-audit.json', audit)
        (target / 'transcript.srt').write_text(srt, encoding='utf-8')
        cases[case['id']] = case
        changes.append({'id': case['id'], 'label': case['label'], 'repairs': audit,
                        'cards': {k: {'before': old_score['cards'][k], 'after': v}
                                  for k, v in case['score']['cards'].items() if v != old_score['cards'][k]}})
    selection = tracked(source / 'selection.json')
    for model, item in selection.items():
        base = cases[item['baseline']]
        candidate = min((c for c in cases.values() if c['config']['model'] == model
                         and c['stage'] not in ('control', 'validation')), key=ranking)
        item['candidate'] = candidate['id']
        item['checks'] = [c for c in item['checks'] if c['source'] in (base['id'], candidate['id'])]
        lookup = {(c['source'], c['tag']): cases[c['id']] for c in item['checks']}
        item['repeat_equal'] = {c['id']: lookup[c['id'], 'repeat']['score'] == c['score']
                                for c in (base, candidate) if (c['id'], 'repeat') in lookup}
        item['shift_comparisons'] = []
        for tag in ('shift-5', 'shift+5'):
            if (base['id'], tag) not in lookup or (candidate['id'], tag) not in lookup:
                item['shift_comparisons'].append({'tag': tag, 'status': 'not_measured'})
                continue
            a, b = lookup[base['id'], tag], lookup[candidate['id'], tag]
            item['shift_comparisons'].append({'tag': tag, 'baseline': a['id'], 'candidate': b['id'],
                'wer_delta': b['score']['total']['wer'] - a['score']['total']['wer'],
                'card_deltas': {k: v['wer'] - a['score']['cards'][k]['wer'] for k, v in b['score']['cards'].items()}})
    write(destination / 'metadata.json', {**metadata, 'evaluation_source': str(source),
          'card_method': 'group-alignment-v1', 'reference': reference,
          'review_sha256': hashlib.sha256(review_bytes).hexdigest(), 'postprocessing_source': metadata.get('postprocessing_source', str(source)),
          'postprocessing_code': code})
    write(destination / 'reference.json', reference)
    (destination / 'review-snapshot.md').write_bytes(review_bytes)
    write(destination / 'selection.json', selection)
    write(destination / 'execution.json', {'status': 'complete', 'cases': len(cases), 'inference_rerun': False})
    write(destination / 'remerge-audit.json', {'source_hashes': hashes, 'changes': changes,
        'previous_reference': old_reference, 'reference_normalization_unchanged': True})
    # Ensure the original inputs have not changed during rebuilding.
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    write(ROOT / '.lecture-cache/gigaam-tuning/latest-report.json', {'run': str(destination)})
    print(destination)
    print(json.dumps({m: {k: cases[v[k]]['score']['total'] for k in ('baseline', 'candidate')}
                      for m, v in selection.items()}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--scores-only', action='store_true', help='Reuse merged words without modifying transcripts')
    parser.add_argument('--review', type=Path, help='Update reference spelling, preserving normalized content')
    args = parser.parse_args()
    rebuild(args.source, args.scores_only, args.review)
