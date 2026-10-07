"""Audit cached chunk seams and remaining adjacent lexical repetitions without ASR."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from lecture_recognition.timeline import merge_words  # noqa: E402


def audit(source, output):
    remaining, repairs, hashes = [], [], {}
    count, seams = 0, 0

    def read(path):
        data = path.read_bytes()
        hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    for path in sorted((source / 'cases').glob('*/case.json')):
        case = read(path)
        if case['status'] != 'ok':
            raise ValueError(f'Incomplete case: {path}')
        aligned = []
        for transcript in read(path.parent / 'alignment/request.json')['transcripts']:
            key = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
            aligned.append(read(path.parent / 'alignment-cache/alignment' / (key + '.json')))
        events = []
        words = merge_words(aligned, seam_audit=events)
        seams += len(aligned) - 1
        count += 1
        repairs.extend({'case': case['id'], 'label': case['label'], **event} for event in events)
        tokens = [(m.group().casefold().replace('ё', 'е'), word)
                  for word in words for m in re.finditer(r'\w+', word['text'])]
        for start in range(len(tokens)):
            for size in range(1, 9):
                if start + 2 * size > len(tokens):
                    break
                left, right = tokens[start:start + size], tokens[start + size:start + 2 * size]
                if [t[0] for t in left] == [t[0] for t in right]:
                    remaining.append({'case': case['id'], 'label': case['label'],
                        'phrase': ' '.join(t[0] for t in left), 'start': left[0][1]['start'],
                        'cross_chunk': not bool(left[-1][1]['chunks'] & right[0][1]['chunks'])})
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {'source': str(source.resolve()), 'cases': count, 'seams_checked': seams,
        'timeline_sha256': hashlib.sha256((ROOT / 'src/lecture_recognition/timeline.py').read_bytes()).hexdigest(),
        'source_hashes': hashes,
        'detection': 'adjacent exact lexical repetitions, 1-8 tokens; source chunk provenance',
        'repairs': repairs, 'remaining_repetitions': remaining}
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'cases': count, 'seams': seams, 'repairs': len(repairs),
                     'remaining_cross_chunk': sum(r['cross_chunk'] for r in remaining)}, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/gigaam-seam-audit.json')
    args = parser.parse_args()
    audit(args.source, args.output)
