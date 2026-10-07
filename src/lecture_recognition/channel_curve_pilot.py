"""R005-only screening of channel curves with frozen neighboring context."""

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .dynamic_channels import sharpen_weights
from .dynamic_study import FIXED, PARENTS
from .evaluation import edit_score, project_cards, select_text
from .experiments import identity
from .model_benchmark import ROOT, read, worker, write
from .timeline import merge_words

GRIDS = {
    'energy-focus': {'thresholds': (51, 52, 53), 'steepness': (16, 32, 64),
                     'bases': (('energy', 'fast'),)},
    'lower': {'thresholds': (52, 54, 56, 58, 60), 'steepness': (2, 4, 8, 16)},
    'legacy': {'thresholds': tuple(range(60, 68)), 'steepness': (2, 4, 8)},
}
BASES = tuple((method, speed) for method in ('energy', 'ev-normalized') for speed in ('fast', 'slow'))
INDICES = (10, 11)


def r005_score(words, reference):
    group = next(g for g in reference['groups'] if 'R005' in g['cards'])
    text = select_text(words, *group['window'])
    alignment = edit_score(group['text'], text, 2, with_alignment=True)['alignment']
    only = {**group, 'cards': {'R005': group['cards']['R005']}}
    hypothesis = project_cards(only, text, alignment, 2)['R005']
    return {**edit_score(group['cards']['R005']['text'], hypothesis, 2),
            'reference': group['cards']['R005']['text'], 'hypothesis': hypothesis}


def choose_shortlist(cases, baseline_errors):
    successful = [c for c in cases if c['status'] == 'ok']
    ordered = sorted(successful, key=lambda c: (c['score']['errors'], c['score']['cer'],
                     -c['threshold_percent'], c['steepness'], c['label']))
    top = {c['id'] for c in ordered[:10]}
    return [{**c, 'reached_rnnt_left': c['score']['errors'] <= baseline_errors,
             'in_top10': c['id'] in top}
            for c in ordered if c['id'] in top or c['score']['errors'] <= baseline_errors]


def quality_ties(cases):
    groups = {}
    for case in cases:
        if case['status'] == 'ok':
            key = (case['score']['errors'], case['score']['cer'])
            groups.setdefault(key, []).append(case['id'])
    return [{'errors': errors, 'cer': cer, 'case_ids': sorted(ids)}
            for (errors, cer), ids in sorted(groups.items()) if len(ids) > 1]


class Pilot:
    def __init__(self, args):
        self.args = args
        self.grid = GRIDS[args.grid]
        self.bases = self.grid.get('bases', BASES)
        self.expected_cases = len(PARENTS) * len(self.bases) * len(self.grid['thresholds']) * len(self.grid['steepness'])
        self.source = args.source.resolve()
        source_metadata = read(self.source / 'metadata.json')
        historical = read(Path(source_metadata['source']) / 'metadata.json')
        self.prepared = read(Path(historical['source']) / 'prepared.json')
        self.reference = read(self.source / 'reference.json')
        if digest(args.audio) != source_metadata['source_audio_sha256']:
            raise ValueError('Source recording changed')
        if digest(args.review) != source_metadata['review_sha256']:
            raise ValueError('Review changed')
        if digest(Path(self.prepared['audio'])) != self.prepared['masked_sha256']:
            raise ValueError('Alignment audio changed')
        existing = [read(p) for p in (self.source / 'cases').glob('*/case.json')]
        self.fixed = {label: next(c for c in existing if c['label'] == label and c['status'] == 'ok')
                      for label in FIXED}
        self.controls = {}
        files = [self.source / 'metadata.json', self.source / 'reference.json', args.review,
                 Path(historical['source']) / 'prepared.json']
        for label, case in self.fixed.items():
            folder = self.source / 'cases' / case['id']
            files.extend(folder / n for n in ('case.json', 'alignment/result.json'))
            score = r005_score(read(folder / 'alignment/result.json')['words'], self.reference)
            if score != case['score']['cards']['R005']:
                raise ValueError('Cannot reproduce fixed R005: ' + label)
            self.controls[label] = {'id': case['id'], 'score': score}
        self.parent_aligned = {}
        self.chunks = {}
        for model, label in PARENTS.items():
            case = self.fixed[label]
            raw = Path(case['evaluation_source']).parent
            request = raw / 'alignment/request.json'
            files.append(request)
            aligned = []
            for tr in read(request)['transcripts']:
                key = hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest()
                path = raw / 'alignment-cache/alignment' / (key + '.json')
                files.append(path)
                aligned.append(read(path))
            score = r005_score(merge_words(aligned), self.reference)
            if score != self.controls[label]['score']:
                raise ValueError('Current merge changes fixed R005: ' + label)
            self.parent_aligned[model] = aligned
            self.chunks[model] = [read(request)['transcripts'][i]['chunk'] for i in INDICES]
        if (self.controls['gigaam-rnnt left']['score']['errors'],
                self.controls['gigaam-rnnt left']['score']['words']) != (2, 27):
            raise ValueError('Expected frozen RNNT left R005 control: 2/27')
        self.base_diagnostics = {}
        for method, speed in self.bases:
            path = self.source / 'audio' / f'dynamic_{method}_{speed}.json'
            files.append(path)
            self.base_diagnostics[method, speed] = read(path)
        code = [ROOT / 'src/lecture_recognition' / n for n in
                ('channel_curve_pilot.py', 'dynamic_channels.py', 'timeline.py', 'evaluation.py',
                 'gigaam_decoding.py', 'model_benchmark.py', 'models.py')]
        code += [ROOT / 'scripts/asr_worker.py', ROOT / 'uv.lock', ROOT / 'experiments/gigaam/uv.lock']
        metadata = {'series': 'r005-channel-curves-v1', 'source': str(self.source),
                    'source_audio_sha256': digest(args.audio), 'reference_sha256': digest(self.source / 'reference.json'),
                    'source_hashes': {str(p.resolve()): digest(p) for p in files},
                    'code': {str(p.relative_to(ROOT)): digest(p) for p in code},
                    'grid': args.grid, 'thresholds_percent': self.grid['thresholds'],
                    'steepness': self.grid['steepness'], 'bases': self.bases,
                    'expected_cases': self.expected_cases, 'chunk_indices': INDICES, 'card': 'R005',
                    'curve_position': 'after_smoothing_and_sample_interpolation'}
        self.root = args.output.resolve() / identity(metadata)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        write(self.root / 'metadata.json', metadata)
        write(self.root / 'reference.json', self.reference)
        write(self.root / 'controls.json', self.controls)
        self.stereo = None
        self.cases = []
        for model in PARENTS:
            for method, speed in self.bases:
                for threshold in self.grid['thresholds']:
                    for steepness in self.grid['steepness']:
                        settings = {'model': model, 'method': method, 'speed': speed,
                                    'threshold_percent': threshold, 'steepness': steepness}
                        key = identity(settings)[:16]
                        path = self.root / 'cases' / key / 'case.json'
                        case = read(path) if path.exists() else {
                            **settings, 'id': key, 'status': 'pending',
                            'label': f'{model} {method} {speed} t{threshold} k{steepness}'}
                        write(path, case)
                        self.cases.append(case)
        write(args.output.resolve() / 'latest.json', {'run': str(self.root)})
        print('PILOT', self.root, flush=True)

    def audio(self, case):
        stem = f"{case['method']}_{case['speed']}_t{case['threshold_percent']}_k{case['steepness']}"
        path = self.root / 'audio' / (stem + '.f32')
        info = path.with_suffix('.json')
        if path.exists() and info.exists():
            if digest(path) != read(info)['sha256']:
                raise ValueError('Pilot audio cache changed')
            return path
        started = time.monotonic()
        if self.stereo is None:
            end = max(c['end'] for chunks in self.chunks.values() for c in chunks)
            buffer = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(self.args.audio),
                                              '-t', str(end), '-ar', str(RATE), '-ac', '2', '-f', 'f32le', '-'])
            self.stereo = np.frombuffer(buffer, dtype='<f4').reshape(-1, 2).astype(np.float64)
            if len(self.stereo) != round(end * RATE):
                raise ValueError('Pilot stereo duration mismatch')
        d = self.base_diagnostics[case['method'], case['speed']]
        times = np.arange(len(self.stereo)) / RATE
        before = np.interp(times, d['times'], d['weights_left'])
        after = sharpen_weights(before, case['threshold_percent'] / 100, case['steepness'])
        if not np.allclose(after + sharpen_weights(1 - before, case['threshold_percent'] / 100,
                                                   case['steepness']), 1, atol=1e-12):
            raise ValueError('Curve symmetry failed')
        wave = after * self.stereo[:, 0] + (1 - after) * self.stereo[:, 1]
        for a, b in self.prepared['excluded']:
            wave[round(a * RATE):round(b * RATE)] = 0
        if not np.isfinite(wave).all() or np.max(np.abs(wave)) > np.max(np.abs(self.stereo)) + 1e-12:
            raise ValueError('Invalid mixture')
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix('.tmp')
        wave.astype('<f4').tofile(temporary)
        temporary.replace(path)
        card = next(g['cards']['R005'] for g in self.reference['groups'] if 'R005' in g['cards'])
        a, b = (round(t * RATE) for t in card['window'])
        def stats(v):
            return {'mean': float(v.mean()), 'min': float(v.min()), 'max': float(v.max()),
                    'above80_fraction': float(np.mean(v > .8)), 'above90_fraction': float(np.mean(v > .9))}
        step = round(.1 * RATE)
        write(info, {'sha256': digest(path), 'samples': len(wave), 'preparation_seconds': time.monotonic() - started,
                     'r005_before': stats(before[a:b]), 'r005_after': stats(after[a:b]),
                     'times': times[::step].tolist(), 'weights_before': before[::step].tolist(),
                     'weights_after': after[::step].tolist()})
        return path

    def execute(self, case):
        if case['status'] == 'ok' or (case['status'] == 'error' and not self.args.retry_failed):
            return
        folder = self.root / 'cases' / case['id']
        case.update(status='running')
        write(folder / 'case.json', case)
        started = time.monotonic()
        try:
            audio = self.audio(case)
            wave = np.memmap(audio, dtype='<f4', mode='r')
            cfg = {k: v for k, v in self.fixed[PARENTS[case['model']]]['config'].items()
                   if k not in ('audio_variant', 'maximum', 'context')}
            replacements = []
            cache_keys = []
            for chunk in self.chunks[case['model']]:
                pcm = wave[round(chunk['start'] * RATE):round(chunk['end'] * RATE)]
                key = identity({'pcm': hashlib.sha256(pcm.tobytes()).hexdigest(), 'config': cfg, 'chunk': chunk})
                shared = self.root / 'inference-cache' / key
                cache_keys.append(key)
                case['inference_cache_keys'] = list(cache_keys)
                write(folder / 'case.json', case)
                if (shared / 'asr/result.json').exists() and read(shared / 'asr/result.json')['status'] == 'ok':
                    asr = read(shared / 'asr/result.json')
                else:
                    request = {'operation': 'asr', 'audio': str(audio), 'config': cfg, 'chunks': [chunk]}
                    if (shared / 'asr/request.json').exists():
                        request = read(shared / 'asr/request.json')
                    asr = worker(request, shared / 'asr', 'gigaam', self.args.retry_failed)
                if asr['status'] != 'ok':
                    raise RuntimeError(asr.get('error', 'ASR failed'))
                transcript = {k: asr['transcripts'][0][k] for k in ('chunk', 'text')}
                aligned = worker({'operation': 'align', 'audio': self.prepared['audio'],
                                  'transcripts': [transcript], 'cache': str(shared / 'alignment-cache')},
                                 shared / 'alignment', retry=self.args.retry_failed)
                if aligned['status'] != 'ok':
                    raise RuntimeError(aligned.get('error', 'Alignment failed'))
                align_key = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
                replacements.append(read(shared / 'alignment-cache/alignment' / (align_key + '.json')))
            hybrid = copy.deepcopy(self.parent_aligned[case['model']])
            for index, replacement in zip(INDICES, replacements):
                hybrid[index] = replacement
            audit = []
            words = merge_words(hybrid, seam_audit=audit)
            score = r005_score(words, self.reference)
            if score['words'] != 27:
                raise ValueError('R005 denominator changed')
            write(folder / 'seam-audit.json', audit)
            write(folder / 'hybrid-words.json', [{k: v for k, v in w.items() if k != 'chunks'} for w in words])
            case.update(status='ok', score=score, audio=str(audio), audio_diagnostics=str(audio.with_suffix('.json')),
                        inference_cache_keys=cache_keys, context_parent=PARENTS[case['model']])
        except Exception as exc:
            case.update(status='error', error=str(exc))
        case['elapsed_seconds'] = time.monotonic() - started
        write(folder / 'case.json', case)
        self.select()
        print('CASE', case['label'], case['status'], case.get('score', {}).get('errors'), flush=True)

    def select(self, blocked=False):
        complete = len(self.cases) == self.expected_cases and all(c['status'] == 'ok' for c in self.cases)
        baseline = self.controls['gigaam-rnnt left']['score']
        selected = choose_shortlist(self.cases, baseline['errors'])
        write(self.root / 'results.json', {'complete': complete, 'card': 'R005', 'controls': self.controls,
                                         'cases': self.cases})
        write(self.root / 'shortlist.json', {'complete': complete, 'provisional': not complete,
                                            'rnnt_left': baseline, 'count': len(selected), 'cases': selected,
                                            'quality_ties': quality_ties(self.cases)})
        write(self.root / 'execution.json', {'status': 'blocked' if blocked else 'complete' if complete else 'incomplete',
                                            'reason': 'cuda_unavailable' if blocked else None,
                                            'completed': sum(c['status'] == 'ok' for c in self.cases), 'expected': self.expected_cases})
        lines = [f"R005 pilot: {'complete' if complete else 'provisional'}", f'Selected: {len(selected)}',
                 f"RNNT left: {baseline['errors']}/27, WER {baseline['wer']:.2%}",
                 'Equal errors and CER mean equal measured quality; remaining ordering is a tie-break.']
        for c in selected:
            lines.append(f"{c['label']}: {c['score']['errors']}/27, WER {c['score']['wer']:.2%}; "
                         f"reached RNNT left: {c['reached_rnnt_left']}")
        (self.root / 'summary.txt').write_text('\n'.join(lines) + '\n')


def progress(root):
    done = sum(1 for p in (root / 'inference-cache').glob('*/asr/result.json') if read(p)['status'] == 'ok')
    metadata = read(root / 'metadata.json')
    planned_chunks = metadata['expected_cases'] * len(metadata['chunk_indices'])
    total_steps = planned_chunks * 2
    logical = 0
    for path in (root / 'cases').glob('*/case.json'):
        for key in read(path).get('inference_cache_keys', []):
            for stage in ('asr', 'alignment'):
                result = root / 'inference-cache' / key / stage / 'result.json'
                logical += int(result.exists() and read(result)['status'] == 'ok')
    return {'execution': read(root / 'execution.json'), 'unique_asr_chunks': done,
                      'completed_logical_steps': logical, 'total_logical_steps': total_steps,
                      'percent': round(100 * logical / total_steps, 1), 'planned_case_chunks': planned_chunks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'progress', 'select'])
    parser.add_argument('--grid', choices=tuple(GRIDS), default='lower')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '.lecture-cache/gigaam-channel-curves-r005')
    parser.add_argument('--audio', type=Path, default=ROOT / 'record/20260925_101716.m4a')
    parser.add_argument('--review', type=Path, default=ROOT / 'record/20260925_101716.review.md')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.stage == 'progress':
        root = Path(read(args.output / 'latest.json')['run'])
        print(json.dumps(progress(root)))
        return
    args.source = args.source or Path(read(ROOT / '.lecture-cache/gigaam-dynamic-channels/latest.json')['run'])
    pilot = Pilot(args)
    pilot.select()
    if args.stage == 'select':
        print((pilot.root / 'summary.txt').read_text())
        return
    if args.stage == 'run':
        probe = subprocess.run([str(ROOT / 'experiments/gigaam/.venv/bin/python'), '-c',
                                'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'])
        if probe.returncode:
            pilot.select(blocked=True)
            raise RuntimeError('CUDA unavailable; no pilot ASR started')
        for case in pilot.cases:
            pilot.execute(case)
    else:
        for case in pilot.cases:
            pilot.audio(case)
    pilot.select()


if __name__ == '__main__':
    main()
