"""Independent, resumable GigaAM dynamic-channel series."""

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .dynamic_channels import METHODS, SETTINGS, SPEEDS, features, mix
from .evaluation import score_case
from .experiments import identity
from .gigaam_tuning import Study
from .model_benchmark import ROOT, extra_metrics, read, write
from .timeline import merge_words

FIXED = ('gigaam-ctc original', 'gigaam-rnnt original', 'gigaam-rnnt left',
         'gigaam-ctc combination', 'gigaam-ctc bias4')
PARENTS = {'gigaam-ctc': 'gigaam-ctc combination', 'gigaam-rnnt': 'gigaam-rnnt left'}


class DynamicStudy(Study):
    def __init__(self, args):
        self.args = args
        self.source = args.source.resolve()
        historical = read(self.source / 'metadata.json')
        self.prepared = read(Path(historical['source']) / 'prepared.json')
        self.reference = read(self.source / 'reference.json')
        self.duration = self.prepared['prefix_end']
        self.raw = np.memmap(self.prepared['audio'], mode='r', dtype='<f4')[:round(self.duration * RATE)]
        if digest(args.review) != historical['review_sha256']:
            raise ValueError('Review changed; rebuild historical comparisons first')
        if digest(args.audio) != historical['source_audio_sha256']:
            raise ValueError('Source stereo differs from historical recording')
        if digest(Path(self.prepared['audio'])) != self.prepared['masked_sha256']:
            raise ValueError('Historical alignment audio changed')
        for name in ('timeline.py', 'evaluation.py', 'model_benchmark.py'):
            relative = 'src/lecture_recognition/' + name
            if digest(ROOT / relative) != historical['postprocessing_code'][relative]:
                raise ValueError('Historical postprocessing differs; rebuild comparisons first')
        old = [read(p) for p in (self.source / 'cases').glob('*/case.json')]
        self.fixed = {label: next(c for c in old if c['label'] == label) for label in FIXED}
        self.originals = {model: self.fixed[model + ' original'] for model in PARENTS}
        source_files = [self.source / 'reference.json', self.source / 'metadata.json', args.review,
                        Path(historical['source']) / 'prepared.json']
        for case in self.fixed.values():
            directory = self.source / 'cases' / case['id']
            words = read(directory / 'alignment/result.json')['words']
            if case['status'] != 'ok' or extra_metrics(score_case(words, self.reference, 2)) != case['score']:
                raise ValueError('Imported comparison score cannot be reproduced')
            source_files.extend(directory / name for name in
                                ('case.json', 'alignment/result.json', 'seam-audit.json', 'transcript.srt'))
        code = [ROOT / 'src/lecture_recognition' / name for name in
                ('dynamic_channels.py', 'dynamic_study.py', 'gigaam_tuning.py', 'gigaam_decoding.py',
                 'evaluation.py', 'timeline.py', 'model_benchmark.py')]
        code += [ROOT / 'scripts/asr_worker.py', ROOT / 'uv.lock', ROOT / 'experiments/gigaam/uv.lock']
        metadata = {'series': 'dynamic-channels-v1', 'source': str(self.source),
                    'review_sha256': digest(args.review), 'source_audio_sha256': digest(args.audio),
                    'code': {str(p.relative_to(ROOT)): digest(p) for p in code},
                    'source_hashes': {str(p.resolve()): digest(p) for p in source_files},
                    'report_selection': 'all', 'expected_cases': 17,
                    'methods': METHODS, 'speeds': SPEEDS, 'settings': SETTINGS,
                    'parent_ids': {model: self.fixed[label]['id'] for model, label in PARENTS.items()}}
        self.root = args.output.resolve() / identity(metadata)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        write(self.root / 'metadata.json', metadata)
        write(self.root / 'reference.json', self.reference)
        shutil.copyfile(args.review, self.root / 'review-snapshot.md')
        for case in self.fixed.values():
            target = self.root / 'cases' / case['id']
            if not target.exists():
                for name in ('case.json', 'alignment/result.json', 'seam-audit.json', 'transcript.srt'):
                    destination = target / name
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(self.source / 'cases' / case['id'] / name, destination)
                imported = {**case, 'imported_from': str(self.source / 'cases' / case['id'])}
                write(target / 'case.json', imported)
        self.cases = {p.parent.name: read(p) for p in (self.root / 'cases').glob('*/case.json')}
        self.audio_hashes = {}
        self.stereo = self.analysis = None
        write(args.output.resolve() / 'latest.json', {'run': str(self.root)})
        print('SERIES', self.root, flush=True)

    def audio(self, variant):
        path = self.root / 'audio' / (variant + '.f32')
        metadata_path = path.with_suffix('.json')
        if path.exists() and metadata_path.exists():
            metadata = read(metadata_path)
            if path.stat().st_size == len(self.raw) * 4 and digest(path) == metadata['sha256']:
                return path
            raise ValueError('Cached dynamic audio changed')
        if self.stereo is None:
            buffer = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(self.args.audio),
                                              '-t', str(self.duration), '-ar', str(RATE), '-ac', '2',
                                              '-f', 'f32le', '-'])
            self.stereo = np.frombuffer(buffer, dtype='<f4').reshape(-1, 2)
            if len(self.stereo) != len(self.raw):
                raise ValueError('Stereo and historical sample lengths differ')
            self.analysis = features(self.stereo, self.prepared['excluded'])
        _, method, speed = variant.split('_')
        started = time.monotonic()
        wave, diagnostic = mix(self.stereo, method, speed, self.prepared['excluded'], self.analysis)
        mirrored, mirror_diag = mix(self.stereo[:, ::-1], method, speed, self.prepared['excluded'],
                                    {**self.analysis, 'energy': self.analysis['energy'][:, ::-1],
                                     'envelopes': self.analysis['envelopes'][:, ::-1]})
        error = float(np.max(np.abs(wave - mirrored)))
        weight_error = float(np.max(np.abs(np.array(diagnostic['weights_left']) +
                                           np.array(mirror_diag['weights_left']) - 1)))
        if error > 1e-7 or weight_error > 1e-12:
            raise ValueError('Channel swap symmetry failed')
        path.parent.mkdir(exist_ok=True)
        temporary = path.with_suffix('.tmp')
        wave.tofile(temporary)
        temporary.replace(path)
        write(metadata_path, {**diagnostic, 'sha256': digest(path), 'samples': len(wave),
                              'peak': float(np.max(np.abs(wave))), 'preparation_seconds': time.monotonic() - started,
                              'mirror_max_sample_error': error, 'mirror_max_weight_error': weight_error})
        return path

    def summary(self):
        complete = len(self.cases) == 17 and all(c['status'] == 'ok' for c in self.cases.values())
        write(self.root / 'execution.json', {'status': 'complete' if complete else 'incomplete',
                                           'expected_cases': 17, 'cases': len(self.cases),
                                           'completed': sum(c['status'] == 'ok' for c in self.cases.values())})
        write(self.root / 'summary.json', {'selection': 'all', 'cases': [
            {'id': c['id'], 'label': c['label'], 'status': c['status'], 'included': c['status'] == 'ok'}
            for c in self.cases.values()]})

    def prepare_cases(self):
        for model, label in PARENTS.items():
            parent = self.fixed[label]
            for method in METHODS:
                for speed in SPEEDS:
                    cfg = {**parent['config'], 'audio_variant': f'dynamic_{method}_{speed}'}
                    audio = self.audio(cfg['audio_variant'])
                    settings = {'config': cfg, 'chunks': self.originals[model]['chunks'], 'stage': 'dynamic',
                                'label': f'{model} dynamic {method} {speed}', 'audio': str(audio), 'repeat': 0}
                    key = identity(settings)[:16]
                    if key not in self.cases:
                        case = {'id': key, **settings, 'status': 'pending'}
                        write(self.root / 'cases' / key / 'case.json', case)
                        self.cases[key] = case
        self.summary()

    def run_dynamic(self):
        probe = subprocess.run([str(ROOT / 'experiments/gigaam/.venv/bin/python'), '-c',
                                'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],
                               capture_output=True, text=True)
        if probe.returncode:
            write(self.root / 'execution.json', {'status': 'blocked', 'reason': 'cuda_unavailable',
                                                'expected_cases': 17, 'cases': len(self.cases),
                                                'completed': 5, 'probe_stderr': probe.stderr})
            raise RuntimeError('CUDA unavailable in GigaAM environment: ' + probe.stderr)
        for model, label in PARENTS.items():
            parent = self.fixed[label]
            for method in METHODS:
                for speed in SPEEDS:
                    options = {k: v for k, v in parent['config'].items() if k != 'model'}
                    options['audio_variant'] = f'dynamic_{method}_{speed}'
                    case = self.case(model, f'{model} dynamic {method} {speed}', 'dynamic', **options)
                    if case['status'] == 'ok':
                        directory = self.root / 'cases' / case['id']
                        if not (directory / 'seam-audit.json').exists():
                            aligned = []
                            for transcript in read(directory / 'alignment/request.json')['transcripts']:
                                key = hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()
                                aligned.append(read(directory / 'alignment-cache/alignment' / (key + '.json')))
                            audit = []
                            words = [{k: v for k, v in w.items() if k != 'chunks'}
                                     for w in merge_words(aligned, seam_audit=audit)]
                            if words != read(directory / 'alignment/result.json')['words']:
                                raise ValueError('Worker and audited merge disagree')
                            write(directory / 'seam-audit.json', audit)
        self.summary()


def progress(root):
    metadata = read(root / 'metadata.json')
    total = done = 0
    for model, parent_id in metadata['parent_ids'].items():
        chunks = len(read(root / 'cases' / parent_id / 'case.json')['chunks'])
        total += 6 * chunks * 2
    for directory in (root / 'cases').iterdir():
        case = read(directory / 'case.json')
        if case['stage'] != 'dynamic':
            continue
        done += min(len(case['chunks']), len(list((directory / 'asr/raw-chunks').glob('*.json'))))
        done += min(len(case['chunks']), len(list((directory / 'alignment-cache/alignment').glob('*.json'))))
    return {'completed_chunks': done, 'total_chunks': total, 'percent': round(100 * done / total, 1),
            'execution': read(root / 'execution.json')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run', 'report', 'progress'])
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '.lecture-cache/gigaam-dynamic-channels')
    parser.add_argument('--review', type=Path, default=ROOT / 'record/20260925_101716.review.md')
    parser.add_argument('--audio', type=Path, default=ROOT / 'record/20260925_101716.m4a')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.stage in ('report', 'progress'):
        root = Path(read(args.output / 'latest.json')['run'])
        if args.stage == 'progress':
            print(json.dumps(progress(root)))
        else:
            subprocess.run(['python3', str(ROOT / 'scripts/create_gigaam_report.py'), '--run', str(root),
                            '--output', str(ROOT / 'output/pdf/r001-r008-gigaam-dynamic-channels.pdf')], check=True)
        return
    args.source = args.source or Path(read(ROOT / '.lecture-cache/gigaam-tuning/latest-report.json')['run'])
    study = DynamicStudy(args)
    for method in METHODS:
        for speed in SPEEDS:
            study.audio(f'dynamic_{method}_{speed}')
    study.prepare_cases()
    if args.stage == 'run':
        study.run_dynamic()


if __name__ == '__main__':
    main()
