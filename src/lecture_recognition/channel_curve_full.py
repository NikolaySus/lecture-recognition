"""Full R001-R008 validation of Energy fast t52/k32, with five rescored controls."""

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .dynamic_channels import sharpen_weights
from .dynamic_study import FIXED, PARENTS
from .evaluation import score_case
from .experiments import identity
from .model_benchmark import ROOT, extra_metrics, read, validate_srt, worker, write
from .timeline import merge_words, to_srt


def chunk_key(pcm, config, chunk):
    return identity({'pcm': hashlib.sha256(pcm.tobytes()).hexdigest(), 'config': config, 'chunk': chunk})


def alignment_key(transcript):
    return hashlib.sha256(json.dumps(transcript, sort_keys=True).encode()).hexdigest()


class FullStudy:
    def __init__(self, args):
        self.args = args
        self.pilot = args.pilot.resolve()
        pilot_metadata = read(self.pilot / 'metadata.json')
        if not read(self.pilot / 'results.json')['complete']:
            raise ValueError('A completed pilot is required')
        self.source = Path(pilot_metadata['source'])
        source_metadata = read(self.source / 'metadata.json')
        historical = read(Path(source_metadata['source']) / 'metadata.json')
        prepared_path = Path(historical['source']) / 'prepared.json'
        self.prepared = read(prepared_path)
        self.reference = read(self.pilot / 'reference.json')
        if digest(args.audio) != pilot_metadata['source_audio_sha256']:
            raise ValueError('Recording changed')
        if digest(args.review) != source_metadata['review_sha256']:
            raise ValueError('Review changed')
        if digest(Path(self.prepared['audio'])) != self.prepared['masked_sha256']:
            raise ValueError('Alignment audio changed')
        for name in ('dynamic_channels.py', 'timeline.py', 'evaluation.py', 'gigaam_decoding.py',
                     'model_benchmark.py', 'models.py'):
            relative = 'src/lecture_recognition/' + name
            if digest(ROOT / relative) != pilot_metadata['code'][relative]:
                raise ValueError('Pilot code differs: ' + relative)
        for relative in ('scripts/asr_worker.py', 'uv.lock', 'experiments/gigaam/uv.lock'):
            if digest(ROOT / relative) != pilot_metadata['code'][relative]:
                raise ValueError('Pilot worker environment differs: ' + relative)
        files = [self.pilot / 'metadata.json', self.pilot / 'reference.json',
                 self.pilot / 'results.json', prepared_path, args.review]
        controls = read(self.pilot / 'controls.json')
        self.fixed, self.aligned = {}, {}
        for label in FIXED:
            case_path = self.source / 'cases' / controls[label]['id'] / 'case.json'
            case = read(case_path)
            raw = Path(case['evaluation_source']).parent
            request = raw / 'alignment/request.json'
            files.extend((case_path, request, raw / 'asr/result.json'))
            aligned = []
            for transcript in read(request)['transcripts']:
                path = raw / 'alignment-cache/alignment' / (alignment_key(transcript) + '.json')
                files.append(path)
                aligned.append(read(path))
            self.fixed[label], self.aligned[label] = case, aligned
        candidates = read(self.pilot / 'results.json')['cases']
        self.pilot_cases = {model: next(c for c in candidates if c['model'] == model
                            and c['method'] == 'energy' and c['speed'] == 'fast'
                            and c['threshold_percent'] == 52 and c['steepness'] == 32
                            and c['status'] == 'ok') for model in PARENTS}
        for case in self.pilot_cases.values():
            audio = Path(case['audio'])
            if digest(audio) != read(audio.with_suffix('.json'))['sha256']:
                raise ValueError('Pilot audio changed')
            files.extend((audio, audio.with_suffix('.json')))
            for key in case['inference_cache_keys']:
                files.extend((self.pilot / 'inference-cache' / key).rglob('*.json'))
        self.diagnostic_path = self.source / 'audio/dynamic_energy_fast.json'
        files.append(self.diagnostic_path)
        code_paths = [ROOT / 'src/lecture_recognition' / name for name in
                      ('channel_curve_full.py', 'dynamic_channels.py', 'timeline.py', 'evaluation.py',
                       'gigaam_decoding.py', 'model_benchmark.py', 'models.py')]
        code_paths += [ROOT / 'scripts/asr_worker.py', ROOT / 'uv.lock', ROOT / 'experiments/gigaam/uv.lock']
        metadata = {'series': 'channel-curve-full-v1', 'source': str(self.source),
                    'pilot': str(self.pilot), 'review_sha256': digest(args.review),
                    'source_audio_sha256': digest(args.audio), 'report_selection': 'all',
                    'expected_cases': 7, 'new_cases': 2, 'method': 'energy', 'speed': 'fast',
                    'threshold_percent': 52, 'steepness': 32,
                    'curve_position': 'after_smoothing_and_sample_interpolation',
                    'source_hashes': {str(p.resolve()): digest(p) for p in files},
                    'code': {str(p.relative_to(ROOT)): digest(p) for p in code_paths},
                    'parent_ids': {model: self.fixed[label]['id'] for model, label in PARENTS.items()}}
        self.root = args.output.resolve() / identity(metadata)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        write(self.root / 'metadata.json', metadata)
        write(self.root / 'reference.json', self.reference)
        shutil.copyfile(args.review, self.root / 'review-snapshot.md')
        self.cases = {}
        changes = []
        for label, case in self.fixed.items():
            directory = self.root / 'cases' / case['id']
            result = self.finish(case, self.aligned[label], directory)
            result.update(imported_from=str(self.source / 'cases' / case['id']), previous_score=case['score'])
            write(directory / 'case.json', result)
            self.cases[result['id']] = result
            changes.append({'label': label, 'before': case['score'], 'after': result['score']})
        write(self.root / 'control-rescore-audit.json', changes)
        for model, label in PARENTS.items():
            chunks = [a['chunk'] for a in self.aligned[label]]
            if len(chunks) != 22:
                raise ValueError('Expected 22 historical chunks')
            config = {**self.fixed[label]['config'], 'audio_variant': 'dynamic_energy_fast_t52_k32'}
            settings = {'label': model + ' Energy fast t52/k32', 'config': config,
                        'stage': 'dynamic', 'chunks': chunks, 'context_parent': label}
            key = identity(settings)[:16]
            path = self.root / 'cases' / key / 'case.json'
            self.cases[key] = read(path) if path.exists() else {'id': key, **settings, 'status': 'pending'}
            write(path, self.cases[key])
        write(args.output.resolve() / 'latest.json', {'run': str(self.root)})
        self.summary()
        print('SERIES', self.root, flush=True)

    def finish(self, case, aligned, directory):
        audit = []
        words = [{k: v for k, v in w.items() if k != 'chunks'} for w in merge_words(aligned, seam_audit=audit)]
        score = extra_metrics(score_case(words, self.reference, 2))
        if score['total']['words'] != 246:
            raise ValueError('Expected 246 reference words')
        srt = to_srt(words, self.prepared['regions'])
        write(directory / 'alignment/result.json', {'status': 'ok', 'words': words})
        write(directory / 'seam-audit.json', audit)
        (directory / 'transcript.srt').write_text(srt)
        return {**case, 'status': 'ok', 'score': score,
                'legacy_score': extra_metrics(score_case(words, self.reference, 1)),
                'validation': validate_srt(srt, self.prepared['duration'], self.prepared['excluded'])}

    def prepare(self):
        path = self.root / 'audio/dynamic_energy_fast_t52_k32.f32'
        if path.exists():
            if digest(path) != read(path.with_suffix('.json'))['sha256']:
                raise ValueError('Full mixture changed')
        else:
            data = subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(self.args.audio),
                    '-t', str(self.prepared['prefix_end']), '-ar', str(RATE), '-ac', '2', '-f', 'f32le', '-'])
            stereo = np.frombuffer(data, dtype='<f4').reshape(-1, 2).astype(np.float64)
            if len(stereo) != round(self.prepared['prefix_end'] * RATE):
                raise ValueError('Unexpected stereo length')
            diagnostic = read(self.diagnostic_path)
            times = np.arange(len(stereo)) / RATE
            before = np.interp(times, diagnostic['times'], diagnostic['weights_left'])
            after = sharpen_weights(before, .52, 32)
            mirrored = sharpen_weights(1 - before, .52, 32)
            if not np.allclose(after + mirrored, 1, atol=1e-12):
                raise ValueError('Curve symmetry failed')
            wave = after * stereo[:, 0] + (1 - after) * stereo[:, 1]
            if not np.isfinite(wave).all() or np.max(np.abs(wave)) > np.max(np.abs(stereo)) + 1e-12:
                raise ValueError('Invalid mixture')
            for a, b in self.prepared['excluded']:
                wave[round(a * RATE):round(b * RATE)] = 0
            path.parent.mkdir(exist_ok=True)
            wave.astype('<f4').tofile(path)
            step = round(.1 * RATE)
            write(path.with_suffix('.json'), {'sha256': digest(path), 'samples': len(wave),
                  'times': times[::step].tolist(), 'weights_before': before[::step].tolist(),
                  'weights_after': after[::step].tolist(), 'method': 'energy', 'speed': 'fast',
                  'threshold_percent': 52, 'steepness': 32})
        wave = np.memmap(path, dtype='<f4', mode='r')
        for case in self.cases.values():
            if case['stage'] != 'dynamic':
                continue
            pilot = self.pilot_cases[case['config']['model']]
            pilot_wave = np.memmap(pilot['audio'], dtype='<f4', mode='r')
            if wave[:len(pilot_wave)].tobytes() != pilot_wave.tobytes():
                raise ValueError('Full PCM differs from pilot prefix')
            cfg = self.decoder(case)
            keys = [chunk_key(wave[round(c['start'] * RATE):round(c['end'] * RATE)], cfg, c)
                    for c in case['chunks']]
            for key in pilot['inference_cache_keys']:
                if key not in keys:
                    raise ValueError('Pilot chunk PCM, decoder or bounds differ')
                src = self.pilot / 'inference-cache' / key
                if any(read(src / stage / 'result.json')['status'] != 'ok' for stage in ('asr', 'alignment')):
                    raise ValueError('Incomplete pilot cache')
                dst = self.root / 'inference-cache' / key
                if not dst.exists():
                    shutil.copytree(src, dst)
            case.update(audio=str(path), inference_cache_keys=keys, reused_pilot_keys=pilot['inference_cache_keys'])
            write(self.root / 'cases' / case['id'] / 'case.json', case)
        self.summary()

    @staticmethod
    def decoder(case):
        return {k: v for k, v in case['config'].items() if k not in ('audio_variant', 'maximum', 'context')}

    def execute(self, case):
        if case['status'] == 'ok' or (case['status'] == 'error' and not self.args.retry_failed):
            return
        directory = self.root / 'cases' / case['id']
        case.update(status='running')
        write(directory / 'case.json', case)
        try:
            transcripts, aligned = [], []
            for chunk, key in zip(case['chunks'], case['inference_cache_keys'], strict=True):
                shared = self.root / 'inference-cache' / key
                if (shared / 'asr/result.json').exists() and read(shared / 'asr/result.json')['status'] == 'ok':
                    asr = read(shared / 'asr/result.json')
                else:
                    asr = worker({'operation': 'asr', 'audio': case['audio'],
                                  'config': self.decoder(case), 'chunks': [chunk]},
                                 shared / 'asr', 'gigaam', self.args.retry_failed)
                if asr['status'] != 'ok':
                    raise RuntimeError(asr.get('error', 'ASR failed'))
                transcript = {k: asr['transcripts'][0][k] for k in ('chunk', 'text')}
                if transcript['chunk'] != chunk:
                    raise ValueError('Cached transcript has different bounds')
                result_path = shared / 'alignment/result.json'
                if result_path.exists() and read(result_path)['status'] == 'ok':
                    result = read(result_path)
                else:
                    result = worker({'operation': 'align', 'audio': self.prepared['audio'],
                                     'transcripts': [transcript], 'cache': str(shared / 'alignment-cache')},
                                    shared / 'alignment', retry=self.args.retry_failed)
                if result['status'] != 'ok':
                    raise RuntimeError(result.get('error', 'Alignment failed'))
                transcripts.append(transcript)
                aligned.append(read(shared / 'alignment-cache/alignment' / (alignment_key(transcript) + '.json')))
                self.summary()
            write(directory / 'asr/result.json', {'status': 'ok', 'transcripts': transcripts})
            case.update(self.finish(case, aligned, directory))
            pilot_score = self.pilot_cases[case['config']['model']]['score']
            full_score = case['score']['cards']['R005']
            write(directory / 'pilot-r005-check.json', {'matches': full_score == pilot_score,
                  'pilot': pilot_score, 'full': full_score})
            if full_score != pilot_score:
                raise ValueError('Full R005 differs from pilot; inspect pilot-r005-check.json')
            windows = {name: card['window'] for group in self.reference['groups']
                       for name, card in group['cards'].items()}
            write(directory / 'focus-audit.json', {name: {
                  'score': case['score']['cards'][name],
                  'raw_transcripts': [t for t in transcripts if t['chunk']['end'] > windows[name][0]
                                     and t['chunk']['start'] < windows[name][1]]}
                  for name in ('R004', 'R005')})
        except Exception as exc:
            case.update(status='error', error=str(exc))
        write(directory / 'case.json', case)
        self.summary()
        print('CASE', case['label'], case['status'], flush=True)

    def summary(self, blocked=False):
        complete = len(self.cases) == 7 and all(c['status'] == 'ok' for c in self.cases.values())
        write(self.root / 'execution.json', {'status': 'blocked' if blocked else 'complete' if complete else 'incomplete',
              'reason': 'cuda_unavailable' if blocked else None, 'expected_cases': 7,
              'completed': sum(c['status'] == 'ok' for c in self.cases.values())})
        write(self.root / 'results.json', {'complete': complete, 'cases': list(self.cases.values())})


def progress(root):
    cases = [read(p) for p in (root / 'cases').glob('*/case.json')]
    new = [c for c in cases if c['stage'] == 'dynamic']
    total = sum(len(c['chunks']) * 2 for c in new)
    done = 0
    for case in new:
        for key in case.get('inference_cache_keys', []):
            for stage in ('asr', 'alignment'):
                path = root / 'inference-cache' / key / stage / 'result.json'
                done += int(path.exists() and read(path)['status'] == 'ok')
    return {'execution': read(root / 'execution.json'), 'completed_logical_steps': done,
            'total_logical_steps': total, 'percent': round(100 * done / total, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'run', 'progress', 'report'))
    parser.add_argument('--pilot', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / '.lecture-cache/gigaam-channel-curve-full')
    parser.add_argument('--audio', type=Path, default=ROOT / 'record/20260925_101716.m4a')
    parser.add_argument('--review', type=Path, default=ROOT / 'record/20260925_101716.review.md')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    if args.stage in ('progress', 'report'):
        root = Path(read(args.output / 'latest.json')['run'])
        if args.stage == 'progress':
            print(json.dumps(progress(root)))
        else:
            if read(root / 'execution.json')['status'] != 'complete':
                raise ValueError('Final report requires all seven successful configurations')
            subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/create_gigaam_report.py'),
                  '--run', str(root), '--output', str(ROOT / 'output/pdf/r001-r008-gigaam-energy-fast-t52-k32.pdf')], check=True)
        return
    args.pilot = args.pilot or Path(read(ROOT / '.lecture-cache/gigaam-channel-curves-r005/latest.json')['run'])
    study = FullStudy(args)
    study.prepare()
    if args.stage == 'run':
        probe = subprocess.run([str(ROOT / 'experiments/gigaam/.venv/bin/python'), '-c',
                  'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'])
        if probe.returncode:
            study.summary(blocked=True)
            raise RuntimeError('CUDA unavailable; full ASR not started')
        for case in study.cases.values():
            if case['stage'] == 'dynamic':
                study.execute(case)
        if read(study.root / 'execution.json')['status'] == 'complete':
            subprocess.run([str(ROOT / '.venv/bin/python'), str(ROOT / 'scripts/benchmark_channel_curve_full.py'),
                            'report', '--output', str(args.output)], check=True)


if __name__ == '__main__':
    main()
