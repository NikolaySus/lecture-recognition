"""Bounded, resumable GigaAM inference-only study on the historical prefix."""

import argparse
import copy
import hashlib
import subprocess
import time
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .decoding_experiments import TERMS
from .evaluation import combine, reference_cards, score_case
from .experiments import identity
from .model_benchmark import MODELS, ROOT, extra_metrics, read, validate_srt, worker, write
from .timeline import Chunk, quiet_cut, to_srt

BASE_IDS = {'gigaam-ctc': 'd08a9d751d2c098d', 'gigaam-rnnt': 'd1615af5a08d5697'}
TERMS_FIXED = sorted(set(TERMS + [
    'корреляционно-спектрального анализа', 'временных рядов', 'временного ряда',
    'стационарности', 'нестационарности', 'нестационарных', 'скользящего анализа',
    'сегмента', 'сегментов', 'отсчета', 'отсчетов', 'перекрытия', 'статистики',
    'квазистационарность', 'квазистационарности',
]))
CONTROL_TERMS = ['автокорреляция', 'периодограмма', 'преобразование Фурье']


def eligible(candidate, baseline):
    if candidate.get('status') != 'ok' or baseline.get('status') != 'ok':
        return False
    cards = baseline['score']['cards']
    if set(candidate['score']['cards']) != set(cards):
        return False
    return not all(candidate['score']['cards'][key]['wer'] > value['wer'] for key, value in cards.items())


def ranking(case):
    if case.get('status') != 'ok':
        return (float('inf'),) * 4 + (case['id'],)
    s = case['score']['total']
    return (s['wer'], s['number_errors'] + s['negation_errors'], s['cer'],
            case.get('asr_seconds') or float('inf'), case['id'])


def make_layout(audio, duration, maximum=20, context=1, shift=0):
    core = maximum - 2 * context
    cuts, cursor = [0.0], 0.0
    while duration - cursor > core:
        upper = min(duration, cursor + core)
        cut = quiet_cut(audio, max(cursor + 1, upper - min(4, core / 4)), upper)
        if cursor == 0 and shift:
            cut = max(1, min(maximum - context, cut + shift))
        cuts.append(cut)
        cursor = cut
    cuts.append(duration)
    chunks = [Chunk(max(0, a-context), min(duration, b+context), a, b).dict()
              for a, b in zip(cuts, cuts[1:])]
    assert all(c['end'] - c['start'] <= maximum + 1e-6 for c in chunks)
    return chunks


class Study:
    def __init__(self, args):
        self.args = args
        self.source = args.source.resolve()
        self.old_reference = read(self.source / 'reference.json')
        self.prepared = read(self.source / 'prepared.json')
        self.duration = self.prepared['prefix_end']
        self.raw = np.memmap(self.prepared['audio'], mode='r', dtype='<f4')[:round(self.duration * RATE)]
        self.audio_hashes = {'original': hashlib.sha256(self.raw.tobytes()).hexdigest()}
        self.originals = {m: read(self.source / 'cases' / key / 'case.json') for m, key in BASE_IDS.items()}
        cards = reference_cards(args.review.read_text())
        reference = copy.deepcopy(self.old_reference)
        for group in reference['groups']:
            tokens, spans = combine(cards, group['cards'])
            group['text'] = ' '.join(tokens)
            group.pop('words', None)
            for name, card in group['cards'].items():
                card.update(text=cards[name], span=spans[name])
        reference['method'] = 'historical-fixed-windows-current-review'
        reference['normalization_version'] = 2
        code = [Path(__file__), ROOT/'scripts/asr_worker.py', Path(__file__).with_name('gigaam_decoding.py'),
                Path(__file__).with_name('evaluation.py'), Path(__file__).with_name('timeline.py'),
                Path(__file__).with_name('experiments.py'), Path(__file__).with_name('model_benchmark.py')]
        metadata = {'source': str(self.source), 'review_sha256': digest(args.review), 'reference': reference,
                    'code': {str(p.relative_to(ROOT)): digest(p) for p in code},
                    'terms': TERMS_FIXED, 'control_terms': CONTROL_TERMS,
                    'locks': {str(p.relative_to(ROOT)): digest(p) for p in
                              [ROOT/'uv.lock', ROOT/'experiments/gigaam/uv.lock']},
                    'baseline_files': {m: digest(self.source/'cases'/key/'case.json') for m, key in BASE_IDS.items()},
                    'audio_sha256': digest(Path(self.prepared['audio'])), 'source_audio_sha256': digest(args.audio)}
        if metadata['audio_sha256'] != self.prepared['masked_sha256']:
            raise ValueError('Historical masked audio was modified')
        self.root = args.output.resolve()/identity(metadata)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root/'metadata.json').exists():
            write(self.root/'metadata.json', metadata)
            write(self.root/'reference.json', reference)
            (self.root/'review-snapshot.md').write_text(args.review.read_text(), encoding='utf-8')
        self.reference = reference
        self.cases = {p.parent.name: read(p) for p in (self.root/'cases').glob('*/case.json')}
        print('SERIES', self.root, flush=True)
        write(args.output.resolve()/'latest.json', {'run': str(self.root)})

    def audio(self, variant):
        if variant == 'original':
            return Path(self.prepared['audio'])
        path = self.root/'audio'/f'{variant}.f32'
        if path.exists():
            return path
        path.parent.mkdir(exist_ok=True)
        if variant in ('left', 'right'):
            cmd = ['ffmpeg', '-v', 'error', '-i', str(self.args.audio), '-t', str(self.duration),
                   '-af', f'pan=mono|c0=c{0 if variant == "left" else 1}', '-ar', str(RATE),
                   '-f', 'f32le', '-']
            wave = np.frombuffer(subprocess.check_output(cmd), dtype='<f4').copy()
        elif variant == 'highpass':
            cmd = ['ffmpeg', '-v', 'error', '-f', 'f32le', '-ar', str(RATE), '-ac', '1', '-i', '-',
                   '-af', 'highpass=f=80', '-f', 'f32le', '-']
            wave = np.frombuffer(subprocess.check_output(cmd, input=self.raw.tobytes()), dtype='<f4').copy()
        elif variant == 'rms':
            rms = float(np.sqrt(np.mean(self.raw.astype(np.float64)**2)))
            gain = min(10**(-20/20)/max(rms, 1e-12), 10**(-1/20)/max(float(np.max(np.abs(self.raw))), 1e-12))
            wave = np.array(self.raw * gain, dtype='<f4')
        else:
            raise ValueError(variant)
        if len(wave) != len(self.raw) or not np.isfinite(wave).all():
            raise ValueError(f'Invalid audio: {variant}')
        for start, end in self.prepared['excluded']:
            wave[round(start*RATE):round(end*RATE)] = 0
        temporary = path.with_suffix('.tmp')
        wave.tofile(temporary)
        temporary.replace(path)
        write(path.with_suffix('.json'), {'sha256': digest(path), 'samples': len(wave),
                                         'peak': float(np.abs(wave).max())})
        return path

    def case(self, model, label, stage, **options):
        cfg = {**MODELS[model], 'model': model, 'dictionary': None, 'decoding': 'greedy',
               'maximum': 20, 'context': 1, 'audio_variant': 'original', 'precision': 'default',
               'diagnostic_greedy': model == 'gigaam-rnnt', **options}
        audio = self.audio(cfg['audio_variant'])
        if cfg['maximum'] == 20 and cfg['context'] == 1 and not cfg.get('shift'):
            chunks = self.originals[model]['chunks']
        else:
            chunks = make_layout(self.raw, self.duration, cfg['maximum'], cfg['context'], cfg.get('shift', 0))
        settings = {'config': cfg, 'chunks': chunks, 'stage': stage, 'label': label,
                    'audio': str(audio), 'repeat': cfg.get('repeat', 0)}
        if cfg['audio_variant'] not in self.audio_hashes:
            self.audio_hashes[cfg['audio_variant']] = digest(audio)
        inference_cfg = {k: v for k, v in cfg.items() if k not in ('audio_variant', 'maximum', 'context')}
        signature = identity({'config': inference_cfg, 'chunks': chunks,
                              'audio': self.audio_hashes[cfg['audio_variant']]})
        key = identity(settings)[:16]
        directory = self.root/'cases'/key
        if key in self.cases and (self.cases[key]['status'] == 'ok' or
                                  (self.cases[key]['status'] == 'error' and not self.args.retry_failed)):
            return self.cases[key]
        if stage not in ('validation', 'dynamic'):
            for previous in self.cases.values():
                if previous.get('signature') == signature and previous['status'] == 'ok':
                    aliases = read(self.root/'aliases.json') if (self.root/'aliases.json').exists() else {}
                    aliases[label] = {'case_id': previous['id'], 'reason': 'identical_input_and_decoder'}
                    write(self.root/'aliases.json', aliases)
                    return previous
        directory.mkdir(parents=True, exist_ok=True)
        record = {'id': key, **settings, 'signature': signature, 'status': 'running'}
        write(directory/'case.json', record)
        started = time.monotonic()
        try:
            result = worker({'operation': 'asr', 'audio': str(audio), 'config': cfg, 'chunks': chunks},
                            directory/'asr', 'gigaam', self.args.retry_failed)
            if result['status'] != 'ok':
                raise RuntimeError(result.get('error', 'ASR failed'))
            transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in result['transcripts']]
            aligned = worker({'operation': 'align', 'audio': self.prepared['audio'],
                              'transcripts': transcripts, 'cache': str(directory/'alignment-cache')},
                             directory/'alignment', retry=self.args.retry_failed)
            if aligned['status'] != 'ok':
                raise RuntimeError(aligned.get('error', 'Alignment failed'))
            words = aligned['words']
            score = extra_metrics(score_case(words, self.reference, version=2))
            legacy = extra_metrics(score_case(words, self.reference, version=1))
            srt = to_srt(words, self.prepared['regions'])
            validation = validate_srt(srt, self.prepared['duration'], self.prepared['excluded'])
            (directory/'transcript.srt').write_text(srt, encoding='utf-8')
            record.update(status='ok', score=score, legacy_score=legacy, validation=validation,
                          limit_hits=sum(t.get('limit_hits', 0) for t in result['transcripts']),
                          asr_seconds=result.get('measured_asr_seconds'), cached_chunks=result.get('cached_chunks'),
                          alignment_seconds=aligned.get('seconds'), peak_vram_gib=result.get('peak_vram_gib'))
        except Exception as exc:
            record.update(status='error', error=str(exc))
        record['elapsed_seconds'] = time.monotonic() - started
        write(directory/'case.json', record)
        self.cases[key] = record
        self.summary()
        print('CASE', label, record['status'], record.get('score', {}).get('total', {}).get('wer'), flush=True)
        return record

    def summary(self):
        bases = {c['config']['model']: c for c in self.cases.values() if c['stage'] == 'baseline'}
        results = []
        for c in self.cases.values():
            base = bases.get(c['config']['model'])
            keep = base is not None and eligible(c, base)
            results.append({'id': c['id'], 'label': c['label'], 'stage': c['stage'], 'status': c['status'],
                            'included': keep and c['stage'] != 'validation',
                            'reason': 'validation' if c['stage'] == 'validation' else
                            'incomplete' if c['status'] != 'ok' else 'included' if keep else 'worse_on_all_cards'})
        write(self.root/'summary.json', {'cases': results, 'normalization_version': 2,
                                       'baseline_ids': {m: c['id'] for m, c in bases.items()}})

    def rescore(self):
        for case in self.cases.values():
            if case['status'] != 'ok':
                continue
            words = read(self.root/'cases'/case['id']/'alignment/result.json')['words']
            case['score'] = extra_metrics(score_case(words, self.reference, version=2))
            case['legacy_score'] = extra_metrics(score_case(words, self.reference, version=1))
            write(self.root/'cases'/case['id']/'case.json', case)
        self.summary()

    def run(self, stage):
        interpreter = ROOT/'experiments/gigaam/.venv/bin/python'
        probe = subprocess.run([str(interpreter), '-c',
                                'import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)'],
                               capture_output=True, text=True)
        if probe.returncode:
            write(self.root/'execution.json', {'status': 'blocked', 'reason': 'cuda_unavailable',
                                                'probe_stderr': probe.stderr})
            raise RuntimeError('CUDA is unavailable in the GigaAM environment; no ASR cases started.')
        bases = {}
        for model in BASE_IDS:
            b = self.case(model, model+' original', 'baseline')
            if b['status'] != 'ok':
                raise RuntimeError(f'Baseline failed: {model}')
            old = read(self.source/'cases'/BASE_IDS[model]/'asr/result.json')
            new = read(self.root/'cases'/b['id']/'asr/result.json')
            if [t['text'] for t in old['transcripts']] != [t['text'] for t in new['transcripts']]:
                raise RuntimeError(f'Baseline transcripts differ: {model}')
            bases[model] = b
        for model in BASE_IDS:
            for tag, opts in [('12s', {'maximum': 12}), ('25s', {'maximum': 25}),
                              ('context2', {'context': 2}),
                              *[(v, {'audio_variant': v}) for v in ('left', 'right', 'highpass', 'rms')],
                              ('fp32', {'precision': 'fp32'})]:
                self.case(model, model+' '+tag, 'input', **opts)
        if stage == 'input':
            return
        for model in BASE_IDS:
            widths = (8, 32) if model.endswith('ctc') else (4, 8)
            for beam in widths:
                self.case(model, f'{model} beam{beam}', 'decoder', decoding='beam', beam=beam)
            for weight in (1, 2, 4):
                self.case(model, f'{model} bias{weight}', 'dictionary', decoding='beam', beam=widths[-1],
                          terms=TERMS_FIXED, bias_weight=weight)
            self.case(model, model+' control-bias2', 'control', decoding='beam', beam=widths[-1],
                      terms=CONTROL_TERMS, bias_weight=2)
        if stage == 'decode':
            return
        selection = {}
        decisions = []
        for model, base in bases.items():
            cases = [c for c in self.cases.values() if c['config']['model'] == model and c['status'] == 'ok']
            inp = min([base]+[c for c in cases if c['stage'] == 'input'], key=ranking)
            dec = min([base]+[c for c in cases if c['stage'] in ('decoder', 'dictionary')], key=ranking)
            if inp != base and dec != base:
                opts = {k: inp['config'][k] for k in ('maximum', 'context', 'audio_variant', 'precision')}
                opts.update({k: v for k, v in dec['config'].items() if k in ('decoding', 'beam', 'terms', 'bias_weight')})
                combo = self.case(model, model+' combination', 'combination', **opts)
                if combo['status'] == 'ok':
                    cases.append(combo)
            else:
                decisions.append({'model': model, 'stage': 'combination', 'reason': 'one_or_both_factors_did_not_improve'})
            if model.endswith('rnnt'):
                best_decoder = min([base]+[c for c in cases if c['stage'] in ('decoder', 'dictionary')], key=ranking)
                for source in {base['id']: base, best_decoder['id']: best_decoder}.values():
                    if source.get('limit_hits', 0):
                        opts = dict(source['config'])
                        opts.pop('model')
                        opts['max_symbols'] = 20
                        test = self.case(model, source['label']+' limit20', 'limit', **opts)
                        if test['status'] == 'ok':
                            cases.append(test)
                    else:
                        decisions.append({'model': model, 'stage': 'limit', 'source': source['id'],
                                          'reason': 'symbol_limit_not_reached'})
            candidate = min([base]+[c for c in cases if c['stage'] not in ('control', 'validation')], key=ranking)
            checks = []
            for source in {base['id']: base, candidate['id']: candidate}.values():
                for tag, changes in [('repeat', {'repeat': 1}), ('shift-5', {'shift': -5}), ('shift+5', {'shift': 5})]:
                    opts = dict(source['config'])
                    opts.pop('model')
                    opts.update(changes)
                    check = self.case(model, source['label']+' '+tag, 'validation', **opts)
                    checks.append({'source': source['id'], 'tag': tag, 'id': check['id']})
            lookup = {(c['source'], c['tag']): self.cases[c['id']] for c in checks}
            repeat_equal = {source['id']: (
                lookup[source['id'], 'repeat'].get('score') == source['score']
            ) for source in (base, candidate)}
            shifts = []
            for tag in ('shift-5', 'shift+5'):
                control, changed = lookup[base['id'], tag], lookup[candidate['id'], tag]
                if control['status'] == changed['status'] == 'ok':
                    shifts.append({'tag': tag, 'baseline': control['id'], 'candidate': changed['id'],
                                   'wer_delta': changed['score']['total']['wer'] - control['score']['total']['wer'],
                                   'card_deltas': {name: score['wer'] - control['score']['cards'][name]['wer']
                                                   for name, score in changed['score']['cards'].items()}})
                else:
                    shifts.append({'tag': tag, 'status': 'incomplete'})
            selection[model] = {'baseline': base['id'], 'candidate': candidate['id'], 'checks': checks,
                                'repeat_equal': repeat_equal, 'shift_comparisons': shifts}
        write(self.root/'selection.json', selection)
        write(self.root/'decisions.json', decisions)
        write(self.root/'execution.json', {'status': 'complete' if all(c['status'] == 'ok' for c in self.cases.values())
                                            else 'incomplete', 'cases': len(self.cases)})
        self.summary()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'input', 'decode', 'all', 'score'])
    parser.add_argument('--source', type=Path, default=ROOT/'.lecture-cache/model-benchmark/9915a85fd8f93d0c')
    parser.add_argument('--output', type=Path, default=ROOT/'.lecture-cache/gigaam-tuning')
    parser.add_argument('--review', type=Path, default=ROOT/'record/20260925_101716.review.md')
    parser.add_argument('--audio', type=Path, default=ROOT/'record/20260925_101716.m4a')
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    study = Study(args)
    if args.stage == 'score':
        study.rescore()
    elif args.stage == 'prepare':
        study.summary()
    else:
        study.run(args.stage)


if __name__ == '__main__':
    main()
