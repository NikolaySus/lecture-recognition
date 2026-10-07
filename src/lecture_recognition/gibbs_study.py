"""Fixed CTC Gibbs experiment, using posterior caches and unchanged ASR units."""

import argparse
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed
from difflib import SequenceMatcher
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .audio import RATE, digest
from .channel_research import REVIEW, Research, decoder, require_cuda
from .channel_utility import aggregate_scores, assemble, entropy_confidence, word_confidence
from .evaluation import edit_score, lexical, score_case, select_text
from .experiments import identity
from .model_benchmark import ROOT, extra_metrics, read, worker, write
from .model_benchmark import layout as historical_layout

MODEL = 'gigaam-ctc'
SCOPES = ('full', 'core', 'disagreement')
RULES = ('hold', 'independent')
LIMITS = (0., .02, .05)


def disagreement_words(left, right):
    """Map differing greedy lexical spans plus one neighbour, without references."""
    a, b = lexical(left), lexical(right)
    selected = [set(), set()]
    for tag, i, j, k, end in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == 'equal':
            continue
        selected[0].update(range(max(0, i - 1), min(len(a), j + 1)))
        selected[1].update(range(max(0, k - 1), min(len(b), end + 1)))
    return selected


def local_confidence(trace, chunk, scope, chosen=None):
    """Use native CTC symbol runs; filter their frames before word aggregation."""
    p = np.exp(trace['log_probs'])
    if scope == 'full':
        return word_confidence(p, trace['tokens'], trace['token_words'], trace['blank'], ctc=True)['gibbs']
    confidence = entropy_confidence(p)
    runs, previous = [], None
    for i, token in enumerate(trace['tokens']):
        if token == trace['blank']:
            previous = None
            continue
        if token == previous:
            runs[-1].append(i)
        else:
            runs.append([i])
        previous = token
    words = {}
    for indices in runs:
        word = int(trace['token_words'][indices[0]])
        if scope == 'disagreement' and word not in chosen:
            continue
        inside = [i for i in indices if chunk['core_start'] <= trace['times'][i] < chunk['core_end']]
        if inside:
            words.setdefault(word, []).append(float(np.min(confidence[inside])))
    return float(np.mean([min(v) for v in words.values()])) if len(words) >= 3 else None


def decisions(rows, scope, rule, threshold):
    current, result = 'left', []
    for row in rows:
        lhs, rhs = row[scope]
        if row['same_text']:
            current, reason = 'left', 'identical_text'
        elif lhs is None or rhs is None or not np.isfinite([lhs, rhs]).all():
            current, reason = 'left', 'insufficient_support'
        elif rule == 'independent':
            current = 'right' if rhs - lhs > threshold else 'left'
            reason = 'independent_advantage' if current == 'right' else 'left_default'
        elif abs(lhs - rhs) > threshold:
            current = 'left' if lhs > rhs else 'right'
            reason = 'advantage'
        else:
            reason = 'hold'
        result.append({'channel': current, 'reason': reason, 'left': lhs, 'right': rhs})
    return result


def load_research(source, retry):
    """Open the existing fixed series without advancing its latest pointer."""
    meta = read(source / 'metadata.json')
    if (meta.get('fixed_layout'), meta.get('merger_override')) != ('historical', 'current'):
        raise ValueError('Expected the fixed historical/current research series')
    # Guard cached inference compatibility without recreating the earlier run.
    for name, expected in meta['code'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Research code changed: ' + name)
    research = Research.__new__(Research)
    research.root = source
    research.source = Path(meta['source'])
    research.metadata = read(research.source / 'metadata.json')
    dynamic = Path(research.metadata['source'])
    historical = read(Path(read(dynamic / 'metadata.json')['source']) / 'metadata.json')
    research.prepared = read(Path(historical['source']) / 'prepared.json')
    research.reference = read(source / 'reference.json')
    cases = [read(p) for p in (research.source / 'cases').glob('*/case.json')]
    research.parents = {MODEL: next(c for c in cases if c['label'] == 'gigaam-ctc combination' and c['status'] == 'ok')}
    research.args = SimpleNamespace(output=source.parent, retry_failed=retry)
    research.diagnostic_costs = {}
    return research


def traces(research, channel, chunks, competitors, inference_root=None):
    settings = {'operation': 'channel-diagnostics', 'audio': str(research.audio(channel)),
                'config': decoder(research.parents[MODEL]['config']), 'chunks': chunks, 'competitors': competitors}
    folder = research.root / 'diagnostics' / identity(settings)[:16]
    if not (folder / 'result.json').exists() or read(folder / 'result.json')['status'] != 'ok':
        folder = (inference_root or research.root) / 'diagnostics' / identity(settings)[:16]
        if not (folder / 'result.json').exists() or read(folder / 'result.json')['status'] != 'ok':
            require_cuda()
        result = worker({**settings, 'trace_dir': str(folder / 'traces')}, folder, 'gigaam', research.args.retry_failed)
    else:
        result = read(folder / 'result.json')
    if result['status'] != 'ok':
        raise RuntimeError(result.get('error', 'Diagnostics failed'))
    values = []
    for item in result['items']:
        if digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Posterior trace changed')
        with np.load(item['path']) as arrays:
            values.append({**{k: arrays[k].copy() for k in arrays.files},
                           'blank': item['blank'], 'text': item['text'], 'sha256': item['sha256']})
    return values


def measure_rows(research, chunks, left, right, root=None):
    competitors = {str(i): [a['text'], b['text']] for i, (a, b) in enumerate(zip(left['transcripts'], right['transcripts']))}
    lt, rt = [traces(research, c, chunks, competitors, root) for c in ('left', 'right')]
    rows = []
    for i, chunk in enumerate(chunks):
        sets = disagreement_words(lt[i]['text'], rt[i]['text'])
        row = {'chunk': chunk, 'same_text': competitors[str(i)][0] == competitors[str(i)][1],
               'trace_sha256': [lt[i]['sha256'], rt[i]['sha256']]}
        for scope in SCOPES:
            row[scope] = [local_confidence(t, chunk, scope, s) for t, s in zip((lt[i], rt[i]), sets)]
        rows.append(row)
    return rows


def selected_words(left, right, chosen):
    aligned = [({'left': left, 'right': right}[d['channel']])['aligned'][i] for i, d in enumerate(chosen)]
    return assemble(aligned, 'current')


def case_name(scope, rule, threshold):
    return f'{scope}/{rule}/t{threshold:g}'


def development(research, root):
    data = read(research.root / 'diagnose.json')['models'][MODEL]
    chunks, left, right = data['layout']['chunks'], data['left'], data['right']
    rows = measure_rows(research, chunks, left, right)
    write(root / 'dev-inputs.json', {'rows': rows, 'left': left, 'right': right})
    cases = []
    baseline = extra_metrics(score_case(assemble(left['aligned'], 'current'), research.reference, 2))
    if baseline['total']['errors'] != 18:
        raise ValueError('Left control did not reproduce 18 errors')
    for scope, rule, threshold in itertools.product(SCOPES, RULES, LIMITS):
        chosen = decisions(rows, scope, rule, threshold)
        score = extra_metrics(score_case(selected_words(left, right, chosen), research.reference, 2))
        case = {'name': case_name(scope, rule, threshold), 'scope': scope, 'rule': rule,
                'threshold': threshold, 'score': score, 'decisions': chosen,
                'right_chunks': sum(d['channel'] == 'right' for d in chosen),
                'switches': sum(a['channel'] != b['channel'] for a, b in zip(chosen, chosen[1:]))}
        cases.append(case)
        write(root / 'development.json', {'complete': False, 'baseline': baseline, 'cases': cases})
        print(case['name'], score['total']['errors'], 'errors', flush=True)
    original = next(c for c in cases if c['name'] == 'full/hold/t0.02')
    prior = next(c for c in read(research.root / 'develop.json')['cases']
                 if c['model'] == MODEL and c['policy'] == 'gibbs' and c['threshold'] == .02)
    if original['score'] != prior['score'] or original['score']['total']['errors'] != 17:
        raise ValueError('Historical Gibbs did not reproduce the exact original score')
    def critical(s):
        return s['number_errors'] + s['negation_errors']
    eligible = [c for c in cases if c['score']['total']['errors'] <= baseline['total']['errors']
                and critical(c['score']['total']) <= critical(baseline['total'])
                and c['score']['cards']['R004']['errors'] < baseline['cards']['R004']['errors']]
    eligible.sort(key=lambda c: (c['score']['total']['errors'], critical(c['score']['total']),
                               c['score']['total']['cer'], c['right_chunks'], c['rule'] != 'independent',
                               -c['threshold'], c['name']))
    # Two distinct decision sequences, not duplicate thresholds of the same output.
    selected, seen = [], set()
    for c in eligible:
        signature = tuple(d['channel'] for d in c['decisions'])
        if signature in seen:
            continue
        selected.append({k: c[k] for k in ('name', 'scope', 'rule', 'threshold')})
        seen.add(signature)
        if len(selected) == 2:
            break
    write(root / 'development.json', {'complete': True, 'baseline': baseline, 'cases': cases})
    frozen = {'candidates': selected, 'development_sha256': digest(root / 'development.json'),
              'protocol_sha256': digest(root / 'protocol.json'), 'selection_data': 'R001-R008 only'}
    path = root / 'frozen.json'
    if path.exists() and read(path) != frozen:
        raise ValueError('Frozen candidates changed')
    write(path, frozen)
    return read(root / 'development.json')


def validation(research, root, jobs):
    frozen = read(root / 'frozen.json')
    if frozen['development_sha256'] != digest(root / 'development.json') or frozen['protocol_sha256'] != digest(root / 'protocol.json'):
        raise ValueError('Frozen protocol changed')
    if not frozen['candidates']:
        write(root / 'validation.json', {'complete': True, 'skipped': 'no eligible dev candidates', 'fragments': []})
        return
    # This is the first access to test reference contents, after candidate freezing.
    references = research.validation_reference()
    reference_sha = digest(REVIEW)
    windows = read(research.root / 'validation-windows.json')
    mono = np.memmap(research.prepared['audio'], dtype='<f4', mode='r')
    configs = [{'name': 'left'}] + frozen['candidates']
    def fragment(item):
        path = root / 'validation-cases' / (item['id'] + '.json')
        if path.exists():
            cached = read(path)
            if cached['reference_sha256'] != reference_sha or cached['frozen_sha256'] != digest(root / 'frozen.json'):
                raise ValueError('Test cache inputs changed')
            return cached
        start, end = item['input_window']
        regions = [(a - start, b - start) for a, b in research.prepared['regions']]
        local = historical_layout(mono[round(start * RATE):round(end * RATE)], end - start, 20, regions)
        chunks = [{k: v + start for k, v in c.items()} for c in local]
        print('TEST', item['id'], flush=True)
        left, right = [research.infer(MODEL, c, chunks, split='test') for c in ('left', 'right')]
        if any(v['status'] != 'ok' for v in (left, right)):
            raise RuntimeError('Test inference failed: ' + item['id'])
        rows = measure_rows(research, chunks, left, right, root)
        results = {}
        for config in configs:
            chosen = ([{'channel': 'left'} for _ in chunks] if config['name'] == 'left' else
                      decisions(rows, config['scope'], config['rule'], config['threshold']))
            words = selected_words(left, right, chosen)
            text = select_text(words, *item['window'])
            results[config['name']] = {'hypothesis': text, 'score': edit_score(references[item['id']], text, 2),
                                       'words': words, 'decisions': chosen}
        result = {**item, 'reference': references[item['id']], 'reference_sha256': reference_sha,
                  'frozen_sha256': digest(root / 'frozen.json'), 'results': results,
                  'rows': rows, 'chunks': chunks, 'left': left, 'right': right}
        write(path, result)
        return result
    completed = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [pool.submit(fragment, item) for item in windows]
        for future in as_completed(futures):
            completed.append(future.result())
            write(root / 'status.json', {'stage': 'validation', 'status': 'running', 'completed': len(completed), 'total': 12})
    completed.sort(key=lambda f: f['id'])
    metrics = {c['name']: aggregate_scores([f['results'][c['name']]['score'] for f in completed]) for c in configs}
    base = metrics['left']
    passed = {c['name']: (metrics[c['name']]['wer'] <= base['wer'] and
                         metrics[c['name']]['number_errors'] <= base['number_errors'] and
                         metrics[c['name']]['negation_errors'] <= base['negation_errors']) for c in frozen['candidates']}
    if base['errors'] != 60 or base['words'] != 948:
        raise ValueError('Historical S001-S012 baseline not reproduced')
    write(root / 'validation.json', {'complete': True, 'fragments': completed, 'metrics': metrics,
                                    'transfer_passed': passed, 'independent': False, 'production_admission': False})


def report(root):
    from .channel_research_report import comparison
    reference = read(root / 'reference.json')
    dev = read(root / 'development.json')
    cases = [{'name': 'left', 'score': dev['baseline']}] + dev['cases']
    cases.sort(key=lambda c: c['score']['total']['wer'], reverse=True)
    columns = [c['name'] for c in cases]
    rows = []
    for sid, card in {k: v for g in reference['groups'] for k, v in g['cards'].items()}.items():
        rows.append({'label': f"{sid}\n{card['window'][0]:.2f}-{card['window'][1]:.2f} с", 'reference': card['text'],
                     'hypotheses': {c['name']: c['score']['cards'][sid]['hypothesis'] for c in cases},
                     'scores': {c['name']: c['score']['cards'][sid] for c in cases}})
    appendix = [f"{c['name']}: WER {c['score']['total']['wer']:.2%}; ошибок {c['score']['total']['errors']}; "
                f"CER {c['score']['total']['cer']:.2%}; правых чанков {c.get('right_chunks', 0)}; "
                f"переключений {c.get('switches', 0)}" for c in cases]
    pdf = ROOT / 'output/pdf/r001-r008-ctc-gibbs.pdf'
    comparison(pdf, 'GigaAM CTC combination: устойчивость Gibbs', columns, rows, appendix,
               '18 конфигураций и left. Историческая раскладка/current, без нового ASR. '
               'Общий WER по трём группам, 246 слов; карточки перекрываются. Порядок: убывание общего WER. '
               'core: центральные кадры; disagreement: различающиеся greedy-слова с соседями. '
               'Локальная оценка требует >=3 слов на каждом канале, иначе left. Это разработочная оценка.')
    produced = [str(pdf)]
    if (root / 'validation.json').exists():
        val = read(root / 'validation.json')
        if val.get('metrics'):
            columns = sorted(val['metrics'], key=lambda c: val['metrics'][c]['wer'], reverse=True)
            rows = [{'label': f"{f['id']}\n{f['window'][0]:.2f}-{f['window'][1]:.2f} с", 'reference': f['reference'],
                     'hypotheses': {c: f['results'][c]['hypothesis'] for c in columns},
                     'scores': {c: f['results'][c]['score'] for c in columns}} for f in val['fragments']]
            pdf = ROOT / 'output/pdf/s001-s012-ctc-gibbs.pdf'
            comparison(pdf, 'Gibbs CTC: проверка переноса на S001-S012', columns, rows,
                       [f"{c}: {val['metrics'][c]}" for c in columns],
                       'Кандидаты зафиксированы на R001-R008. S001-S012 уже просмотрены: проверка не независимая. '
                       'Эталон исправлен вручную из ASR-черновика CTC. Параметры по S не подбирались. '
                       'Историческая раскладка/current, границы оценки не изменены. Допуска в production нет.')
            produced.append(str(pdf))
    write(root / 'reports.json', produced)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('development', 'validation', 'report', 'run'))
    parser.add_argument('--research', type=Path)
    parser.add_argument('--jobs', type=int, choices=(1, 2), default=2)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    source = (args.research or Path(read(ROOT / '.lecture-cache/channel-research/latest.json')['run'])).resolve()
    research = load_research(source, args.retry_failed)
    protocol = {'series': 'ctc-gibbs-v1', 'source': str(source), 'scopes': SCOPES, 'rules': RULES,
                'thresholds': LIMITS, 'min_local_words': 3, 'selection': 'top two distinct dev decision sequences',
                'code_sha256': digest(Path(__file__)), 'diagnose_sha256': digest(source / 'diagnose.json'),
                'reference_sha256': digest(source / 'reference.json'),
                'windows_sha256': digest(source / 'validation-windows.json'),
                'research_code': read(source / 'metadata.json')['code'], 'test_tuning': False}
    root = ROOT / '.lecture-cache/gibbs-study' / identity(protocol)[:16]
    write(root / 'protocol.json', protocol)
    write(root / 'reference.json', research.reference)
    write(root.parent / 'latest.json', {'run': str(root)})
    print('GIBBS', root, flush=True)
    try:
        if args.stage in ('development', 'run'):
            development(research, root)
            report(root)
        if args.stage in ('validation', 'run'):
            validation(research, root, args.jobs)
            report(root)
        if args.stage == 'report':
            report(root)
        write(root / 'status.json', {'stage': args.stage, 'status': 'complete'})
    except Exception as exc:
        write(root / 'status.json', {'stage': args.stage, 'status': 'blocked' if 'CUDA unavailable' in str(exc) else 'error',
                                     'reason': str(exc)})
        raise


if __name__ == '__main__':
    main()
