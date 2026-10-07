"""Cache-only Gibbs/fullness study; never modify or splice source hypotheses."""

import argparse
import itertools
import math
import unicodedata
from pathlib import Path

from .audio import digest
from .channel_research_report import comparison
from .channel_utility import aggregate_scores
from .evaluation import edit_score, lexical, score_case, select_text
from .experiments import identity
from .gibbs_study import decisions, selected_words
from .model_benchmark import ROOT, extra_metrics, read, write

METRICS = ('letters', 'words')
POLICIES = ('symmetric_guard',)
ALLOWANCES = (0., .02, .05)


def length(text, metric):
    text = unicodedata.normalize('NFC', text)
    if metric == 'letters':
        return sum(c.isalpha() for c in text)
    if metric == 'words':
        return len(lexical(text))
    raise ValueError('Unknown length metric')


def completeness_decisions(rows, left, right, policy, metric='letters', allowance=0.):
    if policy not in (*POLICIES, 'pure_longer', 'gibbs', 'left'):
        raise ValueError('Unknown completeness policy')
    if not math.isfinite(allowance) or not 0 <= allowance < 1:
        raise ValueError('Invalid allowed shortening')
    if len(rows) != len(left) or len(rows) != len(right):
        raise ValueError('Input chunk counts differ')
    proposals = decisions(rows, 'core', 'independent', .05)
    output = []
    for row, proposal, lhs, rhs in zip(rows, proposals, left, right):
        sizes = {'left': length(lhs['text'], metric), 'right': length(rhs['text'], metric)}
        longer = 'right' if sizes['right'] > sizes['left'] else 'left'
        shorter = 'left' if longer == 'right' else 'right'
        significant = sizes[shorter] < sizes[longer] * (1 - allowance)
        channel, reason = proposal['channel'], proposal['reason']
        if policy == 'left' or row['same_text']:
            channel, reason = 'left', 'fixed_or_identical'
        elif policy == 'pure_longer':
            channel, reason = longer, 'length_only'
        elif proposal['reason'] == 'insufficient_support':
            channel, reason = 'left', 'insufficient_support'
        elif policy == 'symmetric_guard' and significant and channel != longer:
            channel, reason = longer, 'shorter_candidate_veto'
        output.append({**proposal, 'channel': channel, 'reason': reason,
                       'gibbs_channel': proposal['channel'], 'lengths': sizes})
    return output


def configurations():
    result = [{'name': 'left', 'policy': 'left', 'metric': 'letters', 'allowance': 0.},
              {'name': 'Gibbs core t0.05', 'policy': 'gibbs', 'metric': 'letters', 'allowance': 0.}]
    result += [{'name': f'pure_longer/{m}', 'policy': 'pure_longer', 'metric': m, 'allowance': 0.} for m in METRICS]
    result += [{'name': f'{p}/{m}/loss{a:.0%}', 'policy': p, 'metric': m, 'allowance': a}
               for p, m, a in itertools.product(POLICIES, METRICS, ALLOWANCES)]
    return result


def choose(inputs, config):
    selected = completeness_decisions(inputs['rows'], inputs['left']['transcripts'], inputs['right']['transcripts'],
                                      config['policy'], config['metric'], config['allowance'])
    return selected_words(inputs['left'], inputs['right'], selected), selected


def build_report(root, result, reference):
    cases = sorted(result['development'], key=lambda c: c['score']['total']['wer'], reverse=True)
    columns = [c['name'] for c in cases]
    rows = []
    for sid, card in {k: v for g in reference['groups'] for k, v in g['cards'].items()}.items():
        rows.append({'label': f"{sid}\n{card['window'][0]:.2f}-{card['window'][1]:.2f} с", 'reference': card['text'],
                     'hypotheses': {c['name']: c['score']['cards'][sid]['hypothesis'] for c in cases},
                     'scores': {c['name']: c['score']['cards'][sid] for c in cases}})
    notice = ('Gibbs core/independent/t0.05 + полнота исходных ASR-чанков. Буквы: Unicode-буквы без пробелов, '
              'пунктуации и цифр; слова: lexical-токены. symmetric_guard блокирует более короткий '
              'выбранный канал в обе стороны при превышении допуска, иначе сохраняет Gibbs. '
              'pure_longer выбирает только по длине. Тексты не редактируются и не смешиваются. '
              'Историческая раскладка/current, новый ASR не запускался. '
              'Общий dev WER по трём группам (246 слов), R-карточки перекрываются. '
              'S001-S012 уже просмотрены, включая пример Эйлера: это разработочная проверка.')
    appendix = [f"{c['name']}: dev WER {c['score']['total']['wer']:.2%}; "
                f"ошибки {c['score']['total']['errors']}; числа/отрицания "
                f"{c['score']['total']['number_errors']}/{c['score']['total']['negation_errors']}; "
                f"S WER {result['metrics'][c['name']]['wer']:.2%}; ошибки {result['metrics'][c['name']]['errors']}; "
                f"S/D/I {result['metrics'][c['name']]['S']}/{result['metrics'][c['name']]['D']}/{result['metrics'][c['name']]['I']}"
                for c in cases]
    pdf = ROOT / 'output/pdf/r001-r008-gibbs-completeness.pdf'
    comparison(pdf, 'CTC Gibbs: защита полноты', columns, rows, appendix, notice)
    paths = [str(pdf)]
    columns = sorted(result['metrics'], key=lambda c: result['metrics'][c]['wer'], reverse=True)
    rows = [{'label': f"{f['id']}\n{f['window'][0]:.2f}-{f['window'][1]:.2f} с", 'reference': f['reference'],
             'hypotheses': {c: f['results'][c]['hypothesis'] for c in columns},
             'scores': {c: f['results'][c]['score'] for c in columns}} for f in result['fragments']]
    pdf = ROOT / 'output/pdf/s001-s012-gibbs-completeness.pdf'
    comparison(pdf, 'CTC Gibbs: полнота на просмотренных S001-S012', columns, rows, appendix, notice)
    write(root / 'reports.json', paths + [str(pdf)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    args = parser.parse_args()
    source = (args.source or Path(read(ROOT / '.lecture-cache/gibbs-study/latest.json')['run'])).resolve()
    configs = configurations()
    # Freeze the complete grid before reading reference contents or evaluating S.
    protocol = {'series': 'gibbs-completeness-v1', 'source': str(source), 'configs': configs,
                'dev_inputs_sha256': digest(source / 'dev-inputs.json'),
                'validation_sha256': digest(source / 'validation.json'),
                'reference_sha256': digest(source / 'reference.json'),
                'code_sha256': digest(Path(__file__)), 'gibbs_code_sha256': digest(ROOT / 'src/lecture_recognition/gibbs_study.py'),
                'evaluation_code_sha256': digest(ROOT / 'src/lecture_recognition/evaluation.py'),
                'utility_code_sha256': digest(ROOT / 'src/lecture_recognition/channel_utility.py'),
                'timeline_code_sha256': digest(ROOT / 'src/lecture_recognition/timeline.py'),
                'selection_data': 'none; all predetermined configurations displayed',
                'independent': False, 'new_asr_runs': 0, 'motivation': 'already observed S010 omission'}
    root = ROOT / '.lecture-cache/gibbs-completeness' / identity(protocol)[:16]
    write(root / 'protocol.json', protocol)
    write(root.parent / 'latest.json', {'run': str(root)})
    print('COMPLETENESS', root, flush=True)
    write(root / 'status.json', {'status': 'running'})
    reference, inputs = read(source / 'reference.json'), read(source / 'dev-inputs.json')
    development = []
    for config in configs:
        words, selected = choose(inputs, config)
        score = extra_metrics(score_case(words, reference, 2))
        development.append({**config, 'score': score, 'decisions': selected})
        print(config['name'], 'dev errors', score['total']['errors'], flush=True)
    fragments = []
    prior = read(source / 'validation.json')
    if not prior['complete'] or len(prior['fragments']) != 12:
        raise ValueError('Need the complete previous S validation')
    for fragment in prior['fragments']:
        outcomes = {}
        for config in configs:
            words, selected = choose(fragment, config)
            text = select_text(words, *fragment['window'])
            score = edit_score(fragment['reference'], text, 2)
            outcomes[config['name']] = {'hypothesis': text, 'score': score, 'words': words, 'decisions': selected}
        for name, original in [('left', 'left'), ('Gibbs core t0.05', 'core/independent/t0.05')]:
            if outcomes[name]['score'] != fragment['results'][original]['score']:
                raise ValueError('Control not reproduced: ' + fragment['id'])
        fragments.append({'id': fragment['id'], 'window': fragment['window'], 'reference': fragment['reference'], 'results': outcomes})
    metrics = {c['name']: aggregate_scores([f['results'][c['name']]['score'] for f in fragments]) for c in configs}
    original_dev = read(source / 'development.json')
    for name, baseline in [('left', original_dev['baseline']), ('Gibbs core t0.05', next(
            c['score'] for c in original_dev['cases'] if c['name'] == 'core/independent/t0.05'))]:
        if next(c for c in development if c['name'] == name)['score'] != baseline:
            raise ValueError('Dev control changed')
    result = {'complete': True, 'development': development, 'fragments': fragments, 'metrics': metrics,
              'independent': False, 'production_admission': False, 'new_asr_runs': 0}
    write(root / 'results.json', result)
    build_report(root, result, reference)
    write(root / 'status.json', {'status': 'complete', 'configs': len(configs), 'fragments': 12})
    for name, metrics in result['metrics'].items():
        print(name, 'S errors', metrics['errors'], 'D', metrics['D'], flush=True)


if __name__ == '__main__':
    main()
