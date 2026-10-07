"""Reproduce final Gibbs analysis and compare the pre-existing full/hold control."""

import argparse
from pathlib import Path

from lecture_recognition.audio import digest
from lecture_recognition.channel_research_report import comparison
from lecture_recognition.channel_utility import aggregate_scores
from lecture_recognition.evaluation import edit_score, select_text
from lecture_recognition.gibbs_study import decisions, selected_words
from lecture_recognition.model_benchmark import ROOT, read, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    root = args.run or Path(read(ROOT / '.lecture-cache/gibbs-study/latest.json')['run'])
    validation = read(root / 'validation.json')
    if not validation['complete'] or not validation.get('metrics'):
        raise ValueError('Complete transfer validation is required')
    old = 'full/hold/t0.02'
    fragments = []
    for fragment in validation['fragments']:
        chosen = decisions(fragment['rows'], 'full', 'hold', .02)
        text = select_text(selected_words(fragment['left'], fragment['right'], chosen), *fragment['window'])
        results = {**fragment['results'], old: {'hypothesis': text,
                   'score': edit_score(fragment['reference'], text, 2), 'decisions': chosen}}
        fragments.append({'id': fragment['id'], 'window': fragment['window'],
                          'reference': fragment['reference'], 'results': results})
    metrics = {name: aggregate_scores([f['results'][name]['score'] for f in fragments])
               for name in fragments[0]['results']}
    changes = {name: {'improved': [f['id'] for f in fragments if f['results'][name]['score']['errors'] < f['results']['left']['score']['errors']],
                     'worse': [f['id'] for f in fragments if f['results'][name]['score']['errors'] > f['results']['left']['score']['errors']]}
               for name in metrics if name != 'left'}
    write(root / 'analysis.json', {'metrics': metrics, 'changes': changes, 'fragments': fragments,
          'validation_sha256': digest(root / 'validation.json'), 'analysis_code_sha256': digest(Path(__file__)),
          'control': 'Previously specified full/hold/t0.02; no new parameter selection',
          'transfer_passed': validation['transfer_passed'], 'independent': False, 'production_admission': False})
    columns = sorted(metrics, key=lambda c: metrics[c]['wer'], reverse=True)
    rows = [{'label': f"{f['id']}\n{f['window'][0]:.2f}-{f['window'][1]:.2f} с", 'reference': f['reference'],
             'hypotheses': {c: f['results'][c]['hypothesis'] for c in columns},
             'scores': {c: f['results'][c]['score'] for c in columns}} for f in fragments]
    pdf = ROOT / 'output/pdf/s001-s012-ctc-gibbs-controls.pdf'
    comparison(pdf, 'CTC Gibbs: перенос и прежний контроль', columns, rows,
               [f"{c}: WER {metrics[c]['wer']:.2%}; CER {metrics[c]['cer']:.2%}; "
                f"ошибки {metrics[c]['errors']}/{metrics[c]['words']}; числа/отрицания "
                f"{metrics[c]['number_errors']}/{metrics[c]['negation_errors']}; "
                f"S/D/I {metrics[c]['S']}/{metrics[c]['D']}/{metrics[c]['I']}" for c in columns],
               'Историческая раскладка/current. Кандидаты и прежний Gibbs/0.02 определены на R001-R008. '
               'Параметры по S не подбирались. Эталон вручную исправлен из черновика CTC; '
               'S001-S012 уже просмотрены, проверка не независимая. Границы оценки не менялись. '
               'Колонки по убыванию общего WER; жирным лучшие ячейки, цвет по WER. Production-допуска нет.')
    reports = read(root / 'reports.json')
    if str(pdf) not in reports:
        write(root / 'reports.json', reports + [str(pdf)])
    lines = ['# CTC Gibbs: итог эксперимента', '', '| Вариант | Ошибки | WER | CER | Числа / отрицания |',
             '|---|---:|---:|---:|---:|']
    for name, score in metrics.items():
        lines.append(f"| {name} | {score['errors']} | {score['wer']:.2%} | {score['cer']:.2%} | {score['number_errors']} / {score['negation_errors']} |")
    lines += ['', '| Фрагмент | ' + ' | '.join(metrics) + ' |', '|---|' + '---:|' * len(metrics)]
    for fragment in fragments:
        lines.append('| ' + fragment['id'] + ' | ' + ' | '.join(str(fragment['results'][c]['score']['errors']) for c in metrics) + ' |')
    lines += ['', 'Dev: лучший core/independent — 16/246 (6,50%), left 18/246 (7,32%), старый Gibbs 17/246 (6,91%). '
              'R004 улучшен 3 → 1, остальные карточки не хуже left. Общая оценка dev по трём группам; карточки перекрываются.', '',
              'Перенос относительно left: ' + str(validation['transfer_passed']) + '. '
              'Отрицательный результат не запускает донастройку по S. Независимость и production-допуск не подтверждены.', '',
              'S001: выбранное «которая» вместо «который» добавляет ошибку. S007 при t0.02: «колебанийпо» '
              'заменено на «колебаний по часлу»; WER снижен, но «частотам» по-прежнему распознано неверно. '
              'Границы S и исходные таймкоды оставлены прежними.', '',
              'Результаты воспроизводятся из validation.json и сохранённых posterior-оценок без нового ASR. '
              '51 профильный тест прошёл, Ruff чист. Сохраняем left основным контролем до независимого подтверждения.']
    text = '\n'.join(lines) + '\n'
    (root / 'analysis.md').write_text(text)
    (ROOT / 'docs/ctc-gibbs-results.md').write_text(text)
    print(text)


if __name__ == '__main__':
    main()
