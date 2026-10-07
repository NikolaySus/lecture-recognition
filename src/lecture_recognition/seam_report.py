"""A focused before/after seam report, with the full 30-case search appendix."""
from pathlib import Path

from .channel_research_report import comparison
from .model_benchmark import ROOT, read, write


def build(root):
    root = Path(root)
    result = read(root / 'results.json')
    assert result['complete']
    source = Path(read(root / 'metadata.json')['source'])
    frozen = read(source / 'chunk-frozen.json')['models']
    displayed = []
    for case in result['cases']:
        if case['layout'] == 'historical' or case['layout'].startswith(frozen[case['model']]['layout']):
            versions = ('baseline',) if case['layout'] == 'historical' else ('baseline', 'score')
            for version in versions:
                label = f"{case['model']} {case['layout']} " + ('current' if version == 'baseline' else 'contextual')
                displayed.append((label, case[version]))
    displayed.sort(key=lambda entry: entry[1]['total']['wer'], reverse=True)
    columns = [name for name, _ in displayed]
    rows = []
    reference = read(root / 'reference.json')
    for name, card in {name: card for g in reference['groups'] for name, card in g['cards'].items()}.items():
        a, b = card['window']
        rows.append({'label': f'{name}\n{a:.2f}-{b:.2f} с', 'reference': card['text'],
                     'hypotheses': {label: score['cards'][name]['hypothesis'] for label, score in displayed},
                     'scores': {label: score['cards'][name] for label, score in displayed}})
    appendix = []
    for case in result['cases']:
        a, b = case['baseline']['total'], case['score']['total']
        appendix.append(f"{case['model']} {case['layout']}: {a['errors']} -> {b['errors']} ошибок; "
                        f"WER {a['wer']:.2%} -> {b['wer']:.2%}; CER {a['cer']:.2%} -> {b['cer']:.2%}; "
                        f"числа/отрицания {a['number_errors']}/{a['negation_errors']} -> "
                        f"{b['number_errors']}/{b['negation_errors']}; "
                        f"ухудшенные карточки {case['regressed_cards']}.")
        for action in case['audit']:
            if action['reason'] == 'contextual_overlap_copy':
                appendix.append('  Исправление: ' + action['phrase'] + '; ' + action['evidence'] + '; ' + str(action['overlap']))
    path = ROOT / 'output/pdf/r001-r008-seam-contextual.pdf'
    comparison(path, 'Склейка полного перекрытия: current / contextual', columns, rows, appendix,
               'Все варианты используют одни и те же сырые ASR-гипотезы и выравнивания. Новых ASR-прогонов: 0. '
               'Проверено 26 раскладок и 4 сдвига; регрессий по карточкам, группам и защищённым словам нет. '
               'Общий WER считается по трём непересекающимся группам (246 слов). Карточки перекрываются. '
               'В основной таблице исторические контроли и финалисты ±2 с; все 30 случаев в приложении. '
               'S001-S012 не использовались. Спорные повторы без достаточных свидетельств сохраняются. '
               'Временные границы составных единиц остаются исходными; их точность отдельно не подтверждена.')
    write(root / 'reports.json', [str(path)])
    return path
