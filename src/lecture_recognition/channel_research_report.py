"""Separate layout and channel-selection research PDFs with complete search appendices."""
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .model_benchmark import ROOT, read, write


def comparison(path, title, columns, rows, appendix, notice):
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    for name, file in [('Research', 'DejaVuSans.ttf'), ('ResearchBold', 'DejaVuSans-Bold.ttf')]:
        if name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(name, str(fonts / file)))
    pdfmetrics.registerFontFamily('Research', normal='Research', bold='ResearchBold')
    style = ParagraphStyle('body', fontName='Research', fontSize=9, leading=13)
    heading = ParagraphStyle('heading', parent=style, fontSize=16, leading=21)
    def paragraph(text, bold=False):
        value = escape(str(text)).replace('\n', '<br/>')
        return Paragraph('<b>' + value + '</b>' if bold else value, style)
    widths = [115] + [220] * len(columns) + [240]
    width = max(800, sum(widths) + 48)
    height = max(850, 210 + sum(max(80, len(r['reference']) * .36) for r in rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    story = [Paragraph(escape(title), heading), Spacer(1, 12), paragraph(notice), Spacer(1, 12)]
    data = [[paragraph('Время / фрагмент')] + [paragraph(c) for c in columns] + [paragraph('Эталон')]]
    commands = [('VALIGN', (0, 0), (-1, -1), 'TOP'), ('GRID', (0, 0), (-1, -1), .4, colors.HexColor('#CBD5E1')),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E2E8F0')),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7)]
    for i, row in enumerate(rows, 1):
        scores = [row['scores'][c]['wer'] for c in columns]
        best = min(scores)
        cells = [paragraph(row['label'])]
        for j, (column, wer) in enumerate(zip(columns, scores), 1):
            score = row['scores'][column]
            cells.append(paragraph(f"{row['hypotheses'][column]}\nWER {wer:.2%} ({score['errors']}/{score['words']})",
                                   abs(wer - best) < 1e-12))
            t = min(1., wer / .5)
            shade = colors.Color(.85 + .15 * t, .97 - .22 * t, .86 - .13 * t)
            commands.append(('BACKGROUND', (j, i), (j, i), shade))
        cells.append(paragraph(row['reference']))
        data.append(cells)
    table = Table(data, colWidths=widths, repeatRows=1)
    table.setStyle(TableStyle(commands))
    story.extend([table, Spacer(1, 18), paragraph('Полная сетка экспериментов и метрики'), Spacer(1, 8)])
    for item in appendix:
        story.append(paragraph(item))
        story.append(Spacer(1, 4))
    height = max(height, sum(flow.wrap(width - 48, 1_000_000)[1] for flow in story) + 100)
    doc = SimpleDocTemplate(str(path), pagesize=(width, height), leftMargin=24, rightMargin=24,
                            topMargin=24, bottomMargin=24)
    doc.build(story)


def build(root):
    root = Path(root)
    reference = read(root / 'reference.json')
    output = ROOT / 'output/pdf'
    produced = []
    for stage, title in [('chunk-study', 'Разбиение и склейка: фиксированный левый канал'),
                         ('develop', 'Полезность каналов: разработочная выборка R001-R008')]:
        source = root / (stage + '.json')
        if not source.exists():
            continue
        result = read(source)
        cases = [c for c in result['cases'] if c.get('status', 'ok') == 'ok']
        if not cases:
            continue
        # Keep every successful experiment; width is deliberately unconstrained.
        cases.sort(key=lambda c: c['score']['total']['wer'], reverse=True)
        columns = [f"{c['model']} {str(c.get('layout', c.get('policy'))).removeprefix(c['model'] + ' ')}"
                   + (f" t{c['threshold']:g}" if 'threshold' in c else '') for c in cases]
        rows = []
        for name, card in {name: card for g in reference['groups'] for name, card in g['cards'].items()}.items():
            a, b = card['window']
            rows.append({'label': f'{name}\n{a:.2f}-{b:.2f} с', 'reference': card['text'],
                         'hypotheses': {label: case['score']['cards'][name]['hypothesis']
                                        for label, case in zip(columns, cases)},
                         'scores': {label: case['score']['cards'][name] for label, case in zip(columns, cases)}})
        appendix = [f"{label}: WER {case['score']['total']['wer']:.2%}; "
                    f"CER {case['score']['total']['cer']:.2%}; "
                    f"ошибки {case['score']['total']['errors']}/{case['score']['total']['words']}; "
                    f"S/D/I {case['score']['total']['S']}/{case['score']['total']['D']}/{case['score']['total']['I']}; "
                    f"числа/отрицания {case['score']['total']['number_errors']}/{case['score']['total']['negation_errors']}; "
                    + (str(case['channel_quality']) if 'channel_quality' in case else '')
                    for label, case in zip(columns, cases)]
        appendix += [f"FAILED {c['model']} {c.get('layout', '')}: {c.get('error', '')}"
                     for c in result['cases'] if c.get('status') == 'error']
        if stage == 'chunk-study' and (root / 'chunk-frozen.json').exists():
            for model, layout in read(root / 'chunk-frozen.json')['models'].items():
                for shift in layout['shifts']:
                    appendix.append(f"{model} finalist {layout['layout']}, shift {shift['shift']:+d}s: "
                                    + (f"WER {shift['score']['total']['wer']:.2%}" if shift['score'] else shift['status']))
        path = output / ('channel-research-' + stage + '.pdf')
        comparison(path, title, columns, rows, appendix,
                   ('Серия завершена. ' if result.get('complete') else 'Промежуточный отчёт: серия ещё не завершена. ') +
                   ('Склейка: ' + ', '.join(sorted({c['merger'] for c in cases if 'merger' in c})) + '. '
                    if stage == 'chunk-study' else '') +
                   'Это разработочная оценка, не независимая проверка. Общий WER считается по трём '
                   'непересекающимся группам; карточки R001-R008 перекрываются. Жирным выделены все '
                   'лучшие результаты строки; цвет соответствует WER. Сетка и результаты сохраняются в JSON по мере выполнения.')
        produced.append(str(path))
    assistance = read(root / 'reference-assistance.json') if (root / 'reference-assistance.json').exists() else {}
    validation = root / 'validation.json'
    if validation.exists():
        for model, result in read(validation)['models'].items():
            columns = sorted(result['wer'], key=result['wer'].get, reverse=True)
            rows = [{**f, 'label': f"{f['id']}\n{f['window'][0]:.2f}-{f['window'][1]:.2f} с"} for f in result['fragments']]
            path = output / ('channel-research-test-' + model + '.pdf')
            comparison(path, ('Проверка на новых фрагментах: ' if assistance else 'Независимая проверка: ') + model, columns, rows,
                       [f"{name}: WER {wer:.2%}; " + str(result.get('metrics', {}).get(name, {}))
                        for name, wer in result['wer'].items()] +
                       ['Ранжирование/oracle: ' + str(result.get('channel_quality', {})),
                        'ASR+alignment seconds (cached/shared): ' + str(result.get('seconds'))],
                       ('12 новых фрагментов; эталон исправлен вручную из черновиков GigaAM CTC combination. '
                        'Разметка не слепая, возможен сдвиг оценки в пользу CTC. '
                        if assistance else '12 независимых фрагментов. ') +
                       'Профиль выбран без подбора по этим фрагментам. '
                       f"Допуск selector={result['selector_passed']}; left={result['left_passed']}.")
            produced.append(str(path))
    write(root / 'reports.json', produced)
    return produced
