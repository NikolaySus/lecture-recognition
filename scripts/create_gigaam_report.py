"""Render a variable-width GigaAM comparison plus A4 experiment audit pages.

python3 scripts/create_gigaam_report.py --run .lecture-cache/gigaam-tuning/RUN
Use --baseline-preview to report rescored historical baselines without new ASR.
"""

import argparse
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path
from xml.sax.saxutils import escape

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import LongTable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from create_asr_comparison_pdf import INK, Legend, timestamp, wer_color  # noqa: E402

from lecture_recognition.evaluation import score_case  # noqa: E402
from lecture_recognition.gigaam_tuning import BASE_IDS, ranking  # noqa: E402
from lecture_recognition.model_benchmark import extra_metrics  # noqa: E402


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def report_config():
    path = ROOT/'experiments/gigaam/report-config.json'
    return load(path) if path.exists() else {'always_include': []}


def pareto_dominators(cases):
    """Compare all main configurations across models; equal vectors survive."""
    names = {f'R{i:03d}' for i in range(1, 9)}
    pool = [c for c in cases if c['status'] == 'ok' and c['stage'] != 'validation'
            and set(c['score']['cards']) == names]
    vectors = {c['id']: tuple(Fraction(c['score']['cards'][k]['errors'], c['score']['cards'][k]['words'])
                             for k in sorted(names)) for c in pool}
    return {c['id']: [other['id'] for other in pool
                     if all(a <= b for a, b in zip(vectors[other['id']], vectors[c['id']]))
                     and any(a < b for a, b in zip(vectors[other['id']], vectors[c['id']]))]
            for c in pool}


def collect(run, preview=False):
    metadata = load(run/'metadata.json')
    reference = load(run/'reference.json')
    if preview:
        source = Path(metadata['source'])
        cases = []
        for model, key in BASE_IDS.items():
            old = load(source/'cases'/key/'case.json')
            words = load(source/'cases'/key/'alignment/result.json')['words']
            cases.append({**old, 'label': model+' original', 'stage': 'baseline',
                          'score': extra_metrics(score_case(words, reference, 2)),
                          'legacy_score': extra_metrics(score_case(words, reference, 1))})
    else:
        cases = [load(p) for p in sorted((run/'cases').glob('*/case.json'))]
    bases = {c['config']['model']: c for c in cases if c['stage'] == 'baseline' and c['status'] == 'ok'}
    if set(bases) != set(BASE_IDS):
        raise ValueError('Both successful baselines are required; use --baseline-preview for historical data.')
    dominators = pareto_dominators(cases)
    pinned = set(report_config()['always_include']) if not preview else set()
    selected = [bases[m] for m in BASE_IDS]
    selected += sorted([c for c in cases if c['stage'] not in ('baseline', 'validation')
                        and c['id'] in dominators and (metadata.get('report_selection') == 'all'
                        or not dominators[c['id']] or c['label'] in pinned)], key=ranking)
    selected.sort(key=lambda c: (-c['score']['total']['wer'], c['label'], c['id']))
    rows = []
    for group in reference['groups']:
        for name, card in group['cards'].items():
            cells = []
            for case in selected:
                score = case['score']['cards'][name]
                baseline = bases[case['config']['model']]['score']['cards'][name]
                cells.append({'model': case['label'], 'case_id': case['id'], 'text': score['hypothesis'],
                              'wer': score['wer'], 'errors': score['errors'], 'words': score['words'],
                              'delta': score['wer'] - baseline['wer']})
            best = min(Fraction(c['errors'], c['words']) for c in cells)
            for cell in cells:
                cell['best'] = Fraction(cell['errors'], cell['words']) == best
                cell['background'] = wer_color(cell['wer']).hexval()
            rows.append({'id': name, 'window': card['window'], 'reference': card['text'], 'models': cells})
    return sorted(rows, key=lambda r: r['id']), selected, cases, bases


def build(run, output, preview=False):
    rows, selected, cases, bases = collect(run, preview)
    metadata = load(run/'metadata.json')
    dynamic = metadata.get('report_selection') == 'all'
    full_curve = metadata.get('series') == 'channel-curve-full-v1'
    if full_curve and (len(cases) != 7 or any(c['status'] != 'ok' for c in cases)):
        raise ValueError('Full curve PDF requires seven successful configurations')
    dominators = pareto_dominators(cases)
    by_id = {c['id']: c for c in cases}
    fonts = Path('/usr/share/fonts/truetype/dejavu')
    for name, file in [('DejaVu', 'DejaVuSans.ttf'), ('DejaVuBold', 'DejaVuSans-Bold.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(fonts/file)))
    styles = {name: ParagraphStyle(name, fontName='DejaVuBold' if name in ('best', 'header', 'title') else 'DejaVu',
                                   fontSize=size, leading=leading, textColor=colors.white if name == 'header' else INK)
              for name, size, leading in [('body', 10, 13), ('best', 10, 13), ('small', 8, 11),
                                          ('header', 10, 13), ('title', 18, 23)]}
    def p(text, style='body'):
        return Paragraph(escape(str(text)), styles[style])
    headings = ['ID', 'Время'] + [c['label'].replace('gigaam-ctc', 'GigaAM CTC').replace('gigaam-rnnt', 'GigaAM RNNT').replace('original', 'исходная') for c in selected] + ['Ground truth']
    widths = [18*mm, 40*mm] + [70*mm]*(len(selected)+1)
    table_rows = [[p(h, 'header') for h in headings]]
    commands = [('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#21485A')),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                ('TOPPADDING', (0, 0), (-1, -1), 8), ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), .35, colors.HexColor('#C8D1D7'))]
    for i, row in enumerate(rows, 1):
        values = [p(row['id'], 'best'), p(' - '.join(timestamp(t) for t in row['window']), 'small')]
        for col, cell in enumerate(row['models'], 2):
            change = f"{cell['delta']*100:+.2f} п.п."
            mark = ' / ХУЖЕ БАЗЫ' if cell['delta'] > 0 else ''
            values.append([p(f"WER {cell['wer']:.2%} / {change}{mark}", 'small'), Spacer(1, 5),
                           p(cell['text'], 'best' if cell['best'] else 'body')])
            commands.append(('BACKGROUND', (col, i), (col, i), wer_color(cell['wer'])))
        values.append(p(row['reference']))
        table_rows.append(values)
        for col in (0, 1, len(headings)-1):
            commands.append(('BACKGROUND', (col, i), (col, i), colors.HexColor('#F2F5F7')))
    table = Table(table_rows, colWidths=widths, style=TableStyle(commands))
    complete = (len(cases) == metadata['expected_cases'] if dynamic else (run/'selection.json').exists()) \
        and all(c['status'] == 'ok' for c in cases)
    title = 'GigaAM: исходные модели, исправленная оценка' if preview else 'GigaAM: эксперименты без дообучения'
    if dynamic:
        title = 'GigaAM: динамические пропорции каналов'
    if full_curve:
        title = 'GigaAM: Energy fast t52/k32, проверка R001-R008'
    status = ('Предварительная таблица: сохранённые транскрипции, новые эксперименты не выполнены.' if preview
              else 'Серия завершена.' if complete else 'Неполная серия: результаты и пропуски перечислены в приложении.')
    story = [p(title, 'title'), Spacer(1, 6), p(status, 'small'), Spacer(1, 8), Legend(sum(widths)),
             Spacer(1, 8), table, Spacer(1, 10),
             p('WER v2: числительные и их падежные формы нормализованы; «тире» между числами равно знаку диапазона. '
               'Исходные тексты моделей сохранены. Отрицания, неверные значения чисел и «или» не удаляются.', 'small'),
             p(('Карточки выделены по текстовому сопоставлению объединённых участков с эталоном; '
                'временные метки обозначают исходные интервалы. Вставки на границе относятся к слову справа, '
                'в конце участка — к последнему слову. '
                if selected[0]['score'].get('card_method') == 'group-alignment-v1' else
                'Окна оценки фиксированы; пограничные слова зависят от временного выравнивания. ') +
               'Карточки пересекаются: их ошибки нельзя суммировать. R001-R008 не являются независимым тестом.', 'small'),
             p(('Включены все успешные эксперименты динамических каналов и пять фиксированных сравнений, '
                'без исключения по WER или Парето. Колонки упорядочены по убыванию общего WER.' if dynamic else
                'Отбор по Парето среди основных конфигураций обеих моделей: вариант исключён, если другой имеет '
               'WER не выше во всех R001-R008 и ниже хотя бы в одной карточке. Обе базы сохранены; '
               'равные векторы WER не исключаются. Проверки устойчивости не участвуют в отборе. '
               'Колонки упорядочены слева направо по убыванию общего WER. ' +
               ('По запросу дополнительно включены: ' + ', '.join(report_config()['always_include']) + '.'
                if not preview and report_config()['always_include'] else '')), 'small'),
             p(f'Серия: {run.name}. Эталон: {load(run/"metadata.json")["review_sha256"][:16]}.', 'small')]
    available = sum(widths)
    height = sum(item.wrap(available, 100000)[1] for item in story)
    page = (available+24*mm+12, height+24*mm+24)
    if max(page) > 14400:
        raise ValueError('PDF exceeds the 200-inch compatibility limit; split columns explicitly.')
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = ROOT/'tmp/pdfs/gigaam'
    tmp.mkdir(parents=True, exist_ok=True)
    main_pdf, appendix = tmp/'table.pdf', tmp/'appendix.pdf'
    def footer(canvas, doc):
        canvas.setFont('DejaVu', 7)
        canvas.setFillColor(INK)
        number = doc.page if canvas._pagesize == page else doc.page + 1
        canvas.drawString(12*mm, 5*mm, f'R001-R008 | GigaAM | Страница {number}')

    SimpleDocTemplate(str(main_pdf), pagesize=page, leftMargin=12*mm, rightMargin=12*mm,
                      topMargin=12*mm, bottomMargin=12*mm, title=title).build(story, onFirstPage=footer, onLaterPages=footer)
    if len(PdfReader(main_pdf).pages) != 1:
        raise ValueError('Comparison table unexpectedly split across pages')
    audit = [p('Параметры и результаты серии', 'title'), Spacer(1, 10), p(status), Spacer(1, 8)]
    if dynamic:
        execution = load(run/'execution.json')
        audit += [p(f"Запланировано {metadata['expected_cases']} конфигураций: пять импортированных сравнений "
                    f"и {metadata.get('new_cases', 12)} новых экспериментов. Сейчас успешны {len(selected)}. "
                    f"Статус запуска: {execution['status']}; причина: {execution.get('reason', '-')}. "
                    "Невыполненные варианты перечислены ниже; их результатов в таблице нет.", 'small'),
                  Spacer(1, 8)]
    if complete and not preview and not dynamic:
        selection = load(run/'selection.json')
        by_id = {c['id']: c for c in cases}
        audit += [p(f"Успешно завершены {len(cases)} прогонов. В основной таблице {len(selected)} "
                    "конфигурации, включая две исходные модели; проверки устойчивости приведены отдельно."),
                  Spacer(1, 8)]
        for model, item in selection.items():
            baseline, candidate = by_id[item['baseline']], by_id[item['candidate']]
            old, new = baseline['score']['total'], candidate['score']['total']
            worse = ', '.join(k for k, v in candidate['score']['cards'].items()
                              if v['wer'] > baseline['score']['cards'][k]['wer']) or 'нет'
            audit += [p(f"{candidate['label']}: WER {old['wer']:.2%} -> {new['wer']:.2%}; "
                        f"ошибок {old['errors']} -> {new['errors']} из {new['words']} слов. "
                        f"Ухудшения относительно своей базы: {worse}."), Spacer(1, 5)]
        requested = [c for c in selected if c['label'] in report_config()['always_include']
                     and c['id'] not in {item['candidate'] for item in selection.values()}]
        for case in requested:
            audit += [p(f"Дополнительно включён по запросу {case['label']}: "
                        f"WER {case['score']['total']['wer']:.2%}, "
                        f"ошибок {case['score']['total']['errors']} из {case['score']['total']['words']}. "
                        "Отдельные проверки повтора и сдвигов для этого варианта не выполнялись."), Spacer(1, 5)]
        if load(run/'metadata.json').get('postprocessing_source'):
            audit += [p('Результаты пересобраны из сохранённых выравниваний слов без повторного ASR. '
                        'Исправлены дубли и потери слов на стыках по сохранённым чанкам и опорным словам выравнивания. '
                        'Одинаковое правило применено ко всем конфигурациям, включая исходные модели. '
                        'Исходные результаты сохранены отдельно.'), Spacer(1, 5)]
        audit += [p('Выводы относятся к диагностической выборке R001-R008; '
                    'перенос на новые лекции ещё не проверен.'), Spacer(1, 10)]
    grid = [[p(s, 'header') for s in ['Вариант / параметры', 'Статус / отбор', 'WER v2 / v1', 'Числа / отрицания', 'ASR, с / VRAM, ГиБ']]]
    for c in sorted(cases, key=lambda c: (c['stage'] != 'baseline', ranking(c))):
        cfg = c['config']
        parameters = ', '.join(f'{k}={cfg[k]}' for k in ('maximum', 'context', 'audio_variant', 'precision',
                               'decoding', 'beam', 'bias_weight', 'shift', 'repeat', 'max_symbols') if k in cfg)
        if c['status'] != 'ok':
            status_text = c.get('error', c['status'])
            metrics, critical = '-', '-'
        else:
            status_text = ('Проверка устойчивости' if c['stage'] == 'validation' else 'Включён' if c in selected
                           else 'Исключён по Парето; доминирует: ' + by_id[dominators[c['id']][0]]['label']
                           if dominators.get(c['id']) else 'Неполная оценка')
            if c in selected and c['stage'] != 'baseline' and dominators.get(c['id']) and not dynamic:
                status_text = 'Включён по запросу; по Парето доминирует: ' + by_id[dominators[c['id']][0]]['label']
            s, old = c['score']['total'], c['legacy_score']['total']
            metrics = f"{s['wer']:.2%} / {old['wer']:.2%}"
            critical = f"{s['number_errors']} / {s['negation_errors']}"
        grid.append([p(c['label']+'\n'+parameters, 'small'), p(status_text, 'small'), p(metrics, 'small'),
                     p(critical, 'small'), p(' / '.join(f'{c[k]:.2f}' if c.get(k) is not None else '-' for k in ('asr_seconds', 'peak_vram_gib')), 'small')])
    audit.append(LongTable(grid, colWidths=[95*mm, 58*mm, 35*mm, 32*mm, 48*mm], repeatRows=1,
                          style=TableStyle([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#21485A')),
                                            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                                            ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey),
                                            ('TOPPADDING', (0, 0), (-1, -1), 5),
                                            ('BOTTOMPADDING', (0, 0), (-1, -1), 5)])))
    audit += [Spacer(1, 10), p('Общий WER рассчитан по трём объединённым участкам без двойного подсчёта. '
                              'WER v1 использует тот же эталон, включая числа словами. Время отдельного прогона '
                              'не является статистикой производительности.', 'small')]
    if dynamic:
        from dynamic_report import appendix_items
        audit += appendix_items(run, cases, p)
    if (run/'selection.json').exists() and not preview:
        selection = load(run/'selection.json')
        for model, item in selection.items():
            audit += [Spacer(1, 8), p(f"{model}: кандидат {item['candidate']}", 'best')]
            by_id = {c['id']: c for c in cases}
            for check in item['checks']:
                source, result = by_id[check['source']], by_id[check['id']]
                if result['status'] != 'ok':
                    text = f"{source['label']} / {check['tag']}: проверка не выполнена."
                else:
                    delta = result['score']['total']['wer'] - source['score']['total']['wer']
                    text = f"{source['label']} / {check['tag']}: изменение WER {delta*100:+.2f} п.п."
                audit.append(p(text, 'small'))
            if item['candidate'] not in item.get('repeat_equal', {}):
                audit.append(p('Для выбранного кандидата отдельный повтор ещё не выполнен.', 'small'))
            for source, equal in item.get('repeat_equal', {}).items():
                audit.append(p(f"Повтор {source}: " + ('полное совпадение оценки и текста.' if equal
                                                       else 'есть расхождение или ошибка исполнения.'), 'small'))
            for shift in item.get('shift_comparisons', []):
                if 'wer_delta' in shift:
                    worse = ', '.join(k for k, d in shift['card_deltas'].items() if d > 0) or 'нет'
                    audit.append(p(f"{shift['tag']}: кандидат против базы на соответствующем сдвиге: "
                                   f"{shift['wer_delta']*100:+.2f} п.п.; ухудшения: {worse}.", 'small'))
                else:
                    audit.append(p(f"{shift['tag']}: сравнение нового кандидата со сдвинутой базой ещё не выполнено.", 'small'))
        if (run/'decisions.json').exists():
            audit += [Spacer(1, 8), p('Условные пропуски', 'best')]
            for decision in load(run/'decisions.json'):
                audit.append(p(f"{decision['model']} / {decision['stage']}: {decision['reason']}", 'small'))
    SimpleDocTemplate(str(appendix), pagesize=landscape(A4), leftMargin=10*mm, rightMargin=10*mm,
                      topMargin=10*mm, bottomMargin=10*mm).build(audit, onFirstPage=footer, onLaterPages=footer)
    writer = PdfWriter()
    writer.append(main_pdf)
    writer.append(appendix)
    temporary = output.with_suffix('.tmp.pdf')
    with temporary.open('wb') as stream:
        writer.write(stream)
    extracted = ''.join(''.join(p.extract_text().split()) for p in PdfReader(temporary).pages)
    for row in rows:
        for text in [row['reference']]+[c['text'] for c in row['models']]:
            if ''.join(text.split()) not in extracted:
                raise ValueError(f"Missing text: {row['id']}")
        assert all(timestamp(t) in extracted for t in row['window'])
    temporary.replace(output)
    sources = [run/'metadata.json', run/'reference.json', Path(__file__).resolve(), ROOT/'scripts/create_asr_comparison_pdf.py']
    sources.extend(sorted((run/'cases').glob('*/case.json')))
    if dynamic:
        sources.extend(sorted((run/'audio').glob('dynamic_*.json')))
        sources.append(ROOT/'scripts/dynamic_report.py')
    sources.extend(run/name for name in ('selection.json', 'execution.json', 'decisions.json', 'remerge-audit.json') if (run/name).exists())
    if full_curve:
        sources.append(run/'control-rescore-audit.json')
        sources.extend(sorted((run/'cases').glob('*/focus-audit.json')))
        sources.extend(sorted((run/'cases').glob('*/pilot-r005-check.json')))
    config_path = ROOT/'experiments/gigaam/report-config.json'
    if config_path.exists():
        sources.append(config_path)
    if preview:
        source = Path(load(run/'metadata.json')['source'])
        sources.extend(source/'cases'/key/'alignment/result.json' for key in BASE_IDS.values())
    manifest = {'run': str(run), 'preview': preview, 'complete': complete and not preview, 'rows': rows,
                'case_ids': [c['id'] for c in selected], 'all_cases': cases,
                'selection_rule': 'all_dynamic_and_five_fixed' if dynamic else
                    'pareto_weak_all_strict_any_across_models_keep_baselines_and_requested_variants',
                'requested_variants': [] if dynamic or preview else report_config()['always_include'],
                'dominators': dominators,
                'sources': {str(s): hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    output.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Created {output}; {len(selected)} model columns; all {len(rows)*(len(selected)+1)} texts verified.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--baseline-preview', action='store_true')
    args = parser.parse_args()
    pointer = ROOT/'.lecture-cache/gigaam-tuning/latest-report.json'
    if not pointer.exists() or args.baseline_preview:
        pointer = ROOT/'.lecture-cache/gigaam-tuning/latest.json'
    run = args.run or Path(load(pointer)['run'])
    name = 'r001-r008-gigaam-baselines.pdf' if args.baseline_preview else 'r001-r008-gigaam-experiments.pdf'
    build(run, args.output or ROOT/'output/pdf'/name, args.baseline_preview)


if __name__ == '__main__':
    main()
