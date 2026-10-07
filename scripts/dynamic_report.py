"""Dynamic-channel experiment explanations and weight plots for the PDF appendix."""

import json
import os
from pathlib import Path

from reportlab.platypus import Image, PageBreak, Spacer


def appendix_items(run, cases, p):
    if json.loads((run / 'metadata.json').read_text()).get('series') == 'channel-curve-full-v1':
        return curve_appendix(run, cases, p)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/lecture-recognition-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metadata = json.loads((run / 'metadata.json').read_text())
    by_id = {c['id']: c for c in cases}
    items = [Spacer(1, 10), p('Динамические каналы: методика и диагностика', 'title'),
             p('Energy - средняя энергия; SNR-proxy - приближение по десятому процентилю энергии '
               'предшествующих 20 с; EV-normalized - дисперсия кубического корня энергии в 20 mel-полосах, '
               'делённая на квадрат среднего. Это адаптации критериев к плавному смешиванию, '
               'а не точное воспроизведение статей.', 'small'),
             p('Fast: окно 1 с, сглаживание 0,3 с. Slow: окно 3 с, сглаживание 1 с. '
               'Обновление 100 мс; начальный вес 0,5; при тишине веса сохраняются. '
               'Одна смесь для всего префикса, без коррекции задержки, усиления или дообучения. '
               'Все варианты используют исторические чанки, общее аудио выравнивания и актуальную склейку.', 'small'),
             p('Зеркальная проверка меняет каналы местами и сравнивает смесь и дополнительные веса. '
               'Она не заменяет проверку реальных участков, где лучше правый канал. '
               'Повторы ASR и сдвиги границ в этой серии не запланированы.', 'small')]
    for case in cases:
        if case['stage'] == 'dynamic' and case['status'] == 'ok':
            parent = by_id[metadata['parent_ids'][case['config']['model']]]
            delta = case['score']['total']['wer'] - parent['score']['total']['wer']
            worse = ', '.join(k for k, v in case['score']['cards'].items()
                              if v['wer'] > parent['score']['cards'][k]['wer']) or 'нет'
            items.append(p(f"{case['label']}: против {parent['label']}, общий WER {delta * 100:+.2f} п.п.; "
                           f"ухудшения карточек: {worse}.", 'small'))
    sources = [
        ('Wolf, Nadeu (2010)', 'https://www.isca-archive.org/interspeech_2010/wolf10_interspeech.pdf'),
        ('Himawan et al. (2015)', 'https://publications.idiap.ch/attachments/reports/2015/Himawan_Idiap-RR-30-2015.pdf'),
        ('CHiME-7 baseline', 'https://www.chimechallenge.org/challenges/chime7/task1/baseline')]
    for title, url in sources:
        items.append(p(title + ': ' + url, 'small'))
    plots = Path('tmp/pdfs/dynamic-channels')
    plots.mkdir(parents=True, exist_ok=True)
    reference = json.loads((run / 'reference.json').read_text())
    for path in sorted((run / 'audio').glob('dynamic_*.json')):
        d = json.loads(path.read_text())
        items += [PageBreak(), p(f"Пропорции: {d['method']} / {d['speed']}", 'title'),
                  p(f"Подготовка и зеркальная проверка: {d['preparation_seconds']:.2f} с. "
                    f"Максимальная разница смеси после перестановки: {d['mirror_max_sample_error']:.3g}; "
                    f"подавление более 3 dB: {len(d['suppressed_intervals'])} интервалов по 100 мс. "
                    f"Полный аудит: {path.resolve()}", 'small')]
        fig, axes = plt.subplots(3, 1, figsize=(10.8, 5.6), constrained_layout=True)
        for ax, group in zip(axes, reference['groups']):
            ax.plot(d['times'], d['weights_left'], label='Left channel weight', linewidth=1)
            ax.set_xlim(*group['window'])
            ax.set_ylim(-.02, 1.02)
            ax.set_ylabel('Left weight')
            ax.set_title(group['name'])
            ax.grid(alpha=.3)
        axes[-1].set_xlabel('Time, seconds')
        target = plots / (path.stem + '.png')
        fig.savefig(target, dpi=160)
        plt.close(fig)
        items.append(Image(str(target), width=760, height=394))
    return items


def curve_appendix(run, cases, p):
    """Describe this curve, not the unrelated first dynamic-channel grid."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/lecture-recognition-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    metadata = json.loads((run / 'metadata.json').read_text())
    by_id = {c['id']: c for c in cases}
    items = [PageBreak(), p('Energy fast t52/k32: методика и изменения', 'title'),
             p('Средняя энергия: окно 1 с, сглаживание 0,3 с, обновление 100 мс. '
               'Симметричная кривая применяется после сглаживания и интерполяции весов на 16 кГц. '
               't=0,52, k=32; a=max(w,1-w), u=max(a-t,0)/(1-t), '
               "a'=1-(1-a)*exp(-k*u*u). Ниже порога веса не меняются.", 'small'),
             p('Каждая модель обработала все 22 исторических чанка префикса 0-345,92 с. '
               'Два пилотных чанка переиспользованы после проверки PCM, декодера и границ. '
               'Соседний контекст полностью распознан заново. Пять контролей пересобраны '
               'из сохранённых выравниваний с тем же текущим алгоритмом склейки.', 'small'),
             p('Настройка выбрана по R005. Перенос выигрыша на остальные карточки проверяется '
               'на той же лекции; это не независимый тест. Дообучение, повторы и сдвиги границ '
               'в этой серии не выполнялись.', 'small')]
    for case in cases:
        if case['stage'] != 'dynamic':
            continue
        parent = by_id[metadata['parent_ids'][case['config']['model']]]
        old, new = parent['score']['total'], case['score']['total']
        items += [Spacer(1, 10), p(case['label'], 'best'),
                  p(f"Относительно {parent['label']}: общий WER {old['wer']:.2%} -> {new['wer']:.2%}; "
                    f"ошибки {old['errors']} -> {new['errors']} из {new['words']} слов.", 'small')]
        for name, card in sorted(case['score']['cards'].items()):
            previous = parent['score']['cards'][name]
            items.append(p(f"{name}: ошибки {previous['errors']} -> {card['errors']}; "
                           f"WER {(card['wer'] - previous['wer']) * 100:+.2f} п.п.", 'small'))
        focus = json.loads((run / 'cases' / case['id'] / 'focus-audit.json').read_text())
        for name in ('R004', 'R005'):
            items += [Spacer(1, 5), p(f'{name}: итоговая транскрипция', 'best'),
                      p(focus[name]['score']['hypothesis'], 'small'),
                      p('Сырые ответы до склейки:', 'small')]
            for transcript in focus[name]['raw_transcripts']:
                chunk = transcript['chunk']
                items.append(p(f"{chunk['start']:.2f}-{chunk['end']:.2f} с: {transcript['text']}", 'small'))
        check = json.loads((run / 'cases' / case['id'] / 'pilot-r005-check.json').read_text())
        items.append(p('Оценка и текст R005 совпадают с пилотом: ' + ('да.' if check['matches'] else 'нет.'), 'small'))
    rescored = json.loads((run / 'control-rescore-audit.json').read_text())
    items += [Spacer(1, 10), p('Пересчёт контролей', 'best')]
    for row in rescored:
        changed = ', '.join(name for name, score in row['after']['cards'].items()
                            if score != row['before']['cards'][name]) or 'нет'
        items.append(p(f"{row['label']}: изменения карточек относительно сохранённых оценок: {changed}.", 'small'))
    reference = json.loads((run / 'reference.json').read_text())
    diagnostic = json.loads((run / 'audio/dynamic_energy_fast_t52_k32.json').read_text())
    fig, axes = plt.subplots(3, 1, figsize=(10.8, 5.6), constrained_layout=True)
    for ax, group in zip(axes, reference['groups']):
        ax.plot(diagnostic['times'], diagnostic['weights_before'], label='Before curve', linewidth=.8)
        ax.plot(diagnostic['times'], diagnostic['weights_after'], label='After t52/k32', linewidth=1)
        ax.set_xlim(*group['window'])
        ax.set_ylim(-.02, 1.02)
        ax.set_ylabel('Left weight')
        ax.set_title(group['name'])
        ax.grid(alpha=.3)
        ax.legend(loc='best')
    axes[-1].set_xlabel('Time, seconds')
    plots = run / 'report-plots'
    plots.mkdir(exist_ok=True)
    target = plots / 'energy-fast-t52-k32.png'
    fig.savefig(target, dpi=160)
    plt.close(fig)
    items += [PageBreak(), p('Доля левого канала до и после кривой', 'title'),
              Image(str(target), width=760, height=394)]
    return items
