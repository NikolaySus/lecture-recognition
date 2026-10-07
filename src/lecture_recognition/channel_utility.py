"""Reference-free layouts, overlap assembly, confidence and channel selection."""

import math
from difflib import SequenceMatcher

import numpy as np

from .evaluation import lexical
from .timeline import Chunk, merge_words, quiet_cut

MERGERS = ('current', 'core', 'overlap', 'contextual')
THRESHOLDS = {'gibbs': (0., .02, .05), 'tsallis': (0., .02, .05),
              'snr': (0., 3., 6.), 'c50': (0., 3., 6.), 'margin': (0., .01, .05)}


def overlap_duration(window, regions):
    a, b = window
    spans = sorted((max(a, x), min(b, y)) for x, y in regions if x < b and y > a)
    total, end = 0., a
    for x, y in spans:
        total += max(0., y - max(end, x))
        end = max(end, y)
    return total


def validation_windows(duration, regions, excluded, start=345.92, count=12, length=60.):
    edges = np.linspace(start, duration, count + 1)
    result = []
    for index, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        candidates = np.arange(a + 2, b - length - 2 + 1e-6, .1)
        candidates = sorted(candidates, key=lambda t: (abs(t + length / 2 - (a + b) / 2), t))
        window = next(([round(float(t), 2), round(float(t + length), 2)] for t in candidates
                       if overlap_duration((t, t + length), regions) >= .7 * length
                       and overlap_duration((t - 2, t + length + 2), excluded) == 0), None)
        if window is None:
            raise ValueError(f'No eligible {length:g}s speech window in bin {index + 1}')
        result.append({'id': f'S{index + 1:03d}', 'window': window,
                       'input_window': [window[0] - 2, window[1] + 2], 'split': 'test'})
    return result


def make_layout(audio, duration, maximum, context, mode, regions, shift=0.):
    if mode not in ('regular', 'vad') or maximum <= 2 * context or context < 0:
        raise ValueError('Invalid layout parameters')
    core = maximum - 2 * context
    cuts, cursor = [0.], 0.
    while duration - cursor > core + 1e-6:
        upper = cursor + core
        lower = max(cursor + min(1., core / 2), upper - 4)
        cut = upper
        if mode == 'vad':
            # VAD complement, bounded to the last four seconds of a feasible core.
            candidates, end = [], lower
            for a, b in sorted(regions):
                if b <= lower or a >= upper:
                    continue
                if a > end:
                    candidates.append((end, min(a, upper)))
                end = max(end, min(b, upper))
            if end < upper:
                candidates.append((end, upper))
            if candidates:
                a, b = candidates[-1]
                cut = (a + b) / 2
            else:
                cut = quiet_cut(audio, lower, upper)
        cuts.append(float(cut))
        cursor = cut
    cuts.append(float(duration))
    if shift:
        if abs(shift) >= core:
            raise ValueError('Boundary shift must be smaller than the maximum core')
        # Translate the entire grid, including its endpoints. Short edge cores
        # preserve coverage and the input-length limit without undoing the shift
        # at every saturated regular core.
        cuts = sorted({0., float(duration), *(c + shift for c in cuts if 0 < c + shift < duration)})
    return [Chunk(max(0., a - context), min(duration, b + context), a, b).dict()
            for a, b in zip(cuts[:-1], cuts[1:])]


def shift_layout(chunks, duration, maximum, context, shift):
    """Translate historical core boundaries; retain the full bounded timeline."""
    core = maximum - 2 * context
    if abs(shift) >= core:
        raise ValueError('Boundary shift must be smaller than the maximum core')
    original = {0., duration, *(c[k] for c in chunks for k in ('core_start', 'core_end'))}
    cuts = sorted({0., float(duration), *(c + shift for c in original if 0 < c + shift < duration)})
    result = [Chunk(max(0., a - context), min(duration, b + context), a, b).dict()
              for a, b in zip(cuts[:-1], cuts[1:])]
    if any(c['end'] - c['start'] > maximum + 1e-6 for c in result):
        raise ValueError('Historical cores exceed requested input limit')
    return result


def global_units(aligned, atoms=False):
    groups = []
    for index, item in enumerate(aligned):
        c = item['chunk']
        group = []
        for word in item['words']:
            start = c['start'] + word['start_time']
            end = c['start'] + word['end_time']
            if not np.isfinite([start, end]).all() or end <= start:
                raise ValueError('Invalid alignment timestamps')
            parts = lexical(word['text']) if atoms else [word['text']]
            # Compound aligner units have no native sub-word timestamps. These
            # approximate subdivisions are explicitly recorded in overlap audit.
            sizes = np.array([len(s) for s in parts], dtype=float)
            bounds = start + (end - start) * np.r_[0, np.cumsum(sizes)] / max(1, sizes.sum())
            for j, text in enumerate(parts):
                a, b = float(bounds[j]), float(bounds[j + 1])
                midpoint = (a + b) / 2
                group.append({'text': text, 'start': a, 'end': b,
                              'owner': c['core_start'] <= midpoint < c['core_end'],
                              'margin': min(midpoint - c['start'], c['end'] - midpoint),
                              'source_chunk': index, 'compound': len(parts) > 1})
        groups.append(group)
    return groups


def assemble(aligned, method='current', audit=None):
    audit = audit if audit is not None else []
    if method == 'current':
        return [{k: v for k, v in w.items() if k != 'chunks'} for w in merge_words(aligned, seam_audit=audit)]
    if method == 'contextual':
        from .seam_merge import merge_overlap_words
        return [{k: v for k, v in w.items() if k != 'chunks'}
                for w in merge_overlap_words(aligned, seam_audit=audit)]
    if method not in MERGERS:
        raise ValueError(method)
    groups = global_units(aligned, atoms=method == 'overlap')
    if method == 'overlap':
        for index in range(1, len(groups)):
            a = max(aligned[index - 1]['chunk']['start'], aligned[index]['chunk']['start'])
            b = min(aligned[index - 1]['chunk']['end'], aligned[index]['chunk']['end'])
            left = [w for w in groups[index - 1] if w['end'] > a and w['start'] < b and not w.get('removed')]
            right = [w for w in groups[index] if w['end'] > a and w['start'] < b and not w.get('removed')]
            matcher = SequenceMatcher(None, [w['text'].casefold() for w in left],
                                      [w['text'].casefold() for w in right], autojunk=False)
            for block in matcher.get_matching_blocks():
                if not block.size:
                    continue
                pairs = list(zip(left[block.a:block.a + block.size], right[block.b:block.b + block.size]))
                # A lexical match alone must never erase a real repeated phrase.
                if any(min(x['end'], y['end']) <= max(x['start'], y['start']) for x, y in pairs):
                    continue
                for x, y in pairs:
                    keep, remove = (x, y) if x['margin'] >= y['margin'] else (y, x)
                    if remove.get('removed'):
                        continue
                    keep['owner'] |= remove['owner']
                    remove['removed'] = True
                    audit.append({'reason': 'matched_overlap_copy', 'chunks': [index - 1, index],
                                  'text': keep['text'], 'window': [a, b],
                                  'approximate_compound_times': bool(x['compound'] or y['compound'])})
    words = [w for group in groups for w in group if w['owner'] and not w.get('removed')]
    return sorted(words, key=lambda w: (w['start'], w['end'], w['source_chunk']))


def entropy_confidence(probabilities, kind='gibbs'):
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] < 2 or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError('Expected finite probability matrix')
    if not np.allclose(p.sum(axis=1), 1, atol=1e-4):
        raise ValueError('Probabilities are not normalized')
    v = p.shape[1]
    if kind == 'gibbs':
        entropy = -np.sum(p * np.log(np.maximum(p, 1e-300)), axis=1)
        maximum = math.log(v)
    elif kind == 'tsallis':
        alpha = 1 / 3
        entropy = (np.sum(p ** alpha, axis=1) - 1) / (1 - alpha)
        maximum = (v ** (1 - alpha) - 1) / (1 - alpha)
    else:
        raise ValueError(kind)
    return np.clip((np.exp(-entropy) - math.exp(-maximum)) / (-math.expm1(-maximum)), 0, 1)


def word_confidence(probabilities, tokens, token_words, blank, ctc=False):
    p = np.asarray(probabilities)
    if len(p) != len(tokens) or len(tokens) != len(token_words):
        raise ValueError('Token trace lengths differ')
    selected, previous = [], None
    for i, token in enumerate(tokens):
        if token == blank:
            previous = None
            continue
        if ctc and token == previous:
            selected[-1].append(i)
        else:
            selected.append([i])
        previous = token
    result = {}
    for kind in ('gibbs', 'tsallis'):
        conf = entropy_confidence(p, kind)
        words = {}
        for indices in selected:
            name = token_words[indices[0]]
            words.setdefault(name, []).append(float(np.min(conf[indices])))
        result[kind] = float(np.mean([min(values) for values in words.values()])) if words else None
    return result


def ctc_log_likelihood(log_probs, labels, blank):
    p = np.asarray(log_probs, dtype=np.float64)
    labels = list(labels)
    if p.ndim != 2 or not np.isfinite(p).all():
        raise ValueError('Expected finite log probabilities')
    if any(v == blank or v < 0 or v >= p.shape[1] for v in labels):
        raise ValueError('Invalid target labels')
    if len(labels) + sum(a == b for a, b in zip(labels, labels[1:])) > len(p):
        return float('-inf')
    target = [blank]
    for label in labels:
        target.extend((label, blank))
    state = np.full(len(target), -np.inf)
    state[0] = 0
    target = np.asarray(target)
    skip = np.r_[False, False, (target[2:] != blank) & (target[2:] != target[:-2])] if len(target) > 1 else np.array([False])
    for frame in p:
        one = np.r_[-np.inf, state[:-1]]
        two = np.r_[-np.inf, -np.inf, state[:-2]] if len(state) > 1 else np.array([-np.inf])
        state = np.logaddexp(np.logaddexp(state, one), np.where(skip, two, -np.inf)) + frame[target]
    return float(np.logaddexp.reduce(state[-2:])) if labels else float(state[0])


def likelihood_margin(left, right, hleft, hright, blank):
    if hleft is None or hright is None:
        return None
    if hleft == hright:
        return 0.
    values = [ctc_log_likelihood(left, hleft, blank), ctc_log_likelihood(left, hright, blank),
              ctc_log_likelihood(right, hright, blank), ctc_log_likelihood(right, hleft, blank)]
    if not np.isfinite(values).all():
        return None
    return (values[0] - values[1]) / len(left) - (values[2] - values[3]) / len(right)


def acoustic_statistics(times, speech, snr, c50, window):
    times, speech, snr, c50 = map(np.asarray, (times, speech, snr, c50))
    if (len({len(times), len(speech), len(snr), len(c50)}) != 1 or len(times) < 2
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError('Invalid acoustic trace')
    mask = (times >= window[0]) & (times < window[1]) & (speech >= .5)
    step = float(np.median(np.diff(times)))
    if mask.sum() * step < 1 - 1e-9:
        return {'snr': None, 'c50': None, 'reason': 'less_than_one_second_speech'}
    return {name: float(np.median(values[mask])) if np.isfinite(values[mask]).all() else None
            for name, values in [('snr', snr), ('c50', c50)]}


def select_channels(rows, policy, threshold=0., hybrid_thresholds=None):
    if policy not in {*THRESHOLDS, 'hybrid', 'left'} or not math.isfinite(threshold) or threshold < 0:
        raise ValueError('Invalid selector policy or threshold')
    current, decisions = 'left', []
    for row in rows:
        reason, votes = 'hold', []
        if policy == 'left':
            current, reason = 'left', 'fixed_left'
        elif row['left_text'] == row['right_text']:
            current, reason = 'left', 'identical_text'
        else:
            criteria = ('tsallis', 'c50', 'margin') if policy == 'hybrid' else (policy,)
            available = 0
            for criterion in criteria:
                value = row.get('margin') if criterion == 'margin' else None
                if criterion != 'margin':
                    lhs, rhs = row['left'].get(criterion), row['right'].get(criterion)
                    if lhs is not None and rhs is not None:
                        value = lhs - rhs
                if value is None or not math.isfinite(value):
                    continue
                available += 1
                limit = hybrid_thresholds[criterion] if policy == 'hybrid' else threshold
                if abs(value) > limit:
                    votes.append('left' if value > 0 else 'right')
            required = 2 if policy == 'hybrid' else 1
            if votes.count('left') >= required:
                current, reason = 'left', 'advantage'
            elif votes.count('right') >= required:
                current, reason = 'right', 'advantage'
            elif available == 0:
                current, reason = 'left', 'no_scores_fallback'
        decisions.append({'channel': current, 'reason': reason, 'votes': votes})
    return decisions


def aggregate_scores(scores):
    """Word-weighted totals; never average fragment WERs."""
    result = {key: sum(s[key] for s in scores) for key in
              ('words', 'errors', 'chars', 'char_errors', 'S', 'D', 'I')}
    result['wer'] = result['errors'] / max(1, result['words'])
    result['cer'] = result['char_errors'] / max(1, result['chars'])
    result['number_errors'] = sum(sum(e['ref'].isdigit() or e['hyp'].isdigit() for e in s['edits']) for s in scores)
    result['negation_errors'] = sum(sum(e['ref'] == 'не' or e['hyp'] == 'не'
               or e['ref'].startswith('нестационар') or e['hyp'].startswith('нестационар')
               for e in s['edits']) for s in scores)
    return result


def channel_quality_metrics(units):
    """Coarse reference-only oracle and channel ranking, outside inference.

    Each unit is a whole dev group or test fragment. A weighted vote of selected
    chunk cores predicts its channel; mixed text can beat this coarse oracle.
    """
    oracle_errors, ranked, correct = 0, 0, 0
    for unit in units:
        lhs, rhs = unit['scores']['new_left']['errors'], unit['scores']['new_right']['errors']
        oracle_errors += min(lhs, rhs)
        if lhs == rhs:
            continue
        ranked += 1
        weights = {'left': 0., 'right': 0.}
        a, b = unit['window']
        for chunk, decision in zip(unit['chunks'], unit['decisions']):
            weights[decision['channel']] += max(0., min(b, chunk['core_end']) - max(a, chunk['core_start']))
        predicted = 'left' if weights['left'] >= weights['right'] else 'right'
        correct += predicted == ('left' if lhs < rhs else 'right')
    selector_errors = sum(u['scores']['selector']['errors'] for u in units)
    return {'oracle_errors': oracle_errors, 'regret_errors': selector_errors - oracle_errors,
            'ranking_units_non_ties': ranked, 'ranking_correct': correct,
            'ranking_accuracy': correct / ranked if ranked else None,
            'oracle_scope': 'whole group/fragment L/R; mixed chunks may beat it',
            'switches': sum(sum(a['channel'] != b['channel'] for a, b in zip(u['decisions'], u['decisions'][1:]))
                            for u in units)}
