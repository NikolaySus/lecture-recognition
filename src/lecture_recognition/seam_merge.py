"""Conservative full-overlap lexical reconciliation above the legacy merger.

No references, new recognition or inferred sub-word timestamps are used.
Ambiguous repeated speech is retained and reported rather than erased.
"""
import re
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from statistics import median

from .evaluation import NUMBER_FORMS, NUMBERS, THOUSANDS
from .timeline import merge_words


def values(text):
    return tuple(m.group().casefold().replace('ё', 'е') for m in re.finditer(r'\w+', text))


@dataclass(frozen=True)
class Token:
    value: str
    key: tuple
    offset: int
    position: int
    start: float
    end: float


def raw_tokens(item, index):
    c, result = item['chunk'], []
    for word in item['words']:
        start = max(c['start'], c['start'] + word['start_time'])
        end = min(c['end'], c['start'] + word['end_time'])
        tokens = values(word['text'])
        key = (index, start, end, tokens)
        for offset, value in enumerate(tokens):
            result.append(Token(value, key, offset, len(result), start, end))
    return result


def coincides(a, b):
    overlap = min(a.end, b.end) - max(a.start, b.start)
    return overlap > .5 * min(a.end - a.start, b.end - b.start)


def protected(value):
    return (value.isdigit() or value in NUMBERS or value in NUMBER_FORMS or value in THOUSANDS
            or value == 'не' or value.startswith('нестационар'))


def tail_variant(a, b):
    function_words = {'про', 'для', 'при', 'под', 'над', 'без', 'что', 'это', 'так', 'как',
                      'его', 'ее', 'них', 'нас', 'вас', 'мой', 'наш', 'тут', 'там', 'был', 'были', 'свой', 'свои', 'все'}
    if protected(a.value) or protected(b.value) or a.value in function_words or a.value == b.value:
        return None
    prefix = 0
    for x, y in zip(a.value, b.value):
        if x != y:
            break
        prefix += 1
    if prefix >= 3 and len(a.value) <= 4 and b.value.startswith(a.value) and len(b.value) > len(a.value):
        return 'clipped_terminal_token'
    if prefix >= 4 and prefix / max(len(a.value), len(b.value)) >= .6:
        return 'terminal_form_disagreement'
    return None


def occurrences(sequence, phrase):
    tokens = [t.value for t in sequence]
    return sum(tuple(tokens[i:i + len(phrase)]) == phrase for i in range(len(tokens) - len(phrase) + 1))


def merge_overlap_words(aligned, *, seam_audit=None):
    audit = seam_audit if seam_audit is not None else []
    words = merge_words(aligned, seam_audit=audit)
    lookup = defaultdict(list)
    for index, word in enumerate(words):
        # Previously reconciled units have multiple sources. Do not dismantle
        # legacy recovery decisions or guess how a modified compound was split.
        if len(word['chunks']) == 1:
            chunk = next(iter(word['chunks']))
            lookup[(chunk, word['start'], word['end'], values(word['text']))].append(index)
    raw = [raw_tokens(item, i) for i, item in enumerate(aligned)]
    removals = defaultdict(set)
    def retained(token):
        indices = lookup.get(token.key, [])
        if len(indices) != 1 or token.offset in removals[indices[0]]:
            return None
        return indices[0]
    for i in range(len(aligned) - 1):
        a, b = aligned[i]['chunk'], aligned[i + 1]['chunk']
        lo, hi = max(a['start'], b['start']), min(a['end'], b['end'])
        if hi <= lo or abs(a['core_end'] - b['core_start']) > 1e-6:
            continue
        left = [t for t in raw[i] if t.end > lo and t.start < hi]
        right = [t for t in raw[i + 1] if t.end > lo and t.start < hi]
        matcher = SequenceMatcher(None, [t.value for t in left], [t.value for t in right], autojunk=False)
        for block in matcher.get_matching_blocks():
            if block.size < 2:
                continue
            pairs = list(zip(left[block.a:block.a + block.size], right[block.b:block.b + block.size]))
            phrase = tuple(x.value for x, _ in pairs)
            if len(set(phrase)) < 2 or occurrences(left, phrase) != 1 or occurrences(right, phrase) != 1:
                continue
            duplicates = [(x, y) for x, y in pairs if retained(x) is not None and retained(y) is not None]
            if not duplicates:
                continue
            anchored_units = {(x.key, y.key) for x, y in pairs if coincides(x, y)}
            all_timed = all(coincides(x, y) for x, y in duplicates)
            evidence = 'temporal_overlap' if all_timed else 'independent_context_anchors' if len(anchored_units) >= 2 else None
            extension = None
            li, ri = block.a + block.size, block.b + block.size
            if block.size >= 3 and li < len(left) and ri < len(right):
                x, y = left[li], right[ri]
                variant = tail_variant(x, y)
                terminal = x.position == len(raw[i]) - 1 and y.position < len(raw[i + 1]) - 1
                if variant and terminal and retained(x) is not None and retained(y) is not None:
                    extension = (x, y, variant)
                    duration = max(t.end for t, _ in pairs) - min(t.start for t, _ in pairs)
                    collapsed = len(x.key[-1]) > 1 and x.end - x.start <= .1 * max(duration, 1e-9)
                    compound = len(y.key[-1]) >= 3
                    if evidence is None and compound and (variant == 'clipped_terminal_token' or collapsed):
                        evidence = variant
            if evidence is None:
                audit.append({'reason': 'contextual_ambiguous_repeat', 'chunks': [i, i + 1],
                              'overlap': [lo, hi], 'phrase': ' '.join(phrase), 'action': 'retained',
                              'evidence': 'no independent temporal anchors or supported terminal fragment'})
                continue
            left_margin = median(words[retained(x)]['margin'] for x, _ in duplicates)
            right_margin = median(words[retained(y)]['margin'] for _, y in duplicates)
            keep_right = right_margin > left_margin
            # A clipped tail needs the complete right continuation, even if
            # the incorrect collapsed alignment happens to have a large margin.
            if extension and evidence in ('clipped_terminal_token', 'terminal_form_disagreement'):
                keep_right = True
            if extension:
                duplicates.append(extension[:2])
            deleted = []
            for x, y in duplicates:
                token = x if keep_right else y
                index = retained(token)
                if index is not None:
                    removals[index].add(token.offset)
                    deleted.append({'text': token.value, 'start': token.start, 'end': token.end})
            if deleted:
                audit.append({'reason': 'contextual_overlap_copy', 'chunks': [i, i + 1],
                              'overlap': [lo, hi], 'phrase': ' '.join(phrase), 'kept_chunk': i + 1 if keep_right else i,
                              'evidence': evidence, 'removed': deleted,
                              'tail_variant': extension[2] if extension else None})
    result = []
    for index, word in enumerate(words):
        cuts = removals.get(index, set())
        if not cuts:
            result.append(word)
            continue
        matches = list(re.finditer(r'\w+', word['text']))
        keep = [j for j in range(len(matches)) if j not in cuts]
        if not keep:
            continue
        # Preserve native unit bounds; no fabricated per-token timestamps.
        pieces, cursor = [], 0
        for j, match in enumerate(matches):
            if j in cuts:
                pieces.append(word['text'][cursor:match.start()])
                cursor = match.end()
        pieces.append(word['text'][cursor:])
        text = re.sub(r'\s+', ' ', ''.join(pieces)).strip()
        result.append({**word, 'text': text})
    return result
