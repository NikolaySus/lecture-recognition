"""Model-independent reference parsing, normalization and text scoring."""

import re
import unicodedata

from .experiments import distance

GROUPS = [("R001", "R002"), ("R003", "R004"), ("R005", "R006", "R007", "R008")]

_SMALL = (
    "ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать "
    "тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать"
).split()
NUMBERS = {s: n for n, s in enumerate(_SMALL)}
NUMBERS.update(
    dict(
        zip(
            "двадцать тридцать сорок пятьдесят шестьдесят семьдесят восемьдесят девяносто".split(),
            range(20, 100, 10),
        )
    )
)
NUMBERS.update(
    dict(
        zip(
            "сто двести триста четыреста пятьсот шестьсот семьсот восемьсот девятьсот".split(),
            range(100, 1000, 100),
        )
    )
)
NUMBERS.update({"одна": 1, "одно": 1, "две": 2})
THOUSANDS = {"тысяча", "тысячи", "тысяч", "тысячу"}

# Explicit cardinal forms only: ordinal adjectives and ordinary nouns stay lexical.
NUMBER_FORMS = {}
for _value, _forms in {
    0: "нуля нулю нулем нуле", 1: "одного одному одним одной одну одними",
    2: "двух двум двумя", 3: "трех трем тремя", 4: "четырех четырем четырьмя",
    5: "пяти пятью", 6: "шести шестью", 7: "семи семью", 8: "восьми восемью восьмью",
    9: "девяти девятью", 10: "десяти десятью", 11: "одиннадцати одиннадцатью",
    12: "двенадцати двенадцатью", 13: "тринадцати тринадцатью",
    14: "четырнадцати четырнадцатью", 15: "пятнадцати пятнадцатью",
    16: "шестнадцати шестнадцатью", 17: "семнадцати семнадцатью",
    18: "восемнадцати восемнадцатью", 19: "девятнадцати девятнадцатью",
    20: "двадцати двадцатью", 30: "тридцати тридцатью", 40: "сорока",
    50: "пятидесяти пятьюдесятью", 60: "шестидесяти шестьюдесятью",
    70: "семидесяти семьюдесятью", 80: "восьмидесяти восемьюдесятью",
    90: "девяноста", 100: "ста", 200: "двухсот двумстам двумястами двухстах",
    300: "трехсот тремстам тремястами трехстах",
    400: "четырехсот четыремстам четырьмястами четырехстах",
    500: "пятисот пятистам пятьюстами пятистах", 600: "шестисот шестистам шестьюстами шестистах",
    700: "семисот семистам семьюстами семистах", 800: "восьмисот восьмистам восемьюстами восьмистах",
    900: "девятисот девятистам девятьюстами девятистах",
}.items():
    NUMBER_FORMS.update(dict.fromkeys(_forms.split(), _value))


def lexical(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("ё", "е").replace("%", " процентов ")
    return re.findall(r"[а-яa-z]+|\d+", text)


def normalize(text, version=1, *, with_spans=False):
    """Normalize written/spoken integer values, not negations or split terminology.

    Supports cardinal integers below one million; non-descending components start
    another number (95 98 must never become 193). Range punctuation is ignored,
    but spoken conjunctions such as 'или' remain lexical content.
    """
    if version not in (1, 2):
        raise ValueError(f"Unknown normalization version: {version}")
    numbers = NUMBERS if version == 1 else {**NUMBERS, **NUMBER_FORMS}
    thousands = THOUSANDS if version == 1 else THOUSANDS | {"тысяче", "тысячей", "тысячам", "тысячами", "тысячах"}
    tokens, result, i = lexical(text), [], 0
    spans = []
    while i < len(tokens):
        begin = i
        word = tokens[i]
        # Ambiguous standalone forms (сорока, ста, одной...) require numeric context.
        ambiguous = word in {"сорока", "ста", "одного", "одному", "одним", "одной", "одну", "одними", "семью"}
        numeric_context = (i + 1 < len(tokens) and (
            tokens[i + 1] in numbers or tokens[i + 1] in thousands
            or re.match(r"(?:отсчет|отчет|процент|секунд|минут|час|лет|год|раз|сегмент|слов|человек|метр|рубл)", tokens[i + 1])
        ))
        if (word in numbers or word in thousands) and not (version == 2 and ambiguous and not numeric_context):
            value, part, last, used_thousand = 0, 0, 1000, False
            while i < len(tokens):
                t = tokens[i]
                if t in thousands and not used_thousand:
                    value = (part or 1) * 1000
                    part, last, used_thousand = 0, 1000, True
                elif t in numbers:
                    n = numbers[t]
                    rank = 100 if n >= 100 else 10 if n >= 10 else 1
                    if rank >= last or (last == 10 and part % 100 in range(10, 20)):
                        break
                    part += n
                    last = rank
                else:
                    break
                i += 1
            result.append(str(value + part))
            spans.append((begin, i))
            continue
        result.append("процент" if re.fullmatch(r"процент(?:а|ов|ы)?", word) else word)
        i += 1
        spans.append((begin, i))
    if version == 2:
        keep = [j for j, token in enumerate(result) if not (
            token == "тире" and 0 < j < len(result) - 1
            and result[j - 1].isdigit() and result[j + 1].isdigit()
        )]
        result, spans = [result[j] for j in keep], [spans[j] for j in keep]
    return (result, spans) if with_spans else result



def reference_cards(markdown, names=None):
    """Read explicitly selected filled cards; defaults preserve the legacy study."""
    expected = set(names) if names is not None else {r for group in GROUPS for r in group}
    cards = {}
    for section in re.split(r"(?m)^### ", markdown)[1:]:
        name = section.splitlines()[0].strip()
        if name not in expected:
            continue
        match = re.search(r"\*\*Правильная транскрипция:\*\*\s*([^\n]+)", section)
        if not match or match[1].startswith("**"):
            raise ValueError(f"Missing reference: {name}")
        cards[name] = match[1].replace("(!)", "").strip()
    if set(cards) != expected:
        raise ValueError(f"Missing references: {sorted(expected - set(cards))}")
    return cards


def combine(cards, names):
    """Join only exact normalized suffix/prefix overlap; never silently invent text."""
    words, spans = [], {}
    for name in names:
        extra = [word for word in cards[name].split() if lexical(word)]
        overlap = 0
        if words:
            for n in range(1, min(len(words), len(extra)) + 1):
                if lexical(" ".join(words[-n:])) == lexical(" ".join(extra[:n])):
                    overlap = n
            if not overlap:
                raise ValueError(f"No exact reference overlap before {name}")
        spans[name] = [len(words) - overlap, len(words) - overlap + len(extra)]
        words.extend(extra[overlap:])
    return words, spans


def edit_score(reference, hypothesis, version=1, *, with_alignment=False):
    """Levenshtein backtrace exposes correct reference positions and all edits."""
    a, b = normalize(reference, version), normalize(hypothesis, version)
    d = [list(range(len(b) + 1))]
    for i, x in enumerate(a, 1):
        d.append([i] + [0] * len(b))
        for j, y in enumerate(b, 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (x != y))
    i, j, edits, correct = len(a), len(b), [], []
    alignment = []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            alignment.append((i - 1, j - 1))
            if a[i - 1] == b[j - 1]:
                correct.append(i - 1)
            else:
                edits.append({"op": "S", "index": i - 1, "ref": a[i - 1], "hyp": b[j - 1]})
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            alignment.append((i - 1, None))
            edits.append({"op": "D", "index": i - 1, "ref": a[i - 1], "hyp": ""})
            i -= 1
        else:
            alignment.append((i, j - 1))
            edits.append({"op": "I", "index": i, "ref": "", "hyp": b[j - 1]})
            j -= 1
    chars_a, chars_b = " ".join(a), " ".join(b)
    char_errors = distance(chars_a, chars_b)
    protected = [n for n, t in enumerate(a) if t.isdigit() or t == "не" or t.startswith("нестационар")]
    return {
        "words": len(a),
        "errors": d[-1][-1],
        "wer": d[-1][-1] / max(1, len(a)),
        "char_errors": char_errors,
        "chars": len(chars_a),
        "cer": char_errors / max(1, len(chars_a)),
        "S": sum(e["op"] == "S" for e in edits),
        "D": sum(e["op"] == "D" for e in edits),
        "I": sum(e["op"] == "I" for e in edits),
        "edits": list(reversed(edits)),
        "protected_correct": sorted(set(correct) & set(protected)),
        "protected": protected,
        **({"alignment": list(reversed(alignment))} if with_alignment else {}),
    }


def term_counts(text):
    value = " ".join(normalize(text))
    patterns = {
        "quasi": r"\b(?:квази )?квазистационар\w*\b|\bквази стационар\w*\b",
        "nonstationary": r"\bнестационар\w*\b",
        "stationary": r"\bстационар\w*\b",
        "correlation": r"\bкорреляционно спектраль\w*",
        "series": r"\bвременн\w* ряд\w*",
        "sliding": r"\bскользящ\w*",
        "segment": r"\bсегмент\w*",
        "sample": r"\bотсчет\w*",
        "overlap": r"\bперекрыти\w*",
        "statistics": r"\bстатистик\w*",
        "autocorrelation": r"\bавтокорреляци\w*",
        "periodogram": r"\bпериодограмм\w*",
        "fourier": r"\bфурье\b",
    }
    return {key: len(re.findall(pattern, value)) for key, pattern in patterns.items()}


def select_text(words, start, end):
    return " ".join(w["text"] for w in words if start <= (w["start"] + w["end"]) / 2 < end)


def project_cards(group, text, alignment, version):
    """Project one deterministic group alignment onto frozen reference spans.

    Insertions belong to the reference token on their right (trailing insertions
    to the last token). Overlapping cards deliberately share the same tokens.
    This changes evaluation boundaries only, never the saved ASR transcript.
    """
    ref_tokens, ref_spans = normalize(group['text'], version, with_spans=True)
    _, hyp_spans = normalize(text, version, with_spans=True)
    raw_ref = group['text'].split()
    raw_hyp = text.split()
    # Map normalized tokens back to original whitespace words, preserving model
    # spelling and complete spoken numbers in the displayed hypothesis.
    hyp_owners = [i for i, word in enumerate(raw_hyp) for _ in lexical(word)]
    result = {}
    for name, card in group['cards'].items():
        if 'span' in card:
            lo, hi = card['span']
            lex_lo = len(lexical(' '.join(raw_ref[:lo])))
            lex_hi = len(lexical(' '.join(raw_ref[:hi])))
            indices = [i for i, (a, b) in enumerate(ref_spans) if a >= lex_lo and b <= lex_hi]
        else:
            target = normalize(card['text'], version)
            starts = [i for i in range(len(ref_tokens) - len(target) + 1)
                      if ref_tokens[i:i + len(target)] == target]
            if len(starts) != 1:
                raise ValueError(f'Ambiguous reference span for {name}')
            indices = list(range(starts[0], starts[0] + len(target)))
        if not indices or [ref_tokens[i] for i in indices] != normalize(card['text'], version):
            raise ValueError(f'Reference span/normalization mismatch for {name}')
        left, right = indices[0], indices[-1] + 1
        selected = [j for i, j in alignment if j is not None
                    and left <= min(i, len(ref_tokens) - 1) < right]
        if selected:
            # Include punctuation / spoken range separators between matched
            # tokens as originally recognized, without rewriting the text.
            first = hyp_owners[hyp_spans[selected[0]][0]]
            last = hyp_owners[hyp_spans[selected[-1]][1] - 1]
            result[name] = ' '.join(raw_hyp[first:last + 1])
        else:
            result[name] = ''
    return result


def score_case(words, reference, version=1, *, card_method="group-alignment-v1"):
    if card_method not in ("group-alignment-v1", "time-window-v1"):
        raise ValueError(f"Unknown card scoring method: {card_method}")
    groups, cards = {}, {}
    for group in reference["groups"]:
        text = select_text(words, *group["window"])
        score = edit_score(group["text"], text, version, with_alignment=True)
        alignment = score.pop("alignment")
        projected = project_cards(group, text, alignment, version) if card_method == "group-alignment-v1" else {}
        expected, actual = term_counts(group["text"]), term_counts(text)
        score.update(
            hypothesis=text,
            reference=group["text"],
            excess_terms={t: max(0, actual[t] - expected[t]) for t in actual},
        )
        groups[group["name"]] = score
        for name, card in group["cards"].items():
            hyp = projected[name] if card_method == "group-alignment-v1" else select_text(words, *card["window"])
            cards[name] = {**edit_score(card["text"], hyp, version), "reference": card["text"], "hypothesis": hyp}
    total = {
        key: sum(g[key] for g in groups.values())
        for key in ("words", "errors", "chars", "char_errors", "S", "D", "I")
    }
    total.update(
        wer=total["errors"] / total["words"],
        cer=total["char_errors"] / total["chars"],
        worst_wer=max(g["wer"] for g in groups.values()),
        false_quasi=sum(g["excess_terms"]["quasi"] for g in groups.values()),
    )
    target = {}
    for name, window in [("first", [145, 155]), ("second", reference["target_window"])]:
        text = select_text(words, *window)
        tokens = normalize(text)
        target[name] = {
            "text": text,
            "exact": "квазистационарности" in tokens,
            "split": "квази стационарности" in " ".join(tokens),
        }
    return {"total": total, "groups": groups, "cards": cards, "target": target, "card_method": card_method}


def regression(candidate, baseline):
    if candidate["total"]["false_quasi"] > baseline["total"]["false_quasi"]:
        return True
    return any(
        not set(b["protected_correct"]) <= set(candidate["groups"][name]["protected_correct"])
        for name, b in baseline["groups"].items()
    )
