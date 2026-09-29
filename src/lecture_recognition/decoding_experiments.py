"""Frozen-reference scoring and configuration for decoding experiments (no audio filters)."""

import re
import unicodedata

from .experiments import distance

TOPIC = "Лекция по математической статистике и анализу временных рядов."
TERMS = [
    "корреляционно-спектральный анализ",
    "временной ряд",
    "стационарность",
    "нестационарность",
    "скользящий анализ",
    "сегмент",
    "отсчёт",
    "перекрытие",
    "статистика",
]
BIAS_TERMS = TERMS + ["квазистационарность", "квазистационарности", "нестационарных", "отсчётов"]
PROMPTS = {
    "empty": None,
    "topic_ru": TOPIC,
    "topic_en": "A lecture on mathematical statistics and time series analysis.",
    "dictionary": "Термины: " + "; ".join(TERMS) + ".",
    "dictionary_target": "Термины: " + "; ".join(TERMS + ["квазистационарность"]) + ".",
    "dictionary_forms": "Термины: " + "; ".join(BIAS_TERMS) + ".",
    "distractors": TOPIC + " Термины: автокорреляция; периодограмма; преобразование Фурье.",
}
GROUPS = [("R001", "R002"), ("R003", "R004"), ("R005", "R006", "R007", "R008")]
# Broad search windows, independent of any experimental hypothesis.
REFERENCE_WINDOWS = [(10.8, 52.8), (71.4, 123.3), (155.4, 260.5)]

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


def lexical(text):
    text = unicodedata.normalize("NFC", text).casefold().replace("ё", "е").replace("%", " процентов ")
    return re.findall(r"[а-яa-z]+|\d+", text)


def normalize(text):
    """Normalize written/spoken integer values, not negations or split terminology.

    Supports cardinal integers below one million; non-descending components start
    another number (95 98 must never become 193). Range punctuation is ignored,
    but spoken conjunctions such as 'или' remain lexical content.
    """
    tokens, result, i = lexical(text), [], 0
    while i < len(tokens):
        word = tokens[i]
        if word in NUMBERS or word in THOUSANDS:
            value, part, last, used_thousand = 0, 0, 1000, False
            while i < len(tokens):
                t = tokens[i]
                if t in THOUSANDS and not used_thousand:
                    value = (part or 1) * 1000
                    part, last, used_thousand = 0, 1000, True
                elif t in NUMBERS:
                    n = NUMBERS[t]
                    rank = 100 if n >= 100 else 10 if n >= 10 else 1
                    if rank >= last or (last == 10 and part % 100 in range(10, 20)):
                        break
                    part += n
                    last = rank
                else:
                    break
                i += 1
            result.append(str(value + part))
            continue
        result.append("процент" if re.fullmatch(r"процент(?:а|ов|ы)?", word) else word)
        i += 1
    return result


def reference_cards(markdown):
    cards = {}
    for section in re.split(r"(?m)^### ", markdown)[1:]:
        name = section.splitlines()[0].strip()
        if name not in {r for group in GROUPS for r in group}:
            continue
        match = re.search(r"\*\*Правильная транскрипция:\*\*\s*([^\n]+)", section)
        if not match or match[1].startswith("**"):
            raise ValueError(f"Missing reference: {name}")
        cards[name] = match[1].replace("(!)", "").strip()
    if len(cards) != 8:
        raise ValueError("Exactly R001–R008 must be present")
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


def edit_score(reference, hypothesis):
    """Levenshtein backtrace exposes correct reference positions and all edits."""
    a, b = normalize(reference), normalize(hypothesis)
    d = [list(range(len(b) + 1))]
    for i, x in enumerate(a, 1):
        d.append([i] + [0] * len(b))
        for j, y in enumerate(b, 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (x != y))
    i, j, edits, correct = len(a), len(b), [], []
    while i or j:
        if i and j and d[i][j] == d[i - 1][j - 1] + (a[i - 1] != b[j - 1]):
            if a[i - 1] == b[j - 1]:
                correct.append(i - 1)
            else:
                edits.append({"op": "S", "index": i - 1, "ref": a[i - 1], "hyp": b[j - 1]})
            i, j = i - 1, j - 1
        elif i and d[i][j] == d[i - 1][j] + 1:
            edits.append({"op": "D", "index": i - 1, "ref": a[i - 1], "hyp": ""})
            i -= 1
        else:
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


def score_case(words, reference):
    groups, cards = {}, {}
    for group in reference["groups"]:
        text = select_text(words, *group["window"])
        score = edit_score(group["text"], text)
        expected, actual = term_counts(group["text"]), term_counts(text)
        score.update(
            hypothesis=text,
            reference=group["text"],
            excess_terms={t: max(0, actual[t] - expected[t]) for t in actual},
        )
        groups[group["name"]] = score
        for name, card in group["cards"].items():
            hyp = select_text(words, *card["window"])
            cards[name] = {**edit_score(card["text"], hyp), "reference": card["text"], "hypothesis": hyp}
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
    return {"total": total, "groups": groups, "cards": cards, "target": target}


def regression(candidate, baseline):
    if candidate["total"]["false_quasi"] > baseline["total"]["false_quasi"]:
        return True
    return any(
        not set(b["protected_correct"]) <= set(candidate["groups"][name]["protected_correct"])
        for name, b in baseline["groups"].items()
    )


def config(prompt="empty", beams=1, penalty=1.0, bias=0.0, **extra):
    return {
        "prompt": prompt,
        "beams": beams,
        "length_penalty": penalty,
        "bias": bias,
        "batch": 1,
        "layout": "production",
        "repeat": 0,
        **extra,
    }


def initial_matrix():
    return [config(p, b) for p in list(PROMPTS)[:6] for b in (1, 2, 4)]


def generation_kwargs(cfg, tokenizer):
    result = {"do_sample": False, "num_beams": cfg["beams"]}
    if cfg["beams"] > 1:
        result["length_penalty"] = cfg["length_penalty"]
    if cfg["bias"]:
        sequences = {
            tuple(tokenizer.encode(prefix + term, add_special_tokens=False))
            for term in BIAS_TERMS
            for prefix in ("", " ")
        }
        result["sequence_bias"] = {seq: float(cfg["bias"]) for seq in sequences if seq}
    return result
