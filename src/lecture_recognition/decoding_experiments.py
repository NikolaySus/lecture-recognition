"""Frozen-reference scoring and configuration for decoding experiments (no audio filters)."""

from .evaluation import (  # noqa: F401 -- legacy public imports
    GROUPS,
    combine,
    edit_score,
    lexical,
    normalize,
    reference_cards,
    regression,
    score_case,
)

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
# Broad search windows, independent of any experimental hypothesis.
REFERENCE_WINDOWS = [(10.8, 52.8), (71.4, 123.3), (155.4, 260.5)]


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
