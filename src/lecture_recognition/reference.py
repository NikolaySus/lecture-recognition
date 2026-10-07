"""Frozen alignment recipe for the historical R001–R008 diagnostic set."""

import json

import numpy as np

from .audio import RATE
from .cli import atomic_text
from .decoding_experiments import REFERENCE_WINDOWS
from .evaluation import GROUPS, combine, lexical
from .models import MODEL_REVISIONS, align_wave, load, release


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def make_reference(audio, cards, root):
    """Locate the reference once, before decoding; preserve raw word/card identities."""
    path = root / "reference.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    processor, model = load("alignment", MODEL_REVISIONS["alignment"])
    groups = []
    try:
        for names, (start, end) in zip(GROUPS, REFERENCE_WINDOWS):
            tokens, spans = combine(cards, names)
            text = " ".join(tokens)
            # The aligner strips hyphens (95-98 -> 9598). Explicitly separate
            # components for alignment, then map back to original reference words.
            pieces = [lexical(token) for token in tokens]
            alignment_tokens = [part for parts in pieces for part in parts]
            alignment_text = " ".join(alignment_tokens)
            wave = np.asarray(audio[round(start * RATE) : round(end * RATE)])
            for pad in (0, 0.25, 0.5, 1.0):
                padded = np.pad(wave, (round(pad * RATE), round(pad * RATE))) if pad else wave
                if names == GROUPS[1]:
                    # The long reference gives a collapsed opening despite padding.
                    # Align its first complete sentence independently; this split is
                    # frozen before ASR experiments and never changes ASR inputs.
                    split = 77.4 - start
                    first = align_wave(
                        processor,
                        model,
                        padded[: round((split + pad) * RATE)],
                        " ".join(alignment_tokens[:6]),
                        "ru",
                    )
                    rest = align_wave(
                        processor,
                        model,
                        padded[round((split + pad) * RATE) :],
                        " ".join(alignment_tokens[6:]),
                        "ru",
                    )
                    for word in rest:
                        word["start_time"] += split + pad
                        word["end_time"] += split + pad
                    aligned = first + rest
                else:
                    aligned = align_wave(processor, model, padded, alignment_text, "ru")
                for word in aligned:
                    word["start_time"] = max(0, word["start_time"] - pad)
                    word["end_time"] = min(len(wave) / RATE, word["end_time"] - pad)
                if aligned[0]["end_time"] > aligned[0]["start_time"] and all(
                    w["end_time"] >= w["start_time"] for w in aligned
                ):
                    break
            else:
                raise RuntimeError(f"Reference has degenerate leading timestamps: {names}")
            if len(aligned) != len(alignment_tokens):
                raise RuntimeError(
                    f"Reference alignment token mismatch for {names}: {len(aligned)} vs {len(alignment_tokens)}"
                )
            for token, word in zip(alignment_tokens, aligned):
                if lexical(token) != lexical(word["text"]):
                    raise RuntimeError(f"Reference alignment changed token: {token!r} vs {word['text']!r}")
                word["start_time"] += start
                word["end_time"] += start
            merged, offset = [], 0
            for token, parts in zip(tokens, pieces):
                merged.append(
                    {
                        "text": token,
                        "start_time": aligned[offset]["start_time"],
                        "end_time": aligned[offset + len(parts) - 1]["end_time"],
                    }
                )
                offset += len(parts)
            aligned = merged
            if any(b["start_time"] + 0.15 < a["start_time"] for a, b in zip(aligned, aligned[1:])):
                raise RuntimeError(f"Non-monotonic reference alignment: {names}")

            def bounds(left, right):
                return [
                    max(start, aligned[left]["start_time"] - 0.15),
                    min(end, aligned[right - 1]["end_time"] + 0.15),
                ]

            groups.append(
                {
                    "name": "+".join(names),
                    "text": text,
                    "window": bounds(0, len(tokens)),
                    "search_window": [start, end],
                    "words": aligned,
                    "cards": {
                        name: {"text": cards[name], "window": bounds(*span), "span": span}
                        for name, span in spans.items()
                    },
                }
            )
    finally:
        del processor, model
        release()
    target = [w for g in groups for w in g["words"] if "квазистационарности" in lexical(w["text"])]
    if len(target) != 1:
        raise RuntimeError("Expected one labeled target word in R008")
    reference = {
        "groups": groups,
        "target_window": [target[0]["start_time"] - 2, target[0]["end_time"] + 2],
        "boundary_margin_seconds": 0.15,
        "method": "fixed-reference-forced-alignment",
    }
    write_json(path, reference)
    return reference
