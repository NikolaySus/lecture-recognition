"""Deterministic preprocessing and scoring for isolated ASR experiments."""

import hashlib
import json
import subprocess
import unicodedata

import numpy as np

from .audio import RATE

FILTERS = {
    "none": None,
    "peak": None,
    "rms": None,
    "highpass": "highpass=f=80",
    "bandpass": "highpass=f=80,lowpass=f=7000",
    "denoise6": "afftdn=nr=6:nf=-40:tn=1",
    "denoise12": "afftdn=nr=12:nf=-40:tn=1",
    "highpass_denoise": "highpass=f=80,afftdn=nr=6:nf=-40:tn=1",
    "compress": "acompressor=threshold=0.063095734:ratio=2:attack=20:release=250:makeup=1",
}


def identity(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def normalized_token(text):
    text = unicodedata.normalize("NFC", text).casefold()
    while text and unicodedata.category(text[0])[0] in {"P", "Z"}:
        text = text[1:]
    while text and unicodedata.category(text[-1])[0] in {"P", "Z"}:
        text = text[:-1]
    return text


def distance(a, b):
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(new[-1] + 1, row[j] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def evaluate(raw_text, words, target, window):
    """Only exact whitespace-delimited tokens qualify; never concatenate fragments."""
    target = normalized_token(target)
    raw_hit = target in [normalized_token(t) for t in raw_text.split()]
    nearby, matches = [], []
    for w in words:
        if not w["end"] > w["start"] or not window[0] <= (w["start"] + w["end"]) / 2 <= window[1]:
            continue
        for token in w["text"].split():
            normalized = normalized_token(token)
            if not normalized:
                continue
            item = {
                "text": token,
                "start": w["start"],
                "end": w["end"],
                "distance": distance(target, normalized),
            }
            nearby.append(item)
            if normalized == target:
                matches.append(item)
    return {
        "pass": raw_hit and bool(matches),
        "raw_hit": raw_hit,
        "matches": matches,
        "nearby": nearby,
        "distance": min((w["distance"] for w in nearby), default=len(target)),
    }


def preprocess(audio, name, excluded):
    wave = np.asarray(audio, dtype="<f4").copy()
    if name not in FILTERS:
        raise ValueError(f"Unknown filter: {name}")
    if name in {"peak", "rms"}:
        peak = float(np.max(np.abs(wave)))
        rms = float(np.sqrt(np.mean(wave.astype(np.float64) ** 2)))
        if peak > 0:
            gain = 10 ** (-1 / 20) / peak
            if name == "rms" and rms > 0:
                gain = min(gain, 10 ** (-20 / 20) / rms)
            wave *= gain
    elif FILTERS[name]:
        proc = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                str(RATE),
                "-ac",
                "1",
                "-i",
                "pipe:0",
                "-af",
                FILTERS[name],
                "-f",
                "f32le",
                "-ar",
                str(RATE),
                "-ac",
                "1",
                "pipe:1",
            ],
            input=wave.tobytes(),
            capture_output=True,
            check=True,
        )
        wave = np.frombuffer(proc.stdout, dtype="<f4").copy()
    if len(wave) != len(audio) or not np.isfinite(wave).all():
        raise ValueError("Preprocessing changed sample count or produced non-finite samples")
    # Floating-point AAC decoding can already exceed full scale. Preserve the
    # unmodified control; reject only newly introduced overload from filters.
    if name != "none" and np.max(np.abs(wave), initial=0) >= max(1.0, float(np.max(np.abs(audio)))):
        raise ValueError("Preprocessing produced clipping")
    for a, b in excluded:
        wave[max(0, round(a * RATE)) : min(len(wave), round(b * RATE))] = 0
    return wave
