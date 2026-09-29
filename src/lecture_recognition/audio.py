"""Decode once to disk; retain sample-accurate source coordinates."""

import hashlib
import shutil
import subprocess
from pathlib import Path

import numpy as np

RATE = 16000


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def masked_audio(audio, target: Path, excluded, padded_samples):
    """Separate atomic PCM cache, bounded memory; source audio is never modified."""
    if padded_samples < len(audio):
        raise ValueError("Padded audio cannot be shorter than source")
    if not target.exists() or target.stat().st_size != padded_samples * 4:
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            for start in range(0, padded_samples, RATE * 60):
                end = min(padded_samples, start + RATE * 60)
                block = np.zeros(end - start, dtype="<f4")
                available = max(0, min(len(audio), end) - start)
                block[:available] = audio[start : start + available]
                for a, b in excluded:
                    left, right = max(start, round(a * RATE)), min(end, round(b * RATE))
                    if left < right:
                        block[left - start : right - start] = 0
                f.write(block.tobytes())
        tmp.replace(target)
    return np.memmap(target, dtype="<f4", mode="r")


def decode(source: Path, target: Path) -> np.memmap:
    if not target.exists():
        if not shutil.which("ffmpeg"):
            raise RuntimeError("FFmpeg is required on PATH.")
        tmp = target.with_suffix(".tmp")
        result = subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:a:0",
                "-vn",
                "-ac",
                "1",
                "-ar",
                str(RATE),
                "-f",
                "f32le",
                str(tmp),
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"FFmpeg could not decode input: {result.stderr.strip()}")
        if tmp.stat().st_size == 0:
            tmp.unlink()
            raise RuntimeError("Input contains no audio samples.")
        tmp.replace(target)
    return np.memmap(target, dtype="<f4", mode="r")
