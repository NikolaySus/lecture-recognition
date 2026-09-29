"""Search preprocessing without using the test word as an ASR prompt."""

import argparse
import hashlib
import json
import logging
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from lecture_recognition.audio import RATE, decode, digest
from lecture_recognition.cli import Cache, atomic_text
from lecture_recognition.experiments import FILTERS, evaluate, identity, preprocess
from lecture_recognition.models import MODEL_REVISIONS, align, load, release
from lecture_recognition.timeline import (
    Chunk,
    lecturer_regions,
    make_packed_chunks,
    merge_words,
    packing_size,
    student_only_regions,
    to_srt,
)


def transcribe(audio, chunks, batch_size):
    """No hidden chunk splitting or automatic batch-size fallback in measurements."""
    processor, model = load("asr", MODEL_REVISIONS["asr"])
    result = []
    try:
        for offset in range(0, len(chunks), batch_size):
            group = chunks[offset : offset + batch_size]
            waves = [np.asarray(audio[round(c.start * RATE) : round(c.end * RATE)]) for c in group]
            inputs = output = None
            try:
                inputs = processor.apply_transcription_request(audio=waves, language=["ru"] * len(group))
                inputs = inputs.to("cuda", dtype=model.dtype)
                limit = max(1024, math.ceil(max(map(len, waves)) / RATE * 12 + 256))
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=limit, do_sample=False)
                rows = output[:, inputs["input_ids"].shape[1] :].cpu().tolist()
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                for c, ids in zip(group, rows):
                    end = next((i for i, token in enumerate(ids) if token in eos), None)
                    if end is None:
                        raise RuntimeError("token_limit: generation did not reach EOS")
                    text = processor.tokenizer.decode(ids[:end], skip_special_tokens=True).strip()
                    result.append({"chunk": c.dict(), "text": text.split("<asr_text>", 1)[-1].strip()})
            finally:
                del inputs, output
        return result
    finally:
        del model, processor
        release()


def srt_words(srt):
    def seconds(stamp):
        h, m, s = stamp.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)

    words = []
    for block in srt.strip().split("\n\n") if srt.strip() else []:
        lines = block.splitlines()
        start, end = map(seconds, lines[1].split(" --> "))
        words.append({"text": " ".join(lines[2:]), "start": start, "end": end})
    return words


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--limit-seconds", type=float, default=300)
    parser.add_argument("--target", default="квазистационарности")
    parser.add_argument("--window", type=float, nargs=2, default=[145, 155])
    parser.add_argument("--max-cases", type=int, default=30)
    parser.add_argument("--output", type=Path, default=Path(".lecture-cache/preprocessing-benchmark"))
    args = parser.parse_args()
    if not 0 < args.max_cases <= 30 or not 0 <= args.window[0] < args.window[1] < args.limit_seconds <= 300:
        parser.error("Require 1..30 cases, and a valid target window within a prefix of at most 300s")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    torch.cuda.set_per_process_memory_fraction(0.95)
    source_hash = digest(args.audio)
    base = Path(".lecture-cache") / source_hash
    base.mkdir(parents=True, exist_ok=True)
    original = decode(args.audio, base / "audio.f32")
    # Existing whole-recording diarization identifies the same global lecturer.
    candidates = []
    for path in base.glob("*/settings.json"):
        settings = json.loads(path.read_text(encoding="utf-8"))
        if (
            settings.get("version") == 4
            and settings.get("samples") == len(original)
            and settings.get("revisions") == MODEL_REVISIONS
        ):
            c = Cache(path.parent)
            if c.read("diarization", {}) is not None and c.read("chunking", {}) is not None:
                candidates.append(c)
    if not candidates:
        raise RuntimeError("A completed current production diarization/chunking cache is required")
    production = max(candidates, key=lambda c: (c.root / "settings.json").stat().st_mtime_ns)
    segments = production.read("diarization", {})
    speaker, regions, _ = lecturer_regions(segments, len(original) / RATE)
    excluded = student_only_regions(segments, len(original) / RATE, speaker)
    samples = round(args.limit_seconds * RATE)
    if samples > len(original):
        raise ValueError("Requested prefix exceeds recording")
    original = np.asarray(original[:samples])
    regions = [(a, min(b, args.limit_seconds)) for a, b in regions if a < args.limit_seconds]
    target_mid = sum(args.window) / 2
    current = next(
        Chunk(**c)
        for c in production.read("chunking", {})["chunks"]
        if c["core_start"] <= target_mid < c["core_end"]
    )
    if current.end > args.limit_seconds:
        raise ValueError("Production target chunk extends beyond experimental prefix")
    # Relative alternatives retain the specified 60, 90, and 118 second cores.
    contexts = [("current", current, 118.0)]
    for size in (60, 90, 118):
        start = target_mid - (size / 2 if size != 118 else 60)
        contexts.append((f"core{size}", Chunk(start - 1, start + size + 1, start, start + size), float(size)))
    if any(c.start < 0 or c.end > args.limit_seconds for _, c, _ in contexts):
        raise ValueError("Context matrix does not fit inside prefix")
    metadata = {
        "version": 2,
        "source": source_hash,
        "revisions": MODEL_REVISIONS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "ffmpeg": subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, check=True
        ).stdout.splitlines()[0],
        "samples": samples,
        "target": args.target,
        "window": args.window,
        "diarization": identity(segments),
        "current": current.dict(),
        "language": "ru",
        "dtype": "bfloat16",
        "greedy": True,
        "memory_fraction": 0.95,
    }
    root = args.output / identity(metadata)[:16]
    root.mkdir(parents=True, exist_ok=True)
    atomic_text(root / "metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"RESULTS {root}", flush=True)
    waves = {"mono": original}
    prepared = {}
    records = []
    seen = {}
    winner = None

    def audio_for(channel, effect):
        key = (channel, effect)
        if key in prepared:
            return prepared[key]
        if channel not in waves:
            channel_id = {"left": 0, "right": 1}[channel]
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-i",
                    str(args.audio),
                    "-t",
                    str(args.limit_seconds),
                    "-af",
                    f"pan=mono|c0=c{channel_id}",
                    "-ar",
                    str(RATE),
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            )
            waves[channel] = np.frombuffer(proc.stdout, dtype="<f4").copy()
            if len(waves[channel]) != samples:
                raise ValueError("Channel decode changed sample count")
        prepared[key] = preprocess(waves[channel], effect, excluded)
        return prepared[key]

    def report():
        atomic_text(
            root / "summary.json",
            json.dumps({"winner": winner, "cases": records}, ensure_ascii=False, indent=2),
        )
        rows = [
            "# Preprocessing experiment",
            "",
            "| Case | Status | Hit | Distance | Seconds | GiB |",
            "|---|---|---|---|---|---|",
        ]
        for r in records:
            score = r.get("score", {})
            rows.append(
                f"| {r['name']} | {r['status']} | {score.get('pass', False)} | "
                f"{score.get('distance', '')} | {r.get('seconds', 0):.2f} | {r.get('vram', 0):.2f} |"
            )
        rows += [
            "",
            f"Stable winner: {winner or 'not found'}",
            "",
            "This checks one term, not whole-transcript WER. No term was supplied to ASR.",
        ]
        atomic_text(root / "comparison.md", "\n".join(rows) + "\n")

    def run_case(name, channel, effect, chunk, max_core, full=False):
        config = {
            "name": name,
            "channel": channel,
            "effect": effect,
            "chunk": chunk.dict(),
            "max_core": max_core,
            "full": full,
            "batch": 2 if full else 1,
        }
        key = identity(config)
        if key in seen:
            return seen[key]
        if len(records) >= args.max_cases:
            return None
        path = root / f"{len(records) + 1:02}-{name}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["config"] != config:
                raise RuntimeError("Experiment ordering changed; choose a new output directory")
        else:
            started = time.perf_counter()
            result = {"name": name, "config": config, "status": "ok"}
            torch.cuda.reset_peak_memory_stats()
            try:
                audio = audio_for(channel, effect)
                if full:
                    size, *_ = packing_size(len(audio), 60, max_core)
                    audio = np.pad(audio, (0, size - len(audio)))
                    chunks = make_packed_chunks(audio, regions, 60, max_core)
                else:
                    chunks = [chunk]
                if any(c.core_end - c.core_start < 60 - 1e-6 or c.end - c.start > 120 + 1e-6 for c in chunks):
                    raise ValueError("Experiment violates chunk limits")
                # Audio hashes deduplicate equivalent transformations, not just labels.
                audio_hash = hashlib.sha256(audio.tobytes()).hexdigest()
                inference_key = {
                    "audio": audio_hash,
                    "chunks": [c.dict() for c in chunks],
                    "batch": config["batch"],
                    "revisions": MODEL_REVISIONS,
                }
                cache = Cache(root / "inference" / identity(inference_key))
                transcripts = cache.read("transcripts", {})
                if transcripts is None:
                    transcripts = transcribe(audio, chunks, config["batch"])
                    cache.write("transcripts", {}, transcripts)
                raw_text = "\n".join(t["text"] for t in transcripts)
                aligned = align(
                    audio,
                    transcripts,
                    "ru",
                    MODEL_REVISIONS["alignment"],
                    cache.read,
                    cache.write,
                    context_regions=[(0, len(audio) / RATE)],
                )
                words = merge_words(aligned)
                score = evaluate(raw_text, words, args.target, args.window)
                srt = to_srt(words, regions)
                if full:
                    score["srt_pass"] = evaluate(raw_text, srt_words(srt), args.target, args.window)["pass"]
                    score["pass"] = score["pass"] and score["srt_pass"]
                result.update(
                    score=score,
                    text=raw_text,
                    words=[{k: v for k, v in w.items() if k != "chunks"} for w in words],
                    chunks=[c.dict() for c in chunks],
                    audio_sha256=audio_hash,
                )
                atomic_text(path.with_suffix(".txt"), raw_text)
                atomic_text(path.with_suffix(".srt"), srt)
            except torch.cuda.OutOfMemoryError:
                result.update(status="oom")
            except (RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
                result.update(status="error", error=str(exc))
            finally:
                release()
            result.update(
                seconds=time.perf_counter() - started, vram=torch.cuda.max_memory_allocated() / 2**30
            )
            atomic_text(path, json.dumps(result, ensure_ascii=False, indent=2))
        seen[key] = result
        records.append(result)
        report()
        print(
            json.dumps(
                {k: result[k] for k in ("name", "status", "score", "seconds") if k in result},
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    def trial(name, channel, effect, chunk, max_core):
        nonlocal winner
        r = run_case(name, channel, effect, chunk, max_core)
        if r is None or not r.get("score", {}).get("pass"):
            return False
        for shift in (-5, 5):
            shifted = Chunk(
                chunk.start + shift, chunk.end + shift, chunk.core_start + shift, chunk.core_end + shift
            )
            r = run_case(f"{name}-shift{shift:+}", channel, effect, shifted, max_core)
            if r is None or not r.get("score", {}).get("pass"):
                return False
        r = run_case(f"{name}-full", channel, effect, chunk, max_core, full=True)
        if r is not None and r.get("score", {}).get("pass"):
            winner = name
            report()
            return True
        return False

    for name, c, size in contexts:
        if trial(name, "mono", "none", c, size):
            return
    for channel in ("left", "right"):
        for name, c, size in (contexts[0], contexts[2]):
            if trial(f"{channel}-{name}", channel, "none", c, size):
                return
    ranked = []
    for effect in list(FILTERS)[1:]:
        if trial(effect, "mono", effect, current, 118):
            return
        matching = next((r for r in records if r["name"] == effect and r.get("score")), None)
        if matching:
            ranked.append(matching)
    ranked.sort(key=lambda r: (r["score"]["distance"], r["seconds"], list(FILTERS).index(r["name"])))
    for r in ranked[:2]:
        for name, c, size in contexts:
            effect = r["name"]
            if name == "current":
                continue  # Already tested, including confirmation if it matched.
            if trial(f"{effect}-{name}", "mono", effect, c, size):
                return
    report()
    print("No stable winner within the matrix/budget", flush=True)


if __name__ == "__main__":
    main()
