import argparse
import hashlib
import json
import logging
import math
import time
from pathlib import Path

from .audio import RATE, decode, digest, masked_audio
from .timeline import (
    lecturer_regions,
    make_packed_chunks,
    merge_words,
    packing_size,
    student_only_regions,
    to_srt,
)

CACHE_VERSION = 4


def atomic_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


class Cache:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, stage, key):
        digest = hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()
        return self.root / stage / (digest + ".json")

    def read(self, stage, key):
        path = self.path(stage, key)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write(self, stage, key, value):
        atomic_text(self.path(stage, key), json.dumps(value, ensure_ascii=False))


def run(args):
    import torch

    from .models import MODEL_REVISIONS, align, diarize, recognize, release

    if not args.input.is_file():
        raise ValueError(f"Audio file does not exist: {args.input}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Run uv sync and check your NVIDIA driver.")
    supported = {"ru", "en", "zh", "yue", "fr", "de", "it", "pt", "es"}
    if args.language not in supported:
        raise ValueError("Supported language codes: " + ", ".join(sorted(supported)))
    output = args.output or args.input.with_suffix(".preview.srt" if args.limit_seconds else ".srt")
    if output.resolve() == args.input.resolve():
        raise ValueError("Output must not overwrite the source audio.")
    logging.info("CUDA: %s", torch.cuda.get_device_name())
    started = time.perf_counter()
    source_hash = digest(args.input)
    base = args.cache_dir / source_hash
    base.mkdir(parents=True, exist_ok=True)
    audio = decode(args.input, base / "audio.f32")
    if args.limit_seconds:
        audio = audio[: round(args.limit_seconds * RATE)]
    logging.info("Audio: %.2f seconds", len(audio) / RATE)
    # Pin all weights as well as dependencies for reproducible new recordings.
    revisions_path = base / "revisions.json"
    revisions = MODEL_REVISIONS
    atomic_text(revisions_path, json.dumps(revisions))
    import transformers

    settings = {
        "version": CACHE_VERSION,
        "revisions": revisions,
        "language": args.language,
        "samples": len(audio),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "chunk_seconds": args.chunk_seconds,
        "min_chunk_seconds": args.min_chunk_seconds,
        "masking": "student-only-silence-v1",
    }
    if args.batch_size != 1:
        settings["batch_size"] = args.batch_size
    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:16]
    cache = Cache(base / key)
    atomic_text(cache.root / "settings.json", json.dumps(settings, indent=2))
    timings = {}

    def stage(name, work):
        release()
        torch.cuda.reset_peak_memory_stats()
        t = time.perf_counter()
        result = work()
        timings[name] = {
            "seconds": time.perf_counter() - t,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        }
        logging.info(
            "%s: %.1fs, peak %.2f GiB", name, timings[name]["seconds"], timings[name]["peak_vram_gb"]
        )
        return result

    segments = cache.read("diarization", {})
    if segments is None:
        segments = stage("diarization", lambda: diarize(audio, revisions["diarization"]))
        cache.write("diarization", {}, segments)
    speaker, regions, totals = lecturer_regions(segments, len(audio) / RATE)
    logging.info("Lecturer: %s; speaker durations: %s", speaker, totals)
    if speaker is None:
        text = ""
    else:
        original_duration = len(audio) / RATE
        excluded = student_only_regions(segments, original_duration, speaker)
        padded, _, _, _ = packing_size(len(audio), args.min_chunk_seconds, args.chunk_seconds - 2)
        audio = masked_audio(audio, cache.root / "masked.f32", excluded, padded)
        context_regions = [(0, len(audio) / RATE)]
        chunks = make_packed_chunks(
            audio, regions, min_core=args.min_chunk_seconds, max_core=args.chunk_seconds - 2
        )
        stats = {}
        for label, values in (
            ("core", [c.core_end - c.core_start for c in chunks]),
            ("input", [c.end - c.start for c in chunks]),
        ):
            stats[label] = {"min": min(values), "mean": sum(values) / len(values), "max": max(values)}
            logging.info(
                "ASR %s: %d chunks, min %.2fs, mean %.2fs, max %.2fs",
                label,
                len(chunks),
                stats[label]["min"],
                stats[label]["mean"],
                stats[label]["max"],
            )
        cache.write(
            "chunking",
            {},
            {
                "chunks": [c.dict() for c in chunks],
                "stats": stats,
                "padding_seconds": len(audio) / RATE - original_duration,
            },
        )
        transcripts = stage(
            "asr",
            lambda: recognize(
                audio,
                chunks,
                args.language,
                revisions["asr"],
                cache.read,
                cache.write,
                args.batch_size,
                min_core=args.min_chunk_seconds,
            ),
        )
        aligned = stage(
            "alignment",
            lambda: align(
                audio,
                transcripts,
                args.language,
                revisions["alignment"],
                cache.read,
                cache.write,
                context_regions=context_regions,
            ),
        )
        text = to_srt(merge_words(aligned), regions)
    atomic_text(output, text)
    atomic_text(
        cache.root / "run.json",
        json.dumps(
            {
                "timings": timings,
                "speaker": speaker,
                "speaker_seconds": totals,
                "regions": regions,
                "elapsed": time.perf_counter() - started,
            },
            indent=2,
        ),
    )
    logging.info("Saved %s (%.1f minutes)", output, (time.perf_counter() - started) / 60)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe the longest-speaking lecturer to SRT using CUDA."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--language", default="ru")
    parser.add_argument("--batch-size", type=int, default=1, help="ASR CUDA batch size (default: 1)")
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=120,
        help="Maximum ASR audio duration including overlap (default: 120)",
    )
    parser.add_argument(
        "--min-chunk-seconds",
        type=float,
        default=60,
        help="Strict minimum core duration without overlap (default: 60)",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".lecture-cache"))
    parser.add_argument("--limit-seconds", type=float, help="Process only a prefix for a smoke test")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not math.isfinite(args.chunk_seconds) or args.chunk_seconds <= 2:
        parser.error("--chunk-seconds must be finite and greater than 2")
    try:
        packing_size(1, args.min_chunk_seconds, args.chunk_seconds - 2)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit_seconds is not None and args.limit_seconds <= 0:
        parser.error("--limit-seconds must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    try:
        run(args)
    except (RuntimeError, ValueError, OSError) as exc:
        logging.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
