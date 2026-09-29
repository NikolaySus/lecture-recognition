"""Measure actual ASR limits; keep diagnostics separate from production SRT.

Each capacity case is one uninterrupted prefix, with no hidden chunking or OOM
fallback. Comparison cases cover the same audio using fixed, contiguous cuts.
Transcript disagreement is not WER: this script has no human reference.
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch

from lecture_recognition.audio import RATE, decode, digest
from lecture_recognition.cli import atomic_text
from lecture_recognition.models import MODEL_REVISIONS, load, release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--durations", nargs="+", type=int, default=[30, 60, 120, 180, 300, 600, 1200])
    parser.add_argument("--batches", nargs="+", type=int, default=[1])
    parser.add_argument(
        "--compare-seconds", type=int, default=180, help="Shared comparison window; 0 disables"
    )
    parser.add_argument("--compare-chunks", nargs="+", type=int, default=[30, 60, 120, 180])
    parser.add_argument("--language", default="ru")
    parser.add_argument(
        "--memory-fraction",
        type=float,
        default=0.95,
        help="CUDA allocator cap, e.g. 0.95 to prevent WDDM spilling into system RAM",
    )
    parser.add_argument("--output", type=Path, default=Path(".lecture-cache/asr-benchmark-bounded"))
    args = parser.parse_args()
    if any(n <= 0 for n in args.durations + args.batches + args.compare_chunks):
        parser.error("Durations and batch sizes must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    if args.memory_fraction is not None:
        if not 0 < args.memory_fraction <= 1:
            parser.error("--memory-fraction must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    source_hash = digest(args.audio)
    source_cache = Path(".lecture-cache") / source_hash
    source_cache.mkdir(parents=True, exist_ok=True)
    audio = decode(args.audio, source_cache / "audio.f32")
    args.output.mkdir(parents=True, exist_ok=True)
    import transformers

    metadata = {
        "source_sha256": source_hash,
        "model_revision": MODEL_REVISIONS["asr"],
        "gpu": torch.cuda.get_device_name(),
        "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "language": args.language,
        "dtype": "bfloat16",
        "sample_rate": RATE,
        "note": "Raw source audio; continuous silence is preserved. No reference transcript or WER.",
    }
    meta_path = args.output / "metadata.json"
    if args.memory_fraction is not None:
        metadata["memory_fraction"] = args.memory_fraction
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != metadata:
        raise RuntimeError("Benchmark output belongs to different input/model/settings; choose --output")
    atomic_text(meta_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    processor, model = load("asr", MODEL_REVISIONS["asr"])

    def case(start, duration, batch_size=1, save=True):
        key = f"start-{start:g}_seconds-{duration:g}_batch-{batch_size}"
        path = args.output / f"{key}.json"
        if save and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        end = start + duration
        if round(end * RATE) > len(audio):
            raise ValueError(f"Requested audio past EOF: {end}")
        wave = np.asarray(audio[round(start * RATE) : round(end * RATE)])
        # Capacity batches intentionally repeat the identical waveform, avoiding
        # padding-length confounds. Throughput includes all batch rows.
        waves = [wave] * batch_size
        token_limit = max(1024, math.ceil(duration * 12 + 256))
        inputs = generated = token_ids = None
        release()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = {
            "start": start,
            "seconds": duration,
            "batch_size": batch_size,
            "max_new_tokens": token_limit,
            "status": "ok",
        }
        try:
            inputs = processor.apply_transcription_request(audio=waves, language=[args.language] * batch_size)
            result["input_tokens"] = inputs["input_ids"].shape[1]
            result["mel_frames"] = inputs["input_features"].shape[-1]
            inputs = inputs.to("cuda", dtype=model.dtype)
            torch.cuda.synchronize()
            result["preprocess_seconds"] = time.perf_counter() - started
            generation_started = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
            torch.cuda.synchronize()
            result["generation_seconds"] = time.perf_counter() - generation_started
            token_ids = generated[:, inputs["input_ids"].shape[1] :].cpu().tolist()
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            texts, counts, finished = [], [], []
            for ids in token_ids:
                stop = next((i for i, token in enumerate(ids) if token in eos), None)
                finished.append(stop is not None)
                ids = ids[:stop] if stop is not None else ids
                counts.append(len(ids))
                text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                texts.append(text.split("<asr_text>", 1)[-1].strip())
            result.update(
                texts=texts, generated_tokens=counts, eos_reached=finished, token_limit_hit=not all(finished)
            )
        except torch.cuda.OutOfMemoryError:
            result["status"] = "oom"
        except (RuntimeError, ValueError) as exc:
            result.update(status="error", error=str(exc))
        finally:
            torch.cuda.synchronize()
            result["elapsed_seconds"] = time.perf_counter() - started
            result["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
            result["peak_reserved_gib"] = torch.cuda.max_memory_reserved() / 2**30
            result["real_time_factor"] = result["elapsed_seconds"] / (duration * batch_size)
            del inputs, generated, token_ids
            release()
        if save:
            atomic_text(path, json.dumps(result, ensure_ascii=False, indent=2))
            print(
                json.dumps({k: v for k, v in result.items() if k != "texts"}, ensure_ascii=False), flush=True
            )
        return result

    print("Warmup (10s, not measured in report)", flush=True)
    case(0, 10, save=False)
    for batch_size in args.batches:
        for seconds in args.durations:
            result = case(0, seconds, batch_size)
            if result["status"] != "ok":
                print(
                    f"Stopping larger inputs for batch {batch_size} after {seconds}s {result['status']}",
                    flush=True,
                )
                break
    for size in args.compare_chunks if args.compare_seconds > 0 else []:
        parts = []
        for start in range(0, args.compare_seconds, size):
            parts.append(case(start, min(size, args.compare_seconds - start)))
        combined = {
            "window_seconds": args.compare_seconds,
            "chunk_seconds": size,
            "status": "ok" if all(p["status"] == "ok" for p in parts) else "incomplete",
            "elapsed_seconds": sum(p["elapsed_seconds"] for p in parts),
            "peak_allocated_gib": max(p["peak_allocated_gib"] for p in parts),
            "text": "\n\n".join(p.get("texts", [""])[0] for p in parts),
        }
        atomic_text(
            args.output / f"comparison-{args.compare_seconds}s-chunks-{size}s.json",
            json.dumps(combined, ensure_ascii=False, indent=2),
        )
    model = processor = None
    release()


if __name__ == "__main__":
    main()
