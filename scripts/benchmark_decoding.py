"""Run the fixed R001–R008 context/decoding study on CUDA, with resumable cases."""

import argparse
import hashlib
import json
import logging
import math
import platform
import time
from pathlib import Path

import numpy as np
import torch
import transformers

from lecture_recognition.audio import RATE, decode, digest
from lecture_recognition.cli import Cache, atomic_text
from lecture_recognition.decoding_experiments import (
    GROUPS,
    PROMPTS,
    REFERENCE_WINDOWS,
    combine,
    config,
    generation_kwargs,
    initial_matrix,
    lexical,
    reference_cards,
    regression,
    score_case,
)
from lecture_recognition.experiments import identity, preprocess
from lecture_recognition.models import MODEL_REVISIONS, align, align_wave, load, release
from lecture_recognition.timeline import (
    Chunk,
    lecturer_regions,
    make_packed_chunks,
    merge_words,
    student_only_regions,
    to_srt,
)


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


def transcribe(audio, chunks, cfg, cache):
    processor, model = load("asr", MODEL_REVISIONS["asr"])
    result = []
    try:
        kwargs = generation_kwargs(cfg, processor.tokenizer)
        for offset in range(0, len(chunks), cfg["batch"]):
            group = chunks[offset : offset + cfg["batch"]]
            key = {"offset": offset, "chunks": [c.dict() for c in group]}
            saved = cache.read("asr_batch", key)
            if saved is not None:
                result.extend(saved)
                continue
            inputs = output = None
            try:
                waves = [np.asarray(audio[round(c.start * RATE) : round(c.end * RATE)]) for c in group]
                inputs = processor.apply_transcription_request(
                    audio=waves,
                    language=["ru"] * len(group),
                    prompt=PROMPTS[cfg["prompt"]],
                ).to("cuda", dtype=model.dtype)
                limit = max(1024, math.ceil(max(map(len, waves)) / RATE * 12 + 256))
                with torch.inference_mode():
                    output = model.generate(**inputs, max_new_tokens=limit, **kwargs)
                rows = output[:, inputs["input_ids"].shape[1] :].cpu().tolist()
                eos = model.generation_config.eos_token_id
                eos = {eos} if isinstance(eos, int) else set(eos or [])
                saved = []
                for c, ids in zip(group, rows):
                    finish = next((i for i, token in enumerate(ids) if token in eos), None)
                    if finish is None:
                        raise RuntimeError("token_limit: EOS not reached; no implicit splitting")
                    text = processor.tokenizer.decode(ids[:finish], skip_special_tokens=True).strip()
                    saved.append(
                        {
                            "chunk": c.dict(),
                            "text": text.split("<asr_text>", 1)[-1].strip(),
                            "generated_tokens": finish + 1,
                        }
                    )
                cache.write("asr_batch", key, saved)
                result.extend(saved)
                print(f"  chunk {offset + 1}..{offset + len(group)}/{len(chunks)}", flush=True)
            finally:
                del inputs, output
    finally:
        del processor, model
        release()
    return result


def rank(records):
    valid = [r for r in records if r["status"] == "ok" and r["config"]["prompt"] != "distractors"]
    baseline = next(r for r in valid if r["config"] == config())

    def key(r):
        s = r["score"]["total"]
        return (regression(r["score"], baseline["score"]), s["wer"], s["worst_wer"], r["seconds"], r["name"])

    return sorted(valid, key=key)


def report(root, records, selection=None):
    write_json(root / "summary.json", {"selection": selection, "cases": records})
    lines = [
        "# Контекст и декодирование: R001–R008",
        "",
        "Все восемь карточек используются для подбора. Это не независимый тест.",
        "",
        "WER считается по трём объединённым участкам; карточки ниже могут пересекаться.",
        "",
        "| Вариант | Статус | WER | CER | S/D/I | квази: 1 / 2 | Ложные квази | с | GiB |",
        "|---|---|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in records:
        if r["status"] != "ok":
            lines.append(
                f"| {r['name']} | {r['status']} | — | — | — | — | — | {r['seconds']:.1f} | {r['vram']:.2f} |"
            )
            continue
        t, target = r["score"]["total"], r["score"]["target"]
        hits = [
            "точно" if target[k]["exact"] else "раздельно" if target[k]["split"] else "нет"
            for k in ("first", "second")
        ]
        lines.append(
            f"| {r['name']} | ok | {t['wer']:.2%} | {t['cer']:.2%} | {t['S']}/{t['D']}/{t['I']} | "
            f"{' / '.join(hits)} | {t['false_quasi']} | {r['seconds']:.1f} | {r['vram']:.2f} |"
        )
    if selection:
        lines += ["", "## Выбор", "", "```json", json.dumps(selection, ensure_ascii=False, indent=2), "```"]
    for r in records:
        lines += [
            "",
            f"## {r['name']}",
            "",
            "```json",
            json.dumps(r["config"], ensure_ascii=False),
            "```",
            "",
        ]
        if r["status"] != "ok":
            lines.append(r.get("error", r["status"]))
            continue
        for name, card in r["score"]["cards"].items():
            lines += [
                f"### {name}: WER {card['wer']:.2%}",
                "",
                f"Эталон: {card['reference']}",
                "",
                f"ASR: {card['hypothesis']}",
                "",
                "Изменения: " + "; ".join(f"{e['op']} [{e['ref']}] → [{e['hyp']}]" for e in card["edits"]),
                "",
            ]
    atomic_text(root / "comparison.md", "\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--production-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(".lecture-cache/decoding-benchmark"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--extended-verification",
        action="store_true",
        help="Also verify the best safe greedy dictionary and target-recovering candidate",
    )
    parser.add_argument("--max-cases", type=int, help="Optional checkpoint after this many completed cases")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    if not torch.cuda.is_available():
        parser.error("CUDA required")
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.manual_seed(0)
    review_text = args.review.read_text(encoding="utf-8")
    cards = reference_cards(review_text)
    source_hash = digest(args.audio)
    production = Cache(args.production_cache)
    settings = json.loads((production.root / "settings.json").read_text(encoding="utf-8"))
    if (
        production.root.parent.name != source_hash
        or settings["revisions"] != MODEL_REVISIONS
        or settings["version"] != 4
        or settings["chunk_seconds"] != 120
    ):
        raise ValueError("Production cache must match source, model revisions, v4, and 120-second chunks")
    chunks = [Chunk(**c) for c in production.read("chunking", {})["chunks"][:3]]
    original = decode(args.audio, production.root.parent / "audio.f32")
    segments = production.read("diarization", {})
    speaker, regions, _ = lecturer_regions(segments, len(original) / RATE)
    excluded = student_only_regions(segments, len(original) / RATE, speaker)
    duration = chunks[-1].end
    audio = preprocess(original[: round(duration * RATE)], "none", excluded)
    regions = [(a, min(b, duration)) for a, b in regions if a < duration]
    layouts = {"production": chunks, "90s": make_packed_chunks(audio, regions, 60, 88)}
    for layout in layouts.values():
        if any(c.core_end - c.core_start < 60 - 1e-6 or c.end - c.start > 120 + 1e-6 for c in layout):
            raise ValueError("Invalid chunk bounds")
    metadata = {
        "version": 3,
        "source": source_hash,
        "references": cards,
        "reference_windows": REFERENCE_WINDOWS,
        "reference_alignment_split": {"R003+R004": 77.4},
        "audio_sha256": hashlib.sha256(audio.tobytes()).hexdigest(),
        "diarization": identity(segments),
        "layouts": {k: [c.dict() for c in v] for k, v in layouts.items()},
        "prompts": PROMPTS,
        "revisions": MODEL_REVISIONS,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "language": "ru",
        "memory_fraction": 0.95,
        "seed": 0,
    }
    root = args.output / identity(metadata)[:16]
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "metadata.json", metadata)
    if not (root / "review-snapshot.md").exists():
        atomic_text(root / "review-snapshot.md", review_text)
    print(f"RESULTS {root}", flush=True)
    reference = make_reference(audio, cards, root)
    if args.prepare_only:
        return
    records, seen = [], {}

    def run(cfg):
        key = identity(cfg)[:16]
        if key in seen:
            return seen[key]
        if args.max_cases is not None and len(records) >= args.max_cases:
            raise SystemExit("Requested checkpoint reached; rerun without --max-cases to continue")
        name = (
            f"{cfg['prompt']}-b{cfg['beams']}-lp{cfg['length_penalty']:g}-bias{cfg['bias']:g}"
            f"-{cfg['layout']}-batch{cfg['batch']}-r{cfg['repeat']}"
        )
        path = root / "cases" / f"{key}.json"
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["config"] != cfg:
                raise RuntimeError("Cache configuration mismatch")
        else:
            print(f"START {len(records) + 1} {name}", flush=True)
            cache = Cache(root / "inference" / key)
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats()
            result = {"name": name, "config": cfg, "status": "ok", "phase": "asr"}
            try:
                transcripts = cache.read("transcripts", {})
                if transcripts is None:
                    transcripts = transcribe(audio, layouts[cfg["layout"]], cfg, cache)
                    cache.write("transcripts", {}, transcripts)
                result["asr_seconds"] = time.perf_counter() - started
                result["phase"] = "alignment"
                aligned = align(
                    audio,
                    transcripts,
                    "ru",
                    MODEL_REVISIONS["alignment"],
                    cache.read,
                    cache.write,
                    context_regions=[(0, duration)],
                )
                words = merge_words(aligned)
                result.update(
                    score=score_case(words, reference),
                    transcripts=transcripts,
                    words=[{k: v for k, v in w.items() if k != "chunks"} for w in words],
                    phase="complete",
                )
                atomic_text(root / "cases" / f"{key}.srt", to_srt(words, regions))
            except torch.cuda.OutOfMemoryError as exc:
                result.update(status="oom", error=str(exc))
            except (RuntimeError, ValueError, TypeError, NotImplementedError) as exc:
                result.update(status="error", error=str(exc))
            finally:
                release()
            result.update(
                seconds=time.perf_counter() - started, vram=torch.cuda.max_memory_allocated() / 2**30
            )
            write_json(path, result)
        seen[key] = result
        records.append(result)
        report(root, records)
        print(
            json.dumps(
                {
                    "case": len(records),
                    "name": name,
                    "status": result["status"],
                    "score": result.get("score", {}).get("total"),
                    "seconds": result["seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        return result

    for cfg in initial_matrix():
        r = run(cfg)
        if cfg == config() and r["status"] != "ok":
            raise RuntimeError("Baseline failed; fix measurement before comparisons")
    ranked = rank(records)
    best_prompts = list(dict.fromkeys(r["config"]["prompt"] for r in ranked))[:2]
    write_json(root / "phase2-selection.json", {"prompts": best_prompts})
    for prompt in best_prompts:
        for beams in (2, 4):
            for penalty in (0.8, 1.2):
                run(config(prompt, beams, penalty))
    dictionary = next(
        (r["config"]["prompt"] for r in rank(records) if r["config"]["prompt"].startswith("dictionary")),
        "dictionary_target",
    )
    write_json(root / "phase3-selection.json", {"dictionary": dictionary})
    for prompt in ("empty", dictionary):
        for bias in (0.25, 0.5, 1.0):
            run(config(prompt, 2, bias=bias))
    for beams in (1, 2, 4):
        run(config("distractors", beams))
    main_records = list(records)
    baseline = next(r for r in records if r["config"] == config())
    finalists = [r for r in rank(main_records) if r["config"] != config()][:2]
    if args.extended_verification:
        eligible = [
            r
            for r in rank(main_records)
            if not regression(r["score"], baseline["score"])
            and r["score"]["total"]["wer"] < baseline["score"]["total"]["wer"]
        ]
        for candidates in (
            [
                r
                for r in eligible
                if r["config"]["prompt"].startswith("dictionary") and r["config"]["beams"] == 1
            ],
            [r for r in eligible if r["score"]["target"]["first"]["exact"]],
        ):
            if candidates and candidates[0] not in finalists:
                finalists.append(candidates[0])
        selected_names = {r["name"] for r in finalists}
        finalists = [r for r in rank(main_records) if r["name"] in selected_names]
    checks = []
    for r in [baseline] + finalists:
        for update in ({"layout": "90s"}, {"batch": 2}, {"repeat": 1}):
            checks.append(run({**r["config"], **update}))
    winner = baseline
    for candidate in finalists:
        if regression(candidate["score"], baseline["score"]):
            continue
        if candidate["score"]["total"]["wer"] >= baseline["score"]["total"]["wer"]:
            continue
        stable = True
        for update in ({"layout": "90s"}, {"batch": 2}, {"repeat": 1}):
            a = seen[identity({**candidate["config"], **update})[:16]]
            b = seen[identity({**baseline["config"], **update})[:16]]
            if (
                a["status"] != "ok"
                or b["status"] != "ok"
                or regression(a["score"], b["score"])
                or a["score"]["total"]["wer"] > b["score"]["total"]["wer"]
            ):
                stable = False
        if stable:
            winner = candidate
            break
    selection = {
        "winner": winner["name"],
        "config": winner["config"],
        "finalists": [r["name"] for r in finalists],
        "main_cases": len(main_records),
        "verification_cases": len(checks),
        "extended_verification": args.extended_verification,
        "independent_validation": "pending new reference fragments",
        "rule": "no new protected errors/false quasi; lower WER; no WER regression in robustness checks",
    }
    report(root, records, selection)
    write_json(root / "selection.json", selection)
    print("COMPLETE " + json.dumps(selection, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
