"""Resumable multi-model diagnostic experiment, separate from the production CLI."""

import argparse
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

from .audio import RATE, decode, digest, masked_audio
from .cli import atomic_text
from .decoding_experiments import TERMS
from .evaluation import reference_cards, score_case, term_counts
from .experiments import identity
from .timeline import (
    Chunk,
    lecturer_regions,
    make_packed_chunks,
    packing_size,
    quiet_cut,
    student_only_regions,
    to_srt,
)

ROOT = Path(__file__).resolve().parents[2]
VERSION = 1
MODELS = {
    "qwen": {
        "backend": "qwen",
        "repo": "Qwen/Qwen3-ASR-1.7B-hf",
        "revision": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
        "maximum": 120,
        "dtype": "bfloat16",
    },
    "gigaam-rnnt": {
        "backend": "gigaam",
        "repo": "ai-sage/GigaAM-v3",
        "revision": "c7f128b8accdd9624df905e5c2d7b7a48c27c0d8",
        "maximum": 20,
        "dtype": "float32",
    },
    "gigaam-ctc": {
        "backend": "gigaam",
        "repo": "ai-sage/GigaAM-v3",
        "revision": "15ef3b5a88da78f93134b3cb7f015c70aefa8946",
        "maximum": 20,
        "dtype": "float32",
    },
    "whisper": {
        "backend": "whisper",
        "repo": "openai/whisper-large-v3",
        "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1",
        "maximum": 30,
        "dtype": "bfloat16",
    },
    "parakeet": {
        "backend": "parakeet",
        "repo": "nvidia/parakeet-tdt-0.6b-v3",
        "revision": "541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        "maximum": 60,
        "dtype": "float32",
    },
}
DICTIONARIES = {
    "dictionary": TERMS,
    "target": TERMS + ["квазистационарность", "квазистационарности"],
    "control": ["автокорреляция", "периодограмма", "преобразование Фурье"],
}


def write(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def child_env():
    env = os.environ.copy()
    # httpx rejects the desktop's non-standard "socks://" scheme. Only remove
    # this invalid optional proxy, preserving valid HTTP/SOCKS5 configurations.
    for key in ("ALL_PROXY", "all_proxy"):
        if env.get(key, "").startswith("socks://"):
            env.pop(key)
    env.setdefault("HF_HUB_DISABLE_XET", "1")
    return env


def worker(request, directory, backend="qwen", retry=False):
    directory.mkdir(parents=True, exist_ok=True)
    request_path, output = directory / "request.json", directory / "result.json"
    if request_path.exists() and read(request_path) != request:
        raise ValueError(f"Request changed in existing worker directory: {directory}")
    write(request_path, request)
    if output.exists() and (read(output)["status"] == "ok" or not retry):
        return read(output)
    command = [sys.executable]
    if backend in {"gigaam", "parakeet"}:
        environment = ROOT / "experiments" / backend
        interpreter = environment / ".venv/bin/python"
        command = ([str(interpreter)] if interpreter.exists() else
                   ["uv", "run", "--locked", "--project", str(environment), "python"])
    command += [str(ROOT / "scripts/asr_worker.py"), str(request_path), str(output)]
    print("WORKER", backend, directory, flush=True)
    with (directory / "worker.log").open("a", encoding="utf-8") as log:
        code = subprocess.run(
            command, cwd=ROOT, env=child_env(), stdout=log, stderr=subprocess.STDOUT
        ).returncode
    if not output.exists():
        write(output, {"status": "error", "error": f"Worker exited {code}; see worker.log"})
    return read(output)


def layout(audio, duration, maximum, regions, shift=0):
    """Bounded cores on a shared timeline; shifted layouts keep maximum input length."""
    core = maximum - 2
    cuts, cursor = [0.0], 0.0
    while duration - cursor > core:
        upper = min(duration, cursor + core)
        if cursor == 0 and shift:
            upper = max(2.0, upper - shift)
        cut = quiet_cut(audio, max(cursor + 1, upper - min(4, core / 4)), upper)
        cuts.append(cut)
        cursor = cut
    cuts.append(duration)
    return [
        Chunk(max(0, a - 1), min(duration, b + 1), a, b).dict()
        for a, b in zip(cuts, cuts[1:])
        if any(x < b and y > a for x, y in regions)
    ]


def shifted(chunks, duration, maximum, minimum):
    """Move interior ownership boundaries +5s without breaking input/core constraints."""
    # Qwen packing covers a continuous prefix in this study. A new regular layout
    # is not substituted: each existing cut is shifted and feasibility-clamped.
    cuts = [chunks[0]["core_start"]] + [c["core_end"] for c in chunks]
    old = list(cuts)
    for i in range(1, len(cuts) - 1):
        lo = max(cuts[i - 1] + minimum, old[i + 1] - (maximum - 2))
        hi = min(cuts[i - 1] + maximum - 2, old[i + 1] - minimum)
        cuts[i] = max(lo, min(hi, old[i] + 5))
    return [Chunk(max(0, a - 1), min(duration, b + 1), a, b).dict() for a, b in zip(cuts, cuts[1:])]


def extra_metrics(score):
    tp = expected = actual = critical = 0
    excess = 0
    number_errors = negation_errors = 0
    for group in score["groups"].values():
        ref, hyp = term_counts(group["reference"]), term_counts(group["hypothesis"])
        tp += sum(min(ref[t], hyp[t]) for t in ref)
        expected += sum(ref.values())
        actual += sum(hyp.values())
        excess += sum(max(0, hyp[t] - ref[t]) for t in ref)
        critical += len(set(group["protected"]) - set(group["protected_correct"]))
        for edit in group["edits"]:
            tokens = (edit["ref"], edit["hyp"])
            number_errors += int(any(t.isdigit() for t in tokens))
            negation_errors += int(any(t == "не" or t.startswith("нестационар") for t in tokens))
    score["total"].update(
        term_precision=tp / actual if actual else None,
        term_recall=tp / expected if expected else None,
        excess_terms=excess,
        critical_errors=critical,
        number_errors=number_errors,
        negation_errors=negation_errors,
    )
    return score


def rank(record):
    total = record["score"]["total"]
    return total["wer"], total["critical_errors"], record.get("asr_seconds") or float("inf"), record["id"]


def safe(candidate, baseline, improve=True):
    if candidate.get("status") != "ok" or baseline.get("status") != "ok":
        return False
    a, b = candidate["score"], baseline["score"]
    if a["total"]["wer"] > b["total"]["wer"] or (improve and a["total"]["wer"] == b["total"]["wer"]):
        return False
    for name, group in b["groups"].items():
        other = a["groups"][name]
        if not set(group["protected_correct"]) <= set(other["protected_correct"]):
            return False
        if any(other["excess_terms"][t] > value for t, value in group["excess_terms"].items()):
            return False
    return True


def validate_srt(text, duration, excluded):
    def seconds(value):
        h, m, s, ms = map(int, re.split("[:,]", value))
        return h * 3600 + m * 60 + s + ms / 1000

    previous, count = 0.0, 0
    for a, b in re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", text):
        start, end = seconds(a), seconds(b)
        if not previous <= start < end <= duration + 0.001 or end - start > 7.001:
            raise ValueError("Invalid SRT timeline")
        if any(start < right - 0.001 and end > left + 0.001 for left, right in excluded):
            raise ValueError("SRT overlaps excluded speaker-only speech")
        previous, count = end, count + 1
    if not count:
        raise ValueError("Empty SRT")
    return {"blocks": count, "last_end": previous, "timeline_valid": True}


class Benchmark:
    def __init__(self, args):
        import torch

        self.args = args
        self.config = read(args.config) if args.config else {"models": MODELS, "dictionaries": DICTIONARIES}
        self.names = args.models or list(self.config["models"])
        for name in self.names:
            if name not in self.config["models"]:
                raise ValueError(f"Unknown model: {name}")
        self.source_hash = digest(args.audio)
        self.snapshot = args.review.read_text(encoding="utf-8")
        self.cards = reference_cards(self.snapshot)
        self.locks = {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                ROOT / "uv.lock",
                ROOT / "experiments/gigaam/uv.lock",
                ROOT / "experiments/parakeet/uv.lock",
            ]
            if p.exists()
        }
        settings = {
            "version": VERSION,
            "audio_sha256": self.source_hash,
            "review_sha256": digest(args.review),
            "config": self.config,
            "locks": self.locks,
            "split": "diagnostic-dev",
            "seed": 0,
            "batch_size": 1,
            "selected_models": sorted(set(self.names) | {"qwen"}),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "torch": torch.__version__,
            "implementation": {
                str(p.relative_to(ROOT)): digest(p)
                for p in [
                    Path(__file__),
                    ROOT / "scripts/asr_worker.py",
                    Path(__file__).with_name("evaluation.py"),
                    Path(__file__).with_name("reference.py"),
                    Path(__file__).with_name("models.py"),
                    Path(__file__).with_name("timeline.py"),
                ]
            },
        }
        self.root = args.output.resolve() / identity(settings)[:16]
        self.root.mkdir(parents=True, exist_ok=True)
        if not (self.root / "metadata.json").exists():
            write(
                self.root / "metadata.json",
                {**settings, "python": platform.python_version(), "platform": platform.platform()},
            )
        with (self.root / "commands.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(sys.argv, ensure_ascii=False) + "\n")
        atomic_text(self.root / "review-snapshot.md", self.snapshot)
        write(self.root / "config.json", self.config)
        self.prepared = read(self.root / "prepared.json") if (self.root / "prepared.json").exists() else None
        print("RESULTS", self.root, flush=True)

    def prepare(self):
        if self.prepared:
            if digest(self.root / "reference.json") != self.prepared["reference_sha256"]:
                raise ValueError("Frozen reference was modified; use a new experiment directory")
            return
        from .models import MODEL_REVISIONS
        from .reference import make_reference

        source = self.args.output.resolve() / "audio" / self.source_hash
        source.mkdir(parents=True, exist_ok=True)
        raw = decode(self.args.audio, source / "audio.f32")
        diar = worker(
            {"operation": "diarize", "audio": str(source / "audio.f32")},
            source
            / (
                "diarization-"
                + identity(
                    {
                        "revision": MODEL_REVISIONS["diarization"],
                        "lock": self.locks["uv.lock"],
                        "code": digest(Path(__file__).with_name("models.py")),
                    }
                )[:16]
            ),
            retry=self.args.retry_failed,
        )
        if diar["status"] != "ok":
            raise RuntimeError(f"Diarization failed: {diar.get('error')}")
        duration = len(raw) / RATE
        speaker, regions, totals = lecturer_regions(diar["segments"], duration)
        if speaker is None:
            raise ValueError("No lecturer speech found")
        excluded = student_only_regions(diar["segments"], duration, speaker)
        padded, *_ = packing_size(len(raw), 60, 118)
        mask_path = self.root / "masked.f32"
        audio = masked_audio(raw, mask_path, excluded, padded)
        chunks = [c.dict() for c in make_packed_chunks(audio, regions, 60, 118)]
        prefix = chunks[:3]
        endpoint = prefix[-1]["end"]
        make_reference(audio, self.cards, self.root)
        self.prepared = {
            "audio": str(mask_path),
            "duration": duration,
            "padded_duration": len(audio) / RATE,
            "speaker": speaker,
            "speaker_seconds": totals,
            "regions": regions,
            "excluded": excluded,
            "qwen_chunks": chunks,
            "prefix_chunks": prefix,
            "prefix_end": endpoint,
            "reference_sha256": digest(self.root / "reference.json"),
            "masked_sha256": digest(mask_path),
            "revisions": MODEL_REVISIONS,
        }
        write(self.root / "prepared.json", self.prepared)

    def chunks(self, name, mode, shift):
        cfg = self.config["models"][name]
        audio = np.memmap(self.prepared["audio"], dtype="<f4", mode="r")
        if mode == "smoke":
            return [Chunk(0, 20, 0, 20).dict()]
        duration = self.prepared["padded_duration"] if mode == "full" else self.prepared["prefix_end"]
        if cfg["backend"] == "qwen":
            chunks = self.prepared["qwen_chunks" if mode == "full" else "prefix_chunks"]
            return shifted(chunks, duration, 120, 60) if shift else chunks
        return layout(audio, duration, cfg["maximum"], self.prepared["regions"], shift)

    def case(self, name, dictionary=None, alpha=0, mode="compare", repeat=0, shift=0):
        cfg = dict(self.config["models"][name])
        cfg.update(model=name, dictionary=dictionary, alpha=alpha)
        if dictionary:
            cfg["terms"] = self.config["dictionaries"][dictionary]
            if cfg["backend"] == "parakeet":
                cfg["terms"] = sorted(
                    set(cfg["terms"]) | {term[0].upper() + term[1:] for term in cfg["terms"]}
                )
            if cfg["backend"] in {"qwen", "whisper"}:
                cfg["prompt"] = "Термины: " + "; ".join(cfg["terms"]) + "."
        chunks = self.chunks(name, mode, shift)
        settings = {
            "config": cfg,
            "chunks": chunks,
            "repeat": repeat,
            "shift": shift,
            "mode": mode,
            "worker_sha256": digest(ROOT / "scripts/asr_worker.py"),
            "evaluation_sha256": digest(Path(__file__).with_name("evaluation.py")),
            "reference_sha256": self.prepared["reference_sha256"],
        }
        key = identity(settings)[:16]
        directory = self.root / "cases" / key
        directory.mkdir(parents=True, exist_ok=True)
        result_file = directory / "case.json"
        if result_file.exists() and (read(result_file)["status"] == "ok" or not self.args.retry_failed):
            return read(result_file)
        record = {"id": key, **settings, "status": "error", "phase": "asr"}
        write(directory / "settings.json", settings)
        request = {"operation": "asr", "audio": self.prepared["audio"], "config": cfg, "chunks": chunks}
        asr = worker(request, directory / "asr", cfg["backend"], self.args.retry_failed)
        record.update(
            status=asr["status"],
            asr_seconds=None if asr.get("cached_chunks") else asr.get("measured_asr_seconds"),
            cached_chunks=asr.get("cached_chunks", 0),
            peak_vram_gib=asr.get("peak_vram_gib"),
            error=asr.get("error"),
        )
        if asr["status"] == "ok":
            record["phase"] = "alignment"
            transcripts = [{"chunk": t["chunk"], "text": t["text"]} for t in asr["transcripts"]]
            aligned = worker(
                {
                    "operation": "align",
                    "audio": self.prepared["audio"],
                    "transcripts": transcripts,
                    "cache": str(directory / "alignment-cache"),
                },
                directory / "alignment",
                retry=self.args.retry_failed,
            )
            record.update(
                status=aligned["status"], error=aligned.get("error"), alignment_seconds=aligned.get("seconds")
            )
            if aligned["status"] == "ok":
                words = aligned["words"]
                try:
                    text = to_srt(words, self.prepared["regions"])
                    record["validation"] = validate_srt(
                        text, self.prepared["duration"], self.prepared["excluded"]
                    )
                    atomic_text(directory / "transcript.srt", text)
                    if mode not in {"smoke", "full"}:
                        record["score"] = extra_metrics(score_case(words, read(self.root / "reference.json")))
                    record["phase"] = "complete"
                except (ValueError, RuntimeError) as exc:
                    record.update(status="error", phase="evaluation", error=str(exc))
        write(result_file, record)
        print(
            "CASE",
            name,
            dictionary,
            mode,
            record["status"],
            record.get("score", {}).get("total", {}).get("wer"),
            flush=True,
        )
        self.report()
        return record

    def baseline(self):
        baseline = self.case("qwen")
        if baseline["status"] != "ok":
            raise RuntimeError("Qwen baseline failed; inspect saved ASR/alignment logs")
        self.case("qwen", "dictionary")

    def compare(self):
        self.baseline()
        for name in self.names:
            if name != "qwen":
                self.case(name)

    def records(self):
        return [read(p) for p in sorted((self.root / "cases").glob("*/case.json"))]

    def bias(self):
        self.compare()
        bases = {
            r["config"]["model"]: r
            for r in self.records()
            if r["status"] == "ok"
            and r["mode"] == "compare"
            and not r["config"]["dictionary"]
            and not r["repeat"]
            and not r["shift"]
        }
        names = [
            r["config"]["model"]
            for r in sorted(bases.values(), key=rank)
            if r["config"]["backend"] in {"qwen", "whisper", "parakeet"}
        ][:2]
        write(self.root / "bias-selection.json", {"models": names})
        accepted = []
        for name in names:
            alphas = [0.5, 1, 2] if self.config["models"][name]["backend"] == "parakeet" else [0]
            cases = [self.case(name, d, a) for d in ("dictionary", "target") for a in alphas]
            successful = sorted([r for r in cases if r["status"] == "ok"], key=rank)
            if not successful:
                continue
            control = self.case(name, "control", successful[0]["config"]["alpha"])
            eligible = sorted([r for r in cases if safe(r, bases[name])], key=rank)
            if not eligible:
                continue
            best = eligible[0]
            cfg = best["config"]
            if cfg["alpha"] != control["config"]["alpha"]:
                control = self.case(name, "control", cfg["alpha"])
            repeat = self.case(name, cfg["dictionary"], cfg["alpha"], repeat=1)
            shifted_base = self.case(name, shift=5)
            shifted_best = self.case(name, cfg["dictionary"], cfg["alpha"], shift=5)
            control_ok = control["status"] == "ok" and all(
                control["score"]["groups"][g]["excess_terms"][t]
                <= bases[name]["score"]["groups"][g]["excess_terms"][t]
                for g in bases[name]["score"]["groups"]
                for t in ("autocorrelation", "periodogram", "fourier")
            )
            if control_ok and safe(repeat, bases[name]) and safe(shifted_best, shifted_base, improve=False):
                accepted.append(best)
        baseline = bases["qwen"]
        candidates = [r for r in bases.values() if rank(r)[0] < rank(baseline)[0]] + accepted
        selected = min(candidates, key=rank) if candidates else baseline
        write(
            self.root / "selection.json",
            {
                "case_id": selected["id"],
                "config": selected["config"],
                "accepted_bias": [r["id"] for r in accepted],
                "diagnostic_only": True,
            },
        )
        self.report()

    def full(self):
        selection = self.root / "selection.json"
        if not selection.exists():
            self.bias()
        cfg = read(selection)["config"]
        result = self.case(cfg["model"], cfg["dictionary"], cfg["alpha"], mode="full")
        if result["status"] != "ok":
            raise RuntimeError("Full transcription failed; resumable artifacts saved")
        self.review_clips(result)
        self.report()

    def review_clips(self, record):
        import soundfile as sf

        audio = decode(self.args.audio, self.args.output.resolve() / "audio" / self.source_hash / "audio.f32")
        duration = self.prepared["duration"]
        centers = [15, duration / 2, duration - 15]
        centers += [c["core_end"] for c in record["chunks"][:-1]][:: max(1, len(record["chunks"]) // 6)]
        directory = self.root / "listening"
        directory.mkdir(exist_ok=True)
        manifest = []
        for i, center in enumerate(centers):
            a, b = max(0, center - 10), min(duration, center + 10)
            path = directory / f"{i:02d}.wav"
            sf.write(path, audio[round(a * RATE) : round(b * RATE)], RATE, subtype="FLOAT")
            manifest.append({"file": path.name, "start": a, "end": b})
        write(directory / "manifest.json", manifest)

    def report(self):
        records = self.records()
        write(self.root / "summary.json", records)
        lines = [
            "# Сравнение ASR одного лектора",
            "",
            "Диагностическая выборка R001–R008; независимого test нет. Сравниваются модель + нарезка.",
            "",
            "| Модель / словарь | Режим | Статус / этап | WER | S/D/I | Критические | ASR, с | SRT |",
            "|---|---|---|---:|---|---:|---:|---|",
        ]
        for r in records:
            score = r.get("score", {}).get("total", {})
            wer = f"{score['wer']:.2%}" if score else "—"
            edits = "/".join(str(score[k]) for k in ("S", "D", "I")) if score else "—"
            cfg = r["config"]
            label = f"{cfg['model']} / {cfg['dictionary'] or 'без словаря'} α={cfg['alpha']}"
            mode = f"{r['mode']}; shift={r['shift']}; repeat={r['repeat']}"
            srt = f"[SRT](cases/{r['id']}/transcript.srt)" if r["status"] == "ok" else "—"
            seconds = r.get("asr_seconds")
            lines.append(
                f"| {label} | {mode} | {r['status']}/{r['phase']} | {wer} | {edits} | "
                f"{score.get('critical_errors', '—')} | {seconds if seconds is not None else '—'} | {srt} |"
            )
        bases = [
            r
            for r in records
            if r.get("score")
            and r["status"] == "ok"
            and r["mode"] == "compare"
            and not r["config"]["dictionary"]
            and not r["repeat"]
            and not r["shift"]
        ]
        if bases:
            errors = sum(
                min(r["score"]["groups"][g]["errors"] for r in bases) for g in bases[0]["score"]["groups"]
            )
            count = bases[0]["score"]["total"]["words"]
            lines += [
                "",
                f"Oracle выбора модели на каждом целом участке: {errors}/{count} = {errors / count:.2%}.",
                "Это выбор с использованием эталона, не автоматический результат.",
                "",
                "## Ошибки по участкам",
            ]
            for r in sorted(bases, key=rank):
                lines += ["", f"### {r['config']['model']}"]
                for name, group in r["score"]["groups"].items():
                    lines += [
                        "",
                        f"**{name}: {group['errors']} ошибок**",
                        "",
                        f"Эталон: {group['reference']}",
                        "",
                        f"Гипотеза: {group['hypothesis']}",
                    ]
        selection = self.root / "selection.json"
        if selection.exists():
            lines += [
                "",
                "## Выбор",
                "",
                "```json",
                json.dumps(read(selection), ensure_ascii=False, indent=2),
                "```",
            ]
        lines += [
            "",
            "Raw ASR и нативные времена — в asr/result.json; общее выравнивание — в alignment/result.json.",
            "Время ASR включает загрузку; сведения о частичном возобновлении — в cached_chunks.",
            "Метрики полного SRT вне размеченных участков не вычисляются.",
        ]
        atomic_text(self.root / "comparison.md", "\n".join(lines) + "\n")


def main():
    for key in ("ALL_PROXY", "all_proxy"):
        if os.environ.get(key, "").startswith("socks://"):
            os.environ.pop(key)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=["prepare", "smoke", "baseline", "compare", "bias", "full", "report", "all"]
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".lecture-cache/model-benchmark"))
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    benchmark = Benchmark(args)
    if args.stage == "report":
        benchmark.report()
        return
    benchmark.prepare()
    if args.stage in {"smoke", "all"}:
        for name in benchmark.names:
            benchmark.case(name, mode="smoke")
    if args.stage == "baseline":
        benchmark.baseline()
    if args.stage == "compare":
        benchmark.compare()
    if args.stage in {"bias", "all"}:
        benchmark.bias()
    if args.stage in {"full", "all"}:
        benchmark.full()
