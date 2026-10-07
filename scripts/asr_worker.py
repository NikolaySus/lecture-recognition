"""Isolated CUDA worker. Requests contain audio/configuration, never references."""

import argparse
import importlib.metadata
import json
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")

    def scalar(item):
        if isinstance(item, (np.ndarray, torch.Tensor)):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Non-serializable worker output: {type(item)}")

    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=scalar), encoding="utf-8")
    tmp.replace(path)


def versions():
    result = {}
    for package in ("torch", "transformers", "nemo_toolkit", "torchaudio", "numpy"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def whisper_generated_ids(output, eos):
    """Exclude decoder prompt tokens and check EOS before Whisper strips it."""
    if not output.scores:
        raise RuntimeError("Whisper returned no generation steps")
    ids = output.sequences[0, -len(output.scores) :]
    if not (ids == eos).any():
        raise RuntimeError("Generation limit reached")
    return ids


def run(request, output):
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(0.95)
    torch.cuda.reset_peak_memory_stats()
    audio = np.memmap(request["audio"], dtype="<f4", mode="r")
    operation = request.get("operation", "asr")
    if operation == "channel-diagnostics":
        from lecture_recognition.channel_diagnostics import run_diagnostics
        return run_diagnostics(request)
    if operation != "asr":
        from lecture_recognition.models import MODEL_REVISIONS, align, diarize

        if operation == "diarize":
            return {"segments": diarize(audio, MODEL_REVISIONS["diarization"])}
        if operation == "align":
            from lecture_recognition.cli import Cache
            from lecture_recognition.timeline import merge_words

            cache = Cache(Path(request["cache"]))
            aligned = align(
                audio,
                request["transcripts"],
                "ru",
                MODEL_REVISIONS["alignment"],
                cache.read,
                cache.write,
                context_regions=[(0, len(audio) / 16000)],
            )
            words = [{k: v for k, v in w.items() if k != "chunks"} for w in merge_words(aligned)]
            return {"words": words}
        raise ValueError(f"Unknown operation: {operation}")

    cfg = request["config"]
    backend, prompt = cfg["backend"], cfg.get("prompt")
    from huggingface_hub import hf_hub_download, snapshot_download

    download_start = time.perf_counter()
    checkpoint = None
    if backend == "parakeet":
        checkpoint = hf_hub_download(cfg["repo"], "parakeet-tdt-0.6b-v3.nemo", revision=cfg["revision"])
    else:
        snapshot_download(
            cfg["repo"],
            revision=cfg["revision"],
            allow_patterns=[
                "*.json",
                "*.txt",
                "*.model",
                "*.tiktoken",
                "*.py",
                "*.jinja",
                "pytorch_model.bin" if backend == "gigaam" else "*.safetensors",
            ],
            ignore_patterns=["*fp32*"],
            max_workers=2,
            local_files_only=bool(cfg.get("offline", False)),
        )
    download_seconds = time.perf_counter() - download_start
    started = time.perf_counter()
    if backend == "qwen":
        from transformers import AutoModelForMultimodalLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            cfg["repo"], revision=cfg["revision"], local_files_only=True
        )
        model = (
            AutoModelForMultimodalLM.from_pretrained(
                cfg["repo"], revision=cfg["revision"], dtype=torch.bfloat16, local_files_only=True
            )
            .to("cuda")
            .eval()
        )
    elif backend == "whisper":
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        processor = AutoProcessor.from_pretrained(
            cfg["repo"], revision=cfg["revision"], local_files_only=True
        )
        model = (
            AutoModelForSpeechSeq2Seq.from_pretrained(
                cfg["repo"], revision=cfg["revision"], dtype=torch.bfloat16, local_files_only=True
            )
            .to("cuda")
            .eval()
        )
    elif backend == "gigaam":
        from transformers import AutoModel

        model = (
            AutoModel.from_pretrained(
                cfg["repo"],
                revision=cfg["revision"],
                code_revision=cfg["revision"],
                trust_remote_code=True,
                local_files_only=True,
            )
            .to("cuda")
            .eval()
        )
    elif backend == "parakeet":
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.parts.context_biasing.boosting_graph_batched import BoostingTreeModelConfig
        from omegaconf import OmegaConf, open_dict

        model = ASRModel.restore_from(checkpoint, map_location="cuda").float().eval()
        with open_dict(model.cfg.decoding):
            model.cfg.decoding.strategy = "greedy_batch"
            if cfg.get("alpha"):
                model.cfg.decoding.greedy.boosting_tree = OmegaConf.structured(
                    BoostingTreeModelConfig(
                        key_phrases_list=cfg["terms"], context_score=1.0, depth_scaling=2.0, source_lang="ru"
                    )
                )
                model.cfg.decoding.greedy.boosting_tree_alpha = cfg["alpha"]
        model.change_decoding_strategy(model.cfg.decoding)
    else:
        raise ValueError(backend)
    load_seconds = time.perf_counter() - started
    items = []
    chunk_dir = Path(output).parent / "raw-chunks"
    chunk_dir.mkdir(exist_ok=True)
    compute_seconds = 0.0
    cached_chunks = 0
    for index, chunk in enumerate(request["chunks"]):
        saved = chunk_dir / f"{index:05d}.json"
        if saved.exists():
            item = json.loads(saved.read_text())
            if item["chunk"] != chunk:
                raise ValueError("Cached chunk does not match request")
            items.append(item)
            cached_chunks += 1
            continue
        wave = np.array(audio[round(chunk["start"] * 16000) : round(chunk["end"] * 16000)])
        start = time.perf_counter()
        native = None
        limit_hits = 0
        with torch.inference_mode():
            if backend == "qwen":
                kwargs = {"audio": wave, "language": "ru"}
                if prompt:
                    kwargs["prompt"] = prompt
                inputs = processor.apply_transcription_request(**kwargs).to("cuda", dtype=model.dtype)
                limit = max(1024, math.ceil(len(wave) / 16000 * 12 + 256))
                result = model.generate(**inputs, max_new_tokens=limit, do_sample=False, num_beams=1)
                ids = result[0, inputs["input_ids"].shape[1] :]
                if len(ids) >= limit:
                    raise RuntimeError("Generation limit reached")
                text = (
                    processor.tokenizer.decode(ids, skip_special_tokens=True).split("<asr_text>")[-1].strip()
                )
            elif backend == "whisper":
                inputs = processor(
                    wave, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
                ).to("cuda", dtype=model.dtype)
                kwargs = dict(
                    language="ru",
                    task="transcribe",
                    do_sample=False,
                    num_beams=1,
                    return_timestamps=False,
                    max_new_tokens=440,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
                if prompt:
                    kwargs["prompt_ids"] = processor.get_prompt_ids(prompt, return_tensors="pt").to("cuda")
                    kwargs["max_new_tokens"] = min(
                        440, model.config.max_target_positions - 4 - len(kwargs["prompt_ids"])
                    )
                result = model.generate(**inputs, **kwargs)
                ids = whisper_generated_ids(result, model.generation_config.eos_token_id)
                text = processor.decode(ids, skip_special_tokens=True).strip()
            elif backend == "gigaam":
                from lecture_recognition.gigaam_decoding import decode_gigaam

                text, limit_hits = decode_gigaam(model, wave, cfg)
            else:
                hypothesis = model.transcribe([wave], batch_size=1, return_hypotheses=True, timestamps=True)[
                    0
                ]
                text = hypothesis.text
                stamps = hypothesis.timestamp
                native = stamps.get("word", []) if isinstance(stamps, dict) else None
        torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        compute_seconds += seconds
        item = {"chunk": chunk, "text": str(text), "seconds": seconds, "native_words": native, "limit_hits": limit_hits}
        write(saved, item)
        items.append(item)
        print(f"Chunk {index + 1}/{len(request['chunks'])}: {seconds:.2f}s", flush=True)
    return {
        "transcripts": items,
        "load_seconds": load_seconds,
        "download_seconds": download_seconds,
        "compute_seconds": compute_seconds,
        "measured_asr_seconds": load_seconds + compute_seconds,
        "cached_chunks": cached_chunks,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    started = time.perf_counter()
    value = {"status": "error", "versions": versions()}
    try:
        value.update(run(request, args.output), status="ok")
    except Exception as exc:
        value.update(
            status="oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        traceback.print_exc()
    value.update(
        seconds=time.perf_counter() - started,
        peak_vram_gib=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0,
    )
    write(args.output, value)
    raise SystemExit(0 if value["status"] == "ok" else 1)


if __name__ == "__main__":
    main()
