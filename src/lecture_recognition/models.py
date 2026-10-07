"""CUDA adapters. Model lifetimes are deliberately limited to one stage."""

import gc
import logging
import math

import numpy as np
import torch
from tqdm import tqdm
from transformers import (
    AutoModelForAudioFrameClassification,
    AutoModelForMultimodalLM,
    AutoModelForTokenClassification,
    AutoProcessor,
)

from .audio import RATE
from .timeline import Chunk, merge_words, split_chunk

MODEL_IDS = {
    "diarization": "nvidia/Nemotron-3-Diarization",
    "asr": "Qwen/Qwen3-ASR-1.7B-hf",
    "alignment": "Qwen/Qwen3-ForcedAligner-0.6B-hf",
}
MODEL_REVISIONS = {
    "diarization": "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
    "asr": "bcd2b5b7f32b480ab5790554cfa8347f246a14f3",
    "alignment": "c07281df297b9905d24a508279258cccf987a064",
}
log = logging.getLogger(__name__)


def release():
    gc.collect()
    torch.cuda.empty_cache()


def load(kind, revision):
    cls = {
        "diarization": AutoModelForAudioFrameClassification,
        "asr": AutoModelForMultimodalLM,
        "alignment": AutoModelForTokenClassification,
    }[kind]
    name = MODEL_IDS[kind]
    dtype = torch.float32 if kind == "diarization" else torch.bfloat16
    for local_only in (True, False):
        try:
            processor = AutoProcessor.from_pretrained(name, revision=revision, local_files_only=local_only)
            model = cls.from_pretrained(name, revision=revision, dtype=dtype, local_files_only=local_only).to("cuda").eval()
            return processor, model
        except OSError:
            if not local_only:
                raise
            log.info("Missing cached %s files; downloading the pinned revision", kind)
    raise RuntimeError("Model loading failed")


def diarization_inputs(processor, audio):
    """Processor accounts for STFT context as well as right look-ahead."""
    size = processor.num_samples_first_audio_chunk
    if len(audio) <= size:
        yield (
            processor(
                np.asarray(audio),
                sampling_rate=RATE,
                is_streaming=True,
                is_first_audio_chunk=True,
                is_last_audio_chunk=True,
            ),
            0,
        )
        return
    yield (
        processor(np.asarray(audio[:size]), sampling_rate=RATE, is_streaming=True, is_first_audio_chunk=True),
        0,
    )
    frame = processor.num_mel_frames_per_step
    while True:
        start = processor.audio_chunk_start(frame)
        end = start + processor.num_samples_per_audio_chunk
        last = end >= len(audio)
        wave = np.asarray(audio[start : min(end, len(audio))])
        if last:
            # Offline centered STFT pads the recording's right edge. Reproduce
            # exactly floor(total_samples / hop) valid frames with center=False.
            pad = processor.feature_extractor.n_fft // 2 - processor.feature_extractor.hop_length
            wave = np.pad(wave, (0, pad))
        yield (
            processor(
                wave,
                sampling_rate=RATE,
                is_streaming=True,
                is_first_audio_chunk=False,
                is_last_audio_chunk=last,
            ),
            frame,
        )
        if last:
            break
        frame += processor.num_mel_frames_per_step


def configure_diarization(processor, model):
    processor.streaming_modes["lecture_offline"] = (340, 40)
    processor.set_streaming_mode("lecture_offline")
    model.config.streaming_config.fifo_length = 40
    model.config.streaming_config.speaker_cache_update_period = 300
    model.config.streaming_config.speaker_cache_length = 264


def diarize(audio, revision):
    processor, model = load("diarization", revision)
    configure_diarization(processor, model)
    cache, logits = None, []
    try:
        with torch.inference_mode(), tqdm(total=len(audio) / RATE, unit="s", desc="Diarization") as bar:
            for inputs, frame in diarization_inputs(processor, audio):
                inputs = inputs.to("cuda")
                # Explicit cache also selects streaming for a single final chunk.
                if cache is None and "num_lookahead_frames" not in inputs:
                    inputs["num_lookahead_frames"] = 0
                output = model(**inputs, speaker_cache=cache)
                cache = output.speaker_cache
                logits.append(output.logits.cpu())
                bar.update(min(output.logits.shape[1] / 100, bar.total - bar.n))
        return processor.extract_speaker_dict(torch.cat(logits, dim=1))[0]
    finally:
        del model, processor, cache, logits
        release()


def recognize(audio, chunks, language, revision, cache_read, cache_write, batch_size=1, min_core=1):
    saved_chunks = [cache_read("asr", chunk.dict()) for chunk in chunks]
    if all(saved is not None for saved in saved_chunks):
        return [item for saved in saved_chunks for item in saved]
    processor, model = load("asr", revision)

    def one(chunk):
        saved = cache_read("asr", chunk.dict())
        if saved is not None:
            return saved
        inputs = output = ids = None
        divide = False
        token_limit = max(1024, math.ceil((chunk.end - chunk.start) * 12 + 256))
        try:
            wave = np.asarray(audio[round(chunk.start * RATE) : round(chunk.end * RATE)])
            inputs = processor.apply_transcription_request(audio=wave, language=language).to("cuda")
            inputs = inputs.to(dtype=model.dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
            ids = output[:, inputs["input_ids"].shape[1] :]
            if ids.shape[1] >= token_limit:
                divide = True
            else:
                # Avoid the processor's heuristic repetition deletion: actual lecture
                # repetitions must survive. Remove only model control tokens.
                text = processor.tokenizer.decode(ids[0], skip_special_tokens=True).strip()
                if "<asr_text>" in text:
                    text = text.split("<asr_text>", 1)[1].strip()
                result = [{"chunk": chunk.dict(), "text": text}]
        except torch.cuda.OutOfMemoryError:
            divide = True
        finally:
            del inputs, output, ids
        if divide:
            release()
            log.warning("Splitting ASR chunk %.2f–%.2f (memory/token limit)", chunk.start, chunk.end)
            result = [r for child in split_chunk(audio, chunk, min_core=min_core) for r in one(child)]
        cache_write("asr", chunk.dict(), result)
        return result

    def batch(group):
        missing = [c for c in group if cache_read("asr", c.dict()) is None]
        if len(missing) <= 1:
            return [r for c in group for r in one(c)]
        inputs = output = None
        failed = False
        completed = []
        token_limit = max(1024, math.ceil(max(c.end - c.start for c in missing) * 12 + 256))
        try:
            waves = [np.asarray(audio[round(c.start * RATE) : round(c.end * RATE)]) for c in missing]
            inputs = processor.apply_transcription_request(audio=waves, language=[language] * len(waves))
            inputs = inputs.to("cuda", dtype=model.dtype)
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=token_limit, do_sample=False)
            generated = output[:, inputs["input_ids"].shape[1] :].cpu().tolist()
            eos = model.generation_config.eos_token_id
            eos = {eos} if isinstance(eos, int) else set(eos or [])
            for c, ids in zip(missing, generated):
                finish = next((i for i, token in enumerate(ids) if token in eos), None)
                if finish is None and len(ids) >= token_limit:
                    continue
                if finish is not None:
                    ids = ids[:finish]
                text = processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
                if "<asr_text>" in text:
                    text = text.split("<asr_text>", 1)[1].strip()
                completed.append((c, [{"chunk": c.dict(), "text": text}]))
        except torch.cuda.OutOfMemoryError:
            failed = True
        finally:
            del inputs, output
        if failed:
            release()
            log.warning("ASR batch exceeded CUDA memory; retrying individually")
        for c, value in completed:
            cache_write("asr", c.dict(), value)
        return [r for c in group for r in one(c)]

    try:
        result = []
        with tqdm(total=len(chunks), desc="ASR", unit="chunk") as bar:
            for start in range(0, len(chunks), batch_size):
                group = chunks[start : start + batch_size]
                result.extend(batch(group) if batch_size > 1 else one(group[0]))
                bar.update(len(group))
        return result
    finally:
        model = processor = None
        release()


def align_wave(processor, model, wave, text, language):
    """Retry quantized, degenerate timings with silence; never invent a word span."""
    duration = len(wave) / RATE
    for pad in (0.0, 0.25, 0.5, 1.0):
        padded = np.pad(wave, (round(pad * RATE), round(pad * RATE))) if pad else wave
        inputs, lists = processor.prepare_forced_aligner_inputs(
            audio=padded, transcript=text, language=language
        )
        inputs = inputs.to("cuda", dtype=model.dtype)
        with torch.inference_mode():
            output = model(**inputs)
        words = processor.decode_forced_alignment(
            logits=output.logits,
            input_ids=inputs["input_ids"],
            word_lists=lists,
            timestamp_token_id=model.config.timestamp_token_id,
        )[0]
        del inputs, output
        for w in words:
            w["start_time"] = round(w["start_time"] - pad, 3)
            w["end_time"] = round(w["end_time"] - pad, 3)
        valid = bool(words) and all(
            np.isfinite([w["start_time"], w["end_time"]]).all()
            and -0.1 <= w["start_time"] <= w["end_time"] <= duration + 0.2
            # A word entirely beyond the audio cannot be repaired by clipping
            # only its end: that would create a negative duration.
            and w["start_time"] < duration
            and w["end_time"] >= 0
            for w in words
        )
        if valid:
            for w in words:
                w["start_time"] = max(0.0, w["start_time"])
                w["end_time"] = min(duration, w["end_time"])
            if any(w["end_time"] > w["start_time"] for w in words):
                if pad:
                    log.info("Recovered short-utterance alignment with %.2fs silence padding", pad)
                return words
    raise RuntimeError("Forced aligner could not measure valid timestamps after silence-padding retries")


def align_with_context(processor, model, audio, chunk, text, language, regions):
    """Retry with real neighboring audio without crossing excluded speakers.

    Core ownership stays unchanged, so added context cannot duplicate subtitles.
    """

    def attempt(c):
        wave = np.asarray(audio[round(c.start * RATE) : round(c.end * RATE)])
        return align_wave(processor, model, wave, text, language)

    try:
        return chunk, attempt(chunk)
    except RuntimeError:
        bounds = next(((a, b) for a, b in regions if a <= chunk.start and chunk.end <= b), None)
        if bounds is None:
            raise
        expanded = Chunk(
            max(bounds[0], chunk.start - 2),
            min(bounds[1], chunk.end + 2),
            chunk.core_start,
            chunk.core_end,
        )
        if expanded == chunk:
            raise
        log.warning(
            "Retrying alignment %.2f–%.2f with real audio context %.2f–%.2f",
            chunk.start,
            chunk.end,
            expanded.start,
            expanded.end,
        )
        return expanded, attempt(expanded)


def align(audio, transcripts, language, revision, cache_read, cache_write, context_regions=()):
    saved_chunks = []
    for item in transcripts:
        saved = cache_read("alignment", item)
        if saved is not None:
            try:
                merge_words([saved])
            except RuntimeError:
                log.warning("Recomputing invalid cached alignment at %.2fs", item["chunk"]["start"])
                saved = None
        saved_chunks.append(saved)
    if all(saved is not None for saved in saved_chunks):
        return saved_chunks
    processor, model = load("alignment", revision)
    result = []
    try:
        for item, saved in tqdm(
            zip(transcripts, saved_chunks), total=len(transcripts), desc="Alignment", unit="chunk"
        ):
            if saved is not None:
                result.append(saved)
                continue
            c = Chunk(**item["chunk"])
            words = []
            if item["text"].strip():
                try:
                    c, words = align_with_context(
                        processor, model, audio, c, item["text"], language, context_regions
                    )
                except RuntimeError as exc:
                    raise RuntimeError(f"Alignment failed at {c.start:.2f}s: {exc}") from exc
                # Russian/Latin whitespace tokens retain original punctuation.
                original = item["text"].split()
                if len(original) == len(words):
                    for w, text in zip(words, original):
                        w["text"] = text
                # Quantized timestamps occasionally give a word zero duration.
                # Attach it to an adjacent measured span, preserving its text.
                positive, pending = [], []
                for w in words:
                    if w["end_time"] == w["start_time"]:
                        pending.append(w["text"])
                    else:
                        w["text"] = " ".join(pending + [w["text"]])
                        pending = []
                        positive.append(w)
                if pending and positive:
                    positive[-1]["text"] += " " + " ".join(pending)
                if not positive:
                    raise RuntimeError(f"No valid alignment at {c.start:.2f}s")
                words = positive
            saved = {"chunk": c.dict(), "words": words}
            # Apply the same final validation before persisting resumable work.
            merge_words([saved])
            cache_write("alignment", item, saved)
            result.append(saved)
        return result
    finally:
        del model, processor
        release()
