"""Pure timeline operations, independent of model execution."""

import re
from dataclasses import asdict, dataclass

import numpy as np

from .audio import RATE


def runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def speaker_masks(segments, duration):
    """10 ms masks match Nemotron resolution and count union durations."""
    n = int(np.ceil(duration * 100))
    masks = np.zeros((8, n), dtype=bool)
    for s in segments:
        a, b = max(0, round(s["Start"] * 100)), min(n, round(s["End"] * 100))
        masks[int(s["Speaker"]), a:b] = True
    return masks


def student_only_regions(segments, duration, speaker):
    masks = speaker_masks(segments, duration)
    excluded = masks[np.arange(8) != speaker].any(axis=0) & ~masks[speaker]
    return [(a / 100, min(duration, b / 100)) for a, b in runs(excluded)]


def lecturer_regions(segments, duration, max_gap=0.8):
    masks = speaker_masks(segments, duration)
    n = masks.shape[1]
    totals = masks.sum(axis=1)
    if not totals.any():
        return None, [], totals.tolist()
    speaker = int(totals.argmax())
    selected = masks[speaker].copy()
    other_only = masks[np.arange(8) != speaker].any(axis=0) & ~selected
    # Extend into silence only; never leap across another speaker's turn.
    for a, b in runs(selected):
        left = a
        while left > max(0, a - 20) and not other_only[left - 1]:
            left -= 1
        right = b
        while right < min(n, b + 20) and not other_only[right]:
            right += 1
        selected[left:right] = True
    spans = runs(selected)
    merged = []
    for a, b in spans:
        if merged and a - merged[-1][1] <= max_gap * 100 and not other_only[merged[-1][1] : a].any():
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return speaker, [(a / 100, min(duration, b / 100)) for a, b in merged], (totals / 100).tolist()


@dataclass(frozen=True)
class Chunk:
    start: float
    end: float
    core_start: float
    core_end: float

    def dict(self):
        return asdict(self)


def quiet_cut(audio, lower, upper):
    a, b = round(lower * RATE), round(upper * RATE)
    # 100 ms energy windows, advanced at 10 ms; bounded memory.
    samples = np.asarray(audio[a:b], dtype=np.float64)
    width = 1600
    if len(samples) <= width:
        return (lower + upper) / 2
    sums = np.r_[0.0, np.cumsum(samples * samples)]
    offsets = np.arange(0, len(samples) - width + 1, 160)
    energy = sums[offsets + width] - sums[offsets]
    offset = int(offsets[int(energy.argmin())]) + width // 2
    return (a + offset) / RATE


def make_chunks(audio, regions, max_core=28.0):
    if not np.isfinite(max_core) or max_core <= 0:
        raise ValueError("Chunk core duration must be positive and finite")
    chunks = []
    for start, end in regions:
        cursor = start
        while end - cursor > max_core:
            cut = quiet_cut(audio, cursor + max_core - min(8, max_core / 2), cursor + max_core)
            chunks.append(Chunk(max(start, cursor - 1), min(end, cut + 1), cursor, cut))
            cursor = cut
        if end > cursor:
            chunks.append(Chunk(max(start, cursor - 1), end, cursor, end))
    return chunks


def packing_size(samples, min_core=60, max_core=118):
    if not np.isfinite([min_core, max_core]).all() or not 0 < min_core <= max_core:
        raise ValueError("Chunk minimum must be positive and no greater than maximum core duration")
    minimum, maximum = int(np.ceil(min_core * RATE)), int(np.floor(max_core * RATE))
    if minimum > maximum or minimum < 1:
        raise ValueError("Chunk limits are incompatible at sample resolution")
    count = max(1, (samples + maximum - 1) // maximum)
    return max(samples, count * minimum), count, minimum, maximum


def make_packed_chunks(audio, speech_regions, min_core=60, max_core=118):
    """Partition a padded timeline; each cut leaves a feasible remaining tail."""
    size, count, minimum, maximum = packing_size(len(audio), min_core, max_core)
    if size != len(audio):
        raise ValueError("Audio must be padded before packing")
    chunks, cursor = [], 0
    for remaining in range(count, 0, -1):
        if remaining == 1:
            cut = size
        else:
            lower = max(cursor + minimum, size - (remaining - 1) * maximum)
            upper = min(cursor + maximum, size - (remaining - 1) * minimum)
            lower = max(lower, upper - 8 * RATE)
            cut = min(upper, max(lower, round(quiet_cut(audio, lower / RATE, upper / RATE) * RATE)))
        c = Chunk(max(0, cursor - RATE) / RATE, min(size, cut + RATE) / RATE, cursor / RATE, cut / RATE)
        if any(a < c.core_end and b > c.core_start for a, b in speech_regions):
            chunks.append(c)
        cursor = cut
    return chunks


def split_chunk(audio, chunk, min_core=1):
    if chunk.core_end - chunk.core_start < 2 * min_core:
        raise RuntimeError(
            f"ASR memory/token limit: cannot split while preserving {min_core:g}s minimum core; "
            "completed chunks remain cached. Free CUDA memory or use a GPU with more memory."
        )
    if chunk.core_end - chunk.core_start <= 2:
        raise RuntimeError("Cannot process even a 2-second ASR chunk; check available CUDA memory.")
    mid = (chunk.core_start + chunk.core_end) / 2
    cut = quiet_cut(
        audio, max(chunk.core_start + min_core, mid - 0.5), min(chunk.core_end - min_core, mid + 0.5)
    )
    return [
        Chunk(chunk.start, min(chunk.end, cut + 1), chunk.core_start, cut),
        Chunk(max(chunk.start, cut - 1), chunk.end, cut, chunk.core_end),
    ]


def merge_words(aligned):
    candidates = []
    for chunk_index, item in enumerate(aligned):
        c = Chunk(**item["chunk"])
        for word in item["words"]:
            start, end = c.start + word["start_time"], c.start + word["end_time"]
            if (
                not np.isfinite([start, end]).all()
                or end <= start
                or start < c.start - 0.1
                or end > c.end + 0.2
            ):
                raise RuntimeError("Forced aligner returned invalid word timestamps.")
            mid = (start + end) / 2
            candidates.append(
                {
                    "start": max(start, c.start),
                    "end": min(end, c.end),
                    "text": word["text"].strip(),
                    "margin": min(mid - c.start, c.end - mid),
                    "owner": c.core_start <= mid < c.core_end,
                    "chunks": {chunk_index},
                }
            )
    candidates.sort(key=lambda w: (w["start"], w["end"]))
    result = []
    for w in candidates:
        if not w["text"]:
            continue

        def norm(t):
            return re.sub(r"[^\w]", "", t.casefold())

        duplicate = False
        for index in range(len(result) - 1, -1, -1):
            prev = result[index]
            if w["start"] - prev["end"] > 1:
                break
            if w["chunks"] & prev["chunks"] or norm(w["text"]) != norm(prev["text"]):
                continue
            overlap = min(prev["end"], w["end"]) - max(prev["start"], w["start"])
            if overlap > 0.5 * min(prev["end"] - prev["start"], w["end"] - w["start"]):
                chosen = dict(w if w["margin"] > prev["margin"] else prev)
                chosen["owner"] = w["owner"] or prev["owner"]
                chosen["chunks"] = w["chunks"] | prev["chunks"]
                result[index] = chosen
                duplicate = True
                break
        if not duplicate:
            result.append(w)
    # Two independently aligned copies can fall on opposite sides of a cut,
    # each just outside its own core. Keep their matched word once.
    return sorted((w for w in result if w["owner"] or len(w["chunks"]) > 1), key=lambda w: w["start"])


def stamp(seconds):
    ms = max(0, round(seconds * 1000))
    sec, ms = divmod(ms, 1000)
    minute, sec = divmod(sec, 60)
    hour, minute = divmod(minute, 60)
    return f"{hour:02}:{minute:02}:{sec:02},{ms:03}"


def to_srt(words, regions):
    import textwrap

    groups, group = [], []
    region_index = 0
    last_region = -1
    for w in words:
        mid = (w["start"] + w["end"]) / 2
        while region_index < len(regions) and mid >= regions[region_index][1]:
            region_index += 1
        if region_index == len(regions) or mid < regions[region_index][0]:
            continue
        w = dict(
            w, start=max(w["start"], regions[region_index][0]), end=min(w["end"], regions[region_index][1])
        )
        if round(w["end"] * 1000) <= round(w["start"] * 1000):
            continue
        text = " ".join(x["text"] for x in group + [w])
        if group and (
            region_index != last_region
            or w["start"] - group[-1]["end"] > 0.8
            or w["end"] - group[0]["start"] > 7
            or len(text) > 84
        ):
            groups.append(group)
            group = []
        group.append(w)
        last_region = region_index
        if re.search(r"[.!?…][\"»)]?$", w["text"]):
            groups.append(group)
            group = []
    if group:
        groups.append(group)
    blocks = []
    for i, g in enumerate(groups):
        start, end = g[0]["start"], min(g[-1]["end"], g[0]["start"] + 7)
        if i + 1 < len(groups):
            end = min(end, groups[i + 1][0]["start"])
        if round(end * 1000) <= round(start * 1000):
            raise RuntimeError("Non-monotonic aligned subtitle timestamps.")
        lines = textwrap.wrap(
            " ".join(w["text"] for w in g), width=42, break_long_words=False, break_on_hyphens=False
        )
        # Rebalance if greedy wrapping produces three short lines.
        if len(lines) > 2:
            tokens = " ".join(lines).split()
            cut = min(
                range(1, len(tokens)),
                key=lambda j: abs(len(" ".join(tokens[:j])) - len(" ".join(tokens[j:]))),
            )
            lines = [" ".join(tokens[:cut]), " ".join(tokens[cut:])]
        blocks.append(f"{len(blocks) + 1}\n{stamp(start)} --> {stamp(end)}\n" + "\n".join(lines))
    return "\n\n".join(blocks) + ("\n" if blocks else "")
