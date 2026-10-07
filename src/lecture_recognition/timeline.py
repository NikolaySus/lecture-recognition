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


def _merge_shifted_seams(groups, aligned, audit):
    """Reconcile unique phrase copies at adjacent audio seams.

    Match lexical tokens inside compound alignment units; trimming text keeps
    the original unit interval and never estimates timestamps for its parts.
    """
    def flatten(words):
        return [(m.group().casefold().replace("ё", "е"), w, m.start(), m.end())
                for w in words for m in re.finditer(r"\w+", w["text"])]

    def midpoint(token):
        w = token[1]
        return (w["start"] + w["end"]) / 2

    def snapshot(copy):
        return [{"text": t[0], "start": t[1]["start"], "end": t[1]["end"]} for t in copy]

    def remove(words, copy, chunk):
        by_word = {}
        for _, w, lo, hi in copy:
            by_word.setdefault(id(w), set()).add((lo, hi))
        for w in list(words):
            cuts = by_word.get(id(w))
            if not cuts:
                continue
            remaining = [m for m in re.finditer(r"\w+", w["text"]) if (m.start(), m.end()) not in cuts]
            if not remaining:
                words.remove(w)
            else:
                # Matches are suffixes/prefixes, so retained words are contiguous.
                w["text"] = w["text"][remaining[0].start():remaining[-1].end()]
                # A compound unit can straddle the ownership boundary. Its
                # remaining words must not disappear solely because the unit
                # midpoint describes the removed phrase as well.
                w["owner"] |= chunk.core_start <= w["start"] < chunk.core_end and w["end"] > chunk.core_start

    for i in range(len(groups) - 1):
        left, right = groups[i], groups[i + 1]
        a, b = Chunk(**aligned[i]["chunk"]), Chunk(**aligned[i + 1]["chunk"])
        lo, hi = max(a.start, b.start), min(a.end, b.end)
        if not left or not right or hi <= lo or abs(a.core_end - b.core_start) > 1e-6:
            continue
        ltokens, rtokens = flatten(left), flatten(right)
        matches = []
        # Leading context can be misrecognized (e.g. a word ending); skip only
        # tokens outside the right chunk's ownership, never owned speech.
        for skip in range(min(4, len(rtokens))):
            if any(t[1]["owner"] for t in rtokens[:skip]):
                break
            for n in range(1, min(12, len(ltokens), len(rtokens) - skip) + 1):
                phrase = [t[0] for t in ltokens[-n:]]
                if n == 1 and not any(len(re.findall(r"\w+", t[1]["text"])) > 1
                                      for t in (ltokens[-1], rtokens[skip])):
                    continue
                if (n == 1 or len(set(phrase)) >= 2) and phrase == [t[0] for t in rtokens[skip:skip + n]]:
                    matches.append((n, -skip, phrase))
        for n, negative_skip, phrase in sorted(matches, reverse=True):
            skip = -negative_skip
            lcopy, rcopy = ltokens[-n:], rtokens[skip:skip + n]
            local_lo = min(lo, lcopy[0][1]["start"], rcopy[0][1]["start"]) - .15
            local_hi = max(hi, lcopy[-1][1]["end"], rcopy[-1][1]["end"]) + .15

            def occurrences(ts):
                values = [t[0] for t in ts]
                return sum(values[j:j + n] == phrase and
                           all(local_lo <= midpoint(t) <= local_hi
                               for t in ts[j:j + n])
                           for j in range(len(values) - n + 1))

            if occurrences(ltokens) != 1 or occurrences(rtokens) != 1:
                continue
            la = all(lo - .15 <= midpoint(t) <= hi + .15 for t in lcopy)
            ra = all(lo - .15 <= midpoint(t) <= hi + .15 for t in rcopy)
            temporal_pairs = [
                min(x[1]["end"], y[1]["end"]) - max(x[1]["start"], y[1]["start"])
                > .5 * min(x[1]["end"] - x[1]["start"], y[1]["end"] - y[1]["start"])
                for x, y in zip(lcopy, rcopy)]
            temporal = all(temporal_pairs)
            # Several correctly aligned anchors can identify a phrase even
            # when one compound unit has collapsed onto its neighbour.
            hull_overlap = min(lcopy[-1][1]["end"], rcopy[-1][1]["end"]) - max(
                lcopy[0][1]["start"], rcopy[0][1]["start"])
            anchored_phrase = (n >= 3 and sum(temporal_pairs) >= max(2, (n + 1) // 2)
                               and hull_overlap > .5 * min(
                                   lcopy[-1][1]["end"] - lcopy[0][1]["start"],
                                   rcopy[-1][1]["end"] - rcopy[0][1]["start"]))
            lduration = lcopy[-1][1]["end"] - lcopy[0][1]["start"]
            rduration = rcopy[-1][1]["end"] - rcopy[0][1]["start"]
            stretched_pair = (n == 2 and any(temporal_pairs) and
                              hull_overlap > .5 * min(lduration, rduration) and
                              max(t[1]["end"] - t[1]["start"] for t in lcopy + rcopy)
                              > 2 * min(lduration, rduration))
            if la and ra and (temporal or anchored_phrase or stretched_pair):
                # Same measured speech represented by different compound units.
                keep_left = sum(t[1]["margin"] for t in lcopy) >= sum(t[1]["margin"] for t in rcopy)
                if stretched_pair:
                    keep_left = lduration <= rduration
                reason = "coincident_phrase"
            elif la != ra:
                if max(abs(lcopy[0][1]["start"] - rcopy[0][1]["start"]),
                       abs(lcopy[-1][1]["end"] - rcopy[-1][1]["end"])) > hi - lo + 1:
                    continue
                keep_left = la
                if not all(t[1]["owner"] for t in (lcopy if la else rcopy)):
                    continue
                reason = "shifted_phrase"
            else:
                continue
            # Keep a left compound containing speech before the shared phrase
            # when the right chunk starts directly with that phrase. Otherwise
            # deleting its suffix can discard the orphan prefix ("ряда а").
            # Keeping the unit intact also preserves order despite time jitter.
            leading = lcopy[0]
            orphan_prefix = (not keep_left and skip == 0 and not leading[1]["owner"]
                             and b.core_start <= leading[1]["start"] < hi
                             and bool(re.search(r"\w+", leading[1]["text"][:leading[2]])))
            if orphan_prefix:
                keep_left = True
            kept, removed = (lcopy, rcopy) if keep_left else (rcopy, lcopy)
            event = {"chunks": [i, i + 1], "phrase": " ".join(phrase), "reason": reason,
                     "overlap": [lo, hi], "kept_chunk": i if keep_left else i + 1,
                     "kept": snapshot(kept), "removed": snapshot(removed)}
            if orphan_prefix:
                event['preserved_prefix'] = leading[1]['text'][:leading[2]].strip()
            kept_chunk = a if keep_left else b
            seen_units = set()
            for _, w, _, _ in kept:
                if id(w) in seen_units:
                    continue
                seen_units.add(id(w))
                if not w["owner"] and w["start"] < kept_chunk.core_start:
                    positions = [(start, end) for _, unit, start, end in kept if unit is w]
                    # Matching words are supported by both chunks. Leading
                    # context attached to the same unit is not: retain only
                    # the matched part rather than promote all context words.
                    w["text"] = w["text"][min(start for start, _ in positions):max(end for _, end in positions)]
                w["chunks"].update({i, i + 1})
            remove(right if keep_left else left, removed, b if keep_left else a)
            if audit is not None:
                audit.append(event)
            break


def _recover_seam_words(groups, aligned, audit):
    """Recover dropped boundary words using independent overlap anchors.

    No reference text is consulted. A rescue needs a counterpart or adjacent
    anchor in the other chunk; it never promotes all words from context.
    """
    def tokens(word):
        return re.findall(r"\w+", word["text"].casefold().replace("ё", "е"))

    def coincides(x, y):
        overlap = min(x["end"], y["end"]) - max(x["start"], y["start"])
        return overlap > .5 * min(x["end"] - x["start"], y["end"] - y["start"])

    for i in range(len(groups) - 1):
        left, right = groups[i], groups[i + 1]
        a, b = Chunk(**aligned[i]["chunk"]), Chunk(**aligned[i + 1]["chunk"])
        lo, hi = max(a.start, b.start), min(a.end, b.end)
        if not left or not right or hi <= lo or abs(a.core_end - b.core_start) > 1e-6:
            continue

        def record(word, reason, evidence):
            word["owner"] = True
            if audit is not None:
                audit.append({"chunks": [i, i + 1], "phrase": word["text"], "reason": reason,
                              "overlap": [lo, hi], "kept_chunk": i if any(word is w for w in left) else i + 1,
                              "kept": [{k: word[k] for k in ("text", "start", "end")}],
                              "removed": [], "evidence": evidence})

        first = right[0]
        # A compound ending in the right chunk's first anchored word can
        # contain a preceding word absent from the right transcript ("ряда а").
        if first["owner"] and len(tokens(first)) == 1:
            for word in left[-3:]:
                ts = tokens(word)
                if (not word["owner"] and len(ts) > 1 and ts[-1:] == tokens(first)
                        and b.core_start <= word["start"] < hi and coincides(word, first)):
                    original = word["text"]
                    # Keep the complete compound to preserve its lexical order;
                    # splitting "ряда а" would put "ряда" after the independently
                    # aligned "а" because of small timestamp jitter.
                    word["chunks"].update({i, i + 1})
                    right.remove(first)
                    record(word, "recovered_compound_prefix", {"compound": original, "anchor": first["text"]})
                    if audit is not None:
                        audit[-1]["removed"] = [{k: first[k] for k in ("text", "start", "end")}]
                    break

        if first["owner"] or len(right) < 2 or len(tokens(first)) != 1:
            continue
        following = right[1]
        if not following["owner"]:
            continue
        for index in range(max(0, len(left) - 4), len(left) - 1):
            word, next_word = left[index:index + 2]
            # Both timestamp ownership tests can reject a word. Keep the
            # tighter copy when its whole interval belongs to the other core
            # and both chunks agree on the immediately following anchor.
            if (not word["owner"] and len(word["chunks"]) == 1 and len(tokens(word)) == 1
                    and b.core_start <= word["start"] < word["end"] <= hi + .15
                    and first["start"] < b.core_start and word["end"] - word["start"]
                    < .75 * (first["end"] - first["start"])
                    and coincides(word, first) and tokens(next_word) == tokens(following)
                    and coincides(next_word, following)):
                record(word, "recovered_unowned_word", {"other_copy": first["text"],
                                                       "anchor": following["text"]})
                break
            # A short attached prefix can stretch the right copy across the
            # ownership boundary ("спроцент"). Rescue the complete left word
            # only with both a preceding temporal anchor and a following
            # lexical/temporal anchor. Do not rewrite the distorted right text.
            if (index > 0 and not word["owner"] and len(tokens(word)) == 1
                    and len(tokens(next_word)) == len(tokens(following)) == 1
                    and b.core_start <= word["start"] < word["end"] <= hi + .15
                    and first["start"] < b.core_start and coincides(word, first)):
                complete, distorted = tokens(word)[0], tokens(first)[0]
                prefix_length = len(distorted) - len(complete)
                anchor = tokens(next_word)[0]
                if (len(complete) >= 5 and 1 <= prefix_length <= 3 and distorted.endswith(complete)
                        and len(anchor) >= 3 and tokens(following)[0].startswith(anchor)
                        and coincides(next_word, following) and coincides(left[index - 1], first)):
                    record(word, "recovered_prefixed_copy", {"other_copy": first["text"],
                           "preceding_anchor": left[index - 1]["text"], "following_anchor": following["text"]})
                    break
            # A clipped suffix of the preceding word can be attached to the
            # first word ("выбирался" + "сяпроцент"). Require a second,
            # independently timed prefix anchor after it before removing that
            # repeated fragment. Retain the original interval for the remainder.
            if (word["owner"] and len(tokens(word)) == 1 and first["start"] < b.core_start < first["end"]
                    and first["end"] - first["start"] >= 2 * (word["end"] - word["start"])
                    and coincides(word, first)):
                before, after = tokens(word)[0], tokens(first)[0]
                suffixes = [n for n in (2, 3) if len(after) >= n + 5 and before[-n:] == after[:n]]
                anchors = [w for w in left[index + 1:]
                           if len(tokens(w)) == 1 and len(tokens(w)[0]) >= 3
                           and tokens(following)[0].startswith(tokens(w)[0]) and coincides(w, following)]
                if suffixes and anchors:
                    fragment = max(suffixes)
                    original = first["text"]
                    match = re.search(r"\w+", original)
                    first["text"] = original[match.start() + fragment:]
                    record(first, "recovered_clipped_prefix", {"original": original,
                           "preceding_anchor": word["text"], "following_anchor": following["text"],
                           "repeated_fragment": after[:fragment]})
                    break


def merge_words(aligned, *, seam_audit=None):
    groups = []
    for chunk_index, item in enumerate(aligned):
        group = []
        groups.append(group)
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
            group.append(
                {
                    "start": max(start, c.start),
                    "end": min(end, c.end),
                    "text": word["text"].strip(),
                    "margin": min(mid - c.start, c.end - mid),
                    "owner": c.core_start <= mid < c.core_end,
                    "chunks": {chunk_index},
                }
            )
    _merge_shifted_seams(groups, aligned, seam_audit)
    _recover_seam_words(groups, aligned, seam_audit)
    candidates = [w for group in groups for w in group]
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
