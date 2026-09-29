"""Create local listening samples from a completed run; no extra exported transcript."""

import argparse
import json
import subprocess
from pathlib import Path

from lecture_recognition.audio import digest

parser = argparse.ArgumentParser()
parser.add_argument("audio", type=Path)
args = parser.parse_args()
base = Path(".lecture-cache") / digest(args.audio)
runs = list(base.glob("*/run.json"))
if not runs:
    parser.error("Run transcription first.")
run_path = max(runs, key=lambda p: json.loads((p.parent / "settings.json").read_text())["samples"])
run = json.loads(run_path.read_text())
settings = json.loads((run_path.parent / "settings.json").read_text())
duration = settings["samples"] / 16000
diar_path = next((run_path.parent / "diarization").glob("*.json"))
segments = json.loads(diar_path.read_text())
points = [(0, "beginning"), (duration / 2, "middle"), (max(0, duration - 25), "ending")]
students = sorted(
    (s for s in segments if s["Speaker"] != run["speaker"]), key=lambda s: s["End"] - s["Start"], reverse=True
)
points.extend((max(0, s["Start"] - 5), "student_turn") for s in students[:4])
asr_files = sorted((run_path.parent / "asr").glob("*.json"))
cuts = sorted(
    {
        r["chunk"]["core_end"]
        for p in asr_files
        for r in json.loads(p.read_text(encoding="utf-8"))
        if r["chunk"]["end"] > r["chunk"]["core_end"]
    }
)
if cuts:
    points.extend((max(0, cuts[i] - 5), "chunk_boundary") for i in {0, len(cuts) // 2, len(cuts) - 1})
review = run_path.parent / "review"
review.mkdir(exist_ok=True)
manifest = []
for i, (start, reason) in enumerate(points):
    end = min(duration, start + 20)
    path = review / f"{i + 1:02}_{reason}_{start:.2f}.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-ss",
            str(start),
            "-i",
            str(args.audio),
            "-t",
            str(end - start),
            "-acodec",
            "pcm_s16le",
            str(path),
        ],
        check=True,
    )
    manifest.append(
        {"file": path.name, "source_start": start, "source_end": end, "reason": reason, "reviewed": False}
    )
(review / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(review)
