"""Compare bounded streaming with native offline inference on real audio (CUDA)."""

import argparse
from pathlib import Path

import numpy as np
import torch

from lecture_recognition.audio import decode
from lecture_recognition.models import MODEL_REVISIONS, configure_diarization, diarization_inputs, load

parser = argparse.ArgumentParser()
parser.add_argument("audio", type=Path)
args = parser.parse_args()
cache = Path(".lecture-cache/check")
cache.mkdir(parents=True, exist_ok=True)
audio = decode(args.audio, cache / "audio.f32")
revision = MODEL_REVISIONS["diarization"]
processor, model = load("diarization", revision)
configure_diarization(processor, model)
for seconds in [5.37, 30.4, 61.37]:
    wave = np.asarray(audio[: round(seconds * 16000)])
    with torch.inference_mode():
        inputs = processor(wave, sampling_rate=16000).to("cuda")
        expected = model(**inputs).logits.cpu()[:, : int(inputs.attention_mask.sum())]
        state, parts = None, []
        for chunk, _ in diarization_inputs(processor, wave):
            if state is None and "num_lookahead_frames" not in chunk:
                chunk["num_lookahead_frames"] = 0
            output = model(**chunk.to("cuda"), speaker_cache=state)
            state = output.speaker_cache
            parts.append(output.logits.cpu())
        actual = torch.cat(parts, dim=1)
    assert actual.shape == expected.shape, (seconds, actual.shape, expected.shape)
    difference = (actual.sigmoid() - expected.sigmoid()).abs()
    disagreement = ((actual > 0) != (expected > 0)).float().mean().item()
    print(
        seconds, "max probability difference", difference.max().item(), "activity disagreement", disagreement
    )
    assert disagreement < 0.01
