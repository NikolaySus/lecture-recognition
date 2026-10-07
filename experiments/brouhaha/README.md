# Isolated Brouhaha worker

NumPy 1.26 / torch+torchaudio 2.5.1 / pyannote.audio 3.3.0.
The older torchaudio interface is required by pyannote 3.x; GigaAM keeps its own torch 2.9 environment.

```bash
uv sync --locked --project experiments/brouhaha
.venv/bin/python scripts/setup_brouhaha.py
```

Setup downloads official public Git blobs at revision
`9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891`, verifies their Git SHA1,
and records the checkpoint SHA256 in `model.json`. No gated-model consent is automated.
Vendor sources and weights are local generated artifacts; reproduce them with the setup script.

The worker verifies the checkpoint SHA256 before loading it and accepts only PCM and time windows,
never transcripts or references. Its outputs are speech probability, SNR and C50 in native model units.
A failed download or smoke test blocks Brouhaha candidates; ASR-only development remains provisional.

Official source: https://github.com/marianne-m/brouhaha-vad
