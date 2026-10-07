"""Reference-free Brouhaha inference in the isolated NumPy 1.x environment."""
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments/brouhaha/vendor'))
# pyannote 3.x checkpoints predate PyTorch's weights_only default. This worker
# loads only the pinned official checkpoint after its SHA256 has been verified.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'


def run(request):
    import numpy as np
    import torch
    from brouhaha.inference import BrouhahaInference
    from pyannote.audio import Model

    checkpoint = request['checkpoint']
    path = Path(checkpoint['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint['sha256']:
        raise ValueError('Official checkpoint SHA256 mismatch')
    signature = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()
    output = Path(request['output'])
    if output.exists():
        cached = json.loads(output.read_text())
        if cached['request_sha256'] == signature:
            return
        raise ValueError('Brouhaha cached request changed')
    model = Model.from_pretrained(str(path)).to('cuda').eval()
    inference = BrouhahaInference(model, device=torch.device('cuda'), batch_size=16)
    wave = np.memmap(request['audio'], dtype='<f4', mode='r')
    collected = []
    for a, b in request['windows']:
        data = torch.from_numpy(np.array(wave[round(a * 16000):round(b * 16000)], dtype=np.float32)).unsqueeze(0)
        with torch.inference_mode():
            prediction = inference({'waveform': data, 'sample_rate': 16000})
        values = np.asarray(prediction.data)
        if values.ndim != 2 or values.shape[1] != 3:
            raise ValueError('Expected Brouhaha speech/SNR/C50 output; got ' + str(values.shape))
        grid = prediction.sliding_window
        times = a + grid.start + grid.duration / 2 + np.arange(len(values)) * grid.step
        collected.extend((float(t), *map(float, v)) for t, v in zip(times, values) if a <= t < b)
    collected.sort()
    if not collected:
        raise ValueError('Empty Brouhaha prediction')
    times, speech, snr, c50 = map(list, zip(*collected))
    if not np.isfinite([times, speech, snr, c50]).all():
        raise ValueError('Non-finite Brouhaha output')
    result = {'times': times, 'speech': speech, 'snr': snr, 'c50': c50,
              'checkpoint': checkpoint, 'request_sha256': signature}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(output)
    print('Brouhaha:', len(times), 'frames', flush=True)


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
