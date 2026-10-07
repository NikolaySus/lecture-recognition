"""Uncalibrated greedy CTC entropy proxy, alongside the beam transcript."""
import math

import numpy as np


def gibbs(model, wave):
    import torch

    net = model.model
    wav = torch.from_numpy(wave.copy()).unsqueeze(0).to('cuda')
    length = torch.tensor([len(wave)], device='cuda')
    encoded, lengths = net(wav, length)
    count = int(lengths[0])
    logits = net.head(encoded)[0, :count].float()
    p = torch.softmax(logits, dim=-1).cpu().numpy().astype(np.float64)
    tokens = p.argmax(axis=1)
    decoder = net.decoding
    entropy = -(p * np.log(np.maximum(p, 1e-300))).sum(axis=1)
    floor = 1 / p.shape[1]
    values = np.clip((np.exp(-entropy) - floor) / (1 - floor), 0, 1)
    words, prefix, previous, run = {}, [], None, []
    for i, token in enumerate(tokens):
        token = int(token)
        if token == decoder.blank_id:
            previous = None
            continue
        if token == previous:
            run.append(i)
        else:
            prefix.append(token)
            text = decoder.tokenizer.decode(prefix)
            word = max(0, len(text.split()) - 1)
            run = []
            words.setdefault(word, []).append(run)
            run.append(i)
        previous = token
    score = float(np.mean([min(float(values[r].min()) for r in runs) for runs in words.values()])) if words else None
    if score is not None and not math.isfinite(score):
        raise ValueError('Non-finite confidence')
    return {'gibbs': score, 'kind': 'greedy_ctc_entropy', 'scope': 'input_chunk',
            'calibrated': False, 'greedy_text': decoder.tokenizer.decode(prefix)}
