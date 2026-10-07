"""Isolated GigaAM posterior diagnostics; no references and no decoder changes."""
from pathlib import Path

import numpy as np

from .audio import RATE
from .channel_utility import word_confidence
from .evaluation import lexical


def encode_hypothesis(tokenizer, text):
    """Encode with GigaAM's actual vocabulary; unknown characters are unavailable."""
    if tokenizer.charwise:
        vocabulary = {char: i for i, char in enumerate(tokenizer.vocab)}
        return [vocabulary[char] for char in text] if all(char in vocabulary for char in text) else None
    return list(tokenizer.model.encode(text, out_type=int))


def trace_gigaam(model, wave, config):
    import torch

    net = model.model
    wav = torch.from_numpy(np.array(wave, dtype=np.float32)).unsqueeze(0).to('cuda')
    length = torch.tensor([len(wave)], device='cuda')
    with torch.inference_mode():
        if config.get('precision') == 'fp32':
            features, lengths = net.preprocessor(wav, length)
            with torch.autocast('cuda', enabled=False):
                encoded, encoded_len = net.encoder(features.float(), lengths)
        else:
            encoded, encoded_len = net(wav, length)
        count = int(encoded_len[0])
        head, decoder = net.head, net.decoding
        blank, tokenizer = decoder.blank_id, decoder.tokenizer
        vectors, tokens, frame_ids, prefix = [], [], [], []
        previous = None
        word_ids = []
        hits = 0
        if config['model'] == 'gigaam-ctc':
            logits = head(encoded)[0, :count].float()
            log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            tokens = np.argmax(log_probs, axis=1).tolist()
            for t, token in enumerate(tokens):
                if token != blank and token != previous:
                    prefix.append(token)
                previous = token
                word_ids.append(max(0, len(lexical(tokenizer.decode(prefix))) - 1))
                frame_ids.append(t)
            probabilities = np.exp(log_probs)
        else:
            encoded = encoded.transpose(1, 2)
            predictor, state = head.decoder.predict(None, None)
            for t in range(count):
                for _ in range(config.get('max_symbols', 10)):
                    logits = head.joint.joint(encoded[:, t:t + 1], predictor)[0, 0, 0].float()
                    probability = torch.softmax(logits, dim=-1).cpu().numpy()
                    token = int(np.argmax(probability))
                    vectors.append(probability)
                    tokens.append(token)
                    frame_ids.append(t)
                    if token != blank:
                        prefix.append(token)
                    word_ids.append(max(0, len(lexical(tokenizer.decode(prefix))) - 1))
                    if token == blank:
                        break
                    label = torch.tensor([[token]], device='cuda')
                    predictor, state = head.decoder.predict(label, state)
                else:
                    hits += 1
            probabilities = np.asarray(vectors)
            log_probs = np.log(np.maximum(probabilities, 1e-30))
    text = tokenizer.decode(prefix)
    return {'log_probs': log_probs, 'tokens': np.array(tokens), 'token_words': np.array(word_ids),
            'frame_ids': np.array(frame_ids), 'encoder_frames': count, 'blank': blank,
            'text': text, 'limit_hits': hits,
            'confidence': word_confidence(probabilities, tokens, word_ids, blank,
                                          ctc=config['model'] == 'gigaam-ctc')}


def run_diagnostics(request):
    import torch
    from transformers import AutoModel

    config = request['config']
    model = AutoModel.from_pretrained(config['repo'], revision=config['revision'],
              code_revision=config['revision'], trust_remote_code=True, local_files_only=True).to('cuda').eval()
    wave = np.memmap(request['audio'], dtype='<f4', mode='r')
    folder = Path(request['trace_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    items = []
    for i, chunk in enumerate(request['chunks']):
        summary_path = folder / f'{i:05d}.json'
        from .model_benchmark import read, write
        if summary_path.exists():
            saved = read(summary_path)
            from .audio import digest
            if digest(Path(saved['path'])) != saved['sha256']:
                raise ValueError('Diagnostic matrix changed')
            if saved['chunk'] != chunk:
                raise ValueError('Diagnostic chunk changed')
            items.append(saved)
            continue
        trace = trace_gigaam(model, wave[round(chunk['start'] * RATE):round(chunk['end'] * RATE)], config)
        path = folder / f'{i:05d}.npz'
        frame_times = chunk['start'] + (np.arange(trace['encoder_frames']) + .5) * (chunk['end'] - chunk['start']) / trace['encoder_frames']
        trace['times'] = frame_times[trace['frame_ids']]
        np.savez_compressed(path, times=trace.pop('times'), encoder_times=frame_times, **{k: trace.pop(k) for k in ('log_probs', 'tokens', 'token_words', 'frame_ids')})
        if config['model'] == 'gigaam-ctc':
            # All competitors are known to the reference-free caller. Encode once
            # with the actual tokenizer; no inferred vocabulary mapping.
            trace['competitor_labels'] = {text: encode_hypothesis(model.model.decoding.tokenizer, text)
                                          for text in request.get('competitors', {}).get(str(i), [])}
        from .audio import digest
        summary = {**trace, 'chunk': chunk, 'path': str(path), 'sha256': digest(path),
                   'time_grid': 'encoder frame centres uniformly spanning input; approximate'}
        write(summary_path, summary)
        items.append(summary)
        print(f'Diagnostics {i + 1}/{len(request["chunks"])}', flush=True)
    torch.cuda.empty_cache()
    return {'items': items}
