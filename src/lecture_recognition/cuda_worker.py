"""Isolated model process; requests contain audio and configuration, never references."""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from .runtime import read, verify_runtime, write


def ensure_gigaam_snapshot(config):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    try:
        snapshot = Path(snapshot_download(config['repo'], revision=config['revision'], local_files_only=True))
        if not (snapshot / 'config.json').is_file() or not (snapshot / 'pytorch_model.bin').is_file():
            raise LocalEntryNotFoundError('Incomplete cached GigaAM snapshot')
    except LocalEntryNotFoundError:
        snapshot_download(config['repo'], revision=config['revision'],
                          allow_patterns=['*.json', '*.txt', '*.model', '*.py', 'pytorch_model.bin'], max_workers=2)


def run(request, output):
    import torch

    verify_runtime(request.get('runtime_code', {}))
    torch.manual_seed(0)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in the model worker')
    audio = np.memmap(request['audio'], dtype='<f4', mode='r')
    operation = request['operation']
    if operation in ('diarize', 'align'):
        from .models import MODEL_REVISIONS, align, diarize
        if operation == 'diarize':
            return {'segments': diarize(audio, MODEL_REVISIONS['diarization'])}
        from .cli import Cache
        cache = Cache(Path(request['cache']))
        aligned = align(audio, request['transcripts'], 'ru', MODEL_REVISIONS['alignment'], cache.read, cache.write,
                        context_regions=[(0, len(audio) / 16000)])
        return {'aligned': aligned}
    if operation != 'asr':
        raise ValueError('Unknown model operation')
    from transformers import AutoModel

    from .confidence import gibbs
    from .gigaam_decoding import decode_gigaam
    config = request['config']
    ensure_gigaam_snapshot(config)
    model = AutoModel.from_pretrained(config['repo'], revision=config['revision'], code_revision=config['revision'],
                                     trust_remote_code=True, local_files_only=True).to('cuda').eval()
    cache = Path(output).parent / 'raw-chunks'
    cache.mkdir(exist_ok=True)
    transcripts = []
    for i, chunk in enumerate(request['chunks']):
        path = cache / f'{i:05d}.json'
        if path.exists():
            item = read(path)
            if item['chunk'] != chunk:
                raise ValueError('Raw chunk bounds changed')
        else:
            start = time.monotonic()
            wave = np.array(audio[round(chunk['start'] * 16000):round(chunk['end'] * 16000)])
            with torch.inference_mode():
                text, _ = decode_gigaam(model, wave, config)
                confidence = gibbs(model, wave)
            item = {'chunk': chunk, 'text': text, 'confidence': confidence, 'seconds': time.monotonic() - start}
            write(path, item)
        transcripts.append(item)
        print('ASR', i + 1, '/', len(request['chunks']), flush=True)
    return {'transcripts': transcripts}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        result = {**run(read(args.request), args.output), 'status': 'ok'}
    except Exception as exc:
        result = {'status': 'error', 'error': str(exc), 'traceback': traceback.format_exc()}
        traceback.print_exc()
    result['seconds'] = time.monotonic() - started
    write(args.output, result)
    raise SystemExit(0 if result['status'] == 'ok' else 1)


if __name__ == '__main__':
    main()
