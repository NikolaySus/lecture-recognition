"""Equal-channel CTC pipeline with the frozen historical layout and merger."""
import math
import subprocess
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

from .audio import RATE, digest, masked_audio
from .runtime import cuda_worker, verify_runtime, write
from .timeline import Chunk, lecturer_regions, merge_words, quiet_cut, student_only_regions


def historical_layout(audio, duration, regions):
    cuts, cursor = [0.], 0.
    while duration - cursor > 18:
        upper = cursor + 18
        cut = quiet_cut(audio, max(cursor + 1, upper - 4), upper)
        cuts.append(cut)
        cursor = cut
    cuts.append(duration)
    return [Chunk(max(0, a - 1), min(duration, b + 1), a, b).dict()
            for a, b in zip(cuts, cuts[1:]) if any(x < b and y > a for x, y in regions)]


def probe(source):
    import json
    result = subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'a:0', '-show_entries',
                             'stream=channels:format=duration', '-of', 'json', str(source)],
                            capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    if not info.get('streams'):
        raise ValueError('File contains no audio stream')
    channels = int(info['streams'][0]['channels'])
    duration = float(info['format']['duration'])
    if channels not in (1, 2):
        raise ValueError('Only mono or stereo audio is supported')
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError('Invalid audio duration')
    return {'channels': channels, 'duration': duration}


def decode_channel(source, target, pan, limit):
    if target.exists():
        return np.memmap(target, dtype='<f4', mode='r')
    temporary = target.with_suffix('.tmp')
    command = ['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(source), '-map', '0:a:0', '-vn']
    if limit is not None:
        command += ['-t', str(limit)]
    command += ['-af', 'pan=' + pan, '-ar', str(RATE), '-f', 'f32le', str(temporary)]
    subprocess.run(command, check=True)
    if not temporary.stat().st_size:
        raise ValueError('Audio contains no samples')
    temporary.replace(target)
    return np.memmap(target, dtype='<f4', mode='r')


def conflicts(a, b):
    left, right = a.split(), b.split()
    return [{'a': ' '.join(left[i:j]), 'b': ' '.join(right[k:end]), 'operation': tag}
            for tag, i, j, k, end in SequenceMatcher(None, left, right, autojunk=False).get_opcodes()
            if tag != 'equal']


def comparison_segments(chunks, channels):
    """Partition whole measured units once; preserve every raw input transcript."""
    assigned = {}
    for channel, value in channels.items():
        groups = [[] for _ in chunks]
        for word in merge_words(value['aligned']):
            midpoint = (word['start'] + word['end']) / 2
            owner = next((i for i, c in enumerate(chunks) if c['core_start'] <= midpoint < c['core_end']), None)
            if owner is None:
                owner = min(range(len(chunks)), key=lambda i: min(abs(midpoint - chunks[i]['core_start']),
                                                                 abs(midpoint - chunks[i]['core_end'])))
            groups[owner].append({k: word[k] for k in ('start', 'end', 'text')})
        assigned[channel] = groups
    result = []
    for i, chunk in enumerate(chunks):
        variants = {channel: {'text': ' '.join(w['text'] for w in assigned[channel][i]),
                               'words': assigned[channel][i], 'input_text': value['transcripts'][i]['text'],
                               'confidence': value['transcripts'][i]['confidence']}
                    for channel, value in channels.items()}
        result.append({'id': f'S{i + 1:06d}', 'window': [chunk['core_start'], chunk['core_end']],
                       'input_window': [chunk['start'], chunk['end']], 'variants': variants,
                       'conflicts': conflicts(variants['A']['text'], variants['B']['text']) if 'B' in variants else [],
                       'timing': 'native aligned units; edits use segment bounds'})
    return result


def run_pipeline(store, job_id):
    request = store.get(job_id)['request']
    verify_runtime(request['runtime_code'])
    folder = store.folder(job_id)
    source = Path(request['audio_path'])
    if digest(source) != request['audio_sha256']:
        raise ValueError('Source audio changed; create a new transcription job')
    store.state(job_id, status='running', stage='decode')
    audio_dir = folder / 'audio'
    audio_dir.mkdir(exist_ok=True)
    pans = {'A': 'mono|c0=c0'}
    if request['audio_info']['channels'] == 2:
        pans['B'] = 'mono|c0=c1'
    pans['mono'] = 'mono|c0=0.5*c0+0.5*c1' if 'B' in pans else 'mono|c0=c0'
    waves = {name: decode_channel(source, audio_dir / (name + '-raw.f32'), pan, request['limit_seconds'])
             for name, pan in pans.items()}
    duration = len(waves['mono']) / RATE
    if any(len(w) != len(waves['mono']) for w in waves.values()):
        raise ValueError('Channel sample counts differ')
    store.state(job_id, stage='diarization', duration=duration, channel_count=len(pans) - 1)
    diar = cuda_worker({'operation': 'diarize', 'audio': str(audio_dir / 'mono-raw.f32'),
                       'runtime_code': request['runtime_code']}, folder / 'diarization')
    speaker, regions, totals = lecturer_regions(diar['segments'], duration, speaker=request['speaker_id'])
    if speaker is None:
        raise ValueError('No speech detected')
    excluded = student_only_regions(diar['segments'], duration, speaker)
    prepared = {name: masked_audio(wave, audio_dir / (name + '.f32'), excluded, len(wave)) for name, wave in waves.items()}
    chunks = historical_layout(prepared['mono'], duration, regions)
    if not chunks:
        raise ValueError('No lecturer chunks')
    write(folder / 'prepared.json', {'duration': duration, 'lecturer': speaker, 'speaker_seconds': totals,
                                     'regions': regions, 'excluded': excluded, 'chunks': chunks})
    store.state(job_id, stage='asr', chunks=len(chunks), lecturer=speaker, speaker_seconds=totals)
    outputs = {}
    for name in ('A', 'B') if 'B' in pans else ('A',):
        store.state(job_id, stage='asr', channel=name)
        asr = cuda_worker({'operation': 'asr', 'audio': str(audio_dir / (name + '.f32')),
                           'config': request['profile']['model_config'], 'chunks': chunks,
                           'runtime_code': request['runtime_code']}, folder / name / 'asr',
                          request['gigaam_python'])
        store.state(job_id, stage='alignment', channel=name)
        transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in asr['transcripts']]
        aligned = cuda_worker({'operation': 'align', 'audio': str(audio_dir / 'mono.f32'),
                               'transcripts': transcripts, 'cache': str(folder / name / 'alignment-cache'),
                               'runtime_code': request['runtime_code']}, folder / name / 'alignment')
        outputs[name] = {'transcripts': asr['transcripts'], 'aligned': aligned['aligned']}
    # Original input/chunk texts remain independent of merged/proposed output.
    write(folder / 'channels.json', outputs)
    segments = comparison_segments(chunks, outputs)
    store.initialize_segments(job_id, segments)
    store.state(job_id, status='ready', stage='review', segments=len(segments), channel=None)
