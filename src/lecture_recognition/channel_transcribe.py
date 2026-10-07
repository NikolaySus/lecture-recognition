"""Transcribe new stereo recordings using an independently admitted channel profile."""
import argparse
import hashlib
import subprocess
from pathlib import Path

import numpy as np

from .audio import RATE, digest
from .channel_curve_full import alignment_key
from .channel_utility import acoustic_statistics, assemble, likelihood_margin, make_layout, select_channels
from .experiments import identity
from .model_benchmark import ROOT, read, worker, write
from .model_benchmark import layout as historical_layout
from .timeline import to_srt


def ensure_admitted(profile):
    result = profile['validation']
    if profile['policy'] == 'left':
        passed = result['left_passed']
    else:
        passed = result['selector_passed']
    if not passed:
        raise ValueError('Profile did not pass independent validation')
    for name, expected in profile['code'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Profile implementation changed; revalidate: ' + name)


def transcribe(audio, profile_path, model, output, retry=False):
    profile = read(profile_path)
    ensure_admitted(profile)
    config = profile['model_config']
    if config['model'] != model:
        raise ValueError('Profile model mismatch')
    key = identity({'audio': digest(audio), 'profile': digest(profile_path)})[:16]
    folder = ROOT / '.lecture-cache/channel-transcription' / key
    folder.mkdir(parents=True, exist_ok=True)
    for channel, pan in [('left', 'mono|c0=c0'), ('right', 'mono|c0=c1'), ('mono', 'mono|c0=0.5*c0+0.5*c1')]:
        path = folder / (channel + '.f32')
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-y', '-i', str(audio),
                            '-af', 'pan=' + pan, '-ar', str(RATE), '-f', 'f32le', str(temporary)], check=True)
            temporary.replace(path)
    wave = np.memmap(folder / 'left.f32', dtype='<f4', mode='r')
    duration = len(wave) / RATE
    label = profile['layout']['layout']
    mode, maximum, context = ('vad', 20, 1) if label == 'historical' else label.split('-')
    maximum = int(str(maximum).removeprefix('m'))
    context = int(str(context).removeprefix('c'))
    from .timeline import lecturer_regions, student_only_regions
    diar = worker({'operation': 'diarize', 'audio': str(folder / 'mono.f32')}, folder / 'vad', retry=retry)
    if diar['status'] != 'ok':
        raise RuntimeError(diar.get('error'))
    speaker, regions, _ = lecturer_regions(diar['segments'], duration)
    excluded = student_only_regions(diar['segments'], duration, speaker)
    for channel in ('left', 'right', 'mono'):
        masked = np.memmap(folder / (channel + '.f32'), dtype='<f4', mode='r+')
        for a, b in excluded:
            masked[round(a * RATE):round(b * RATE)] = 0
        masked.flush()
        del masked
    mono = np.memmap(folder / 'mono.f32', dtype='<f4', mode='r')
    chunks = historical_layout(mono, duration, 20, regions) if label == 'historical' else make_layout(
        wave, duration, maximum, context, mode, regions)
    channels = ['left'] if profile['policy'] == 'left' else ['left', 'right']
    decoded, traces = {}, {}
    for channel in channels:
        result = worker({'operation': 'asr', 'audio': str(folder / (channel + '.f32')),
                         'config': {**config, 'offline': True}, 'chunks': chunks},
                        folder / channel / 'asr', 'gigaam', retry)
        if result['status'] != 'ok':
            raise RuntimeError(result.get('error'))
        transcripts = [{k: t[k] for k in ('chunk', 'text')} for t in result['transcripts']]
        aligned = worker({'operation': 'align', 'audio': str(folder / 'mono.f32'), 'transcripts': transcripts,
                          'cache': str(folder / channel / 'alignment-cache')}, folder / channel / 'align', retry=retry)
        if aligned['status'] != 'ok':
            raise RuntimeError(aligned.get('error'))
        decoded[channel] = {'transcripts': transcripts, 'aligned': [read(folder / channel / 'alignment-cache/alignment' /
                                               (alignment_key(t) + '.json')) for t in transcripts]}
    if len(channels) == 1:
        decisions = [{'channel': 'left', 'reason': 'fixed_left'} for _ in chunks]
    else:
        competitors = {str(i): [decoded[ch]['transcripts'][i]['text'] for ch in channels] for i in range(len(chunks))}
        for channel in channels:
            for scorer in (model, 'gigaam-ctc') if model != 'gigaam-ctc' else (model,):
                cfg = config if scorer == model else profile['ctc_scorer_config']
                target = folder / channel / ('trace-' + scorer)
                request = {'operation': 'channel-diagnostics', 'audio': str(folder / (channel + '.f32')),
                           'config': cfg, 'chunks': chunks, 'competitors': competitors, 'trace_dir': str(target / 'traces')}
                result = worker(request, target, 'gigaam', retry)
                if result['status'] != 'ok':
                    raise RuntimeError(result.get('error'))
                traces[channel, scorer] = result['items']
        acoustic = {}
        if profile['policy'] in ('snr', 'c50', 'hybrid'):
            for channel in channels:
                request = {'audio': str(folder / (channel + '.f32')), 'checkpoint': profile['brouhaha_checkpoint'],
                           'output': str(folder / channel / 'acoustic.json'), 'windows': [[0, duration]]}
                path = folder / channel / 'acoustic-request.json'
                write(path, request)
                subprocess.run([str(ROOT / 'experiments/brouhaha/.venv/bin/python'),
                                str(ROOT / 'scripts/brouhaha_worker.py'), str(path)], check=True)
                acoustic[channel] = read(Path(request['output']))
        rows = []
        for i, chunk in enumerate(chunks):
            lc, rc = traces['left', 'gigaam-ctc'][i], traces['right', 'gigaam-ctc'][i]
            texts = competitors[str(i)]
            with np.load(lc['path']) as lp, np.load(rc['path']) as rp:
                margin = likelihood_margin(lp['log_probs'], rp['log_probs'], lc['competitor_labels'][texts[0]],
                                           lc['competitor_labels'][texts[1]], lc['blank'])
            row = {'left_text': texts[0], 'right_text': texts[1], 'margin': margin}
            for channel in channels:
                row[channel] = dict(traces[channel, model][i]['confidence'])
                if channel in acoustic:
                    v = acoustic[channel]
                    row[channel].update(acoustic_statistics(v['times'], v['speech'], v['snr'], v['c50'],
                                                            [chunk['start'], chunk['end']]))
            rows.append(row)
        decisions = select_channels(rows, profile['policy'], profile['threshold'], profile.get('hybrid_thresholds'))
    words = assemble([decoded[d['channel']]['aligned'][i] for i, d in enumerate(decisions)], profile['layout']['merger'])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix('.srt').write_text(to_srt(words, regions))
    write(output.with_suffix('.json'), {'audio_sha256': digest(audio), 'profile_sha256': digest(profile_path),
          'model': model, 'lecturer': speaker, 'regions': regions, 'excluded': excluded, 'chunks': chunks, 'decisions': decisions, 'words': words,
          'pcm_sha256': hashlib.sha256(wave.tobytes()).hexdigest()})
    write(folder / 'completed.json', {'output': str(output), 'chunks': len(chunks)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('audio', type=Path)
    parser.add_argument('--model', choices=('gigaam-ctc', 'gigaam-rnnt'), required=True)
    parser.add_argument('--selector-config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--retry-failed', action='store_true')
    args = parser.parse_args()
    transcribe(args.audio, args.selector_config, args.model, args.output, args.retry_failed)
