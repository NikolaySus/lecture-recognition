"""Deterministic causal channel scores and sample-continuous stereo mixtures."""

import numpy as np

from .audio import RATE

METHODS = ('energy', 'snr-proxy', 'ev-normalized')
SPEEDS = {'fast': (1.0, 0.3), 'slow': (3.0, 1.0)}
SETTINGS = {'frame_seconds': .03, 'hop_seconds': .01, 'update_seconds': .1,
            'noise_seconds': 20, 'noise_percentile': 10, 'silence_power': 1e-6,
            'mel_bands': 20, 'epsilon': 1e-12}


def sharpen_weights(weights, threshold, steepness):
    """Symmetric, smooth amplification after interpolation; identity below threshold."""
    weights = np.asarray(weights, dtype=np.float64)
    if (not .5 < threshold < 1 or not np.isfinite(steepness) or steepness < 0
            or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1))):
        raise ValueError('Invalid channel curve parameters or weights')
    dominant = np.maximum(weights, 1 - weights)
    u = np.maximum(dominant - threshold, 0) / (1 - threshold)
    amplified = 1 - (1 - dominant) * np.exp(-steepness * u**2)
    output = np.where(weights >= .5, amplified, 1 - amplified)
    return np.where(dominant <= threshold, weights, output)


def features(stereo, excluded=()):
    stereo = np.asarray(stereo, dtype=np.float64)
    if stereo.ndim != 2 or stereo.shape[1] != 2 or not len(stereo) or not np.isfinite(stereo).all():
        raise ValueError('Expected finite, nonempty stereo samples')
    size, hop = round(.03 * RATE), round(.01 * RATE)
    if len(stereo) < size:
        raise ValueError('Audio must contain at least one analysis frame')
    frames = np.lib.stride_tricks.sliding_window_view(stereo, size, axis=0)[::hop]
    window = np.hamming(size)
    energy = np.sum((frames * window)**2, axis=2) / np.sum(window**2)
    spectrum = np.abs(np.fft.rfft(frames * window, n=512, axis=2))**2
    hz = np.fft.rfftfreq(512, 1 / RATE)
    edges = 700 * (10**(np.linspace(0, 2595 * np.log10(1 + RATE / 2 / 700), 22) / 2595) - 1)
    filters = np.array([np.maximum(0, np.minimum((hz - a) / (b - a), (c - hz) / (c - b)))
                        for a, b, c in zip(edges, edges[1:], edges[2:])])
    envelopes = np.cbrt(spectrum @ filters.T)
    ends = (np.arange(len(frames)) * hop + size) / RATE
    valid = np.ones(len(ends), dtype=bool)
    for a, b in excluded:
        valid &= ~((ends > a) & (ends - size / RATE < b))
    return {'ends': ends, 'energy': energy, 'envelopes': envelopes, 'valid': valid,
            'samples': len(stereo)}


def mix(stereo, method, speed, excluded=(), analysis=None):
    if method not in METHODS or speed not in SPEEDS:
        raise ValueError('Unknown dynamic channel configuration')
    stereo = np.asarray(stereo, dtype=np.float64)
    f = features(stereo, excluded) if analysis is None else analysis
    if f['samples'] != len(stereo):
        raise ValueError('Analysis length mismatch')
    span, tau = SPEEDS[speed]
    times = np.arange(0, len(stereo) / RATE, .1)
    weights, targets, scores, correlation, suppression = [], [], [], [], []
    previous = .5
    for i, t in enumerate(times):
        end = np.searchsorted(f['ends'], t, side='right')
        begin = np.searchsorted(f['ends'], max(0, t - span))
        idx = np.flatnonzero(f['valid'][begin:end]) + begin
        q = np.zeros(2)
        active = False
        if len(idx):
            energy = np.mean(f['energy'][idx], axis=0)
            active = np.max(energy) >= SETTINGS['silence_power']
            if method == 'energy':
                q = energy
            elif method == 'snr-proxy':
                start = np.searchsorted(f['ends'], max(0, t - 20))
                noise_idx = np.flatnonzero(f['valid'][start:end]) + start
                noise = np.percentile(f['energy'][noise_idx], 10, axis=0)
                q = np.maximum(energy - noise, 0) / np.maximum(noise, 1e-12)
            else:
                u = f['envelopes'][idx]
                q = np.mean(np.var(u, axis=0) / np.maximum(np.mean(u, axis=0)**2, 1e-12), axis=1)
            q[energy <= 1e-12] = 0
            if active and np.count_nonzero(energy > 1e-12) == 1:
                q = (energy > 1e-12).astype(float)
        target = float(q[0] / q.sum()) if q.sum() > 0 else .5
        if active and i:
            previous += (1 - np.exp(-(t - times[i - 1]) / tau)) * (target - previous)
        weights.append(float(previous))
        targets.append(target)
        scores.append(q.tolist())
    sample_weights = np.interp(np.arange(len(stereo)) / RATE, times, weights)
    wave = sample_weights * stereo[:, 0] + (1 - sample_weights) * stereo[:, 1]
    for t in times:
        a, b = round(t * RATE), min(len(wave), round((t + .1) * RATE))
        block, w = stereo[a:b], sample_weights[a:b]
        powers = np.mean(block**2, axis=0)
        correlation.append(float(np.mean(block[:, 0] * block[:, 1]) /
                                 max(np.sqrt(powers.prod()), 1e-12)))
        expected = np.mean((w * block[:, 0])**2 + ((1 - w) * block[:, 1])**2)
        suppression.append(float(10 * np.log10(max(np.mean(wave[a:b]**2), 1e-12) / max(expected, 1e-12))))
    for a, b in excluded:
        wave[round(a * RATE):round(b * RATE)] = 0
    if not np.isfinite(wave).all() or np.max(np.abs(wave)) > np.max(np.abs(stereo)) + 1e-12:
        raise ValueError('Mixture is invalid or introduces an overload')
    diagnostics = {'method': method, 'speed': speed, 'window_seconds': span, 'smoothing_seconds': tau,
                   'settings': SETTINGS, 'times': times.tolist(), 'weights_left': weights,
                   'targets_left': targets, 'scores': scores, 'correlation': correlation,
                   'suppression_db': suppression,
                   'suppressed_intervals': [[float(t), min(float(t + .1), len(wave) / RATE)]
                                            for t, db in zip(times, suppression) if db < -3]}
    return wave.astype('<f4'), diagnostics
