"""Inference-only beam decoders; pure search is testable without CUDA or weights."""

from functools import lru_cache

import numpy as np

NEG = float('-inf')


def add(a, b):
    return float(np.logaddexp(a, b))


class ContextBias:
    """Phrase-prefix potential over decoded token prefixes, refunded on mismatch.

    A character trie makes matching independent of SentencePiece segmentation.
    Only word-boundary starts qualify; completed phrases retain their bonus.
    An unfinished prefix has zero terminal bonus. Longest match wins at a start.
    """

    def __init__(self, decode, terms=(), weight=0):
        self.decode = decode
        self.weight = weight
        self.trie = {}
        for term in sorted(set(terms)):
            node = self.trie
            term = term.casefold().replace('ё', 'е')
            for char in term:
                node = node.setdefault(char, {})
                node.setdefault('_lengths', []).append(len(term))
            node['_end'] = True

    @lru_cache(maxsize=65536)
    def __call__(self, prefix, final=False):
        if not self.weight:
            return 0.0
        text = self.decode(list(prefix)).casefold().replace('ё', 'е')
        total = 0.0
        cursor = 0
        while cursor < len(text):
            if cursor and (text[cursor - 1].isalnum() or text[cursor - 1] == '_'):
                cursor += 1
                continue
            node, j, completed = self.trie, cursor, None
            while j < len(text) and text[j] in node:
                node = node[text[j]]
                j += 1
                if '_end' in node and (j == len(text) or not (text[j].isalnum() or text[j] == '_')):
                    completed = j
            if completed is not None:
                total += 1
                cursor = completed
                continue
            if not final and j == len(text) and j > cursor:
                total += (j - cursor) / min(node['_lengths'])
            cursor += 1
        return self.weight * total


def ctc_beam(log_probs, blank, beam, bias=None):
    """Prefix search sums CTC paths, keeping blank/nonblank masses separate."""
    bias = bias or (lambda prefix, final=False: 0.0)
    states = {(): (0.0, NEG)}
    for index, frame in enumerate(np.asarray(log_probs)):
        nxt = {}

        def update(prefix, pb=NEG, pn=NEG):
            old_b, old_n = nxt.get(prefix, (NEG, NEG))
            nxt[prefix] = (add(old_b, pb), add(old_n, pn))

        for prefix, (pb, pn) in states.items():
            total = add(pb, pn)
            update(prefix, pb=total + frame[blank])
            for token, probability in enumerate(frame):
                if token == blank:
                    continue
                if prefix and token == prefix[-1]:
                    update(prefix, pn=pn + probability)
                    update(prefix + (token,), pn=pb + probability)
                else:
                    update(prefix + (token,), pn=total + probability)
        final = index == len(log_probs) - 1
        states = dict(sorted(nxt.items(), key=lambda x: (-(add(*x[1]) + bias(x[0], final=final)), x[0]))[:beam])
    return min(states, key=lambda p: (-(add(*states[p]) + bias(p, final=True)), p))


def rnnt_beam(step, frames, blank, beam, max_symbols=10, bias=None, topk=None):
    """Time-synchronous search. step(t, prefix) supplies predictor/joint log P.

    Prefixes at the same frame/depth share predictor state and path mass.
    Blank always remains available, including at the emission limit.
    """
    bias = bias or (lambda prefix, final=False: 0.0)
    states, limit_hits = {(): 0.0}, 0
    for t in range(frames):
        active, complete = states, {}
        for depth in range(max_symbols + 1):
            expansions = {}
            for prefix, mass in active.items():
                probs = np.asarray(step(t, prefix))
                complete[prefix] = add(complete.get(prefix, NEG), mass + probs[blank])
                if depth == max_symbols:
                    limit_hits += int(int(np.argmax(probs)) != blank)
                    continue
                tokens = [int(k) for k in np.argsort(-probs, kind='stable') if k != blank]
                if topk is not None:
                    tokens = tokens[:topk]
                for token in tokens:
                    p = prefix + (token,)
                    expansions[p] = add(expansions.get(p, NEG), mass + probs[token])
            if not expansions:
                break
            active = dict(sorted(expansions.items(), key=lambda x: (-(x[1] + bias(x[0])), x[0]))[:beam])
        final = t == frames - 1
        states = dict(sorted(complete.items(), key=lambda x: (-(x[1] + bias(x[0], final=final)), x[0]))[:beam])
    best = min(states, key=lambda p: (-(states[p] + bias(p, final=True)), p))
    return best, limit_hits


def decode_gigaam(model, wave, cfg):
    """Adapter for the pinned HF GigaAM model; no changes to upstream code."""
    import torch

    net = model.model
    wav = torch.from_numpy(wave).unsqueeze(0).to('cuda')
    length = torch.tensor([len(wave)], device='cuda')
    if cfg.get('precision') == 'fp32':
        features, lengths = net.preprocessor(wav, length)
        with torch.autocast('cuda', enabled=False):
            encoded, encoded_len = net.encoder(features.float(), lengths)
    else:
        encoded, encoded_len = net(wav, length)
    decoding = cfg.get('decoding', 'greedy')
    head, decoder = net.head, net.decoding
    limit = cfg.get('max_symbols', 10)
    if decoding == 'greedy' and not cfg.get('diagnostic_greedy'):
        return decoder.decode(head, encoded, encoded_len)[0], 0
    tokenizer = decoder.tokenizer
    bias = ContextBias(tokenizer.decode, cfg.get('terms', ()), cfg.get('bias_weight', 0))
    count = int(encoded_len[0])
    if cfg['model'] == 'gigaam-ctc':
        probs = head(encoded)[0, :count].float().cpu().numpy()
        tokens = ctc_beam(probs, decoder.blank_id, cfg['beam'], bias)
        hits = 0
    else:
        encoded = encoded.transpose(1, 2)
        predictor = {}

        def predict(prefix):
            if prefix not in predictor:
                if prefix:
                    _, state = predict(prefix[:-1])
                    label = torch.tensor([[prefix[-1]]], device='cuda')
                    predictor[prefix] = head.decoder.predict(label, state)
                else:
                    predictor[prefix] = head.decoder.predict(None, None)
            return predictor[prefix]

        def step(t, prefix):
            g, _ = predict(prefix)
            return head.joint.joint(encoded[:, t:t + 1], g)[0, 0, 0].float().cpu().numpy()

        if decoding == 'greedy':
            tokens, hits = (), 0
            for t in range(count):
                for _ in range(limit):
                    token = int(np.argmax(step(t, tokens)))
                    if token == decoder.blank_id:
                        break
                    tokens += (token,)
                else:
                    hits += 1
        else:
            tokens, hits = rnnt_beam(step, count, decoder.blank_id, cfg['beam'], limit, bias,
                                     topk=2 * cfg['beam'])
    return tokenizer.decode(list(tokens)), hits
