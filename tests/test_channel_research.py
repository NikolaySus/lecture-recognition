from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

from lecture_recognition import channel_research as research
from lecture_recognition.channel_transcribe import ensure_admitted
from lecture_recognition.channel_utility import aggregate_scores, channel_quality_metrics
from lecture_recognition.evaluation import edit_score
from lecture_recognition.model_benchmark import read, write


def test_totals_weight_words_and_keep_critical_errors():
    score = aggregate_scores([edit_score('один два не', 'один два', 2), edit_score('слово', 'слово', 2)])
    assert score['wer'] == .25
    assert score['negation_errors'] == 1 and score['D'] == 1


def test_no_test_reference_before_freeze(tmp_path, monkeypatch):
    review = tmp_path / 'review.md'
    review.write_text('SECRET TEST REFERENCE')
    monkeypatch.setattr(research, 'REVIEW', review)
    instance = research.Research.__new__(research.Research)
    instance.root = tmp_path
    instance.state = lambda *args, **kwargs: None
    write(tmp_path / 'develop.json', {'complete': True, 'selected': {'gigaam-ctc': {'policy': 'gibbs'}}})
    write(tmp_path / 'chunk-frozen.json', {})
    write(tmp_path / 'validation-windows.json', [])
    write(tmp_path / 'metadata.json', {'code': {}})
    instance.freeze()
    assert 'SECRET' not in (tmp_path / 'frozen.json').read_text()
    instance.freeze()
    write(tmp_path / 'develop.json', {'complete': True, 'selected': {'gigaam-ctc': {'policy': 'margin'}}})
    with pytest.raises(ValueError, match='cannot be changed'):
        instance.freeze()


def test_manual_reference_requires_all_fragments(tmp_path, monkeypatch):
    review = tmp_path / 'review.md'
    monkeypatch.setattr(research, 'REVIEW', review)
    instance = research.Research.__new__(research.Research)
    instance.root = tmp_path
    write(tmp_path / 'validation-windows.json', [{'id': 'S001'}, {'id': 'S002'}])
    review.write_text('## S001\nvalid: yes\nТранскрипция:\nне десять\n## S002\nvalid: pending\nТранскрипция:\n')
    with pytest.raises(ValueError, match='S002'):
        instance.validation_reference()
    review.write_text(review.read_text().replace('valid: pending', 'valid: yes') + 'двадцать')
    assert instance.validation_reference() == {'S001': 'не десять', 'S002': 'двадцать'}


def test_inference_cache_resume_and_reference_free_requests(tmp_path, monkeypatch):
    instance = research.Research.__new__(research.Research)
    instance.root = tmp_path / 'run'
    instance.root.mkdir()
    instance.args = Namespace(output=tmp_path, retry_failed=False)
    instance.metadata = {'source_audio_sha256': 'example'}
    instance.parents = {'gigaam-ctc': {'config': {'model': 'gigaam-ctc', 'backend': 'gigaam'}}}
    instance.prepared = {'audio': str(tmp_path / 'mono.f32')}
    instance.historical = lambda model: ([], [])
    audio = tmp_path / 'left.f32'
    np.zeros(16000, dtype='<f4').tofile(audio)
    instance.audio = lambda channel: audio
    requests = []
    chunk = {'start': 0., 'end': 1., 'core_start': 0., 'core_end': 1.}
    def worker(request, folder, backend='align', retry=False):
        requests.append(request)
        assert not any(k in request for k in ('reference', 'ground_truth', 'review'))
        if request['operation'] == 'asr':
            return {'status': 'ok', 'transcripts': [{'chunk': chunk, 'text': 'текст'}]}
        transcript = request['transcripts'][0]
        write(Path(request['cache']) / 'alignment' / (research.alignment_key(transcript) + '.json'),
              {**transcript, 'words': [{'text': 'текст', 'start_time': .1, 'end_time': .9}]})
        return {'status': 'ok'}
    monkeypatch.setattr(research, 'worker', worker)
    monkeypatch.setattr(research, 'require_cuda', lambda: None)
    first = instance.infer('gigaam-ctc', 'left', [chunk], 'test')
    assert first['status'] == 'ok'
    assert instance.infer('gigaam-ctc', 'left', [chunk], 'test') == first
    assert len(requests) == 2
    assert read(next((instance.root / 'inference').glob('*/result.json'))) == first


def test_unadmitted_profile_rejected():
    with pytest.raises(ValueError, match='independent validation'):
        ensure_admitted({'policy': 'left', 'validation': {'left_passed': False}, 'code': {}})


def test_oracle_counts_whole_fragment_and_ignores_ties():
    unit = {'window': [0, 1], 'chunks': [{'core_start': 0, 'core_end': 1}],
            'decisions': [{'channel': 'right'}],
            'scores': {'new_left': {'errors': 2}, 'new_right': {'errors': 1}, 'selector': {'errors': 1}}}
    metrics = channel_quality_metrics([unit])
    assert metrics['oracle_errors'] == 1 and metrics['ranking_accuracy'] == 1
    assert metrics['regret_errors'] == 0


def test_actual_tokenizer_encoding_handles_unavailable_hypotheses():
    from types import SimpleNamespace

    from lecture_recognition.channel_diagnostics import encode_hypothesis
    from lecture_recognition.channel_utility import likelihood_margin

    tokenizer = SimpleNamespace(charwise=True, vocab=['а', ' ', 'б'])
    assert encode_hypothesis(tokenizer, 'а б') == [0, 1, 2]
    assert encode_hypothesis(tokenizer, 'в') is None
    assert likelihood_margin(np.log([[.5, .5]]), np.log([[.5, .5]]), None, [0], 1) is None


def test_cuda_block_preserves_partial_cache_and_is_not_recorded_as_model_failure(tmp_path, monkeypatch):
    instance = research.Research.__new__(research.Research)
    instance.root = tmp_path / 'run'
    instance.root.mkdir()
    instance.args = Namespace(output=tmp_path, retry_failed=True)
    instance.metadata = {'source_audio_sha256': 'example'}
    instance.parents = {'gigaam-ctc': {'config': {'model': 'gigaam-ctc', 'backend': 'gigaam'}}}
    instance.historical = lambda model: ([], [])
    audio = tmp_path / 'left.f32'
    np.zeros(16000, dtype='<f4').tofile(audio)
    instance.audio = lambda channel: audio
    instance.seed_chunks = lambda *args: []
    def unavailable():
        raise research.CUDAUnavailable('test unavailable')
    monkeypatch.setattr(research, 'require_cuda', unavailable)
    def unexpected_worker(*args, **kwargs):
        pytest.fail('A GPU worker was started despite the preflight block')
    monkeypatch.setattr(research, 'worker', unexpected_worker)
    with pytest.raises(research.CUDAUnavailable):
        instance.infer('gigaam-ctc', 'left', [{'start': 0., 'end': 1., 'core_start': 0., 'core_end': 1.}])
    assert not list((instance.root / 'inference').glob('*/result.json'))


def test_cli_continuation_inherits_explicit_merger(tmp_path, monkeypatch):
    import sys

    run = tmp_path / 'run'
    write(tmp_path / 'latest.json', {'run': str(run)})
    write(run / 'metadata.json', {'merger_override': 'contextual'})
    seen = []
    class FakeResearch:
        def __init__(self, args):
            seen.append(args.merger)
        def diagnose(self):
            pass
    monkeypatch.setattr(research, 'Research', FakeResearch)
    monkeypatch.setattr(sys, 'argv', ['benchmark', 'diagnose', '--output', str(tmp_path), '--source', str(tmp_path)])
    research.main()
    assert seen == ['contextual']


def test_cli_historical_baseline_and_continuation_keep_user_choice(tmp_path, monkeypatch):
    import sys

    run = tmp_path / 'run'
    write(tmp_path / 'latest.json', {'run': str(run)})
    write(run / 'metadata.json', {'merger_override': 'contextual'})
    seen = []
    class FakeResearch:
        def __init__(self, args):
            seen.append((args.fixed_layout, args.merger))
        def historical_baseline(self):
            write(run / 'metadata.json', {'fixed_layout': 'historical', 'merger_override': 'current'})
        def diagnose(self):
            pass
    monkeypatch.setattr(research, 'Research', FakeResearch)
    for stage in ('historical-baseline', 'diagnose'):
        monkeypatch.setattr(sys, 'argv', ['benchmark', stage, '--output', str(tmp_path), '--source', str(tmp_path)])
        research.main()
    assert seen == [('historical', 'current'), ('historical', 'current')]


def test_fixed_layout_rejects_another_search(tmp_path, monkeypatch):
    import sys

    run = tmp_path / 'run'
    write(tmp_path / 'latest.json', {'run': str(run)})
    write(run / 'metadata.json', {'fixed_layout': 'historical', 'merger_override': 'current'})
    monkeypatch.setattr(sys, 'argv', ['benchmark', 'chunk-study', '--output', str(tmp_path)])
    with pytest.raises(SystemExit) as exc:
        research.main()
    assert exc.value.code == 2
