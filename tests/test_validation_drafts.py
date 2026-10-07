import pytest

from lecture_recognition.validation_drafts import insert_draft, prepare_review, timestamp


@pytest.mark.parametrize('seconds,expected', [(523.62, '00:08:43.62'), (944.95, '00:15:44.95'),
                                              (3599.999, '01:00:00.00'), (3600, '01:00:00.00')])
def test_clock_format(seconds, expected):
    assert timestamp(seconds) == expected


def test_review_time_duplicates_preserve_parseable_seconds_and_existing_text():
    text = '# Old\n\n## S001\n\nВремя: 523.62–583.62 с\n\nАудио: WAV\n\nvalid: pending\n\nТранскрипция:\n\nисправил вручную\n'
    result = prepare_review(text)
    assert 'Время: 523.62–583.62 с' in result
    assert '00:08:43.62–00:09:43.62' in result
    assert 'исправил вручную' in result
    assert 'разметка уже не слепая' in result
    again = prepare_review(result)
    assert again == result


def test_insert_only_missing_draft_and_leave_other_section_untouched():
    other = '## S002\n\nvalid: pending\n\nТранскрипция:\n\nмой текст\n'
    text = '## S001\n\nvalid: pending\n\nТранскрипция:\n\n<!-- Введите ручной эталон -->\n\n' + other
    result, inserted = insert_draft(text, 'S001', 'не десять, а двадцать')
    assert inserted and 'не десять, а двадцать' in result
    assert other in result
    assert 'valid: yes' not in result
    assert insert_draft(result, 'S001', 'другой ASR')[0] == result
    assert insert_draft(text, 'S002', 'другой ASR') == (text, False)


def test_empty_and_missing_drafts_are_not_silently_inserted():
    with pytest.raises(ValueError, match='Empty draft'):
        insert_draft('## S001', 'S001', '')
    with pytest.raises(ValueError, match='Missing transcription'):
        insert_draft('## S001', 'S002', 'текст')
