import shutil
import subprocess

import numpy as np
import pytest
import soundfile as sf

from lecture_recognition.audio import decode


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="FFmpeg unavailable")
@pytest.mark.parametrize("suffix", ["wav", "mp3", "flac", "m4a"])
def test_decode_formats_and_unicode_paths(tmp_path, suffix):
    source = tmp_path / "исходный звук.wav"
    sf.write(source, np.zeros((48000, 2), dtype=np.float32), 48000)
    encoded = tmp_path / f"запись с пробелами.{suffix}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), str(encoded)], check=True)
    target = tmp_path / "audio.f32"
    audio = decode(encoded, target)
    assert 15000 <= len(audio) <= 17000
    assert np.isfinite(audio).all()
