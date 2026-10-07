"""Small, reference-free runtime primitives shared by the service and workers."""
import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def verify_runtime(code):
    from .audio import digest
    for name, expected in code.items():
        path = Path(__file__).parent / name if name.endswith('.py') and '/' not in name else ROOT / name
        if not path.is_file() or digest(path) != expected:
            raise ValueError('Runtime changed; start a new job instead of mixing old and new caches: ' + name)


def worker_env():
    env = os.environ.copy()
    for key in ('ALL_PROXY', 'all_proxy'):
        if env.get(key, '').startswith('socks://'):
            env.pop(key)
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1]) + os.pathsep + env.get('PYTHONPATH', '')
    env.setdefault('HF_HUB_DISABLE_XET', '1')
    return env


def cuda_worker(request, directory, gigaam_python=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    req, result = directory / 'request.json', directory / 'result.json'
    if req.exists() and read(req) != request:
        raise ValueError('Worker request changed in existing cache')
    write(req, request)
    if result.exists() and read(result)['status'] == 'ok':
        return read(result)
    interpreter = sys.executable
    if request['operation'] == 'asr':
        interpreter = gigaam_python or os.environ.get('LECTURE_GIGAAM_PYTHON')
        if not interpreter:
            base = ROOT / 'experiments/gigaam/.venv'
            interpreter = str(base / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python'))
        if not Path(interpreter).is_file():
            raise RuntimeError('GigaAM environment missing: uv sync --locked --project experiments/gigaam')
    with (directory / 'worker.log').open('a', encoding='utf-8') as log:
        process = subprocess.run([interpreter, '-m', 'lecture_recognition.cuda_worker', str(req), str(result)],
                                 env=worker_env(), stdout=log, stderr=subprocess.STDOUT)
    if process.returncode or not result.exists() or read(result)['status'] != 'ok':
        error = read(result).get('error') if result.exists() else 'Worker produced no result'
        raise RuntimeError(f'{error}; log: {directory / "worker.log"}')
    return read(result)
