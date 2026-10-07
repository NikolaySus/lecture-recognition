"""Download the official pinned public Brouhaha release without gated services."""
import base64
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = '9132cbe62ac78f90abdbc21bcf6ec6cfe9bb4891'
API = 'https://api.github.com/repos/marianne-m/brouhaha-vad'


def git_blob_sha(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def get(url):
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1)


def main():
    folder = ROOT / 'experiments/brouhaha/vendor'
    tree = get(API + '/git/trees/' + REVISION + '?recursive=1')['tree']
    for item in tree:
        name = item['path']
        if item['type'] != 'blob' or not (name.startswith(('brouhaha/', 'models/best/'))
                                        or name in ('setup.py', 'requirements.txt', 'README.md')):
            continue
        path = folder / name
        if path.exists() and git_blob_sha(path.read_bytes()) == item['sha']:
            continue
        blob = {'content': ''} if not item['size'] else get(API + '/contents/' + name + '?ref=' + REVISION)
        data = base64.b64decode(blob['content'])
        if item['size'] and not data:
            # Contents API omits large files; Git blob API returns their data.
            data = base64.b64decode(get(API + '/git/blobs/' + item['sha'])['content'])
        if git_blob_sha(data) != item['sha']:
            raise ValueError('Git blob verification failed')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        print(name, len(data), flush=True)
    checkpoint = folder / 'models/best/checkpoints/best.ckpt'
    metadata = {'repository': 'https://github.com/marianne-m/brouhaha-vad', 'revision': REVISION,
                'path': str(checkpoint), 'sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    (folder.parent / 'model.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
