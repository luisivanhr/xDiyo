import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / '.pytest_tmp/splits_verification'
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
def read(name):
    return json.loads((SCRATCH / name).read_text(encoding='utf-8-sig'))
def guard():
    expected = read('source_freeze.json')['source_sha256']
    actual = {name: sha(ROOT / name) for name in expected}
    assert actual == expected, 'Source changed; do not transfer a pass.'
    return actual
def save(name, record):
    (SCRATCH / name).write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')
