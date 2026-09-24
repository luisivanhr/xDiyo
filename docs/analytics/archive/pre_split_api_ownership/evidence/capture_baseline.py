from datetime import datetime, timezone
import hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
SCRATCH = ROOT / '.pytest_tmp/splits_verification'
ARCHIVE = ROOT / 'docs/analytics/archive/pre_splits_implementation'
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert not (SCRATCH / 'baseline.json').exists()
docs = ['IMPLEMENTATION_PROGRESS.md', 'football_analytics_working_notes.md', 'coaching_history.md', 'src/README.md', 'docs/analytics/datasets.md', 'docs/analytics/datasets_reference.md']
for name in docs:
    target = ARCHIVE / name
    if target.exists():
        assert name in docs[:2], ('Do not overwrite an existing archive', name)
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
sources = [*sorted((ROOT / 'src/xdiyo_analytics').rglob('*.py')), ROOT / 'utils/glicko_rating.py', ROOT / 'utils/team_seasons.py', ROOT / 'pyproject.toml']
protected = set((ROOT / 'tests/analytics').glob('*.py')) | set((ROOT / 'notebooks').glob('*.ipynb'))
protected -= {ROOT / 'tests/analytics' / name for name in ['split_samples.py', 'test_split_core.py', 'test_temporal_splits.py', 'test_cpcv.py']}
protected |= set((ROOT / 'docs/analytics').glob('*check.json'))
protected |= {p for p in (ROOT / 'docs/analytics/archive').rglob('*') if p.is_file()}
protected |= {p for p in (ROOT / 'docs/analytics').glob('*.md') if p.relative_to(ROOT).as_posix() not in docs}
record = {'captured_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sources}, 'protected_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(protected)}, 'archived_docs': docs, 'current_doc_sha256': {n: sha(ROOT / n) for n in docs}, 'reference_notebook': {'path': 'C:/Users/luisi/Escritorio/dsr_pbo_cpcv.ipynb', 'sha256': sha(Path('C:/Users/luisi/Escritorio/dsr_pbo_cpcv.ipynb')), 'read_only_not_executed': True}}
(SCRATCH / 'baseline.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
(SCRATCH / 'source_freeze.json').write_text(json.dumps({k: record[k] for k in ['captured_at_utc', 'source_sha256']}, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'sources': len(sources), 'protected_files': len(protected), 'archived_docs': len(docs)}))
