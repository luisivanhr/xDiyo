from datetime import datetime, timezone
import re, subprocess, sys
from support import ROOT, SCRATCH, guard, save, sha
before = guard()
command = [sys.executable, '-B', '-m', 'pytest', 'tests/analytics', '-q', '-p', 'no:cacheprovider', '--basetemp=' + str(SCRATCH / 'full_suite')]
result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
(SCRATCH / 'full_suite.log').write_text(result.stdout + result.stderr, encoding='utf-8')
assert result.returncode == 0, result.stdout[-8000:] + result.stderr
assert guard() == before
assert '632 passed' in result.stdout
record = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'source_sha256': before, 'full_passed': 632, 'new_cases': 142, 'prior_cases': 490, 'test_sha256': {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / 'tests/analytics').glob('*.py'))}, 'log_sha256': sha(SCRATCH / 'full_suite.log')}
save('suite_check.json', record)
print(result.stdout[-1800:])
