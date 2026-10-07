"""Exercise new APIs from an isolated built wheel, without editable imports."""
import json
import site
import sys
from pathlib import Path
from zipfile import ZipFile
from test_packaging import _copy_publishable_source, _run, git_checkout


def test_installed_composition_examples(git_checkout,tmp_path):
    source=tmp_path/'source'
    source.mkdir()
    _copy_publishable_source(git_checkout,source)
    wheels=tmp_path/'wheels'
    _run([sys.executable,'-I','-m','pip','wheel','--no-deps','--no-index','--no-build-isolation',
          '--no-cache-dir','--disable-pip-version-check','--wheel-dir',str(wheels),str(source)],cwd=tmp_path)
    wheel,=wheels.glob('*.whl')
    with ZipFile(wheel) as archive:
        names=archive.namelist()
    assert 'xdiyo_analytics/composition/adapter.py' in names
    assert 'xdiyo_analytics/evaluation/bankroll.py' in names
    assert 'xdiyo_analytics/evaluation/audit_storage.py' in names
    assert not any(n.startswith(('data/','scraping/','experiments/')) for n in names)
    install=tmp_path/'installed'
    _run([sys.executable,'-I','-m','pip','install','--no-deps','--no-index','--no-compile','--no-cache-dir',
          '--disable-pip-version-check','--target',str(install),str(wheel)],cwd=tmp_path)
    example=(git_checkout/'examples/composition_and_staking.py').read_text(encoding='utf-8')
    script='''
import sys,json
from pathlib import Path
sys.path[:0]=[sys.argv[1],*json.loads(sys.argv[2])]
from importlib.metadata import distribution
commands={e.name for e in distribution('xdiyo-analytics').entry_points if e.group=='console_scripts'}
assert {'xdiyo-ui','xdiyo-report'} <= commands
exec(Path(sys.argv[3]).read_text(encoding='utf-8'))
quote_example={'__name__':'wheel_quote_example'}
exec(Path(sys.argv[4]).read_text(encoding='utf-8'),quote_example)
summary=quote_example['consensus'](*quote_example['synthetic_example'](),audit_level='summary')
assert summary[0].attrs['audit_manifest']['audit_level']=='summary'
assert summary[0].attrs['selected_model_values']['data']
for name,module in list(sys.modules.items()):
    if name=='xdiyo_analytics' or name.startswith('xdiyo_analytics.'):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()),name
print('Isolated wheel composition passed')
'''
    copied_example=tmp_path/'example.py'
    copied_example.write_text(example,encoding='utf-8')
    copied_quote=tmp_path/'quote_example.py'
    copied_quote.write_text((git_checkout/'examples/research_quote_consensus.py').read_text(encoding='utf-8'),encoding='utf-8')
    output=_run([sys.executable,'-I','-S','-c',script,str(install),json.dumps(site.getsitepackages()),str(copied_example),str(copied_quote)],cwd=tmp_path,text=True)
    assert 'Isolated wheel composition passed' in output
