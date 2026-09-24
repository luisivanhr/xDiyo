"""Fallback: saved notebook in installed nbconvert's Lab HTML template, no live server."""
from datetime import datetime, timezone
import importlib.util
import json
import subprocess
import nbformat
import nbconvert
from nbconvert import HTMLExporter
from support import ROOT, SCRATCH, guard, read, save, sha

source = guard()
notebook = ROOT / 'notebooks/08_reporting_quickstart.ipynb'
before = sha(notebook)
assert before == read('notebook_check.json')['sha256']
work = SCRATCH / 'jupyter_frontend'
work.mkdir(exist_ok=True)
exporter = HTMLExporter(template_name='lab', require_js_url='', mathjax_url='',
                        jquery_url='', mermaid_js_url='')
# No RequireJS outputs, mathematical Markdown, widgets or Mermaid appear in this notebook.
# Suppress template-only CDN loaders while leaving the saved notebook output unchanged.
template = '''{% extends 'lab/index.html.j2' %}
{% block html_head_js %}{% endblock html_head_js %}
{% block html_head_js_mathjax %}{% endblock html_head_js_mathjax %}
{% block html_head_js_mermaidjs %}{% endblock html_head_js_mermaidjs %}
'''
(work / 'offline.html.j2').write_text(template, encoding='utf-8')
exporter.raw_template = template
body, _ = exporter.from_notebook_node(nbformat.read(notebook, as_version=4))
output = work / 'saved_notebook.html'
output.write_text(body, encoding='utf-8')
node = 'C:/Users/luisi/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin/node.exe'
completed = subprocess.run([node, str(SCRATCH / 'check_notebook_browser.mjs'), '--export'],
    cwd=ROOT, capture_output=True, text=True, timeout=90, creationflags=subprocess.CREATE_NO_WINDOW)
print(completed.stdout)
if completed.returncode:
    raise RuntimeError(completed.stderr)
outcome = read('jupyter_frontend/browser.json')
assert all(item['passed'] for item in outcome['checks'])
assert not outcome['page_errors'] and not outcome['remote_requests']
assert sha(notebook) == before and guard() == source
save('notebook_export_check.json', {'checked_at_utc':datetime.now(timezone.utc).isoformat(),
     'status':'verified', 'source_sha256':source, 'notebook_sha256':before,
     'notebook_unchanged':True, 'no_cells_reexecuted':True, 'nbconvert_version':nbconvert.__version__,
     'export_path':output.relative_to(ROOT).as_posix(), 'export_sha256':sha(output), 'browser':outcome,
     'live_jupyter_frontend':{'status':'unavailable',
         'reason':'jupyterlab/notebook/nbclassic/jupyter_server are absent from the project, base and bundled Python runtimes checked; installation is outside this batch.',
         'project_module_availability':{name:importlib.util.find_spec(name) is not None for name in ('jupyterlab','notebook','nbclassic','jupyter_server')},
         'server_started':False, 'packages_installed':False}})
print({'status':'verified', 'mode':'saved_nbconvert_lab_export', 'checks':len(outcome['checks']),
       'live_frontend':'unavailable', 'saved_notebook_unchanged':True})
