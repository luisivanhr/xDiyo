"""Disposable native UI fixture for browser verification; no production data."""
from pathlib import Path
import tempfile
import threading
import json
from test_ui_workflow import ui_recipe
from xdiyo_analytics.ui import launch_ui

root = Path(tempfile.mkdtemp(prefix='xdiyo-native-ui-completion-'))
recipe = ui_recipe.__wrapped__(root)
handle = launch_ui(workspace=root, open_browser=False, retention_bytes=0)
handle.state.dispatch('save', {'recipe': recipe})
print(json.dumps({'url': handle.url, 'workspace': str(root)}), flush=True)
try:
    threading.Event().wait()
finally:
    handle.close()
