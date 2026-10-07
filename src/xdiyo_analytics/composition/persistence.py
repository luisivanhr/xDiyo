"""Complete trusted fitted graphs plus an inspectable, non-executable manifest."""
from dataclasses import dataclass
from pathlib import Path
import importlib.metadata
import json
import hashlib
import inspect
from ..training.persistence import JoblibSerializer


@dataclass(frozen=True)
class CompositeSerializer:
    """Uses trusted joblib state, never imports registrations named by an artifact."""
    format_id: str = 'xdiyo.composition.joblib.v1'
    required_components: tuple = ()

    def save(self,adapter,directory):
        from .adapter import CompositeModelAdapter
        if not isinstance(adapter,CompositeModelAdapter) or not hasattr(adapter,'models_'):
            raise TypeError('CompositeSerializer requires a fitted CompositeModelAdapter.')
        adapter.graph.order()
        from ..ui.recipe import catalog_for_ui
        catalog = catalog_for_ui()
        try:
            configuration = catalog.encode({'graph':adapter.graph, 'training_plan':adapter.training_plan, 'target_spec':adapter.target_spec})
        except TypeError:
            configuration = None  # explicit Python-only extension, retained by trusted joblib
        components = {}
        def visit(value):
            if isinstance(value, dict):
                if 'component' in value:
                    key = value['component']
                    constructor = catalog.entries[key]['constructor']
                    try:
                        code = inspect.getsource(constructor)
                        fingerprint = hashlib.sha256(code.encode()).hexdigest()
                    except (OSError, TypeError):
                        fingerprint = None
                    components[key] = dict(implementation_sha256=fingerprint)
                for item in value.values():
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(configuration)
        manifest = dict(version=1, required_components=list(self.required_components),
                        configuration=configuration, portability='registered_configuration' if configuration is not None else 'trusted_python_extension',
                        components=components,
                        dependencies={name:importlib.metadata.version(name) for name in ('numpy','pandas','scikit-learn','joblib')},
                        nodes={name:dict(adapter=type(model).__module__+'.'+type(model).__qualname__) for name,model in adapter.models_.items()},
                        outputs={name:dict(node=ref.node,output=ref.output) for name,ref in adapter.graph.outputs.items()},
                        training_summary=adapter.training_summary_)
        (Path(directory)/'composition.json').write_text(json.dumps(manifest,default=str,indent=2),encoding='utf-8')
        JoblibSerializer().save(adapter,directory)

    def load(self,directory):
        manifest = json.loads((Path(directory)/'composition.json').read_text(encoding='utf-8'))
        if manifest.get('version') != 1 or tuple(manifest.get('required_components',())) != tuple(self.required_components):
            raise ValueError('Unsupported graph version or missing trusted component registrations.')
        from ..ui.recipe import catalog_for_ui
        catalog = catalog_for_ui()
        if any(key not in catalog.entries for key in (*self.required_components, *manifest.get('components',{}))):
            raise ValueError('Install/register required components explicitly before loading.')
        adapter = JoblibSerializer().load(directory)
        from .adapter import CompositeModelAdapter
        if not isinstance(adapter, CompositeModelAdapter):
            raise ValueError('Composite artifact restored an incompatible adapter.')
        adapter.graph.order()
        if set(adapter.models_) != set(manifest['nodes']):
            raise ValueError('Fitted graph children differ from manifest.')
        outputs = {name:dict(node=ref.node,output=ref.output) for name,ref in adapter.graph.outputs.items()}
        if outputs != manifest['outputs']:
            raise ValueError('Fitted graph public outputs differ from manifest.')
        return adapter
