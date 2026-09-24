"""Loopback-only UI service. Reports reuse retained results; jobs call the library."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import re
import secrets
import threading
from urllib.parse import urlparse
import uuid
import webbrowser

from .recipe import (catalog_for_ui, default_recipe, export_notebook, export_python,
                     prepare_recipe, predict_recipe, run_recipe, validate_recipe)
from .schema import stage_schema


def render_result_report(result, recipe):
    """Refresh local badge presentation without fitting or changing saved results."""
    from dataclasses import replace
    from ..reporting import TeamCatalog

    path = Path(recipe.get('team_badges') or
                Path(__file__).resolve().parents[3] / 'docs/analytics/team_assets/catalog.json')
    badges = TeamCatalog.from_json(path) if path.is_file() else TeamCatalog({})
    specs = recipe.get('post_reporters', {})
    report = result.report
    refreshed = []
    for study in report.studies:
        name = study.name.removeprefix('post/')
        enabled = specs.get(name, {}).get('params', {}).get('show_badges', True)
        artifacts = []
        for artifact in study.result.artifacts:
            if artifact.kind == 'match_results':
                teams = {}
                for key, entry in artifact.options.get('teams', {}).items():
                    _, badge, _ = badges.display(key, badges=enabled)
                    teams[key] = {**entry, 'badge': (badge or entry.get('badge')) if enabled else None}
                artifact = replace(artifact, options={**artifact.options, 'teams': teams})
            artifacts.append(artifact)
        refreshed.append(replace(study, result=replace(study.result, artifacts=artifacts)))
    return replace(report, studies=refreshed).to_html()


def _frame(frame, limit=40):
    # JSON decimal rendering must never round identity integers via a float cast.
    import pandas as pd
    def value(v):
        if v is None or v is pd.NA or v is pd.NaT:
            return None
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v) if abs(v) > 2**53 - 1 else v
        if isinstance(v, (str, bool)):
            return v
        if hasattr(v, 'item'):
            v = v.item()
        if isinstance(v, int):
            return str(v) if abs(v) > 2**53 - 1 else v
        if isinstance(v, float):
            import math
            return v if math.isfinite(v) else None
        return str(v)
    return {'columns': [str(c) for c in frame.columns],
            'rows': [[value(v) for v in row] for row in frame.head(limit).itertuples(index=False, name=None)],
            'count': len(frame)}


def _preparation_preview(prepared):
    """Actual assembled columns for inspection and reporter selection."""
    return {'features': _frame(prepared.dataset.X), 'labels': _frame(prepared.dataset.y),
            'metadata': _frame(prepared.dataset.metadata),
            'classes': json.loads(prepared.dataset.y.stack().drop_duplicates().head(100).to_json(orient='values')),
            'folds': [{'fold': i, 'train': len(f.train), 'test': len(f.test), 'score': len(f.score),
                       'description': _fold_description(prepared.dataset.metadata, f)}
                      for i, f in enumerate(prepared.split_plan.folds)]}


def _fold_description(metadata, fold):
    """Human-readable population and season context for the fold picker."""
    train, test = metadata.iloc[fold.train], metadata.iloc[fold.test]
    parts = []
    league_column = 'source_league' if 'source_league' in metadata else 'competition_id'
    if league_column in test:
        leagues = test[league_column].dropna().unique()
        if len(leagues) == 1:
            parts.append(str(leagues[0]).replace('_', ' '))
        elif len(leagues):
            parts.append(f'{len(leagues)} leagues')
    if 'source_season' in metadata:
        for title, rows in [('train', train), ('test', test)]:
            if 'kickoff_at' in rows:
                rows = rows.sort_values('kickoff_at', kind='stable')
            seasons = rows.source_season.dropna().astype(str).drop_duplicates().tolist()
            if seasons:
                label = ', '.join(s.replace('_', '/') for s in seasons)
                parts.append(f'{title} {label}')
    if fold.metadata.get('gap_unit') == 'rounds' and fold.metadata.get('gap') and 'round' in test:
        import pandas as pd
        rounds = pd.to_numeric(test['round'], errors='coerce').dropna()
        if len(rounds):
            parts.append(f'test rounds {rounds.min():g}–{rounds.max():g}')
    return ' · '.join(parts)


class BuilderState:
    def __init__(self, workspace, catalog):
        self.workspace = Path(workspace).resolve()
        self.catalog = catalog
        self.token = secrets.token_urlsafe(32)
        self.jobs = {}
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='football-ui')
        self.recipes = self.workspace / 'experiments' / '_recipes'
        self.discovery_cache = {}

    def resolve(self, path):
        path = Path(path)
        return path.resolve() if path.is_absolute() else (self.workspace / path).resolve()

    def recipe_paths(self, recipe):
        recipe = validate_recipe(recipe)
        # Resolve known paths without chdir (which would affect notebook threads).
        recipe['data']['data_root'] = str(self.resolve(recipe['data']['data_root']))
        recipe['output_dir'] = str(self.resolve(recipe.get('output_dir', 'experiments')))
        if recipe.get('team_badges'):
            recipe['team_badges'] = str(self.resolve(recipe['team_badges']))
        if recipe['data'].get('record_dir'):
            recipe['data']['record_dir'] = str(self.resolve(recipe['data']['record_dir']))
        if recipe.get('prediction', {}).get('model_path'):
            recipe['prediction']['model_path'] = str(self.resolve(recipe['prediction']['model_path']))
        def paths(value):
            if isinstance(value, list):
                return [paths(v) for v in value]
            if isinstance(value, dict):
                value = {k: paths(v) for k, v in value.items()}
                if value.get('component', '').startswith('input.') and value.get('params', {}).get('path'):
                    value['params']['path'] = str(self.resolve(value['params']['path']))
            return value
        return paths(recipe)

    def start(self, action, recipe):
        job_id = str(uuid.uuid4())
        job = {'id': job_id, 'action': action, 'status': 'queued', 'message': 'Waiting for the current job', 'recipe': self.recipe_paths(recipe)}
        self.jobs[job_id] = job
        def work():
            job.update(status='running', message={'prepare': 'Loading histories and preparing features and folds',
                       'run': 'Preparing, fitting and generating reports', 'predict': 'Loading saved model and aligning fixtures'}[action])
            try:
                if action == 'prepare':
                    result = prepare_recipe(job['recipe'], catalog=self.catalog)
                    job['object'] = result
                    job['preview'] = _preparation_preview(result)
                elif action == 'run':
                    previous = next((j.get('object') for j in reversed(list(self.jobs.values()))
                                     if j['action'] == 'prepare' and j['status'] == 'complete' and j['recipe'] == job['recipe']), None)
                    result = run_recipe(job['recipe'], prepared=previous, catalog=self.catalog)
                    job['object'] = result
                    job['html'] = result.to_html()
                    job['path'] = str(result.path)
                    job['reused'] = result.reused
                    job['preview'] = _preparation_preview(result.prepared)
                else:
                    dataset, predictions = predict_recipe(job['recipe'], catalog=self.catalog)
                    job['object'] = (dataset, predictions)
                    job['preview'] = {'metadata': _frame(dataset.metadata), **{k: _frame(v) for k, v in predictions.items()}}
                job.update(status='complete', message='Completed')
            except Exception as exc:
                job.update(status='failed', message=f'{type(exc).__name__}: {exc}')
        job['future'] = self.executor.submit(work)
        return {'id': job_id}

    def dispatch(self, route, request):
        from ..data import inspect_season, list_stat_bundles
        if route == 'catalog':
            from .inventory import inventory
            return {'components': self.catalog.schema(), 'stages': stage_schema(),
                    'recipe': default_recipe(str(self.workspace / 'data/xDiyo_data')),
                    'bundles': list_stat_bundles(), 'metrics': inventory()['metrics']}
        if route == 'discover':
            root = self.resolve(request['root'])
            if not root.is_dir():
                raise ValueError(f'Data folder does not exist: {root}')
            publications = []
            for path in sorted(root.glob('*.manifest.json')):
                stem = path.name.removesuffix('.manifest.json')
                parts = stem.rsplit('_', 2)
                if len(parts) == 3:
                    publications.append({'stem': stem, 'league': parts[0], 'season': '_'.join(parts[1:])})
            return {'publications': publications}
        if route == 'choices':
            from ..data._source import _open_source
            import pandas as pd
            root = self.resolve(request['root'])
            publications = self.dispatch('discover', request)['publications']
            selected = [p for p in publications if p['season'] in request.get('seasons', [])
                        and (request.get('leagues') is None or p['league'] in request['leagues'])]
            identities, teams, rounds = set(), {}, set()
            for publication in selected:
                stem = publication['stem']
                manifest = root / (stem + '.manifest.json')
                cache_key = (str(manifest), manifest.stat().st_mtime_ns)
                if cache_key not in self.discovery_cache:
                    source = _open_source(root, stem)
                    stats, names, round_values = [], {}, []
                    if 'statistics' in source.manifest['tables']:
                        frame = pd.read_parquet(source.table_path('statistics'), columns=['period', 'group_name', 'key'])
                        stats = list(frame.drop_duplicates().itertuples(index=False, name=None))
                    if 'matches' in source.manifest['tables']:
                        import pyarrow.parquet as pq
                        path = source.table_path('matches')
                        columns = pq.read_schema(path).names
                        use = [c for c in ('home_id','home_name','away_id','away_name','round') if c in columns]
                        frame = pd.read_parquet(path, columns=use)
                        for side in ('home', 'away'):
                            if {f'{side}_id',f'{side}_name'} <= set(frame):
                                names.update({str(i): str(n) for i,n in frame[[f'{side}_id',f'{side}_name']].dropna().drop_duplicates().itertuples(index=False,name=None)})
                        if 'round' in frame: round_values = frame['round'].dropna().unique().tolist()
                    self.discovery_cache[cache_key] = (stats, names, round_values)
                stats, names, round_values = self.discovery_cache[cache_key]
                identities.update(stats)
                teams.update(names)
                rounds.update(round_values)
            return {'stats': [dict(period=p,group_name=g,key=k) for p,g,k in sorted(identities)],
                    'teams': [{'value': k, 'label': v} for k,v in sorted(teams.items(),key=lambda x:x[1])],
                    'rounds': sorted(rounds, key=str), 'publications': selected}
        if route == 'inspect':
            from ..data._source import _open_source
            import pandas as pd
            root = self.resolve(request['root'])
            source = _open_source(root, request['stem'])
            overview = inspect_season(root, request['stem'])
            stats = []
            if 'statistics' in source.manifest['tables']:
                frame = pd.read_parquet(source.table_path('statistics'), columns=['period', 'group_name', 'key'])
                stats = frame.drop_duplicates().to_dict('records')
            overview = overview.reset_index()
            overview = overview.loc[overview['table'].isin(['matches','statistics','pregame','shots'])]
            return {'tables': _frame(overview[['table','rows','description']]), 'stats': stats}
        if route == 'save':
            recipe = validate_recipe(request['recipe'])
            name = re.sub(r'[^A-Za-z0-9_-]+', '_', request.get('name') or recipe['name']).strip('_')
            if not name:
                raise ValueError('Choose a recipe file name.')
            self.recipes.mkdir(parents=True, exist_ok=True)
            path = self.recipes / (name + '.json')
            temporary = path.with_suffix('.tmp')
            temporary.write_text(json.dumps(recipe, indent=2, ensure_ascii=False), encoding='utf-8')
            temporary.replace(path)
            return {'path': str(path)}
        if route == 'recipes':
            return {'recipes': [{'file': p.name, 'name': json.loads(p.read_text(encoding='utf-8'))['name']}
                                for p in sorted(self.recipes.glob('*.json'))]}
        if route == 'open':
            name = request['file']
            if Path(name).name != name or not name.endswith('.json'):
                raise ValueError('Choose a saved recipe file.')
            return {'recipe': validate_recipe(json.loads((self.recipes / name).read_text(encoding='utf-8')))}
        if route == 'export':
            recipe = self.recipe_paths(request['recipe'])
            notebook = request.get('format') == 'notebook'
            return {'text': json.dumps(export_notebook(recipe), indent=2) if notebook else export_python(recipe),
                    'filename': 'football_experiment.ipynb' if notebook else 'football_experiment.py'}
        if route in ('prepare', 'run', 'predict'):
            return self.start(route, request['recipe'])
        if route == 'job':
            job = self.jobs[request['id']]
            return {k: v for k, v in job.items() if k not in ('object', 'recipe', 'future', 'html')}
        if route == 'report':
            job = self.jobs[request['id']]
            return {'html': render_result_report(job['object'], job['recipe'])}
        if route == 'runs':
            from ..experiments import ExperimentStore
            recipe = self.recipe_paths(request['recipe'])
            store = ExperimentStore(recipe['output_dir'], recipe['name'])
            return {'runs': [{'id': r['run_id'], 'name': r['name'], 'status': r['status']}
                             for r in store.read_runs()]}
        if route == 'load_run':
            from ..experiments.football import FootballExperiment
            recipe = self.recipe_paths(request['recipe'])
            experiment = FootballExperiment(recipe['name'], output_dir=recipe['output_dir'])
            result = experiment.load(request['id'])
            return {'html': render_result_report(result, recipe)}
        if route == 'save_model':
            from ..training import save_model
            result = self.jobs[request['id']]['object']
            model = result.refit if request.get('refit') else result.training
            if model is None:
                raise ValueError('This run has no final refitted model.')
            path = self.resolve(request['path'])
            save_model(model, path, fold_id=request.get('fold_id'))
            return {'path': str(path)}
        raise ValueError(f'Unknown action: {route}')


@dataclass
class BuilderHandle:
    server: object
    thread: object
    state: BuilderState

    @property
    def url(self):
        return f'http://127.0.0.1:{self.server.server_port}/#{self.state.token}'

    def show(self, height=850):
        from IPython.display import IFrame, display
        display(IFrame(self.url, width='100%', height=height))
        return self

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.state.executor.shutdown(wait=False, cancel_futures=True)


def launch_ui(workspace='.', *, port=0, open_browser=True, catalog=None):
    state = BuilderState(workspace, catalog or catalog_for_ui())
    assets = Path(__file__).parent / 'static'
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Never log the access token or recipe contents.

        def respond(self, payload, status=200, content_type='application/json'):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode() if content_type == 'application/json' else payload
            self.send_response(status)
            self.send_header('Content-Type', content_type + ('; charset=utf-8' if content_type.startswith(('text/', 'application/json')) else ''))
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            name = urlparse(self.path).path.lstrip('/') or 'index.html'
            if name == 'report.css':
                from ..reporting.viewer import CSS
                return self.respond(CSS.encode(), content_type='text/css')
            if name not in ('index.html', 'app.js', 'forms.js', 'feature-bundles.js', 'style.css'):
                return self.respond({'error': 'Not found'}, 404)
            self.respond((assets / name).read_bytes(), content_type=mimetypes.guess_type(name)[0] or 'text/plain')

        def do_POST(self):
            host = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != host or self.headers.get('Origin', 'http://' + host) != 'http://' + host:
                return self.respond({'error': 'Use this local builder URL.'}, 403)
            if not secrets.compare_digest(self.headers.get('X-Builder-Token', ''), state.token):
                return self.respond({'error': 'Open the URL returned by launch_ui().' }, 403)
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if length < 0 or length > 10_000_000:
                    raise ValueError('Request exceeds 10 MB.')
                request = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path.removeprefix('/api/')
                self.respond(state.dispatch(route, request))
            except Exception as exc:
                self.respond({'error': f'{type(exc).__name__}: {exc}'}, 400)
    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True, name='football-builder')
    handle = BuilderHandle(server, thread, state)
    thread.start()
    if open_browser:
        webbrowser.open(handle.url)
    return handle


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Open the local football experiment builder.')
    parser.add_argument('--workspace', default='.')
    parser.add_argument('--port', type=int, default=0)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    handle = launch_ui(args.workspace, port=args.port, open_browser=not args.no_browser)
    print(handle.url, flush=True)
    try:
        handle.thread.join()
    except KeyboardInterrupt:
        handle.close()


if __name__ == '__main__':
    main()
