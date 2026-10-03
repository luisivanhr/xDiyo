"""Native report shell around exact, already-calculated Plotly/CSV evidence."""
import base64
from copy import deepcopy
from datetime import timedelta
from html import escape
import json
from pathlib import Path
import re

from .common import (manifest, require_text, decimal, reconcile, timestamp, read_csv,
                     json_read, js_json, digest, output_directory, attribution_files,
                     write_evidence)


def vector(value):
    """Decode Plotly 6 typed-array payloads for validation only; never rewrite them."""
    if isinstance(value, dict) and 'bdata' in value:
        import numpy as np
        dtype = np.dtype(value['dtype'])
        if dtype.kind not in 'iuf' or dtype.itemsize not in (1, 2, 4, 8) or value.get('shape'):
            raise ValueError('Expected a one-dimensional numeric Plotly vector.')
        return np.frombuffer(base64.b64decode(value['bdata'], validate=True), dtype=dtype).tolist()
    if not isinstance(value, list):
        raise ValueError('Expected a Plotly vector.')
    return value


def validate_figure(spec, item, root, tolerance):
    if not isinstance(spec, dict) or not isinstance(spec.get('data'), list) or not isinstance(spec.get('layout'), dict):
        raise ValueError('Figure requires data and layout.')
    if spec.get('frames'):
        raise ValueError('Animated figures are outside this static evidence format.')
    # These formats use exported local curves, not remote map/image resources.
    def offline(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == 'source' and child:
                    raise ValueError('Remote/image/map Plotly resources are not supported.')
                offline(child)
        elif isinstance(value, list):
            for child in value:
                offline(child)
    offline(spec)
    if any(t.get('type', 'scatter') not in ('scatter', 'scattergl', 'bar') for t in spec['data']):
        raise ValueError('Saved curve figures support scatter, scattergl and bar traces.')
    ends = item.get('endpoints')
    if not isinstance(ends, list) or len(ends) != len(spec['data']):
        raise ValueError('Supply endpoints for every trace, in source trace order (null for an empty trace).')
    for n, (trace, end) in enumerate(zip(spec['data'], ends)):
        x, y = vector(trace.get('x', [])), vector(trace.get('y', []))
        if len(x) != len(y):
            raise ValueError(f'Trace {n}: unequal x/y lengths.')
        if not y:
            if end is not None:
                raise ValueError(f'Trace {n}: empty trace endpoint must be null.')
            continue
        for v in y:
            decimal(str(v), f'trace {n} value')
        reconcile(decimal(str(y[-1]), f'trace {n} end'), decimal(end, f'endpoints[{n}]'), tolerance, f'trace {n} endpoint')
    for menu in spec['layout'].get('updatemenus', []):
        for button in menu.get('buttons', []):
            for arg in button.get('args', [])[:1]:
                if isinstance(arg, dict) and isinstance(arg.get('visible'), list) and len(arg['visible']) != len(spec['data']):
                    raise ValueError('Selector visible mask does not match trace count.')
    checks = []
    for check in item.get('curve_tables', []):
        index = check['trace']
        if type(index) is not int or not 0 <= index < len(spec['data']):
            raise ValueError('curve_tables trace must be a valid zero-based index.')
        raw, headers, rows = read_csv(root / check['csv'])
        xc, yc = headers.index(check['x']), headers.index(check['y'])
        if check.get('where'):
            for col, value in check['where'].items():
                rows = [row for row in rows if row[headers.index(col)] == value]
        x, y = vector(spec['data'][index]['x']), vector(spec['data'][index]['y'])
        if len(rows) != len(y):
            raise ValueError(f'Trace {index}: source CSV row count differs.')
        dates = []
        for i, row in enumerate(rows):
            a = timestamp(row[xc], 'curve CSV timestamp')
            if a != timestamp(x[i], 'figure timestamp'):
                raise ValueError(f'Trace {index} point {i}: timestamp mismatch.')
            dates.append(a)
            reconcile(decimal(str(y[i]), 'figure y'), decimal(row[yc], 'CSV y'), tolerance, f'trace {index} point {i}')
        if any(b < a for a, b in zip(dates, dates[1:])):
            raise ValueError('Curve timestamps must retain chronological order; equal timestamps are allowed.')
        if check.get('calendar'):
            calendar = check['calendar']
            start, end = timestamp(calendar['start'], 'calendar start'), timestamp(calendar['end'], 'calendar end')
            if not dates or dates[0] != start or dates[-1] != end or any(b-a != timedelta(days=1) for a,b in zip(dates,dates[1:])):
                raise ValueError('Calendar coverage must include every supplied UTC day, without gaps.')
            if calendar.get('initial_zero') and decimal(rows[0][yc], 'initial zero') != 0:
                raise ValueError('Calendar initial zero is missing; the renderer will not insert one.')
        checks.append({'csv': check['csv'], 'sha256': digest(raw), 'trace': index, 'points': len(rows)})
    return checks


def _build(manifest_path, output_path, kind):
    path, m = manifest(manifest_path, kind)
    root = path.parent
    for key in ('scope_label', 'timing_note', 'units', 'aggregation_note'):
        require_text(m.get(key), key)
    js_path = root / require_text(m.get('plotly_js'), 'plotly_js')
    runtime = js_path.read_bytes()
    version = re.search(rb'plotly\.js v([0-9.]+)', runtime[:600])
    if not version or version[1].decode() != m.get('plotly_js_version'):
        raise ValueError('plotly_js_version must match the supplied local Plotly runtime header.')
    tolerance = decimal(m.get('numeric_tolerance', '0'), 'numeric_tolerance')
    if tolerance < 0:
        raise ValueError('numeric_tolerance cannot be negative.')
    sections = m.get('sections')
    if not isinstance(sections, list) or not sections:
        raise ValueError('sections must be a nonempty list.')
    attribution = attribution_files(m, root)
    portable = deepcopy(m)
    portable['plotly_js'] = 'plotly.min.js'
    portable['attribution_files'] = [name for name, _ in attribution]
    inputs = [{'plotly_js_version': m['plotly_js_version'], 'sha256': digest(runtime)}]
    from .. import Artifact, StudyResult, StudyRun, AnalysisReport
    runs, files, seen = [], [], set()
    for section_index, section in enumerate(sections):
        name = require_text(section.get('title'), 'section title')
        if name in seen:
            raise ValueError('Section titles must be distinct.')
        seen.add(name)
        artifacts = []
        notes = [m['timing_note'], m['units'], m['aggregation_note'], *section.get('notes', [])]
        for figure_index, item in enumerate(section.get('figures', [])):
            portable_figure = portable['sections'][section_index]['figures'][figure_index]
            revision = require_text(item.get('producer_revision'), 'figure producer_revision')
            if item.get('plotly_js_version') != m['plotly_js_version']:
                raise ValueError('Every figure must declare the matching supplied runtime version; use separate bundles for incompatible versions.')
            raw = (root / item['json']).read_bytes()
            spec = json_read(raw.decode('utf-8-sig'))
            checks = validate_figure(spec, item, root, tolerance)
            index = len(files)
            filename = f'evidence/figure-{index:04d}.plotly.json'
            files.append((filename, raw))
            portable_figure['json'] = filename
            inputs.append({'figure': item['json'], 'producer_revision': revision,
                           'sha256': digest(raw), 'trace_count': len(spec['data']), 'curve_tables': checks})
            for check_index, check in enumerate(item.get('curve_tables', [])):
                csv_name = f'evidence/curve-{len(files):04d}.csv'
                files.append((csv_name, (root / check['csv']).read_bytes()))
                portable_figure['curve_tables'][check_index]['csv'] = csv_name
            spec_id = f'shared-figure-{index}'
            height = spec['layout'].get('height', 430)
            if isinstance(height, bool) or not isinstance(height, (int, float)) or not 100 <= height <= 10000:
                raise ValueError('Saved figure height must be numeric, between 100 and 10000 pixels.')
            fragment = (f'<div class="shared-figure-viewport" style="max-width:100%;overflow-x:auto">'
                        f'<div class="plot" data-spec="{spec_id}" style="height:{height}px"></div></div>'
                        f'<script type="application/json" id="{spec_id}">{js_json(spec)}</script>'
                        f'<a class="download" download href="{filename}">Download original Plotly JSON</a>')
            artifacts.append(Artifact('html', fragment, item.get('title', name)))
            notes.append(f'Figure producer revision: {revision}')
        for table_index, table in enumerate(section.get('tables', [])):
            raw, headers, rows = read_csv(root / table['csv'])
            filename = f'evidence/table-{len(files):04d}.csv'
            files.append((filename, raw))
            portable['sections'][section_index]['tables'][table_index]['csv'] = filename
            inputs.append({'table': table['csv'], 'sha256': digest(raw), 'rows': len(rows)})
            # Same native table classes, string-preserving preview/download.
            heads = ''.join(f'<th>{escape(h)}</th>' for h in headers)
            body = ''.join('<tr>'+''.join(f'<td>{escape(v)}</td>' for v in row)+'</tr>' for row in rows[:200])
            fragment = (f'<p class="muted">{len(rows)} source rows; preview limited to 200.</p>'
                        f'<div class="table-scroll"><table><thead><tr>{heads}</tr></thead><tbody>{body}</tbody></table></div>'
                        f'<a class="download" download href="{filename}">Download original CSV</a>')
            artifacts.append(Artifact('html', fragment, table.get('title', 'Supplied table')))
        if not artifacts:
            raise ValueError(f'{name}: include at least one figure or table.')
        runs.append(StudyRun(name, 'overall', 'all', None, 'exported evidence', [], 0,
                            StudyResult(name, artifacts=artifacts, notes=notes), scope_label=m['scope_label']))
    report = AnalysisReport(runs, title=m['title'])
    html = report.to_html()
    # Public HTML contract, no private style/plotting helper imports. Runtime
    # matches original JSON rather than the currently installed Plotly version.
    html = html.replace('</head>', '<script src="plotly.min.js"></script></head>', 1)
    out = output_directory(output_path)
    (out / 'evidence').mkdir()
    for name, raw in files:
        (out / name).write_bytes(raw)
    (out / 'plotly.min.js').write_bytes(runtime)
    (out / 'INDEX.html').write_text(html, encoding='utf-8')
    write_evidence(out, portable, inputs, attribution)
    return out / 'INDEX.html'


def build_curves(manifest_path, output_path):
    """Preserve supplied strategy curves/tables; never derive profits or risk."""
    return _build(manifest_path, output_path, 'curves')


def build_portfolio(manifest_path, output_path):
    """Display an already-aggregated portfolio with its declared overlap policy."""
    return _build(manifest_path, output_path, 'portfolio')
