"""Build the preserved paged ticket explorer from general exported leg arrays."""
import base64
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

from .common import (manifest, require_text, decimal, exact_sum, reconcile, timestamp,
                     read_csv, json_read, js_json, encoded, digest, output_directory,
                     attribution_files, write_evidence)

ASSETS = Path(__file__).with_name('assets')
REQUIRED = {'ticket_id', 'league', 'result', 'first_kickoff_utc', 'legs',
            'stake_units', 'payout_units', 'profit_units'}
OPTIONAL = {'season', 'round', 'stage', 'combined_odds', 'settlement_time', 'timing_note', 'details'}


def legacy_draw_pair(row):
    """Explicit adapter for Ayre's flat audited two-draw-leg CSV, never gzip reconstruction."""
    legs = []
    for n in (1, 2):
        prefix = f'leg{n}_'
        leg = {'market': 'Match outcome', 'selection': 'Draw'}
        for target, suffix in [('event_id','event_id'), ('kickoff_utc','kickoff_utc'),
                               ('home_id','home_id'), ('away_id','away_id'),
                               ('home_name','home_name'), ('away_name','away_name'),
                               ('home_score','home_score_current'), ('away_score','away_score_current'),
                               ('odds','opening_draw_odds'), ('probability','draw_probability'),
                               ('result','draw_settlement')]:
            value = row.get(prefix + suffix)
            if value not in (None, ''):
                leg[target] = value
        for key in ('competition_id', 'season_id'):
            if row.get(key):
                leg[key] = row[key]
        legs.append(leg)
    return json.dumps(legs)


def badge_uri(value):
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError('Badge must be an image data URI.')
    match = re.fullmatch(r'data:image/(png|jpeg|webp|svg\+xml);base64,([A-Za-z0-9+/=\s]+)', value)
    if not match:
        raise ValueError('Badge must be an embedded PNG, JPEG, WebP or SVG data URI.')
    raw = base64.b64decode(re.sub(r'\s', '', match[2]), validate=True)
    if len(raw) > 2_000_000:
        raise ValueError('Badge exceeds 2 MB.')
    kind = match[1]
    valid = (kind == 'png' and raw.startswith(b'\x89PNG\r\n\x1a\n') or
             kind == 'jpeg' and raw.startswith(b'\xff\xd8\xff') or
             kind == 'webp' and raw[:4] == b'RIFF' and raw[8:12] == b'WEBP')
    if kind == 'svg+xml':
        source = raw.decode('utf-8-sig')
        if re.search(r'<!\s*(?:DOCTYPE|ENTITY)|@import', source, re.I):
            raise ValueError('SVG badge contains declarations or external styles.')
        root = ET.fromstring(source)
        if root.tag.split('}')[-1] != 'svg':
            raise ValueError('Badge is not SVG.')
        for node in root.iter():
            if node.tag.split('}')[-1].lower() in {'script', 'foreignobject', 'iframe', 'object', 'embed', 'animate', 'set'}:
                raise ValueError('SVG badge contains active content.')
            for key, text in node.attrib.items():
                key = key.split('}')[-1].lower()
                embedded_raster = text.startswith(('data:image/png;base64,', 'data:image/jpeg;base64,'))
                if embedded_raster:
                    badge_uri(text)
                if key.startswith('on') or (key in {'href', 'src'} and not text.startswith('#') and not embedded_raster):
                    raise ValueError('SVG badge contains active/external references.')
            for url in re.findall(r'url\(([^)]*)\)', ' '.join([node.text or '', *node.attrib.values()]), re.I):
                if not url.strip().strip('\"\'').startswith('#'):
                    raise ValueError('SVG badge contains external styles.')
        valid = True
    if not valid:
        raise ValueError('Badge data does not match its image type.')
    return f'data:image/{kind};base64,{encoded(raw)}'


def read_badges(m, root):
    badges = dict(m.get('badges', {}))
    if m.get('team_catalog'):
        from .. import TeamCatalog
        catalog = TeamCatalog.from_json(root / m['team_catalog'])
        for key in catalog.entries:
            name, badge, note = catalog.display(key, badges=True)
            badges.setdefault(key, {'name': name, 'badge': badge, 'note': note})
    for key, info in badges.items():
        if not isinstance(info, dict):
            raise ValueError(f'Badge record {key!r} must be an object.')
        info['badge'] = badge_uri(info.get('badge'))
    if any(x['badge'] for x in badges.values()) and not m.get('attribution_files'):
        raise ValueError('Include attribution_files when redistributing badges.')
    return badges


def validate_legs(raw, name):
    legs = json_read(raw)
    if not isinstance(legs, list) or not legs:
        raise ValueError(f'{name}: legs must be a nonempty JSON array.')
    for i, leg in enumerate(legs):
        if not isinstance(leg, dict):
            raise ValueError(f'{name} leg {i+1}: expected an object.')
        for key in ('event_id', 'kickoff_utc', 'market', 'selection'):
            require_text(leg.get(key), f'{name} leg {i+1} {key}')
        timestamp(leg['kickoff_utc'], f'{name} leg {i+1}')
        for key in ('home_id', 'away_id', 'competition_id', 'season_id'):
            if key in leg and leg[key] is not None:
                require_text(leg[key], f'{name} leg {key}')
        for key in ('odds', 'probability', 'home_score', 'away_score'):
            if leg.get(key) is not None:
                number = decimal(leg[key], f'{name} leg {key}')
                if (key == 'probability' and not 0 <= number <= 1) or (key == 'odds' and number < 1):
                    raise ValueError(f'{name}: invalid {key}.')
    return legs


def load_policy(policy, root, tolerance):
    fields = policy.get('fields', {})
    adapter = policy.get('adapter', 'general')
    if adapter not in ('general', 'legacy_draw_pairs'):
        raise ValueError('adapter must be general or legacy_draw_pairs.')
    required = REQUIRED - {'legs'} if adapter == 'legacy_draw_pairs' else REQUIRED
    if adapter == 'legacy_draw_pairs' and 'legs' in fields:
        raise ValueError('legacy_draw_pairs supplies legs from original leg1_/leg2_ columns; omit the legs mapping.')
    if not isinstance(fields, dict) or required - fields.keys() or fields.keys() - (REQUIRED | OPTIONAL):
        raise ValueError(f'{policy.get("id")}: fields need {sorted(REQUIRED)}; unknown mapping keys are rejected.')
    raw, headers, source_rows = read_csv(root / require_text(policy.get('csv'), 'policy csv'))
    if any(name not in headers for name in fields.values()):
        raise ValueError(f'{policy["id"]}: missing mapped CSV columns.')
    positions = {key: headers.index(name) for key, name in fields.items()}
    records, ids = [], set()
    amounts = {key: [] for key in ('stake_units', 'payout_units', 'profit_units')}
    for n, row in enumerate(source_rows, 2):
        r = {key: row[index] for key, index in positions.items()}
        label = f'{policy["id"]} CSV row {n}'
        tid = require_text(r['ticket_id'], label + ' ticket_id')
        if tid in ids:
            raise ValueError(f'{label}: duplicate ticket_id {tid!r}.')
        ids.add(tid)
        r['legs'] = validate_legs(legacy_draw_pair(dict(zip(headers, row)))
                                  if adapter == 'legacy_draw_pairs' else r['legs'], label)
        date = timestamp(r['first_kickoff_utc'], label)
        r['filter_date'] = date.date().isoformat()
        if r.get('settlement_time'):
            timestamp(r['settlement_time'], label + ' settlement_time')
        if r.get('combined_odds'):
            if decimal(r['combined_odds'], label + ' combined_odds') < 1:
                raise ValueError(f'{label}: combined_odds must be at least 1.')
        for key in amounts:
            amounts[key].append(decimal(r[key], label + ' ' + key))
        if amounts['stake_units'][-1] < 0 or amounts['payout_units'][-1] < 0:
            raise ValueError(f'{label}: stake/payout cannot be negative.')
        reconcile(exact_sum([amounts['payout_units'][-1], amounts['stake_units'][-1].copy_negate()]),
                  amounts['profit_units'][-1], tolerance, label + ' payout - stake = profit')
        r['source'] = row
        records.append(r)
    summary = policy.get('summary', {})
    if type(summary.get('rows')) is not int or summary['rows'] != len(records):
        raise ValueError(f'{policy["id"]}: summary rows must equal {len(records)}.')
    for key, values in amounts.items():
        reconcile(exact_sum(values), decimal(summary.get(key), 'summary ' + key), tolerance, 'summary ' + key)
    if 'roi_percent' in summary:
        decimal(summary['roi_percent'], 'summary roi_percent')
    return raw, {'headers': headers, 'rows': records, 'csv_base64': encoded(raw)}


def build_tickets(manifest_path, output_path):
    """Validate a v1 manifest and write a file://-friendly INDEX.html bundle.

    The destination must be empty. Full downloads preserve source CSV bytes.
    Validation precedes writes; original evidence is never modified.
    """
    path, m = manifest(manifest_path)
    root = path.parent
    tolerance = decimal(m.get('accounting_tolerance', '0'), 'accounting_tolerance')
    if tolerance < 0:
        raise ValueError('accounting_tolerance must be nonnegative.')
    policies = m.get('policies')
    if not isinstance(policies, list) or not policies:
        raise ValueError('policies must be a nonempty list.')
    ids, index, chunks, inputs, csv_files = set(), [], [], [], []
    portable = deepcopy(m)
    badges, attribution = read_badges(m, root), attribution_files(m, root)
    links = []
    for i, link in enumerate(m.get('links', [])):
        source = root / require_text(link.get('file'), 'link file')
        if source.suffix.lower() not in ('.txt', '.md', '.json', '.csv', '.pdf'):
            raise ValueError('Linked attachments must be local TXT, Markdown, JSON, CSV or PDF evidence.')
        links.append((f'attachments/{i:03d}-{source.name}', source.read_bytes(), require_text(link.get('label'), 'link label')))
    for i, policy in enumerate(policies):
        key = require_text(policy.get('id'), 'policy id')
        if key in ids:
            raise ValueError(f'Duplicate policy id: {key}')
        ids.add(key)
        raw, data = load_policy(policy, root, tolerance)
        filename = f'policies/policy-{i:05d}.js'
        index.append({'id': key, 'label': require_text(policy.get('label'), 'policy label'),
                      'group': require_text(policy.get('group'), 'policy group'),
                      'summary': policy['summary'], 'file': filename, 'note': policy.get('note', '')})
        chunks.append((filename, f'window.XDIYO_TICKET_CHUNK({js_json(key)},{js_json(data)});'))
        csv_name = f'evidence/policy-{i:05d}.csv'
        csv_files.append((csv_name, raw))
        portable['policies'][i]['csv'] = csv_name
        inputs.append({'policy': key, 'csv': policy['csv'], 'sha256': digest(raw), 'rows': len(data['rows'])})
    out = output_directory(output_path)
    (out / 'policies').mkdir()
    (out / 'evidence').mkdir()
    for name, raw in csv_files:
        (out / name).write_bytes(raw)
    for name, raw, _ in links:
        (out / name).parent.mkdir(exist_ok=True)
        (out / name).write_bytes(raw)
    for name, data in chunks:
        (out / name).write_text(data, encoding='utf-8')
    for name in ('tickets.css', 'tickets.js'):
        (out / name).write_bytes((ASSETS / name).read_bytes())
    (out / 'index.js').write_text('window.TICKET_INDEX=' + js_json({'policies': index, 'badges': badges}) + ';', encoding='utf-8')
    html = (ASSETS / 'tickets.html').read_text(encoding='utf-8')
    texts = {key: escape(str(m.get(key, '')), quote=True) for key in ('title', 'subtitle', 'notice', 'footer')}
    texts['attribution'] = ' · '.join(f'<a href="{escape(name, quote=True)}">{escape(Path(name).name)}</a>' for name, _ in attribution)
    texts['links'] = ' · '.join(f'<a href="{escape(name, quote=True)}">{escape(label)}</a>' for name, _, label in links)
    # Single pass: supplied text cannot introduce another template placeholder.
    html = re.sub(r'\{\{(\w+)\}\}', lambda x: texts[x[1]], html)
    (out / 'INDEX.html').write_text(html, encoding='utf-8')
    portable.pop('team_catalog', None)
    portable['badges'] = badges
    portable['attribution_files'] = [name for name, _ in attribution]
    portable['links'] = [{'file': name, 'label': label} for name, _, label in links]
    write_evidence(out, portable, inputs, attribution)
    return out / 'INDEX.html'
