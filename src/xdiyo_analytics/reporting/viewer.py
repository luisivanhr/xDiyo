"""Standalone navigable HTML for arbitrary study collections and artifacts."""

from collections import OrderedDict
import base64
from html import escape
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .match_view import MATCH_CSS, MATCH_JS, render_match_results


def _e(value):
    return escape(str(value), quote=True)


def _csv_link(frame, name):
    content = frame.to_csv(index=False).encode("utf-8-sig")
    encoded = base64.b64encode(content).decode("ascii")
    return (f'<a class="download" download="{_e(name)}.csv" '
            f'href="data:text/csv;base64,{encoded}">Download CSV</a>')


def _table(frame, *, coefficient_columns=(), percentile_columns=(), preview=200):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Table artifacts must contain a pandas DataFrame.")
    heads = ''.join(f'<th><button type="button" data-sort="{i}" '
                    f'data-absolute="{str(col in coefficient_columns).lower()}">{_e(col)} ↕</button></th>'
                    for i, col in enumerate(frame.columns))
    rows = []
    for values in frame.head(preview).itertuples(index=False, name=None):
        cells = []
        for col, value in zip(frame.columns, values):
            missing = pd.isna(value) if np.isscalar(value) or value is pd.NA else False
            numeric = isinstance(value, (int, float, np.number)) and not missing
            text = "—" if missing else (f"{value:.4f}" if isinstance(value, (float, np.floating)) else str(value))
            bar = ""
            if numeric and col in coefficient_columns and np.isfinite(value):
                width = min(abs(float(value)), 1) * 50
                left = 50 if value >= 0 else 50 - width
                color = "positive" if value >= 0 else "negative"
                bar = (f'<span class="zero"></span><span class="bar {color}" '
                       f'style="left:{left:.5f}%;width:{width:.5f}%"></span>')
            elif numeric and col in percentile_columns and np.isfinite(value):
                bar = f'<span class="bar rank" style="left:0;width:{np.clip(value, 0, 100):.5f}%"></span>'
            sort_value = "" if missing else str(value)
            cells.append(f'<td data-value="{_e(sort_value)}" data-numeric="{str(numeric).lower()}">'
                         f'{bar}<span class="cell-value">{_e(text)}</span></td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    note = f'<p class="muted">Preview: first {preview} of {len(frame):,} rows. Download includes every row.</p>' if len(frame) > preview else ''
    return note + '<div class="table-scroll"><table><thead><tr>' + heads + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'


CSS = r"""
:root{color-scheme:dark;--bg:#0d141d;--panel:#17202b;--border:#2e3c4e;--muted:#9eb0c5;--ink:#e8eef7;--accent:#58c7b2}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif}
a{color:var(--accent)}button,input,select{font:inherit;color:inherit;background:#1d2a39;border:1px solid var(--border);border-radius:7px;padding:7px 11px}
button,select,summary{cursor:pointer}button:hover,a:hover{filter:brightness(1.15)}button:focus-visible,a:focus-visible,select:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid #b9a3ff;outline-offset:3px}
aside{position:fixed;inset:0 auto 0 0;width:265px;padding:25px 18px;background:#111b27;border-right:1px solid var(--border);overflow:auto}
.brand{font-weight:750;font-size:18px;letter-spacing:.02em}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:.14em;text-transform:uppercase}
aside input{width:100%;margin:22px 0 12px}nav a{display:block;padding:10px;border-radius:7px;color:var(--ink);text-decoration:none}nav a:hover{background:#223245}
main{margin-left:265px;padding:35px clamp(20px,4vw,65px);max-width:1800px}h1{font-size:32px;line-height:1.2;margin:10px 0}h2,h3{line-height:1.35}header{margin-bottom:28px}
.muted,.notes{color:var(--muted)}.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}.study{background:var(--panel);border:1px solid var(--border);border-radius:12px;margin:20px 0;scroll-margin-top:18px}
.study>summary{list-style-position:inside;padding:20px 24px;font-size:18px;font-weight:650}.study>summary small{font-weight:400;font-size:12px;color:var(--muted);margin-left:12px}
.study-content{padding:0 24px 24px}.scope-controls{display:flex;gap:15px;align-items:center;flex-wrap:wrap;margin:0 0 15px}.scope{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}
.badge{border:1px solid #405068;border-radius:30px;padding:3px 11px;font-size:12px;color:#c7d9ee}.plot{height:430px;min-width:0}.artifact{margin:18px 0;border-top:1px solid var(--border);padding-top:14px}
.artifact>summary{font-size:15px;font-weight:600}.table-scroll{overflow:auto;max-height:560px;margin:14px 0;border:1px solid var(--border);border-radius:8px}
table{border-collapse:separate;border-spacing:0;width:100%;font-variant-numeric:tabular-nums;font-size:13px}th{position:sticky;top:0;z-index:2;background:#243245;text-align:left;white-space:nowrap}th button{width:100%;border:0;background:transparent;text-align:left;font-size:12px;padding:13px}
td{position:relative;min-width:120px;padding:10px 12px;border-bottom:1px solid #293749;white-space:nowrap}td:first-child{font-weight:600}tr:hover td{background-color:#ffffff09}.bar{position:absolute;top:5px;bottom:5px;border-radius:3px;opacity:.42}.positive{background:#3bb999}.negative{background:#ed7975}.rank{background:#8d7bd9;opacity:.19}.zero{position:absolute;left:50%;top:5px;bottom:5px;border-left:1px solid #b9c6d233}.cell-value{position:relative;z-index:1}
.download{display:inline-block;border:1px solid #3c5269;border-radius:6px;padding:4px 10px;margin:4px 8px 4px 0;font-size:12px;text-decoration:none}.legend{font-size:12px;color:var(--muted)}.legend .pos{color:#69dfbf}.legend .neg{color:#f4a29d}.empty{padding:30px;border:1px dashed var(--border);border-radius:12px;color:var(--muted)}[hidden]{display:none!important}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.notes{font-size:13px}footer{color:var(--muted);font-size:12px;padding:25px 0}
@media(max-width:850px){aside{position:relative;width:auto;border-bottom:1px solid var(--border)}main{margin:0;padding:20px}aside nav{display:flex;flex-wrap:wrap}aside input{margin:10px 0}h1{font-size:25px}.study-content{padding:0 12px 16px}.plot{height:380px}}
@media print{aside,.toolbar,.download,.scope-controls{display:none}main{margin:0;padding:0}.study{break-inside:avoid}.table-scroll{max-height:none}body{print-color-adjust:exact}}
"""


JS = r"""
function visiblePlots(){
  document.querySelectorAll('.plot[data-spec]').forEach(el=>{
    if(el.closest('details:not([open]), [hidden]'))return;
    if(!el.getClientRects().length)return;
    if(el.dataset.loaded){if(window.Plotly)Plotly.Plots.resize(el);return;}
    const spec=JSON.parse(document.getElementById(el.dataset.spec).textContent);
    el.dataset.loaded='true';
    Plotly.newPlot(el,spec.data,spec.layout,{responsive:true,displaylogo:false,
      toImageButtonOptions:{format:'svg',filename:'analysis_figure'}});
  });
}
document.querySelectorAll('.fold-selector').forEach(select=>select.addEventListener('change',()=>{
  const section=select.closest('.study');
  section.querySelectorAll('.variant').forEach(v=>v.hidden=v.dataset.fold!==select.value);
  requestAnimationFrame(visiblePlots);
}));
document.querySelectorAll('details').forEach(el=>el.addEventListener('toggle',()=>requestAnimationFrame(visiblePlots)));
document.getElementById('study-search').addEventListener('input',e=>{
  const query=e.target.value.toLowerCase();
  document.querySelectorAll('.study').forEach(s=>s.hidden=!s.dataset.search.includes(query));
  document.querySelectorAll('nav a').forEach(a=>a.hidden=!a.textContent.toLowerCase().includes(query));
  requestAnimationFrame(visiblePlots);
});
document.querySelectorAll('[data-expand]').forEach(button=>button.addEventListener('click',()=>{
  document.querySelectorAll('.study:not([hidden])').forEach(s=>s.open=button.dataset.expand==='true');
}));
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',event=>{
  // srcdoc inherits the embedding document's base URI: do not navigate its href.
  event.preventDefault();
  const section=document.querySelector(a.getAttribute('href'));section.hidden=false;section.open=true;
  section.scrollIntoView({block:'start'});
  requestAnimationFrame(visiblePlots);
}));
document.querySelectorAll('th button[data-sort]').forEach(button=>button.addEventListener('click',()=>{
  const table=button.closest('table'),body=table.querySelector('tbody'),column=Number(button.dataset.sort);
  const direction=button.dataset.direction==='desc'?'asc':'desc';button.dataset.direction=direction;
  const rows=Array.from(body.rows),absolute=button.dataset.absolute==='true';
  rows.sort((a,b)=>{
    const ac=a.cells[column],bc=b.cells[column];
    if(ac.dataset.value==='')return bc.dataset.value===''?0:1;if(bc.dataset.value==='')return -1;
    let diff;
    if(ac.dataset.numeric==='true'&&bc.dataset.numeric==='true'){
      const number=value=>/^[+-]?\d+$/.test(value)?BigInt(value):
        value==='inf'?Infinity:value==='-inf'?-Infinity:
        value==='True'?1:value==='False'?0:Number(value);
      let av=number(ac.dataset.value),bv=number(bc.dataset.value);
      if(absolute){av=av<0?-av:av;bv=bv<0?-bv:bv;}
      // Relational comparisons preserve BigInt precision, including mixed floats.
      diff=av<bv?-1:av>bv?1:0;
    }else diff=ac.dataset.value.localeCompare(bc.dataset.value);
    return direction==='desc'?-diff:diff;
  });rows.forEach(row=>body.appendChild(row));
}));
requestAnimationFrame(visiblePlots);
"""


def render_report(report, *, path=None, renderers=None):
    """Render already calculated results without executing reporter code.

    Plotly's JS is embedded once when required; the document needs no server or
    CDN. Plot initialization is deferred until its section/fold becomes visible.
    Custom artifact renderers extend display kinds without changing orchestration.
    Downloads include all computed table rows even when HTML previews are capped.
    """
    renderers = {} if renderers is None else dict(renderers)
    groups = OrderedDict()
    for run in report.studies:
        groups.setdefault(run.name, []).append(run)
    sections, navigation = [], []
    has_plots, plot_number = False, 0

    def artifact_html(artifact):
        nonlocal has_plots, plot_number
        if artifact.kind in renderers:
            return str(renderers[artifact.kind](artifact))
        if artifact.kind == "text":
            return f'<pre>{_e(artifact.data)}</pre>'
        if artifact.kind == "html":
            return str(artifact.data)
        if artifact.kind == "table":
            return _table(artifact.data) + _csv_link(artifact.data, artifact.title or "table")
        if artifact.kind == "match_results":
            return render_match_results(artifact)
        if artifact.kind == "leaderboard":
            legend = ('<p class="legend">' + _e(artifact.options["legend"]) + '</p>'
                      if "legend" in artifact.options else
                      '<p class="legend"><span class="pos">Positive</span> / <span class="neg">negative</span> coefficients · bar length = magnitude · click headers to sort</p>')
            return legend + _table(artifact.data,
                                  coefficient_columns=artifact.options.get("coefficient_columns", ()),
                                  percentile_columns=artifact.options.get("percentile_columns", ()),
                                  preview=len(artifact.data))
        if artifact.kind == "plotly":
            from plotly.io import to_json
            has_plots = True
            plot_number += 1
            spec_id = f"plot-spec-{plot_number}"
            spec = to_json(artifact.data, validate=True)
            # JSON in a script-data element must not be able to close the element.
            spec = spec.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            height = getattr(artifact.data.layout, 'height', None)
            style = f' style="height:{max(300, min(1600, int(height)))}px"' if height is not None else ''
            return f'<div class="plot" data-spec="{spec_id}"{style}></div><script type="application/json" id="{spec_id}">{spec}</script>'
        raise ValueError(f"No renderer for artifact kind {artifact.kind!r}; supply renderers={{kind: callable}}.")

    for i, (name, runs) in enumerate(groups.items()):
        section_id = f"study-{i}"
        navigation.append(f'<a href="#{section_id}">{_e(name)}</a>')
        options = ''.join(f'<option value="{j}">Fold {run.fold_id}</option>' for j, run in enumerate(runs))
        controls = (f'<div class="scope-controls"><label>Inspect fold <select class="fold-selector">{options}</select></label></div>'
                    if runs[0].type == "per_fold" else '')
        variants = []
        for j, run in enumerate(runs):
            badges = [run.type, f"partition: {run.partition}", f"layout: {run.layout}",
                      f"{len(run.row_positions):,} rows", f"{run.n_matches:,} matches"]
            if run.fold_id is not None:
                badges.append(f"fold {run.fold_id}")
            elif run.scope_label is None and run.partition != "all":
                badges.append("unique union across selected folds")
            if run.scope_label is not None:
                badges.append(run.scope_label)
            scope = '<div class="scope">' + ''.join(f'<span class="badge">{_e(b)}</span>' for b in badges) + '</div>'
            notes = '<ul class="notes">' + ''.join(f'<li>{_e(n)}</li>' for n in run.result.notes) + '</ul>'
            artifacts = []
            for k, artifact in enumerate(run.result.artifacts):
                artifacts.append(f'<details class="artifact" {"open" if k == 0 else ""}>'
                                 f'<summary>{_e(artifact.title or artifact.kind)}</summary>{artifact_html(artifact)}</details>')
            downloads = []
            scope_table = run.scope if run.scope is not None else pd.DataFrame({"row_position": run.row_positions})
            downloads.append(_csv_link(scope_table, f"{name}-scope-{j}"))
            for table_name, table in run.result.tables.items():
                downloads.append(f'<div class="download-row">{_e(table_name)} · {len(table):,} rows '
                                 + _csv_link(table, f"{name}-{table_name}-{j}") + '</div>')
            variants.append(f'<div class="variant" data-fold="{j}" {"hidden" if j else ""}>'
                            f'{scope}<h3>{_e(run.result.title)}</h3>{notes}{"".join(artifacts)}'
                            f'<details class="artifact"><summary>Download underlying data</summary>{"".join(downloads)}</details></div>')
        search = f"{name} {runs[0].result.title}".lower()
        sections.append(f'<details class="study" id="{section_id}" data-search="{_e(search)}" {"open" if i == 0 else ""}>'
                        f'<summary>{_e(name)}<small>{len(runs)} view(s)</small></summary>'
                        f'<div class="study-content">{controls}{"".join(variants)}</div></details>')
    plotly_script = ''
    if has_plots:
        from plotly.offline import get_plotlyjs
        plotly_script = '<script>' + get_plotlyjs() + '</script>'
    empty = '<div class="empty">No reporters selected. Add studies when you need them.</div>' if not groups else ''
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_e(report.title)}</title><style>{CSS}{MATCH_CSS}</style>{plotly_script}</head><body>
<aside><div class="eyebrow">xDiyo analytics</div><div class="brand">Study explorer</div>
<label for="study-search" class="muted">Find a study</label><input id="study-search" type="search" placeholder="Filter studies…">
<nav aria-label="Study navigation">{''.join(navigation)}</nav></aside>
<main><header><div class="eyebrow">Analysis workspace</div><h1>{_e(report.title)}</h1>
<p class="muted">{len(groups)} selected studies · {len(report.studies)} computed views · explicit row scopes</p>
<div class="toolbar"><button type="button" data-expand="true">Expand studies</button><button type="button" data-expand="false">Collapse studies</button></div></header>
{empty}{''.join(sections)}<footer>Computed once. Fold and entity selection changes the view without recalculating studies. Figure toolbar exports SVG; data panels download CSV.</footer></main>
<script>{JS}{MATCH_JS}</script></body></html>'''
    if path is not None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(html, encoding="utf-8")
    return html
