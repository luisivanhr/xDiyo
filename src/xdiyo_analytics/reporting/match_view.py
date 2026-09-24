"""Offline interactive fixture tables used by the shared report viewer."""

from html import escape
import json

import numpy as np
import pandas as pd

from .teams import team_key


def render_match_results(artifact):
    """Embed already-computed rows and names; no filesystem/network/model access."""
    rows = []
    numeric = {"result", "prediction", "error"}
    for record in artifact.data.to_dict("records"):
        row = {}
        for key, value in record.items():
            if value is None or value is pd.NA or (np.isscalar(value) and pd.isna(value)):
                row[key] = None
            elif key in numeric:
                value = value.item() if isinstance(value, np.generic) else value
                row[key] = str(value) if isinstance(value, int) and abs(value) > 2**53-1 else value
            else:
                # Identities are strings in JavaScript: never round large integers.
                row[key] = team_key(value)
        rows.append(row)
    payload = json.dumps(dict(rows=rows, **artifact.options), ensure_ascii=False, allow_nan=False)
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    return ('<div class="match-results">'
            f'<script type="application/json" class="match-data">{payload}</script>'
            f'<div class="match-app" aria-label="{escape(artifact.title, quote=True)}"></div>'
            '<noscript>Interactive fixture filters require JavaScript. Full results remain available in the data download below.</noscript></div>')


MATCH_CSS = r"""
.match-filters{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0}.match-filters label{display:flex;flex-direction:column;gap:5px;color:var(--muted);font-size:12px}.match-filters select{min-width:125px;max-width:260px;color:var(--ink)}
.match-controls{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin:12px 0}.match-count{color:var(--muted);font-size:13px}.match-pager{display:flex;align-items:center;gap:12px;margin:12px 0}.match-pager button:disabled{opacity:.4;cursor:default}
.match-table th{padding:12px}.match-table td{min-width:85px}.match-table .team-cell{min-width:160px;white-space:normal;font-weight:600}.team-inline{display:flex;align-items:center;gap:9px}.team-inline img{width:28px;height:28px;object-fit:contain;flex-shrink:0}
.match-table .fixture-heading td{background:#243245;color:var(--muted);font-size:12px;padding:9px 12px;white-space:normal}.match-table td.good{background:#3bb99918}.match-table td.bad{background:#ed79751a}.match-table td.neutral{color:var(--muted)}
.match-table tr.result-good{box-shadow:inset 3px 0 #58c7b2}.match-table tr.result-bad{box-shadow:inset 3px 0 #ed7975}.match-table tr.result-neutral{box-shadow:inset 3px 0 #9eb0c5}.match-table .prob-cell{white-space:normal;min-width:150px;font-size:12px}.match-table .subhead th{font-size:12px;top:45px;background:#1d2a39}
"""


MATCH_JS = r"""
document.querySelectorAll('.match-results').forEach(root=>{
  const spec=JSON.parse(root.querySelector('.match-data').textContent),app=root.querySelector('.match-app');
  const filters={...spec.initial},selects={},fields=['league','season','team','round'];
  let page=0,sort='original',direction=1,visibleRows=[],visibleFixtures=[];
  const text=(tag,value,className)=>{const node=document.createElement(tag);node.textContent=value;if(className)node.className=className;return node;};
  const addButton=(label,fn)=>{const button=text('button',label);button.type='button';button.addEventListener('click',fn);return button;};
  const labelFor=(key,value)=>key==='team'?(spec.teams[value]?.name||`Team ${value}`):value;
  const matches=(row,except=null)=>fields.every(key=>key===except||filters[key]===null||
    (key==='team'?(row.home_id===filters[key]||row.away_id===filters[key]):row[key]===filters[key]));
  const filterBar=text('div','','match-filters');app.appendChild(filterBar);
  fields.forEach(key=>{const label=text('label',key[0].toUpperCase()+key.slice(1));
    const select=document.createElement('select');select.setAttribute('aria-label',key[0].toUpperCase()+key.slice(1));
    select.addEventListener('change',()=>{filters[key]=select.value===''?null:select.value;page=0;refresh();});
    label.appendChild(select);filterBar.appendChild(label);selects[key]=select;
  });
  const controls=text('div','','match-controls');app.appendChild(controls);
  controls.appendChild(addButton('Reset filters',()=>{fields.forEach(key=>filters[key]=null);page=0;refresh();}));
  const sortLabel=text('label','Sort '),sortSelect=document.createElement('select');sortSelect.setAttribute('aria-label','Sort fixtures');
  const sorts=spec.layout==='team_match'?['original','home','away','home_result','home_prediction','away_result','away_prediction']:['original','home','away','result','prediction'];
  if(spec.numeric)sorts.push(...(spec.layout==='team_match'?['home_error','away_error']:['error']));
  sorts.forEach(key=>{const option=text('option',key==='original'?'Original order':key.replaceAll('_',' '));option.value=key;sortSelect.appendChild(option);});
  sortSelect.addEventListener('change',()=>{sort=sortSelect.value;page=0;refresh();});sortLabel.appendChild(sortSelect);controls.appendChild(sortLabel);
  const reverse=addButton('Ascending',()=>{direction*=-1;reverse.textContent=direction===1?'Ascending':'Descending';page=0;refresh();});controls.appendChild(reverse);
  controls.appendChild(addButton('Download filtered CSV',()=>{
    const keys=Object.keys(spec.rows[0]||{}).filter(key=>key!=='fixture');
    const cell=value=>'"'+String(value??'').replaceAll('"','""')+'"';
    const csv='\ufeff'+[keys.map(cell).join(','),...visibleRows.map(row=>keys.map(key=>cell(row[key])).join(','))].join('\r\n');
    const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link=document.createElement('a');link.href=url;link.download='filtered-match-results.csv';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }));
  const count=text('p','','match-count');count.setAttribute('aria-live','polite');app.appendChild(count);
  const legend=text('p',spec.numeric?`Green: |prediction − result| ≤ ${spec.tolerance}. Red: outside tolerance. Neutral: unavailable, push or void.`:
    'Green: correct. Red: incorrect. Neutral: unavailable, no decision, push or void.','legend');app.appendChild(legend);
  const scroll=text('div','','table-scroll'),table=text('table','','match-table');scroll.appendChild(table);app.appendChild(scroll);
  const pager=text('div','','match-pager'),previous=addButton('Previous',()=>{page--;draw();}),next=addButton('Next',()=>{page++;draw();}),pageLabel=text('span','');
  pager.append(previous,pageLabel,next);app.appendChild(pager);
  const format=value=>value===null||value===undefined?'—':typeof value==='number'?(Number.isInteger(value)?String(value):value.toLocaleString(undefined,{maximumFractionDigits:4})):String(value);
  const mood=row=>row&&['correct','within tolerance'].includes(row.status)?'good':row&&['incorrect','outside tolerance'].includes(row.status)?'bad':'neutral';
  function teamCell(id){const cell=text('td','','team-cell'),inline=text('span','','team-inline'),entry=spec.teams[id]||{name:`Team ${id}`};
    if(entry.badge){const img=document.createElement('img');img.src=entry.badge;img.alt='';img.loading='lazy';img.addEventListener('error',()=>img.remove());inline.appendChild(img);}
    inline.appendChild(text('span',entry.name));cell.appendChild(inline);return cell;
  }
  function outcomeCells(tr,row){const state=mood(row);
    const observed=row&&['push','void','missing'].includes(row.settlement)?row.settlement:format(row?.result);
    tr.append(text('td',observed,state),text('td',format(row?.prediction),state));
    if(spec.numeric)tr.appendChild(text('td',format(row?.error),state));
    if(spec.probabilities){let display='—';if(row?.probabilities){const p=JSON.parse(row.probabilities);display=Object.entries(p).map(([k,v])=>`${k}: ${(v*100).toFixed(1)}%`).join(' · ');}tr.appendChild(text('td',display,`prob-cell ${state}`));}
  }
  function valueFor(fixture){if(sort==='home'||sort==='away')return fixture[0][sort];
    const parts=sort.split('_'),side=parts.length>1?parts[0]:null,key=parts.at(-1);
    return (side?fixture.find(row=>row.side===side):fixture[0])?.[key]??null;
  }
  function refresh(){
    fields.forEach(key=>{
      const values=new Set();spec.rows.filter(row=>matches(row,key)).forEach(row=>{
        if(key==='team'){values.add(row.home_id);values.add(row.away_id);}else if(row[key]!==null)values.add(row[key]);
      });
      const chosen=filters[key],select=selects[key];select.replaceChildren();const all=text('option','All');all.value='';select.appendChild(all);
      if(chosen!==null)values.add(chosen);
      [...values].sort((a,b)=>labelFor(key,a).localeCompare(labelFor(key,b),undefined,{numeric:true})).forEach(value=>{
        const option=text('option',labelFor(key,value));option.value=value;select.appendChild(option);
      });select.value=chosen??'';
    });
    visibleRows=spec.rows.filter(row=>matches(row));const fixtures=new Map();
    visibleRows.forEach(row=>{if(!fixtures.has(row.fixture))fixtures.set(row.fixture,[]);fixtures.get(row.fixture).push(row);});
    visibleFixtures=[...fixtures.values()];
    if(sort!=='original')visibleFixtures.sort((a,b)=>{const av=valueFor(a),bv=valueFor(b);if(av===null)return bv===null?0:1;if(bv===null)return -1;
      const diff=typeof av==='number'&&typeof bv==='number'?av-bv:String(av).localeCompare(String(bv),undefined,{numeric:true});return direction*diff;});
    else if(direction<0)visibleFixtures.reverse();
    draw();
  }
  function draw(){
    const pages=Math.max(1,Math.ceil(visibleFixtures.length/spec.page_size));page=Math.max(0,Math.min(page,pages-1));
    count.textContent=`${visibleFixtures.length.toLocaleString()} fixture occurrences · ${visibleRows.length.toLocaleString()} observations · ${spec.rows.length.toLocaleString()} observations in full scope`;
    table.replaceChildren();const head=document.createElement('thead'),top=document.createElement('tr'),body=document.createElement('tbody');
    const metrics=['Result','Prediction',...(spec.numeric?['Error']:[]),...(spec.probabilities?['Probabilities']:[])];
    if(spec.layout==='team_match'){
      ['Home','Away'].forEach(name=>{const cell=text('th',name);cell.rowSpan=2;top.appendChild(cell);});
      ['Home label','Away label'].forEach(name=>{const cell=text('th',name);cell.colSpan=metrics.length;top.appendChild(cell);});head.appendChild(top);
      const sub=text('tr','','subhead');[...metrics,...metrics].forEach(name=>sub.appendChild(text('th',name)));head.appendChild(sub);
    }else{['Home','Away',...metrics].forEach(name=>top.appendChild(text('th',name)));head.appendChild(top);}
    const columnCount=2+metrics.length*(spec.layout==='team_match'?2:1);
    let lastHeading=null;
    visibleFixtures.slice(page*spec.page_size,(page+1)*spec.page_size).forEach(fixture=>{
      const row=fixture[0],heading=[row.league,row.season,row.stage,row.round===null?null:`Round ${row.round}`,row.fold_id==='-1'?'Pooled predictions':`Fold ${row.fold_id}`].filter(value=>value!==null&&value!=='').join(' · ');
      if(heading!==lastHeading){const group=text('tr','','fixture-heading'),cell=text('td',heading);cell.colSpan=columnCount;group.appendChild(cell);body.appendChild(group);lastHeading=heading;}
      const states=fixture.map(mood),state=states.includes('bad')?'bad':states.every(s=>s==='good')&&(spec.layout!=='team_match'||fixture.length===2)?'good':'neutral';
      const tr=text('tr','',`result-${state}`);tr.append(teamCell(row.home_id),teamCell(row.away_id));
      if(spec.layout==='team_match'){outcomeCells(tr,fixture.find(r=>r.side==='home'));outcomeCells(tr,fixture.find(r=>r.side==='away'));}else outcomeCells(tr,row);
      body.appendChild(tr);
    });
    if(!visibleFixtures.length){const row=document.createElement('tr'),cell=text('td','No fixtures match these filters.');cell.colSpan=columnCount;row.appendChild(cell);body.appendChild(row);}
    table.append(head,body);previous.disabled=page===0;next.disabled=page>=pages-1;pageLabel.textContent=`Page ${page+1} of ${pages}`;
  }
  refresh();
});
"""
