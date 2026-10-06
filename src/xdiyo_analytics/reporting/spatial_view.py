"""Two reusable plot panels and a bounded client cache for spatial artifacts."""
import json
from html import escape
from uuid import uuid4

CSS = """
.spatial-controls{display:flex;gap:12px;flex-wrap:wrap;margin:15px 0}.spatial-controls label{display:flex;flex-direction:column;gap:5px;min-width:180px;max-width:100%}.spatial-controls select{max-width:100%;width:360px}.spatial-controls input{width:220px;max-width:100%}.spatial-panels{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.spatial-panel{min-width:0;border:1px solid var(--border);border-radius:8px;padding:10px}.spatial-team{display:flex;align-items:center;gap:10px;font-weight:600;min-height:45px}.spatial-team img{height:38px;width:38px;object-fit:contain}.spatial-chart{height:390px;min-width:0}.spatial-detail{font-size:12px;color:var(--muted)}.spatial-status{font-size:13px;color:var(--muted)}@media(max-width:1150px){.spatial-panels{grid-template-columns:1fr}}@media(max-width:600px){.spatial-controls select{width:100%}}
"""


def render_spatial(data, *, transport=None, limit=None):
    identifier = 'spatial-' + uuid4().hex
    if transport is None:
        payload = data.offline(limit)
    else:
        payload = {'manifest':data.manifest(limit), 'remote':{k:v for k,v in transport.items() if k != 'register'}}
        payload['remote']['id'] = transport['register'](data)
    spec = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    return f'''<div class="spatial-view" id="{identifier}"><div class="spatial-controls">
<label>Find fixture<input class="spatial-search" type="search" placeholder="Team, league, season or round"></label>
<label>Fixture<select class="spatial-fixture"></select></label>
<label>Spatial feature<select class="spatial-feature"></select></label></div>
<p class="spatial-status" role="status"></p><div class="spatial-panels">
{''.join('<div class="spatial-panel"><div class="spatial-team"></div><p class="spatial-detail"></p><div class="spatial-chart"></div></div>' for _ in range(2))}
</div><button type="button" class="spatial-download">Download selected values</button>
<script type="application/json" class="spatial-spec">{spec}</script></div>'''

JS = r"""
function initializeSpatial(root){
 if(root.dataset.ready)return;root.dataset.ready='true';
 const cfg=JSON.parse(root.querySelector('.spatial-spec').textContent),m=cfg.manifest;
 const fixture=root.querySelector('.spatial-fixture'),feature=root.querySelector('.spatial-feature'),status=root.querySelector('.spatial-status');
 const cache=new Map();let revision=0,current=null;
 const option=(id,label)=>{const e=document.createElement('option');e.value=id;e.textContent=label;return e;};
 const populate=query=>{const old=fixture.value;fixture.replaceChildren(...m.fixtures.filter(f=>f.label.toLowerCase().includes(query.toLowerCase())).map(f=>option(f.id,f.label)));if([...fixture.options].some(o=>o.value===old))fixture.value=old;};
 populate('');if(m.fixtures.length)fixture.value=m.fixtures[m.fixtures.length-1].id;feature.replaceChildren(...m.maps.map(f=>option(f.id,f.label)));
 root.querySelector('.spatial-search').oninput=e=>{populate(e.target.value);draw();};
 const pitch=()=>{const line={color:'#b9ccd0',width:1};return [{type:'rect',x0:0,x1:105,y0:0,y1:68,line},{type:'line',x0:52.5,x1:52.5,y0:0,y1:68,line},{type:'circle',x0:43.35,x1:61.65,y0:24.85,y1:43.15,line},...[[0,16.5],[88.5,105]].map(([x0,x1])=>({type:'rect',x0,x1,y0:13.84,y1:54.16,line}))];};
 async function retrieve(f,k){
  const key=JSON.stringify([f,k]);if(cache.has(key)){const v=cache.get(key);cache.delete(key);cache.set(key,v);return v;}
  let value;
  if(cfg.remote){const response=await fetch(cfg.remote.url,{method:'POST',headers:{'Content-Type':'application/json','X-Builder-Token':cfg.remote.token},body:JSON.stringify({id:cfg.remote.id,fixture:f,map:k})});if(!response.ok)throw new Error((await response.json()).error||'Unable to load fixture');value=await response.json();}
  else value=cfg.pairs[f][k];
  cache.set(key,value);while(cache.size>m.cache_size)cache.delete(cache.keys().next().value);root.dataset.cached=String(cache.size);return value;
 }
 async function draw(){
  const ticket=++revision;
  if(!fixture.value||!feature.value){status.textContent='No fixtures match this selection.';root.querySelector('.spatial-panels').hidden=true;current=null;return;}
  status.textContent='Loading prepared feature values…';
  try{
   const pair=await retrieve(fixture.value,feature.value);if(ticket!==revision)return;current=pair;root.querySelector('.spatial-panels').hidden=false;
   const minimum=pair.scale[0],maximum=pair.scale[1],extent=Math.max(Math.abs(minimum),Math.abs(maximum))||1;
   for(let i=0;i<2;i++){
    if(ticket!==revision)return;
    const panel=pair.panels[i],host=root.querySelectorAll('.spatial-panel')[i],team=m.teams[panel.team_id],spec=panel.spec||{};
    const heading=host.querySelector('.spatial-team');heading.replaceChildren();
    if(team.badge){const img=document.createElement('img');img.src=team.badge;img.alt='';heading.append(img);}
    heading.append(document.createTextNode(team.name+' · '+(panel.side==='home'?'Home':'Away')));
    host.querySelector('.spatial-detail').textContent=spec.derived?'Derived spatial expression · '+(spec.calculation_label||''):(spec.side==='against'?'Opponents historically faced by this team':'This team’s historical activity')+' · '+(spec.orientation==='home'?'shared home-oriented pitch':'team-relative pitch');
    if(spec.units)host.querySelector('.spatial-detail').textContent+=' · '+spec.units;
    const annotations=[],shapes=pitch(),traces=[];
    if(panel.missing)annotations.push({text:'Feature unavailable',x:.5,y:.5,xref:'paper',yref:'paper',showarrow:false});
    else if(spec.kind==='grid'){
     const n=spec.grid_size,z=Array.from({length:n},(_,y)=>panel.values.slice(y*n,(y+1)*n));
     traces.push({type:'heatmap',z,x:Array.from({length:n},(_,x)=>(x+.5)*105/n),y:Array.from({length:n},(_,y)=>(y+.5)*68/n),zmin:minimum<0?-extent:0,zmax:minimum<0?extent:maximum||1,colorscale:minimum<0?'RdBu':'Viridis',reversescale:minimum<0,colorbar:{thickness:10,title:{text:'Value'}},hovertemplate:'x %{x:.1f}<br>y %{y:.1f}<br>%{z:.5g}<extra></extra>'});
    }else{
     if(spec.kind==='region'){
      const rectangles=spec.rectangles||(spec.region==='own_half'?[[0,50,0,100]]:[[50,100,0,100]]);
      for(const rectangle of rectangles){
       let [x0,x1,y0,y1]=rectangle;
       if(spec.orientation==='home'&&panel.side==='away'){[x0,x1,y0,y1]=[100-x1,100-x0,100-y1,100-y0];}
       shapes.unshift({type:'rect',x0:x0*1.05,x1:x1*1.05,y0:y0*.68,y1:y1*.68,fillcolor:'rgba(88,199,178,.3)',line:{width:0}});
      }
     }
     annotations.push({text:(spec.field_label||spec.field||spec.region||'Value').replaceAll('_',' ')+': '+(panel.values[0]===null?'Unavailable':Number(panel.values[0]).toPrecision(5)),x:.5,y:1.05,xref:'paper',yref:'paper',showarrow:false});
    }
    await Plotly.react(host.querySelector('.spatial-chart'),traces,{template:'plotly_dark',paper_bgcolor:'#17202b',plot_bgcolor:'#17202b',font:{color:'#e8eef7'},height:390,margin:{l:28,r:35,t:35,b:32},shapes,annotations,xaxis:{range:[0,105],showgrid:false,zeroline:false,constrain:'domain'},yaxis:{range:[0,68],showgrid:false,zeroline:false,scaleanchor:'x',scaleratio:1,constrain:'domain'}},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'fixture_spatial_feature'}});
   }
   if(ticket===revision)status.textContent='Exact prepared values · '+(cfg.remote?'loaded on demand':'offline data')+' · '+m.included_fixtures+' of '+m.total_fixtures+' fixtures included · shared scale across this feature’s report population';
  }catch(error){if(ticket===revision){current=null;root.querySelector('.spatial-panels').hidden=true;status.textContent=error.message;}}
 }
 fixture.onchange=draw;feature.onchange=draw;
 root.spatialDispose=()=>{revision++;cache.clear();current=null;root.querySelectorAll('.spatial-chart').forEach(el=>Plotly.purge(el));fixture.onchange=null;feature.onchange=null;root.querySelector('.spatial-search').oninput=null;root.querySelector('.spatial-download').onclick=null;delete root.dataset.ready;root.dataset.cached='0';root.spatialDispose=null;};
 root.querySelector('.spatial-download').onclick=()=>{if(!current)return;const url=URL.createObjectURL(new Blob([JSON.stringify({fixture:fixture.value,feature:feature.value,...current},null,2)],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='fixture-spatial-values.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),0);};
 draw();
}
"""
