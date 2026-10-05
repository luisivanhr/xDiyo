"""Lazy, offline rating charts with league/team filters and badge legends."""

import json


def render_ratings(data):
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('&', '\\u0026')
    return ('<div class="rating-view"><div class="rating-controls"></div>'
            '<div class="rating-teams" aria-label="Toggle team lines"></div><div class="rating-charts"></div>'
            f'<script type="application/json">{payload}</script></div>')


CSS = """
.rating-controls,.rating-teams{display:flex;flex-wrap:wrap;gap:12px;margin:14px 0;align-items:center}
.rating-controls label{display:flex;gap:8px;align-items:center}.rating-teams button{display:flex;gap:6px;align-items:center;border-bottom:3px solid var(--team-color)}
.rating-teams button[aria-pressed=false]{opacity:.4}.rating-teams img{width:25px;height:25px;object-fit:contain}
.rating-teams[hidden]{display:none}
.rating-chart{height:430px;min-width:0}.rating-view{min-width:0}.rating-controls select{max-width:100%}
"""


JS = r"""
function initializeRatings(root){
 if(root.ratingReady){root.querySelectorAll('.rating-chart').forEach(p=>{if(p.data)Plotly.Plots.resize(p);});return;}
 root.ratingReady=true;
 const data=JSON.parse(root.querySelector('script').textContent),controls=root.querySelector('.rating-controls'),charts=root.querySelector('.rating-charts'),legend=root.querySelector('.rating-teams');
 // Solarized seeds extended for perceptual separation, then visually balanced
 // to avoid clustering around yellow-green in a full-league chart.
 const hidden=new Set(),colors=['#268bd2','#dc322f','#238b57','#b58900','#805ad5','#06dbbb','#b8ca16','#27b927','#b927a3','#06bcf9','#9f5904','#f9a406','#13ae9f','#1961e6','#f96706','#1d7787','#06db70','#b35572','#d22d77','#a44ee4'];
 const teamIds=Object.keys(data.teams).sort((a,b)=>data.teams[a].name.localeCompare(data.teams[b].name));
 const color=id=>{const i=teamIds.indexOf(id);return i<colors.length?colors[Math.max(i,0)]:`hsl(${(i*137.508)%360},78%,43%)`;};
 const shade=c=>c.startsWith('#')?`rgba(${parseInt(c.slice(1,3),16)},${parseInt(c.slice(3,5),16)},${parseInt(c.slice(5,7),16)},0.13)`:c.replace('hsl(','hsla(').replace(')',',0.13)');
 const makeSelect=(label,items)=>{const wrap=document.createElement('label'),s=document.createElement('select');wrap.append(document.createTextNode(label+' '),s);s.setAttribute('aria-label',label);for(const [v,t] of items){const o=document.createElement('option');o.value=v;o.textContent=t;s.append(o);}controls.append(wrap);return s;};
 const leagues=[...new Map(data.rows.map(r=>[r.competition,r.league])).entries()].sort((a,b)=>a[1].localeCompare(b[1]));
 const sources=[...new Map(data.rows.map(r=>[JSON.stringify([r.source,r.stream]),r.source+' · '+r.stream])).entries()];
 const league=makeSelect('League',leagues),source=makeSelect('Rating source',sources),team=makeSelect('Teams',[['','All teams']]);
 source.parentElement.hidden=sources.length<=1;
 const bandLabel=document.createElement('label'),bands=document.createElement('input');bands.type='checkbox';bands.checked=true;bandLabel.append(bands,document.createTextNode('Uncertainty bands'));controls.append(bandLabel);
 const all=document.createElement('button');all.textContent='Show all lines';all.type='button';all.onclick=()=>{hidden.clear();draw(true);};controls.append(all);
 function available(){return data.rows.filter(r=>r.competition===league.value&&JSON.stringify([r.source,r.stream])===source.value);}
 function updateTeams(){const old=team.value;team.replaceChildren();const ids=new Set(available().map(r=>r.team).filter(v=>v!==null));for(const id of ['',...teamIds.filter(id=>ids.has(id))]){const o=document.createElement('option');o.value=id;o.textContent=id?data.teams[id].name:'All teams';team.append(o);}if([...team.options].some(o=>o.value===old))team.value=old;draw();}
 let drawVersion=0;
 function draw(preserveLegend=false){
  const version=++drawVersion,renders=[];
  // Retain the chart area's height while replacing plots so the document
  // cannot collapse and clamp the reader's scroll position during a toggle.
  charts.style.minHeight=charts.getBoundingClientRect().height+'px';
  charts.querySelectorAll('.rating-chart').forEach(p=>{if(p.data)Plotly.purge(p);});charts.replaceChildren();if(preserveLegend!==true)legend.replaceChildren();
  legend.hidden=Boolean(team.value);
  const rows=available(),panels=[...new Set(rows.map(r=>r.panel))],folds=[...new Set(rows.map(r=>r.fold))];
  const present=new Set(rows.map(r=>r.team).filter(v=>v!==null));
  if(preserveLegend!==true)for(const id of teamIds.filter(id=>!team.value&&present.has(id))){const b=document.createElement('button');b.type='button';b.dataset.team=id;b.style.setProperty('--team-color',color(id));b.setAttribute('aria-pressed',String(!hidden.has(id)));b.title='Show or hide '+data.teams[id].name;const badge=data.teams[id].badge;if(badge){const img=document.createElement('img');img.src=badge;img.alt='';b.append(img);}b.append(document.createTextNode(data.teams[id].name));b.onclick=()=>{hidden.has(id)?hidden.delete(id):hidden.add(id);draw(true);};legend.append(b);}
  for(const b of legend.querySelectorAll('button'))b.setAttribute('aria-pressed',String(!hidden.has(b.dataset.team)));
  for(const panel of panels){
   const el=document.createElement('div');el.className='rating-chart';el.setAttribute('aria-label',panel);charts.append(el);
   const selected=rows.filter(r=>r.panel===panel&&(r.team===null||((!team.value||team.value===r.team)&&!hidden.has(r.team))));
   const groups=new Map();for(const row of selected){const key=JSON.stringify([row.fold,row.team]);if(!groups.has(key))groups.set(key,[]);groups.get(key).push(row);}
   const traces=[],origins=new Map();
   for(const points of groups.values()){
    points.sort((a,b)=>a.kickoff.localeCompare(b.kickoff));const r=points[0],c=r.team===null?colors[folds.indexOf(r.fold)%colors.length]:color(r.team);
    const label=(r.team===null?'Shared home advantage':data.teams[r.team].name)+(folds.length>1?' · fold '+r.fold:'');
    const x=points.map(p=>p.kickoff),y=points.map(p=>p.mean),valid=points.some(p=>p.lower!==null&&p.upper!==null);
    const first=points.find(p=>Number.isFinite(p.mean));
    if(r.team!==null&&first&&data.teams[r.team].badge&&(!origins.has(r.team)||first.kickoff<origins.get(r.team).kickoff))origins.set(r.team,first);
    if(bands.checked&&valid){traces.push({x,y:points.map(p=>p.lower),mode:'lines',line:{width:0},showlegend:false,hoverinfo:'skip',connectgaps:false});traces.push({x,y:points.map(p=>p.upper),mode:'lines',line:{width:0},fill:'tonexty',fillcolor:shade(c),showlegend:false,hoverinfo:'skip',connectgaps:false});}
    traces.push({x,y,customdata:points.map(p=>[p.cutoff,p.lower,p.upper]),name:label,mode:'lines+markers',marker:{size:4},line:{color:c,width:2,dash:['solid','dash','dot','dashdot'][folds.indexOf(r.fold)%4]},connectgaps:false,hovertemplate:'%{x}<br>%{y:.3f}<br>Interval: %{customdata[1]:.3f} – %{customdata[2]:.3f}<br>Cutoff: %{customdata[0]}<extra>%{fullData.name}</extra>'});
   }
   const shared=panel==='Home advantage';if(shared)el.style.height='480px';
   renders.push(Plotly.newPlot(el,traces,{title:{text:panel},paper_bgcolor:'#16212d',plot_bgcolor:'#16212d',font:{color:'#dce5f0'},margin:{t:45,l:origins.size?90:65,r:20,b:shared?140:100},xaxis:{type:'date',title:{text:'Kickoff time',standoff:12},tickformat:'%d %b<br>%Y',nticks:4,automargin:true,gridcolor:'#2a394a'},yaxis:{title:{text:panel},ticklabelstandoff:origins.size?16:0,automargin:true,gridcolor:'#2a394a'},showlegend:shared,legend:{orientation:'h',y:-.4,font:{size:11}},hovermode:'closest'},{responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'rating_history'}}).then(()=>{
    // Separate from the team selector: one badge per team's earliest visible
    // rating, projected onto the y axis. Paper sizing keeps badges at 22 px.
    // Read Plotly's resolved axis range so zooming preserves the value anchor.
    let placing=false,initialRange=true;
    function placeBadges(){
     if(placing||!el.isConnected||!el._fullLayout||!origins.size)return;
     const layout=el._fullLayout,axis=layout.yaxis,size=layout._size;
     if(size.w<=0||size.h<=0)return;
     const images=[...origins].map(([id,p])=>({source:data.teams[id].badge,name:data.teams[id].name,xref:'paper',yref:'y',x:0,y:p.mean,xanchor:'center',yanchor:'middle',sizex:22/size.w,sizey:Math.abs(axis.range[1]-axis.range[0])*22/size.h,sizing:'contain',layer:'above'}));
     const update={images};
     // Replace Plotly's percentage-based date padding with 14 px: half a
     // badge (11 px) plus a 3 px gap before the first marker.
     if(initialRange){
      initialRange=false;
      let start=Infinity,end=-Infinity;
      for(const row of selected){const time=Date.parse(row.kickoff);if(Number.isFinite(time)){start=Math.min(start,time);end=Math.max(end,time);}}
      if(end>start&&size.w>28){const pad=(end-start)*14/(size.w-28);update['xaxis.range']=[new Date(start-pad).toISOString(),new Date(end+pad).toISOString()];}
     }
     placing=true;Plotly.relayout(el,update).finally(()=>{placing=false;});
    }
    if(el.isConnected&&typeof el.on==='function'){el.on('plotly_relayout',placeBadges);placeBadges();}
   }));
  }
  Promise.all(renders).finally(()=>{if(version===drawVersion)charts.style.minHeight='';});
 }
 league.onchange=source.onchange=updateTeams;team.onchange=bands.onchange=draw;updateTeams();
}
"""
