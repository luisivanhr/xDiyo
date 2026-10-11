"""Square, navigable embedding plots with trace-owned, fixed-size badges."""
from colorsys import hsv_to_rgb
from html import escape

import numpy as np

from .studies import _plotting, _style


def embedding_figure(table, badges, title):
    go = _plotting()
    figure = go.Figure()
    colors = ['#268bd2', '#dc322f', '#238b57', '#b58900', '#805ad5', '#2aa198']
    xy = table[['x', 'y']].to_numpy(dtype=float)
    span = max(float(np.ptp(xy, axis=0).max()), 1e-6) * 1.2
    center = (xy.max(axis=0) + xy.min(axis=0)) / 2
    rows = table.reset_index(drop=True)
    for n, (league, group) in enumerate(rows.groupby('league', dropna=False, sort=False)):
        color = colors[n] if n < len(colors) else '#%02x%02x%02x' % tuple(round(255*v) for v in hsv_to_rgb((n*.61803398875)%1, .85, .8))
        has_badge = [bool(badges[i]) for i in group.index]
        figure.add_trace(go.Scatter(x=group.x, y=group.y, mode='markers',
            text=[escape(str(x)) for x in group.team_name], textposition='top center',
            name=f'League {league}', marker=dict(size=[30 if b else 12 for b in has_badge],
                color=['#eee8d5' if b else color for b in has_badge], line=dict(color=color, width=2)),
            meta=dict(embedding_color=color), customdata=group[['support', 'gate']].to_numpy(),
            hovertemplate='%{text}<br>Support %{customdata[0]:.2f}<br>Gate %{customdata[1]:.3f}<extra></extra>'))
        for point, (row, badge) in enumerate(zip(group.itertuples(), [badges[i] for i in group.index])):
            if badge:
                figure.add_layout_image(dict(source=badge, name=f'badge:{n}:{point}', x=row.x, y=row.y,
                    xref='x', yref='y', sizex=span*24/600, sizey=span*24/600,
                    sizing='contain', xanchor='center', yanchor='middle', layer='above'))
    _style(figure, title, 'Component 1', 'Component 2')
    figure.update_layout(height=720, width=720, showlegend=False, dragmode='pan',
        margin=dict(l=52, r=52, t=68, b=68), meta=dict(xdiyo_embedding=True),
        xaxis=dict(range=[center[0]-span/2, center[0]+span/2], constrain='domain'),
        yaxis=dict(range=[center[1]-span/2, center[1]+span/2], scaleanchor='x', scaleratio=1, constrain='domain'))
    return figure


CSS = r"""
.embedding-controls{max-width:860px;margin:12px auto;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.embedding-controls label{display:flex;align-items:center;gap:5px}.embedding-controls select{max-width:100%;min-width:0}
.embedding-leagues{max-width:860px;margin:12px auto;display:flex;gap:8px;flex-wrap:wrap}
.embedding-leagues button[aria-pressed=false]{opacity:.45;text-decoration:line-through}
.plot.embedding-plot{width:100%;max-width:860px;height:auto;aspect-ratio:1;margin:auto}
.embedding-help{max-width:860px;margin:8px auto;font-size:12px;color:var(--muted)}
.embedding-plot .imagelayer image{pointer-events:none}
"""


JS = r"""
async function initializeEmbedding(el,spec){
  el.classList.add('embedding-plot');
  el.style.height='auto';
  const controls=document.createElement('div');controls.className='embedding-controls';
  const leagues=document.createElement('div');leagues.className='embedding-leagues';
  const help=document.createElement('p');help.className='embedding-help';
  help.textContent='Drag to pan; use the chart toolbar to zoom. Select a team to locate it. League buttons hide points, names and badges together.';
  el.before(controls,leagues,help);
  function button(text,fn,parent=controls){const b=document.createElement('button');b.type='button';b.textContent=text;b.onclick=fn;parent.append(b);return b;}
  function check(text,checked){const l=document.createElement('label'),i=document.createElement('input');i.type='checkbox';i.checked=checked;l.append(i,document.createTextNode(text));controls.append(l);return i;}
  const names=check('Team names',false), badges=check('Team badges',true);
  const sizeLabel=document.createElement('label'),badgeSize=document.createElement('input');
  badgeSize.type='range';badgeSize.min='12';badgeSize.max='36';badgeSize.value='24';badgeSize.style.width='90px';
  sizeLabel.append(document.createTextNode('Badge size'),badgeSize);controls.append(sizeLabel);
  const find=document.createElement('select');find.setAttribute('aria-label','Find team');find.add(new Option('Find a team…',''));controls.append(find);
  const sourceImages=spec.layout.images||[], failed=new Set(), visible=spec.data.map(()=>true);
  const originalX=[...spec.layout.xaxis.range],originalY=[...spec.layout.yaxis.range];
  const teamOptions=[];
  spec.data.forEach((trace,t)=>trace.text.forEach((name,p)=>{const decoder=document.createElement('span');decoder.innerHTML=name;teamOptions.push({name:decoder.textContent,t,p});}));
  teamOptions.sort((a,b)=>a.name.localeCompare(b.name)).forEach(o=>find.add(new Option(o.name,`${o.t}:${o.p}`)));
  const toggles=spec.data.map((trace,t)=>{const b=button(trace.name,async()=>{visible[t]=!visible[t];await redraw();},leagues);b.style.borderBottom=`3px solid ${trace.meta.embedding_color}`;b.setAttribute('aria-pressed','true');return b;});
  const width=Math.round(el.getBoundingClientRect().width);
  await Plotly.newPlot(el,spec.data,{...spec.layout,width,height:width},{responsive:false,displaylogo:false,
    toImageButtonOptions:{format:'svg',filename:'team_embeddings'}});
  let updating=false, pending=false;
  async function placeBadges(){
    if(updating){pending=true;return;}if(!el._fullLayout||!el.getClientRects().length)return;
    const f=el._fullLayout,s=f._size;if(s.w<=0||s.h<=0)return;
    const images=sourceImages.map(im=>{const t=Number(im.name.split(':')[1]);return {...im,
      visible:visible[t]&&badges.checked&&!failed.has(im.name),
      sizex:Math.abs(f.xaxis.range[1]-f.xaxis.range[0])*Number(badgeSize.value)/f.xaxis._length,
      sizey:Math.abs(f.yaxis.range[1]-f.yaxis.range[0])*Number(badgeSize.value)/f.yaxis._length};});
    updating=true;
    try{await Plotly.relayout(el,{images});}finally{updating=false;if(pending){pending=false;requestAnimationFrame(placeBadges);}}
  }
  async function redraw(){
    for(let t=0;t<el.data.length;t++){
      toggles[t].setAttribute('aria-pressed',String(visible[t]));
      const owners=new Set(sourceImages.filter(im=>im.name.startsWith(`badge:${t}:`)&&!failed.has(im.name)).map(im=>Number(im.name.split(':')[2])));
      const sizes=el.data[t].text.map((_,p)=>badges.checked&&owners.has(p)?Number(badgeSize.value)+6:12);
      const colors=el.data[t].text.map((_,p)=>badges.checked&&owners.has(p)?'#eee8d5':el.data[t].meta.embedding_color);
      await Plotly.restyle(el,{visible:visible[t],mode:names.checked?'markers+text':'markers','marker.size':[sizes],'marker.color':[colors]},[t]);
    }
    await placeBadges();
  }
  names.onchange=redraw;badges.onchange=redraw;badgeSize.onchange=redraw;
  button('Show all leagues',async()=>{visible.fill(true);await redraw();});
  button('Hide all leagues',async()=>{visible.fill(false);await redraw();});
  button('Reset view',()=>{find.value='';Plotly.relayout(el,{'xaxis.range':originalX,'yaxis.range':originalY});});
  find.onchange=async()=>{if(!find.value)return;const [t,p]=find.value.split(':').map(Number);visible[t]=true;await redraw();
    const trace=el.calcdata[t][p],span=(originalX[1]-originalX[0])*.18;
    await Plotly.relayout(el,{'xaxis.range':[trace.x-span,trace.x+span],'yaxis.range':[trace.y-span,trace.y+span]});
    Plotly.Fx.hover(el,[{curveNumber:t,pointNumber:p}]);};
  el.on('plotly_relayout',event=>{if(!('images' in event))placeBadges();});
  let lastWidth=width;
  el.embeddingResize=()=>{const w=Math.round(el.getBoundingClientRect().width);if(w>0&&w!==lastWidth){lastWidth=w;Plotly.relayout(el,{width:w,height:w});}};
  new ResizeObserver(el.embeddingResize).observe(el);
  await placeBadges();
  sourceImages.forEach(im=>{const img=new Image();img.onerror=()=>{failed.add(im.name);redraw();};img.src=im.source;});
}
"""
