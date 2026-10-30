"""Native, scrollable attention selectors; display ordering only."""
from html import escape

from .studies import _plotting, _style
from .teams import team_key


def attention_roster(model, queries, metadata, catalog, show_badges, notes):
    """Evaluation teams, including cold teams, using membership at issuance."""
    members = model.event_store.team_seasons
    members = members[members.entry_at.le(queries.issuance_at.iloc[0])]
    leagues = dict(zip(members.team_id, members.competition_id))
    league_names = {}
    if {'competition_id', 'source_league'} <= set(metadata):
        league_names = dict(metadata[['competition_id', 'source_league']].dropna().itertuples(index=False, name=None))
    roster = []
    for team in queries.team_id:
        name, badge, note = catalog.display(team, badges=show_badges)
        if note:
            notes.append(note)
        league = leagues.get(team)
        roster.append(dict(team=team_key(team), name=name, badge=badge,
                           league=team_key(league), league_name=str(league_names.get(league, f'League {league}' if league is not None else 'Unknown league'))))
    return sorted(roster, key=lambda row: (row['league_name'], row['name'], row['team']))


def attention_figure(table, roster, catalog, top_edges, fold):
    go = _plotting()
    figure = go.Figure()
    categories = []
    first = roster[0]['team'] if roster else None
    if not table.empty:
        for (team, layer, head), group in table.groupby(['team_id', 'layer', 'head'], sort=False):
            # Selection remains by strength; ordering is strictly chronological.
            top = group.nlargest(top_edges, 'attention').sort_values(['kickoff_at', 'event_id'], ascending=[False, True], kind='stable')
            labels = [escape(catalog.display(r.opponent_id)[0])+' · '+r.kickoff_at.strftime('%Y-%m-%d')+' · '+str(r.event_id) for r in top.itertuples()]
            categories.extend(zip(top.kickoff_at, labels))
            figure.add_trace(go.Bar(x=labels, y=top.attention, name=f'Layer {layer+1} / head {head+1}',
                                   meta=dict(team=team_key(team)), visible=team_key(team) == first))
    # A shared order is needed when different heads select different matches.
    order = list(dict.fromkeys(label for _, label in sorted(categories, key=lambda pair: pair[0], reverse=True)))
    _style(figure, f'Incoming attention · Fold {fold}', 'Historical matches (newest first)', 'Normalized weight')
    figure.update_layout(height=650, colorway=['#268bd2','#dc322f','#238b57','#b58900','#805ad5','#2aa198'],
        title=dict(y=.98, yanchor='top'),
        # Keep a separate right gutter for the selected team's badge. The
        # horizontal legend wraps within the plotting width, outside this gutter.
        margin=dict(t=110, b=150, r=90),
        legend=dict(orientation='h', x=0, xanchor='left', y=.92, yref='container', yanchor='top'),
        xaxis=dict(tickangle=-40, automargin=True, categoryorder='array', categoryarray=order),
        meta=dict(xdiyo_attention=True, roster=roster))
    return figure


CSS = r"""
.attention-controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:12px 0}
.attention-controls label{display:flex;gap:6px;align-items:center;min-width:0;max-width:100%}
.attention-controls select{min-width:0;max-width:100%;width:250px}
.attention-controls [hidden]{display:none!important}
.attention-help{color:var(--muted);font-size:12px}
.attention-chart-scroll{max-width:100%;overflow-x:auto}.attention-chart-scroll>.plot{min-width:0;width:100%}
"""

JS = r"""
async function sizeAttentionBadge(el){
  const size=el._fullLayout?._size, badge=el.layout?.images?.[0];
  if(!size||!badge||el.attentionBadgeUpdating)return;
  const sx=32/size.w, sy=32/size.h, x=1+8/size.w;
  if(Math.abs(badge.sizex-sx)<1e-9&&Math.abs(badge.sizey-sy)<1e-9&&Math.abs(badge.x-x)<1e-9)return;
  el.attentionBadgeUpdating=true;
  try{await Plotly.relayout(el,{'images[0].sizex':sx,'images[0].sizey':sy,'images[0].x':x});}
  finally{el.attentionBadgeUpdating=false;}
}

async function drawAttention(el,spec,team){
  const selected=spec.layout.meta.roster.find(r=>r.team===team), traces=spec.data.filter(t=>t.meta.team===team).map(t=>({...t,visible:true}));
  const labels=new Set(traces.flatMap(t=>t.x)), order=spec.layout.xaxis.categoryarray.filter(label=>labels.has(label));
  const images=selected?.badge?[{source:selected.badge,x:1.03,y:1.12,xref:'paper',yref:'paper',sizex:.06,sizey:.12,sizing:'contain',xanchor:'left',yanchor:'top'}]:[];
  const annotations=traces.length?[]:[{text:'No eligible historical edges for this team at this issuance.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
  // Hidden teams must not leave empty categories or consume grouped-bar widths.
  await Plotly.react(el,traces,{...spec.layout,images,annotations,
    xaxis:{...spec.layout.xaxis,categoryarray:order,autorange:true},yaxis:{...spec.layout.yaxis,autorange:true}},
    {responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'attention_weights'}});
  if(!el.attentionBadgeListener){el.on('plotly_relayout',()=>sizeAttentionBadge(el));el.attentionBadgeListener=true;}
  await sizeAttentionBadge(el);
}

async function initializeFixtureAttention(el,spec){
  const {fixtures,views}=spec.layout.meta;
  el.style.height='auto';el.replaceChildren();
  if(!fixtures.length){el.textContent='No evaluation fixtures at the selected issuance.';return;}
  const controls=document.createElement('div');controls.className='attention-controls';el.append(controls);
  function selector(text){const label=document.createElement('label'),select=document.createElement('select');
    select.setAttribute('aria-label',text);label.append(document.createTextNode(text),select);controls.append(label);return select;}
  const home=selector('Home team'),away=selector('Away team'),date=selector('Kickoff date');
  const help=document.createElement('p');help.className='attention-help';el.append(help);
  const panels=['Home','Away'].map(side=>{const heading=document.createElement('h4'),scroll=document.createElement('div'),plot=document.createElement('div');
    scroll.className='attention-chart-scroll';plot.className='plot';plot.style.height='650px';
    plot.setAttribute('aria-label',side+' attention');scroll.append(plot);el.append(heading,scroll);return {heading,plot};});
  function teams(select,rows,side){const old=select.value;select.replaceChildren();
    [...new Map(rows.map(f=>[f[side],f[side+'_name']]))].sort((a,b)=>a[1].localeCompare(b[1]))
      .forEach(([id,name])=>select.add(new Option(name,id)));
    if([...select.options].some(o=>o.value===old))select.value=old;}
  function dates(){const old=date.value;date.replaceChildren();const rows=fixtures.filter(f=>f.home===home.value&&f.away===away.value);
    rows.forEach(f=>{const index=fixtures.indexOf(f), tie=rows.filter(r=>r.kickoff===f.kickoff).length>1;
      date.add(new Option(f.kickoff.replace('T',' ')+(tie?' · '+f.competition+' / '+f.season+' / '+f.event:''),String(index)));});
    if([...date.options].some(o=>o.value===old))date.value=old;
    date.parentElement.hidden=rows.length<2;
  }
  teams(home,fixtures,'home');teams(away,fixtures.filter(f=>f.home===home.value),'away');dates();
  async function redraw(){const fixture=fixtures[Number(date.value)];
    help.textContent='Kickoff: '+fixture.kickoff+' · Inference issuance: '+fixture.issuance;
    for(const [i,side] of ['home','away'].entries()){
      panels[i].heading.textContent=(i?'Away · ':'Home · ')+fixture[side+'_name'];
      const view=views[fixture[side+'_view']];
      await drawAttention(panels[i].plot,{data:view.data,layout:{...spec.layout,meta:view.layout.meta,
        xaxis:{...spec.layout.xaxis,...view.layout.xaxis}}},fixture[side]);
    }
  }
  let updating=Promise.resolve();function update(){updating=updating.then(redraw);return updating;}
  home.onchange=()=>{teams(away,fixtures.filter(f=>f.home===home.value),'away');dates();return update();};
  away.onchange=()=>{dates();return update();};date.onchange=update;
  el.attentionResize=()=>panels.forEach(p=>Plotly.Plots.resize(p.plot));
  await update();
}

async function initializeAttention(el,spec){
  const roster=spec.layout.meta.roster, controls=document.createElement('div');controls.className='attention-controls';
  function selector(text){const label=document.createElement('label'),select=document.createElement('select');
    select.setAttribute('aria-label',text);label.append(document.createTextNode(text),select);controls.append(label);return select;}
  const league=selector('League'),team=selector('Team'),help=document.createElement('p');help.className='attention-help';
  help.textContent='Evaluation teams at this fold’s selected issuance. Strongest edges are shown newest first; downloads retain all edges.';
  el.before(controls,help);
  const scroll=document.createElement('div');scroll.className='attention-chart-scroll';el.before(scroll);scroll.append(el);
  league.add(new Option('All leagues',''));
  const leagues=new Map(roster.map(r=>[r.league,r.league_name]));
  [...leagues].sort((a,b)=>a[1].localeCompare(b[1])).forEach(([id,name])=>league.add(new Option(name,id)));
  function choices(){const old=team.value;team.replaceChildren();roster.filter(r=>!league.value||r.league===league.value)
    .sort((a,b)=>a.name.localeCompare(b.name)).forEach(r=>team.add(new Option(r.name,r.team)));
    if([...team.options].some(o=>o.value===old))team.value=old;}
  choices();
  async function redraw(){
    await drawAttention(el,spec,team.value);
  }
  // Serialize rapid changes so an earlier update cannot overwrite the selection.
  let updating=Promise.resolve();function update(){updating=updating.then(redraw);return updating;}
  league.onchange=()=>{choices();return update();};team.onchange=update;
  await update();
}
"""
