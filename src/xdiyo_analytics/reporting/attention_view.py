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
        margin=dict(t=110, b=150), legend=dict(orientation='h', x=0, y=1.05, yanchor='bottom'),
        xaxis=dict(tickangle=-40, automargin=True, categoryorder='array', categoryarray=order),
        meta=dict(xdiyo_attention=True, roster=roster))
    return figure


CSS = r"""
.attention-controls{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin:12px 0}
.attention-controls label{display:flex;gap:6px;align-items:center;min-width:0;max-width:100%}
.attention-controls select{min-width:0;max-width:100%;width:250px}
.attention-help{color:var(--muted);font-size:12px}
.attention-chart-scroll{max-width:100%;overflow-x:auto}.attention-chart-scroll>.plot{min-width:720px}
"""

JS = r"""
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
    const selected=roster.find(r=>r.team===team.value), traces=spec.data.filter(t=>t.meta.team===team.value).map(t=>({...t,visible:true}));
    const labels=new Set(traces.flatMap(t=>t.x)), order=spec.layout.xaxis.categoryarray.filter(label=>labels.has(label));
    const images=selected?.badge?[{source:selected.badge,x:1,y:1.12,xref:'paper',yref:'paper',sizex:.06,sizey:.12,xanchor:'right',yanchor:'top'}]:[];
    const annotations=traces.length?[]:[{text:'No eligible historical edges for this team at this issuance.',xref:'paper',yref:'paper',x:.5,y:.5,showarrow:false}];
    // Plot only the selected team's traces. Hidden teams must not leave empty
    // categorical slots or consume grouped-bar widths in the current chart.
    await Plotly.react(el,traces,{...spec.layout,images,annotations,
      xaxis:{...spec.layout.xaxis,categoryarray:order,autorange:true},yaxis:{...spec.layout.yaxis,autorange:true}},
      {responsive:true,displaylogo:false,toImageButtonOptions:{format:'svg',filename:'attention_weights'}});
  }
  // Serialize rapid changes so an earlier update cannot overwrite the selection.
  let updating=Promise.resolve();function update(){updating=updating.then(redraw);return updating;}
  league.onchange=()=>{choices();return update();};team.onchange=update;
  await update();
}
"""
