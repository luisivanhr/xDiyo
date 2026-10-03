/* Presentation only. Accounting is decimal-string arithmetic; no bet calculations. */
(() => {'use strict';
const $=id=>document.getElementById(id), esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const D=window.TICKET_INDEX, idx=D?.policies||[], B=D?.badges||{};
let current=null, rows=[], filtered=[], payload=null, page=0, generation=0, activeScript=null, timer=null;
// Integer mantissa + decimal scale. All source values were validated in Python.
function dec(s){const [a,e='0']=String(s).toLowerCase().split('e');const [i,f='']=a.split('.');let n=BigInt(i+f),scale=f.length-Number(e);if(scale<0){n*=10n**BigInt(-scale);scale=0}return [n,scale]}
function textDecimal(n,scale){const sign=n<0n?'-':'';let s=(n<0n?-n:n).toString().padStart(scale+1,'0');return sign+(scale?s.slice(0,-scale)+'.'+s.slice(-scale):s)}
function sum(values){const parts=values.map(dec);let scale=0;for(const p of parts)scale=Math.max(scale,p[1]);return textDecimal(parts.reduce((a,[n,s])=>a+n*10n**BigInt(scale-s),0n),scale)}
function fmt(value,places=2){let [n,scale]=dec(value),sign=n<0n?'-':'';n=n<0n?-n:n;if(scale>places){const div=10n**BigInt(scale-places);n=(n+div/2n)/div}else n*=10n**BigInt(places-scale);return sign+textDecimal(n,places)}
function percent(value){const [n,s]=dec(value);return fmt(textDecimal(n*100n,s))+'%'}
const negative=v=>dec(v)[0]<0n;
const qa=window.TICKET_VIEWER_QA={loaded:false,policyCount:idx.length,index:idx,currentPolicy:null,getRows:()=>rows,getFiltered:()=>filtered,getPayload:()=>payload,decimalSum:sum};
function options(id,values,all){$(id).innerHTML=(all?`<option value="">${esc(all)}</option>`:'')+values.map(v=>`<option value="${esc(v)}">${esc(v)}</option>`).join('')}
function fail(message){clearTimeout(timer);qa.loaded=false;$('loading').hidden=false;$('loading').className='error';$('loading').textContent=message;$('csv').disabled=true;$('filteredCsv').disabled=true;$('tickets').innerHTML='';$('policyTotals').innerHTML='';$('filteredSummary').textContent='';}
function policies(){const list=idx.filter(p=>p.group===$('group').value);$('policy').innerHTML=list.map(p=>`<option value="${esc(p.id)}">${esc(p.label)}</option>`).join('');choose()}
function choose(){
  const request=++generation;clearTimeout(timer);activeScript?.remove();current=idx.find(p=>p.id===$('policy').value);
  rows=[];filtered=[];payload=null;qa.loaded=false;qa.currentPolicy=current.id;
  $('loading').hidden=false;$('loading').className='';$('loading').textContent='Loading '+current.label+'…';$('app').hidden=false;
  $('tickets').innerHTML='';$('policyTotals').innerHTML='';$('policyNote').textContent='';$('filteredSummary').textContent='';$('page').textContent='';
  for(const id of ['csv','filteredCsv','prev','next'])$(id).disabled=true;
  const script=document.createElement('script');activeScript=script;script.dataset.generation=String(request);script.src=current.file;
  script.onerror=()=>{if(request===generation)fail('Unable to open policy data. Keep the entire report folder together and try selecting the policy again.')};
  timer=setTimeout(()=>{if(request===generation){generation++;fail('Policy data did not load. Select a policy to retry.')}},15000);
  document.body.appendChild(script);
}
window.XDIYO_TICKET_CHUNK=(id,data)=>{
  // A removed script can still finish. Only the active request may change UI.
  if(document.currentScript!==activeScript||Number(document.currentScript.dataset.generation)!==generation||id!==current.id)return;
  clearTimeout(timer);payload=data;rows=data.rows;qa.loaded=true;qa.currentPolicy=id;page=0;
  options('league',[...new Set(rows.map(r=>r.league))].sort(),'All leagues');
  options('result',[...new Set(rows.map(r=>r.result))].sort(),'All results');
  for(const key of ['from','to','search'])$(key).value='';
  $('loading').hidden=true;$('csv').disabled=false;$('filteredCsv').disabled=false;
  $('policyNote').textContent=current.label+' · '+rows.length+' selected tickets · '+(current.note||'');
  const s=current.summary,metrics=[['Taken tickets',String(s.rows),0],['Stake · units',s.stake_units,2],['Payout · units',s.payout_units,2],['Net P&L · units',s.profit_units,2]];
  if(s.roi_percent!==undefined)metrics.push(['ROI · %',s.roi_percent,2]);
  $('policyTotals').innerHTML=metrics.map(([name,value,places])=>`<div class="metric"><small>${esc(name)}</small><div class="value ${name.startsWith('Net')?(negative(value)?'negative':'positive'):''}">${esc(fmt(value,places))}</div></div>`).join('');
  const ids=new Set(rows.flatMap(r=>r.legs.flatMap(l=>[l.home_id,l.away_id])).filter(x=>x!=null));
  $('badgeNote').textContent=`Local badges: ${[...ids].filter(id=>B[id]?.badge).length} of ${ids.size} teams. ◇ marks an unavailable badge.`;
  filter();activeScript.remove();activeScript=null;
};
function teamName(leg,side){return leg[side+'_name']||B[leg[side+'_id']]?.name||(leg[side+'_id']?'Team '+leg[side+'_id']:'Unknown team')}
function team(leg,side){const id=leg[side+'_id'],b=B[id]?.badge,away=side==='away',image=b?`<img src="${esc(b)}" alt="" loading="lazy">`:'';return `<span class="team ${away?'away':''}">${!away?image:''}<span>${esc(teamName(leg,side))}${!b?'<small title="No saved badge"> ◇</small>':''}</span>${away?image:''}</span>`}
function leg(l,n){const teams=(l.home_id||l.home_name||l.away_id||l.away_name)?`<div class="teams">${team(l,'home')}<span class="score">${esc(l.home_score??'—')} – ${esc(l.away_score??'—')}</span>${team(l,'away')}</div>`:`<p>${esc(l.label||l.event_id)}</p>`;
  return `<div class="match"><small>Leg ${n+1} · ${esc(l.kickoff_utc.replace('T',' ').replace('Z',' UTC'))} · Event ${esc(l.event_id)}</small>${teams}<div class="facts"><span>${esc(l.market)} · <b>${esc(l.selection)}</b>${l.odds!=null?` @ <b title="${esc(l.odds)}">${esc(fmt(l.odds))}</b>`:''}</span>${l.result!=null?`<span>Result: ${esc(l.result)}</span>`:''}${l.probability!=null?`<span title="${esc(l.probability)}">Probability: ${esc(percent(l.probability))}</span>`:''}</div></div>`}
function filter(){if(!qa.loaded)return;const league=$('league').value,result=$('result').value,from=$('from').value,to=$('to').value,q=$('search').value.toLowerCase().trim();
  filtered=rows.filter(r=>(!league||r.league===league)&&(!result||r.result===result)&&(!from||r.filter_date>=from)&&(!to||r.filter_date<=to)&&(!q||[r.ticket_id,r.round,...r.legs.flatMap(l=>[l.event_id,teamName(l,'home'),teamName(l,'away'),l.label,l.market,l.selection])].join(' ').toLowerCase().includes(q)));page=0;render()}
function render(){const size=+$('size').value,total=Math.ceil(filtered.length/size);page=Math.min(page,Math.max(0,total-1));$('page').textContent=filtered.length?`Page ${page+1} of ${total}`:'No matching tickets';$('prev').disabled=page===0;$('next').disabled=page+1>=total;
  $('filteredSummary').textContent=filtered.length+' tickets · filtered net '+fmt(sum(filtered.map(r=>r.profit_units)))+' units';
  $('tickets').innerHTML=filtered.slice(page*size,(page+1)*size).map(r=>`<article class="ticket"><div class="ticket-head"><b>${esc(r.league)}${r.round?' · Round '+esc(r.round):''}</b><span class="outcome ${r.result==='win'?'positive':r.result==='loss'?'negative':''}">${esc(r.result.toUpperCase()||'UNKNOWN')}</span></div><small>${r.season?'Season '+esc(r.season)+' · ':''}${r.stage?'Stage '+esc(r.stage)+' · ':''}${r.legs.length} leg${r.legs.length===1?'':'s'}</small>${r.legs.map(leg).join('')}<div class="accounting">${r.combined_odds?`<span>Combined odds<b>${esc(fmt(r.combined_odds,4))}</b></span>`:''}<span>Stake<b>${esc(fmt(r.stake_units))} u</b></span><span>Payout<b>${esc(fmt(r.payout_units))} u</b></span><span>Net P&L<b class="${negative(r.profit_units)?'negative':'positive'}">${esc(fmt(r.profit_units))} u</b></span></div><details><summary>Ticket ID, timing and model details</summary><p class="id">${esc(r.ticket_id)}</p>${r.settlement_time?'<p>'+esc(r.settlement_time)+'</p>':''}<p>${esc(r.timing_note||'')}</p><p>${esc(r.details||'')}</p></details></article>`).join('')||'<p class="empty">No tickets match these filters</p>';
}
function download(data,suffix){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([data],{type:'text/csv;charset=utf-8'}));a.download='policy-'+idx.indexOf(current)+suffix+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000)}
const quote=s=>/[",\r\n]/.test(s)?'"'+s.replaceAll('"','""')+'"':s;
$('csv').onclick=()=>download(Uint8Array.from(atob(payload.csv_base64),c=>c.charCodeAt(0)),'');
$('filteredCsv').onclick=()=>download([payload.headers,...filtered.map(r=>r.source)].map(r=>r.map(quote).join(',')).join('\r\n')+'\r\n','-filtered');
$('group').onchange=policies;$('policy').onchange=choose;for(const id of ['league','result','from','to'])$(id).onchange=filter;$('search').oninput=filter;
$('size').onchange=()=>{page=0;render()};$('prev').onclick=()=>{page--;render()};$('next').onclick=()=>{page++;render()};$('reset').onclick=()=>{for(const id of ['league','result','from','to','search'])$(id).value='';filter()};
if(!idx.length){fail('No report index found. Keep index.js alongside INDEX.html.');return}
options('group',[...new Set(idx.map(p=>p.group))]);policies();
})();
