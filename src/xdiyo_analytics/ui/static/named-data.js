import {el} from './forms.js';

export function namedDataPanel(recipe,api,changed,refreshChoices){
 const box=el('section',{class:'section named-datasets'},el('h2',{text:'Named datasets'}),el('p',{text:'Explicit versioned selections keep existing experiments unchanged. Preview eligibility before accepting a new population.'}));
 const root=el('input',{'aria-label':'Dataset catalogue folder',value:recipe.data_selection?.root||'',placeholder:'Local canonical dataset folder'});
 const mode=el('select',{'aria-label':'Dataset selection mode'});
 for(const name of ['single_dataset','union','extend_history'])mode.append(el('option',{value:name,text:name.replaceAll('_',' ')}));
 const profile=el('select',{'aria-label':'Dataset evidence profile'});
 for(const name of ['retrospective_reported_goals','strict_as_of'])profile.append(el('option',{value:name,text:name.replaceAll('_',' ')}));
 const train=el('input',{'aria-label':'Fixed training fixture IDs',placeholder:'Training IDs, comma separated'});
 const test=el('input',{'aria-label':'Fixed evaluation fixture IDs',placeholder:'Evaluation IDs, comma separated'});
 const requirements=el('input',{'aria-label':'Dataset requirements',value:'goals',placeholder:'goals, stat:ALL:cornerKicks'});
 const versions=el('div',{class:'named-fields'}),status=el('pre',{class:'help'});let selectors={},preview=null,generation=0;
 function invalid(){generation++;preview=null;accept.disabled=true;}
 for(const input of [root,mode,profile,train,test,requirements])input.addEventListener('input',invalid);
 const accept=el('button',{text:'Use previewed selection',disabled:true,onclick:async()=>{if(!preview||preview.generation!==generation)return;recipe.data_selection={root:preview.root,selection:preview.selection_json};changed();status.textContent='Frozen selection accepted: '+preview.fingerprint;try{await refreshChoices();}catch(e){status.textContent=e.message;}}});
 const refresh=el('button',{text:'List dataset versions',onclick:async()=>{try{invalid();versions.replaceChildren();selectors={};const requestGeneration=generation,requestRoot=root.value;const items=await api('named-datasets',{root:requestRoot});if(requestGeneration!==generation||root.value!==requestRoot)return;for(const dataset of ['comprehensive_dataset','international_dataset']){const select=el('select',{'aria-label':dataset});select.append(el('option',{value:'',text:'Not selected'}));for(const item of items.filter(i=>i.dataset===dataset))select.append(el('option',{value:item.version,text:item.version}));select.addEventListener('change',invalid);selectors[dataset]=select;versions.append(el('label',{text:dataset},select));}status.textContent=items.length+' published versions';}catch(e){status.textContent=e.message;}}});
 const inspect=el('button',{text:'Preview selection',onclick:async()=>{
  invalid();const requestGeneration=generation,requestRoot=root.value;
  try{
   const ids=input=>input.value.trim()?input.value.split(',').map(x=>{if(!/^\d+$/.test(x.trim()))throw Error('Use integer fixture IDs');return x.trim();}):[];
   const snapshots=Object.fromEntries(Object.entries(selectors).filter(([_,s])=>s.value).map(([n,s])=>[n,s.value]));
   const options={snapshots,mode:mode.value,profile:profile.value,train_ids:ids(train),evaluation_ids:ids(test),requirements:requirements.value.split(',').map(x=>x.trim()).filter(Boolean)};
   const result=await api('named-selection',{root:requestRoot,options});
   if(requestGeneration!==generation||root.value!==requestRoot)return;
   preview={...result,root:requestRoot,generation:requestGeneration};
   status.textContent=JSON.stringify({classification:preview.classification,targets:preview.targets,history:preview.history,exclusions:preview.exclusions,assumptions:preview.assumptions,coverage:preview.coverage},null,2);accept.disabled=false;
  }catch(e){if(requestGeneration===generation)status.textContent=e.message;}
 }});
 const field=(text,input)=>el('label',{class:'named-field'},el('span',{text}),input);
 box.append(field('Catalogue folder',root),refresh,versions,el('div',{class:'named-fields'},field('Selection mode',mode),field('Evidence profile',profile),field('Required capabilities',requirements),field('Fixed training fixtures',train),field('Fixed evaluation fixtures',test)),el('div',{class:'named-actions'},inspect,accept,el('button',{text:'Use legacy loader',onclick:async()=>{invalid();recipe.data_selection=null;changed();try{await refreshChoices();}catch(e){status.textContent=e.message;}}})),status);
 if(recipe.data_selection)status.textContent='A frozen named selection is active. Legacy source settings are ignored until you select Use legacy loader.';
 return box;
}
