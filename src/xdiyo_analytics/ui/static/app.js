import {el,clone,configure,discovered,fieldsEditor,componentEditor,mapEditor,listEditor,valueEditor,spec,makeNode,featureMapEditor} from './forms.js';

const token=location.hash.slice(1), content=document.querySelector('#content'), notification=document.querySelector('#notification');
let catalog,recipe,stage='data',job=null,dirty=false;
let publications=[], available={}, discoveryKey='', discoveryBusy=false,preparedColumns=null,preparedKey='',jobPreparationKey='',appliedPreviewJob=null,jobRequestedFoldIds=null;
let outerFoldMetadata=[],evaluatedFoldMetadata=[],evaluatedFoldSelection=null,postSelectionScope=null;
const stages=[
 ['data','Data','Choose available leagues and seasons; inspect the tables and discover statistics.'],
 ['target','Target & layout','Define the observed quantity to predict, then choose one row per match or per team.'],
 ['features','Features & ratings','Compose historical features, rating states, league populations and optional warm-up.'],
 ['evaluation','Evaluation','Choose folds independently of model fitting and model selection.'],
 ['pre','Pre-training analysis','Add descriptive studies and training-only feature selectors.'],
 ['model','Model & training','Choose an estimator, fitted preprocessing, adapter and training controls.'],
 ['search','Model selection','Optionally compare candidates on inner folds and evaluate the selection on outer data.'],
 ['execution','Execution & refit','Set fold jobs, device preferences, checkpoints and optional final refitting.'],
 ['post','Post-training analysis','Choose the diagnostics, match results, betting studies and leaderboard to retain.'],
 ['results','Run & results','Inspect preparation, reopen saved reports and retain fitted models.'],
 ['prediction','Future predictions','Load a trusted saved model and align the requested upcoming rounds.']
];
function notice(message,error=false){notification.textContent=message;notification.classList.toggle('error',error);}
async function api(action,data={}){const response=await fetch('/api/'+action,{method:'POST',headers:{'Content-Type':'application/json','X-Builder-Token':token},body:JSON.stringify(data)});const result=await response.json();if(!response.ok)throw Error(result.error);return result;}
function action(fn){return async (...args)=>{try{await fn(...args);}catch(e){notice(e.message,true);}};}
const changed=()=>{dirty=true;document.title='• Football experiment builder';syncNames();try{localStorage.setItem('football-builder-draft-v2',JSON.stringify(recipe));}catch{}};
const preparationKey=()=>JSON.stringify([recipe.data,recipe.stat_selection,recipe.features,recipe.feature_options,recipe.cutoff_hours,recipe.labels,recipe.target,recipe.assembly,recipe.ratings,recipe.history,recipe.split,recipe.split_options]);
const foldDetails=f=>(f.description?' · '+f.description:'')+' · '+f.train+' train / '+f.test+' test';
function syncNames(){
 if(!recipe)return;
 const key=preparationKey(),selection=JSON.stringify(recipe.fold_ids??null),scope=JSON.stringify([key,selection]);
 // Report IDs are local to the selected outer population. Never silently reuse
 // an explicit report subset for a different population or selection order.
 if(postSelectionScope!==null&&postSelectionScope!==scope&&recipe.analysis_options?.post?.fold_ids!=null)delete recipe.analysis_options.post.fold_ids;
 postSelectionScope=scope;
 if(preparedKey!==key){preparedColumns=null;outerFoldMetadata=[];evaluatedFoldMetadata=[];evaluatedFoldSelection=null;discovered({classes:[]});}
 const outerChoices=outerFoldMetadata.map(f=>({value:f.fold,label:'Fold '+f.fold+foldDetails(f)}));
 let selected=[];
 if(outerFoldMetadata.length){
  const ids=recipe.fold_ids??outerFoldMetadata.map(f=>f.fold);
  if(ids.every(id=>outerFoldMetadata.some(f=>f.fold===id)))selected=ids.map(id=>outerFoldMetadata.find(f=>f.fold===id));
 }else if(selection===evaluatedFoldSelection)selected=evaluatedFoldMetadata;
 const postChoices=selected.map((f,i)=>({value:i,label:'Fold '+i+(recipe.fold_ids!=null?' (outer '+recipe.fold_ids[i]+')':'')+foldDetails(f)}));
 if(selected.length&&recipe.analysis_options?.post?.fold_ids?.some(id=>!postChoices.some(choice=>choice.value===id)))delete recipe.analysis_options.post.fold_ids;
 discovered({fold_ids:postChoices,outer_fold_ids:outerChoices,features:preparedColumns?.features||[],targets:preparedColumns?.targets||Object.keys(recipe.labels||{}),selectors:Object.keys(recipe.fitted_reporters||{}),correlations:Object.entries({...recipe.pre_reporters,...recipe.fitted_reporters}).filter(([,v])=>v?.component==='reporting.CorrelationAnalysis').map(([k])=>k),ratings:Object.keys(recipe.ratings||{})});
}
async function discoverData(force=false){const source=clone(recipe.data),key=JSON.stringify(source);if(!force&&key===discoveryKey)return;discoveryBusy=true;try{const data=await api('discover',{root:source.data_root});const choices=await api('choices',{root:source.data_root,seasons:source.seasons,leagues:source.leagues});if(key!==JSON.stringify(recipe.data))return;publications=data.publications;discovered({seasons:[...new Set(publications.map(p=>p.season))].sort(),leagues:[...new Set(publications.map(p=>p.league))].sort()});available=choices;discovered(available);discoveryKey=key;}finally{discoveryBusy=false;}if(stage==='data')render();}
function optionalComponent(title,key,id,description,ids=[id]){return optional(title,key,()=>{if(!recipe[key]?.component)recipe[key]=makeNode(id);return el('div',{},el('p',{class:'help',text:description}),componentEditor(recipe[key],v=>set(key,v),null,ids));});}
function set(key,value){recipe[key]=value;changed();}
function section(title,description,...children){return el('section',{class:'section'},el('h2',{text:title}),description?el('p',{class:'help',text:description}):null,...children);}
function optional(title,key,editor){const box=el('section',{class:'section'}), checked=recipe[key]!==null&&recipe[key]!==undefined;
 const body=el('div',{class:'section-body'});const draw=()=>{body.replaceChildren();if(recipe[key]!=null)body.append(editor());};
 box.append(el('label',{class:'inline'},el('input',{type:'checkbox',checked,onchange:e=>{set(key,e.target.checked?{}:null);draw();}}),el('h2',{text:title})),body);draw();return box;}
function parameterFields(fields,values,onChange){const shown=fields.filter(f=>f.required||f.primary||Object.hasOwn(values,f.name));const other=fields.filter(f=>!shown.includes(f));return el('div',{},fieldsEditor(shown,values,onChange),other.length?el('details',{class:'advanced'},el('summary',{text:'Additional settings'}),fieldsEditor(other,values,onChange)):null);}
function options(key,schemaKey=key){recipe[key]??={};let fs=catalog.stages[schemaKey].fields;if(key==='split_options')fs=fs.filter(f=>(spec(recipe.split.component)?.inputs||[]).includes(f.name));return parameterFields(fs,recipe[key],v=>set(key,v));}
function analysisTitle(phase){recipe.analysis_options??={};recipe.analysis_options[phase]??={};return fieldsEditor([manual('title','Report title','text','Football analysis')],recipe.analysis_options[phase],()=>changed());}
const manual=(name,title,kind='value',defaultValue=null,extra={})=>({name,title,kind,default:defaultValue,required:false,...extra});
function table(data){const wrap=el('div',{class:'table-scroll'});const t=el('table'), head=el('tr');data.columns.forEach(c=>head.append(el('th',{},el('span',{text:c}))));t.append(el('thead',{},head));const body=el('tbody');for(const row of data.rows){const tr=el('tr');row.forEach(v=>tr.append(el('td',{text:v===null?'—':typeof v==='object'?JSON.stringify(v):v})));body.append(tr);}t.append(body);wrap.append(t);return wrap;}
function showPreview(data){const view=el('div');for(const [name,value] of Object.entries(data)){if(value?.columns){const panel=section(name,`${value.count} rows · preview of the first ${value.rows.length}`,table(value));view.append(name==='metadata'?el('details',{class:'advanced'},el('summary',{text:'Fixture identifiers and context'}),panel):panel);}else if(name==='folds')view.append(section('Fold sizes','Counts refer to assembled dataset rows.',table({columns:['Fold','Train','Test','Score'],rows:value.map(v=>[v.fold,v.train,v.test,v.score])})));}return view;}
function gridEditor(){const box=el('div');const grid=recipe.search.grid??={};const modelFields=spec(recipe.model.component)?.fields||[];const draw=()=>{box.replaceChildren();for(const [key,values] of Object.entries(grid)){const field=modelFields.find(f=>f.name===key)||{kind:'number',title:key};const picker=valueEditor(key,next=>{if(next===key||Object.hasOwn(grid,next))return;delete grid[key];grid[next]=values;changed();draw();},{title:'Tuned parameter',choices:modelFields.filter(f=>['number','boolean','text','select'].includes(f.kind)&&!f.hidden).map(f=>({value:f.name,label:f.title}))});box.append(el('div',{class:'map-row'},el('div',{class:'inline'},picker,el('button',{text:'Remove',onclick:()=>{delete grid[key];changed();draw();}})),el('p',{class:'field-help',text:field.help}),listEditor(values,v=>{grid[key]=v;changed();},{item:{...field,default:field.default??1}})));}box.append(el('button',{text:'+ Add tuned parameter',onclick:()=>{const field=modelFields.find(f=>f.kind==='number'&&!Object.hasOwn(grid,f.name));if(field){grid[field.name]=[field.default??1];changed();draw();}}}));};draw();return box;}
function showHtml(html){
 const frame=()=>el('iframe',{class:'report-frame',title:'Experiment report',sandbox:'allow-scripts allow-downloads',srcdoc:html});
 const wrapper=el('div',{class:'report-view'}),embedded=frame();
 const expand=()=>{
  const dialog=el('dialog',{class:'report-dialog','aria-label':'Expanded experiment report'});
  const close=el('button',{text:'Close expanded report',onclick:()=>dialog.close()});
  dialog.append(el('div',{class:'report-toolbar'},el('h2',{text:'Experiment report'}),close),frame());
  dialog.addEventListener('close',()=>dialog.remove(),{once:true});document.body.append(dialog);dialog.showModal();
 };
 wrapper.append(el('div',{class:'report-toolbar'},el('h2',{text:'Experiment report'}),el('button',{text:'Expand report',onclick:expand})),embedded);
 embedded.addEventListener('load',()=>wrapper.scrollIntoView({block:'start'}),{once:true});
 return wrapper;
}
function modal(...children){document.querySelector('#modal-content').replaceChildren(...children);document.querySelector('#modal').showModal();}

function render(){
 syncNames();
 content.replaceChildren();const definition=stages.find(s=>s[0]===stage);document.querySelector('#stage-title').textContent=definition[1];document.querySelector('#stage-help').textContent=definition[2];
 document.querySelectorAll('nav a').forEach(a=>a.classList.toggle('active',a.dataset.stage===stage));
 if(stage==='data'){
  const select=el('select',{'aria-label':'Season file to inspect'});(available.publications||[]).forEach(p=>select.append(el('option',{value:p.stem,text:p.stem.replaceAll('_',' ')})));
  content.append(section('Source data','Choose the available leagues and seasons. Awarded matches are excluded by default.',parameterFields(catalog.stages.data.fields,recipe.data,()=>{changed();}),el('button',{text:'Refresh available data',onclick:action(()=>discoverData(true))}),el('p',{class:'help',text:discoveryBusy?'Reading available data…':`${publications.length} season files · ${available.stats?.length||0} available statistic identities`})));
  content.append(section('Available observations','Inspect the selected season’s usable tables, without collection bookkeeping.',el('div',{class:'inline'},select,el('button',{text:'Inspect season tables',disabled:!select.options.length,onclick:action(async()=>{const info=await api('inspect',{root:recipe.data.data_root,stem:select.value});modal(el('h2',{text:select.value.replaceAll('_',' ')}),table(info.tables),el('p',{class:'help',text:`${info.stats.length} statistic identities. Choose them by name under Target or Features.`}));})}))));
  content.append(optional('Statistic bundles and extra statistics','stat_selection',()=>options('stat_selection')));
  content.append(section('Team history','Optional fields used when building one row per team and match.',options('history')));
 }else if(stage==='target'){
  content.append(section('Named labels','Label creators use observed outcomes. Future fixtures retain missing outcomes.',mapEditor(recipe.labels,v=>set('labels',v),{componentCategory:['label']})));
  const selector=el('select',{'aria-label':'Training target',onchange:e=>set('target',e.target.value)});Object.keys(recipe.labels).forEach(n=>selector.append(el('option',{value:n,text:n})));selector.value=recipe.target;
  content.append(section('Dataset layout','Select the label and row layout consumed by the downstream model.',el('label',{class:'inline'},'Target',selector),options('assembly')));
 }else if(stage==='features'){
  content.append(section('Named features','Choose a raw Stat as a source, then wrap it in Lag, RollingMean, EMA or another historical operator. Current-match raw observations cannot be used directly as predictors.',featureMapEditor(recipe.features,v=>set('features',v))));
  content.append(section('Historical timing and grouping','Only eligible past results enter historical features. Leave the cutoff disabled for the existing retrospective assumption.',fieldsEditor([manual('cutoff_hours','Hours before kickoff','number',null,{initial:2,help:'Example: 48 simulates a prediction two days before kickoff. Disabled uses earlier finished matches.'})],recipe,()=>changed()),options('feature_options')));
  content.append(section('Named rating states','Add one stream for match results, or enable Statistic to rate corners or another quantity. Reuse its name with a Rating feature. Strength, uncertainty and volatility are separate outputs.',mapEditor(recipe.ratings,v=>set('ratings',v),{itemFields:catalog.stages.rating_options.fields})));
 }else if(stage==='evaluation'){
  content.append(section('Outer evaluation folds','Temporal, grouped, match K-fold and CPCV schemes preserve whole matches.',componentEditor(recipe.split,v=>set('split',v),['split'],['splits.TemporalSplit','splits.MatchKFold','splits.GroupKFold','splits.CPCV'])));
  content.append(section('Fold inputs','Timing vectors or metadata column names must align with this dataset. Override only arguments accepted by the selected splitter.',options('split_options')));
  content.append(section('Selected folds','By default every prepared split is evaluated. For one holdout, enable this setting and choose one fold.',el('button',{text:'Discover folds',onclick:()=>start('prepare',true)}),fieldsEditor([manual('fold_ids','Choose folds','multiselect',null,{discovery:'outer_fold_ids',help:'Discover folds lists all available outer folds, even when this selection is empty. Check exactly one for holdout search; leave disabled to evaluate all folds.'})],recipe,()=>changed())));
 }else if(stage==='pre'){
  content.append(section('Available feature columns',preparedColumns?`${preparedColumns.features.length} assembled feature columns are ready to select.`:'Discover the actual model-input columns before choosing which features to plot. This prepares histories, features and folds without fitting a model.',el('button',{text:preparedColumns?'Refresh feature columns':'Discover feature columns',onclick:()=>start('prepare',true)}),el('p',{class:'field-help',text:'When discovery finishes, enable Features in the reporter and check the columns to plot. Leave it disabled to plot all features. Changes to data, features or layout require discovery again.'})));

  content.append(section('Exploratory reporters','These studies run once for the prepared experiment using each reporter’s explicit scope.',analysisTitle('pre'),mapEditor(recipe.pre_reporters,v=>set('pre_reporters',v),{componentCategory:['pre_reporter']})));
  content.append(section('Fitted feature selection','These reporters run inside candidate fitting on the eligible training population. Set Features from under Model & training to the selector’s name.',analysisTitle('fitted'),mapEditor(recipe.fitted_reporters,v=>set('fitted_reporters',v),{componentCategory:['pre_reporter']})));
 }else if(stage==='model'){
  content.append(section('Estimator','Parameters and explanations come from the maintained catalog. Additional settings expose optional controls. Installed optional estimators appear in this catalog.',componentEditor(recipe.model,v=>{const previous=recipe.model?.component;set('model',v);if(previous!==v?.component)render();},['model'])));
  content.append(section('Fitted preprocessing','Order matters. Each step fits only on fitting rows and is retained with the model. Validation, test and future rows reuse that transform. These steps transform input features; target scaling is configured separately below.',listEditor(recipe.preprocessors,v=>set('preprocessors',v),{categories:['preprocessor']})));
  const countModel=recipe.model?.component==='training.NegativeBinomialRegressor';
  content.append(countModel
   ? section('Raw count labels','Negative Binomial regression fits the original count-scale label. Input preprocessing is still available above. Mean and mode predictions are already in count units, so no target inverse transform is needed.',...(recipe.target_transformer?[el('p',{class:'field-help',text:'Your recipe still has a target scaler selected. Disable it before running this count model.'}),el('button',{text:'Use raw count labels',onclick:()=>{set('target_transformer',null);render();}})]:[]))
   : spec('targets.StandardScaler')
   ? optionalComponent('Scale the target','target_transformer','targets.StandardScaler','Optional for regression: choose a scaler or distribution transform fitted only on training labels. Predictions are automatically converted back to the original units before metrics, reports and future prediction. Disabled keeps labels unscaled.',catalog.components.filter(c=>c.category==='target_transformer').map(c=>c.id))
   : section('Scale the target','Save your recipe, restart the notebook kernel and relaunch the builder to enable target scaling in this older running server.'));
  content.append(section('Prediction outputs','The adapter is chosen automatically for the estimator. Probabilities are available for classifiers.',valueEditor(recipe.prediction_methods,v=>set('prediction_methods',v),{kind:'multiselect',choices:['predict','predict_proba']})));
  content.append(optionalComponent('Custom training adapter','adapter','training.BoostingAdapter','Only needed to override the automatic adapter. Iterative Adapter supports SGD-style partial_fit training controls; Boosting Adapter supports native validation.',['training.EstimatorAdapter','training.BoostingAdapter','training.IterativeAdapter']));
  content.append(section('Candidate, feature selection and training controls','Validation, stopping, scheduling, restart and custom fit statistics are optional and depend on adapter support.',options('candidate')));
 }else if(stage==='search'){
  content.append(optional('Enable model selection','search',()=>{
    recipe.search.options??={metrics:['mse']};recipe.search.split??={component:'splits.MatchKFold',params:{n_splits:3}};
    return el('div',{},el('h3',{text:'Parameter grid'}),el('p',{class:'help',text:'Select an estimator parameter and add its candidate values. The search evaluates their combinations on inner splits. The outer test remains reserved for evaluating the selected model.'}),gridEditor(),
      el('h3',{text:'Selection rule and numerical evidence'}),parameterFields(catalog.stages.search_options.fields,recipe.search.options,v=>{recipe.search.options=v;changed();}),
      el('h3',{text:'Inner folds'}),componentEditor(recipe.search.split,v=>{recipe.search.split=v;changed();},['split'],['splits.TemporalSplit','splits.MatchKFold','splits.GroupKFold','splits.CPCV']),
      fieldsEditor([manual('nested','Repeat selection within each outer fold','boolean',false,{help:'Disabled: tune inside one outer training set, then evaluate its untouched holdout. Enabled: repeat that search independently within each outer fold; this costs more fits.'})],recipe.search,()=>changed()));
  }));
 }else if(stage==='execution'){
  content.append(section('Parallel processing and device','One fold is one job. Select CPU or a device supported by the chosen model.',componentEditor(recipe.execution,v=>set('execution',v),null,['training.ExecutionPolicy'])));
  content.append(optionalComponent('Save training checkpoints','checkpoint','training.CheckpointPolicy','Recover intermediate training only when the adapter supports resumable fitting. Completed runs and search candidates are saved independently.'));
  content.append(optional('Final deployment refit','refit',()=>el('div',{},el('p',{class:'help',text:'Train positions can be “training”, “all”, or an explicit list. For nested selection, Candidate is an explicit configuration override: add model (a component), preprocessors (a list) and candidate (training options) as needed.'}),options('refit'))));
  content.append(section('Run settings','Reuse completed results by default. Completed search candidates remain recoverable after interruption.',options('run')));
 }else if(stage==='post'){
  recipe.analysis_options??={};recipe.analysis_options.post??={};
  content.append(section('Test folds to report','All post-training reporters use held-out predictions. Select which evaluated folds to include; this changes reporting only, not model fitting. Pooled views combine only these folds.',fieldsEditor([manual('fold_ids','Choose test folds','multiselect',null,{discovery:'fold_ids',help:'Disabled includes every evaluated fold. Inspect preparation to see fold sizes, then select one or more folds. Internal early-stopping validation rows are separate.'})],recipe.analysis_options.post,()=>changed())));
  content.append(section('Post-training reporters','New reporters pool evaluation predictions by default. Choose per-fold only when you want separate diagnostics. Test is held-out evaluation; score is its selected scoring subset. Internal validation losses appear in Learning Curve.',analysisTitle('post'),mapEditor(recipe.post_reporters,v=>set('post_reporters',v),{componentCategory:['post_reporter']})));
  content.append(section('Team badges','Known local badges are connected automatically by team ID. Enable Show badges in Match Result Reporter.',fieldsEditor([manual('team_badges','Alternative badge catalog','text',null,{help:'Optional path to a TeamCatalog JSON. Disabled uses the project’s collected local badges.'})],recipe,()=>changed())));
 }else if(stage==='results'){
  content.append(section('Experiment identity','Earlier runs in the same named experiment folder are available to the leaderboard.',fieldsEditor([manual('name','Experiment name','text','', {required:true}),manual('output_dir','Artifacts folder','text','experiments'),manual('config','Experiment metadata','map',{})],recipe,()=>changed())));
  const saved=el('div');content.append(section('Saved runs','Open a retained report without loading data or fitting a model.',el('button',{text:'List saved runs',onclick:action(async()=>{const data=await api('runs',{recipe});saved.replaceChildren();if(!data.runs.length)saved.append(el('p',{class:'empty',text:'No saved runs in this experiment yet.'}));data.runs.forEach(r=>saved.append(el('div',{class:'saved-run'},el('span',{text:r.name}),el('button',{text:'Open report',onclick:action(async()=>{const result=await api('load_run',{recipe,id:r.id});saved.replaceChildren(showHtml(result.html));})}))));})}),saved));
  content.append(section('Latest job','Preparation is inspectable before training.',el('div',{id:'job-result'})));if(job)refreshJob(false);
 }else if(stage==='prediction'){
  content.append(section('Saved model and fixture alignment','Use a model artifact from a trusted producer. Prediction uses the data and features configured in this recipe, including the full history.',fieldsEditor([manual('model_path','Saved model folder','text','',{required:true,help:'Folder created by Save model. The retained model includes its fitted preprocessing.'}),...catalog.stages.prediction.fields,manual('serializer','Custom serializer','component',null,{components:['training.JoblibSerializer'],help:'Optional serialization backend. Disabled uses the saved artifact’s ordinary model loader.'})],recipe.prediction,()=>changed()),el('button',{class:'primary',text:'Predict selected fixtures',onclick:()=>start('predict')})));
  content.append(el('div',{id:'job-result'}));if(job)refreshJob(false);
 }
}

function emptyReporterSelection(){
 for(const [section,page] of [['pre_reporters','pre'],['fitted_reporters','pre'],['post_reporters','post']]){
  for(const [name,reporter] of Object.entries(recipe[section]||{})){
   if(!reporter?.component?.startsWith('reporting.'))continue;
   for(const field of ['features','targets']){
    const selected=reporter.params?.[field];
    if(Array.isArray(selected)&&selected.length===0){
     const labelPlot=field==='targets'&&reporter.component==='reporting.FeatureDistributionReporter',title=labelPlot?'Labels':field==='features'?'Features':'Targets';
     return {page,message:`${page==='pre'?'Pre-training':'Post-training'} analysis → ${name} → ${title}: no columns selected. Choose at least one, or untick ${title} to ${labelPlot?'omit label plots':'use all available columns'}. Use Discover feature columns if choices are not loaded.`};
    }
   }
  }
 }
}
async function start(kind,stayOnPage=false){try{if(kind==='run'){const issue=emptyReporterSelection();if(issue){stage=issue.page;render();notice(issue.message,true);return;}}const submittedKey=preparationKey(),submittedRecipe=clone(recipe);if(kind==='prepare')submittedRecipe.fold_ids=null;const data=await api(kind,{recipe:submittedRecipe});job=data.id;jobRequestedFoldIds=submittedRecipe.fold_ids;jobPreparationKey=submittedKey;appliedPreviewJob=null;notice(kind==='prepare'?'Preparation started. You can continue editing while it runs.':kind==='run'?'Experiment started. The running job uses a snapshot of this recipe.':'Prediction started.');if(!stayOnPage)stage='results';render();refreshJob();}catch(e){notice(e.message,true);}}
async function refreshJob(poll=true){if(!job)return;try{const id=job,data=await api('job',{id});if(id!==job)return;const status=document.querySelector('#job-status');status.textContent=data.message;status.classList.toggle('status-error',data.status==='failed');if(data.preview?.features&&jobPreparationKey===preparationKey()&&appliedPreviewJob!==id){
 appliedPreviewJob=id;preparedKey=jobPreparationKey;
 preparedColumns={features:data.preview.features.columns,targets:data.preview.labels.columns};
 if(data.action==='prepare'||jobRequestedFoldIds==null)outerFoldMetadata=data.preview.folds||[];
 if(data.action==='run'){evaluatedFoldMetadata=data.preview.folds||[];evaluatedFoldSelection=JSON.stringify(jobRequestedFoldIds??null);}
 discovered({classes:data.preview.classes||[]});syncNames();
 if(['pre','post','model','evaluation'].includes(stage))render();
 notice(stage==='evaluation'?'Available folds are ready. Enable Choose folds and check one for a single holdout.':'Feature columns are ready. Enable Features in a reporter to choose what to plot.');
}
if(stage==='prediction'&&data.action!=='predict')return;
 const host=document.querySelector('#job-result');if(host){host.replaceChildren(el('p',{text:data.message}));if(data.preview){host.append(showPreview(data.preview));}if(data.status==='complete'&&data.action==='run'){
  host.append(el('p',{class:'path help',text:(data.reused?'Loaded retained results · ':'Saved · ')+data.path}));
  host.append(el('button',{text:'Open report',onclick:action(async()=>{const r=await api('report',{id});host.append(showHtml(r.html));})}));
  const path=el('input',{placeholder:'experiments/saved_models/lasso','aria-label':'Model save folder'}),fold=el('input',{type:'number',min:0,placeholder:'Fold index (if needed)','aria-label':'Model fold index'}),refit=el('input',{type:'checkbox','aria-label':'Save final refitted model'});
  host.append(el('h3',{text:'Save a fitted model'}),el('div',{class:'inline'},path,fold,el('label',{},refit,' Final refit'),el('button',{text:'Save model',onclick:action(async()=>{const r=await api('save_model',{id,path:path.value,fold_id:fold.value===''?null:Number(fold.value),refit:refit.checked});notice('Model saved to '+r.path);})})));
 }}if(poll&&['queued','running'].includes(data.status))setTimeout(()=>refreshJob(),1500);
 }catch(e){notice(e.message,true);}}

function download(text,filename,mime='text/plain'){const url=URL.createObjectURL(new Blob([text],{type:mime}));const a=el('a',{href:url,download:filename});a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
document.querySelector('#prepare').onclick=()=>start('prepare');document.querySelector('#run').onclick=()=>start('run');
document.querySelector('#save-recipe').onclick=action(async()=>{const r=await api('save',{recipe});dirty=false;document.title='Football experiment builder';notice('Recipe saved to '+r.path);});
document.querySelector('#open-recipe').onclick=action(async()=>{const r=await api('recipes');modal(el('h2',{text:'Saved recipes'}),...r.recipes.map(p=>el('div',{class:'saved-run'},el('span',{text:p.name}),el('button',{text:'Open',onclick:action(async()=>{recipe=(await api('open',{file:p.file})).recipe;document.querySelector('#modal').close();render();await discoverData(true);notice('Recipe opened.');})}))));});
document.querySelector('#import-recipe').onclick=()=>document.querySelector('#import-file').click();
document.querySelector('#import-file').onchange=action(async e=>{const value=JSON.parse(await e.target.files[0].text());await api('export',{recipe:value});recipe=value;changed();render();await discoverData(true);notice('Recipe imported.');});
for(const format of ['python','notebook'])document.querySelector('#export-'+format).onclick=action(async()=>{const r=await api('export',{recipe,format});download(r.text,r.filename);});
window.addEventListener('beforeunload',e=>{if(dirty){e.preventDefault();e.returnValue='';}});
try{catalog=await api('catalog');recipe=catalog.recipe;const draft=localStorage.getItem('football-builder-draft-v2');if(draft){try{const saved=JSON.parse(draft);if(saved.version===1)recipe=saved;}catch{}}configure(catalog.components,changed,catalog.metrics);discovered({bundles:Object.keys(catalog.bundles)});const navigation=document.querySelector('#navigation');stages.forEach(([key,title])=>navigation.append(el('a',{href:'#'+key,'data-stage':key,text:title,onclick:action(async e=>{e.preventDefault();if(stage==='data'&&key!=='data')await discoverData();stage=key;render();})})));render();notice('Loading choices from the selected data…');await discoverData();notice('Data choices are ready. Changes are saved as a local draft; Save recipe creates a reusable file.');}catch(e){notice(e.message,true);}
