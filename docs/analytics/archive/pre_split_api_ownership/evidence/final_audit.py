"""Stable-source, documentation and preservation audit; --check-only is read-only."""
from datetime import datetime, timezone
from math import comb
from pathlib import Path
import ast, hashlib, inspect, json, re, sys
import nbformat
from support import ROOT, SCRATCH, guard, read, sha
from xdiyo_analytics import splits as api
from xdiyo_analytics.splits import core, temporal, cpcv

CHECK_ONLY = '--check-only' in sys.argv
baseline, freeze = read('baseline.json'), read('source_freeze.json')
source = guard()
assert source == baseline['source_sha256'] and len(source)==33
protected = baseline['protected_sha256']
assert len(protected)==164
for name,digest in protected.items():
    assert sha(ROOT/name)==digest, ('Earlier protected file changed',name)
reference_notebook = baseline['reference_notebook']
assert sha(reference_notebook['path'])==reference_notebook['sha256']
suite, real, examples, notebook, wheel = [read(name) for name in ['suite_check.json','real_check.json','examples_check.json','notebook_check.json','wheel_check.json']]
for result in [suite,real,examples,notebook]:assert result['source_sha256']==source
assert suite['full_passed']==632 and suite['new_cases']==142 and suite['prior_cases']==490
assert sha(SCRATCH/'full_suite.log')==suite['log_sha256']
for name,digest in suite['test_sha256'].items():assert sha(ROOT/name)==digest
assert '89 passed' in (SCRATCH/'core_temporal.log').read_text(encoding='utf-8-sig')
assert '53 passed' in (SCRATCH/'cpcv.log').read_text(encoding='utf-8-sig')
assert real['status']=='verified' and real['matches']==1372 and real['history_rows']==2744
assert real['publications']==4 and len(real['checks'])==14
assert real['membership_positions_checked']==192240 and real['reconstructed_prediction_cells']==41160
assert real['inputs_unchanged'] and len(real['prepared_source_sha256'])==20
for name,digest in real['prepared_source_sha256'].items():assert sha(name)==digest
assert wheel['no_installs'] and wheel['source_file_count']==77
assert sha(ROOT/wheel['wheel'])==wheel['sha256']
for name,digest in wheel['source_sha256'].items():assert sha(ROOT/name)==digest, ('Wheel source changed',name)
assert wheel['probe']['isolated_wheel_import'] and not wheel['probe']['sklearn_imported']
assert set(wheel['probe']['exports'])==set(api.__all__)

guides = ['docs/analytics/splits.md','docs/analytics/splits_reference.md','docs/analytics/splits_documentation_checklist.md']
docs = [*baseline['archived_docs'],*guides]
def code_hashes(path):
    return [hashlib.sha256(code.encode()).hexdigest() for code in re.findall(r'^```python\n(.*?)^```',path.read_text(encoding='utf-8-sig'),re.M|re.S)]
for guide in ['splits.md','splits_reference.md']:
    assert code_hashes(ROOT/'docs/analytics'/guide)==[entry['code_sha256'] for entry in examples['examples'] if entry['guide']==guide]
assert examples['example_count']==10 and examples['status']=='passed'
archive = ROOT/'docs/analytics/archive/pre_splits_implementation'
preserved_examples = {}
for name in baseline['archived_docs']:
    assert code_hashes(ROOT/name)==code_hashes(archive/name), ('Earlier code example changed',name)
    preserved_examples[name]=len(code_hashes(ROOT/name))
text_reference = (ROOT/guides[1]).read_text(encoding='utf-8')
signatures = {name:str(inspect.signature(getattr(api,name))) for name in api.__all__}
assert len(api.__all__)==8
for name in api.__all__:
    assert name in text_reference
    for parameter in inspect.signature(getattr(api,name)).parameters:
        assert parameter in text_reference, ('Missing public argument',name,parameter)
for cls in [api.Fold,api.SplitPlan]:
    for field in cls.__dataclass_fields__:assert f'`{cls.__name__}.{field}`' in text_reference
for cls in [api.TemporalSplit,api.MatchKFold,api.GroupKFold,api.CPCV]:
    for parameter in inspect.signature(cls).parameters:assert f'`{parameter}`' in text_reference
    signatures[cls.__name__+'.folds']=str(inspect.signature(cls.folds))
for method in ['split','get_n_splits']:signatures['SplitPlan.'+method]=str(inspect.signature(getattr(api.SplitPlan,method)))
signatures['CPCV.path_map']=str(inspect.signature(api.CPCV.path_map))
helpers = ['Splitter','_integer','_duration','_times','_Matches','_Matches.rows','_Matches.constant','_Matches.columns','_Matches.times','_fold','CPCV._check']
for helper in helpers:assert helper in text_reference
for module in [core,temporal,cpcv]:
    tree=ast.parse(inspect.getsource(module))
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):assert all(not item.name.startswith('sklearn') for item in node.names)
        if isinstance(node,ast.ImportFrom):assert not (node.module or '').startswith('sklearn')
matrix = (ROOT/guides[2]).read_text(encoding='utf-8')
assert len(re.findall(r'^\| `',matrix,re.M))==8
if not CHECK_ONLY:assert 'Status: verified.' in matrix and '- [ ]' not in matrix

nb=nbformat.read(ROOT/notebook['notebook'],as_version=4);nbformat.validate(nb)
assert sha(ROOT/notebook['notebook'])==notebook['sha256']
codes=[cell for cell in nb.cells if cell.cell_type=='code']
assert len(nb.cells)==9 and len(codes)==5 and [c.execution_count for c in codes]==[1,2,3,4,5]
assert not any(output.output_type=='error' for cell in codes for output in cell.outputs)
assert notebook['fresh_top_to_bottom_execution'] and notebook['error_outputs']==0
earlier_notebooks=[name for name in protected if name.startswith('notebooks/') and name.endswith('.ipynb')]
assert len(earlier_notebooks)==6
render=read('render_after.json')
assert render['rendered_math_count']==27 and render['rendered_error_count']==0 and not render['errors']
assert sum(document['display_count'] for document in render['documents'])==6
assert not any(item['overflow'] for item in render['layout'])
for document in render['documents']:assert sha(ROOT/document['file'])==document['sha256'], ('Rendered document changed',document['file'])
for table in render['table_rows']:assert all(count==len(table['headers']) for count in table['rows'])
screenshots={f'.pytest_tmp/splits_verification/render_after_{n}.png':sha(SCRATCH/f'render_after_{n}.png') for n in range(1,14)}
for n in range(2,13):
    for k in range(1,n):assert k*comb(n,k)==n*comb(n-1,k-1)

output=ROOT/'docs/analytics/splits_check.json'
assert not output.exists() and not (archive/'preservation_manifest.json').exists(), 'Preserve existing evidence.'
link_count=0
def slug(value):return re.sub(r'\s','-',re.sub(r'[^\w\s-]','',value.lower()))
def links(text,path):
    global link_count
    for target in re.findall(r'\[[^\]\n]*\]\(([^)\s]+)\)',text):
        if target.startswith(('http:','https:','mailto:')):continue
        location,_,anchor=target.partition('#')
        resolved=(path.parent/location).resolve() if location else path
        assert resolved==output or resolved.exists(),('Broken link',path,target)
        if anchor and resolved.suffix=='.md':
            headings=[slug(h.strip()) for h in re.findall(r'^#{1,6}\s+(.+)$',resolved.read_text(encoding='utf-8'),re.M)]
            assert anchor in headings,('Broken anchor',path,target)
        link_count+=1
for name in guides:
    text=(ROOT/name).read_text(encoding='utf-8')
    assert not re.search(r'[ \t]+$',text,re.M)
    links(text,ROOT/name)
for cell in nb.cells:
    if cell.cell_type=='markdown':links(cell.source,ROOT/notebook['notebook'])
for name in baseline['archived_docs']:
    for line in (ROOT/name).read_text(encoding='utf-8').splitlines():
        if re.search(r'\]\([^)]*(?:splits|07_splits)',line):links(line,ROOT/name)
archived={}
for name in baseline['archived_docs']:
    path=archive/name; relative=path.relative_to(ROOT).as_posix()
    archived[relative]=sha(path)
    assert archived[relative]==protected[relative]
assert len(archived)==6
summary={'status':'preflight_passed' if CHECK_ONLY else 'verified','tests_passed':632,'new_tests':142,'source_files_stable':33,'prior_files_unchanged':164,'prior_notebooks_unchanged':6,'matches':1372,'membership_positions_checked':192240,'path_value_cells':41160,'guide_examples':10,'notebook_cells':9,'latex_expressions':27,'local_links_checked':link_count,'evidence':output.relative_to(ROOT).as_posix()}
if CHECK_ONLY:
    print(json.dumps(summary,indent=2));raise SystemExit(0)
scripts=['capture_baseline.py','support.py','run_suite.py','check_real.py','check_examples.py','build_notebook.py','check_wheel.py','render_docs.mjs','complete_docs.py','final_audit.py']
record={
    'status':'verified','verified_at_utc':datetime.now(timezone.utc).isoformat(),
    'scope':'Whole-match split membership, temporal eligibility, retrospective K-fold/CPCV and prediction-path assembly; no worker library edits, model training or reports.',
    'source_freeze_utc':freeze['captured_at_utc'],'source_sha256':source,'tests':suite,
    'focused_tests':{'core_and_temporal_passed':89,'cpcv_passed':53,'first_runs_passed':True,'log_sha256':{name:sha(SCRATCH/name) for name in ['core_temporal.log','cpcv.log']}},
    'real_data':real,'documentation_examples':examples,'notebook':notebook,'wheel':wheel,
    'documentation':{'public_exports':api.__all__,'signatures':signatures,'coverage_rows':8,'internal_helpers':helpers,'all_acceptance_boxes_checked':True,'guide_sha256':{name:sha(ROOT/name) for name in docs},'local_links_checked':link_count,'earlier_code_examples_preserved':preserved_examples,
        'equation_review':{'scoring':'Score is contained in held-out membership and excluded from training.','eligibility':'Earlier kickoff, availability by boundary and all selected target cells present; block and duration gaps are separate.','combinatorics':'All k-block combinations and k*C(N,k)=N*C(N-1,k-1); checked exactly for 66 count pairs.','overlap':'Two closed intervals overlap iff both cross-end inequalities hold; disjoint gaps remain open.','embargo':'Strictly after the latest test end in each contiguous block run, inclusive at end plus duration.','path_limit':'Occurrence assignment gives one copy of every block per path, not independence.'},
        'rendering':{'katex_version':render['katex_version'],'math_expressions':27,'display_equations':6,'errors':0,'display_overflows':0,'sections_visually_reviewed':13,'table_columns_valid':True,'method':'Local KaTeX and marked with protected LaTeX delimiters; Playwright/Chrome at 1200px. Other viewers need compatible math support.','report':'.pytest_tmp/splits_verification/render_after.json','report_sha256':sha(SCRATCH/'render_after.json'),'screenshot_sha256':screenshots},
        'methodology_sources':['https://random-docs.readthedocs.io/en/latest/implementations/cross_validation.html','https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html','https://smallake.kr/wp-content/uploads/2018/07/SSRN-id3104816.pdf']},
    'preservation':{'prior_files_unchanged':164,'protected_sha256':protected,'earlier_notebooks_unchanged':6,'prepared_source_files_unchanged':20,'archived_originals':archived,'reference_notebook':reference_notebook,'source_files_stable':33,'source_changed_since_dispatch':False},
    'workflow':{'implementation_defects':0,'failure_callbacks':0,'success_callbacks':0,'worker_library_edits':0,'installs':0,'primary_thread_id':'01a0a37d-7dc6-7422-9402-227a13350f37'},
    'warnings':['Existing pandas all-NA concatenation FutureWarning in the history test.','Existing Windows/ZMQ selector-thread RuntimeWarning during notebook execution.','Initial focused run reported an unrelated pytest cache-path warning; subsequent suites disabled the cache provider without repairing or deleting that cache.'],
    'limits':['Kickoff is a retrospective availability proxy unless actual timestamps are supplied.','Real explicit availability and path values were synthetic verification inputs, not observed completion times or forecasts.','Retrospective feature/state reconstruction and fitting provenance remain outside this module.','CPCV paths are not independent samples; no DSR/PBO/FDR or finance calculations were validated or adopted.'],
    'review_scripts_sha256':{'.pytest_tmp/splits_verification/'+name:sha(SCRATCH/name) for name in scripts},
}
assert guard()==source
with output.open('x',encoding='utf-8') as stream:json.dump(record,stream,indent=2,allow_nan=False);stream.write('\n')
manifest={'created_at_utc':record['verified_at_utc'],'original_sha256':archived,'verified_unchanged':True,'current_document_sha256':record['documentation']['guide_sha256'],'evidence':{'path':output.relative_to(ROOT).as_posix(),'sha256':sha(output)},'primary_archived_first':['IMPLEMENTATION_PROGRESS.md','football_analytics_working_notes.md']}
with (archive/'preservation_manifest.json').open('x',encoding='utf-8') as stream:json.dump(manifest,stream,indent=2);stream.write('\n')
print(json.dumps(summary,indent=2))
