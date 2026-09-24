from copy import deepcopy
import json
import numpy as np
import pandas as pd
from support import ROOT, SCRATCH, read, sha
from reporting_samples import sample, plan, Custom, BASE
from xdiyo_analytics.analysis import PreTrainingAnalysis
from xdiyo_analytics.reporting import Artifact, StudyResult, FeatureDistributionReporter, CorrelationAnalysis, FeatureTimeline, TopKCorrelationSelector

data = sample(layout='team_match', shuffle=True)
data.X.loc[data.metadata.case.isin([3, 8]), 'linear'] = np.nan
ids = pd.DataFrame({'exact_id': pd.Series([BASE+2, BASE, BASE+1], dtype='uint64[pyarrow]'),
                    'label': ['third', '<img src=x onerror="window.evil=1">', 'second']})
large = pd.DataFrame({'row': range(205), 'text': ['full download']*205})
report = PreTrainingAnalysis({
    'distributions': FeatureDistributionReporter(type='per_fold', partition='train', features='cycle', bins=3),
    'associations': CorrelationAnalysis(type='per_fold', partition='train', targets='first', methods='pearson'),
    'timeline': FeatureTimeline(type='timeline', partition='all', features='linear'),
    'custom tables': Custom('overall', 'all', lambda ctx: StudyResult('Custom <content>',
        artifacts=[Artifact('table', ids, 'Exact identifiers'), Artifact('text', '</script><script>window.evil=1</script>'),
                   Artifact('html', '<p id="trusted-local">Trusted local extension</p>'),
                   Artifact('custom', {'note': 'flexible renderer'})], tables={'full': large})),
    'selected': TopKCorrelationSelector(type='per_fold', partition='train', k=2, method='pearson', source='associations', targets='first'),
    'consensus': TopKCorrelationSelector(type='overall', partition='train', k=2, method='pearson', source='associations', targets='first', across_folds=True),
}, title='Synthetic reporting verification — no fitted model').run(data, split_plan=plan(data))
renderers = {'custom': lambda artifact: '<section id="custom-renderer">Flexible custom artifact</section>'}
report.to_html(SCRATCH/'browser_report.html', renderers=renderers)
frame = report.to_notebook(height=800, renderers=renderers)._repr_html_()
(SCRATCH/'browser_iframe.html').write_text('<!doctype html><html><head><style>body{background:white;color:black}h1{color:red}</style></head><body><h1 id="parent-heading">Notebook host</h1>'+frame+'</body></html>', encoding='utf-8')
record = {'source_sha256': {name: sha(ROOT/name) for name in read('source_freeze.json')['source_sha256']},
          'html_sha256': sha(SCRATCH/'browser_report.html'), 'iframe_sha256': sha(SCRATCH/'browser_iframe.html'),
          'studies': 6, 'views': len(report.studies), 'fold_rows': [8, 14],
          'exact_ids_descending': [str(BASE+2), str(BASE+1), str(BASE)],
          'synthetic': True}
(SCRATCH/'browser_fixture.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
print({'studies': record['studies'], 'views': record['views'], 'html_bytes': (SCRATCH/'browser_report.html').stat().st_size})
