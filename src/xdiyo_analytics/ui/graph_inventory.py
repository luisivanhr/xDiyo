"""Form metadata for native graph construction and diagnostics."""
def decorate(components, stages):
    help_text = {
        'timezone': 'IANA timezone for previous-day cutoffs, for example Europe/Berlin or Asia/Tokyo. Use the same setting when loading.',
        'forecast_policy': 'By Round uses the first scheduled kickoff of the complete round. One Day Ahead issues 24 hours before kickoff. Both exclude observations after the previous local day.',
        'round_keys': 'Columns defining a round. Keep competition_id, season_id and round; add a stage column for competitions with repeated round numbers.',
        'embedding_dim': 'Number of learned coordinates in each reusable team representation.',
        'layers': 'Number of synchronous graph message-passing layers; more layers include more distant opponent relationships.',
        'heads': 'Independent attention heads in each layer.',
        'half_life_days': 'Historical edge weight halves after this many days. All eligible history is retained.',
        'seasonal_retention': 'Additional weight multiplier for each crossed season boundary; 1 disables seasonal attenuation.',
        'support_kappa': 'Decayed match mass at which the graph contribution gate reaches one half.',
        'hidden_width': 'Hidden units in the fixture prediction head.',
        'statistics': 'Exact paired native history columns, e.g. team::ALL::Match overview::cornerKicks::value. Include their opponent counterparts in the source history. No xG or rolling columns.',
        'include_goal_features': 'Include native full-time goals for/against as numeric edge predictors and missing-value masks. Disable for explicit period-only statistics; at least one statistic is required. Canonical scores, labels and historical eligibility stay intact.',
        'deterministic': 'Opt in to strict deterministic PyTorch execution for training, predictions and diagnostics, including the optional CNN. Keep the training seed and hardware/runtime fixed. CUDA workspace setup is automatic before first CUDA use; an already initialized session may require a kernel restart. May affect speed. Disabled preserves caller execution settings.',
        'fixture_features': 'Assembled predictor column names to append to the fixture head. Their cutoffs are aligned to the graph during preparation.',
        'learning_rate': 'AdamW learning rate for model construction; tSNE learning rate for EmbeddingReporter (auto or a positive number).',
        'weight_decay': 'AdamW weight decay.',
        'prior_penalty': 'Quadratic penalty on league and entry-context prior embeddings; separate from AdamW decay.',
        'config': 'Architecture, time policy, input columns and optimizer settings for the temporal team graph.',
        'control': 'Native training steps, seed, restarts, early stopping and learning-rate schedule. Candidate validation supplies held-out training rows.',
        'membership_policy': 'First fixture uses the first scheduled kickoff of each competition-season as its effective entry boundary. Explicit requires timezone-aware entry_at in the membership file.',
        'memberships_path': 'Optional CSV or Parquet membership table: team_id, competition_id, season_id, movement and (in explicit mode) entry_at. Disabled uses dataset team_seasons.',
        'mode': 'Choose tSNE for local-neighborhood exploration or PCA for a linear projection with explained variance. Reporting never changes the trained model.',
        'issuance_at': 'Optional timezone-aware issuance present in the selected test/score fixtures. Disabled uses the latest selected issuance separately for each fold.',
        'perplexity': 'tSNE neighborhood scale; capped at number of teams minus one and recorded in the projection summary.',
        'max_iter': 'tSNE optimization iterations (at least 250). This fits the visualization only.',
        'init': 'tSNE initialization. PCA initialization falls back to random for one-dimensional embeddings.',
        'metric': 'Distance metric between original embeddings for tSNE.',
        'early_exaggeration': 'tSNE early exaggeration multiplier.',
        'random_state': 'Seed for reproducible tSNE projections.',
        'whiten': 'PCA-only whitening of projected components.',
        'top_edges': 'Strongest historical edges shown per team/layer/head. Full weights are retained in downloadable tables.',
        'show_badges': 'Display available local team badges with names; unavailable badges fall back to names.',
        'embed_event_store': 'Save the immutable graph inputs inside the model artifact so loading and diagnostics require no source data.',
        'device': 'Device for loading model weights. CPU is portable; CUDA requires a supported installed Torch build.',
    }
    names = ['graphs.TemporalGraphModel', 'graphs.TemporalGraphConfig', 'graphs.TemporalGraphSerializer', 'graphs.GraphHeatmaps', 'graphs.GraphExecution', 'graphs.BoundaryMemory', 'graphs.StatisticTarget', 'graphs.NumericStatisticTarget', 'graphs.StatisticsForecast',
             'neural.HeatmapCNNModel', 'neural.HeatmapCNNConfig', 'neural.Reconstruction', 'neural.HeatmapCNNSerializer',
             'reporting.GraphStatisticsReporter', 'reporting.GATLossReporter', 'reporting.EmbeddingReporter', 'reporting.GraphAttentionWeightReporter']
    choices = {'forecast_policy':['by_round','one_day_ahead'], 'membership_policy':['first_fixture','explicit'],
               'mode':['tSNE','PCA'], 'init':['pca','random'], 'metric':['euclidean','cosine'], 'device':['cpu','auto','cuda:0']}
    for key in names:
        component = components[key]
        for f in component['fields']:
            name = f['name']
            f['help'] = help_text.get(name, f.get('help', name.replace('_',' ').capitalize()))
            f['primary'] = name not in ('catalog','event_store')
            if name in choices:
                f.update(kind='select', choices=choices[name])
            if name == 'config':
                f.update(kind='component', components=['graphs.TemporalGraphConfig'])
            nested = {'execution':'graphs.GraphExecution', 'boundary_memory':'graphs.BoundaryMemory', 'statistics_forecast':'graphs.StatisticsForecast', 'heatmaps':'graphs.GraphHeatmaps', 'encoder':'neural.HeatmapCNNConfig',
                      'reconstruction':'neural.Reconstruction', 'grid':'features.Heatmap'}
            if name in nested and key.startswith(('graphs.', 'neural.')):
                f.update(kind='component', components=[nested[name]], initial_component=nested[name])
                f['help'] = {
                    'execution':'Optional experimental GAT execution. Disabled keeps native execution. Complete required incoming neighborhoods and mathematical objectives are preserved. Floating-point differences can change gradients, optimization trajectories and checkpoint selection; numerical interchangeability is unvalidated. No training or validation passes are skipped. Changing the saved mode or budgets invalidates training resume.',
                    'boundary_memory':'Optional detached state between chronological segments. Changes the model; full history remains the default. States rebuild under each evaluated weight version.',
                    'statistics_forecast':'Optional supervised future-match statistic forecasts. Actual target statistics never enter fixture inputs; choose auxiliary, combined or strict bottleneck.',
                    'heatmaps':'Optional shared CNN over individual historical match maps. Disabled preserves scalar-only GAT. Requires heatmap_points; target-fixture maps are never inputs.',
                    'encoder':'Shared trainable convolutions, spatial pooling and flattened projection. Parameters are optimized with the prediction head.',
                    'reconstruction':'Optional auxiliary decoder with masked MSE or KL divergence. Disabled creates no decoder or extra loss. Validation prediction loss remains the default early-stopping metric.',
                    'grid':'Native deterministic grid construction. Keep side=both and orientation=team for receiver-aligned GAT edges. Default: player-only 10x10 unit-mass grids.',
                }[name]
            if key == 'graphs.GraphExecution':
                f['help'] = {
                    'mode':'Experimental execution ablation: native uses the original full graphs; dependency limits work to complete prediction dependency cones; dependency_lean also prepares only required inputs; dependency_packed also packs bounded disjoint graphs. Numerical results can differ in floating-point rounding. Keep native for bitwise legacy comparisons.',
                    'host_memory_bytes':'Positive byte budget for admission estimates of optimized dependency preparation and packing. Native fallback retains native memory requirements and may exceed this budget. This is not a hard process/RSS cap or a memory reservation; unchanged peak memory requires measured workload evidence. Default 134217728 bytes (128 MiB).',
                    'device_memory_bytes':'Positive byte budget for admission estimates of optimized device workspace. Native fallback retains native requirements and may exceed this budget; allocator and third-party usage are not hard-capped. Unchanged peak memory requires measured workload evidence. Default 268435456 bytes (256 MiB).',
                    'max_graphs':'Maximum disjoint issuance graphs per packed group. Set 1 to disable multi-graph packing while retaining the selected dependency preparation path.',
                    'headroom_bytes':'Minimum free memory headroom in bytes required by execution planning. Default 1073741824 bytes (1 GiB). Budgets do not authorize exceeding available memory.',
                }[name]
                if name == 'mode':
                    f.update(kind='select', choices=['native','dependency','dependency_lean','dependency_packed'])
                else:
                    f.update(kind='number', min=1, step=1)
            if key == 'graphs.BoundaryMemory':
                f['help'] = {'policy':'Season: global union of league entry dates at local midnight. Fixed days: nonoverlapping calendar-day blocks, not a sliding window.',
                    'days':'Number of local calendar days per fixed segment.', 'anchor':'Timezone-aware anchor, at local midnight in the model timezone.',
                    'retention':'Once per boundary, mix carried embedding with current native prior: retention * carry + (1-retention) * prior. One preserves carry.'}.get(name, f['help'])
                if name == 'policy':
                    f.update(kind='select', choices=['season','fixed_days'])
                    for obsolete in ('components','categories','initial_component'): f.pop(obsolete,None)
                if name in ('days','anchor'): f['visible_when'] = {'policy':['fixed_days']}
            if key == 'graphs.StatisticsForecast':
                f['help'] = {'route':'Auxiliary: outcome uses original inputs. Combined: append forecast summaries. Bottleneck: outcome receives only forecast summaries. All train end-to-end.',
                    'weight':'Positive multiplier for the mean of per-target observed-weight-normalized losses.',
                    'point_loss':'Huber or MSE after train-only transformation and standardization. Probabilistic targets use negative log likelihood.',
                    'targets':'Ordered side/period/native statistic targets. Missing targets are masked; each needs usable training coverage.'}.get(name, f['help'])
                if name == 'route': f.update(kind='select', choices=['auxiliary','combined','bottleneck'])
                if name == 'point_loss': f.update(kind='select', choices=['huber','mse'])
                if name == 'targets':
                    f.update(kind='list', item={'kind':'component','components':['graphs.StatisticTarget','graphs.NumericStatisticTarget'],'initial_component':'graphs.StatisticTarget'})
                    f.pop('discovery',None)
            if key == 'graphs.NumericStatisticTarget':
                opts = {'support':['count','nonnegative_continuous','signed_continuous','bounded_continuous'],
                        'transform':['identity','log1p','signed_log1p','affine'],
                        'scaling':['standard','robust','none'], 'side':['home','away'],
                        'period':['ALL','1ST','2ND'], 'family':['quantile']}
                if name in opts: f.update(kind='select', choices=opts[name])
                if name in ('lower','upper'): f.update(kind='number',step='any',required=True,nullable=False,clear_when_hidden=True,visible_when={'support':['bounded_continuous']})
                if name == 'quantiles': f.update(kind='list',item={'kind':'number','step':'any','min':0,'max':1})
                f['help'] = {
                    'source_column':'Exact directed own-team history column (team::...), or goals_for. Missing columns are errors; missing observations are masked.',
                    'semantic_name':'Stable unique identity for this component; home and away are added automatically.',
                    'side':'Upcoming fixture home or away team; independent of historical edge receiver orientation.',
                    'period':'ALL, 1ST or 2ND; must agree with the exact source column.',
                    'lower':'Required lower bound for bounded continuous targets; removed for other support types.',
                    'upper':'Required upper bound for bounded continuous targets; removed for other support types.',
                    'units':'Explicit original units, for example count, percent, or normalized-coordinate squared.',
                    'component':'Exact value, total or named geometric component. Must agree with the source column.',
                    'support':'Counts require integers; signed values remain signed. Bounded continuous requires lower/upper and affine transform.',
                    'transform':'Monotone transform before pinball loss. Use log1p for counts, identity/signed_log1p for signed values, affine for bounded values.',
                    'scaling':'Fit only positive-weight fitting observations: standard deviation, robust interquartile range, or none. Constant scale becomes one.'
                }.get(name, f['help'])
            if key == 'graphs.StatisticTarget':
                from ..graphs.extensions import STATISTICS
                opts = {'statistic':list(STATISTICS), 'side':['home','away'], 'period':['ALL','1ST','2ND'],
                        'family':['point','poisson','negative_binomial','inflated_beta','quantile']}
                if name in opts: f.update(kind='select', choices=opts[name])
                if name == 'quantiles': f.update(kind='list', item={'kind':'number','step':'any','min':0,'max':1}, visible_when={'family':['quantile']})
                f['help'] = {'side':'Upcoming fixture home or away team; independent of historical edge receiver orientation.',
                    'period':'ALL, 1ST or 2ND; exact native period coverage is required.',
                    'statistic':'Exact native statistic identity. Goals currently support ALL only; missing periods are never replaced with totals.',
                    'family':'Point: constrained mean. Counts: Poisson/NB. Possession: endpoint-inflated Beta. Quantile: independently learned ordered quantiles with pinball loss.',
                    'quantiles':'Nonempty unique levels strictly inside (0,1); saved in ascending order. Default .25, .5, .75. Each side/period target learns one output per level. Ties in predicted values are allowed; no rounding.',
                    'weight':'Positive relative target weight; target coverage is normalized separately.'}.get(name, f['help'])
            if key.startswith('neural.') or key == 'graphs.GraphHeatmaps':
                details = {'task':'Classification learns fitting classes; regression predicts numeric targets. Configure the matching target in Labels.',
                           'channels':'Output channels of each convolution, in order. Default 8, 8.',
                           'kernel_size':'Odd convolution kernel width; same padding preserves grid dimensions.',
                           'pool_size':'Average-pooling width and stride. Flattening preserves coarse spatial position.',
                           'embedding_dim':'Number of learned spatial features.',
                           'activation':'Nonlinearity after each convolution and in the standalone prediction/auxiliary heads.',
                           'weight':'Positive multiplier of the selected observed-map reconstruction loss. MSE and KL have different scales. This does not replace prediction loss.',
                           'loss':'MSE averages squared cell errors. KL compares each normalized target map with a per-map softmax decoder distribution, summing over cells; missing maps are excluded.',
                           'hidden_width':'Hidden units in this prediction or reconstruction head.',
                           'batch_size':'Maximum maps/rows processed together. Gradients accumulate before each native optimizer step.',
                           'grid_features':'Full native grid group names, excluding the _yXX_xXX suffix. Empty automatically selects all complete heatmap groups. Use historically wrapped grids of equal resolution.',
                           'numeric_features':'Optional additional prepared numeric columns. Scaling is fitted only on training rows.'}
                f['help'] = details.get(name, f['help'])
                if name in ('task', 'activation'):
                    f.update(kind='select', choices=['classification','regression'] if name == 'task' else ['relu','gelu','tanh'])
                if key == 'neural.Reconstruction' and name == 'loss':
                    f.update(kind='select', choices=['mse','kl'])
                if name in ('channels', 'grid_features', 'numeric_features'):
                    f.update(kind='list', item={'kind':'number','step':1,'min':1} if name == 'channels' else {'kind':'text'})
            if name == 'control':
                f.update(kind='component', components=['training.TrainingControl'])
            if name == 'catalog':
                f.update(kind='reference', initial={'ref':'team_catalog'}, hidden=True)
            if name == 'event_store':
                f.update(hidden=True)
            if name == 'embed_event_store':
                f.update(default=True)
            if name == 'partition':
                f.update(kind='select', choices=['model'] if key.endswith('GATLossReporter') else ['score','test'],
                         default='model' if key.endswith('GATLossReporter') else 'score')
            if name == 'type':
                f.update(kind='select', choices=['overall','per_fold'], default='per_fold')
                f['help'] = 'Per fold selects one fitted model; overall keeps a separate panel or curve for each fold. Encoder coordinates are never averaged.'
            if name == 'metrics' and key.endswith('GATLossReporter'):
                f.update(kind='multiselect', choices=['train_loss','validation_loss','train_reconstruction_loss','validation_reconstruction_loss','train_objective','validation_objective','train_statistics_loss','validation_statistics_loss'],
                         help='Select stored prediction loss, optional reconstruction loss (MSE or KL), or combined objectives. Validation requires a held-out training tail; reconstruction curves require an enabled decoder.')
            if name == 'pooling':
                f.update(choices=['occurrences','first','last'])
            if key.endswith('EmbeddingReporter') and name in ('perplexity','learning_rate','max_iter','init','metric','early_exaggeration','random_state'):
                f['visible_when'] = {'mode':['tSNE']}
            if key.endswith('EmbeddingReporter') and name == 'learning_rate':
                f.update(kind='rule_number', rules=['auto'], number_default=50., min=0.000001)
            if name == 'whiten':
                f['visible_when'] = {'mode':['PCA']}
            if name == 'max_iter': f.update(min=250, step=1)
    for stage in stages.values():
        for f in stage.get('fields', []):
            if f['name'] in ('serializer','model_serializer') and 'components' in f:
                f['components'] = list(dict.fromkeys([*f['components'], 'graphs.TemporalGraphSerializer', 'neural.HeatmapCNNSerializer']))
    export = components.get('experiments.ArtifactExport', {})
    for f in export.get('fields', []):
        if f['name'] == 'serializer':
            f['components'] = list(dict.fromkeys([*f.get('components', []), 'graphs.TemporalGraphSerializer', 'neural.HeatmapCNNSerializer']))
