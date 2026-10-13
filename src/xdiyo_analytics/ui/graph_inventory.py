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
    names = ['graphs.TemporalGraphModel', 'graphs.TemporalGraphConfig', 'graphs.TemporalGraphSerializer', 'graphs.GraphHeatmaps',
             'neural.HeatmapCNNModel', 'neural.HeatmapCNNConfig', 'neural.Reconstruction', 'neural.HeatmapCNNSerializer',
             'reporting.GATLossReporter', 'reporting.EmbeddingReporter', 'reporting.GraphAttentionWeightReporter']
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
            nested = {'heatmaps':'graphs.GraphHeatmaps', 'encoder':'neural.HeatmapCNNConfig',
                      'reconstruction':'neural.Reconstruction', 'grid':'features.Heatmap'}
            if name in nested and key.startswith(('graphs.', 'neural.')):
                f.update(kind='component', components=[nested[name]], initial_component=nested[name])
                f['help'] = {
                    'heatmaps':'Optional shared CNN over individual historical match maps. Disabled preserves scalar-only GAT. Requires heatmap_points; target-fixture maps are never inputs.',
                    'encoder':'Shared trainable convolutions, spatial pooling and flattened projection. Parameters are optimized with the prediction head.',
                    'reconstruction':'Optional auxiliary decoder with masked MSE or KL divergence. Disabled creates no decoder or extra loss. Validation prediction loss remains the default early-stopping metric.',
                    'grid':'Native deterministic grid construction. Keep side=both and orientation=team for receiver-aligned GAT edges. Default: player-only 10x10 unit-mass grids.',
                }[name]
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
                f.update(kind='multiselect', choices=['train_loss','validation_loss','train_reconstruction_loss','validation_reconstruction_loss','train_objective','validation_objective'],
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
