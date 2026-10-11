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
    names = ['graphs.TemporalGraphModel', 'graphs.TemporalGraphConfig', 'graphs.TemporalGraphSerializer',
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
                f.update(kind='multiselect', choices=['train_loss','validation_loss'],
                         help='Select stored cross-entropy loss curves. Validation is available only when a held-out training tail was configured.')
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
                f['components'] = list(dict.fromkeys([*f['components'], 'graphs.TemporalGraphSerializer']))
    export = components.get('experiments.ArtifactExport', {})
    for f in export.get('fields', []):
        if f['name'] == 'serializer':
            f['components'] = list(dict.fromkeys([*f.get('components', []), 'graphs.TemporalGraphSerializer']))
