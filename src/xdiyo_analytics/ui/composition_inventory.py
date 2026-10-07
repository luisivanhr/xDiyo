"""Native forms for composition and opt-in research allocation."""
def decorate(components):
    mappings = {
        'schema':['composition.OutputSchema'], 'target_spec':['composition.TargetSpec'],
        'training':['composition.TrainingPlan'], 'training_plan':['composition.TrainingPlan'],
        'graph':['composition.PredictionGraphSpec'], 'features':['composition.OutputFeatures'],
        'meta_features':['composition.OutputFeatures'], 'reducer':['composition.Mean','composition.HardVote','composition.DistributionMixture'],
        'transform':['composition.OutputFeatures'], 'calibration':['training.ProbabilityCalibrator'],
        'weighting':['weighting.ClassWeightPolicy'],
        'base':['composition.ModelNode'], 'ticket_gate':['evaluation.DecisionLayer'],
        'stake_policy':['evaluation.FixedStake','evaluation.FixedFraction','evaluation.FractionalKelly','evaluation.LearnedAllocation'],
        'stake_context':['evaluation.StakeContext'], 'risk_limits':['evaluation.RiskLimits'],
        'probability_source':['evaluation.ModelProbabilitySource','evaluation.HistoricalRateSource'],
    }
    maps = {'models':['composition.ModelNode','composition.Estimator'], 'base_models':['composition.ModelNode'],
            'nodes':['composition.ModelNode','composition.TransformNode','composition.ReducerNode'],
            'schemas':['composition.OutputSchema'], 'outputs':['composition.OutputRef']}
    for key,entry in components.items():
        relevant = key.startswith(('composition.','evaluation.')) or key in ('reporting.BetOutcomeReporter','reporting.BetPerformanceReporter','experiments.ArtifactExport')
        if not relevant:
            continue
        for f in entry['fields']:
            name=f['name']
            if name in mappings and (key.startswith('composition.') or name in ('ticket_gate','stake_policy','stake_context','risk_limits','probability_source')):
                f.update(kind='component',components=mappings[name],initial_component=mappings[name][0])
            if key.startswith('composition.') and name in maps:
                f.update(kind='map',item={'kind':'component','components':maps[name],'initial_component':maps[name][0]})
            if key.startswith('composition.') and name in ('model','meta_model','correction'):
                f.update(kind='component',components=['composition.Estimator','composition.LearnedTargetAdapter'],initial_component='composition.Estimator')
            if key in ('evaluation.LearnedAllocation','evaluation.LearnedGate') and name=='model':
                f.update(kind='component',components=['composition.SavedUtilityModel'],initial_component='composition.SavedUtilityModel')
            if key=='composition.Estimator' and name=='estimator':
                f.update(kind='component',categories=['model'],initial_component='sklearn.linear_model.Ridge')
                f.pop('initial',None)
            if key=='composition.Estimator' and name=='preprocessors':
                f.update(kind='list',item={'kind':'component','categories':['preprocessor']})
            if key=='composition.Estimator' and name=='target_transformer':
                f.update(kind='component',categories=['target_transformer'])
            if key=='composition.Estimator' and name=='prediction_methods':
                f.update(kind='multiselect',choices=['predict','predict_proba','decision_function'])
            if key.startswith('composition.') and name=='inputs':
                f.update(kind='list',item={'kind':'component','components':['composition.OutputRef'],'initial_component':'composition.OutputRef'})
            if key.startswith('composition.') and name in ('names','passthrough','row_keys','classes','support','parameters','group_by'):
                f.update(kind='list',item={'kind':'text'})
            if key=='composition.Mean' and name=='weights':
                f.update(kind='list',item={'kind':'number','min':0,'default':1})
            if key=='composition.OutputSchema' and name=='kind':
                f.update(kind='select',choices=['regression','probability','labels','vote_fraction','distribution','features'])
            if key=='composition.TrainingPlan' and name=='mode':
                f.update(kind='select',choices=['parallel','chronological','algorithmic_residual','honest_error_meta'])
            if key=='composition.TargetSpec' and name=='kind':
                f.update(kind='select',choices=['ordinary','residual','oof_error','take_skip','allocation'])
            if name in ('payoff','single_payoff'):
                f.update(kind='select',choices=['push_void','binary'],help='Kelly and binary EV gates require an explicitly declared binary payoff. Push/void/partial-return valuations are unsupported by those policies.')
            if name=='probability_columns':
                f.update(kind='map',item={'kind':'text'},help='Model name to retained leg-probability column. Each model also needs issued_at, artifact_vintage and trained_through provenance.')
            if name=='probability_outputs':
                f.update(kind='map',item={'kind':'text'},help='Model name to upstream named probability output. Produces model::probability leg columns for ticket gates.')
            if name=='model_timing':
                f.update(kind='map',item={'kind':'map','item':{'kind':'text'}},help='Per model: issued_at, artifact_vintage and trained_through UTC timestamps. Programmatic recipes may supply {column: name} for row-specific metadata.')
            if key=='evaluation.DecisionLayer' and name=='models':
                f.update(kind='list',item={'kind':'text'})
            if key=='evaluation.DecisionLayer' and name in ('gate','metric','missing'):
                f.update(kind='select',choices={'gate':['and','or'],'metric':['ev','probability'],'missing':['error','reject']}[name])
            if name=='serializer' and key=='experiments.ArtifactExport':
                f.update(kind='component',components=['training.JoblibSerializer','composition.CompositeSerializer','training.BayesianScoreSerializer'])
            if name in ('currency','basis'):
                f['help']='Declare stable research currency/units and either realized_wealth or available_cash as the sizing basis. Illustrative settings are not real-money instructions.'
