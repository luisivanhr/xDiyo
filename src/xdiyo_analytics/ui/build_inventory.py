"""Developer command: python -m xdiyo_analytics.ui.build_inventory.

Refresh after deliberately adding/changing a registered constructor. Review the
JSON diff; the web server reads that file rather than regenerating form schemas.
"""

import inspect
import json
import re
from pathlib import Path

from .catalog import _decorate
from .recipe import catalog_for_ui
from .schema import stage_schema
from ..features.point_geometry import POINT_FIELDS


HELP = {
    'left': 'First single-output feature. For Difference, the right feature is subtracted from this one. Choose historical features or known context.',
    'right': 'Second single-output feature. Select the same period and perspective deliberately; raw current-match statistics cannot be prediction inputs.',
    'numerator': 'Single-output feature above the fraction bar. Example: recent rolling average.',
    'denominator': 'Single-output feature below the fraction bar. Example: long-run rolling average. Zero gives a missing ratio by default.',
    'zero_value': 'Optional finite replacement when the denominator is exactly zero and both inputs are known. Disabled leaves the ratio missing; missing inputs always stay missing.',
    'team_seasons': 'Optional table keyed by competition_id, season_id and team_id, with movement and optional previous-league/season IDs. Disabled recognizes retained teams from loaded history; a new appearance alone does not establish promotion or relegation.',
    'season_starts': 'Optional UTC season-entry boundaries used to freeze transition priors. Disabled uses the earliest prediction cutoff per league-season (first kickoff when no cutoff is supplied). Explicit boundaries cannot follow that first cutoff.',
    'transition_context': 'Optional shared transition inputs for warm-up. This reuses history and membership information across rating streams; it does not choose which stream is warmed.',
    'team_counts': 'Optional count Series indexed by competition_id and season_id for normalizing standings. Disabled counts distinct teams and opponents in the loaded history before fold filtering; a partial league sample may need full-population counts supplied here.',
    'ratings': 'Named rating states available to Rating features. Configure these in Named rating states.',
    'push_value': 'Numeric label assigned when a bet pushes: its stake is returned. This changes label encoding, not settlement accounting.',
    'void_value': 'Numeric label assigned when a bet is void: it is cancelled and its stake is returned.',
    'league': 'Initially filter displayed fixtures to this league. Disabled shows all available leagues.',
    'season': 'Initially filter displayed fixtures to this season. Disabled shows all available seasons.',
    'team': 'Initially show fixtures involving this team, home or away. Disabled includes every team.',
    'round': 'Initially filter displayed fixtures to this round. Disabled includes every round.',
    'league_column': 'Metadata column containing the league identifier used by the report filter.',
    'season_column': 'Metadata column containing the season identifier used by the report filter.',
    'team_column': 'Metadata column identifying the entity whose feature history is plotted.',
    'classes': 'Observed class values, in the order expected by the estimator or probability output.',
    'bets': 'Add named betting options with decimal odds, stakes and explicit placement decisions.',
    'labels': 'Observed outcomes used to settle the selected betting options.',
    'option': 'Betting outcome to settle, such as total corners over a chosen line.',
    'history': 'Team-match history providing observed outcomes and past information. The experiment supplies this automatically.',
    'run_group': 'Optional experiment-run identifier linking one final result and its search trials. Disabled includes every run group in this experiment store. Include trials controls trial visibility separately; statistical comparison groups are a different concept.',
    'path': 'Local file or saved-model directory to load. Relative paths are resolved from the experiment workspace.',
    'directory': 'Directory containing previously saved rating states.',
    'column': 'Column to read from the auxiliary input table; disabled loads the complete table.',
    'record_dir': 'Optional directory for loader resolution records. Disabled uses the loader default.',
    'categories': 'Optional attack/defense category definitions used when expanding statistic bundles.',
    'competition_info': 'League metadata used to identify movement between divisions when constructing team-season transitions.',
    'season_years': 'Optional mapping of season identifiers to their chronological years.',
    'sample_weight_column': 'Prepared metadata column containing observation weights passed to the boosting fit.',
    'partial_fit_kwargs': 'Additional arguments for the estimator’s incremental partial_fit call. Classification classes are handled by the backend.',
    'preprocessor': 'Fitted input transformations retained with the model. Each attempt starts with fresh preprocessing fitted on its training rows.',
    'estimator': 'The estimator selected in Model & training, supplied automatically by the experiment.',
    'name_fields': 'Configuration fields included in the automatically generated run name.',
    'config': 'Additional descriptive configuration stored with the run for identification and comparison.',
    'serializer': 'Serialization backend for reading a previously saved model. The default reads the library’s ordinary saved-model format.',
    'candidate': 'Explicit deployment candidate used for final refitting. Leave disabled to reuse the selected model configuration.',
    'type': 'Overall combines the selected evaluation rows in one study. Per fold provides a separate view of each split. Timeline preserves chronological order.',
    'partition': 'Choose the rows analyzed. Training rows describe fitting data; test contains held-out predictions; score is the configured scoring subset of test.',
    'pooling': 'When a match is predicted in multiple folds, choose which occurrence to retain. First/last uses one prediction per row; mean averages predictions; occurrences retains every prediction.',
    'source': 'Choose the observed statistic or expression consumed by this calculation.',
    'stat': 'Leave disabled for win/draw/loss ratings. Enable to compare a statistic such as corner kicks instead.',
    'side': 'For uses this team’s values; against uses its opponent’s values; both creates separate outputs.',
    'period': 'Full match, first half or second half. Only periods present for the selected statistic are offered.',
    'field': 'Value is the numeric statistic. Total is its denominator when supplied; display is the original formatted value.',
    'window': 'Number of eligible past observations in the rolling window. Example: 5 uses the last five matches; a League source can use rounds or days.',
    'periods': 'How many eligible observations to lag. 1 means the most recent completed match.',
    'min_periods': 'Minimum observed values needed for a result. 1 allows a partial window early in the season.',
    'ddof': 'Variance denominator is n − ddof. 1 estimates sample spread; 0 computes population spread.',
    'reference': 'Historical value compared with the rolling baseline. Disabled uses the latest eligible observation, included in that window.',
    'span': 'EMA memory: alpha = 2 / (span + 1). A smaller span reacts faster to recent results.',
    'fields': 'Choose the rating outputs: rating is strength, rd is uncertainty, sigma is volatility.',
    'scope': 'Keep rating streams separate by these identifiers. Competition gives each league its own ratings.',
    'initial_rating': 'Starting strength for an unseen team. 1500 is the conventional Glicko scale center.',
    'initial_rd': 'Starting uncertainty in rating points. Larger values let early results move ratings more.',
    'initial_sigma': 'Starting volatility: how quickly a team’s underlying strength may change. Typical starting value: 0.06.',
    'tau': 'Limits how rapidly volatility can change. Larger tau allows more adaptation; it is not a rolling window.',
    'engine': 'Rating algorithm and initialization. The same settings apply to both teams before each update.',
    'transition': 'Optional season/movement adjustment. Disabled preserves the engine’s ordinary state evolution.',
    'higher_is_better': 'If enabled, the team with the larger statistic receives the win. Disable for quantities where smaller is better.',
    'phi_scale': 'Multiply rating uncertainty at a season boundary while retaining mean strength. 1 leaves it unchanged.',
    'movement_phi_scale': 'Uncertainty multiplier for a team changing leagues. 1 leaves it unchanged.',
    'shrinkage': 'Fraction of the destination-league rating prior to blend into a moving team’s old mean. 0 keeps its old mean; 1 uses the prior.',
    'top': 'Number of top teams in the destination league used to initialize a relegated team.',
    'bottom': 'Number of bottom teams in the destination league used to initialize a promoted team.',
    'rank_by': 'Rank the rating-prior cohort by frozen standings or rating strength.',
    'aggregate': 'Use the cohort mean or median for the rating transition prior.',
    'schedule': 'Completed rounds freezes one baseline for a target round; kickoff updates as earlier match results become eligible.',
    'window_unit': 'Interpret the league rolling window as completed rounds, matches or elapsed days.',
    'round_keys': 'Identifiers defining a round. Keep competition, season and round together for separate league calendars.',
    'exclude': 'Remove this team’s contributions, or remove all fixtures involving it. The selected window is not refilled.',
    'policy': 'Optional prior-seeded EMA and handoff to ordinary rolling calculations.',
    'handoff': 'Hard switches after a fixed number of rounds; LinearFade blends over rounds; ObservationCount reduces prior weight as samples accumulate.',
    'rounds': 'Number of rounds for this setting. For prediction, choose the upcoming rounds to align.',
    'train_size': 'Amount of past data used for fitting, measured in the selected unit. Example: 2 seasons.',
    'test_size': 'Size of each held-out test window, measured in the same unit as training.',
    'step': 'How far to move the next split. Disabled uses the splitter’s default step.',
    'gap': 'Number of blocks to skip between training and testing, measured in Gap unit. Gap matches do not become training rows.',
    'gap_unit': 'Disabled uses the window unit. Choose Rounds with season windows to skip opening test rounds: gap 10 starts testing at round 11 in each league and keeps the season end. Kickoffs counts simultaneous kickoff batches. Choose the same or a finer unit than the window.',
    'gap_time': 'Elapsed-time gap, for example 2D or 12h, in addition to unit boundaries.',
    'score_start': 'Skip this many held-out blocks before scoring (in the splitter’s chosen unit). Unscored rows stay held out; feature histories remain causal.',
    'score_rounds': 'Inclusive round-number range [first, last] to score in each test season. Either bound may be left empty. Disabled scores all eligible test rounds.',
    'allow_partial_test': 'Include a final test window even if fewer than test_size units remain.',
    'calendar_by': 'Advanced override for independent calendars. Seasons pools leagues by default; League-seasons and Rounds separate competitions. An enabled empty selection pools calendars.',
    'block_by': 'Optional metadata identifiers that must stay together at split boundaries.',
    'n_splits': 'Number of held-out folds. More folds require more model fits.',
    'shuffle': 'Randomize whole matches/groups before splitting. This does not simulate chronological forecasting.',
    'random_state': 'Seed for repeatable randomization. Set a fixed integer to reproduce shuffled splits.',
    'group_by': 'Rows with the same selected metadata identifiers stay together.',
    'n_blocks': 'Number of chronological blocks in CPCV. Every selected combination is evaluated.',
    'n_test_blocks': 'Blocks held out per CPCV split. Example: 6 blocks and 2 test blocks produces 15 fits.',
    'embargo': 'Time excluded after held-out information intervals, for example 2D. This reduces overlap between fitting and evaluation.',
    'methods': 'Choose association measures. Pearson measures linear association; Spearman and Kendall compare ranks; MCC compares categorical decisions.',
    'percentile_ranks': 'Show each feature’s percentile rank by absolute association alongside its coefficient.',
    'categorical_features': 'For MCC only: features already representing categorical decisions. Continuous features need an explicit threshold instead.',
    'categorical_targets': 'For MCC only: targets already representing categories.',
    'feature_thresholds': 'For MCC only: explicit numeric thresholds turning selected features into binary decisions.',
    'target_thresholds': 'For MCC only: explicit numeric thresholds turning selected labels into binary outcomes.',
    'metrics': 'Select the calculations to report. Each metric exposes only its own supported options.',
    'metric': 'Metric used to compare candidates. The registered direction determines whether smaller or larger is better.',
    'features': 'Disabled includes all eligible predictors. When enabled, select at least one checkbox; an empty selection does not mean all. Discover feature columns first if no choices are shown. This reporter setting does not change the model inputs.',
    'targets': 'Disabled includes all labels supplied to this analysis. When enabled, select at least one checkbox; an empty selection does not mean all.',
    'target': 'Observed label to analyze. Disabled uses all available labels.',
    'features_from': 'Reuse a named feature selector’s retained columns.',
    'feature_columns': 'Select model input columns. Disabled uses every assembled feature.',
    'target_columns': 'Select label columns. Disabled uses the assembled target.',
    'k': 'Number of highest-ranked features to keep.',
    'method': 'Association measure used to rank features by absolute strength.',
    'across_folds': 'Choose one set of winners across fold results. Overall scope calculates association once on unique pooled rows.',
    'fold_k': 'Number of candidates retained within each fold before voting across folds.',
    'bins': 'Histogram bin rule. Auto selects the number from the observed distribution.',
    'bandwidth': 'KDE smoothing rule. Scott and Silverman estimate smoothing from the available sample.',
    'grid_size': 'Number of points used to draw the smooth density curve.',
    'entity': 'Show one series per team or a league-level series.',
    'teams': 'Optional team subset for the displayed timeline.',
    'max_points': 'Maximum plotted observations per series. Disabled retains all points.',
    'output': 'Prediction output to analyze: point/class predictions or class probabilities.',
    'comparison': 'Auto follows the label definition; numeric uses a tolerance; categorical compares class identity.',
    'show_badges': 'Embed available local team badges in the saved match report, with team names as fallback.',
    'page_size': 'Number of fixtures shown on one page of the match-results table.',
    'tolerance': 'Allowed numerical tolerance for this calculation. Zero uses exact comparison.',
    'include_zeros': 'Include coefficients eliminated by regularization. Disabled focuses on surviving features.',
    'survival_frequency': 'Summarize how often each feature has a nonzero coefficient across fitted models.',
    'decision': 'Choose the rule for converting evidence into a decision.',
    'threshold': 'Numeric decision boundary. For probability predictions use a value between 0 and 1.',
    'positive_class': 'Class treated as the positive outcome for a probability threshold.',
    'positive_label': 'Observed class treated as positive by this metric.',
    'weights': 'Relative importance of selected metrics. Use nonnegative weights; the scoring rule normalizes them.',
    'direction': 'Minimize losses such as MSE; maximize scores such as accuracy. Disabled uses the metric’s registered direction.',
    'scaling': 'Percentile combines relative metric ranks; fixed uses the supplied reference scales.',
    'odds': 'Decimal odds including returned stake. Example: 2.5 pays 2.5 times the stake on a win.',
    'stake': 'Amount risked on each selected bet.',
    'take': 'Explicit betting decisions aligned with retained predictions.',
    'selection': 'Choose the desired bet outcome or optimizer selection strategy.',
    'line': 'Market threshold, for example 9.5 for over/under total corners.',
    'on_equal': 'Settlement when the observed statistic equals the market line.',
    'draw': 'Settlement of a draw in a selected win/loss market.',
    'perspective': 'Express the outcome from the team, home team or away team perspective.',
    'n_jobs': 'Number of folds trained concurrently. 1 runs sequentially; -1 requests all available CPUs.',
    'inner_threads': 'CPU threads per fold. Limits nested parallelism when several folds run together.',
    'device': 'Requested compute device. CPU works for all standard estimators; CUDA requires a compatible model and installed runtime.',
    'gpu_jobs': 'Maximum simultaneous GPU fits. 1 avoids competing for the same GPU memory.',
    'control': 'Optional iterative stopping, scheduling and restarts, for adapters that support them.',
    'validation': 'Reserve a trailing part of the current training population for training controls. Outer test data is not used.',
    'fraction': 'Fraction of fitting population reserved for internal validation. Example: 0.2 reserves the latest 20%.',
    'max_steps': 'Maximum optimization steps per attempt.',
    'patience': 'Number of steps without sufficient improvement before triggering the control.',
    'min_delta': 'Smallest improvement counted as progress.',
    'factor': 'Multiply the learning rate by this factor when progress stalls.',
    'min_lr': 'Lower bound for the learning rate.',
    'restarts': 'Additional optimization attempts. Zero trains once.',
    'seed': 'Seed used for reproducible optimization attempts.',
    'restore_best': 'Restore the best observed checkpoint when stopping, if supported by the adapter.',
    'monitor': 'Loss or metric name exposed by the training adapter.',
    'early_stopping': 'Stop an iterative fit when validation improvement stalls.',
    'scheduler': 'Optionally reduce learning rate when progress stalls.',
    'every': 'Save an adapter checkpoint after this many training steps.',
    'unsupported': 'Skip checkpointing or raise when the selected adapter cannot save resumable state.',
    'reuse': 'Reuse a matching completed fit and its saved models. Post-training reporters are refreshed without fitting again. Disable to deliberately train a fresh run.',
    'calibration': 'Optional calibration fitted on a separate chronological training tail, excluded from base model/scaler fitting and feature selection. Choose probability input for existing classifiers, or binary SVC decision margins with internal probabilities disabled and explicit label availability.',
    'save_models': 'Save fitted fold models, final refits and their fitted scalers for automatic restoration. Enabled by default. Load only trusted experiment folders.',
    'name': 'Human-readable name used in the recipe and report.',
    'data_root': 'Folder containing published season manifests and tables. Example: data/xDiyo_data.',
    'seasons': 'Select the seasons available in this data folder.',
    'leagues': 'Select available leagues. Disabled includes all leagues for the chosen seasons.',
    'tables': 'Select source tables. Matches identify fixtures and include published season-entry flags; statistics supplies observations; pregame supplies standings. Team seasons adds movement evidence and predecessor IDs for feature warm-up.',
    'stat_fields': 'Numeric statistic fields retained in team history. Value is the ordinary count or percentage; total is an optional denominator.',
    'include_awarded': 'Include administratively awarded matches. Disabled excludes them from training and histories.',
    'bundles': 'Convenience groups of statistics. Combine bundles with individually selected extra statistics.',
    'stats': 'Add specific statistics to the selected bundles.',
    'layout': 'Match creates one row with home/away features. Team match creates two rows, one for each team.',
    'drop_missing_targets': 'Exclude rows without observed labels during training. Future prediction keeps those fixtures.',
    'train_positions': 'Training uses the original fitting partition; all refits on every available labeled row for deployment.',
    'statuses': 'Fixture statuses eligible for future predictions.',
    'as_of': 'UTC reference time for identifying upcoming fixtures. Disabled uses the loader’s current-time default.',
    'cutoffs': 'Optional information cutoff aligned with each row. The hours-before-kickoff control is the simpler choice.',
    'available_at': 'Optional result-availability timestamps. Disabled uses the documented retrospective finished-match assumption.',
    'information_start': 'Start of each row’s information interval for purged splits.',
    'groups': 'Metadata column identifying groups that must remain together.',
}

DESCRIPTIONS = {
    'features.SpatialHistoricalDeviation': 'Observed JS distance from each match’s own prior mean map. Wrap in Rolling mean or Lag to use past deviations as predictors. Inner and outer windows are separate. Default is a 6x6 player-only grid; any grid size of at least 2 is supported.',
    'features.SpatialFixtureDistance': 'One match-level JS distance between the two teams’ prior mean maps. Choose style similarity or a directional opponent-context comparison. Defaults to 6x6; change Source → Grid size for another resolution.',
    'training.SpatialPCA': 'Fit a shared PCA basis on training-fold historical maps only. Declare ordered home/away grid-column blocks with the same resolution and team frame. Missing cells use training medians. Place before ordinary scalers/imputers to retain named columns.',
    'training.SpatialClusters': 'Fit a shared training-fold map codebook and output distances to its centers. Ordered home/away grid blocks share one codebook; cluster numbers are not tactical categories. Place before other preprocessing.',
    'features.SpatialEntropy': 'Measure how evenly activity is spread across a fixed grid. Rolling mean of entropy describes a typical match; entropy of a rolling-mean grid describes the historical mixture. Missing maps stay missing.',
    'features.SpatialConcentration': 'Choose effective occupied cells, HHI or another concentration summary of a unit-mass grid. Choose one alternative at a time; effective cells is a transform of entropy.',

    'features.SpatialPointSummary': 'Exact raw-point means, population spreads, robust widths and covariance geometry. Uses team-relative normalized coordinates and needs a historical wrapper before prediction.',
    'features.Lag': 'Take an earlier eligible observation. A lag of 1 gives the most recent completed match before the prediction cutoff.',
    'features.Abs': 'Absolute value of one feature or finite constant. Missing inputs stay missing and zeros stay zero. Can be used before or after historical reduction; these orders have different meanings.',
    'features.Product': 'Multiply two features, or a feature and a finite constant. Both operands must produce one column. Missing inputs and overflow remain missing. Nest Product to multiply more than two features.',
    'features.RollingMean': 'Average the selected statistic over eligible past observations. A window of 5 uses up to five previous matches, excluding the match being predicted.',
    'features.RollingWeightedMean': 'Average one numeric feature using another nonnegative feature as its weight. Uses one fixed eligible-match window and paired available values. Minimum periods counts only finite pairs with positive weights.',
    'features.RollingSkewness': 'Measure asymmetry in earlier observations: positive means a longer upper tail, negative a longer lower tail. Uses population third and second central moments; at least three finite values and nonzero spread are required. Supports League, LeaveOneOut and SeededEMA warm starts.',
    'features.RollingStd': 'Measure variability across the observations in the past window. For league-round windows this uses individual observations, not averages of rounds.',
    'splits.TemporalSplit': 'Train on earlier matches and test on later matches. Use for forecasting a future season or round. Expanding keeps all earlier training data; sliding keeps a fixed-length window.',
    'splits.MatchKFold': 'Hold out whole matches in K groups. With shuffle enabled this tests generalization across matches, not a realistic future forecast. Both team rows always stay together.',
    'splits.GroupKFold': 'Hold out whole leagues, seasons or other metadata groups. Use to study transfer to groups absent from training; it does not enforce chronological order.',
    'splits.CPCV': 'Combinatorial purged cross-validation holds out combinations of chronological blocks and removes overlapping information intervals. Useful for strategy backtests; it can train on blocks later than a test block and is not a chronological deployment simulation.',
    'features.MatchResultGlicko': 'Evolving team strength from wins, draws and losses. Each feature uses the rating before that match; both teams update from their pre-match states.',
    'features.StatGlicko': 'Evolving strength for a selected statistic. For corner kicks, more corners counts as a win, equal corners as a draw. This rates superiority, not the size of the winning margin.',
    'features.Rating': 'Reuse a named rating stream configured below. Select strength, uncertainty and/or volatility without recalculating separate streams.',
    'features.BayesianRating': 'Evolving attack and defensive vulnerability from goals. Select team-state outputs; larger defensive vulnerability means worse defence. Fixed model parameters are never fitted during feature preparation.',
    'features.BayesianFixture': 'Expected goals and optional score probabilities for the actual home and away teams. These depend on both opponents and venue, and share the score filter with Bayesian team features using the same model.',
    'ratings.BayesianModel': 'Fixed Bayesian parameters with their training cutoff. Pooled models share parameters across leagues; per-league bundles retain separate parameters. Team states remain distinct in either mode.',
    'ratings.BayesianParameters': 'Positive Gamma shape/rate priors, forgetting factors and score dispersion. Defaults are starting values, not parameters trained on your data. Entry shifts mirror promotion and relegation on the log scale.',
    'ratings.Glicko2': 'Glicko-2 maintains strength (rating), uncertainty (rd) and volatility (sigma). Start with the defaults; adjust tau only to change volatility adaptation.',
}


def build():
    catalog = catalog_for_ui()
    _decorate(catalog)
    components = {s['id']: s for s in catalog.schema(refresh=True)}
    stages = stage_schema(refresh=True)
    for key, s in {**components, **stages}.items():
        if key.startswith('splits.') and hasattr(catalog.entries[key]['constructor'], 'folds'):
            s['inputs']=[n for n in inspect.signature(catalog.entries[key]['constructor'].folds).parameters if n not in ('self','dataset')]
        s['description'] = DESCRIPTIONS.get(key, s.get('description', '').split('\n\n')[0])
        doc = inspect.getdoc(catalog.entries[key]['constructor']) if key in catalog.entries else ''
        for f in s['fields']:
            name, default = f['name'], f.get('default')
            f.pop('annotation', None)
            f.pop('suggestions', None)
            f['help'] = HELP.get(name, '')
            if key.startswith(('sklearn.', 'lightgbm.', 'xgboost.')) and doc:
                match = re.search(r'(?m)^[ \t]*' + re.escape(name) + r'[ \t]*:[^\n]+\n(?:[ \t]*\n)*([ \t]+\S[^\n]*(?:\n[ \t]+\S[^\n]*)*)', doc)
                if match:
                    f['help'] = ' '.join(match[1].split())
                enum = re.search(r'(?m)^'+re.escape(name)+r'\s*:\s*\{([^}]+)\}',doc)
                if enum:
                    import ast
                    try:
                        f['choices']=list(ast.literal_eval('['+enum[1]+']'))
                    except (SyntaxError,ValueError):
                        pass
            if not f['help']:
                f['help'] = f"{f['title']} for {s.get('title', key)}." + (f" Default: {default}." if default is not None else ' Optional; leave disabled to use the standard behavior.')
            if f['kind'] == 'value':
                f['kind'] = 'text'
            if name in ('features', 'feature_columns', 'categorical_features'):
                f.update(kind='multiselect', discovery='features')
            if name in ('targets', 'target_columns', 'categorical_targets'):
                f.update(kind='multiselect', discovery='targets')
            if name == 'targets' and key == 'reporting.FeatureDistributionReporter':
                f.update(title='Labels',help='Optional observed-label histograms and KDE curves, using the same fold and rows as the feature plots. Disabled omits labels. Enable and select at least one label, such as total corners. Label plots are identified separately.')
            if name in ('target',): f.update(kind='select', discovery='targets')
            if name == 'features_from': f.update(kind='select', discovery='selectors')
            if name == 'weights_from': f.update(kind='select', discovery='weight_reporters', help='Use observation weights from a named Class Weight Reporter in Fitted preparation. Weights are recalculated on each fit’s training rows, including inner search and final refit.')
            if name in ('scope', 'group_by', 'round_keys', 'calendar_by', 'block_by'):
                f.update(kind='multiselect', choices=['team_id','opponent_id','competition_id','season_id','round','source_league','source_season'])
            if name in ('metrics',) and key != 'reporting.LearningCurveReporter': f.update(kind='metrics')
            if name == 'methods': f.update(kind='multiselect', choices=['pearson','spearman','kendall','mcc'])
            if name == 'method' and key == 'reporting.TopKCorrelationSelector': f.update(choices=['pearson','spearman','kendall','mcc'])
            if name in ('categorical_features','categorical_targets','feature_thresholds','target_thresholds'):
                f['visible_when'] = {'methods': ['mcc']}
            if name in ('feature_thresholds','target_thresholds'): f.update(kind='map', key_discovery='features' if name.startswith('feature') else 'targets', item={'kind':'number'})
            if name == 'weights': f.update(kind='map', key_discovery='metrics', item={'kind':'number','default':1,'min':0})
            if name in ('metric',) or key == 'evaluation.Metric' and name == 'name': f.update(kind='select', discovery='metrics')
            if name == 'fields' and key.startswith('features.'): f.update(kind='multiselect', choices=['rating','rd','sigma'])
            if name == 'output': f.update(choices=['predict','predict_proba'])
            if name == 'prediction_methods': f.update(kind='multiselect', choices=['predict','predict_proba'] + (['decision_function'] if key == 'training.EstimatorAdapter' else []))
            if name == 'engine': f.update(kind='component', categories=['rating'], components=['ratings.Glicko2'], initial_component='ratings.Glicko2')
            if name in ('stat',): f.update(kind='component', components=['features.Stat'], initial_component='features.Stat')
            if name in ('calibration','early_stopping','scheduler','control','validation','transition','handoff','policy'):
                ids={'calibration':['training.ProbabilityCalibrator'],'early_stopping':['training.EarlyStopping'],'scheduler':['training.ReduceOnPlateau'],'control':['training.TrainingControl'],'validation':['training.ValidationTail'],'transition':['ratings.GlickoTransition'],'handoff':['features.Hard','features.LinearFade','features.ObservationCount'],'policy':['features.SeededEMA']}
                f.update(kind='component',components=ids[name],initial_component=ids[name][0])
            if name in ('seed','random_state','max_iter','n_jobs','max_depth','max_leaf_nodes','n_estimators','max_bin','early_stopping_rounds','min_child_weight','reg_alpha','reg_lambda','learning_rate','subsample','colsample_bytree','quantile','max_points','score_rounds','step','line','threshold','odds','stake','k','fold_k'):
                f['kind']='number'
                if default is None: f['initial']= {'max_iter':1000,'n_estimators':100,'learning_rate':0.1,'n_jobs':1,'step':1,'score_rounds':1}.get(name,1)
            if f['kind']=='number': f['step'] = 1 if isinstance(default,int) or name in ('window','periods','min_periods','k','n_splits','train_size','test_size','n_blocks','n_test_blocks','max_iter','random_state') else 'any'
            if key == 'reporting.TopKCorrelationSelector' and name in ('k', 'fold_k'):
                f.update(step='any', min=0, title='Features to select' if name == 'k' else 'Nominees per fold',
                         help='Enter a whole-number count (1 means one feature), or a proportion below 1: 0.8 selects 80% of input features, rounded up. Undefined correlation scores remain ineligible.' + (' Disabled uses the main selection size.' if name == 'fold_k' else ''))
            if name == 'tolerance':
                f['min'] = 0
                if key == 'ratings.Glicko2':
                    f.update(min=5e-324, help='Positive convergence tolerance for the Glicko-2 volatility solver. Smaller values require a tighter numerical solution and may need more iterations. This is not a prediction-error tolerance; zero is invalid.')
                elif key == 'selection.ParsimonySelection':
                    f['help'] = 'Absolute metric difference allowed from the best candidate before choosing the simplest eligible model. Zero only permits candidates tied at the best metric value.'
                elif key == 'reporting.CoefficientReporter':
                    f['help'] = 'A coefficient survives when its absolute value is greater than this threshold. Zero retains every nonzero coefficient.'
                elif key == 'reporting.MatchResultReporter':
                    f['help'] = 'Maximum absolute prediction error marked within tolerance for numeric targets. Zero requires an exact match; categorical targets compare class identity.'
            if name in ('bins','bandwidth'): f['choices']=['auto','fd','sturges','sqrt'] if name=='bins' else ['scott','silverman']
            if name == 'comparison': f['choices']=['auto','numeric','categorical']
            if name == 'entity': f['choices']=['team','league']
            if name == 'device': f['choices']=['cpu','auto','cuda','cuda:0']
            if name == 'partition' and key in ('reporting.LearningCurveReporter','reporting.CoefficientReporter','reporting.FeatureImportanceReporter'): f['choices']=['model']
            if name == 'partition' and key == 'reporting.ExperimentLeaderboardReporter': f['choices']=['experiment']
            if name == 'catalog' and key in ('reporting.MatchResultReporter', 'reporting.HeatmapReporter', 'reporting.RatingReporter'): f.update(kind='reference',initial={'ref':'team_catalog'},hidden=True)
            if name == 'show_badges' and key == 'reporting.MatchResultReporter': f['default'] = True
            if name in ('cutoffs','available_at','information_start'): f.update(kind='component',components=['input.Table'],initial_component='input.Table')
            if name == 'estimator': f.update(kind='reference',initial={'ref':'estimator'})
            if name == 'strategy' and 'SimpleImputer' in key: f['choices']=['mean','median','most_frequent','constant']
            if name == 'selection' and key.startswith('sklearn.linear_model.'): f['choices']=['cyclic','random']
            if name == 'selection' and key == 'labels.BetOption': f['choices']=['yes','no','over','under','win','loss','draw']
            if name == 'source' and key == 'reporting.TopKCorrelationSelector': f.update(kind='select',discovery='correlations')
            if name == 'source' and key in ('labels.TeamValue','labels.MatchTotal','labels.Outcome','features.ForAgainst','features.StatGlicko'):
                f.update(kind='component',components=['features.Stat'],initial_component='features.Stat')
            if name == 'source' and key == 'features.LeaveOneOut':f.update(components=['features.League'],initial_component='features.League')
            if name == 'source' and key == 'features.ForAgainst': f['components'] = ['features.Stat', 'features.MatchScore', 'features.Heatmap', 'features.SpatialPointSummary']
            if key == 'features.MatchScore':
                f['primary'] = True
                if name == 'score_field': f.update(kind='select', choices=['current'], title='Score basis', help='Native current score from the matches table. It is not universally regulation time. No reconstruction from halves, extra time or penalties, and no fallback to another score field.')
                if name == 'side': f.update(kind='select', choices=['for','against','both'], help='For is goals scored by the focal team; against is goals conceded; both creates separate scored and conceded columns. Wrap in Rolling mean or Lag to use previous matches only.')
            if name == 'venue' and key.startswith('features.'):
                f.update(visible_when_contains={'source':['features.Heatmap','features.SpatialPointSummary']}, clear_when_hidden=True)
                f.update(kind='select', choices=[{'value':'all','label':'All venues'}, {'value':'same','label':'Same venue as target fixture'}], primary=True, help='All venues uses every eligible previous match. Same venue uses previous home matches before a home fixture, or previous away matches before an away fixture. The match window is applied after this filter.')
            if key == 'features.RegionMass':
                if name == 'region': f.update(kind='select', choices=['own_half','opponent_half'], primary=True, help='Integrate the grid over the focal team’s own or opponent half. A cell crossing halfway contributes proportionally. Density is multiplied by area; mass sums proportions; count sums points.')
                if name == 'source': f.update(kind='component', components=['features.Heatmap','features.Lag','features.RollingMean','features.EMA'], initial_component='features.RollingMean')
            if key in ('features.SpatialEntropy','features.SpatialConcentration'):
                if name == 'source':
                    f.update(kind='component', components=['features.Heatmap','features.Lag','features.RollingMean','features.EMA'], initial={'component':'features.Heatmap','params':{'grid_size':6,'normalization':'mass','method':'grid','kinds':['player'],'orientation':'team'}}, help='Default source is an unsmoothed 6x6 outfield unit-mass grid. Place this reducer inside Rolling mean for per-match summaries, or outside it for a summary of the historical mixture.')
                    f.pop('initial_component', None)
                if name == 'normalized': f.update(kind='boolean', help='Divide entropy by the maximum for this grid size. Zero is one occupied cell; one is a uniform map. No pseudocount is added.')
                if name == 'base': f.update(kind='number',step='any',visible_when={'normalized':[False]},help='Logarithm base greater than 1. Default e gives natural entropy; 2 gives bits. The base cancels for normalized entropy.')
                if name == 'mass_tolerance': f.update(kind='number',min=0,max=1e-6,step='any',help='Absolute numerical tolerance around total mass 1. Only floating-point drift is normalized away; this does not smooth or repair corrupt maps.')
                if name == 'metric': f.update(kind='select', choices=['effective_cells','effective_fraction','hhi','normalized_hhi','largest_cell_share','occupied_fraction'],help='Effective cells = exp(natural entropy). HHI = sum of squared shares. Normalized HHI is 0 for uniform and 1 for a single cell. Occupied fraction is mainly diagnostic and depends on sampling volume.')
            if key in ('features.SpatialHistoricalDeviation','features.SpatialFixtureDistance'):
                if name == 'source':
                    f.update(kind='component', components=['features.Heatmap'], initial={'component':'features.Heatmap','params':{'grid_size':6,'normalization':'mass','method':'grid','kinds':['player'],'orientation':'team'}}, help='Choose any square grid size of at least 2. Default 6. Gaussian fine resolution must be a multiple of the output grid size. Fixture comparisons choose for/against automatically; keep source side For.')
                    f.pop('initial_component', None)
                if name == 'comparison': f.update(kind='select', choices=['style','home_context','away_context'])
                if name in ('window','min_periods'): f.update(kind='number', min=1, step=1, help='Prior-map history only. Missing maps still occupy eligible-match window slots. The outer historical operator, when present, has independent settings.')
                if name == 'mass_tolerance': f.update(kind='number', min=0, max=1e-6, step='any')
                if name == 'venue': f.pop('visible_when_contains', None)
            if key in ('training.SpatialPCA','training.SpatialClusters'):
                f['primary'] = True
                if name == 'blocks': f.update(kind='list', item={'kind':'list','item':{'kind':'select','discovery':'assembled_features','title':'Grid cell'}}, help='Add one ordered grid-column list per team. Prepare data to discover columns. Include every cell in row-major order, using identical resolution and team orientation. Omit only when the whole input is one map vector. The spatial_extensions example can generate these blocks from prepared metadata.')
                if name in ('n_components','n_clusters'): f.update(kind='number', min=1, step=1)
            if key == 'features.SpatialPointSummary':
                if name == 'field': f.update(kind='select', choices=[{'value':key,'label':label} for key,(label,unit) in POINT_FIELDS.items()], help='Exact geometry on 0–100 coordinates, population moments and linear 10–90% quantile widths. Principal angle is in radians modulo pi, missing for isotropic/zero spread maps; arithmetic rolling averages of angles are not circular averages. For historical axes, roll the double-angle cosine and sine fields separately with identical settings; do not normalize their means. Lateral touchline convention is unverified. Wrap in Rolling mean, Lag or EMA before prediction.')
                if name == 'kinds': f.update(kind='multiselect', choices=['player','goalkeeper'], help='Player excludes goalkeepers. Each selected exported point has equal weight; repeated coordinates are retained.')
                if name == 'min_points': f.update(kind='number', min=1, step=1, help='Minimum selected points for a usable match map. Low-count maps remain missing. This is separate from historical Min periods.')
                if name == 'side': f.update(kind='select', choices=['for','against','both'], help='For summarizes this team. Against rotates historical opponent points 180 degrees into the focal team frame. Scalars are never rotated by the target venue.')
            if key == 'features.Heatmap':
                f['primary'] = True
                if name == 'grid_size': f.update(kind='number', min=2, step=1, help='Number of cells along each coordinate axis. 10 creates 100 features per perspective. Squares are in exported 0–100 coordinates, not physical metres.')
                if name == 'method': f.update(kind='select', choices=[{'value':'grid','label':'Grid pooling'}, {'value':'gaussian','label':'Gaussian-smoothed grid'}], help='Pool points directly, or smooth a fine histogram before pooling. Both return the same shaped feature grid.')
                if name == 'normalization': f.update(kind='select', choices=['mass','density','count'], help='Mass sums to 1 per observed match. Density integrates to 1 over the 0–100 coordinate plane. Count preserves point totals; export sampling intensity can vary.')
                if name == 'sigma': f.update(kind='number', min=0.01, step='any', visible_when={'method':['gaussian']}, help='Gaussian width in fine-grid cells. The reference plot uses 2.6 with resolution 100. Reflected boundaries retain total mass.')
                if name == 'resolution': f.update(kind='number', min=2, step=1, visible_when={'method':['gaussian']}, help='Fine histogram cells per axis before smoothing. Must be a multiple of Grid size. 100 matches the reference plot.')
                if name == 'kinds': f.update(kind='multiselect', choices=['player','goalkeeper'], help='Disabled includes all point kinds. Choose Player to exclude goalkeeper points, or include both.')
                if name == 'use_weights': f.update(help='Disabled counts each exported point once. Enabled uses its supplied weight, with missing weights equal to 1.')
                if name == 'orientation': f.update(kind='select', choices=[{'value':'home','label':'Shared pitch — Home goal left'}, {'value':'team','label':'Team-relative — own goal left'}], help='Shared pitch rotates final Away grids 180 degrees. Historical opponent points are first rotated into the focal team’s frame; history is averaged before final venue alignment. Both coordinate axes reverse.')
            if key == 'reporting.HeatmapReporter':
                if name == 'maps': f.update(kind='multiselect', discovery='heatmaps', primary=True, help='Discover spatial outputs, including grids, point geometry, regional shares, entropy and concentration. Disabled includes all compatible outputs. Select a fixture to inspect its exact prepared Home/Away values.')
                if name == 'max_fixtures': f.update(kind='number', min=1, step=1, help='Optional cap on fixtures retained in this report scope. Useful for small offline exports. Disabled retains every selected fixture.')
                if name == 'cache_size': f.update(kind='number', min=1, step=1, help='Maximum recently viewed fixture/feature pairs kept by the live browser. 8 is the default; this does not cap the experiment population.')
                if name == 'type': f['choices'] = ['overall','per_fold']
                if name == 'show_badges': f['help'] = 'Embed the team’s local badge beside its name. Missing badges fall back to the team name.'
            if name == 'source' and key == 'features.League':f.update(components=['features.Stat','features.ForAgainst'],initial_component='features.Stat')
            if name == 'source' and key == 'features.WarmStart':f.update(components=['features.RollingMean','features.RollingStd','features.RollingSkewness','features.RollingZScore','features.MatchResultGlicko','features.StatGlicko'],initial_component='features.RollingMean')
            if key == 'features.RollingSkewness' and name == 'min_periods':
                f.update(min=3, help='At least three finite observations are always required; increase this to require more. Constant windows remain missing. Skewness uses population moments without a small-sample correction.')
            if key == 'splits.TemporalSplit' and name == 'window':
                f['help']='Expanding retains all eligible earlier training blocks; sliding retains a fixed train_size window as the test window moves forward.'
            if key == 'splits.TemporalSplit' and name == 'unit':
                f['title']='Window unit'
                f['choices']=[{'value':'rounds','label':'Rounds (per league)'},{'value':'seasons','label':'Seasons (pooled leagues)'},{'value':'league_seasons','label':'League-seasons (separate leagues)'},{'value':'kickoffs','label':'Kickoff batches'}]
                f['help']='Training, test and step use this unit. Seasons combines the same season across leagues into one fold. League-seasons creates separate folds and fitted models per league. Example: four seasons with train 2/test 1 gives two pooled folds, or two folds per league. Gap unit can be set independently.'
            if key == 'splits.TemporalSplit' and name == 'gap_unit':
                f.update(kind='select',initial='rounds')
            if name == 'train_size':f['min']=1
            if name == 'test_size':f['min']=1
            if name == 'type':f['title']='Analysis scope'
            if name == 'partition':f['title']='Rows to analyze'
            if name == 'team':f.update(kind='select',discovery='teams')
            if name == 'league':f.update(kind='select',discovery='leagues')
            if name == 'season':f.update(kind='select',discovery='seasons')
            if name == 'round':f.update(kind='select',discovery='rounds')
            if name == 'teams':f.update(kind='multiselect',discovery='teams')
            if name == 'void_statuses':f.update(kind='multiselect',choices=['canceled','postponed','abandoned','awarded'])
            if key == 'labels.BetOption':
                if name in ('line','on_equal'): f['visible_when']={'selection':['over','under']}
                if name == 'draw':f['visible_when']={'selection':['win','loss']}
            if key.startswith('xgboost.'):
                if name=='objective': f['choices']= ['reg:squarederror','reg:absoluteerror','reg:quantileerror','count:poisson','reg:tweedie'] if key.endswith('Regressor') else ['binary:logistic','multi:softprob'] if key.endswith('Classifier') else ['rank:ndcg','rank:pairwise','rank:map']
                if name=='tree_method':f['choices']=['auto','hist','approx','exact']
                if name=='booster':f['choices']=['gbtree','gblinear','dart']
    def patch(key,name,**kw):
        for f in stages[key]['fields']:
            if f['name']==name:f.update(kw)
    patch('data','seasons',kind='multiselect',discovery='seasons')
    patch('data','leagues',kind='multiselect',discovery='leagues',nullable=False,all_when_null=True,help='All available leagues are selected by default. Untick any league to exclude it.')
    patch('data','tables',kind='multiselect',choices=['matches','statistics','pregame','shots','heatmap_points','team_seasons'])
    patch('history','stat_fields',kind='multiselect',choices=['value','total'])
    patch('stat_selection','bundles',kind='multiselect',discovery='bundles')
    patch('stat_selection','stats',kind='list',item={'kind':'component','components':['features.Stat'],'initial_component':'features.Stat'})
    patch('candidate','name',required=False,default=None)
    for name in ('observer','complexity','fit_statistics'):
        patch('candidate',name,kind='callable',help={
            'observer':'Optional registered callback receiving training progress. Available callbacks come from your extension catalog.',
            'complexity':'Optional registered callback measuring fitted model complexity for parsimony selection.',
            'fit_statistics':'Optional registered callback supplying likelihood and parameter counts for AIC/BIC. Leave disabled to use adapter-provided statistics.',
        }[name])
    patch('refit','train_positions',choices=['training','all'])
    patch('search_options','decision',kind='component',categories=['selection'],components=['selection.MetricSelection','selection.WeightedSelection','selection.ParsimonySelection'],initial_component='selection.MetricSelection')
    patch('search_options','on_error',choices=['raise','record'])
    patch('prediction','statuses',kind='multiselect',choices=['notstarted','postponed'])
    patch('prediction','rounds',kind='list',item={'kind':'number','default':1})
    patch('prediction','rounds',kind='multiselect',discovery='rounds')
    for c in components.values():
        primary = {'source','side','window','min_periods','span','periods','type','partition','methods','metrics','features','targets','k','method','train_size','test_size','unit','gap','gap_unit','window','n_splits','shuffle','group_by','n_blocks','n_test_blocks','engine','initial_rating','initial_rd','initial_sigma','tau','n_jobs','device','inner_threads','gpu_jobs','alpha','l1_ratio','max_iter','max_depth','learning_rate','n_estimators','stat','higher_is_better','fields','scope','comparison','tolerance','show_badges','page_size','prediction_methods','strategy','control','validation','early_stopping','scheduler','patience','max_steps','rounds','strength','handoff','league_weight','rank_by','top','bottom','shrinkage'}
        for f in c['fields']:
            name=f['name']
            if name == 'fit_intercept' and c['id'] in ('sklearn.linear_model.Lasso', 'sklearn.linear_model.ElasticNet'):
                f['help'] = 'Usually keep enabled: learns the baseline target level (for example, average corners). Disabling with centered inputs and uncentered labels forces the prediction baseline to zero. Disable only deliberately with appropriately centered targets.'
            if name in ('push_value','void_value','min_frequency','max_categories','max_samples','base_score','colsample_bylevel','colsample_bynode','gamma','max_cat_threshold','max_cat_to_onehot','max_delta_step','max_leaves','num_parallel_tree','scale_pos_weight','verbosity'):
                f.update(kind='number',initial=1)
            if name=='validate_parameters':f.update(kind='boolean',initial=True)
            if name in ('monotonic_cst','monotone_constraints','feature_weights'):
                f.update(kind='list',item={'kind':'number','default':0},initial=[])
            if name=='feature_types':f.update(kind='list',item={'choices':['q','c'],'default':'q'},initial=[])
            if name=='grow_policy':f['choices']=['depthwise','lossguide']
            if name=='sampling_method':f['choices']=['uniform','gradient_based']
            if name=='multi_strategy':f['choices']=['one_output_per_tree','multi_output_tree']
            if name=='importance_type' and c['id'].startswith('xgboost.'):f['choices']=['gain','weight','cover','total_gain','total_cover']
            if name=='eval_metric' and c['id'].startswith('xgboost.'):f['choices']=['rmse','mae','rmsle','logloss','error','auc','aucpr','mlogloss','merror','ndcg','map','quantile']
            if name=='class_weight':f['choices']=['balanced']
            if name=='drop' and c['id'].endswith('OneHotEncoder'):f['choices']=['first','if_binary']
            if name=='league_aggregation':f.update(choices=['mean','median'],visible_when={'entity':['league']},help='Aggregate different feature values at the same league/kickoff. Disabled requires the values to agree.')
            if name in ('teams','team_column') and c['id']=='reporting.FeatureTimeline':f['visible_when']={'entity':['team']}
            if name=='team_column':f['choices']=['team_id','home_id','away_id']
            if name in ('league_column','season_column'):f.update(kind='select',choices=['source_league','source_season','competition_id','season_id'])
            if name=='attempts':f.update(choices=['selected'],help='Disabled shows all attempts; selected shows the retained optimization attempt.')
            if name=='decision' and c['id']=='reporting.MatchResultReporter':f['choices']=['argmax']
            if name=='positive_class':f.update(kind='select',discovery='classes')
            if name=='classes':f.update(kind='multiselect',discovery='classes')
            if name=='callbacks':f.update(kind='list',item={'kind':'callable'},initial=[])
            if name in ('partial_fit_kwargs','transformer_weights'):f.update(kind='map',initial={})
            if name=='history' and c['id']=='reporting.BetPerformanceReporter':f.update(hidden=True,initial={'ref':'history'})
            if name=='labels' and c['id']=='reporting.BetPerformanceReporter':f['hidden']=True
            if name=='bets':f.update(kind='map',item={'kind':'component','components':['evaluation.BetSpec'],'initial_component':'evaluation.BetSpec'})
            if name=='take' and c['id']=='evaluation.BetSpec':f.update(kind='boolean',initial=True,help='Place this fixed-stake option for every evaluated match when enabled; disabled places none. For prediction-dependent policies use an identity-aligned decision table in an imported recipe.')
            if name=='option' and c['id']=='evaluation.BetSpec':f.update(components=['labels.BetOption'],initial_component='labels.BetOption')
            if name=='alpha' and c['id']=='features.SeededEMA':f['help']='Weight assigned to each new observation when updating prior-seeded moments. 0.5 gives the new value half of the update weight.'
            if name=='league_weight':f['help']='Blend a moving team’s previous mean toward its destination league prior. 0 keeps its old mean; 1 uses the league prior.'
            if c['id']=='features.TeamMovement' and name=='movement':
                f.update(kind='select',choices=['promoted','relegated'],help='Known season-entry flag for this team: 1 yes, 0 no, missing when unknown. Administrative entry can be 0/0 without being retained.')
            if c['id']=='features.SeededEMA':
                if name=='mode':f.update(kind='select',choices=[dict(value='legacy',label='Legacy (unchanged)'),dict(value='uniform',label='Uniform — own rolling state'),dict(value='w_league_prior',label='With league prior — mover cohort')],primary=True,help='Legacy preserves old recipes. Uniform seeds from the team’s final eligible pre-break rolling window. With league prior replaces a mover’s seed with an equal-team destination cohort.')
                if name=='league_weight':f.update(visible_when={'mode':['legacy']},help='Legacy only. In explicit modes this field is inactive: uniform never mixes cohorts, while league-prior mode fully replaces the mover seed.')
                if name in ('top','bottom'):f.update(visible_when={'mode':['w_league_prior']},step=1,help=('Relegated teams use the top' if name=='top' else 'Promoted teams use the bottom')+' destination-league cohort, ranked by eligible previous-season pregame standings. Positive integer or -1 for all eligible donors; 0 is invalid.')
                if name=='variance_prior':f.update(kind='select',choices=['within_team'],visible_when={'mode':['uniform','w_league_prior']},help='Average donor within-team variances, never standard deviations. No between-team mean-difference term.')
                if name=='variance_estimator':f.update(kind='select',choices=['population','weighted_sample'],visible_when={'mode':['uniform','w_league_prior']},help='For SD/Z-score: population requires ddof=0; weighted sample requires ddof=1. Corrected cohort seeds also need explicit prior strength. Skewness always uses population moments.')
                if name=='prior_strength':f.update(kind='number',step='any',visible_when={'mode':['w_league_prior'],'variance_estimator':['weighted_sample']},help='SD/Z-score only: effective strength > 1 for the abstract cohort dispersion prior. Not donor count or number of team matches; no production value is selected.')
                if name=='variance_fade':f.update(kind='select',choices=['estimate_interpolation'],visible_when={'mode':['uniform','w_league_prior']},help='For SD/Z-score, blend compatible variance estimates, then take the square root. Skewness always blends the first three population moments, including between-means terms.')
            if name=='strength':f['help']='Prior effective observation count. EMA weight is strength / (strength + new observations).'
            if name=='start' and c['id']=='features.LinearFade':f['help']='Completed rounds before the gradual fade begins.'
            if name=='rounds' and c['id'] in ('features.Hard','features.LinearFade'):f['help']='Number of completed current-season league rounds before switching (Hard), or duration of the fade after Start (LinearFade). Postponed rounds remain incomplete.'
            f['primary'] = f['name'] in primary or name == 'venue' or c['id'] in ('features.Heatmap','features.SpatialPointSummary','features.RegionMass','features.SpatialEntropy','features.SpatialConcentration','features.MatchScore','features.TeamMovement','features.SeededEMA') or (c['id'] == 'reporting.HeatmapReporter' and name == 'maps')
            if f['name']=='type' and c['category'] in ('pre_reporter','post_reporter'):
                f['choices']=[v for v in ['overall','per_fold','timeline'] if v in f.get('choices', ['overall','per_fold'])]
            if f['name']=='type' and c['id']=='reporting.FeatureTimeline':f['choices']=['overall','per_fold','timeline']
            if f['name']=='partition' and c['category']=='post_reporter' and c['id'] not in ('reporting.LearningCurveReporter','reporting.CoefficientReporter','reporting.FeatureImportanceReporter','reporting.ExperimentLeaderboardReporter'):f['choices']=['score','test']
            if c['id']=='reporting.RatingReporter':
                if name=='ratings':f.update(hidden=True,help='Automatically discovers retained feature ratings and fitted Bayesian rating models. Choose the rating source in the rendered report.')
                if name=='interval':f.update(title='Glicko band coverage',kind='number',min=0.001,max=0.999,step=0.01,primary=True,help='Glicko rating uncertainty coverage. Default 0.95 gives rating plus/minus 1.96 rating deviations.')
                if name=='bayesian_interval':f.update(title='Bayesian band coverage',kind='number',min=0.001,max=0.999,step=0.01,primary=True,help='Gamma posterior coverage for attack, defense and home advantage. Default 0.80 shows the 10th to 90th percentiles. This changes visualization only, not the fitted model.')
                if name=='pooling':f.update(kind='select',choices=['occurrences','first','last'],help='Keep fold-specific evaluation occurrences, or retain the first/last occurrence. Rating states are never averaged across fitted folds.')
            if f['name']=='score_rounds':f.update(kind='range',initial=[None,None])
            if f['name']=='name' and c['id']=='features.Rating':f.update(kind='select',discovery='ratings',help='Choose one of the named rating streams configured below.')
        if c['id']=='features.Stat':
            for f in c['fields']:
                if f['name']=='group':f['hidden']=True
        if c['id'].startswith('xgboost.') and c['id'].endswith('Regressor'):
            c['fields'].append(dict(name='quantile_alpha',title='Quantiles',kind='list',required=False,default=[0.1,0.5,0.9],item={'kind':'number','default':0.5,'min':0,'max':1},visible_when={'objective':['reg:quantileerror']},help='Quantiles to predict. 0.5 is the median; 0.1 and 0.9 form an 80% prediction interval.'))
    for key,s in stages.items():
        for f in s['fields']:
            f['primary']=f['name'] in {'seasons','leagues','tables','include_awarded','stat_fields','bundles','stats','layout','drop_missing_targets','name','features_from','control','validation','stat','engine','scope','higher_is_better','transition','metrics','decision','reuse','statuses','rounds','as_of'}
    # Post-assembly operands and configurable notebook feature bank.
    for key, c in components.items():
        for f in c['fields']:
            name = f['name']
            if key.startswith('prepared.'):
                f['primary'] = True
                if name == 'name':
                    f.update(kind='select', discovery='assembled_features', help='Choose an already assembled predictor column. Click Discover assembled columns first; this works even while a derived expression is unfinished. Labels are not available as operands.')
                elif name in ('source','left','right','numerator','denominator'):
                    f.update(kind='component', categories=['derived_feature'], initial_component='prepared.Column', help='Choose an assembled column, constant or nested arithmetic expression.')
            if key == 'features.RollingWeightedMean':
                if name in ('source', 'weights'):
                    f.update(kind='component', categories=['feature'], initial_component='features.Stat', primary=True,
                             help='Select exactly one numeric output. ' + ('This value is averaged.' if name == 'source' else 'Finite negative weights raise; zero weights occupy slots but do not count toward minimum periods.'))
                    for obsolete in ('components', 'key_discovery', 'item'):
                        f.pop(obsolete, None)
                elif name == 'min_periods':
                    f.update(help='Minimum paired finite observations with strictly positive weights, unlike ordinary Rolling mean which counts finite values.')
                elif name == 'venue':
                    f.pop('visible_when_contains', None)
                    f['visible_when_any_contains'] = {operand:['features.Heatmap','features.SpatialPointSummary'] for operand in ('source','weights')}
                elif name in ('source_available_at', 'weights_available_at'):
                    f.update(kind='text', help='Optional UTC datetime column in team history for this input. Missing or late release masks the input within the fixed window. Disabled uses shared fixture availability. The column must already exist in the supplied history.')
            if key == 'context.CalendarFeature' and name == 'kind':
                f.update(kind='select', choices=['month_sin','month_cos','weekday','round'], primary=True,
                         help='UTC fixture context. Month uses sin/cos(2π × month / 12); weekday is Monday=0; round is numeric.')
            if key == 'features.SeasonProgress' and name == 'total_rounds':
                c['title'] = 'Season progress'
                f.update(kind='number', min=1, step=1, primary=True, title='Total rounds override', visible_when={'mode':['rounds']}, clear_when_hidden=True,
                         help='Disabled infers the highest assigned round separately for each league-season from the full loaded schedule, including upcoming matches. Enable for a partial schedule; the override applies to all selected seasons. A postponed round-5 fixture remains 5 / total rounds even when played during round 12. Stage round numbers must not restart.')
            if key == 'features.SeasonProgress' and name == 'mode':
                f.update(kind='select', choices=['rounds','kickoff'], primary=True,
                         help='Rounds uses the originally assigned round. Kickoff uses UTC calendar days from the first through the last scheduled kickoff, counting the opening day as 1. Postponed games use their actual kickoff date in kickoff mode.')
            if key == 'features.SeasonProgress' and name == 'total_days':
                f.update(kind='number', min=1, step=1, primary=True, visible_when={'mode':['kickoff']}, clear_when_hidden=True,
                         help='Optional inclusive season length in calendar days. Disabled infers it from the first and last kickoff in each full league-season schedule. Applies to every selected season; partial schedules may also need a start date override.')
            if key == 'features.SeasonProgress' and name == 'start_date':
                f.update(kind='text', primary=True, visible_when={'mode':['kickoff']}, clear_when_hidden=True,
                         help='Optional season opening date, e.g. 2026-08-01, interpreted in UTC. Disabled uses the earliest kickoff per league-season. This override applies to every selected season.')
            if key == 'preparation.NumericFeatures' and name == 'dtype':
                f.update(kind='select', choices=['float32','float64'], primary=True, help='Numeric precision of model inputs. No scaling or imputation is applied.')
            if key == 'preparation.IdentityFeatureSpec':
                f['primary'] = True
                if name == 'columns': f.update(kind='multiselect', choices=['source_league','home_id','away_id','team_id','opponent_id'], help='Metadata identities to encode. Use source_league for notebook parity; team IDs are an optional extension.')
                if name == 'prefixes': f.update(kind='map', key_discovery='identity_columns', item={'kind':'text'}, help='Optional output prefixes, e.g. source_league → league. Categories are fitted only from common training rows.')
            if key == 'preparation.FeatureBankPreset':
                f['primary'] = name in ('stats','periods','windows','lags','spans','include_loo','include_h2h','include_ratings','warm_policy','rating_warm_policy','include_combinations','include_rest','include_calendar')
                if name == 'stats': f.update(kind='stat_pairs', help='Statistic shortlist. The bank only includes identities observed in the configured development population. Periods are selected separately.')
                elif name in ('half_keys','warm_stat_keys'): f.update(kind='multiselect', discovery='stat_keys', help='Select statistic keys for half-period or warmed copies. Disabled warm keys includes all eligible rolling statistics.')
                elif name == 'periods': f.update(kind='multiselect', choices=['ALL','1ST','2ND'], help='Full match and/or first and second half. Half-period features use their separate operator windows.')
                elif name.endswith('windows') or name.endswith('lags') or name in ('lags','spans'):
                    f.update(kind='list', item={'kind':'number','default':5,'min':1,'step':1}, help='One positive integer per window, lag or EMA span. Each value produces a separate feature.')
                elif name.endswith('reducers'): f.update(kind='multiselect', choices=['mean','std','skew','z'], help='Population or H2H calculations to include; standard deviation and Z-score use ddof=1; skew uses population moments and at least three valid observations.')
                elif name in ('warm_policy','rating_warm_policy'):
                    component = 'features.SeededEMA' if name == 'warm_policy' else 'ratings.GlickoTransition'
                    f.update(kind='component', components=[component], initial_component=component, help='Add warmed copies while retaining every ordinary feature. Disabled omits these additional columns.')
                elif name.startswith('include_'): f.update(help='Include '+name.removeprefix('include_').replace('_',' ')+' in the generated bank. Existing named features remain additional.')
    # Shared, maintained scaler controls for predictor and target transforms.
    for f in stages['candidate']['fields']:
        if f['name'] == 'weights_from':
            f['primary'] = True
    for key in ('reporting.CountClassificationReporter', 'reporting.FeatureImportanceReporter', 'reporting.PredictionTimelineReporter'):
        for f in components[key]['fields']:
            f['primary'] = True
            if f['name'] == 'importance_type':
                f.update(kind='select', choices=['gain','weight','cover','total_gain','total_cover'], help='XGBoost native importance. Gain measures average split improvement; weight counts splits; cover measures observations affected.')
            elif f['name'] == 'top_k':
                f.update(kind='number', min=1, step=1, help='Maximum features shown in the importance plot per fitted model. The retained importance table stays complete.')
            elif f['name'] == 'max_points':
                f.update(kind='number', min=1, step=1, help='Maximum observations plotted per series. The full timeline table is retained for download.')
            elif f['name'] == 'epsilon':
                f.update(kind='number', min=1e-300, max=0.999999, step='any', help='Floor used only for finite clipped log loss. Exact zero probability is still reported separately; this does not calibrate probabilities.')
            elif f['name'] == 'tolerance':
                f.update(kind='number', min=0, step='any', help='Inclusive error band in count units. For corners, 2 counts a prediction within two corners as within tolerance.')
            elif f['name'] == 'probability_output':
                f.update(kind='select', choices=['predict_proba','predict_proba_raw'], help='Use upstream probabilities; calibrated output uses predict_proba when calibration was enabled.')
    for f in components['experiments.ArtifactExport']['fields']:
        f['primary'] = True
        f['help'] = {
            'report_tables': 'Export every retained study table to CSV with its name, fold and partition in a manifest.',
            'predictions': 'Export observed labels, match metadata and all prediction outputs for each evaluated fold.',
            'save_models': 'Save a separately loadable prediction model for each evaluated fold, including fitted preprocessing.',
            'verify_reload': 'Reload each exported model and check its test predictions against the retained values. This predicts again without fitting.'
        }[f['name']]
    for f in components['training.BoostingAdapter']['fields']:
        if f['name'] == 'exact_counts':
            f.update(primary=True, help='For exact-count classifiers: require finite nonnegative integer labels. XGBoost uses multiclass probabilities and discovers its classes from fitting rows only.')
    for f in components['reporting.ClassWeightReporter']['fields']:
        name = f['name']
        f['primary'] = True
        if name == 'mode':
            f.update(kind='select', choices=[{'value':'none','label':'No balancing'}, {'value':'balanced','label':'Balanced'}, {'value':'power','label':'Adjustable balancing'}, {'value':'custom','label':'Custom class weights'}, {'value':'callable','label':'Registered calculation'}],
                     help='Balanced gives each observed fitting class equal total weight. Adjustable balancing controls how strongly rare classes are emphasized. No balancing uses one for every row.')
        elif name == 'power':
            f.update(kind='number', min=0, step='any', visible_when={'mode':['power']}, help='0 means no balancing; 0.5 partially balances; 1 fully balances. Larger values put still more weight on rare classes. Weights are normalized to mean one.')
        elif name == 'class_weights':
            f.update(kind='map', key_discovery='weight_classes', item={'kind':'number','default':1,'min':0,'step':'any'}, initial={}, visible_when={'mode':['custom']},
                     help='Discover feature columns first to list label classes, then assign a nonnegative weight to every fitting class. Keys are original labels, such as 5 corners, not encoded class positions. Extra classes absent from a fold are ignored; missing weights raise an error.')
        elif name == 'calculator':
            f.update(kind='callable', visible_when={'mode':['callable']}, help='Choose a registered callback that receives fitting X, y and metadata and returns a weight Series indexed by those exact rows. Validation and evaluation rows are excluded.')
        elif name == 'type':
            f.update(choices=['per_fold','overall'], help='Per fold computes a separate set of weights for each fitting population. Overall summarizes pooled rows; it can supply training weights only when those rows exactly match a single fit.')
        elif name == 'target':
            f['help'] = 'Original class label to balance. Leave disabled when the dataset has one target. Training with these weights requires that same single target.'
    for f in stages['candidate']['fields']:
        if f['name'] == 'calibration':
            f['primary'] = True
    for f in components['training.ProbabilityCalibrator']['fields']:
        f['primary'] = True
        if f['name'] == 'method':
            f.update(kind='select', choices=['temperature','sigmoid','isotonic'],
                     choices_by={'response_method': {'decision_function':['sigmoid'], 'predict_proba':['temperature','sigmoid','isotonic']}},
                     help='Temperature adjusts overall confidence with one parameter per target. Sigmoid fits a logistic mapping per class; isotonic fits a more flexible monotone mapping. All return a normalized class distribution.')
        elif f['name'] == 'fraction':
            f.update(kind='number', min=0.001, max=0.999, step='any',
                     help='Fraction of latest training kickoff batches reserved only for calibration. 0.2 means 20%; whole matches and equal kickoff times stay together. Early stopping is reserved from the remaining rows.')
        elif f['name'] == 'time_column':
            f.update(kind='select', choices=['kickoff_at'], help='Timestamp used to choose the chronological calibration tail.')
        elif f['name'] == 'update_predict':
            f.update(help='Use calibrated probability argmax for point predictions. Disable to preserve base predictions. Probability mode retains raw probabilities; margin mode retains decision_function instead.')
        elif f['name'] == 'response_method':
            f.update(kind='select', choices=['predict_proba','decision_function'],
                     help='Probability input preserves existing behavior. Decision function fits one sigmoid on binary SVC margins from a later, availability-checked training tail; requires SVC internal probabilities disabled and Method sigmoid.')
        elif f['name'] in ('availability_column','availability_delay','cutoff_column','prediction_lead','issue_at'):
            f.update(kind='text', visible_when={'response_method':['decision_function']}, help={
                'availability_column':'Metadata column with label availability timestamps. Missing availability purges the whole prediction group. Mutually exclusive with delay.',
                'availability_delay':'Explicit outcome-availability assumption, e.g. 3h after kickoff. A proxy, not a verified release time. Set this or an availability column.',
                'cutoff_column':'Optional metadata column for prediction cutoffs. Otherwise use first kickoff of each declared group minus Prediction lead.',
                'prediction_lead':'Duration before the first kickoff of the prediction group, e.g. 1h. Default 1h. Used when no explicit cutoff column is selected.',
                'issue_at':'Explicit UTC model issue time, required for final refit without an outer fold. An outer fold always uses the earlier of this time and its first prediction cutoff.'}[f['name']])
        elif f['name'] == 'prediction_group_by':
            f.update(kind='list', item={'kind':'text','default':'competition_id'}, visible_when={'response_method':['decision_function']},
                     help='Metadata keys defining indivisible prediction groups, e.g. competition_id, season_id, tournament_id, round. Use the actual columns of your dataset. Empty groups by match; equal-kickoff batches remain intact at tail selection. Boundary-crossing/incomplete groups are purged.')
        elif f['name'] in ('min_calibration_rows','min_calibration_per_class'):
            f.update(kind='number', min=1, step=1, visible_when={'response_method':['decision_function']},
                     help='Required calibration observations after purging; counts dataset rows. Both classes must meet the per-class minimum. No automatic tail expansion or fallback.')
    for c in components.values():
        if c['category'] not in ('preprocessor', 'target_transformer'):
            continue
        family = c['id'].split('.')[-1]
        for f in c['fields']:
            name = f['name']
            if family == 'QuantileTransformer':
                if name == 'n_quantiles':
                    f.update(kind='number', min=1, step=1, primary=True,
                             help='Number of empirical quantile landmarks. Uses at most the number of fitting rows; 1000 is the usual starting point.')
                if name == 'output_distribution':
                    f.update(kind='select', choices=['uniform', 'normal'], primary=True,
                             help='Uniform maps ranks to 0–1; normal maps them to a bell-shaped distribution. This transforms values; it does not enable quantile regression.')
                if name == 'subsample':
                    f.update(kind='rule_number', rules=[dict(value=None, label='All fitting rows')],
                             number_default=10000, min=1, step=1, nullable=False, all_when_null=True,
                             help='Maximum fitting samples used to estimate quantiles. None uses every fitting row. If set, keep it at least as large as Quantile landmarks.')
                if name == 'random_state':
                    f.update(kind='number', initial=0,
                             help='Seed for reproducible subsampling. Set a number such as 0 to repeat the same fit sampling.')
            if family == 'PowerTransformer':
                if name == 'method':
                    f.update(kind='select', choices=['yeo-johnson', 'box-cox'], primary=True,
                             help='Yeo-Johnson accepts zero and negative values. Box-Cox requires strictly positive values, so it cannot directly transform zero-corner labels.')
                if name == 'standardize':
                    f.update(primary=True, help='After the power transform, also center to mean zero and scale to unit variance using fitting rows.')
            if family in ('MinMaxScaler', 'RobustScaler') and name in ('feature_range', 'quantile_range'):
                f.update(kind='record', primary=True, fields=[
                    dict(name='0', title='Lower bound' if name=='feature_range' else 'Lower percentile', kind='number', required=True, default=f['default'][0]),
                    dict(name='1', title='Upper bound' if name=='feature_range' else 'Upper percentile', kind='number', required=True, default=f['default'][1])],
                    help='Output interval, for example 0 to 1.' if name=='feature_range' else 'Percentiles defining the robust scale; 25 and 75 use the interquartile range.')
            if family in ('StandardScaler', 'RobustScaler') and name in ('with_mean', 'with_std', 'with_centering', 'with_scaling', 'unit_variance'):
                f['primary'] = True
                f['help'] = {
                    'with_mean': 'Subtract the mean learned from fitting rows. Disable to preserve the original zero point.',
                    'with_std': 'Divide by the standard deviation learned from fitting rows.',
                    'with_centering': 'Subtract the median learned from fitting rows.',
                    'with_scaling': 'Divide by the spread between the selected fitting percentiles.',
                    'unit_variance': 'Adjust the percentile-based scale so a normal distribution would have variance one.',
                }[name]
            if family in ('StandardScaler', 'MinMaxScaler', 'MaxAbsScaler', 'RobustScaler', 'QuantileTransformer', 'PowerTransformer') and name == 'copy':
                f['help'] = 'Keep input arrays unchanged when possible. Disable to allow in-place work; the target adapter still isolates the original labels.'
            if family == 'QuantileTransformer' and name == 'ignore_implicit_zeros':
                f['help'] = 'For sparse input matrices only: exclude unstored zeros when estimating quantiles. Ordinary dense labels are unaffected.'
            if family == 'MinMaxScaler' and name == 'clip':
                f['help'] = 'Clip transformed validation or future observations to the chosen interval. This loses information outside the fitting range; inverse transformation cannot undo clipping.'
    patch('stat_selection','stats',kind='stat_triples')
    patch('prediction','as_of',kind='text',help='UTC timestamp such as 2026-09-19T12:00:00Z. Disabled applies no clock filter.')
    # Explicit auxiliary data controls instead of an untyped object textbox.
    for key in ('feature_options','rating_options'):
        for f in stages[key]['fields']:
            if f['name'] in ('team_counts','team_seasons','season_starts','transition_context'):
                f.update(kind='component',components=['input.Table'],initial_component='input.Table')
            if f['name']=='ratings':
                f.update(kind='map', title='Saved rating runs', primary=True,
                         item={'kind':'component','components':['input.BayesianRatingRun','input.RatingRun'],
                               'initial_component':'input.BayesianRatingRun'},
                         help='Name and load an existing rating artifact. Bayesian Rating Run restores its full checkpoint and fixture predictions; Rating Run loads generic numeric snapshots. Saved names must differ from generated rating names.')
    patch('rating_options','transition_context',kind='component',components=['features.TransitionContext'],initial_component='features.TransitionContext')
    for c in components.values():
        for f in c['fields']:
            name=f['name']
            if name in ('bins','bandwidth'):
                f.update(kind='rule_number',rules=f.pop('choices',[]),number_default=20 if name=='bins' else 1,min=1 if name=='bins' else 0.001)
            if name=='attempts':f.update(kind='attempts');f.pop('choices',None)
            if c['id']=='reporting.CalibrationReporter' and name=='strategy':f['choices']=['uniform','quantile']
            if c['id']=='reporting.PredictionDistributionReporter' and name=='mode':f['choices']=['kde','ecdf','frequency']
            if c['id']=='features.TransitionContext' and name=='history':f.update(kind='reference',initial={'ref':'history'})
            if c['id']=='training.IterativeAdapter' and name=='backend_factory':f.update(kind='factory',initial={'factory':{'component':'training.PartialFitBackend','params':{'estimator':{'ref':'native_estimator'},'preprocessor':{'ref':'preprocessor'}}}},components=['training.PartialFitBackend'],help='Create a fresh incremental backend for each training attempt. The estimator and fitted preprocessing come from Model & training. Choose an estimator supporting partial_fit, such as SGD.')
            if c['id']=='training.PartialFitBackend' and name=='estimator':f.update(kind='reference',initial={'ref':'native_estimator'})
            if c['id']=='training.PartialFitBackend' and name=='loss':f.update(kind='select',discovery='metrics')
            if name=='weights':f['key_discovery']='ranking_metrics'
            if name=='max_features' and c['id'].startswith('sklearn.ensemble.RandomForest'):
                f.pop('choices',None)
                f.update(kind='rule_number',rules=['sqrt','log2'],number_default=1,help='Square root/log2 selects that many predictor candidates per split. A fraction such as 0.7 uses 70% of features; an integer sets a count.')
            if name=='objective' and c['id'].startswith('lightgbm.'):
                f['choices']=['regression','regression_l1','huber','fair','poisson','quantile','mape','gamma','tweedie'] if c['id'].endswith('Regressor') else ['binary','multiclass','multiclassova'] if c['id'].endswith('Classifier') else ['lambdarank','rank_xendcg']
            if c['id']=='features.TransitionContext' and name in ('team_seasons','season_starts','population'):f.update(kind='component',components=['input.Table'],initial_component='input.Table')
            if c['id']=='features.TransitionContext' and name=='population':f['hidden']=True
            if c['id']=='input.Table' and name=='index':f.update(kind='multiselect',choices=['source_league','source_season','competition_id','season_id','event_id','team_id','fold_id','row_position'],help='Exact observation identities: source_league, source_season, competition_id, season_id, event_id; add team_id for team rows.')
            if name in ('steps','transformers') and c['id'] in ('sklearn.pipeline.Pipeline','sklearn.compose.ColumnTransformer'):
                fields=[dict(name='0',title='Step name',kind='text',required=True,default='step',help='Unique name for this preprocessing step.'),dict(name='1',title='Transformation',kind='component',required=True,categories=['preprocessor','model'] if name=='steps' else ['preprocessor'],initial_component='sklearn.preprocessing.StandardScaler',help='Choose the fitted operation to apply.')]
                if name=='transformers':fields.append(dict(name='2',title='Input columns',kind='multiselect',discovery='features',required=True,default=[],help='Prepared predictor columns passed to this operation.'))
                f.update(kind='list',initial=[],item={'kind':'record','fields':fields,'initial':['step',{'component':'sklearn.preprocessing.StandardScaler','params':{}},*([[]] if name=='transformers' else [])]})
        if c['id'].startswith('lightgbm.'):
            c['fields'].append(dict(name='max_bin',title='Maximum bins',kind='number',required=False,default=255,min=2,step=1,help='Maximum histogram bins per feature. Larger values allow finer splits at higher memory cost.'))
            if c['id'].endswith('Regressor'):
                c['fields'].append(dict(name='alpha',title='Quantile / Huber alpha',kind='number',required=False,default=0.9,visible_when={'objective':['quantile','huber']},help='Quantile level (0.5 is the median), or the Huber-loss alpha when that objective is selected.'))
    for key in ('features.SpatialHistoricalDeviation', 'features.SpatialFixtureDistance', 'training.SpatialPCA', 'training.SpatialClusters'):
        for f in components[key]['fields']:
            f['primary'] = True
    for key in ('features.Abs', 'features.Sum', 'features.Product', 'features.Difference', 'features.Ratio', 'features.Constant'):
        for f in components[key]['fields']:
            f['primary'] = f['required']
            if f['name'] == 'zero_value':
                f.update(kind='number', step='any')
            if key == 'features.Constant':
                f['help'] = 'Finite numeric value repeated for every row. Example: 1 as an explicit offset in a denominator.'
    count_model = components.get('training.NegativeBinomialRegressor')
    if count_model:
        for f in count_model['fields']:
            name = f['name']
            f['help'] = {
                'dispersion': 'NB2 variance = mean + dispersion × mean². Fixed value when Learn dispersion is off; starting value when on. For mean 10, dispersion 0.2 gives variance 30. This is separate from regularization strength.',
                'learn_dispersion': 'Estimate one dispersion per fitted model from its training rows, jointly with the mean coefficients. Disabled keeps Dispersion fixed. Betting probabilities and mode predictions use the fitted value. Learned values are numerically bounded to 0.000001–1000000; a boundary flag is retained.',
                'fit_intercept': 'Learn the baseline log mean. Usually enabled; disabling assumes an expected count of 1 when all transformed inputs are zero.',
                'max_iter': 'Maximum iterations used to fit coefficients. Increase if fitting reports that convergence was not reached.',
                'tol': 'Tolerance for convergence of the fitting algorithm. Smaller values require a more precise fit.',
                'penalty': 'None preserves ordinary NB regression. Ridge shrinks coefficients; Lasso can set them to zero; Elastic Net combines both. The intercept is never penalized. Standardize inputs before using a penalty.',
                'alpha': 'Penalty strength, separate from Dispersion. Zero disables shrinkage. Try 0.001, 0.01, 0.1 and 1 in grid search. Larger values shrink fitted-feature coefficients more strongly.',
                'l1_ratio': 'Elastic Net mixture: 0 is pure Ridge, 1 is pure Lasso, and 0.5 mixes both equally. Used only with Elastic Net.',
                'prediction': 'Mean returns the expected count. Mode returns the most probable integer count (upper mode for ties). With dispersion 1 or greater, the mode is always zero. This changes the prediction summary, not the fitting objective, and does not guarantee wider predictions.',
            }[name]
            f['primary'] = name in ('dispersion', 'learn_dispersion', 'fit_intercept', 'penalty', 'alpha', 'l1_ratio', 'prediction')
            if name == 'learn_dispersion':
                f.update(kind='boolean',title='Learn dispersion')
            if name == 'prediction':
                f.update(title='Predict using', kind='select', choices=[dict(value='mean', label='Mean (expected count)'), dict(value='mode', label='Mode (most probable count)')])
            if name == 'penalty':
                f.update(kind='select', choices=[dict(value='none', label='None'), dict(value='l2', label='Ridge (L2)'), dict(value='l1', label='Lasso (L1)'), dict(value='elasticnet', label='Elastic Net')])
            if name == 'alpha':
                f.update(title='Penalty strength', kind='number', min=0, step='any', visible_when={'penalty':['l1','l2','elasticnet']})
            if name == 'l1_ratio':
                f.update(title='L1 ratio', kind='number', min=0, max=1, step='any', visible_when={'penalty':['elasticnet']})
            if name in ('dispersion', 'tol'):
                f.update(kind='number', min=1e-12, step='any')
            if name == 'max_iter':
                f.update(kind='number', min=1, step=1)
    for f in stages['split_options']['fields']:
        if f['name']=='groups':f.update(kind='select',choices=['competition_id','season_id','source_league','source_season'])
    # Betting uses the same discoverable inventory as every other pipeline step.
    bet_help = {
        'offers':'Add the available lines/options for this predicted label. Give each a display name (for example Under 7.5). Quotes are optional unless your policy uses expected profit.',
        'policy':'Choose how one offer is selected for each match (or each team row). No qualifying offer produces No bet. Outcomes never influence selection.',
        'output':'Automatic uses upstream class probabilities (calibrated if enabled), otherwise the Negative Binomial count distribution. Raw probabilities are only available when calibration retained them.',
        'source':'Reuse an earlier Bet Outcome Reporter, with the same scope, rows and pooling. Its exact decisions and settlements feed profit accounting; no second selection step.',
        'default_odds':'Optional explicit fixed-odds scenario for offers without quotes, e.g. 1.1. Disabled leaves odds and profit unavailable. Decimal odds must exceed 1.',
        'min_probability':'Minimum chance of winning, from 0 to 1. Example: 0.90 requires at least 90%. Push probability is separate.',
        'max_probability_loss':'Maximum absolute probability sacrificed from the safest offered line on the same side. Example: 0.03 permits three percentage points. The reference stays fixed.',
        'min_ev':'Minimum expected net profit per stake unit: p(win) × (odds − 1) − p(loss). Requires odds. Zero accepts nonnegative expected profit.',
        'odds':'Optional decimal quote for this option, e.g. 1.85. Leave disabled to use the reporter fallback or show probability-only decisions.',
        'stake':'Fixed stake per selected bet, in your chosen currency or units. Default 1. No bankroll or dynamic staking simulation.',
        'option':'Select a BetOption using the same label source as your model target. Exact-count probabilities are summed across qualifying counts; no interpolation at half lines.',
    }
    for key in ('reporting.BetOutcomeReporter','reporting.BetPerformanceReporter','evaluation.BetOffer','evaluation.TightestLine','evaluation.HighestExpectedProfit'):
        for f in components[key]['fields']:
            name=f['name']
            if name in bet_help:f.update(help=bet_help[name],primary=True)
            if name in ('history','labels'):f.update(hidden=True)
            if name=='catalog':f.update(hidden=True,kind='reference',initial={'ref':'team_catalog'})
            if name=='offers':f.update(kind='map',item={'kind':'component','components':['evaluation.BetOffer'],'initial_component':'evaluation.BetOffer'})
            if name=='policy':f.update(kind='component',categories=['evaluation'],components=['evaluation.TightestLine','evaluation.HighestExpectedProfit','evaluation.BinaryDrawThreshold'],initial_component='evaluation.TightestLine')
            if name=='option':f.update(kind='component',categories=['label'],components=['labels.BetOption'],initial_component='labels.BetOption')
            if name=='source':f.update(kind='select',discovery='bet_outcomes',title='Prepared decisions from')
            if name=='output':f.update(kind='select',choices=['auto','predict_proba','predict_proba_raw','count_distribution'])
            if name=='bets':f.update(visible_when={'source':[None]},help='Manual decisions are optional. Prefer Prepared decisions from to reuse the probability-based outcome reporter.')
            if name in ('min_probability','max_probability_loss'):f.update(kind='number',min=0,max=1,step='any')
            if name in ('min_ev','default_odds','odds','stake'):f.update(kind='number',step='any')
            if name=='min_ev':f['initial']=0.0
            if name=='default_odds':f.update(initial=1.1,min=1.000001)
            if name=='odds':f.update(initial=1.85,min=1.000001)
            if name=='show_badges':f['default']=True
    for key in ('evaluation.BetOffer','evaluation.BetSpec'):
        for f in components[key]['fields']:
            if f['name']=='odds':
                f.update(kind='odds', primary=True, help='Choose a fixed decimal price, an identity-indexed table, or the current odds database. Imported missing quotes mean no bet; no closing-price fallback.')
    odds_fields = {
        'snapshot': dict(hidden=True,initial='data/odds'),
        'crosswalk': dict(title='Fixture mapping',kind='select',discovery='odds_crosswalks',help='Reviewed mapping from provider fixtures to native matches. Build it below after choosing seasons.'),
        'seasons': dict(kind='multiselect',discovery='seasons',help='Explicit seasons to read from the odds database; no implicit future-season inclusion.'),
        'leagues': dict(kind='multiselect',discovery='leagues',help='Optional league subset. Disabled uses all leagues in the database.'),
        'market': dict(kind='select',choices=['corners','yellow_cards','1x2','goals'],help='Market must describe the predicted target. BTTS and Asian handicap prices are stored but not enabled for native settlement.'),
        'selection': dict(kind='select',choices=['over','under'],choices_by={'market':{'corners':['over','under'],'yellow_cards':['over','under'],'goals':['over','under'],'1x2':['home','draw','away']}},help='Choose the quoted event. Home/away are absolute fixture sides; Outcome perspective is checked.'),
        'quote_type': dict(kind='select',choices=['closing','opening'],choices_by={'market':{'corners':['closing'],'yellow_cards':['closing'],'goals':['opening','closing'],'1x2':['opening','closing']}},help='Opening is earliest recorded; closing is last recorded. Actual quote times and bookmaker are unknown. This is a price scenario, not verified early-entry execution.'),
        'line': dict(kind='number',step=0.5,initial=9.5,visible_when={'market':['corners','yellow_cards','goals']},clear_when_hidden=True,help='Integer or half line. Equality on integer totals pushes. Quarter-line split settlement is unsupported.'),
        'settlement_confirmed': dict(kind='boolean',help='Confirm that the native target and void rules match vendor regulation time. For yellow cards verify second yellows/bench cards and exclude red cards. This does not confirm quote availability at prediction time.'),
        'snapshot_hash': dict(hidden=True), 'crosswalk_hash': dict(hidden=True),
    }
    components['input.OddsSeries']['title']='Database odds'
    for f in components['input.OddsSeries']['fields']:
        f.update(primary=True, **odds_fields[f['name']])
    for f in components['evaluation.BinaryDrawThreshold']['fields']:
        f.update(kind='number', min=0, max=1, step='any', primary=True, nullable=True,
                 default=None, initial=0.5, title='Maximum non-draw probability',
                 help='Enable to retain P(non-draw) <= this frozen threshold; equality stays in. Missing probabilities abstain. Disabled explicitly uses None: all quoted draws, including unknown probabilities. Model label must be the draw BetOption (1=draw, 0=non-draw). No EV cutoff or threshold search.')
    ticket_components = ['evaluation.Parlay', 'evaluation.MultiBet', 'evaluation.AllCombinations', 'evaluation.BetSlip']
    for key in ('reporting.BetOutcomeReporter', 'reporting.BetPerformanceReporter'):
        for f in components[key]['fields']:
            if f['name'] == 'composition':
                f.update(kind='component', components=ticket_components, initial_component='evaluation.Parlay',
                         title='Compose tickets', primary=True,
                         help='Parlay uses disjoint batches; MultiBet combines within batches; AllCombinations uses every k-event combination in each whole stage-round. BetSlip names independent templates. Counts appear in reports once eligible predictions/quotes exist; data preparation alone cannot determine filtered counts. Disabled keeps singles. Configure composition on only one reporter.')
    ticket_help = {
        'min_ev': 'Optional ticket EV per unit stake. Accepts strictly greater EV; equality rejects. Requires explicit Independent probability mode and win/loss probabilities. Candidate exposure is counted before filtering; only selected tickets are staked.',
        'size': 'Number of distinct fixtures per batch. Parlay makes one ticket; MultiBet makes the selected combinations from this pool. Incomplete batches are omitted. Use 1 for singles.',
        'grouping': 'Same league and round; same round across leagues (within a common season); or same UTC calendar day. Fold occurrences always stay separate.',
        'order_by': 'Rank eligible selected legs before filling each batch. Kickoff is chronological; probability and expected profit rank highest first. Real outcomes never affect the ranking.',
        'stake': 'Stake per ticket, or total per MultiBet batch when Total is selected. Ticket stakes replace the original single-leg stakes.',
        'sizes': 'System sizes: 2 and 3 with pool size 3 creates three doubles and one treble. Every size must be at most the pool size.',
        'stake_mode': 'Per ticket stakes this amount on every combination. Total divides this amount equally across all combinations in each complete batch.',
        'on_push': 'Remove: leg contributes odds 1. Refund: return the whole ticket stake unless another leg loses. Loss: treat this leg as losing.',
        'on_void': 'Remove: cancelled leg contributes odds 1. Refund: return the whole ticket stake unless another leg loses. Loss: treat this leg as losing.',
        'probability_mode': 'None leaves joint probability blank. Independent multiplies leg win probabilities as an explicit assumption; it does not estimate dependence or push-adjusted return probability.',
        'max_tickets': 'Maximum generated tickets per template in this report scope. System combinations can grow quickly; exceeding the limit gives an error, not a truncated result.',
        'tickets': 'Add named Parlay, MultiBet or AllCombinations templates. Each places separate stakes; a size-1 Parlay adds singles. Repeated legs in different templates are additional bets.',
        'legs': 'Every unordered combination of this many distinct eligible events in the whole group. 2=pairs, 3=triples, 4=quads, 1=singles. With n<k there are no tickets; no incomplete tail is dropped.',
        'stage_column': 'Stage identity column. Tournament id keeps repeated round numbers in different stages separate. Choose Single-stage seasons only when that declaration is correct; missing stage metadata otherwise fails clearly.',
    }
    for key in ticket_components:
        for f in components[key]['fields']:
            name = f['name']
            f.update(help=ticket_help[name], primary=True)
            if name == 'grouping':
                f.update(kind='select', choices=[{'value':'league_round','label':'Same league and round'},
                                                {'value':'round','label':'Same round across leagues'},
                                                {'value':'day','label':'Same calendar day (UTC)'}])
            if name == 'order_by':
                f.update(kind='select', choices=['kickoff', 'probability', 'expected_profit'])
            if name in ('on_push', 'on_void'):
                f.update(kind='select', choices=['remove', 'refund', 'loss'])
            if name == 'probability_mode':
                f.update(kind='select', choices=['none', 'independent'])
            if name == 'stake_mode':
                f.update(kind='select', choices=['per_ticket', 'total'])
            if name in ('size','max_tickets'):
                f.update(kind='number', min=1, step=1)
            if name == 'size':
                f['title'] = 'Pool size' if key.endswith('MultiBet') else 'Legs per ticket'
            if name == 'stake':
                f.update(kind='number', min=0, step='any')
            if name == 'sizes':
                f.update(kind='list', item={'kind':'number', 'min':1, 'step':1}, title='System sizes')
            if name == 'tickets':
                f.update(kind='map', item={'kind':'component', 'components':ticket_components[:-1], 'initial_component':'evaluation.Parlay'})
            if key == 'evaluation.AllCombinations':
                if name == 'min_ev':f.update(kind='number', step='any', nullable=True, initial=0.0, title='Minimum ticket EV (per unit stake)')
                if name == 'legs':f.update(kind='number',min=1,step=1,title='Legs per ticket')
                if name == 'grouping':f.update(choices=[{'value':'league_round','label':'Same league, season, stage and round'}])
                if name == 'stage_column':f.update(kind='select',choices=[{'value':'tournament_id','label':'Tournament id'}, {'value':'stage_id','label':'Stage id'}, {'value':'stage','label':'Stage label'}, {'value':None,'label':'Single-stage seasons (explicit)'}],nullable=False,all_when_null=True)
                if name == 'stake':f['help']='Stake on EACH selected ticket. C(n,k) × stake is candidate exposure before ticket EV filtering. Actual stake is selected tickets × stake. Legs are not separately staked.'
                if name == 'max_tickets':f['help']='Maximum total tickets per template in this report scope (default 100,000). All groups are counted before any expansion. Above the bound: error with exact count; never sample or truncate.'
    components['features.SeasonProgress']['fields'].sort(key=lambda field: field['name'] != 'mode')
    _bayesian_widgets(components, stages)
    from ..evaluation import list_metrics
    metrics = list_metrics().to_dict('records')
    for m in metrics:
        fields=[]
        if m['name'] in ('f1','f1_score','precision','precision_score','recall','recall_score'):
            fields=[dict(name='average',title='Class averaging',kind='text',choices=['macro','micro','weighted','binary'],default='macro',help='Macro gives each class equal weight; weighted uses class frequency; binary focuses on the positive label.'),dict(name='pos_label',title='Positive label',kind='number',default=1,visible_when={'average':['binary']},help='Observed class counted as positive for binary averaging.')]
        elif m['name'] in ('roc_auc','brier_score','brier_score_loss'):
            fields=[dict(name='positive_label',title='Positive label',kind='number',default=1,help=HELP['positive_label'])]
        elif m['name'] in ('log_loss','cross_entropy','binary_cross_entropy'):
            fields=[dict(name='eps',title='Probability floor',kind='number',default=1e-15,help='Clip extremely small probabilities for a finite logarithmic loss.')]
        elif m['name'] == 'count_ou_brier':
            from ..evaluation.count_scores import DEFAULT_OU_LINES
            fields=[dict(name='lines',title='Over/under lines',kind='list',
                         item={'kind':'number','min':0.5,'step':1,'default':6.5},
                         default=list(DEFAULT_OU_LINES),
                         help='Distinct positive half-lines, e.g. 6.5 or 10.5. Both sides and all lines have equal weight; lower Brier score is better.')]
            fields.append(dict(name='distribution',title='Count distribution',kind='select',
                choices=['categorical','negative_binomial','poisson'],default='categorical',
                help='Categorical uses retained class probabilities. Negative binomial uses mean and dispersion; Poisson uses mean with zero dispersion. Native count models retain these in count_distribution.'))
            m['output_by_distribution']={'categorical':'predict_proba','negative_binomial':'count_distribution','poisson':'count_distribution'}
        m['fields']=fields
        m['description']={'numeric':'Compares numeric predictions with observed values.','label':'Compares predicted and observed classes.','probability':'Evaluates predicted class probabilities.','uncertainty':'Describes predicted uncertainty; no observed label is required.'}[m['kind']]
        if m['name'] == 'count_ou_brier':
            m['description']='Equal-weight Over/Under Brier score from exact count probabilities. Lower is better; observed counts outside fitted support remain included.'
    _svm_widgets(components)
    result=dict(version=1,components=components,stages=stages,metrics=metrics)
    Path(__file__).with_name('inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


def _svm_widgets(components):
    """Keep mixed numeric/string SVM parameters usable in the native forms."""
    for model in ('SVR', 'SVC'):
        component = components['sklearn.svm.' + model]
        component['title'] = model + (' — Support vector regression' if model == 'SVR' else ' — Support vector classification')
        component['description'] = (
            'Support vector regression for a single numeric target.' if model == 'SVR' else
            'Support vector classification for a single discrete target; classes are learned from each training fold.'
        ) + ' Use imputation and feature scaling in Preprocessing. Kernel fitting can be expensive for large datasets.'
        fields = {f['name']: f for f in component['fields']}
        fields['C'].update(title='C (inverse regularization)', primary=True, min=1e-300, step='any',
                           help='Positive penalty for fitting errors. Smaller C gives stronger regularization; larger C fits training observations more closely.')
        fields['kernel'].update(kind='select', choices=['linear', 'poly', 'rbf', 'sigmoid'], primary=True,
                                help='Similarity function: linear, polynomial, radial basis (RBF), or sigmoid. Precomputed kernels need a separate Gram-matrix workflow and are not offered by this feature-table builder.')
        fields['gamma'].pop('choices', None)
        fields['gamma'].pop('initial', None)
        fields['gamma'].update(kind='rule_number', rules=['scale', 'auto'], number_default=0.1,
                               min=0, step='any', primary=True, visible_when={'kernel':['poly','rbf','sigmoid']},
                               help='Kernel coefficient. Scale uses 1 / (number of features × fitting-input variance); auto uses 1 / number of features. Or enter a nonnegative numeric value. Computed from fitting rows after preprocessing.')
        fields['degree'].update(min=0, step=1, visible_when={'kernel':['poly']})
        fields['coef0'].update(visible_when={'kernel':['poly','sigmoid']})
        fields['tol'].update(min=1e-300, step='any', help='Positive solver stopping tolerance.')
        fields['cache_size'].update(title='Kernel cache (MB)', min=1e-300, step='any',
                                    help='Memory in megabytes available for the kernel cache for each fitted estimator.')
        fields['max_iter'].update(min=-1, step=1, help='Solver iteration limit. -1 means no limit; a finite limit can stop before convergence.')
        if model == 'SVR':
            fields['epsilon'].update(primary=True, min=0, step='any',
                                     help='Width of the error-insensitive tube on each side of the prediction. Errors within epsilon incur no loss. Uses transformed target units if label scaling is enabled.')
        else:
            fields['probability'].pop('choices', None)
            # sklearn 1.9 uses a deprecation sentinel; the effective default is
            # still disabled. Present a real checkbox on all supported versions.
            fields['probability'].update(title='Enable class probabilities', kind='boolean', default=False, primary=True,
                                         help='Enable for libsvm internal five-fold probability estimation; probabilities are retained automatically. For fully chronological margin calibration, leave disabled and select Probability Calibrator: decision_function + sigmoid with an explicit availability policy. Do not combine the two routes.')
            fields['class_weight'].update(primary=True, choices=['balanced'], initial='balanced',
                                          help='Disabled uses equal class weights. Balanced uses inverse class frequencies from fitting rows. For custom weights or partial balancing, use ClassWeightReporter and Weights from instead; do not combine both weighting mechanisms.')
            fields['decision_function_shape'].update(kind='select', choices=['ovr','ovo'])
            fields['break_ties'].update(visible_when={'decision_function_shape':['ovr']})
            fields['random_state'].update(kind='number', step=1, visible_when={'probability':[True]},
                                          help='Seed for the internal probability-estimation shuffle. Has no effect when class probabilities are disabled.')


def _bayesian_widgets(components, stages):
    """Model-specific hints in the maintained inventory, using existing widgets."""
    from ..ratings.bayesian import FIXTURE_FIELDS, TEAM_FIELDS
    from ..ratings.bayesian_training import DEFAULT_FIT_FIELDS

    for key, name in [('run', 'model_serializer'), ('model_loading', 'serializer')]:
        for field in stages[key]['fields']:
            if field['name'] == name:
                field.update(kind='component', components=['training.JoblibSerializer', 'training.BayesianScoreSerializer'],
                             initial_component='training.JoblibSerializer',
                             help='Explicit persistence backend. Choose Bayesian Score Serializer for Bayesian goal-score models; ordinary estimator models use Joblib. Supply the same serializer when loading.')

    for field in components['labels.MatchGoals']['fields']:
        field.update(kind='select', choices=['current'], primary=True,
                     help='Paired native current scores, producing home_goals and away_goals targets. This is not guaranteed to be regulation time; no other score fields are substituted.')

    for key, choices in [('features.BayesianRating', TEAM_FIELDS),
                         ('features.BayesianFixture', FIXTURE_FIELDS)]:
        for field in components[key]['fields']:
            name = field['name']
            if name == 'fields':
                field.update(kind='multiselect', choices=list(choices), primary=True,
                             help='Select output columns in the requested order. This does not alter filtering, fitted parameters or the full saved state.')
            elif name == 'model':
                field.update(kind='component', components=['input.BayesianModel', 'ratings.BayesianModel'],
                             initial_component='input.BayesianModel', primary=True,
                             help='Optional fixed parameter bundle. Disabled uses documented priors. For evaluation, any fitted bundle must have a training cutoff before every predicted row.')
            elif name == 'name':
                field.update(kind='select', discovery='bayesian_ratings', primary=True,
                             title='Saved Bayesian run',
                             help='Optional saved Bayesian run configured in Saved rating runs. Leave the model disabled when selecting a saved run.')
    for field in components['features.Rating']['fields']:
        if field['name'] == 'fields':
            field['choices'] = list(dict.fromkeys([*field['choices'], *TEAM_FIELDS]))
            field['source_choices'] = {
                'generated': ['rating', 'rd', 'sigma'],
                'input.BayesianRatingRun': list(TEAM_FIELDS),
            }
            field['help'] = 'Select numeric state columns for the named source. Disabled exports all its numeric fields. Bayesian runs offer attack and defence summaries; generated Glicko streams offer rating, rd and sigma.'
    for key in ('ratings.BayesianModel', 'ratings.BayesianScoreAdapter'):
        for field in components[key]['fields']:
            name = field['name']
            if name in ('parameters', 'initial', 'config'):
                component = 'ratings.BayesianConfig' if name == 'config' else 'ratings.BayesianParameters'
                field.update(kind='component', components=[component], initial_component=component)
            elif name == 'mode':
                field.update(kind='select', choices=['pooled', 'per_league'], primary=True,
                             help='Pooled shares fitted parameters across the selected leagues. Per league estimates separate parameters for each league; histories and states retain league identity.')
            elif name in ('home_goal_target', 'away_goal_target'):
                field.update(kind='select', discovery='targets', primary=True,
                             help='Choose the observed integer goal-count target for the actual '+('home' if name.startswith('home') else 'away')+' team. Use match layout and untransformed scores.')
            elif name == 'min_seasons':
                field.update(kind='number', min=1, step=1,
                             help='Optional eligibility override. Defaults require three training seasons for pooled fitting and five for per-league fitting; these are pilot guidelines, not accuracy guarantees.')
            elif name == 'fit_fields':
                field.update(kind='multiselect', choices=list(DEFAULT_FIT_FIELDS),
                             help='Parameters to estimate from fitting rows. Disabled uses the restrained default subset; all other parameters retain their initial values.')
            elif name == 'objective':
                field.update(kind='select', choices=['outcome_log_loss', 'score_log_loss'],
                             help='Optimize sequential win/draw/loss probabilities or the probability of the observed home/away score pair, using training results only.')
            elif name in ('history', 'team_seasons', 'season_starts'):
                field.update(kind='component', components=['input.Table'], initial_component='input.Table',
                             help='Optional explicit contextual table. Fitting still uses only identities and results eligible in the current training partition.')
            elif name == 'available_at':
                field.update(kind='text', help='Optional result-availability column in the supplied history or match metadata. Disabled uses the earlier-finished-kickoff proxy.')
                for obsolete in ('components', 'initial_component'):
                    field.pop(obsolete, None)
            elif name == 'training_cutoff':
                field.update(kind='text', help='Earliest permitted prediction time, respecting parameter calibration and any declared fold fit time. Preserve it on export.')
    for field in components['ratings.BayesianModel']['fields']:
        if field['name'] == 'per_league':
            field.update(kind='list', item={'kind': 'record', 'fields': [
                {'name': '0', 'title': 'Competition ID', 'kind': 'number', 'required': True, 'step': 1},
                {'name': '1', 'title': 'Parameters', 'kind': 'component', 'required': True,
                 'components': ['ratings.BayesianParameters'], 'initial_component': 'ratings.BayesianParameters'},
            ]}, help='Separate competition ID and parameter pairs. Prefer loading a trained JSON bundle to preserve exact identifiers and training provenance.')
    for field in components['ratings.BayesianConfig']['fields']:
        name = field['name']
        choices = {'method': ['vb', 'one_step'], 'clock': ['team_match', 'days'],
                   'score_basis': ['provider_current', 'regulation'],
                   'transition': ['mirrored', 'bridge']}
        if name in choices:
            field.update(kind='select', choices=choices[name], primary=True)
            for obsolete in ('components', 'categories', 'initial_component'):
                field.pop(obsolete, None)
        if name == 'score_basis':
            field['help'] = 'Provider current uses native goals. Regulation requires an externally verified history explicitly marked score_basis=regulation; it does not reconstruct regulation scores.'
        elif name == 'transition':
            field['help'] = 'Mirrored gives promoted/relegated teams direction-aware destination priors. Bridge retains an available source posterior using explicitly supplied league gaps, falling back to mirrored priors when evidence is absent.'
        elif name == 'method':
            field['help'] = 'VB converges variational updates; one_step uses the paper\'s single-update approximation. This is an inference setting, not a parameter-training mode.'
        elif name == 'bridge_gaps':
            field.update(kind='list', item={'kind': 'record', 'fields': [
                {'name': '0', 'title': 'Source competition ID', 'kind': 'number', 'required': True, 'step': 1},
                {'name': '1', 'title': 'Destination competition ID', 'kind': 'number', 'required': True, 'step': 1},
                {'name': '2', 'title': 'Log strength gap', 'kind': 'number', 'required': True, 'step': 'any'},
            ]}, help='Explicit source/destination/gap triples. Positive gap means the source competition is stronger. Reverse movements reuse the negative gap. Import a bundle to preserve string identifiers.')


if __name__ == '__main__':
    build()
