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


HELP = {
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
    'reuse': 'Load a matching completed run rather than fitting it again.',
    'name': 'Human-readable name used in the recipe and report.',
    'data_root': 'Folder containing published season manifests and tables. Example: data/xDiyo_data.',
    'seasons': 'Select the seasons available in this data folder.',
    'leagues': 'Select available leagues. Disabled includes all leagues for the chosen seasons.',
    'tables': 'Select prediction-relevant source tables. Matches identify fixtures; statistics supplies numeric observations; pregame supplies standings.',
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
    'features.Lag': 'Take an earlier eligible observation. A lag of 1 gives the most recent completed match before the prediction cutoff.',
    'features.RollingMean': 'Average the selected statistic over eligible past observations. A window of 5 uses up to five previous matches, excluding the match being predicted.',
    'features.RollingStd': 'Measure variability across the observations in the past window. For league-round windows this uses individual observations, not averages of rounds.',
    'splits.TemporalSplit': 'Train on earlier matches and test on later matches. Use for forecasting a future season or round. Expanding keeps all earlier training data; sliding keeps a fixed-length window.',
    'splits.MatchKFold': 'Hold out whole matches in K groups. With shuffle enabled this tests generalization across matches, not a realistic future forecast. Both team rows always stay together.',
    'splits.GroupKFold': 'Hold out whole leagues, seasons or other metadata groups. Use to study transfer to groups absent from training; it does not enforce chronological order.',
    'splits.CPCV': 'Combinatorial purged cross-validation holds out combinations of chronological blocks and removes overlapping information intervals. Useful for strategy backtests; it can train on blocks later than a test block and is not a chronological deployment simulation.',
    'features.MatchResultGlicko': 'Evolving team strength from wins, draws and losses. Each feature uses the rating before that match; both teams update from their pre-match states.',
    'features.StatGlicko': 'Evolving strength for a selected statistic. For corner kicks, more corners counts as a win, equal corners as a draw. This rates superiority, not the size of the winning margin.',
    'features.Rating': 'Reuse a named rating stream configured below. Select strength, uncertainty and/or volatility without recalculating separate streams.',
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
            if name == 'prediction_methods': f.update(kind='multiselect', choices=['predict','predict_proba'])
            if name == 'engine': f.update(kind='component', categories=['rating'], components=['ratings.Glicko2'], initial_component='ratings.Glicko2')
            if name in ('stat',): f.update(kind='component', components=['features.Stat'], initial_component='features.Stat')
            if name in ('early_stopping','scheduler','control','validation','transition','handoff','policy'):
                ids={'early_stopping':['training.EarlyStopping'],'scheduler':['training.ReduceOnPlateau'],'control':['training.TrainingControl'],'validation':['training.ValidationTail'],'transition':['ratings.GlickoTransition'],'handoff':['features.Hard','features.LinearFade','features.ObservationCount'],'policy':['features.SeededEMA']}
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
            if name == 'partition' and key in ('reporting.LearningCurveReporter','reporting.CoefficientReporter'): f['choices']=['model']
            if name == 'partition' and key == 'reporting.ExperimentLeaderboardReporter': f['choices']=['experiment']
            if name == 'catalog' and key == 'reporting.MatchResultReporter': f.update(kind='reference',initial={'ref':'team_catalog'},hidden=True)
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
            if name == 'source' and key == 'features.League':f.update(components=['features.Stat','features.ForAgainst'],initial_component='features.Stat')
            if name == 'source' and key == 'features.WarmStart':f.update(components=['features.RollingMean','features.RollingStd','features.RollingZScore'],initial_component='features.RollingMean')
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
    patch('data','tables',kind='multiselect',choices=['matches','statistics','pregame','shots'])
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
            if name=='strength':f['help']='Prior effective observation count. EMA weight is strength / (strength + new observations).'
            if name=='start' and c['id']=='features.LinearFade':f['help']='Completed rounds before the gradual fade begins.'
            f['primary'] = f['name'] in primary
            if f['name']=='type' and c['category'] in ('pre_reporter','post_reporter'):
                f['choices']=[v for v in ['overall','per_fold','timeline'] if v in f.get('choices', ['overall','per_fold'])]
            if f['name']=='type' and c['id']=='reporting.FeatureTimeline':f['choices']=['overall','per_fold','timeline']
            if f['name']=='partition' and c['category']=='post_reporter' and c['id'] not in ('reporting.LearningCurveReporter','reporting.CoefficientReporter','reporting.ExperimentLeaderboardReporter'):f['choices']=['score','test']
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
    # Shared, maintained scaler controls for predictor and target transforms.
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
            if f['name']=='ratings':f['hidden']=True
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
            if c['id']=='input.Table' and name=='index':f.update(kind='multiselect',choices=['competition_id','season_id','team_id','event_id','fold_id','row_position'],help='Columns forming the exact row index. Select both competition_id and season_id for league-season counts.')
            if name in ('steps','transformers') and c['id'] in ('sklearn.pipeline.Pipeline','sklearn.compose.ColumnTransformer'):
                fields=[dict(name='0',title='Step name',kind='text',required=True,default='step',help='Unique name for this preprocessing step.'),dict(name='1',title='Transformation',kind='component',required=True,categories=['preprocessor','model'] if name=='steps' else ['preprocessor'],initial_component='sklearn.preprocessing.StandardScaler',help='Choose the fitted operation to apply.')]
                if name=='transformers':fields.append(dict(name='2',title='Input columns',kind='multiselect',discovery='features',required=True,default=[],help='Prepared predictor columns passed to this operation.'))
                f.update(kind='list',initial=[],item={'kind':'record','fields':fields,'initial':['step',{'component':'sklearn.preprocessing.StandardScaler','params':{}},*([[]] if name=='transformers' else [])]})
        if c['id'].startswith('lightgbm.'):
            c['fields'].append(dict(name='max_bin',title='Maximum bins',kind='number',required=False,default=255,min=2,step=1,help='Maximum histogram bins per feature. Larger values allow finer splits at higher memory cost.'))
            if c['id'].endswith('Regressor'):
                c['fields'].append(dict(name='alpha',title='Quantile / Huber alpha',kind='number',required=False,default=0.9,visible_when={'objective':['quantile','huber']},help='Quantile level (0.5 is the median), or the Huber-loss alpha when that objective is selected.'))
    count_model = components.get('training.NegativeBinomialRegressor')
    if count_model:
        for f in count_model['fields']:
            name = f['name']
            f['help'] = {
                'dispersion': 'NB2 variance = mean + dispersion × mean². Positive and fixed during fitting; tune it in Model selection. For mean 10, dispersion 0.2 gives variance 30. This is not a regularization penalty.',
                'fit_intercept': 'Learn the baseline log mean. Usually enabled; disabling assumes an expected count of 1 when all transformed inputs are zero.',
                'max_iter': 'Maximum iterations used to fit coefficients. Increase if fitting reports that convergence was not reached.',
                'tol': 'Tolerance for convergence of the fitting algorithm. Smaller values require a more precise fit.',
                'penalty': 'None preserves ordinary NB regression. Ridge shrinks coefficients; Lasso can set them to zero; Elastic Net combines both. The intercept is never penalized. Standardize inputs before using a penalty.',
                'alpha': 'Penalty strength, separate from Dispersion. Zero disables shrinkage. Try 0.001, 0.01, 0.1 and 1 in grid search. Larger values shrink fitted-feature coefficients more strongly.',
                'l1_ratio': 'Elastic Net mixture: 0 is pure Ridge, 1 is pure Lasso, and 0.5 mixes both equally. Used only with Elastic Net.',
                'prediction': 'Mean returns the expected count. Mode returns the most probable integer count (upper mode for ties). With dispersion 1 or greater, the mode is always zero. This changes the prediction summary, not the fitting objective, and does not guarantee wider predictions.',
            }[name]
            f['primary'] = name in ('dispersion', 'fit_intercept', 'penalty', 'alpha', 'l1_ratio', 'prediction')
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
        m['fields']=fields
        m['description']={'numeric':'Compares numeric predictions with observed values.','label':'Compares predicted and observed classes.','probability':'Evaluates predicted class probabilities.','uncertainty':'Describes predicted uncertainty; no observed label is required.'}[m['kind']]
    result=dict(version=1,components=components,stages=stages,metrics=metrics)
    Path(__file__).with_name('inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')


if __name__ == '__main__':
    build()
