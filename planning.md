## Creation of feature templates
The idea is to have three ingredients:
- A library of _operators_ and _operator building blocks_ that act on the given data and have their own available configuration. These operators could be built or enhanced using the _operator building blocks_ as specified by _templates_ that include all the details about the operators. The details of these templates will be given below. The idea is to have a dynamic library that we can actively update by adding any new element to a registry.
- A list of _templates_ which specify features that can be obtained from data. Those templates have a given name and indicate the way in which _existing operators_ should be
used and with which configuration in order to obtain the desired/required features. At the same time, each operator will expose metadata so our _auditor_ can make sure that the operator being used in a template supports the options that we are asking for in the template.
- An _interface_ (i.e. GUI) that allows to choose _existing templates_ and specify the values of the blank/multiple choice fields or create new templates from scratch (with a predefined structure).

After specifying the set of templates that will be used for feature engineering, an _orchestrator_ will decide the DAG of the tasks required for computation and output some artifacts containing the engineered features and the run metadata. The library of choice for the computation of all the features is Pandas. First we explain the format convention for the templates for both features and targets. Once the conventions are established, we will specify how the templates will be used in the passes through the _orchestrator_ and _auditor_. Finally, we will describe what the output artifacts will be.

## Standard design for templates and artifacts

### Feature template
- name: A string that indicates the name of the feature. Naming format is capitalizing the first letter of each word, with no spaces. Some examples are "RollingMean", "ExponentialMean", "StatRatioDifference", etc.
- family: This field will contain strings separated by a "|" character to enlist which feature families a particular operator supports. The purpose is that this, together with the warm-start status will determine the partition over which the operators will act. The supported families are "league" | "team" | "rating" | "headtohead" | "postprocessing". "postprocessing" is used for example for shrinkage or warm-start operators. Which family is picked when specifying a template will be indicated by the choice in the GUI.
- kernel: Again, this will contain strings separated by a "|" character to enlist which aggregation kinds a particular operator supports. The list is "rolling" |"expanding" |"snapshot" |"point" |"expression" | "teacher". Here the "expression" type is reserved for custom features that we will implement using a _feature builder_ and will have custom aggregation rules, and "teacher" will be reserved by those features that will be given by the output of pretrained models.
- window: Will only have effect for "rolling" kernels and is just to specify the length of the lookback.
- partition: Several types of partitions depending on the family and warm-start status. The indicate how Pandas vectorizes the computations. Some of the options are for league and no warm-start: ["league"], league and warm-start: ["league","season_start"], team and no warm-start: ["team_id"], team and warm-start: ["team_id","season_start"], head to head and no warm-start: ["team_id","opponent_id"], head to head and warm-start: ["team_id","opponent_id","season_start"]. More could be added at a later point.
- inputs: Specifies on which of the columns of our data dataframe the operator acts on. For features that act on single columns we can specify those columns as list[str], for features that act on two columns, we can specify the pairs as list[tuple[str,str]], this could be the case of "StatRatioDifference" acting on [(home_team_45_min_goals,away_team_45_min_goals),(home_team_90_min_corners,away_team_90_min_corners)]. This can be as general as list[tuple[str,...,str]] for features that act on multiple columns, for example "DifferenceByPossession" acting on [(home_team_total_goals,home_opponent_total_goals,home_team_total_possession),(home_team_45_min_corners,home_opponent_45_min_corners,home_team_45_min_possession)], and so on. The convention for the input will in general be prefix_half_stat, where:
    - prefix could be "home_team", "home_opponent", "away_team", or "away_opponent"
    - half could be "45_min", "90_min", or "total"
    - stat is the name of the stat in question, some examples are "Standing", "BallPossession", "Cornerkicks", etc. There is no particular rule in the naming of stats.
- aggregation: This string specifies which is the operation performed on the data and could be any available aggregation method we have in our library. Some examples are "sum", "count","mean","var","std","rate", "custom" (in the case of teacher modules that have their own aggregation), or for the case  of "expression" kernels, we will have the aggregation indicated by full expression _Abstract Syntax Trees_, as will be described later.
- output: This is a string indicating the name of the output column. The naming convention will be prefix_half_stat_feature, where:
    - prefix is as in the input
    - half is as in the input
    - stat is as in the input
    - feature is the name of the feature (the "name" field of this template), e.g. "RollingMean", "ExponentialMean", etc.
- requires: Requirements for the current feature. The idea is that if they are not necessarily kept for the final output, but they may be added to a _snapshots_ table in case they are necessary for some intermediate calculation.
- is_loo: Stands for "is leave-one-out". This will be a Bool True or False that is True if the template describes a leave-one-out feature and False if not.
- shift_policy: Indicates the amount of shift the feature will have. The idea is to prevent information leak from the future or to give information to the model from a more distant past. The values would be either "nolag" for 0 lag or 1,2,3,..., depending on how many steps the feature is delayed.
- warm_start_blend: This will indicate the type of warm-start or shrinkage (if any) that a feature will use for borrowing strength from previous data or data from other teams. The structure is for example {"type": "LinearWarmStart", alpha_schedule: {"start_round": 1, "end_round": 6}}. The warm-start or shrinkage operators will be defined as features from templates that belong to the postprocessing family and their output will be a column with the weights corresponding to each round and also the column with the warm-started or shrunk features.
- audit: This is for ensuring congruence between the input, the settings, and the feature in use. One example could be {"min_count": 3, "allow_zero_var": False}.

### Target template
- name: A string that indicates the name of the target. Naming format is capitalizing the first letter of each word, with no spaces. Some examples are "BettingOption", "Winner", "Rank", "SingleStat" "MultiStat", "MatchTotal", "MultiTotals", etc.
- family: This field will contain strings separated by a "|" character to enlist which target families a particular operator supports. This will determine the partition over which the operators will act. The supported families for the targets are "team" | "match" | "bets" | "odds" | "rank". 
    - "team" targets is for example for a specific stat or group of stats for a single team in a match
    - "match" targets is for the pooled stats from both teams in a single match, for example total_corners = home_corners + away_corners
    - "bets" targets are for selecting suitable bet options for a particular match. They are meant to be used in classification problems where some examples are "more than 1.5 goals", "more than 2.5 goals", "less than 2.5 goals", etc
    - "odds" targets are meant to build our own set of odds as if we were bookmakers. Could be used paired with logit models.
    - "rank" targets are meant to be used to find the positions of the teams relative to each other and for finding tournament winners.
- source: It is the name of the raw columns to use
- scaling: Indicates the desired scaler, some examples are "None", "league_zscore", "league_minmax", "team_zscore", "team_minmax", "quantiles", etc. Scalers will be defined as operators in our library, however, they will be able to check if the output columns already exist to avoid computing them again. For example, if we are already computing the league-wide z-score of a specific stat, then we should already have the league-wide mean and standard deviation computed, so we could reuse those columns for the scaler.
- scaler_partition: It is just the partition for the scaler. In case warm-start is used, we do not warm-start the targets, only the moments or any prior used in the scaler. For example, for a z_score scaler, only the mean and std for the scaler are warm-started, not the target stat.
- scaler_window: Specifies the lookback, we align it with the windows form the feature engineering.
- scaler_shift: Indicates the amount of shift that the scaler will have. In the same way as the targets, this allows to avoid peeking into the future. The values would be either "nolag" for 0 lag (for debugging purposes) or 1.
- transform: Optional operation on the raw target prior to scaling. Given by some operator in our library. Default is None.
- export_mode: "stacked" or "match_wide" depending if we export in long or wide format.
- audit: Same as before, it is for assuring consistency between targets and operators. Some examples could be a fallback to the unscaled target or using a global prior if variance collapses in early rounds {"min_std", "fallback": "use_mean_only"}

### Implementing a feature builder

For implementing a _feature builder_ we will work with the "expression" kernel introduced before. The idea of this type of feature is that the aggregation will be specified by Abstract Syntax Trees (AST) where the operations will be taken from our library of operators. One example of such feature is a difference of the ratios of the rolling means of the stat of one team against the other: Diff(Ratio(RollingMean,RollingMean),Ratio(RollingMean,RollingMean))

- inputs: In this field we list all the raw stats needed to build the expression. In our example ["team_corners_infavor","team_corners_against","opponent_corners_infavor","opponent_corners_against"]
- aggregation: In our example, the syntax tree would be {"operation": Diff, "args": [{"operation": Ratio, "args":}, {"operation": Ratio, "args":}]}