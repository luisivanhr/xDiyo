"""Numerical evaluation shared by reporters and future model selection."""

from .metrics import Metric, MetricDefinition, evaluate_metrics, list_metrics, register_metric
from .betting import BetSpec, evaluate_bets
from .probabilities import bet_probabilities, negative_binomial_bet_probabilities
from .bet_decisions import BetOffer, TightestLine, HighestExpectedProfit, BinaryDrawThreshold, prepare_bets
from .tickets import Parlay, BetSlip, MultiBet, AllCombinations, compose_bets, preview_combinations
from .decision_layer import FrozenTable, DecisionContext, DecisionLayer, DecisionResult, LearnedGate
from .stake_policy import StakeContext, StakePolicy, AllocationResult, FixedStake, FixedFraction, FractionalKelly, RiskLimits, ModelProbabilitySource, HistoricalRateSource, LearnedAllocation, allocate_batch
from .bankroll import BankrollLedger, BankrollResult, replay_bankroll, settlements_from_legs

__all__ = ["AllCombinations", "preview_combinations", "BinaryDrawThreshold", "Parlay", "BetSlip", "MultiBet", "compose_bets", "Metric", "MetricDefinition", "evaluate_metrics", "list_metrics", "register_metric", "BetSpec", "evaluate_bets",
           "bet_probabilities", "negative_binomial_bet_probabilities", "BetOffer", "TightestLine", "HighestExpectedProfit", "prepare_bets"]
__all__ += ['FrozenTable','DecisionContext','DecisionLayer','DecisionResult','LearnedGate',
            'StakeContext','StakePolicy','AllocationResult','FixedStake','FixedFraction','FractionalKelly',
            'RiskLimits','ModelProbabilitySource','HistoricalRateSource','LearnedAllocation','allocate_batch',
            'BankrollLedger','BankrollResult','replay_bankroll','settlements_from_legs']
