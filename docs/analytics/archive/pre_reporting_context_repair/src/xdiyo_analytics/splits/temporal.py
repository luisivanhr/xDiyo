"""Chronological fold selection with match-level eligibility at fitting time."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .core import Splitter, _Matches, _duration, _fold, _integer, _times


@dataclass
class TemporalSplit(Splitter):
    """Expanding/sliding windows in observed rounds, seasons or kickoff batches.

    train_size is the initial/minimum block span for expanding windows and the
    fixed candidate span for sliding windows. test_size and step count blocks;
    step=None advances by test_size. gap excludes blocks between candidate train
    and test. gap_time additionally requires training labels to be available by
    fit_at-gap_time. Neither setting controls feature histories or refitting.

    Rounds/seasons default to separate competition calendars. Kickoffs default
    to one pooled calendar. calendar_by=() explicitly pools; block_by can supply
    shared season/round identifiers for a deliberately synchronized calendar.
    Never equate unrelated leagues' round numbers implicitly. Default round keys
    are (season_id, round); include stage in block_by for multi-stage competitions.
    Blocks are ordered by earliest observed kickoff, including across seasons.
    Candidate train matches postponed past the fitting boundary are excluded.
    Empty training folds raise rather than silently disappearing.

    score_start skips that many held-out blocks for scoring only. score_rounds
    optionally selects an inclusive (low, high) round-number range in EACH test
    season; either bound may be None. Unscored test rows stay held out. Final
    short horizons are omitted unless allow_partial_test=True.

    folds(..., cutoffs=None, available_at=None): timestamp column names or aligned
    vectors. Fit once at the earliest test prediction cutoff (kickoff by default).
    Candidate training requires earlier kickoff, all selected targets nonmissing,
    and all match rows' labels available by the fitting boundary. available_at=None
    uses kickoff as a retrospective proxy; it does not infer actual result times.
    Missing explicit availability excludes a training match. Default cutoff and
    availability assumptions are recorded in every fold's metadata.
    """

    train_size: int
    test_size: int = 1
    step: object = None
    window: str = "expanding"
    unit: str = "rounds"
    gap: int = 0
    gap_time: object = None
    calendar_by: object = None
    block_by: object = None
    score_start: int = 0
    score_rounds: object = None
    allow_partial_test: bool = False

    def folds(self, dataset, *, cutoffs=None, available_at=None):
        for name in ("train_size", "test_size"):
            _integer(getattr(self, name), name)
        for name in ("gap", "score_start"):
            _integer(getattr(self, name), name, 0)
        step = self.test_size if self.step is None else self.step
        _integer(step, "step")
        if self.window not in {"expanding", "sliding"}:
            raise ValueError("window must be expanding or sliding.")
        if self.unit not in {"rounds", "seasons", "kickoffs"}:
            raise ValueError("unit must be rounds, seasons or kickoffs.")
        if self.score_start >= self.test_size:
            raise ValueError("score_start must be smaller than test_size.")
        gap_time = _duration(self.gap_time, "gap_time")
        matches = _Matches(dataset)
        kicks = matches.times("kickoff_at", "kickoff_at")
        predict = kicks if cutoffs is None else matches.times(cutoffs, "cutoffs", "min")
        if (predict > kicks).any():
            raise ValueError("Prediction cutoffs cannot be later than kickoff.")
        if available_at is None:
            available = kicks
        else:
            raw = _times(dataset, available_at, "available_at")
            available = pd.to_datetime([
                raw.iloc[rows].max() if raw.iloc[rows].notna().all() else pd.NaT
                for rows in matches.positions], utc=True)
        observed = np.array([dataset.y.iloc[rows].notna().all().all()
                             for rows in matches.positions])
        calendar_by = self.calendar_by
        if calendar_by is None:
            calendar_by = () if self.unit == "kickoffs" else ("competition_id",)
        calendar_by = matches.columns(calendar_by)
        if calendar_by:
            calendars, calendar_keys = pd.factorize(
                pd.MultiIndex.from_frame(matches.meta[list(calendar_by)]), sort=False)
        else:
            calendars, calendar_keys = np.zeros(len(kicks), dtype=int), [()]
        if self.block_by is None:
            block_by = {"rounds": ("season_id", "round"), "seasons": ("season_id",),
                        "kickoffs": ("kickoff_at",)}[self.unit]
        else:
            block_by = self.block_by
        block_by = matches.columns(block_by)
        if not block_by:
            raise ValueError("block_by cannot be empty.")
        score_mask = np.ones(len(kicks), dtype=bool)
        if self.score_rounds is not None:
            if len(self.score_rounds) != 2:
                raise ValueError("score_rounds must be (low, high), inclusive.")
            matches.columns(("round",))
            rounds = pd.to_numeric(matches.meta["round"], errors="raise")
            low, high = self.score_rounds
            if low is not None:
                score_mask &= (rounds >= low).to_numpy()
            if high is not None:
                score_mask &= (rounds <= high).to_numpy()
            if low is not None and high is not None and low > high:
                raise ValueError("score_rounds low cannot exceed high.")
        folds = []
        for calendar_id, calendar_key in enumerate(calendar_keys):
            ids = np.flatnonzero(calendars == calendar_id)
            codes, keys = pd.factorize(
                pd.MultiIndex.from_frame(matches.meta.iloc[ids][list(block_by)]), sort=False)
            blocks = [ids[codes == i] for i in range(len(keys))]
            order = sorted(range(len(blocks)), key=lambda i: kicks[blocks[i]].min())
            blocks, keys = [blocks[i] for i in order], [keys[i] for i in order]
            for train_end in range(self.train_size, len(blocks) - self.gap, step):
                test_begin = train_end + self.gap
                test_end = min(test_begin + self.test_size, len(blocks))
                if test_end - test_begin < self.test_size and not self.allow_partial_test:
                    break
                train_begin = 0 if self.window == "expanding" else train_end - self.train_size
                candidate = np.concatenate(blocks[train_begin:train_end])
                test = np.concatenate(blocks[test_begin:test_end])
                fit_at = predict[test].min()
                boundary = fit_at - gap_time
                eligible = ((kicks[candidate] < boundary) &
                            (available[candidate] <= boundary) & observed[candidate])
                train = candidate[eligible]
                score_blocks = blocks[test_begin + self.score_start:test_end]
                score = np.concatenate(score_blocks) if score_blocks else np.array([], dtype=int)
                score = score[score_mask[score]]
                folds.append(_fold(matches, train, test, {
                    "scheme": "temporal", "retrospective": False,
                    "calendar": calendar_key, "calendar_by": calendar_by,
                    "block_by": block_by, "unit": self.unit, "window": self.window,
                    "fit_at": fit_at, "training_boundary": boundary,
                    "train_blocks": tuple(keys[train_begin:train_end]),
                    "test_blocks": tuple(keys[test_begin:test_end]),
                    "excluded_train": matches.rows(candidate[~eligible]),
                    "availability": "kickoff_proxy" if available_at is None else "explicit",
                    "cutoffs": "kickoff" if cutoffs is None else "explicit",
                }, score))
        if not folds:
            raise ValueError("No complete fold fits the requested calendar/spans.")
        return folds
