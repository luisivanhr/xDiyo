"""Bounded synthetic parity/profile harness; never loads models or real outcomes.

Record before editing, then compare after editing with the same interpreter:
  python examples/benchmark_quote_consensus.py --output TEMP/before
  python examples/benchmark_quote_consensus.py --output TEMP/after --compare TEMP/before
Only load comparison pickles generated locally by this trusted harness.
"""
import argparse
import cProfile
import io
import json
from pathlib import Path
import pickle
import pstats
import runpy
from time import perf_counter

import pandas as pd

_example = runpy.run_path(str(Path(__file__).with_name('research_quote_consensus.py')))
consensus, synthetic_example = _example['consensus'], _example['synthetic_example']


def population(groups):
    source, _, contract = synthetic_example()
    frames = []
    for group in range(groups):
        frame = pd.concat([source.iloc[:1]] * 8, ignore_index=True)
        frame['event_id'] = range(group * 8 + 1, group * 8 + 9)
        frame['row_position'] = range(group * 8, group * 8 + 8)
        frame['round'] = group + 1
        frame['odds'] = 4.
        frame['settlement'] = 'missing'
        frame['quote_id'] = frame.event_id.map(lambda n: f'synthetic-q{n}')
        for field in ('kickoff_at', 'decision_at', 'assumed_available_at'):
            frame[field] += pd.Timedelta(days=7 * group)
        for model in ('xgb', 'lgbm'):
            for field in contract.identities(frame):
                frame[f'{model}::{field}'] = frame[field]
            for field in ('issued_at', 'trained_through', 'artifact_vintage'):
                frame[f'{model}::{field}'] += pd.Timedelta(days=7 * group)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    pmfs = {model: pd.DataFrame({'draw': p, 'non_draw': q}, index=data.event_id)
            for model, p, q in (('xgb', .30, .70), ('lgbm', .32, .68))}
    return data, pmfs, contract


def assert_metadata_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_metadata_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert type(actual) is type(expected) and len(actual) == len(expected)
        for a, b in zip(actual, expected):
            assert_metadata_equal(a, b)
    elif pd.isna(expected):
        assert pd.isna(actual)
    else:
        assert actual == expected


def assert_outputs_equal(actual, expected):
    assert len(actual) == len(expected)
    for a, b in zip(actual, expected):
        pd.testing.assert_frame_equal(a, b, check_exact=True)
        assert_metadata_equal(a.attrs, b.attrs)


def cases():
    for groups in (3, 6):
        yield f'groups-{groups}', population(groups), True
    data, pmfs, contract = population(1)
    yield 'settled', (data.assign(settlement=['win', 'loss', 'push', 'void'] * 2), pmfs, contract), False
    yield 'all-rejected', (data, {m: p.assign(draw=.01, non_draw=.99) for m, p in pmfs.items()}, contract), False
    for n in (0, 1):
        yield f'rows-{n}', (data.iloc[:n], {m: p.iloc[:n] for m, p in pmfs.items()}, contract), False
    yield 'permuted', (data.iloc[::-1], {m: pmfs[m].iloc[::-1, ::-1] for m in reversed(pmfs)}, contract), False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for name, inputs, profile in cases():
        profiler = cProfile.Profile()
        start = perf_counter()
        if profile:
            profiler.enable()
        outputs = consensus(*inputs)
        if profile:
            profiler.disable()
        elapsed = perf_counter() - start
        if profile:
            assert len(outputs[0]) == (84 if name == 'groups-3' else 168)
            assert outputs[0].quote_at.isna().all()
            profiler.dump_stats(str(args.output / f'{name}.prof'))
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats('cumulative')
            stats.print_stats(35)
            stats.print_callers('deepcopy')
            (args.output / f'{name}.txt').write_text(stream.getvalue(), encoding='utf-8')
        if args.compare:
            with (args.compare / f'{name}.pickle').open('rb') as handle:
                assert_outputs_equal(outputs, pickle.load(handle))
        with (args.output / f'{name}.pickle').open('wb') as handle:
            pickle.dump(outputs, handle)
        results.append(dict(case=name, seconds=elapsed, tickets=len(outputs[0]),
                            parity='exact' if args.compare else 'baseline'))
        print(results[-1], flush=True)
    (args.output / 'timings.json').write_text(json.dumps(results, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
