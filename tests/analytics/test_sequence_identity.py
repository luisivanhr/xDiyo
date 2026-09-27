"""Sequence type and order remain part of executable recovery identities."""
import pytest

from xdiyo_analytics.experiments.recovery import execution_key, signature
from xdiyo_analytics.selection import Candidate


def sequence_factory(values, explicit_key):
    from model_selection_samples import FixedAdapter
    if explicit_key:
        class Factory:
            def __init__(self):
                self.values = values

            def cache_key(self):
                return {"settings": {"values": self.values}}

            def __call__(self):
                return FixedAdapter(1 if isinstance(self.values, list) else 2)
        return Factory()

    def factory():
        return FixedAdapter(1 if isinstance(values, list) else 2)
    return factory


@pytest.mark.parametrize("explicit_key", [False, True])
def test_sequence_type_change_refits_actual_experiment_and_preserves_unchanged_reuse(tmp_path, monkeypatch, explicit_key):
    from football_experiment_samples import prepared
    from xdiyo_analytics.experiments import FootballExperiment, football
    calls = []
    fit = football._fit_candidate

    def count_fit(*args, **kwargs):
        calls.append(True)
        return fit(*args, **kwargs)

    monkeypatch.setattr(football, "_fit_candidate", count_fit)
    experiment = FootballExperiment("Sequence-sensitive factory", output_dir=tmp_path)
    data = prepared(holdout=True)

    def run(values):
        return experiment.run(data, model=Candidate("Factory", sequence_factory(values, explicit_key)))

    first = run([1, 2])
    unchanged = run([1, 2])
    assert unchanged.reused and unchanged.record["run_id"] == first.record["run_id"]
    assert len(calls) == 1
    changed = run((1, 2))
    assert not changed.reused and changed.record["run_id"] != first.record["run_id"]
    assert len(calls) == 2
    assert (first.training.folds[0].predictions["predict"] == 1).all().all()
    assert (changed.training.folds[0].predictions["predict"] == 2).all().all()
    unchanged_tuple = run((1, 2))
    assert unchanged_tuple.reused and unchanged_tuple.record["run_id"] == changed.record["run_id"]
    assert len(calls) == 2


@pytest.mark.parametrize("values", [[], [1], [1, 2], [[1], {"nested": (2, 3)}]])
def test_list_and_tuple_signatures_are_distinct_at_each_nesting_level(values):
    assert signature(values) != signature(tuple(values))
    assert execution_key({"settings": {"values": values}}) != execution_key({"settings": {"values": tuple(values)}})


@pytest.mark.parametrize("container", [list, tuple])
def test_sequence_order_is_preserved_and_literal_mapping_cannot_collide(container):
    original = container([1, {"alpha": 2, "beta": 3}])
    reordered_mapping = container([1, {"beta": 3, "alpha": 2}])
    assert execution_key(original) == execution_key(reordered_mapping)
    assert execution_key(original) != execution_key(container(reversed(original)))
    # Even a mapping copying the complete descriptor must remain a mapping.
    assert signature(original) != signature(signature(original))
    assert execution_key(original) != execution_key({"sequence": container.__name__, "items": list(original)})


def test_nested_candidate_config_retains_sequence_type_and_ignores_mapping_order():
    from model_selection_samples import FixedAdapter
    one = Candidate("Factory", FixedAdapter, config={"model": "fixed", "settings": {"values": [1, 2], "alpha": .1}})
    reordered = Candidate("Factory", FixedAdapter, config={"settings": {"alpha": .1, "values": [1, 2]}, "model": "fixed"})
    changed = Candidate("Factory", FixedAdapter, config={"model": "fixed", "settings": {"values": (1, 2), "alpha": .1}})
    assert execution_key(one) == execution_key(reordered)
    assert execution_key(one) != execution_key(changed)


def test_recursive_sequences_have_bounded_stable_type_specific_signatures():
    def cycle(container):
        link = []
        sequence = container([link])
        link.append(sequence)
        return sequence

    for container in (list, tuple):
        assert execution_key(cycle(container)) == execution_key(cycle(container))
    assert execution_key(cycle(list)) != execution_key(cycle(tuple))
