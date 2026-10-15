"""Explicit numeric placeholders and masks at the existing model boundary."""
import math

import numpy as np


def numeric_values_and_mask(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        pairs = [numeric_values_and_mask(item) for item in value]
        return [p[0] for p in pairs], [p[1] for p in pairs]
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0, False
    return (number, True) if math.isfinite(number) else (0.0, False)


def prepare_team_inputs(data, history):
    """Return finite values; padding, absent facts and absent xG stay masked.

    The legacy sequence includes its current-match slot. This helper preserves
    that contract; it does not define what a prediction model may observe.
    """
    feature_names = [s+"_"+f for s in ("team", "enemy") for f in
                     ("standings", "formations", "scores", "atk", "def", "shots", "penalties", "owngoals")]
    masks = {}
    padding = len(data["team_scores"])-len(history)
    for key in feature_names:
        values, mask = numeric_values_and_mask(data[key])
        mask = np.asarray(mask, dtype=bool)
        mask[:padding] = False
        data[key] = values
        masks[key] = mask.tolist()
    for side in ("team", "enemy"):
        data[side+"_shots_available"] = [False]*padding + [match[side+"_shots_available"] for match in history]
        for group, key in enumerate(("shots", "penalties", "owngoals")):
            name = side+"_"+key
            for i, lengths in enumerate(data[side+"_shots_length"]):
                for j in range(lengths[group], len(masks[name][i])):
                    masks[name][i][j] = [False, False, False]
        for kind in ("player", "goalkeeper"):
            name = side+"_"+kind+"_heatmap_sequence"
            data[name], finite = numeric_values_and_mask(data[name])
            masks[name] = np.asarray(finite, dtype=bool)
            masks[name][:padding] = False
            for i, match in enumerate(history, padding):
                if not match[side+"_"+kind+"_heatmap_available"]:
                    masks[name][i] = False
            masks[name] = masks[name].tolist()
    data["availability"] = masks
    return data
