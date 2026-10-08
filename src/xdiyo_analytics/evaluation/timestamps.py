"""Call-local timestamp parsing shared by quote guards and audit inventory.

Legacy timezone-naive values mean UTC. ISO strings may differ in fractional
precision or offsets. Numeric epoch values are not quote/model timestamps.
Inventory coercion is separate from strict decision validation.
"""
from datetime import date, datetime
from numbers import Number
import numpy as np
import pandas as pd


def timestamps(values, *, utc=True, errors='raise'):
    if isinstance(values, pd.Series):
        if values.dtype == object or pd.api.types.is_numeric_dtype(values.dtype):
            invalid = values.map(lambda v: not pd.isna(v) and (
                isinstance(v, Number) or (isinstance(v, str) and not v.strip()) or not isinstance(v, (str, date, datetime, np.datetime64))))
            if invalid.any():
                if errors == 'raise':
                    raise ValueError('Timestamps require date/time values or strings, not numeric epochs or objects.')
                values = values.mask(invalid)
    return pd.to_datetime(values, utc=utc, errors=errors, format='mixed')
