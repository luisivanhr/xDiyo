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
        # Extension strings and categoricals need the same preflight as object
        # columns: to_datetime would silently turn an empty string into NaT.
        def invalid_value(value):
            if pd.api.types.is_scalar(value) and pd.isna(value):
                return False
            return (isinstance(value, Number)
                    or (isinstance(value, str) and not value.strip())
                    or not isinstance(value, (str, date, datetime, np.datetime64)))
        invalid = np.fromiter((invalid_value(v) for v in values), dtype=bool, count=len(values))
        if invalid.any():
            if errors == 'raise':
                raise ValueError('Timestamps require date/time values or strings, not numeric epochs or objects.')
            values = values.mask(invalid)
    return pd.to_datetime(values, utc=utc, errors=errors, format='mixed')
