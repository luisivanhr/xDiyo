"""Ordered quote products with explicit integer and accounting overflow guards."""
import math
import numpy as np


def checked_product(values, axis=None):
    values = np.asarray(values)
    if values.dtype.kind in 'iu':
        rows = values.reshape(1, -1) if axis is None else values
        for row in rows:
            if not 0 <= math.prod(int(v) for v in row) <= np.iinfo(np.int64).max:
                raise ValueError('Integer quote product overflow; use representable canonical odds.')
    with np.errstate(over='ignore', under='ignore'):
        result = np.prod(values, axis=axis)
    if not np.isfinite(result).all():
        raise ValueError('Combined odds overflow; reduce ticket size.')
    return result


def check_accounting(frame):
    for field in ('stake', 'payout', 'profit'):
        try:
            total = math.fsum(float(v) for v in frame[field].dropna())
        except OverflowError as exc:
            raise ValueError('Selected ticket accounting total overflow.') from exc
        if not np.isfinite(total):
            raise ValueError('Selected ticket accounting total overflow.')
