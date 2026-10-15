"""Internal snapshot boundaries; pandas deep copies alone do not own object cells."""
from copy import deepcopy


def isolated_frame(frame):
    """Independent values/metadata, including nested mutable object-valued cells.

    Preserve column/index order, extension dtypes and duplicate columns. Scalar
    Python values are cheap for deepcopy; never coerce object columns to strings.
    """
    result = frame.copy(deep=True)
    for i, dtype in enumerate(frame.dtypes):
        if dtype == object:
            values = frame.iloc[:, i].to_numpy(dtype=object, copy=True)
            memo = {}
            for row in range(len(values)):
                values[row] = deepcopy(values[row], memo)
            result.isetitem(i, values)
    result.attrs = deepcopy(frame.attrs)
    return result
