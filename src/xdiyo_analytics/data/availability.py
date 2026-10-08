"""Honor sparse, reviewed result-release bounds in historical populations."""


def respect_result_availability(metadata, available):
    """Never let a proxy predate a reviewed result bound; retain unknown releases.

    ``result_available_at`` is sparse UTC datetime metadata, not a predictor.
    It may be a conservative end-of-completion-day bound rather than an exact
    publication timestamp. Explicit later availability remains later. NaT in
    the supplied availability remains unknown even when a bound is present.
    """
    import pandas as pd

    if isinstance(available, pd.Series) and not available.index.equals(metadata.index):
        raise ValueError('Result availability must align with metadata.')
    result = pd.Series(available, index=metadata.index)
    if 'result_available_at' not in metadata:
        return result
    bound = result_availability_bounds(metadata)
    result = pd.to_datetime(result, utc=True, errors='raise')
    return result.mask(result.notna() & bound.notna() & result.lt(bound), bound)


def result_availability_bounds(metadata):
    """Parse sparse reviewed bounds without interpreting numbers as epochs."""
    import pandas as pd
    from pandas.api.types import is_numeric_dtype
    raw = metadata['result_available_at']
    if is_numeric_dtype(raw.dtype) and raw.notna().any():
        raise TypeError('result_available_at must contain datetimes, not unitless numbers.')
    return pd.to_datetime(raw, utc=True, errors='raise')
