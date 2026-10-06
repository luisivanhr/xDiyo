"""Arithmetic for historical expressions and already prepared feature columns."""

from copy import deepcopy
from dataclasses import dataclass
from numbers import Real

from .expressions import Expr


@dataclass(frozen=True)
class Column(Expr):
    """An existing prepared column, used by combine_features after assembly."""
    name: str


@dataclass(frozen=True)
class Constant(Expr):
    """A finite numeric constant, broadcast to every input row."""
    value: float


@dataclass(frozen=True)
class Sum(Expr):
    """Add two single-output expressions or finite numeric constants."""
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Product(Expr):
    """Multiply two single-output expressions or finite numeric constants.

    Missing/nonfinite operands and overflow produce missing values, including
    when the other operand is zero. Nest Product for more than two factors.
    """
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Difference(Expr):
    """Subtract right from left; e.g. recent average minus long-run average."""
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Ratio(Expr):
    """Divide two single-output expressions; zero denominators default to missing.

    zero_value=None returns NaN. A finite fallback applies only when both inputs
    are finite and the denominator is exactly zero. Missing inputs stay missing.
    """
    numerator: Expr
    denominator: Expr
    zero_value: float | None = None


ARITHMETIC = (Sum, Product, Difference, Ratio)


def operands(node):
    return ((node.numerator, node.denominator) if isinstance(node, Ratio)
            else (node.left, node.right))


def constant_frame(value, index):
    import numpy as np
    import pandas as pd
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError("Constants must be finite real numbers.")
    return pd.DataFrame({"value": float(value)}, index=index)


def arithmetic_frame(node, left, right):
    """Combine by exact row alignment, never by source column labels."""
    import numpy as np
    import pandas as pd
    if left.shape[1] != 1 or right.shape[1] != 1:
        raise ValueError("Arithmetic operands must each select one output column; choose an explicit period, side or rating field.")
    if not left.index.equals(right.index):
        raise ValueError("Arithmetic operands must have identical row indices.")
    a = left.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
    b = right.iloc[:, 0].to_numpy(dtype=float, na_value=np.nan)
    valid = np.isfinite(a) & np.isfinite(b)
    values = np.full(len(a), np.nan)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        if isinstance(node, Sum):
            values[valid] = a[valid] + b[valid]
        elif isinstance(node, Product):
            values[valid] = a[valid] * b[valid]
        elif isinstance(node, Difference):
            values[valid] = a[valid] - b[valid]
        else:
            fallback = node.zero_value
            if fallback is not None and (isinstance(fallback, bool) or not isinstance(fallback, Real)
                                         or not np.isfinite(fallback)):
                raise ValueError("zero_value must be None or a finite real number.")
            divide = valid & (b != 0)
            values[divide] = a[divide] / b[divide]
            if fallback is not None:
                values[valid & (b == 0)] = fallback
    values[~np.isfinite(values)] = np.nan
    return pd.DataFrame({"value": values}, index=left.index)


def combine_features(frame, definitions, *, keep_existing=True):
    """Add named arithmetic features to a prepared frame without mutating it.

    Column reads only the supplied feature frame, never labels or metadata.
    Definitions can nest Sum/Product/Difference/Ratio and constants. References address
    original columns; use nested expressions rather than forward references.
    This function does not establish temporal eligibility: its inputs must already
    be legitimate prediction features. Index, including keyed identities, survives.
    """
    import pandas as pd
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise ValueError("Supply a feature DataFrame with unique columns.")
    if not isinstance(keep_existing, bool):
        raise TypeError("keep_existing must be boolean.")
    def evaluate(node):
        if isinstance(node, Column):
            return frame[[node.name]]
        if isinstance(node, Constant):
            return constant_frame(node.value, frame.index)
        if isinstance(node, Real):
            return constant_frame(node, frame.index)
        if isinstance(node, ARITHMETIC):
            left, right = operands(node)
            return arithmetic_frame(node, evaluate(left), evaluate(right))
        raise TypeError("combine_features takes Column, Constant and arithmetic expressions.")
    outputs = {}
    for name, node in definitions.items():
        if not isinstance(name, str) or not name or name in frame.columns:
            raise ValueError("Derived feature names must be nonempty and must not replace input columns.")
        outputs[name] = evaluate(node).iloc[:, 0]
    derived = pd.DataFrame(outputs, index=frame.index)
    result = pd.concat([frame.copy(deep=True), derived], axis=1) if keep_existing else derived
    result.attrs = deepcopy(frame.attrs)
    result.attrs.setdefault("derived_features", {}).update({name: repr(node) for name, node in definitions.items()})
    return result


class IdentityIndicators:
    """Train-fitted team/league identity indicators, independent of row layout.

    Fit on training metadata only, then transform any identically named columns.
    For matches use home_id/away_id; for team rows use team_id. Missing and unseen
    values become all zeros. No implicit fitting or discovery at transform time.
    This small DataFrame helper is not a sklearn estimator. Keep its fitted state
    alongside the model when using it for future prediction.
    """
    def __init__(self, columns=("source_league",), prefixes=None):
        self.columns = (columns,) if isinstance(columns, str) else tuple(columns)
        self.prefixes = dict(prefixes or {})
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ValueError("Choose distinct identity columns.")

    def fit(self, metadata):
        import pandas as pd
        categories, names = {}, []
        for column in self.columns:
            # Object comparison preserves unsigned IDs larger than signed int64.
            values = metadata[column].astype(object)
            categories[column] = tuple(pd.unique(values[values.notna()]))
            prefix = self.prefixes.get(column, column)
            names.extend(f"{prefix}::{value}" for value in categories[column])
        if len(set(names)) != len(names):
            raise ValueError("Identity category labels produce duplicate column names; use distinct prefixes/types.")
        self.categories_, self.feature_names_ = categories, tuple(names)
        return self

    def transform(self, metadata):
        import pandas as pd
        if not hasattr(self, "categories_"):
            raise ValueError("Fit IdentityIndicators on training metadata before transforming.")
        output = {}
        for column, categories in self.categories_.items():
            values = metadata[column].astype(object)
            prefix = self.prefixes.get(column, column)
            for category in categories:
                output[f"{prefix}::{category}"] = values.eq(category).fillna(False).astype("float32")
        return pd.DataFrame(output, index=metadata.index)
