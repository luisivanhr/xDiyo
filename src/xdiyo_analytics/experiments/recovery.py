"""Data-only recovery bundles and automatic execution identities (no model pickle)."""

from dataclasses import fields, is_dataclass
from datetime import datetime, date, time, timedelta
from functools import partial
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
from dateutil.relativedelta import weekday as relative_weekday


# Fixed public constructors only: frequency metadata cannot name imports or
# arbitrary user-defined offset classes.
_FREQUENCY_OFFSETS = {name: getattr(pd.offsets, name) for name in (
    "DateOffset", "Day", "Hour", "Minute", "Second", "Milli", "Micro", "Nano",
    "BusinessDay", "BusinessHour", "CustomBusinessDay", "CustomBusinessHour",
    "MonthBegin", "MonthEnd", "BusinessMonthBegin", "BusinessMonthEnd",
    "CustomBusinessMonthBegin", "CustomBusinessMonthEnd", "SemiMonthBegin", "SemiMonthEnd",
    "QuarterBegin", "QuarterEnd", "BQuarterBegin", "BQuarterEnd", "YearBegin", "YearEnd",
    "BYearBegin", "BYearEnd", "Week", "WeekOfMonth", "LastWeekOfMonth", "Easter", "FY5253", "FY5253Quarter",
)}


def _pack_frequency(frequency):
    if frequency is None:
        return None
    name = type(frequency).__name__
    if _FREQUENCY_OFFSETS.get(name) is not type(frequency):
        raise TypeError(f"Recovery cannot encode {name} frequency; use data-only pandas offsets.")
    parameters = frequency.kwds.copy()
    if type(parameters.get("weekday")) is relative_weekday:
        weekday = parameters["weekday"]
        parameters["weekday"] = {"weekday": weekday.weekday, "n": weekday.n}
    if type(parameters.get("offset")) is timedelta:
        offset = parameters["offset"]
        parameters["offset"] = {"days": offset.days, "seconds": offset.seconds, "microseconds": offset.microseconds}
    for key in ("start", "end"):
        if key in parameters:
            parameters[key] = [item.isoformat() for item in parameters[key]]
    if "calendar" in parameters:
        calendar = parameters["calendar"]
        # A supplied calendar can differ from the offset's exposed weekmask.
        parameters["calendar"] = {"weekmask": calendar.weekmask.tolist(), "holidays": calendar.holidays}
    return {"name": name, "n": frequency.n, "normalize": frequency.normalize, "kwds": pack(parameters)}


def _unpack_frequency(value):
    if value is None:
        return None
    constructor = _FREQUENCY_OFFSETS.get(value["name"])
    if constructor is None:
        raise ValueError(f"Unsupported recovery frequency {value['name']!r}.")
    parameters = unpack(value["kwds"])
    if isinstance(parameters.get("weekday"), dict):
        parameters["weekday"] = relative_weekday(**parameters["weekday"])
    if isinstance(parameters.get("offset"), dict):
        parameters["offset"] = timedelta(**parameters["offset"])
    for key in ("start", "end"):
        if key in parameters:
            parameters[key] = tuple(time.fromisoformat(item) for item in parameters[key])
    if "calendar" in parameters:
        parameters["calendar"] = np.busdaycalendar(**parameters["calendar"])
    return constructor(n=value["n"], normalize=value["normalize"], **parameters)


def _pack_dtype(dtype):
    if isinstance(dtype, pd.StringDtype):
        return {"kind": "string", "storage": dtype.storage, "na_value": pack(getattr(dtype, "na_value", pd.NA))}
    if isinstance(dtype, pd.CategoricalDtype):
        return {"kind": "categorical", "categories": pack(dtype.categories), "ordered": dtype.ordered}
    return str(dtype)


def _unpack_dtype(dtype):
    # Older bundles describe all dtypes by string and remain readable.
    if isinstance(dtype, str):
        return dtype
    if dtype["kind"] == "string":
        na_value = unpack(dtype["na_value"])
        if na_value is pd.NA:
            return pd.StringDtype(storage=dtype["storage"])
        # pandas 2.x cannot represent pandas 3's NaN string sentinel. Preserve
        # storage, values and the null mask while adapting missing values to pd.NA.
        if (isinstance(na_value, float) and np.isnan(na_value) and
                "na_value" not in inspect.signature(pd.StringDtype).parameters):
            return pd.StringDtype(storage=dtype["storage"])
        return pd.StringDtype(storage=dtype["storage"], na_value=na_value)
    if dtype["kind"] == "categorical":
        return pd.CategoricalDtype(unpack(dtype["categories"]), ordered=dtype["ordered"])
    raise ValueError(f"Unknown recovery dtype {dtype['kind']!r}.")


def _pack_series(value, *, include_attrs=True):
    result = {"@": "series", "index": pack(value.index), "name": pack(value.name),
              "values": pack(value.tolist()), "dtype": _pack_dtype(value.dtype)}
    if include_attrs:
        result["attrs"] = pack(value.attrs)
    return result


def pack(value):
    """Encode library results, frames and figures without executable deserialization.

    Fitted adapters are deliberately omitted. Predictions, selections, report
    artifacts, history and configuration remain usable after a process restart.
    Frame/Series attrs use the same data-only encoding, including nested metadata.
    """
    if value is pd.NA:
        return {"@": "NA"}
    if value is pd.NaT:
        return {"@": "NaT"}
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return {"@": "numpy_temporal", "dtype": str(value.dtype), "value": int(value.astype("int64"))}
    if isinstance(value, np.generic):
        return pack(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"@": "float", "value": str(value)}
    if isinstance(value, date) and not isinstance(value, datetime):
        return {"@": "date", "value": value.isoformat()}
    if isinstance(value, (pd.Timestamp, datetime)):
        return {"@": "timestamp", "value": value.isoformat()}
    if isinstance(value, (pd.Timedelta,)):
        return {"@": "timedelta", "value": value.value}
    if isinstance(value, Path):
        return {"@": "path", "value": str(value)}
    if isinstance(value, pd.MultiIndex):
        return {"@": "multiindex", "levels": pack(list(value.levels)), "codes": pack(list(value.codes)),
                "names": pack(value.names), "sortorder": value.sortorder}
    if isinstance(value, pd.RangeIndex):
        return {"@": "rangeindex", "start": value.start, "stop": value.stop,
                "step": value.step, "name": pack(value.name)}
    if isinstance(value, pd.Index):
        result = {"@": "index", "values": pack(value.tolist()), "dtype": _pack_dtype(value.dtype), "name": pack(value.name)}
        if isinstance(value, (pd.DatetimeIndex, pd.TimedeltaIndex)):
            result["freq"] = _pack_frequency(value.freq)
        return result
    if isinstance(value, pd.DataFrame):
        # Validate attrs before pandas can deepcopy them while slicing columns.
        # The frame owns this metadata; inherited column attrs would duplicate it.
        attrs = pack(value.attrs)
        return {"@": "frame", "index": pack(value.index), "columns": pack(value.columns),
                "series": [_pack_series(value.iloc[:, i].reset_index(drop=True), include_attrs=False)
                           for i in range(len(value.columns))], "attrs": attrs}
    if isinstance(value, pd.Series):
        return _pack_series(value)
    if isinstance(value, np.ndarray):
        # Temporal tolist() can discard units or produce unsupported Python
        # timedeltas. Integer conversion preserves ticks, NaT and byte order.
        values = value.astype("int64").tolist() if value.dtype.kind in "mM" else value.tolist()
        return {"@": "array", "values": pack(values), "dtype": str(value.dtype), "shape": list(value.shape)}
    if isinstance(value, tuple):
        return {"@": "tuple", "values": [pack(item) for item in value]}
    if isinstance(value, list):
        return [pack(item) for item in value]
    if isinstance(value, dict):
        return {"@": "dict", "items": [[pack(key), pack(item)] for key, item in value.items()]}
    if type(value).__module__.startswith("plotly.") and callable(getattr(value, "to_json", None)):
        return {"@": "plotly", "value": value.to_json()}
    if is_dataclass(value) and type(value).__module__.startswith("xdiyo_analytics."):
        omitted = {"model"} if type(value).__name__ in {"FoldResult", "FittedModel"} else set()
        return {"@": "record", "module": type(value).__module__, "name": type(value).__name__,
                "fields": {f.name: (None if f.name in omitted else pack(getattr(value, f.name)))
                           for f in fields(value) if f.init}}
    raise TypeError(f"Recovery cannot encode {type(value).__name__}; use data-only report artifacts.")


def unpack(value):
    if isinstance(value, list):
        return [unpack(item) for item in value]
    if not isinstance(value, dict):
        return value
    tag = value["@"]
    if tag == "NA":
        return pd.NA
    if tag == "NaT":
        return pd.NaT
    if tag == "float":
        return float(value["value"])
    if tag == "numpy_temporal":
        dtype = np.dtype(value["dtype"])
        if dtype.kind not in "mM":
            raise ValueError("NumPy temporal scalar requires a datetime64 or timedelta64 dtype.")
        if type(value["value"]) is not int:
            raise ValueError("NumPy temporal scalar requires an integer tick count.")
        return np.asarray(value["value"], dtype="int64").astype(dtype)[()]
    if tag == "date":
        return date.fromisoformat(value["value"])
    if tag == "timestamp":
        # Legacy date-only timestamp records retain their original decoding.
        return pd.Timestamp(value["value"])
    if tag == "timedelta":
        return pd.Timedelta(value["value"], unit="ns")
    if tag == "path":
        return Path(value["value"])
    if tag == "tuple":
        return tuple(unpack(item) for item in value["values"])
    if tag == "dict":
        return {unpack(key): unpack(item) for key, item in value["items"]}
    if tag == "array":
        values = unpack(value["values"])
        if np.dtype(value["dtype"]).kind == "O":
            # tolist() preserves array rank, but each object cell may itself be
            # a sequence. Follow only the recorded shape, never infer cell axes.
            result = np.empty(value["shape"], dtype=object)
            def assign(items, index):
                if len(index) == result.ndim:
                    result[index] = items
                    return
                if not isinstance(items, list) or len(items) != result.shape[len(index)]:
                    raise ValueError("Object array values do not match the recorded shape.")
                for position, item in enumerate(items):
                    assign(item, (*index, position))
            assign(values, ())
            return result
        dtype = np.dtype(value["dtype"])
        if dtype.kind in "mM":
            counts = np.asarray(values)
            if counts.dtype.kind in "iu":
                # astype also supports unitless temporal NaT; direct integer
                # construction of a unitless datetime64 is rejected by NumPy.
                return counts.astype(dtype).reshape(value["shape"])
        return np.array(values, dtype=dtype).reshape(value["shape"])
    if tag == "rangeindex":
        return pd.RangeIndex(value["start"], value["stop"], value["step"], name=unpack(value["name"]))
    if tag == "multiindex":
        if "levels" in value:
            return pd.MultiIndex(levels=unpack(value["levels"]), codes=unpack(value["codes"]),
                                 names=unpack(value["names"]), sortorder=value.get("sortorder"))
        # Legacy tuple-based indexes cannot recover metadata never stored.
        return pd.MultiIndex.from_tuples(unpack(value["values"]), names=unpack(value["names"]))
    if tag == "index":
        index = pd.Index(unpack(value["values"]), dtype=_unpack_dtype(value["dtype"]), name=unpack(value["name"]), tupleize_cols=False)
        if "freq" in value:
            if not isinstance(index, (pd.DatetimeIndex, pd.TimedeltaIndex)):
                raise ValueError("Frequency metadata requires a datetime or timedelta index.")
            frequency = _unpack_frequency(value["freq"])
            if isinstance(index, pd.TimedeltaIndex) and frequency is not None and not isinstance(frequency, (pd.offsets.Tick, pd.offsets.Day)):
                raise ValueError("Timedelta index frequency must be a fixed offset.")
            # Restore recorded metadata without inferring/regenerating values.
            # pandas public freq validation rejects some of its own generated
            # business-offset ranges and singleton calendar offsets.
            index._data._freq = frequency
        return index
    if tag == "series":
        dtype = (pd.CategoricalDtype(unpack(value["categories"]), value["ordered"])
                 if "categories" in value else _unpack_dtype(value["dtype"]))
        series = pd.Series(unpack(value["values"]), index=unpack(value["index"]), dtype=dtype, name=unpack(value["name"]))
        series.attrs = unpack(value["attrs"]) if "attrs" in value else {}
        return series
    if tag == "frame":
        series = [unpack(item) for item in value["series"]]
        frame = pd.concat(series, axis=1) if series else pd.DataFrame(index=range(len(unpack(value["index"]))))
        frame.index, frame.columns = unpack(value["index"]), unpack(value["columns"])
        # Schema-1 bundles written before attrs support remain readable.
        frame.attrs = unpack(value["attrs"]) if "attrs" in value else {}
        return frame
    if tag == "plotly":
        import plotly.io as pio
        return pio.from_json(value["value"])
    if tag == "record":
        # Only library dataclasses, never arbitrary imported constructors.
        module = value["module"]
        if not module.startswith("xdiyo_analytics.") or any(part.startswith("_") for part in module.split(".")):
            raise ValueError("Unsupported recovery record module.")
        cls = getattr(importlib.import_module(module), value["name"])
        if not is_dataclass(cls) or cls.__module__ != module:
            raise ValueError("Unsupported recovery record type.")
        return cls(**{key: unpack(item) for key, item in value["fields"].items()})
    raise ValueError(f"Unknown recovery tag {tag!r}.")


def dump_bundle(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump({"schema": 1, "payload": pack(payload)}, stream, ensure_ascii=False, allow_nan=False)


def load_bundle(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if value.get("schema") != 1:
        raise ValueError("Unsupported recovery schema.")
    return unpack(value["payload"])


def signature(value, _active=None):
    """Describe executable configuration without constructing a model.

    Custom callable objects may implement cache_key(). Factory code, defaults,
    closure values and referenced scalar globals are included. External resources
    or mutable services still need a revision in Candidate.config/cache_key().
    Observers are deliberately excluded by the caller: they do not define fits.
    """
    active = set() if _active is None else _active
    if id(value) in active:
        return {"recursive": f"{type(value).__module__}.{type(value).__qualname__}"}
    active = active | {id(value)}
    sub = lambda item: signature(item, active)
    if isinstance(value, dict):
        return {"mapping": [[sub(k), sub(v)] for k, v in value.items()]}
    if isinstance(value, (list, tuple)):
        return [sub(v) for v in value]
    if isinstance(value, types.ModuleType):
        return {"module": value.__name__, "version": str(getattr(value, "__version__", ""))}
    if isinstance(value, partial):
        return {"partial": sub(value.func), "args": sub(value.args), "keywords": sub(value.keywords)}
    if inspect.ismethod(value):
        return {"method": sub(value.__func__), "self": sub(value.__self__)}
    if inspect.isfunction(value):
        import marshal
        def semantic_code(code):
            # Notebook execution counters/filenames and moved source lines are
            # not parameter changes. Preserve bytecode/constants, not locations.
            return code.replace(co_filename="", co_firstlineno=1, co_linetable=b"",
                                co_consts=tuple(semantic_code(item) if isinstance(item, types.CodeType) else item
                                                for item in code.co_consts))
        return {"function": value.__qualname__, "code": hashlib.sha256(marshal.dumps(semantic_code(value.__code__))).hexdigest(),
                "defaults": sub(value.__defaults__), "kwdefaults": sub(value.__kwdefaults__),
                "closure": [sub(cell.cell_contents) for cell in (value.__closure__ or ())],
                "globals": {name: sub(value.__globals__[name]) for name in value.__code__.co_names
                            if name in value.__globals__ and name != value.__name__}}
    if inspect.isclass(value) or inspect.isbuiltin(value):
        module = sys.modules.get(value.__module__)
        path = getattr(module, "__file__", None)
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest() if path and Path(path).is_file() else None
        result = {"type": f"{value.__module__}.{value.__qualname__}", "source": digest,
                  "version": str(getattr(sys.modules.get(value.__module__.split(".")[0]), "__version__", ""))}
        if inspect.isclass(value) and digest is None:
            result["members"] = {name: sub(item.__func__ if isinstance(item, (staticmethod, classmethod)) else item)
                                 for name, item in vars(value).items()
                                 if inspect.isfunction(item) or isinstance(item, (staticmethod, classmethod))}
        return result
    if callable(getattr(value, "cache_key", None)):
        return {"custom": sub(type(value)), "key": sub(value.cache_key())}
    if is_dataclass(value):
        return {"type": sub(type(value)), "fields": {f.name: sub(getattr(value, f.name)) for f in fields(value)
                if f.init and f.name not in {"observer"}}}
    if callable(getattr(value, "get_params", None)):
        return {"type": sub(type(value)), "parameters": sub(value.get_params(deep=False))}
    try:
        return pack(value)
    except TypeError as error:
        raise TypeError(f"Cannot identify {type(value).__name__} for recovery; implement cache_key() returning configuration data.") from error


def execution_key(*values):
    from .store import configuration_hash
    # Local library changes invalidate reuse without asking users for versions.
    root = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    return configuration_hash({"source": source.hexdigest(), "python": sys.version,
                               "numpy": np.__version__, "pandas": pd.__version__,
                               "inputs": [signature(value) for value in values]})
