"""Hashes and JSON conversion used by the golden tool (no dependency on the implementation)."""
import hashlib
import math

import numpy as np


def sha(a):
    """SHA-256 of an array's dtype, shape and bytes (None for None)."""
    if a is None:
        return None
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def jsonable(x):
    """Converts arrays, numpy scalars, tuples, objects to JSON-compatible values (floats keep full precision)."""
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        return x if math.isfinite(x) else repr(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is None or isinstance(x, (bool, int, str)):
        return x
    if callable(x) and hasattr(x, "__name__"):
        return f"<function {x.__name__}>"
    if hasattr(x, "get_params"):
        return {"class": type(x).__name__, "params": jsonable(x.get_params())}
    return f"<{type(x).__name__}>"


def drop_timing(d):
    """Removes entries whose key mentions time (run times differ between runs)."""
    if isinstance(d, dict):
        return {k: drop_timing(v) for k, v in d.items() if "time" not in str(k).lower()}
    if isinstance(d, list):
        return [drop_timing(v) for v in d]
    return d
