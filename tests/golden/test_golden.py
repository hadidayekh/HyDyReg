"""Golden cases: regression smoke tests. Each case must reproduce its reference file in tests/golden/ref/ within
tolerances that hold on every platform:

- exactly: the number of equivalence queries (NEQ), the number of modes, termination, the number of training points
  kept (shape of Xfit_);
- final arrays (mode coefficients, classifier weights): |new - ref| <= RTOL * max|ref| over each array, so that
  entries that are numerically zero (1e-14 on one platform, 3e-14 on another) compare correctly;
- result values (errors d1-d3): |new - ref| <= RTOL * |ref| + ATOL; below ATOL a value is rounding noise.

d_dot is recorded but not compared: it evaluates the true derivative with the reference labelfun on test points
that can lie on a switching surface, where one rounding step flips the true mode (affine_3modes: one point at
x0 = 1.0 vs 0.9999999999999998 moves d_dot from 4e-6 to 2e-3 with identical learned models).

The trace hashes recorded at every classify_modes call are not compared; when a case fails, the first stage whose
hash differs is printed to help locate where the run diverged (on another platform every stage differs).

    pytest tests/golden
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from cases import CASES  # noqa: E402
import capture  # noqa: E402

REF = HERE / "ref"
RTOL = 1e-6
ATOL = 1e-9
NOT_COMPARED = {"d_dot"}  # see the module docstring


def _params():
    for name, case in CASES.items():
        marks = [pytest.mark.slow] if case["slow"] else []
        yield pytest.param(name, marks=marks, id=name)


def _array_close(new, ref):
    """Same shape and |new - ref| <= RTOL * max|ref| (the array's own scale)."""
    a, b = np.asarray(new, dtype=float), np.asarray(ref, dtype=float)
    if a.shape != b.shape:
        return False, f"shape {a.shape} != {b.shape}"
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    diff = float(np.max(np.abs(a - b))) if b.size else 0.0
    return diff <= RTOL * scale + ATOL * (scale == 0), f"max |diff| = {diff:.3e}, scale {scale:.3e}"


def _first_divergence(ref, new):
    for k, (a, b) in enumerate(zip(ref["trace"], new["trace"])):
        if a != b:
            return f"trace hashes first differ at classify_modes call {k}"
    if len(ref["trace"]) != len(new["trace"]):
        return f"different number of classify_modes calls ({len(ref['trace'])} vs {len(new['trace'])})"
    return "trace hashes identical"


@pytest.mark.parametrize("name", list(_params()))
def test_golden(name, tmp_path):
    ref_file = REF / f"{name}.json"
    if not ref_file.exists():
        pytest.skip(f"no reference file for {name}")
    ref = json.loads(ref_file.read_text())
    p = capture.run_in_subprocess(name, tmp_path)
    assert p.returncode == 0, p.stderr[-3000:]
    new = json.loads((tmp_path / f"{name}.json").read_text())
    where = _first_divergence(ref, new)

    problems = []
    if new["values"]["NEQ"] != ref["values"]["NEQ"]:
        problems.append(f"NEQ {new['values']['NEQ']} != {ref['values']['NEQ']}")
    for key in ("n_modes", "terminated"):
        if new["run_info"].get(key) != ref["run_info"].get(key):
            problems.append(f"{key} {new['run_info'].get(key)} != {ref['run_info'].get(key)}")
    for k, v in ref["final"].items():
        if isinstance(v, list):
            ok, msg = _array_close(new["final"][k], v)
            if not ok:
                problems.append(f"final {k}: {msg}")
        elif isinstance(v, dict) and "shape" in v and new["final"][k]["shape"] != v["shape"]:
            problems.append(f"final {k}: shape {new['final'][k]['shape']} != {v['shape']}")
    for k, v in ref["values"].items():
        if k in NOT_COMPARED:
            continue
        w = new["values"][k]
        if not abs(w - v) <= RTOL * abs(v) + ATOL:
            problems.append(f"value {k}: {w!r} != {v!r} (rel {abs(w - v) / max(abs(v), 1e-300):.1e})")
    assert not problems, f"{name}: " + "; ".join(problems) + f" [{where}]"
