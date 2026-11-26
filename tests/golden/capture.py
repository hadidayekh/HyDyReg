"""Record golden reference files.

    python tests/golden/capture.py G1 G2 --out DIR      # each case in its own process, one JSON file per case
    python tests/golden/capture.py all --out DIR
    python tests/golden/capture.py --one G1 --out DIR   # internal: run one case in this process
    python tests/golden/capture.py --row-settings FILE   # record the settings of the 26 rows

Re-record tests/golden/ref/ after a deliberate change of results, and log the change in CHANGELOG.md.
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("MPLBACKEND", "Agg")

COMPARED = ["d1_avg", "d2_avg", "d3_avg", "d1_max", "d2_max", "d3_max", "NEQ", "d_dot"]


def environment():
    from importlib import metadata
    return dict(platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
                **{k: metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "casadi")})


def record_case(name):
    from cases import CASES
    from hashing import sha, jsonable, drop_timing
    import adapter
    case = CASES[name]
    t0 = time.time()
    values, model, run_info, trace = adapter.run(case["row"], case["overrides"])
    seconds = time.time() - t0
    c = getattr(model, "classifier_", None)
    final = dict(Omegas=jsonable(model.Omegas_), betas=jsonable(model.betas_))
    for attr in ("coef_", "intercept_", "W_", "B_", "order_"):
        if c is not None and hasattr(c, attr):
            final["classifier_" + attr] = jsonable(getattr(c, attr))
    for attr in ("Xfit_", "X_dot_fit_", "modesfit_"):
        a = getattr(model, attr)
        final[attr] = dict(shape=list(a.shape), sha=sha(a))
    return dict(case=name, row=case["row"], overrides=jsonable(case["overrides"]), environment=environment(),
                settings=jsonable(adapter.row_settings(case["row"])), trace=trace, final=final,
                values={k: values[k] for k in COMPARED}, run_info=jsonable(drop_timing(run_info)), seconds=seconds)


def run_in_subprocess(name, out):
    cmd = [sys.executable, str(HERE / "capture.py"), "--one", name, "--out", str(out)]
    return subprocess.run(cmd, capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--out")
    ap.add_argument("--one")
    ap.add_argument("--row-settings")
    a = ap.parse_args()
    if a.one:
        rec = record_case(a.one)
        Path(a.out).mkdir(parents=True, exist_ok=True)
        with open(Path(a.out) / f"{a.one}.json", "w") as f:
            json.dump(rec, f, indent=1)
        return
    if a.row_settings:
        import adapter
        from hashing import jsonable
        from hydyreg.benchmarks.rows import ROW_IDS
        out = {r: jsonable(adapter.row_settings(r)) for r in ROW_IDS}
        with open(a.row_settings, "w") as f:
            json.dump(out, f, indent=1)
        return
    from cases import CASES
    names = list(CASES) if a.cases == ["all"] else a.cases
    for n in names:
        t = time.time()
        p = run_in_subprocess(n, a.out)
        print(f"{n}: {'ok' if p.returncode == 0 else 'FAILED'} ({time.time() - t:.0f} s)", flush=True)
        if p.returncode:
            print(p.stderr[-3000:])


if __name__ == "__main__":
    main()
