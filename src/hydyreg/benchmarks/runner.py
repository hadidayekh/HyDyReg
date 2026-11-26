"""Run any subset of the 26 thesis experiments (Tables 7.2-7.5).

Examples (from any directory, once the package is installed):
    python -m hydyreg.benchmarks.runner --list
    python -m hydyreg.benchmarks.runner incr/isolette              # one row
    python -m hydyreg.benchmarks.runner 1-8                        # rows by index (see --list)
    python -m hydyreg.benchmarks.runner 'incr/*' parametric-4      # globs and short names
    python -m hydyreg.benchmarks.runner all -j 4 --out ~/runs/a
    python -m hydyreg.benchmarks.runner --out ~/runs/a --table --compare thesis

Each row runs in its own fresh Python process; its random draws come from LearnConfig.seed (0 unless --set
seed=K or --seeds). With --seeds 0-4, each seed runs into <out>/seed<K>/ and --summary aggregates them. Results go to
<out>/results/<row>.json, logs to <out>/logs/<row>.log, and the usual experi-plots/ and
experi-info/ outputs to <out>/. Rows that already have a result are skipped unless --force,
so an interrupted run can be resumed with the same command.

The thesis numbers are produced by the tagged version `thesis-reproducible` (see the README there).
This version contains later improvements of the tool, so --compare thesis shows how far results moved.

Thread count: --threads N sets the OpenBLAS threads of the worker processes.
"""
import argparse
import fnmatch
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime

PKG = os.path.dirname(os.path.abspath(__file__))
THESIS_REF = os.path.join(PKG, "thesis_rows.json")
CORE_PACKAGES = ("numpy", "scipy", "scikit-learn", "casadi", "matplotlib")
# Reference environment (requirements-lock.txt), in which the golden references and the reported results were
# recorded. Other versions are supported; they can change results in the last digits (more on fragile rows).
REFERENCE_ENV = {"python": "3.12", "numpy": "2.5.5", "scipy": "1.18.1", "scikit-learn": "1.9.1", "casadi": "3.8.2",
                 "matplotlib": "3.11.2"}

# Rough wall times (minutes, one row, recent laptop), for planning only.
_MIN = {"opti": 2, "incr": 2, ("opti", 2): 1, ("opti", 3): 1, ("opti", 4): 3, ("opti", 5): 1,
        ("incr", 2): 2, ("incr", 3): 2, ("incr", 4): 2, ("incr", 5): 2, ("incr", 10): 2, ("incr", 20): 3}

# Thesis table order: 7.2 opti other, 7.3 incr other, 7.4 opti parametric, 7.5 incr parametric.
from .rows import ROW_IDS  # noqa: E402
TABLES = [("Table 7.2: optimization-based, other systems", ROW_IDS[0:8]),
          ("Table 7.3: incremental, other systems", ROW_IDS[8:16]),
          ("Table 7.4: optimization-based, parametric", ROW_IDS[16:20]),
          ("Table 7.5: incremental, parametric", ROW_IDS[20:26])]


COLS = ["d1_avg", "d2_avg", "d3_avg", "d1_max", "d2_max", "d3_max", "t1", "t2", "t3", "NEQ", "d_dot"]
COMPARED = ["d1_avg", "d2_avg", "d3_avg", "d1_max", "d2_max", "d3_max", "NEQ", "d_dot"]


def minutes(row_id):
    m, s = row_id.split("/")
    return _MIN[(m, int(s.split("-")[1]))] if s.startswith("parametric") else _MIN[m]


def fname(row_id):
    return row_id.replace("/", "__")


# ----------------------------------------------------------------------------- selection

def select(tokens):
    """Resolve selectors (all, 3, 1-8, incr/*, */parametric-*, opti/ham, ham, incr) to row ids,
    in table order."""
    chosen = set()
    for tok in tokens:
        hits = []
        if tok == "all":
            hits = ROW_IDS
        elif tok.isdigit() or (tok.count("-") == 1 and all(p.isdigit() for p in tok.split("-"))):
            a, b = map(int, tok.split("-")) if "-" in tok else (int(tok), int(tok))
            if not (1 <= a <= b <= len(ROW_IDS)):
                sys.exit(f"Row range out of bounds: {tok} (rows are 1-{len(ROW_IDS)})")
            hits = ROW_IDS[a - 1:b]
        else:
            pats = [tok] if "/" in tok else [f"*/{tok}", f"{tok}/*"]
            hits = [r for r in ROW_IDS if any(fnmatch.fnmatchcase(r, p) for p in pats)]
        if not hits:
            sys.exit(f"Selector matches no row: {tok!r} (see --list)")
        chosen.update(hits)
    return [r for r in ROW_IDS if r in chosen]


def print_list():
    print(f"{'#':>3}  {'row id':32s} {'~min':>5}  thesis table")
    for i, r in enumerate(ROW_IDS, 1):
        tab = next(t for t, ids in TABLES if r in ids).split(":")[0]
        print(f"{i:3d}  {r:32s} {minutes(r):5d}  {tab}")
    print(f"\nTotal ~{sum(minutes(r) for r in ROW_IDS) / 60:.1f} h sequentially. "
          "Selectors: all, 5, 1-8, 'incr/*', '*/parametric-*', opti/ham, ham, parametric-4, incr.")


# ----------------------------------------------------------------------------- environment checks

def child_env(args):
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONUNBUFFERED"] = "1"
    if args.threads:
        for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"):
            env[k] = str(args.threads)
    return env


def blas_threads(env):
    """Ask a child process (same env) how many threads OpenBLAS actually uses."""
    code = ("import json, numpy, scipy.linalg, threadpoolctl; print(json.dumps(sorted({(i.get('internal_api'), "
            "i.get('num_threads')) for i in threadpoolctl.threadpool_info() if i.get('user_api') == 'blas'})))")
    try:
        out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception as e:  # report, don't fail
        return f"unknown ({type(e).__name__})"


def platform_report():
    from importlib import metadata
    lines, ok = [], True
    rosetta = None
    if sys.platform == "darwin":
        try:
            rosetta = subprocess.run(["sysctl", "-n", "sysctl.proc_translated"], capture_output=True,
                                     text=True).stdout.strip() == "1"
        except Exception:
            pass
    lines.append(f"python {platform.python_version()} {platform.machine()}"
                 + (" (Rosetta)" if rosetta else "") + f" on {sys.platform}")
    versions = {"python": ".".join(platform.python_version_tuple()[:2]),
                **{k: metadata.version(k) for k in CORE_PACKAGES}}
    lines.append("  " + ", ".join(f"{k} {versions[k]}" for k in CORE_PACKAGES))
    differ = [f"{k} {versions[k]} (reference {v})" for k, v in REFERENCE_ENV.items() if versions[k] != v]
    if differ:
        ok = False
        lines.append("  ! not the reference environment (requirements-lock.txt): " + ", ".join(differ)
                     + "; results can differ from the reported ones in the last digits, more on fragile rows")
    return ok, lines



# ----------------------------------------------------------------------------- tables in the thesis format

def print_experi_data(DATA, both=False, jin=False):
    # Data is made of exp_name, RD1, RD2, RD3, t_best_C, t_active, t_EQ, count, RDdot, RDmerge, tmerge, RDpwa, tpwa
    s = ''
    for j in range(len(DATA)):
        d = DATA[j]
        if not both:
            exp_name, RD1, RD2, RD3, t1, t2, t3, count, RD_dot, RD_merge, t_merge, RD_piecelinear, t_piecelinear = d
            s += d[0] + ' & '
            for i in range(1, 4): #rd1,rd2,rd3
                s += f'{d[i]:.5f} & '
            for i in range(4, 7): #t1,t2,t3
                s += f'{d[i]:.2f} & '
            s += str(d[7]) + ' & ' #count
            s += f'{d[8]:.1e} ' #rddot
            if jin:
                s += f'& {d[9]:.1e} & ' #rdmerge
                s += f'{d[11]:.1e} & ' #rdpl
                s += f'{d[10]:.1f} & ' #tmerge
                s += f'{d[12]:.1f}'    #tpl
            s += '\\\\'
        else:
            exp_name, RD1, RD2, RD3, RD1_max, RD2_max, RD3_max, t1, t2, t3, count, RD_dot, RD_merge, t_merge, RD_piecelinear, t_piecelinear = d
            s += d[0] + ' & '
            for i in range(1, 4): #rd1,rd2,rd3
                s += f'{d[i]:.5f} & '
            for i in range(4, 7): #rd1,rd2,rd3
                s += f'{d[i]:.5f} & '
            for i in range(7, 10): #t1,t2,t3
                s += f'{d[i]:.2f} & '
            s += str(d[10]) + ' & ' #count
            s += f'{d[11]:.1e} ' #rddot
            if jin:
                s += f'& {d[12]:.1e} & ' #rdmerge
                s += f'{d[14]:.1e} & ' #rdpl
                s += f'{d[13]:.1f} & ' #tmerge
                s += f'{d[15]:.1f}\\\\'    #tpl
            s += '\\\\'
        s += '\n'
    print(s)
    return s


# ----------------------------------------------------------------------------- worker (one row)

def to_jsonable(x):
    import numpy as np
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    return x


def parse_overrides(items):
    """['optimization.n_starts=10', 'design.noise_std=1e-3'] -> {'optimization.n_starts': 10, ...}; values are
    Python literals (numbers, strings, lists, True/False/None), true/false as booleans; anything else is kept as a
    string."""
    import ast
    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        v = v.strip()
        if v.lower() in ("true", "false"):  # TOML / command-line spelling of booleans
            out[k.strip()] = v.lower() == "true"
            continue
        try:
            out[k.strip()] = ast.literal_eval(v)
        except (ValueError, SyntaxError):
            out[k.strip()] = v
    return out


def apply_overrides(row, overrides):
    """Applies dotted overrides to a Row: 'design.<field>' to its sampling design, anything else to its
    LearnConfig ('method', 'optimization.n_starts', ...). Two keys of the stress tier modify the design:
    'stress.train_fraction=F' keeps the first ceil(F * n) training initial conditions, and
    'stress.relative_noise=R' sets design.noise_std = R * median ||x|| of the noise-free training trajectories."""
    import math
    import numpy as np
    from ..learning.config import _replace_nested
    from dataclasses import replace
    overrides = dict(overrides)
    frac, rel = overrides.pop("stress.train_fraction", None), overrides.pop("stress.relative_noise", None)
    if frac is not None:
        row.design.x0s = list(row.design.x0s)[:max(1, math.ceil(float(frac) * len(row.design.x0s)))]
    if rel is not None:
        d = row.design
        tr = row.system().simulate_trajectory(d.time_step, d.time_range, d.x0s)
        row.design.noise_std = float(rel) * float(np.median(np.linalg.norm(np.concatenate(tr.x_list), axis=1)))
    cfg_updates = {k: v for k, v in overrides.items() if not k.startswith("design.")}
    for k, v in overrides.items():
        if k.startswith("design."):
            _replace_nested(row.design, {k[len("design."):]: v})
    if cfg_updates:
        _replace_nested(row.config, cfg_updates)
        row.config = replace(row.config)  # validates again
    return row


def worker(row_id, out, threads_requested, overrides=None, plots=False, dpi=150):
    """Run one row in this process; cwd = out."""
    from .rows import get_row, run_row
    from ..learning.reporting import Reporter
    os.chdir(out)
    row = apply_overrides(get_row(row_id), overrides or {})
    t0 = time.time()
    result = run_row(row, reporter=Reporter(row.name, write_on=True, with_plots=plots, dpi=dpi))
    wall = time.time() - t0

    from importlib import metadata
    from threadpoolctl import threadpool_info
    from ..systems import SIM_TOTALS
    rec = {
        "row": row_id, "index": ROW_IDS.index(row_id) + 1,
        "values": dict(zip(COLS, to_jsonable(result.to_row()))),
        "overrides": to_jsonable(overrides or {}),
        "wall_time_s": wall,
        # terminated: the last equivalence query returned no counterexample (the success criterion)
        "run_info": to_jsonable(result.info),
        "simulation": dict(SIM_TOTALS),
        "threads_requested": threads_requested,
        "blas": sorted({(i.get("internal_api"), i.get("num_threads"), i.get("architecture"))
                        for i in threadpool_info() if i.get("user_api") == "blas"}),
        "python": f"{platform.python_version()} {platform.machine()}", "platform": platform.platform(),
        "packages": {k: metadata.version(k) for k in ("numpy", "scipy", "scikit-learn", "casadi")},
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    os.makedirs("results", exist_ok=True)
    tmp = os.path.join("results", fname(row_id) + ".json.tmp")
    with open(tmp, "w") as f:
        json.dump(rec, f, indent=1)
    os.replace(tmp, os.path.join("results", fname(row_id) + ".json"))


# ----------------------------------------------------------------------------- results, tables, comparison

def load_results(out):
    d = os.path.join(out, "results")
    res = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".json"):
                r = json.load(open(os.path.join(d, f)))
                res[r["row"]] = r
    return res


def load_timeouts(out):
    """Rows stopped by --timeout: {row: minutes}. Marker files results/<row>.timeout."""
    d = os.path.join(out, "results")
    res = {}
    for r in ROW_IDS:
        f = os.path.join(d, fname(r) + ".timeout")
        if os.path.exists(f):
            res[r] = float(open(f).read().strip() or 0)
    return res


def terminated(rec):
    """True/False if known; for results written before `run_info` existed, NEQ < 100."""
    info = rec.get("run_info") or {}
    if "terminated" in info:
        return info["terminated"]
    return rec["values"]["NEQ"] < 100


def print_tables(out):
    res = load_results(out)
    for title, ids in TABLES:
        rows = [[r] + [res[r]["values"][c] for c in COLS] + [-1] * 4 for r in ids if r in res]
        timeouts = load_timeouts(out)
        stopped = [r for r in ids if r not in res and r in timeouts]
        missing = [r for r in ids if r not in res and r not in timeouts]
        print(f"\n### {title} ({len(rows)}/{len(ids)} rows)")
        print("name & d1 d2 d3 (avg) & d1 d2 d3 (max) & t1 t2 t3 & NEQ & d_dot")
        if rows:
            print_experi_data(rows, both=True)
        for r in stopped:
            print(f"{r}: did not terminate (stopped by --timeout after {timeouts[r]:g} min)")
        for r in ids:
            if r in res and terminated(res[r]) is False:
                print(f"{r}: did not terminate (last equivalence query still has counterexamples, "
                      f"NEQ = {res[r]['values']['NEQ']})")
        if missing:
            print("missing: " + ", ".join(missing))


def fmt_cell(col, v):
    return str(int(v)) if col == "NEQ" else (f"{v:.1e}" if col == "d_dot" else f"{v:.5f}")


def compare(out, ref, printed=False):
    """Compare RD/NEQ/d_dot columns with the thesis (at printed precision) or with another
    results directory (exactly, or at printed precision if printed=True). Timing columns are
    ignored. Returns the number of differing rows."""
    res = load_results(out)
    if ref == "thesis":
        refvals = {k: v for k, v in json.load(open(THESIS_REF))["rows"].items()}
        same = lambda c, a, b: fmt_cell(c, a) == fmt_cell(c, b)
        label = "thesis (printed precision)"
    else:
        refvals = {k: v["values"] for k, v in load_results(os.path.expanduser(ref)).items()}
        same = (lambda c, a, b: fmt_cell(c, a) == fmt_cell(c, b)) if printed else (lambda c, a, b: a == b)
        label = f"{ref} ({'printed precision' if printed else 'exact'})"
    print(f"\n### Comparison of {out} with {label}")
    n_same = n_diff = 0
    timeouts = load_timeouts(out)
    for r in ROW_IDS:
        if r not in res:
            if r in timeouts:
                print(f"{r:32s} TIMEOUT  did not terminate within {timeouts[r]:g} min")
            continue
        if r not in refvals:
            print(f"{r:32s} no reference")
            continue
        ours, theirs = res[r]["values"], refvals[r]
        diffs = [f"{c} {fmt_cell(c, theirs[c])}->{fmt_cell(c, ours[c])}" if ref == "thesis"
                 else f"{c} {theirs[c]!r}->{ours[c]!r}"
                 for c in COMPARED if not same(c, ours[c], theirs[c])]
        if terminated(res[r]) is False:
            diffs.append("NOT TERMINATED")
        if diffs:
            n_diff += 1
            print(f"{r:32s} DIFF  " + "; ".join(diffs))
        else:
            n_same += 1
            print(f"{r:32s} same")
    print(f"=> {n_same} same, {n_diff} different, {len(ROW_IDS) - len(res) - len(timeouts)} not run")
    return n_diff


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="*", help="row selectors (see --list); none = only --table/--compare")
    ap.add_argument("--list", action="store_true", help="list the 26 rows and exit")
    ap.add_argument("--out", help="output directory (default: <repo>/runs/<timestamp>)")
    ap.add_argument("--threads", type=int, help="OpenBLAS/OpenMP threads for the workers")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="rows to run in parallel (default 1); "
                    "results don't depend on it, timing columns do")
    ap.add_argument("--force", action="store_true", help="rerun rows that already have a result "
                    "(or were stopped by --timeout)")
    ap.add_argument("--timeout", type=float, metavar="MIN",
                    help="stop a row after MIN minutes and record it as not terminated (e.g. stuck in "
                         "equivalence-query rounds); such rows are skipped on reruns unless --force")
    ap.add_argument("--table", action="store_true", help="print thesis-format tables of the results in --out")
    ap.add_argument("--compare", metavar="REF", help="'thesis' or another output directory")
    ap.add_argument("--printed", action="store_true",
                    help="with --compare <dir>: compare at printed precision instead of exactly")
    ap.add_argument("--set", action="append", metavar="KEY=VALUE", default=[],
                    help="override a setting of the rows, e.g. --set optimization.n_starts=10 --set design.noise_std=1e-3 "
                         "(stored in each result file)")
    ap.add_argument("--plots", action="store_true", help="save trajectory figures (experi-plots/)")
    ap.add_argument("--dpi", type=int, default=150, help="resolution of saved figures (default 150; thesis: 600)")
    ap.add_argument("--seeds", metavar="LIST", help="run every row once per seed (e.g. 0-4 or 0,3,7) into <out>/seed<K>/")
    ap.add_argument("--summary", action="store_true",
                    help="summary over the seed<K>/ directories of --out (with --compare DIR: next to those of DIR)")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.out, args.threads, parse_overrides(args.set), args.plots, args.dpi)
        return
    if args.list:
        print_list()
        return
    if not args.rows and not (args.table or args.compare or args.summary):
        ap.error("give row selectors (e.g. 'all' or 1-8), or --table/--compare/--summary with --out")
    if not args.out:
        if not args.rows:
            ap.error("--table/--compare need --out")
        args.out = os.path.abspath(os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S")))
    args.out = os.path.abspath(os.path.expanduser(args.out))

    rc = 0
    if args.rows and args.seeds:
        base, base_set = args.out, list(args.set)
        for k in parse_seeds(args.seeds):
            args.out, args.set = os.path.join(base, f"seed{k}"), base_set + [f"seed={k}"]
            print(f"\n=== seed {k} ===")
            rc = run_rows(args, select(args.rows)) or rc
        args.out, args.set = base, base_set
    elif args.rows:
        rc = run_rows(args, select(args.rows))
    if args.summary:
        print_summary(args.out, args.compare)
        sys.exit(rc)
    if args.table:
        print_tables(args.out)
    if args.compare:
        n_diff = compare(args.out, args.compare, args.printed)
        rc = rc or (1 if n_diff else 0)
    sys.exit(rc)


def parse_seeds(text):
    """'0-4' -> [0, 1, 2, 3, 4]; '0,3,7' -> [0, 3, 7]."""
    seeds = []
    for part in str(text).split(","):
        a, _, b = part.strip().partition("-")
        seeds += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return seeds


def summarize_seeds(out):
    """{row: summary over the seed<K>/ directories of out}: seeds run, terminated count (timeouts count as not
    terminated, and so do crashed runs: a log without a result), largest NEQ, median and largest d3_avg and d3_max, and the median and largest held-out errors."""
    import glob
    import numpy as np
    dirs = sorted(glob.glob(os.path.join(out, "seed*")), key=lambda d: int(os.path.basename(d)[4:] or 0))
    per_row = {}
    for d in dirs:
        res, tos = load_results(d), load_timeouts(d)
        # a row that has a log but neither a result nor a timeout marker crashed: it counts as not terminated
        logs = {r for r in ROW_IDS if os.path.exists(os.path.join(d, "logs", fname(r) + ".log"))}
        for r in set(res) | set(tos) | logs:
            per_row.setdefault(r, []).append(res.get(r))
    summary = {}
    for r in sorted(per_row, key=ROW_IDS.index):
        recs = per_row[r]
        ok = [x for x in recs if x is not None]
        get = lambda f: [f(x) for x in ok if f(x) is not None]
        d3 = get(lambda x: x["values"]["d3_avg"])
        d3m = get(lambda x: x["values"]["d3_max"])
        ha = get(lambda x: x.get("run_info", {}).get("heldout", {}).get("d_avg"))
        hm = get(lambda x: x.get("run_info", {}).get("heldout", {}).get("d_max"))
        summary[r] = dict(
            seeds=len(recs), terminated=sum(1 for x in ok if terminated(x)),
            neq_max=max(get(lambda x: x["values"]["NEQ"]), default=None),
            d3_med=float(np.median(d3)) if d3 else None, d3_max=max(d3, default=None),
            d3max_med=float(np.median(d3m)) if d3m else None,
            hold_med=float(np.median(ha)) if ha else None, hold_max=max(ha, default=None),
            holdmax_med=float(np.median(hm)) if hm else None)
    return summary


def print_summary(out, ref=None):
    """Prints (and writes <out>/summary.json) the summary over seeds; with ref, the medians of ref next to it."""
    s = summarize_seeds(out)
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(s, f, indent=1)
    r = summarize_seeds(os.path.expanduser(ref)) if ref and ref != "thesis" else {}
    g = lambda v: "-" if v is None else f"{v:.2e}"
    print(f"### Summary over seeds of {out}" + (f" (ref: {ref})" if r else ""))
    print(f"{'row':32s} {'term':>6s} {'NEQmax':>6s} {'d3 med':>9s} {'d3 max':>9s} {'hold med':>9s} {'hold max':>9s}"
          + (f" | {'ref d3 med':>10s} {'ref hold med':>12s}" if r else ""))
    n_bad = 0
    for row, v in s.items():
        n_bad += v["terminated"] < v["seeds"]
        line = (f"{row:32s} {v['terminated']:>3d}/{v['seeds']:<2d} {('-' if v['neq_max'] is None else int(v['neq_max'])):>6} "
                f"{g(v['d3_med']):>9s} {g(v['d3_max']):>9s} {g(v['hold_med']):>9s} {g(v['hold_max']):>9s}")
        if r:
            rv = r.get(row, {})
            line += f" | {g(rv.get('d3_med')):>10s} {g(rv.get('hold_med')):>12s}"
        print(line)
    print(f"=> {len(s)} rows; {n_bad} with a seed that did not terminate")


def run_rows(args, rows):
    # created here because an exists-then-makedirs check races when rows start in parallel
    for d in ("logs", "experi-plots", os.path.join("experi-info", "latex-files")):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    env = child_env(args)
    ok, lines = platform_report()
    print("\n".join(lines))
    blas = blas_threads(env) if args.threads else None
    if args.threads:
        print(f"OpenBLAS threads in workers: {blas} (requested {args.threads})")
    if args.threads and isinstance(blas, list) and any(n != args.threads for _, n in blas):
        print(f"! OpenBLAS does not use the requested {args.threads} threads (it is capped at the number "
              "of physical cores).")
    done = load_results(args.out)
    timeouts = load_timeouts(args.out)
    todo = [r for r in rows if args.force or (r not in done and r not in timeouts)]
    skipped = [r for r in rows if r not in todo]
    if skipped:
        print(f"skipping {len(skipped)} rows with existing results or timeouts (use --force to rerun): "
              f"{', '.join(skipped)}")
    for r in todo:
        if r in timeouts:
            os.remove(os.path.join(args.out, "results", fname(r) + ".timeout"))
    print(f"output: {args.out}\nrunning {len(todo)} rows, "
          f"~{sum(minutes(r) for r in todo)} min sequential estimate\n")
    with open(os.path.join(args.out, "runs.log"), "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {' '.join(sys.argv)} | "
                f"threads={args.threads} blas={blas} rows={todo}\n")

    running, failed, stopped, not_terminated = {}, [], [], []
    # in parallel, start the longest rows first so the slowest one doesn't run alone at the end
    queue = sorted(todo, key=minutes, reverse=True) if args.jobs > 1 else list(todo)
    start = time.time()
    while queue or running:
        while queue and len(running) < max(1, args.jobs):
            r = queue.pop(0)
            log = open(os.path.join(args.out, "logs", fname(r) + ".log"), "w")
            cmd = [sys.executable, "-m", "hydyreg.benchmarks.runner", "--worker", r, "--out", args.out,
                   ] + (["--threads", str(args.threads)] if args.threads else []) + \
                sum((["--set", item] for item in args.set), []) + (["--plots"] if args.plots else []) + ["--dpi", str(args.dpi)]
            running[r] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=args.out),
                          log, time.time())
            print(f"[{datetime.now():%H:%M:%S}] start {r} (row {ROW_IDS.index(r) + 1})", flush=True)
        time.sleep(1)
        for r, (p, log, t0) in list(running.items()):
            if p.poll() is None and args.timeout and time.time() - t0 > args.timeout * 60:
                p.kill()
                p.wait()
                log.close()
                del running[r]
                stopped.append(r)
                os.makedirs(os.path.join(args.out, "results"), exist_ok=True)
                with open(os.path.join(args.out, "results", fname(r) + ".timeout"), "w") as f:
                    f.write(f"{args.timeout}\n")
                print(f"[{datetime.now():%H:%M:%S}] TIMEOUT {r}: stopped after {args.timeout:g} min, "
                      f"recorded as not terminated (see logs/{fname(r)}.log)", flush=True)
                continue
            if p.poll() is None:
                continue
            log.close()
            del running[r]
            dt = time.time() - t0
            if p.returncode == 0 and r in load_results(args.out):
                v = load_results(args.out)[r]["values"]
                info = load_results(args.out)[r].get("run_info", {})
                if info.get("terminated") is False:
                    not_terminated.append(r)
                print(f"[{datetime.now():%H:%M:%S}] done  {r} in {dt / 60:.1f} min: d = "
                      f"{v['d1_avg']:.5f} / {v['d2_avg']:.5f} / {v['d3_avg']:.5f}, NEQ {v['NEQ']}, "
                      f"d_dot {v['d_dot']:.1e}, modes {info.get('n_modes', '?')}"
                      + ("" if info.get("terminated", True) else "  ** NOT TERMINATED (last equivalence "
                         "query still has counterexamples) **"), flush=True)
            else:
                failed.append(r)
                print(f"[{datetime.now():%H:%M:%S}] FAIL  {r} (exit {p.returncode}), see logs/{fname(r)}.log",
                      flush=True)
    print(f"\nfinished {len(todo) - len(failed) - len(stopped)}/{len(todo)} rows in "
          f"{(time.time() - start) / 60:.1f} min"
          + (f"; failed: {', '.join(failed)}" if failed else "")
          + (f"; stopped by --timeout: {', '.join(stopped)}" if stopped else "")
          + (f"; NOT terminated: {', '.join(not_terminated)}" if not_terminated else ""))
    return 1 if failed or stopped or not_terminated else 0


if __name__ == "__main__":
    main()
