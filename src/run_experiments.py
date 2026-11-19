"""Run any subset of the 26 thesis experiments (Tables 7.2-7.5).

Examples (from the repository root or from src/):
    python src/run_experiments.py --list
    python src/run_experiments.py incr/isolette              # one row
    python src/run_experiments.py 1-8                        # rows by index (see --list)
    python src/run_experiments.py 'incr/*' parametric-4      # globs and short names
    python src/run_experiments.py all --threads 10 --out ~/runs/thesis
    python src/run_experiments.py --out ~/runs/thesis --table --compare thesis

Each row runs in its own fresh Python process, so it gives the same numbers as when run
inside the original scripts (every row reseeds the global RNG). Results go to
<out>/results/<row>.json, logs to <out>/logs/<row>.log, and the usual experi-plots/ and
experi-info/ outputs to <out>/. Rows that already have a result are skipped unless --force,
so an interrupted run can be resumed with the same command.

Platform: some values depend on last-bit rounding differences of the platform, including the
number of OpenBLAS threads (see README, "Reproducing the thesis tables"). Every run prints how its
platform compares with the thesis setup. --threads N sets the OpenBLAS thread count of the worker
processes (10 on the thesis setup). OpenBLAS never uses more threads than physical cores; to
emulate more cores on macOS (e.g. 10 on an 8-core Mac), pass --fakecores <path to fakecores
dylib> (tools/fakecores).
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

SRC = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SRC)
THESIS_REF = os.path.join(REPO, "thesis", "thesis_rows.json")

BENCHMARKS = ["isolette", "affine_3modes", "lorentz_attractor", "poly2_3modes",
              "poly3_2modes_nonlinclass", "poly2_4d", "bio_sys", "ham"]
# Rough wall times (minutes, one row alone, recent Mac/Linux laptop), for planning only.
_MIN = {"opti": 3, "incr": 2, ("opti", 2): 2, ("opti", 3): 3, ("opti", 4): 3, ("opti", 5): 4,
        ("incr", 2): 4, ("incr", 3): 4, ("incr", 4): 5, ("incr", 5): 20, ("incr", 10): 15, ("incr", 20): 60}

# Thesis table order: 7.2 opti other, 7.3 incr other, 7.4 opti parametric, 7.5 incr parametric.
ROWS = ([("opti", b) for b in BENCHMARKS] + [("incr", b) for b in BENCHMARKS]
        + [("opti", f"parametric-{n}") for n in (2, 3, 4, 5)]
        + [("incr", f"parametric-{n}") for n in (2, 3, 4, 5, 10, 20)])
ROW_IDS = [f"{m}/{s}" for m, s in ROWS]
TABLES = [("Table 7.2: optimization-based, other systems", ROW_IDS[0:8]),
          ("Table 7.3: incremental, other systems", ROW_IDS[8:16]),
          ("Table 7.4: optimization-based, parametric", ROW_IDS[16:20]),
          ("Table 7.5: incremental, parametric", ROW_IDS[20:26])]

# OpenBLAS setup of the thesis runs (x86_64 Python under Rosetta on a 10-core Apple Silicon Mac).
THESIS_BLAS = {"num_threads": 10, "architecture": "Nehalem"}

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
            if not (1 <= a <= b <= len(ROWS)):
                sys.exit(f"Row range out of bounds: {tok} (rows are 1-{len(ROWS)})")
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
    if args.fakecores:
        env["DYLD_INSERT_LIBRARIES"] = os.path.abspath(os.path.expanduser(args.fakecores))
        env["FAKE_NCPU"] = str(args.threads or os.cpu_count())
    return env


def blas_info(env):
    """Ask a child process (same env) which BLAS it loads: [api, threads, kernel architecture]."""
    code = ("import json, numpy, scipy.linalg, threadpoolctl; print(json.dumps(sorted({(i.get('internal_api'), "
            "i.get('num_threads'), i.get('architecture')) for i in threadpoolctl.threadpool_info() "
            "if i.get('user_api') == 'blas'})))")
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
    if platform.python_version_tuple()[:2] != ("3", "9"):
        ok = False
        lines.append("  ! thesis used Python 3.9")
    if not (sys.platform == "darwin" and platform.machine() == "x86_64" and rosetta):
        ok = False
        lines.append("  ! thesis values were produced by x86_64 Python under Rosetta on an Apple Silicon "
                     "Mac; other platforms can give different values")
    pins = {}
    for line in open(os.path.join(REPO, "requirements.txt")):
        line = line.split("#")[0].strip()
        if "==" in line:
            name, rest = line.split("==", 1)
            ver, _, marker = rest.partition(";")
            if "sys_platform" in marker:  # only the 'sys_platform == / != "darwin"' markers occur
                is_darwin = sys.platform == "darwin"
                if ("!=" in marker) == is_darwin:
                    continue
            pins[name.strip()] = ver.strip()
    bad = []
    for name, ver in pins.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        if have != ver:
            bad.append(f"{name} {have} (pinned {ver})")
    if bad:
        ok = False
        lines.append("  ! packages differ from requirements.txt: " + ", ".join(bad))
    else:
        lines.append(f"  all {len(pins)} pinned packages match requirements.txt")
    return ok, lines


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


def worker(row_id, out, threads_requested):
    """Run one row in this process; cwd = out."""
    for p in (SRC, os.path.join(SRC, "experiments"), os.path.join(SRC, "testing")):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(out)
    method, system = row_id.split("/")
    optimization_based = method == "opti"
    t0 = time.time()
    if system.startswith("parametric"):
        import parametric
        res = parametric.run_parametric(int(system.split("-")[1]), optimization_based)
    else:
        import experiment_infos
        src_path = os.path.join(SRC, "pathtoinferdynamic.txt")
        if os.path.exists(src_path) and not os.path.exists("pathtoinferdynamic.txt"):
            with open(src_path) as f, open("pathtoinferdynamic.txt", "w") as g:
                g.write(f.read())
        compare = experiment_infos.load_inferdynamic()
        res = experiment_infos.run_benchmark(system, optimization_based, compare_with_inferDynamic=compare)
    wall = time.time() - t0
    data_avg, data_max, data_both = res

    import numpy as np  # noqa: F401  (already loaded)
    from importlib import metadata
    from threadpoolctl import threadpool_info
    both = to_jsonable(data_both)
    rec = {
        "row": row_id, "index": ROW_IDS.index(row_id) + 1,
        "values": dict(zip(COLS, both[1:12])),
        "inferdynamic": both[12:],
        "data_avg": to_jsonable(data_avg), "data_max": to_jsonable(data_max), "data_both": both,
        "wall_time_s": wall,
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


def print_tables(out):
    from utils import print_experi_data
    res = load_results(out)
    for title, ids in TABLES:
        rows = [res[r]["data_both"] for r in ids if r in res]
        missing = [r for r in ids if r not in res]
        print(f"\n### {title} ({len(rows)}/{len(ids)} rows)")
        print("name & d1 d2 d3 (avg) & d1 d2 d3 (max) & t1 t2 t3 & NEQ & d_dot")
        if rows:
            print_experi_data(rows, both=True)
        if missing:
            print("missing: " + ", ".join(missing))


def fmt_cell(col, v):
    return str(int(v)) if col == "NEQ" else (f"{v:.1e}" if col == "d_dot" else f"{v:.5f}")


def compare(out, ref):
    """Compare RD/NEQ/d_dot columns with the thesis (at printed precision) or with another
    results directory (exactly). Timing columns are ignored. Returns the number of differing rows."""
    res = load_results(out)
    if ref == "thesis":
        refvals = {k: v for k, v in json.load(open(THESIS_REF))["rows"].items()}
        same = lambda c, a, b: fmt_cell(c, a) == fmt_cell(c, b)
        label = "thesis (printed precision)"
    else:
        refvals = {k: v["values"] for k, v in load_results(os.path.expanduser(ref)).items()}
        same = lambda c, a, b: a == b
        label = f"{ref} (exact)"
    print(f"\n### Comparison of {out} with {label}")
    n_same = n_diff = 0
    for r in ROW_IDS:
        if r not in res:
            continue
        if r not in refvals:
            print(f"{r:32s} no reference")
            continue
        ours, theirs = res[r]["values"], refvals[r]
        diffs = [f"{c} {fmt_cell(c, theirs[c])}->{fmt_cell(c, ours[c])}" if ref == "thesis"
                 else f"{c} {theirs[c]!r}->{ours[c]!r}"
                 for c in COMPARED if not same(c, ours[c], theirs[c])]
        if diffs:
            n_diff += 1
            print(f"{r:32s} DIFF  " + "; ".join(diffs))
        else:
            n_same += 1
            print(f"{r:32s} same")
    print(f"=> {n_same} same, {n_diff} different, {len(ROW_IDS) - len(res)} not run")
    return n_diff


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rows", nargs="*", help="row selectors (see --list); none = only --table/--compare")
    ap.add_argument("--list", action="store_true", help="list the 26 rows and exit")
    ap.add_argument("--out", help="output directory (default: <repo>/runs/<timestamp>)")
    ap.add_argument("--threads", type=int, help="OpenBLAS/OpenMP threads for the workers (10 on the thesis "
                                                "setup; default: OpenBLAS's own choice)")
    ap.add_argument("--fakecores", help="path to fakecores dylib (macOS): lets OpenBLAS use --threads "
                                        "threads even beyond the physical core count")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="rows to run in parallel (default 1); "
                    "results don't depend on it, timing columns do")
    ap.add_argument("--force", action="store_true", help="rerun rows that already have a result")
    ap.add_argument("--table", action="store_true", help="print thesis-format tables of the results in --out")
    ap.add_argument("--compare", metavar="REF", help="'thesis' or another output directory")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        worker(args.worker, args.out, args.threads)
        return
    if args.list:
        print_list()
        return
    if not args.rows and not (args.table or args.compare):
        ap.error("give row selectors (e.g. 'all' or 1-8), or --table/--compare with --out")
    if not args.out:
        if not args.rows:
            ap.error("--table/--compare need --out")
        args.out = os.path.join(REPO, "runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    args.out = os.path.abspath(os.path.expanduser(args.out))

    rc = 0
    if args.rows:
        rc = run_rows(args, select(args.rows))
    if args.table:
        sys.path.insert(0, SRC)
        print_tables(args.out)
    if args.compare:
        rc = rc or (1 if compare(args.out, args.compare) else 0)
    sys.exit(rc)


def run_rows(args, rows):
    os.makedirs(os.path.join(args.out, "logs"), exist_ok=True)
    env = child_env(args)
    ok, lines = platform_report()
    print("\n".join(lines))
    blas = blas_info(env)
    print(f"BLAS in workers [api, threads, kernel]: {blas}"
          + (f" (requested {args.threads} threads)" if args.threads else ""))
    if args.threads and isinstance(blas, list) and any(n != args.threads for _, n, _ in blas):
        print(f"! OpenBLAS does not use the requested {args.threads} threads (it is capped at the number "
              "of physical cores). Use --fakecores to emulate more cores; see tools/fakecores and the README.")
    if not (isinstance(blas, list) and blas and all(
            api == "openblas" and n == THESIS_BLAS["num_threads"] and arch == THESIS_BLAS["architecture"]
            for api, n, arch in blas)):
        ok = False
        print(f"  ! thesis runs used OpenBLAS with {THESIS_BLAS['num_threads']} threads and "
              f"{THESIS_BLAS['architecture']} kernels")
    print("platform: same as the thesis setup" if ok else
          "platform differs from the thesis setup: some values can differ from the thesis (see README)")
    done = load_results(args.out)
    todo = [r for r in rows if args.force or r not in done]
    skipped = [r for r in rows if r not in todo]
    if skipped:
        print(f"skipping {len(skipped)} rows with existing results (use --force to rerun): {', '.join(skipped)}")
    print(f"output: {args.out}\nrunning {len(todo)} rows, "
          f"~{sum(minutes(r) for r in todo)} min sequential estimate\n")
    with open(os.path.join(args.out, "runs.log"), "a") as f:
        f.write(f"{datetime.now().isoformat(timespec='seconds')} {' '.join(sys.argv)} | "
                f"threads={args.threads} blas={blas} rows={todo}\n")

    running, failed = {}, []
    # in parallel, start the longest rows first so the slowest one doesn't run alone at the end
    queue = sorted(todo, key=minutes, reverse=True) if args.jobs > 1 else list(todo)
    start = time.time()
    while queue or running:
        while queue and len(running) < max(1, args.jobs):
            r = queue.pop(0)
            log = open(os.path.join(args.out, "logs", fname(r) + ".log"), "w")
            cmd = [sys.executable, os.path.abspath(__file__), "--worker", r, "--out", args.out] + (
                ["--threads", str(args.threads)] if args.threads else [])
            running[r] = (subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=args.out),
                          log, time.time())
            print(f"[{datetime.now():%H:%M:%S}] start {r} (row {ROW_IDS.index(r) + 1})", flush=True)
        time.sleep(1)
        for r, (p, log, t0) in list(running.items()):
            if p.poll() is None:
                continue
            log.close()
            del running[r]
            dt = time.time() - t0
            if p.returncode == 0 and r in load_results(args.out):
                v = load_results(args.out)[r]["values"]
                print(f"[{datetime.now():%H:%M:%S}] done  {r} in {dt / 60:.1f} min: d = "
                      f"{v['d1_avg']:.5f} / {v['d2_avg']:.5f} / {v['d3_avg']:.5f}, NEQ {v['NEQ']}, "
                      f"d_dot {v['d_dot']:.1e}", flush=True)
            else:
                failed.append(r)
                print(f"[{datetime.now():%H:%M:%S}] FAIL  {r} (exit {p.returncode}), see logs/{fname(r)}.log",
                      flush=True)
    print(f"\nfinished {len(todo) - len(failed)}/{len(todo)} rows in {(time.time() - start) / 60:.1f} min"
          + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    main()
