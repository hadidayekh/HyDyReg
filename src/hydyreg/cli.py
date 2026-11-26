"""Command line: `hydyreg fit | learn | simulate | export | bench`.

    hydyreg fit runs/*.csv --method incremental --seg-tol 0.01 --out result/
    hydyreg fit runs/*.csv --config fit.toml --set optimization.n_starts=10 --out result/ --latex --plots
    hydyreg learn --system hydyreg.benchmarks.systems:isolette --config learn_isolette.toml --out result/
    hydyreg simulate result/model.json --x0 100,90 --step 0.25 --time 25 --out sim.csv
    hydyreg export result/model.json --format latex
    hydyreg bench run incr/isolette --out runs/a ;  hydyreg bench compare --out runs/a --ref runs/b

Exit codes: 0 success; 1 a run failed or did not terminate; 2 usage or configuration error.
"""
import argparse
import importlib
import logging
import sys

EXIT_FAILED, EXIT_USAGE = 1, 2

CONFIG_SHORTCUTS = [("--method", "method", str), ("--n-modes", "n_modes", int), ("--poly-degree", "poly_degree", int),
                    ("--boundary-degree", "boundary_degree", int), ("--seg-tol", "seg_tol", float),
                    ("--classifier", "classifier.kind", str)]
DATA_SHORTCUTS = [("--dt", "dt", float), ("--delimiter", "delimiter", str), ("--skip-header", "skip_header", str),
                  ("--time-column", "time_column", str), ("--state-columns", "state_columns", str)]


class UsageError(Exception):
    pass


def _version():
    try:
        from importlib import metadata
        return metadata.version("hydyreg")
    except Exception:
        return "unknown"


def _common(p):
    p.add_argument("-v", "--verbose", action="store_true", help="more output")
    p.add_argument("-q", "--quiet", action="store_true", help="warnings and errors only")


def _config_args(p, data=False):
    p.add_argument("--config", help="TOML configuration file (see examples/configs/)")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="set a configuration key by its dotted name, e.g. --set optimization.n_starts=10 "
                        "(the value is read as a TOML value)")
    for flag, key, typ in CONFIG_SHORTCUTS:
        p.add_argument(flag, dest="cfg_" + key.replace(".", "_"), type=typ, help=f"shortcut for {key}")
    if data:
        for flag, key, typ in DATA_SHORTCUTS:
            p.add_argument(flag, dest="data_" + key, type=typ, help=f"shortcut for data.{key}")


def build_parser():
    ap = argparse.ArgumentParser(prog="hydyreg", description="Identification of switched nonlinear dynamical systems.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n")[1])
    ap.add_argument("--version", action="version", version=f"hydyreg {_version()}")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("fit", help="learn passively from trajectory files (CSV or whitespace-separated text)",
                       description="Learn a switched system from recorded trajectories, one file per trajectory "
                                   "(or one file with a trajectory column).\n\nExample:\n  hydyreg fit runs/*.csv "
                                   "--method incremental --seg-tol 0.01 --out result/",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("data", nargs="*", help="files or glob patterns (default: data.files of the configuration)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--latex", action="store_true", help="also write equations.tex")
    p.add_argument("--plots", action="store_true", help="also write figures")
    _config_args(p, data=True)
    _common(p)

    p = sub.add_parser("learn", help="learn actively from a system defined in Python",
                       description="Active learning on a system given as MODULE:FUNCTION (a function without "
                                   "arguments that returns an SNDS). The sampling design is the [design] table of the "
                                   "configuration file.\n\nExample:\n  hydyreg learn --system "
                                   "hydyreg.benchmarks.systems:isolette --config examples/configs/learn_isolette.toml --out result/",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system", required=True, help="MODULE:FUNCTION returning an SNDS")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--latex", action="store_true", help="also write equations.tex")
    p.add_argument("--plots", action="store_true", help="also write figures")
    _config_args(p)
    _common(p)

    p = sub.add_parser("simulate", help="simulate a saved model",
                       description="Simulate a saved model from initial conditions and write the trajectories as CSV."
                                   "\n\nExample:\n  hydyreg simulate result/model.json --x0 100,90 --step 0.25 --time 25 --out sim.csv",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--x0", action="append", required=True, help="initial condition, comma-separated (repeatable)")
    p.add_argument("--step", type=float, required=True)
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--out", required=True, help="CSV file (one file per initial condition if several: FILE_1.csv, ...)")
    _common(p)

    p = sub.add_parser("export", help="print or write a saved model as text, LaTeX, MATLAB or SpaceEx",
                       description="Example:\n  hydyreg export result/model.json --format latex --out model.tex",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("model")
    p.add_argument("--format", choices=["text", "latex", "matlab", "spaceex"], default="text")
    p.add_argument("--out", help="output file (default: print)")
    _common(p)

    p = sub.add_parser("bench", help="the 26 experiments of the thesis (list, run, table, compare)",
                       description="Examples:\n  hydyreg bench list\n  hydyreg bench run incr/isolette 'opti/*' -j 4 "
                                   "--timeout 30 --out runs/a\n  hydyreg bench table --out runs/a\n  hydyreg bench compare "
                                   "--out runs/a --ref runs/b\n  hydyreg bench run all --seeds 0-4 -j 4 --out runs/m\n"
                                   "  hydyreg bench summary --out runs/m [--ref runs/m_old]",
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("action", choices=["list", "run", "table", "compare", "summary"])
    p.add_argument("rows", nargs="*", help="row selectors for 'run' (all, 1-8, incr/*, ham, ...)")
    p.add_argument("--out")
    p.add_argument("--ref", help="for 'compare': another output directory, or 'thesis'")
    p.add_argument("--printed", action="store_true", help="for 'compare': compare at printed precision")
    p.add_argument("--seeds", help="for 'run': run every row once per seed (e.g. 0-4) into <out>/seed<K>/; "
                                   "then 'summary --out <out>' aggregates them")
    p.add_argument("-j", "--jobs", type=int, default=1)
    p.add_argument("--force", action="store_true")
    p.add_argument("--timeout", type=float, metavar="MIN")
    p.add_argument("--threads", type=int)
    p.add_argument("--plots", action="store_true")
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    _common(p)
    return ap


# ---------------------------------------------------------------------------------------------- configuration

def load_config_args(args, data=False):
    """(LearnConfig, data table, design table) from defaults, --config, shortcuts and --set, in that order."""
    from .learning.config import LearnConfig, parse_toml_value, read_toml
    d = read_toml(args.config) if args.config else {}
    data_table = d.pop("data", {}) or {}
    design_table = d.pop("design", {}) or {}
    for flag, key, typ in CONFIG_SHORTCUTS:
        v = getattr(args, "cfg_" + key.replace(".", "_"), None)
        if v is not None:
            _set_dotted(d, key, v)
    if data:
        for flag, key, typ in DATA_SHORTCUTS:
            v = getattr(args, "data_" + key, None)
            if v is not None:
                data_table[key] = v
    for item in args.set:
        if "=" not in item:
            raise UsageError(f"--set expects KEY=VALUE, got {item!r}")
        k, v = item.split("=", 1)
        v = parse_toml_value(v)
        if k.startswith("data."):
            data_table[k[5:]] = v
        elif k.startswith("design."):
            design_table[k[7:]] = v
        else:
            _set_dotted(d, k, v)
    try:
        config = LearnConfig.from_dict(d)
    except (KeyError, ValueError, TypeError) as e:
        raise UsageError(str(e).strip('"'))
    return config, data_table, design_table


def _set_dotted(d, key, value):
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


def _columns(v):
    if v is None or isinstance(v, (int, list)):
        return v
    v = str(v)
    if "," in v:
        return [int(c) if c.strip().lstrip("-").isdigit() else c.strip() for c in v.split(",")]
    return int(v) if v.lstrip("-").isdigit() else v


def read_data(files, table):
    from .trajectories import Trajectories
    files = list(files) or list(table.get("files", []))
    if not files:
        raise UsageError("no data file given (arguments, or data.files in the configuration)")
    allowed = {"files", "time_column", "state_columns", "delimiter", "skip_header", "dt", "trajectory_column"}
    unknown = set(table) - allowed
    if unknown:
        raise UsageError(f"unknown key(s) in [data]: {sorted(unknown)}; accepted keys: {sorted(allowed)}")
    skip = table.get("skip_header", "auto")
    if isinstance(skip, str) and skip.isdigit():
        skip = int(skip)
    delim = table.get("delimiter", "auto")
    dt = table.get("dt")
    tcol = _columns(table.get("time_column", 0)) if dt is None else None
    return Trajectories.from_csv(files, time_column=tcol, state_columns=_columns(table.get("state_columns")),
                                 delimiter=None if delim in ("whitespace", " ") else delim, skip_header=skip, dt=dt,
                                 trajectory_column=_columns(table.get("trajectory_column")))


# ---------------------------------------------------------------------------------------------- commands

def cmd_fit(args):
    from .learning.pipeline import learn_from_data
    from .learning.report import write_report
    config, data_table, _ = load_config_args(args, data=True)
    trajs = read_data(args.data, data_table)
    logging.getLogger("hydyreg").info("%d trajectories, %d state variables", trajs.num_trajectories, trajs.d)
    result = learn_from_data(trajs, config, name="fit", print_on=args.verbose)
    paths = write_report(result, args.out, latex=args.latex, plots=args.plots)
    print(result.summary())
    print(result.equations())
    print(f"\nwritten: {', '.join(paths.values())}")
    return 0


def cmd_learn(args):
    from .learning.config import SamplingDesign
    from .learning.pipeline import learn_from_system
    from .learning.report import write_report
    config, _, design_table = load_config_args(args)
    if ":" not in args.system:
        raise UsageError("--system expects MODULE:FUNCTION, e.g. hydyreg.benchmarks.systems:isolette")
    mod, fun = args.system.split(":", 1)
    try:
        system = getattr(importlib.import_module(mod), fun)()
    except (ImportError, AttributeError) as e:
        raise UsageError(f"cannot load {args.system}: {e}")
    try:
        design = SamplingDesign(**design_table)
    except TypeError as e:
        raise UsageError(f"[design] table: {e}")
    result = learn_from_system(system, design, config, name="learn", print_on=args.verbose)
    paths = write_report(result, args.out, latex=args.latex, plots=args.plots)
    print(result.summary())
    print(result.equations())
    print(f"\nwritten: {', '.join(paths.values())}")
    return 0 if result.terminated in (True, None) else EXIT_FAILED


def cmd_simulate(args):
    import numpy as np
    from .persistence import load_model
    model = load_model(args.model)
    x0s = [[float(v) for v in x.split(",")] for x in args.x0]
    trajs = model.to_SNDS().simulate_trajectory(args.step, args.time, x0s)
    names = getattr(model, "names_", None) or [f"x{i + 1}" for i in range(len(x0s[0]))]
    outs = [args.out] if len(x0s) == 1 else [args.out.replace(".csv", "") + f"_{i + 1}.csv" for i in range(len(x0s))]
    for t, x, path in zip(trajs.t_list, trajs.x_list, outs):
        np.savetxt(path, np.c_[t, x], delimiter=",", header=",".join(["t"] + list(names)), comments="")
    print("written: " + ", ".join(outs))
    return 0


def cmd_export(args):
    from .persistence import load_model
    from .export.text import model_text
    model = load_model(args.model)
    names = getattr(model, "names_", None)
    if args.format == "text":
        s = model_text(model, names=names)
    elif args.format == "latex":
        from .export.text import model_latex
        s = model_latex(model, names=names)
    elif args.format == "matlab":
        from .export.matlab import gen_full_matlab_hyeq_class
        s = gen_full_matlab_hyeq_class(model)
    else:
        from .export.spaceex import gen_spaceex
        s = gen_spaceex(model)
    if args.out:
        with open(args.out, "w") as f:
            f.write(s)
        print("written: " + args.out)
    else:
        print(s)
    return 0


def cmd_bench(args):
    from .benchmarks import runner
    argv = []
    if args.action == "list":
        argv = ["--list"]
    elif args.action == "run":
        if not args.rows:
            raise UsageError("bench run needs row selectors (e.g. all, 1-8, 'incr/*')")
        argv = list(args.rows) + (["--out", args.out] if args.out else []) + ["-j", str(args.jobs)]
        argv += (["--force"] if args.force else []) + (["--timeout", str(args.timeout)] if args.timeout else [])
        argv += (["--threads", str(args.threads)] if args.threads else []) + (["--plots"] if args.plots else [])
        argv += ["--dpi", str(args.dpi)] + sum((["--set", s] for s in args.set), [])
        argv += ["--seeds", args.seeds] if args.seeds else []
    elif args.action == "summary":
        if not args.out:
            raise UsageError("bench summary needs --out (a directory written by bench run --seeds)")
        argv = ["--out", args.out, "--summary"] + (["--compare", args.ref] if args.ref else [])
    elif args.action == "table":
        if not args.out:
            raise UsageError("bench table needs --out")
        argv = ["--out", args.out, "--table"]
    else:
        if not (args.out and args.ref):
            raise UsageError("bench compare needs --out and --ref")
        argv = ["--out", args.out, "--compare", args.ref] + (["--printed"] if args.printed else [])
    old = sys.argv
    sys.argv = ["hydyreg bench"] + argv
    try:
        runner.main()
    except SystemExit as e:
        return int(e.code or 0) if not isinstance(e.code, str) else (print(e.code, file=sys.stderr) or EXIT_USAGE)
    finally:
        sys.argv = old
    return 0


COMMANDS = {"fit": cmd_fit, "learn": cmd_learn, "simulate": cmd_simulate, "export": cmd_export, "bench": cmd_bench}


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    level = logging.WARNING if args.quiet else (logging.DEBUG if args.verbose else logging.INFO)
    from ._logging import verbose
    verbose(level)
    try:
        return COMMANDS[args.command](args)
    except UsageError as e:
        print(f"hydyreg {args.command}: error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except (FileNotFoundError, ValueError) as e:
        print(f"hydyreg {args.command}: error: {e}", file=sys.stderr)
        return EXIT_USAGE if isinstance(e, FileNotFoundError) else EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
