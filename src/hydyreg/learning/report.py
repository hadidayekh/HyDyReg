"""Files describing a learning run: report.txt, run.json, model.json, and optionally equations.tex and figures."""
import json
import os
import platform


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def versions():
    from importlib import metadata
    out = {"python": f"{platform.python_version()} {platform.machine()}", "platform": platform.platform()}
    for k in ("hydyreg", "numpy", "scipy", "scikit-learn", "casadi"):
        try:
            out[k] = metadata.version(k)
        except Exception:
            out[k] = None
    return out


def write_report(result, out_dir, names=None, latex=False, plots=False, dpi=150, extra=None):
    """Writes the files of a run into out_dir and returns their paths.

    report.txt: modes, equations, regions, errors, warnings. run.json: configuration, run information, errors,
    package versions. model.json: the model (see hydyreg.persistence). equations.tex (latex=True): equations
    and boundaries in LaTeX, to paste into a document. Figures (plots=True): comparison of trajectories and,
    for 2D models, the learned regions.
    """
    from ..persistence import save_model
    from ..export.text import model_text
    names = names or result.names
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    p = os.path.join(out_dir, "model.json")
    save_model(result.model, p, names=names, config=result.config)
    paths["model"] = p

    lines = [result.summary(), "", model_text(result.model, names=names)]
    unresolved = result.info.get("boundary", {}).get("unresolved")
    if unresolved:
        lines += ["", f"Warning: {unresolved} change point(s) could not be explained with the learned modes; they "
                      "have no boundary points (no system to query)."]
    p = os.path.join(out_dir, "report.txt")
    with open(p, "w") as f:
        f.write("\n".join(lines) + "\n")
    paths["report"] = p

    run = dict(name=result.name, n_modes=result.n_modes, terminated=result.terminated, n_eq=result.n_eq,
               errors=result.errors, rd_dot=result.rd_dot, times=result.times, info=result.info,
               config=result.config.to_dict(), versions=versions(), **(extra or {}))
    p = os.path.join(out_dir, "run.json")
    with open(p, "w") as f:
        json.dump(_jsonable(run), f, indent=1)
    paths["run"] = p

    if latex:
        p = os.path.join(out_dir, "equations.tex")
        with open(p, "w") as f:
            f.write(result.equations("latex"))
        paths["latex"] = p

    if plots:
        import matplotlib
        matplotlib.use("Agg")
        fig, _ = result.plot("comparison", names=names)
        p = os.path.join(out_dir, "comparison.png")
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths["comparison"] = p
        if result.model.Xfit_.shape[1] == 2 and result.n_modes > 1:
            fig, _ = result.plot("regions", names=names)
            p = os.path.join(out_dir, "regions.png")
            fig.savefig(p, dpi=dpi, bbox_inches="tight")
            paths["regions"] = p
    return paths
