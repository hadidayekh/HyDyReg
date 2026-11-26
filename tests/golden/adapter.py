"""The only file of the golden tool that imports the implementation.

- row_settings(row_id): the configuration and sampling design of a row, as a JSON-compatible dictionary;
- apply_overrides(row, overrides): the row with LearnConfig overrides ('equivalence.max_rounds') and design
  overrides ('design.noise_std') applied;
- run(row_id, overrides): runs the row with the overrides and returns (values, final model, run_info, trace).
"""
import contextlib
import dataclasses
import os
import tempfile

COLS = ["d1_avg", "d2_avg", "d3_avg", "d1_max", "d2_max", "d3_max", "t1", "t2", "t3", "NEQ", "d_dot"]


def row_settings(row_id):
    """Configuration and sampling design of the registry entry of the row."""
    from hydyreg.benchmarks.rows import get_row
    row = get_row(row_id)
    return dict(config=row.config.to_dict(), design=dataclasses.asdict(row.design))


def apply_overrides(row, overrides):
    design = {k.split(".", 1)[1]: v for k, v in overrides.items() if k.startswith("design.")}
    config = {k: v for k, v in overrides.items() if not k.startswith("design.")}
    if config:
        row.config = row.config.apply_overrides(config)
    if design:
        row.design = dataclasses.replace(row.design, **design)
    return row


def run(row_id, overrides):
    from hydyreg.benchmarks.rows import get_row, run_row
    from hydyreg import model as model_pkg
    trace, last = [], {}
    orig = model_pkg.HybridRegressor.classify_modes

    def classify_modes(self, *a, **k):
        r = orig(self, *a, **k)
        trace.append(_trace_entry(self))
        last["model"] = self
        return r
    model_pkg.HybridRegressor.classify_modes = classify_modes
    row = apply_overrides(get_row(row_id), overrides)
    try:
        with _in_tmpdir():
            result = run_row(row)
    finally:
        model_pkg.HybridRegressor.classify_modes = orig
    values = dict(zip(COLS, [float(v) for v in result.to_row()]))
    return values, last.get("model"), dict(result.info), trace


def _trace_entry(model):
    from hashing import sha
    e = dict(n_modes=int(model.Omegas_.shape[0]) if hasattr(model, "Omegas_") else None,
             Omegas=sha(model.Omegas_), betas=sha(model.betas_),
             X_classify=sha(model.X_classify_), modes_classify=sha(model.modes_classify_))
    c = getattr(model, "classifier_", None)
    for name in ("coef_", "intercept_", "W_", "B_"):
        if c is not None and hasattr(c, name):
            e["classifier_" + name] = sha(getattr(c, name))
    return e


@contextlib.contextmanager
def _in_tmpdir():
    old = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            yield d
        finally:
            os.chdir(old)
