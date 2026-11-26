"""Configuration of a learning run: `LearnConfig` (how to learn) and `SamplingDesign` (which data to ask a
system for, for active learning).

Every field has the value used by the benchmark rows as its default. The behaviour of the thesis is not reproduced by options
of this package: it is the code at the tag `thesis-reproducible`.
"""
from dataclasses import dataclass, field, fields, asdict, is_dataclass, MISSING
from typing import Any, List, Optional, Sequence, Union

METHODS = ("incremental", "optimization")
CLASSIFIER_KINDS = ("logistic", "decision_list", "pairwise")


@dataclass
class ClassifierConfig:
    """Mode classifier.

    kind: 'logistic' (default; L2-regularized logistic regression on standardized features, polytope regions of
        thesis 4.3), 'decision_list' ("if ... else if ... else" regions, represents T-junctions of boundaries) or
        'pairwise' (one-versus-one).
    C: inverse regularization strength (1e6).
    estimator: a classifier object to use instead (API only); overrides the other fields.
    """
    kind: str = "logistic"
    C: float = 1e6
    estimator: Any = None

    def build(self):
        if self.estimator is not None:
            return self.estimator
        from ..classifier import (StandardizedLogisticRegression, DecisionListLogisticRegression,
                                  PairwiseLogisticRegression)
        if self.kind == "logistic":
            return StandardizedLogisticRegression(C=self.C)
        if self.kind == "decision_list":
            return DecisionListLogisticRegression(C=self.C)
        return PairwiseLogisticRegression(C=self.C)


@dataclass
class NoiseConfig:
    """Measurement noise (estimated from the data; nothing changes on noise-free data).

    adapt: the measurement noise is estimated from the data, and (1) the tolerances on estimated derivatives
        follow it, (2) the segmentation and derivative windows of 2m+1 samples are chosen from it, and (3) under
        noticeable noise the incremental method fits the modes by ridge regression instead of least squares.
        False: fixed tolerances, 5-sample windows, least squares (kept for noise experiments).
    integral_refit: under noticeable noise, after the passive fit, refit each mode by integral least squares on the
        runs of consecutive training samples assigned to it (no derivatives). Option, off.
    """
    adapt: bool = True
    integral_refit: bool = False


@dataclass
class BoundaryConfig:
    """Boundary refinement at change points.

    points: 'model' (switches located inside each gap with the learned modes; no query) or 'query' (thesis:
        fine-step simulation of the system across every gap). Without a system to query, gaps that the
        model cannot explain are skipped.
    fine_step_factor: fine step = time step of the trajectories whose gaps are refined (training data, or
        equivalence-query trajectories) / fine_step_factor.
    """
    points: str = "model"
    fine_step_factor: int = 20


@dataclass
class OptimizationConfig:
    """Optimization-based method (thesis 4.2).

    The program is solved for the values of C in increasing order: the smallest from `n_starts` starting points
    (lowest objective kept), each larger one from the solution of the previous one (continuation in C). Slacks
    are kept nonnegative exactly (IPOPT bound_relax_factor = 0). The value of C with the lowest validation error
    is selected; after the selection, each mode is refitted on all points (training and validation) assigned to
    it, unless refit = 'none'. Training segments with an undetected switch are split along the learned modes.

    solver: 'ipopt' (the program of 4.2) or 'clusterwise' (alternating least squares; option).
    C_space: values of the regularization weight C (default [0]: with the refit of the modes, the final coefficients
        depend only on the assignment of segments to modes, so C has no effect and C = 0 was always selected; with
        refit='none' C regularizes the coefficients of the program, and a warning says so when C_space is [0]).
    deltas: insensitivity error, a number or 'auto' (estimated from the derivative error).
    n_starts: IPOPT starts for the smallest C.
    init_seed: seed of the starting points (start k uses init_seed + k), the same for every C; None (default):
        derived from LearnConfig.seed.
    init: starting point of the smallest C: 'random' (n_starts random points), 'segments' (n_starts points, each
        mode from the least-squares fit on one segment) or 'clusterwise' (one start, from the dynamics of the
        clusterwise solution: the program then refines the assignment found by alternating least squares).
    stop_tol: stop the starts once every segment is fitted within deltas; 'auto' = 1e-3 min(deltas), or a number.
    refit: 'ridge', 'ls' or 'none' (refit of each mode on its points after the program; 'none' keeps the
        coefficients computed by IPOPT).
    clusterwise_restarts: starts of the clusterwise solver.
    """
    solver: str = "ipopt"
    C_space: List[float] = field(default_factory=lambda: [0.0])
    deltas: Union[str, float, List[float]] = "auto"
    n_starts: int = 30
    init_seed: Optional[int] = None
    init: str = "random"
    stop_tol: Union[str, float] = "auto"
    refit: str = "ridge"
    clusterwise_restarts: int = 200


@dataclass
class SimulationConfig:
    """Simulation of systems and learned models (sliding motion is simulated with the Filippov vector field).
    max_step_factor: largest integration step = time step / max_step_factor."""
    max_step_factor: float = 10


@dataclass
class EquivalenceConfig:
    """Approximate equivalence queries (active learning only).

    enabled: None = when bounds are given.
    bounds: box of initial conditions, [[lo, hi], ...] per state variable.
    n_trajs: trajectories per query. tol: relative difference above which a trajectory is a counterexample
        (0.01; the thesis used 0.1, which let 10 of the 26 benchmark rows stop after a first query that passed
        too easily: CHANGELOG.md).
    time_step, time_range: of the query trajectories (None: those of the test set of the sampling design).
    max_counterexamples: a query stops after this many counterexamples.
    max_rounds: largest number of queries.
    update_dynamics: counterexamples also update the dynamics (None: True for the incremental method, False
        for the optimization-based method).
    """
    enabled: Optional[bool] = None
    bounds: Optional[List[List[float]]] = None
    n_trajs: int = 50
    tol: float = 0.01
    time_step: Optional[float] = None
    time_range: Optional[float] = None
    max_counterexamples: int = 5
    max_rounds: int = 100
    update_dynamics: Optional[bool] = None


@dataclass
class LearnConfig:
    """How to learn a switched system.

    method: 'incremental' or 'optimization'.
    n_modes: number of modes (required by the optimization-based method; the incremental method finds it).
    poly_degree: largest degree of the polynomial dynamics.
    boundary_degree: largest degree of the polynomial features of the mode boundaries.
    seg_tol: segmentation tolerance (size of a derivative jump that marks a switch).
    rd_tol: tol_RD of the thesis (Algorithm 4): a mode captures a set of points if the mean relative difference
        between their estimated derivatives and its prediction is below it (raised automatically under noise).
    seed: seed of the run. Every random draw of the learner comes from it, in independent streams: initial
        conditions of the equivalence queries, the validation split of the optimization-based method, the
        measurement noise of experiments (SamplingDesign.noise_std), and the starting points of IPOPT and of the
        clusterwise solver (unless optimization.init_seed is given). NumPy's global random state is not used.
        The test initial conditions ('random') do not depend on it.

    Example:
        LearnConfig(method="incremental", poly_degree=1, seg_tol=0.01,
                    equivalence=EquivalenceConfig(bounds=[[95, 105], [70, 110]]))
    """
    method: str = "incremental"
    n_modes: Optional[int] = None
    poly_degree: int = 1
    boundary_degree: int = 1
    seg_tol: float = 0.01
    rd_tol: float = 1e-3
    seed: int = 0
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    boundaries: BoundaryConfig = field(default_factory=BoundaryConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    equivalence: EquivalenceConfig = field(default_factory=EquivalenceConfig)

    def __post_init__(self):
        _check(self.method in METHODS, "method", self.method, METHODS)
        if self.method == "optimization" and self.n_modes is None:
            raise ValueError("LearnConfig.n_modes is required for method='optimization'")
        _check(self.classifier.kind in CLASSIFIER_KINDS, "classifier.kind", self.classifier.kind, CLASSIFIER_KINDS)
        _check(self.boundaries.points in ("model", "query"), "boundaries.points", self.boundaries.points, ("model", "query"))
        _check(self.optimization.solver in ("ipopt", "clusterwise"), "optimization.solver", self.optimization.solver,
               ("ipopt", "clusterwise"))
        _check(self.optimization.refit in ("ridge", "ls", "none"), "optimization.refit", self.optimization.refit,
               ("ridge", "ls", "none"))
        _check(self.optimization.init in ("random", "segments", "clusterwise"), "optimization.init",
               self.optimization.init, ("random", "segments", "clusterwise"))
        if isinstance(self.optimization.deltas, str) and self.optimization.deltas != "auto":
            raise ValueError("LearnConfig.optimization.deltas must be a number, a list of numbers, or 'auto'")
        if isinstance(self.optimization.stop_tol, str) and self.optimization.stop_tol != "auto":
            raise ValueError("LearnConfig.optimization.stop_tol must be a number or 'auto'")
        for name, v in (("noise.adapt", self.noise.adapt), ("noise.integral_refit", self.noise.integral_refit)):
            if not isinstance(v, bool):
                raise ValueError(f"LearnConfig.{name} must be true or false (got {v!r})")
        for name, v in (("seg_tol", self.seg_tol), ("equivalence.tol", self.equivalence.tol)):
            if v is not None and v < 0:
                raise ValueError(f"LearnConfig.{name} must be nonnegative (got {v})")

    # derived switches
    @property
    def equivalence_enabled(self):
        e = self.equivalence
        return (e.bounds is not None) if e.enabled is None else bool(e.enabled)

    @property
    def update_dynamics(self):
        u = self.equivalence.update_dynamics
        return (self.method == "incremental") if u is None else bool(u)

    @classmethod
    def from_dict(cls, d):
        """LearnConfig from a (nested) dictionary with the same names, e.g. the content of a TOML file without its
        [data] and [design] tables. Unknown keys raise KeyError naming the accepted keys."""
        return _from_dict(cls, d or {}, "")

    @classmethod
    def from_toml(cls, path):
        """LearnConfig from a TOML file (the [data] and [design] tables, read by the command line, are ignored)."""
        d = read_toml(path)
        d.pop("data", None)
        d.pop("design", None)
        return cls.from_dict(d)

    def apply_overrides(self, items):
        """A new LearnConfig with overrides applied: a dict {'optimization.n_starts': 10} or a list of
        'key=value' strings (values read as TOML values)."""
        if not isinstance(items, dict):
            items = dict(_split_override(i) for i in items)
        d = self.to_dict()
        est = self.classifier.estimator
        for k, v in items.items():
            target, parts = d, k.split(".")
            for p in parts[:-1]:
                if not isinstance(target.get(p), dict):
                    raise KeyError(f"unknown configuration key {k!r}")
                target = target[p]
            target[parts[-1]] = v
        new = LearnConfig.from_dict({k: v for k, v in d.items()})
        if est is not None and new.classifier.estimator == type(est).__name__:
            new.classifier.estimator = est
        return new

    def to_dict(self):
        """Plain dictionary (the classifier estimator, if any, as its class name)."""
        d = asdict(self)
        if self.classifier.estimator is not None:
            d["classifier"]["estimator"] = type(self.classifier.estimator).__name__
        return d


@dataclass
class SamplingDesign:
    """Data that `learn_from_system` asks the system for.

    time_step, time_range, x0s: training trajectories.
    test_time_step, test_time_range, test_x0s: test trajectories (errors are measured on them); test_x0s =
        'random' draws 10 initial conditions in the equivalence bounds with np.random.default_rng(0).
    noise_std: standard deviation of the measurement noise added to everything the learner observes
        (experiments); the errors are measured against the noise-free system.

    Example:
        SamplingDesign(time_step=0.25, time_range=25, x0s=[[105.5, 80], [97.5, 100]])
    """
    time_step: float
    time_range: float
    x0s: Sequence[Sequence[float]]
    test_time_step: Optional[float] = None
    test_time_range: Optional[float] = None
    test_x0s: Union[str, Sequence[Sequence[float]]] = "random"
    noise_std: float = 0.0

    def __post_init__(self):
        if self.test_time_step is None:
            self.test_time_step = self.time_step
        if self.test_time_range is None:
            self.test_time_range = self.time_range


def _check(ok, name, value, accepted):
    if not ok:
        raise ValueError(f"LearnConfig.{name} = {value!r} is not one of {list(accepted)}")


def _replace_nested(obj, updates):
    """Applies {'group.field': value} or {'field': value} updates to a (nested) dataclass, in place."""
    for key, value in updates.items():
        target, parts = obj, key.split(".")
        for p in parts[:-1]:
            target = getattr(target, p)
        names = {f.name for f in fields(target)}
        if parts[-1] not in names:
            raise KeyError(f"unknown configuration key {key!r}; accepted keys here: {sorted(names)}")
        setattr(target, parts[-1], value)
    return obj


def is_config(x):
    return is_dataclass(x)


def _from_dict(cls, d, prefix):
    if not isinstance(d, dict):
        raise TypeError(f"configuration table {prefix.rstrip('.') or '(top level)'} must be a table")
    known = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(known)
    if unknown:
        raise KeyError(f"unknown configuration key(s) {sorted(prefix + k for k in unknown)}; accepted keys"
                       f"{' in [' + prefix.rstrip('.') + ']' if prefix else ''}: {sorted(known)}")
    kwargs = {}
    for k, v in d.items():
        default = known[k].default_factory() if known[k].default_factory is not MISSING else None
        if is_dataclass(default) and not is_dataclass(v):
            kwargs[k] = _from_dict(type(default), v, prefix + k + ".")
        else:
            kwargs[k] = v
    return cls(**kwargs)


def read_toml(path):
    """The content of a TOML file as a dictionary."""
    import tomllib as toml
    with open(path, "rb") as f:
        return toml.load(f)


def parse_toml_value(text):
    """A value written as in TOML ('10', '1e-3', 'true', '[0, 0.1]', '"ipopt"'); bare words are strings."""
    import tomllib as toml
    try:
        return toml.loads("v = " + text)["v"]
    except Exception:
        return text


def _split_override(item):
    if "=" not in item:
        raise ValueError(f"override {item!r} is not of the form key=value")
    k, v = item.split("=", 1)
    return k.strip(), parse_toml_value(v.strip())
