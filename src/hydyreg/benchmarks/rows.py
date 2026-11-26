"""Settings of the 26 rows of the thesis tables (Tables 7.2-7.5), as (system, sampling design, configuration).

    row = get_row("incr/isolette")
    result = run_row(row)

Row ids: '<method>/<system>' with method 'opti' (optimization-based) or 'incr' (incremental), for the eight
benchmark systems, and 'opti/parametric-N' (N = 2..5), 'incr/parametric-N' (N = 2, 3, 4, 5, 10, 20).
"""
import copy
from dataclasses import dataclass
from typing import Any, Callable


from ..learning.config import (LearnConfig, SamplingDesign, BoundaryConfig, OptimizationConfig,
                               SimulationConfig, EquivalenceConfig)
from . import systems

BENCHMARK_NAMES = ["isolette", "affine_3modes", "lorentz_attractor", "poly2_3modes", "poly3_2modes_nonlinclass",
                   "poly2_4d", "bio_sys", "ham"]
ROW_IDS = ([f"opti/{s}" for s in BENCHMARK_NAMES] + [f"incr/{s}" for s in BENCHMARK_NAMES]
           + [f"opti/parametric-{n}" for n in (2, 3, 4, 5)] + [f"incr/parametric-{n}" for n in (2, 3, 4, 5, 10, 20)])

# Training and test sampling, model degrees, segmentation tolerance and equivalence queries of the benchmark
# systems (thesis Table 7.1). The test initial conditions are drawn at random in the equivalence bounds. The tolerance
# of the equivalence queries is the default of EquivalenceConfig (0.01; the thesis used 0.1, and 0.05 for isolette).
_BENCH = {
    'isolette': dict(time_step=0.25, time_range=25, x0s=[[105.5, 80], [97.5, 100]], test_time_step=0.25,
                     test_time_range=25, n_modes=2, poly_degree=1, seg_tol=0.01, boundary_degree=1,
                     bounds=[[95, 105], [70, 110]], n_trajs=50),
    'affine_3modes': dict(time_step=0.04, time_range=4, x0s=[[-2, 4], [1.5, 10], [0, 5]], test_time_step=0.04,
                          test_time_range=2, n_modes=3, poly_degree=1, seg_tol=0.5, boundary_degree=1,
                          bounds=[[-2, 3], [-1, 8]], n_trajs=50),
    'lorentz_attractor': dict(time_step=0.005, time_range=1, x0s=[[5, 5, 5], [2, 2, 2]], test_time_step=0.005,
                              test_time_range=1, n_modes=2, poly_degree=2, seg_tol=0.5, boundary_degree=1,
                              bounds=[[0, 8], [0, 8], [0, 8]], n_trajs=50),
    'poly2_3modes': dict(time_step=0.04, time_range=4, x0s=[[-1, 1], [2, -3]], test_time_step=0.04, test_time_range=4,
                         n_modes=3, poly_degree=2, seg_tol=0.1, boundary_degree=1, bounds=[[-6, 7], [-5, 15]],
                         n_trajs=50),
    'poly3_2modes_nonlinclass': dict(time_step=0.04, time_range=8, x0s=[[0, 4], [0, 4.5], [0, 5]], test_time_step=0.04,
                                     test_time_range=8, n_modes=2, poly_degree=3, seg_tol=0.5, boundary_degree=2,
                                     bounds=[[-1, 1], [-8, 5]], n_trajs=50),
    'poly2_4d': dict(time_step=0.03, time_range=5, x0s=[[6, 0.7, 2, 0], [4, 0.1, 3.1, 0], [5.9, 0.2, -3, 0], [4.1, 0.5, 2, 0]],
                     test_time_step=0.01, test_time_range=2, n_modes=2, poly_degree=2, seg_tol=0.01, boundary_degree=1,
                     bounds=[[3.5, 6.0], [-1.0, 1.0], [-4.0, 4.0], [-5.0, 5.0]], n_trajs=50),
    'bio_sys': dict(time_step=0.01, time_range=3, x0s=[[1, 7], [0.9, -1.1], [2, 3], [4, 6]], test_time_step=0.01,
                    test_time_range=2.5, n_modes=2, poly_degree=3, seg_tol=0.15, boundary_degree=1,
                    bounds=[[-4, 4], [-6, 12]], n_trajs=50),
    'ham': dict(time_step=0.075, time_range=10, x0s=[[1, 1], [0.9, -1.1], [-0.9, 0.3], [0.4, 0.6], [0.1, 0.1], [0.3, 0.1]],
                test_time_step=0.1, test_time_range=5, n_modes=2, poly_degree=3, seg_tol=0.15, boundary_degree=2,
                bounds=[[-1.5, 1.5], [-1.0, 1.0]], n_trajs=50),
}


@dataclass
class Row:
    """One row: `system()` builds the system under learning. The random draws of a run come from
    `config.seed` (0 for every row)."""
    id: str
    system: Callable[[], Any]
    design: SamplingDesign
    config: LearnConfig

    @property
    def name(self):
        return ("INCR-" if self.config.method == "incremental" else "OPTI-") + self.id.split("/")[1]


def _benchmark_row(row_id, method, name):
    b = _BENCH[name]
    config = LearnConfig(
        method=method, n_modes=b['n_modes'], poly_degree=b['poly_degree'], boundary_degree=b['boundary_degree'],
        seg_tol=b['seg_tol'],
        optimization=OptimizationConfig(deltas='auto'),
        equivalence=EquivalenceConfig(enabled=True, bounds=b['bounds'], n_trajs=b['n_trajs'],
                                      max_counterexamples=5))
    design = SamplingDesign(time_step=b['time_step'], time_range=b['time_range'], x0s=b['x0s'],
                            test_time_step=b['test_time_step'], test_time_range=b['test_time_range'], test_x0s='random')
    return Row(row_id, getattr(systems, name), design, config)


def _parametric_row(row_id, method, nmodes):
    step = 0.001 if method == "incremental" else 0.01
    config = LearnConfig(
        method=method, n_modes=nmodes, poly_degree=1, boundary_degree=2, seg_tol=0.1,
        boundaries=BoundaryConfig(fine_step_factor=10 if nmodes >= 10 else 20),
        optimization=OptimizationConfig(deltas='auto'),
        simulation=SimulationConfig(max_step_factor=2),
        equivalence=EquivalenceConfig(enabled=True, bounds=[[-1.0, 1.0], [-1.0, 1.0]], n_trajs=100,
                                      max_counterexamples=10))
    design = SamplingDesign(time_step=step, time_range=1, x0s=[[-1.0, -1], [1, -1], [-1, 1]], test_time_step=step,
                            test_time_range=1, test_x0s='random')
    return Row(row_id, lambda n=nmodes: systems.parametric(n, 1.0), design, config)


def get_row(row_id):
    """The Row with this id (a fresh copy: changing it does not change the registry)."""
    if row_id not in ROW_IDS:
        raise KeyError(f"unknown row {row_id!r}; see hydyreg.benchmarks.rows.ROW_IDS")
    m, s = row_id.split("/")
    method = "incremental" if m == "incr" else "optimization"
    if s.startswith("parametric-"):
        return _parametric_row(row_id, method, int(s.split("-")[1]))
    return _benchmark_row(row_id, method, s)


def run_row(row, reporter=None, print_on=True):
    """Runs a row. Returns a LearnResult."""
    from ..learning.pipeline import learn_from_system
    return learn_from_system(row.system(), copy.copy(row.design), row.config, name=row.name, reporter=reporter,
                             print_on=print_on)
