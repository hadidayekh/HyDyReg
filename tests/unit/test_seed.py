"""The seed of a run (LearnConfig.seed): the library never uses NumPy's global random state, the same seed gives the
same run, and another seed gives other equivalence-query initial conditions."""
import numpy as np

from hydyreg.benchmarks.rows import get_row, run_row
from hydyreg.learning.equivalence import equivalence_query_by_traj


def _row(seed):
    row = get_row("opti/isolette")  # exercises the validation split, the IPOPT starts and the queries
    row.config = row.config.apply_overrides({"seed": seed, "equivalence.max_rounds": 2})
    return row


def test_global_state_untouched_and_runs_reproducible():
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    a = run_row(_row(0), print_on=False)
    assert np.array_equal(np.random.get_state()[1], before), "the run used NumPy's global random state"
    np.random.random(1000)  # changing the global state must not change the next run
    b = run_row(_row(0), print_on=False)
    keep = [0, 1, 2, 3, 4, 5, 9, 10]  # d1-d3 (avg, max), NEQ, d_dot: not the times
    assert np.array_equal(np.asarray(a.to_row())[keep], np.asarray(b.to_row())[keep])
    assert np.array_equal(a.model.Omegas_, b.model.Omegas_)


def test_query_initial_conditions_follow_the_generator():
    row = get_row("incr/isolette")
    sys_ = row.system()
    bounds = np.asarray(row.config.equivalence.bounds, dtype=float)

    def x0s(seed):
        out = equivalence_query_by_traj(sys_, sys_, 3, bounds, 0.25, 1.0, N_cexs=5, rng=np.random.default_rng(seed))
        return np.array([x[0] for x in out[2].x_list])  # conforming trajectories (the system against itself)
    assert np.array_equal(x0s(0), x0s(0))
    assert not np.array_equal(x0s(0), x0s(1))
