"""Configuration, noisy oracle and passive learning."""
import numpy as np
import pytest

from hydyreg.learning.config import LearnConfig, OptimizationConfig, EquivalenceConfig, SamplingDesign
from hydyreg.learning.oracle import noisy_oracle
from hydyreg.learning.pipeline import check_trajectories
from hydyreg.systems import SNDS_affine
from hydyreg.trajectories import Trajectories


def test_defaults_are_the_benchmark_values():
    c = LearnConfig()
    assert c.method == "incremental" and c.optimization.C_space == [0.0]
    assert c.optimization.deltas == "auto" and c.equivalence.max_counterexamples == 5 and c.equivalence.tol == 0.01
    assert not c.equivalence_enabled and c.update_dynamics
    assert LearnConfig(equivalence=EquivalenceConfig(bounds=[[0, 1]])).equivalence_enabled
    assert not LearnConfig(method="optimization", n_modes=2).update_dynamics


@pytest.mark.parametrize("kwargs, field", [
    (dict(method="foo"), "method"),
    (dict(method="optimization"), "n_modes"),
    (dict(optimization=OptimizationConfig(solver="gurobi")), "optimization.solver"),
    (dict(optimization=OptimizationConfig(refit="lasso")), "optimization.refit"),
    (dict(seg_tol=-1), "seg_tol"),
])
def test_validation_names_the_field(kwargs, field):
    with pytest.raises(ValueError, match=field.replace(".", r"\.")):
        LearnConfig(**kwargs)


def test_design_test_set_defaults_to_training_grid():
    d = SamplingDesign(time_step=0.1, time_range=2, x0s=[[0, 0]])
    assert d.test_time_step == 0.1 and d.test_time_range == 2 and d.test_x0s == "random"


def _system():
    params = np.array([[[-1.0, 0.5, 1.0], [-0.5, -1.0, 0.0]], [[-1.0, 0.0, -1.0], [0.0, -2.0, 0.5]]])
    ev = lambda t, x: x[0]
    ev.terminal, ev.direction = True, 0
    return SNDS_affine(params, [ev], lambda x: 0 if x[0] > 0 else 1)


def test_noisy_oracle_draws_like_the_former_closure():
    s = _system()
    o = noisy_oracle(s, 1e-3)
    a = o.simulate_trajectory(0.05, 1, [[0.5, 0.2]])
    b = o.simulate_trajectory(0.05, 1, [[-0.5, 0.2]])
    clean_a = s.simulate_trajectory(0.05, 1, [[0.5, 0.2]])
    clean_b = s.simulate_trajectory(0.05, 1, [[-0.5, 0.2]])
    rng = np.random.default_rng(12345)
    assert np.array_equal(a.x_list[0], clean_a.x_list[0] + 1e-3 * rng.normal(size=clean_a.x_list[0].shape))
    assert np.array_equal(b.x_list[0], clean_b.x_list[0] + 1e-3 * rng.normal(size=clean_b.x_list[0].shape))
    assert noisy_oracle(s, 0.0) is s


def test_check_trajectories():
    t = np.arange(10) * 0.1
    assert check_trajectories(Trajectories([t], [np.zeros((10, 2))])) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="not uniform"):
        check_trajectories(Trajectories([np.r_[t[:5], t[5:] + 0.01]], [np.zeros((10, 2))]))
    with pytest.raises(ValueError, match="at least 5"):
        check_trajectories(Trajectories([t[:3]], [np.zeros((3, 2))]))


def test_benchmark_rows_use_default_equivalence_tol():
    from hydyreg.benchmarks.rows import ROW_IDS, get_row
    assert {get_row(r).config.equivalence.tol for r in ROW_IDS} == {EquivalenceConfig().tol}
