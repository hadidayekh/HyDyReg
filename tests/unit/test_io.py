"""CSV input and JSON persistence of learned models."""
from pathlib import Path

import numpy as np
import pytest
from sklearn.preprocessing import PolynomialFeatures

from hydyreg.classifier import StandardizedLogisticRegression, DecisionListLogisticRegression, PairwiseLogisticRegression
from hydyreg.model import HybridRegressor
from hydyreg.persistence import save_model, load_model
from hydyreg.trajectories import Trajectories

DATA = Path(__file__).resolve().parents[1] / "data"


def test_from_csv_with_header_and_glob():
    tr = Trajectories.from_csv(str(DATA / "isolette_*.csv"))
    assert tr.num_trajectories == 2 and tr.d == 2 and tr.names == ["T_box", "T_heater"]
    assert tr.check_uniform_step() == pytest.approx(0.25)
    tr2 = Trajectories.from_csv(str(DATA / "isolette_1.csv"), time_column="t", state_columns=["T_heater"])
    assert tr2.d == 1 and np.array_equal(tr2.x_list[0][:, 0], tr.x_list[0][:, 1])


def test_from_csv_variants(tmp_path):
    t = np.arange(20) * 0.1
    x = np.c_[np.sin(t), np.cos(t)]
    np.savetxt(tmp_path / "a.txt", x)                                  # whitespace, no header, no time column
    tr = Trajectories.from_csv(str(tmp_path / "a.txt"), dt=0.1)
    assert np.allclose(tr.t_list[0], t) and np.array_equal(tr.x_list[0], x) and tr.names == ["x1", "x2"]
    ids = np.r_[np.zeros(10), np.ones(10)]
    np.savetxt(tmp_path / "b.csv", np.c_[ids, t, x], delimiter=",", header="id,time,u,v", comments="")
    tr = Trajectories.from_csv(str(tmp_path / "b.csv"), time_column="time", trajectory_column="id")
    assert tr.num_trajectories == 2 and len(tr.t_list[1]) == 10 and tr.names == ["u", "v"]
    paths = tr.to_csv(tmp_path / "out")
    back = Trajectories.from_csv([str(p) for p in paths])
    assert np.array_equal(back.x_list[1], tr.x_list[1])
    np.savetxt(tmp_path / "c.csv", np.c_[np.r_[t[:10], t[10:] + 0.05], x], delimiter=",")
    with pytest.raises(ValueError, match="not uniform"):
        Trajectories.from_csv(str(tmp_path / "c.csv")).check_uniform_step()
    with pytest.raises(FileNotFoundError):
        Trajectories.from_csv(str(tmp_path / "missing_*.csv"))


def _model(n, deg, bdeg, classifier):
    """A model with given dynamics and a classifier fitted on labelled points (regions: sectors / bands)."""
    rng = np.random.default_rng(0)
    m = HybridRegressor(n=n, deg=deg, classifier_poly_deg=bdeg)
    X = rng.uniform(-2, 2, (400, 2))
    m.poly = PolynomialFeatures(degree=deg, include_bias=False).fit(X)
    Q = m.poly.n_output_features_ if deg > 1 else 2
    m.Omegas_ = rng.normal(size=(n, 2, Q)) * 0.5 - np.eye(2, Q)[None] * 1.5
    m.betas_ = rng.normal(size=(n, 2))
    modes = np.where(X[:, 0] > 0.5, 0, np.where(X[:, 1] > 0, 1, 2)) % n
    keep = (np.abs(X[:, 0] - 0.5) > 0.05) & (np.abs(X[:, 1]) > 0.05)
    X, modes = X[keep], modes[keep]
    m.Xfit_, m.Xfit_trans_ = X, (m.poly.transform(X) if deg > 1 else X)
    m.X_dot_fit_, m.modesfit_ = np.zeros_like(X), modes
    m.classify_modes(classifier, X=X, modes=modes, X_dot=np.zeros_like(X))
    return m


@pytest.mark.parametrize("n, deg, bdeg, clf", [
    (2, 1, 1, StandardizedLogisticRegression()),
    (3, 1, 1, StandardizedLogisticRegression(multi_class="multinomial")),
    (2, 2, 2, StandardizedLogisticRegression()),
    (3, 2, 2, DecisionListLogisticRegression()),
    (3, 1, 1, PairwiseLogisticRegression()),
], ids=["2modes", "3modes", "deg2", "decision_list", "pairwise"])
def test_save_load_round_trip(tmp_path, n, deg, bdeg, clf):
    m = _model(n, deg, bdeg, clf)
    save_model(m, tmp_path / "m.json", names=["a", "b"])
    m2 = load_model(tmp_path / "m.json")
    X = np.random.default_rng(1).uniform(-2, 2, (50, 2))
    assert np.array_equal(m.predict(X), m2.predict(X))
    assert [m.labelfun_(x) for x in X] == [m2.labelfun_(x) for x in X]
    t1 = m.to_SNDS().simulate_trajectory(0.01, 2, [[1.5, 1.5], [-1.5, -1.0]])
    t2 = m2.to_SNDS().simulate_trajectory(0.01, 2, [[1.5, 1.5], [-1.5, -1.0]])
    for a, b in zip(t1.x_list, t2.x_list):
        assert np.array_equal(a, b)  # bit for bit
    assert m2.names_ == ["a", "b"] and m2.gen_full_latex_eq()
