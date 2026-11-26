"""Checks of the fixes made after the refactoring into the package (2026-11-16)."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from hydyreg.differentiation import diff_method_fps
from hydyreg.model import HybridRegressor
from hydyreg.trajectories import Trajectories


def _kinked_trajs():
    t = np.arange(60) * 0.1
    x1 = np.c_[np.abs(t - 3.0), np.sin(t)]
    x2 = np.c_[np.cos(t), np.abs(t - 2.0)]
    return Trajectories([t.copy(), t.copy()], [x1, x2])


def test_segment_leaves_input_unchanged_and_matches_in_place():
    trajs = _kinked_trajs()
    x_before = [x.copy() for x in trajs.x_list]
    new, segs = trajs.segment(0.5)
    assert not trajs.segmented
    assert all(np.array_equal(a, b) for a, b in zip(trajs.x_list, x_before))
    assert new.segmented and all(np.array_equal(a, b) for a, b in zip(new.og_x_list, x_before))
    old = _kinked_trajs()
    segs_old = old.segment_trajectories(0.5)
    assert segs == segs_old and any(len(s) for s in segs)
    assert all(np.array_equal(a, b) for a, b in zip(new.x_list, old.x_list))


def _random_model(deg, D=3, M=50, seed=0):
    rng = np.random.default_rng(seed)
    m = HybridRegressor(n=2, deg=deg)
    X = rng.normal(size=(M, D))
    Xt = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    m.Xfit_, m.Xfit_trans_ = X, Xt
    m.Omegas_ = [rng.normal(size=(D, Xt.shape[1])) for _ in range(2)]
    m.betas_ = [rng.normal(size=D) for _ in range(2)]
    labels = rng.integers(0, 2, M)
    ref = np.array([m.Omegas_[j] @ Xt[i] + m.betas_[j] for i, j in enumerate(labels)])
    return m, X, labels, ref


def test_vectorized_predict_and_rhs_match_row_loop():
    for deg in (1, 2, 3):
        m, X, labels, ref = _random_model(deg)
        assert np.allclose(m.predict(X, labels), ref, rtol=1e-12, atol=1e-12)
        assert np.allclose(m.predict(X, lambda x, X=X, l=labels: l[np.argmin(np.abs(X - x).sum(axis=1))]), ref,
                           rtol=1e-12, atol=1e-12)
        rhs = [m.make_eq_predict(j) for j in range(2)]
        got = np.array([rhs[j](0.0, X[i]) for i, j in enumerate(labels)])
        assert np.allclose(got, ref, rtol=1e-12, atol=1e-12)


def test_classify_modes_keeps_point_order():
    m, X, _, _ = _random_model(1, D=2, M=40)
    modes = (X[:, 0] > 0).astype(int)
    m.X_dot_fit_, m.modesfit_ = np.zeros_like(X), modes
    m.X_classify_, m.X_dot_classify_, m.modes_classify_ = X[:5], np.zeros((5, 2)), modes[:5]  # duplicates

    class Recorder:
        def fit(self, Xc, y):
            self.X = Xc
            return self
        def predict(self, Xc):
            return (Xc[:, 0] > 0).astype(int)

    rec = Recorder()
    m._install_region_functions = lambda *a, **k: None
    m.classify_modes(rec)
    assert np.array_equal(rec.X, X)


def test_fps_start_indices_over_several_trajectories():
    t = np.arange(30) * 0.1
    _, _, seg, _ = diff_method_fps([t, t], [np.c_[np.sin(t)], np.c_[np.cos(t)]], segments=[[10, 20], [15]])
    assert seg == [0, 6, 12, 18, 29]


def test_default_classifier_is_not_shared():
    a, X, _, _ = _random_model(1, D=2, M=40, seed=1)
    b, _, _, _ = _random_model(1, D=2, M=40, seed=2)
    for m in (a, b):
        m.X_dot_fit_, m.modesfit_ = np.zeros_like(m.Xfit_), (m.Xfit_[:, 0] > 0).astype(int)
        m.classify_modes()
    assert a.classifier_ is not b.classifier_
