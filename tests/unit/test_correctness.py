"""Correctness checks against known answers or independent references (not regression snapshots):
the Newton logistic solver, switch location, noise estimation, window selection, ridge refit and the decision list."""
import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.special import expit

from hydyreg.classifier import StandardizedLogisticRegression, DecisionListLogisticRegression
from hydyreg.differentiation import estimate_noise_std, choose_derivative_window
from hydyreg.segmentation import choose_segmentation_window
from hydyreg.model import HybridRegressor
from hydyreg.regression import ridge_refit


# ------------------------------------------------------------------ Newton solver of the logistic regression

def _objective(clf, X, y):
    """Objective of StandardizedLogisticRegression at its solution, in standardized features:
    C * sum_i loss_i + 0.5 * ||W||^2 (intercepts not penalized). Binary model for two classes."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    W = clf.coef_ * sd                       # back to standardized features
    b = clf.intercept_ + clf.coef_ @ mu
    classes = np.unique(y)
    if len(classes) == 2:
        z = Z @ W[0] + b[0]
        s = np.where(y == classes[1], 1.0, -1.0)
        return clf.C * np.sum(np.logaddexp(0, -s * z)) + 0.5 * np.sum(W ** 2)
    S = Z @ W.T + b
    idx = np.searchsorted(classes, y)
    lse = np.logaddexp.reduce(S, axis=1)
    return clf.C * np.sum(lse - S[np.arange(len(y)), idx]) + 0.5 * np.sum(W ** 2)


@pytest.mark.parametrize("K", [2, 3])
def test_newton_reaches_the_optimum_found_by_scipy(K):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 3)) * [1.0, 10.0, 0.1]
    y = np.argmax(X @ rng.normal(size=(3, K)) + rng.gumbel(size=(300, K)), axis=1)
    clf = StandardizedLogisticRegression(C=2.0).fit(X, y)
    f_newton = _objective(clf, X, y)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = (X - mu) / sd
    classes = np.unique(y)
    if K == 2:
        s = np.where(y == classes[1], 1.0, -1.0)

        def f(p):
            return 2.0 * np.sum(np.logaddexp(0, -s * (Z @ p[:3] + p[3]))) + 0.5 * np.sum(p[:3] ** 2)
        n = 4
    else:
        def f(p):
            W, b = p[:9].reshape(3, 3), p[9:]
            S = Z @ W.T + b
            return 2.0 * np.sum(np.logaddexp.reduce(S, axis=1) - S[np.arange(len(y)), y]) + 0.5 * np.sum(W ** 2)
        n = 12
    ref = minimize(f, np.zeros(n), method="BFGS", options=dict(gtol=1e-10, maxiter=10000))
    assert f_newton <= ref.fun + 1e-7 * abs(ref.fun)


def test_newton_stationary_on_separable_data():
    """Separable data with C = 1e6: the regularized optimum exists; the gradient vanishes there and every
    training point is classified correctly."""
    rng = np.random.default_rng(2)
    X = rng.uniform(-1, 1, size=(400, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0.1).astype(int)
    clf = StandardizedLogisticRegression().fit(X, y)
    assert np.all(clf.predict(X) == y)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Z = np.c_[(X - mu) / sd, np.ones(len(X))]
    w = np.r_[clf.coef_[0] * sd, clf.intercept_[0] + clf.coef_[0] @ mu]
    s = np.where(y == 1, 1.0, -1.0)
    q = expit(-s * (Z @ w))
    grad = clf.C * (Z.T @ (-s * q)) + np.r_[w[:2], 0.0]
    assert np.max(np.abs(grad)) < 1e-6 * max(1.0, np.max(np.abs(w)))


# ------------------------------------------------------------------ switch location from the learned modes

def _two_mode_model():
    m = HybridRegressor(n=2, deg=1)
    m.Omegas_ = np.array([[[-1.0, 2.0], [-2.0, -1.0]], [[0.5, -1.0], [3.0, 0.2]]])
    m.betas_ = np.array([[1.0, 0.0], [-2.0, 1.0]])
    return m


def _flow(m, j, x0, t):
    sol = solve_ivp(lambda _, x: m.Omegas_[j] @ x + m.betas_[j], (0, t), x0, rtol=1e-12, atol=1e-14)
    return sol.y[:, -1]


@pytest.mark.parametrize("tau_true", [0.03, 0.05, 0.08])
def test_locate_switch_finds_the_true_switch_time(tau_true):
    m = _two_mode_model()
    x_A, T = np.array([1.0, 2.0]), 0.1
    x_s = _flow(m, 0, x_A, tau_true)
    x_B = _flow(m, 1, x_s, T - tau_true)
    found = m.locate_switch(x_A, x_B, T, 0, 1)
    assert found is not None
    tau, x_switch = found
    assert abs(tau - tau_true) < 1e-6 * T
    assert np.allclose(x_switch, x_s, atol=1e-8)


def test_locate_switch_refuses_identical_modes():
    m = _two_mode_model()
    x_A, T = np.array([1.0, 2.0]), 0.1
    assert m.locate_switch(x_A, _flow(m, 0, x_A, T), T, 0, 0) is None


# ------------------------------------------------------------------ noise estimation and window selection

def _smooth(n=2000, h=0.01):
    t = np.arange(n) * h
    return t, np.c_[np.sin(1.3 * t) + 0.2 * t, np.cos(0.7 * t) * np.exp(-0.05 * t)]


@pytest.mark.parametrize("sigma", [1e-4, 1e-3, 1e-2])
def test_estimate_noise_std_recovers_sigma(sigma):
    _, x = _smooth()
    xn = x + sigma * np.random.default_rng(3).normal(size=x.shape)
    assert abs(estimate_noise_std([xn]) - sigma) < 0.1 * sigma


def test_estimate_noise_std_is_negligible_without_noise():
    _, x = _smooth()
    assert estimate_noise_std([x]) < 1e-9


def test_windows_stay_at_five_samples_without_noise_and_grow_with_noise():
    t, x = _smooth()
    h = t[1] - t[0]
    assert choose_segmentation_window([x], h, 0.01, 0.0)[0] == 2
    assert choose_derivative_window([t], [x], [[]], 0.0)[0] == 2
    sigma = 1e-3
    xn = x + sigma * np.random.default_rng(4).normal(size=x.shape)
    assert choose_segmentation_window([xn], h, 0.01, sigma)[0] > 2
    assert choose_derivative_window([t], [xn], [[]], sigma)[0] > 2


# ------------------------------------------------------------------ ridge refit

def test_ridge_refit_is_least_squares_on_clean_data():
    rng = np.random.default_rng(5)
    X = rng.uniform(-1, 1, size=(120, 3))
    W, b = rng.normal(size=(2, 3)), rng.normal(size=2)
    Y = X @ W.T + b
    coef, icpt, alpha, _ = ridge_refit(X, Y, np.repeat(np.arange(6), 20))
    assert alpha == 0
    assert np.allclose(coef, W, atol=1e-10) and np.allclose(icpt, b, atol=1e-10)


# ------------------------------------------------------------------ decision list on a T-junction

def test_decision_list_represents_a_t_junction():
    """Regions of affine_3modes: x1 >= 1: B; else x2 <= 2: A; else C. The decision list classifies a dense grid
    correctly away from the boundaries (the multinomial model can only approach this with infinite weights)."""
    def label(P):
        return np.where(P[:, 0] >= 1, 1, np.where(P[:, 1] <= 2, 0, 2))
    rng = np.random.default_rng(6)
    P = rng.uniform([-2, -1], [3, 8], size=(1500, 2))
    clf = DecisionListLogisticRegression().fit(P, label(P))
    g = np.stack(np.meshgrid(np.linspace(-2, 3, 101), np.linspace(-1, 8, 181)), -1).reshape(-1, 2)
    far = (np.abs(g[:, 0] - 1) > 0.05) & ((g[:, 0] >= 1) | (np.abs(g[:, 1] - 2) > 0.05))
    assert np.all(clf.predict(g[far]) == label(g[far]))


# ------------------------------------------------------------------ two modes with score-per-mode classifiers

@pytest.mark.parametrize("kind", ["decision_list", "pairwise"])
def test_two_mode_model_with_score_classifier_simulates(kind):
    """With two modes, the decision list and one-versus-one give one score per mode; the event function must still
    be a scalar (it was an array: the simulation of every two-mode system failed with these classifiers)."""
    from hydyreg.learning.config import ClassifierConfig
    m = _two_mode_model()
    rng = np.random.default_rng(7)
    X = rng.uniform(-2, 2, size=(300, 2))
    modes = (X[:, 0] + 0.3 * X[:, 1] > 0).astype(int)
    m.Xfit_ = X
    m.classify_modes(ClassifierConfig(kind=kind).build(), X=X, modes=modes)
    assert np.ndim(m.eventlist_(0.0, np.array([0.5, 0.5]))) == 0
    traj = m.to_SNDS().simulate_trajectory(0.05, 1.0, [[1.0, 1.0]])
    assert len(traj.x_list[0]) == 21


# ------------------------------------------------------------------ integral refit under noise

def test_integral_refit_recovers_the_field():
    """Mode 1 of poly3 (x1' = -0.1 x1^2 + 0.06 x2^3, x2' = -0.7 x1, cubic features) from three runs of 60 samples:
    the integral fit (no derivatives) recovers the field along the data, nearly exactly without noise (trapezoid
    bias only) and within a few percent with noise."""
    from sklearn.preprocessing import PolynomialFeatures
    from hydyreg.regression import integral_refit

    def f(t, x):
        return [-0.1 * x[0] ** 2 + 0.06 * x[1] ** 3, -0.7 * x[0]]
    h, rng = 0.04, np.random.default_rng(8)
    clean = []
    for x0 in ([0.5, -1.0], [-0.3, -1.5], [0.8, -0.6]):
        t = np.arange(60) * h
        clean.append(solve_ivp(f, (0, t[-1]), x0, t_eval=t, rtol=1e-10, atol=1e-12).y.T)
    poly = PolynomialFeatures(degree=3, include_bias=False).fit(np.zeros((1, 2)))
    G = np.concatenate(clean)
    F = np.array([f(0, g) for g in G])

    def err(fit):
        W, b = fit
        return np.mean(np.linalg.norm(poly.transform(G) @ W.T + b - F, axis=1)) / np.mean(np.linalg.norm(F, axis=1))
    assert err(integral_refit(clean, poly, h, L=10)) < 2e-3
    noisy = [x + 2e-3 * rng.normal(size=x.shape) for x in clean]
    assert err(integral_refit(noisy, poly, h, L=10)) < 0.05
    assert integral_refit([c[:15] for c in clean], poly, h, L=10) is None  # too few windows for 10 parameters
