"""Ridge refit of the dynamics of a mode, with the strength selected on held-out segments."""
import numpy as np
from sklearn.base import BaseEstimator


RIDGE_ALPHAS = (0.0, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)


def ridge_refit(X_trans, X_dot, groups, alphas=RIDGE_ALPHAS):
    """
    Refit of the dynamics of one mode by ridge regression, with the regularization strength selected on
    held-out segments.

    Model: x_dot = W phi(x) + beta, with the polynomial features phi standardized over the mode's points
    (zero mean, unit variance), so that the penalty does not depend on the scale of the features:

        minimize   sum_i || x_dot_i - W phi~(x_i) - beta ||^2  +  alpha * N * ||W||_F^2

    alpha = 0 is ordinary least squares.

    Selection of alpha: the points are split into groups of consecutive points (`groups`: one identifier
    per point, normally the segment). Each group in turn is held out, the model is fitted on the others
    and evaluated on it; the alpha with the smallest total squared error is kept (the smallest alpha in
    case of a tie). Holding out whole segments tests how the model extrapolates along other pieces of
    trajectory, which a random split of the points does not.

    Holding out a whole segment only estimates the error of the final fit if the fits on the remaining
    points are as well determined as the fit on all points. When some fold would leave fewer than 2(Q+1)
    points (or with a single group), small blocks of consecutive points are held out instead (about a
    tenth of the points each, at least 2, never across two groups): the fold fits then resemble the final
    fit, and a final fit that is itself poorly determined still shows a large held-out error.

    If the smallest error is at the largest alpha of the grid, the selection is not trusted and alpha = 0
    is used: this happens when the fits on the folds are (almost) not determined (few points per
    coefficient), so that any amount of shrinkage looks better on the held-out group although the fit on
    all points is determined.

    Returns (coef (D, Q), intercept (D,), alpha, cv_errors (len(alphas),)); coef and intercept apply to
    the original (not standardized) features.
    """
    X_trans, X_dot, groups = np.asarray(X_trans, dtype=float), np.asarray(X_dot, dtype=float), np.asarray(groups)
    N, Q = X_trans.shape
    mu, sd = X_trans.mean(axis=0), X_trans.std(axis=0)
    sd[sd == 0] = 1.0
    Z = (X_trans - mu) / sd

    def solve(Zt, Yt, alpha):
        zm, ym = Zt.mean(axis=0), Yt.mean(axis=0)
        U, sv, Vt = np.linalg.svd(Zt - zm, full_matrices=False)
        if alpha > 0:
            filt = sv / (sv ** 2 + alpha * len(Zt))
        else:  # least squares (minimum norm if not determined)
            filt = np.where(sv > 1e-12 * sv.max(), 1.0 / np.maximum(sv, 1e-300), 0.0) if len(sv) and sv.max() > 0 else np.zeros_like(sv)
        W = (Vt.T * filt) @ (U.T @ (Yt - ym))   # (Q, D)
        return W, ym - zm @ W

    ids = np.unique(groups)
    blocks = [np.nonzero(groups == g)[0] for g in ids]
    if len(ids) < 2 or min(N - len(bl) for bl in blocks) < 2 * (Q + 1):
        size = max(2, int(np.ceil(N / 10)))
        blocks = [bl[k:k + size] for bl in blocks for k in range(0, len(bl), size)]
        if len(blocks) < 2:
            blocks = []
    cv = np.zeros(len(alphas))
    if len(blocks) >= 2:
        for test in blocks:
            train = np.setdiff1d(np.arange(N), test)
            if len(train) < 2 or len(test) == 0:
                continue
            for k, alpha in enumerate(alphas):
                W, b = solve(Z[train], X_dot[train], alpha)
                cv[k] += np.sum((X_dot[test] - Z[test] @ W - b) ** 2)
        k_best = int(np.argmin(cv))   # first minimum = smallest alpha in case of a tie
        if k_best == len(alphas) - 1 and len(alphas) > 1:
            k_best = 0
    else:
        k_best = 0
    W, b = solve(Z, X_dot, alphas[k_best])
    coef = (W / sd[:, None]).T
    return coef, b - coef @ mu, alphas[k_best], cv


class RidgeRegressor(BaseEstimator):
    """Regressor with the interface used by the incremental method (fit, predict, coef_, intercept_): ridge
    regression on standardized features with the strength selected on held-out blocks of consecutive
    points (`ridge_refit`). Strength 0, ordinary least squares, is among the candidates. Meant for noisy
    data, where least squares on a mode with few points extrapolates badly."""

    def __init__(self, alphas=RIDGE_ALPHAS):
        self.alphas = alphas

    def fit(self, X, y):
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        self.coef_, self.intercept_, self.alpha_, _ = ridge_refit(X, y, np.zeros(len(X), dtype=int), self.alphas)
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_.T + self.intercept_


def integral_refit(runs, poly, h, L=10, min_windows_factor=2):
    """
    Fit of one mode's dynamics x_dot = W phi(x) + beta from runs of consecutive samples, without derivatives:

        x(t_k + L h) - x(t_k) = (integral of phi(x) from t_k to t_k + L h) W^T + L h beta,

    the integrals by the trapezoid rule on the samples, windows of L samples overlapping by half, least squares.
    Measurement noise enters the increments with its own size instead of divided by the time step, so the fit
    stays accurate where derivative estimates are dominated by noise. Without noise the
    trapezoid rule adds a bias of order h^2, so it is meant for noisy data.

    runs: list of arrays (n_i, D) of consecutive samples (time step h). poly: fitted PolynomialFeatures.
    Returns (W, beta) or None when there are fewer than min_windows_factor x (number of parameters) windows.
    """
    A, Y = [], []
    for xr in runs:
        xr = np.asarray(xr, dtype=float)
        if len(xr) <= L:
            continue
        th = poly.transform(xr)
        cum = np.r_[np.zeros((1, th.shape[1])), np.cumsum((th[1:] + th[:-1]) * (h / 2), axis=0)]
        for k in range(0, len(xr) - L, max(1, L // 2)):
            A.append(np.r_[cum[k + L] - cum[k], L * h])
            Y.append(xr[k + L] - xr[k])
    if not A:
        return None
    A, Y = np.array(A), np.array(Y)
    if len(A) < min_windows_factor * A.shape[1]:
        return None
    C = np.linalg.lstsq(A, Y, rcond=None)[0]
    return C[:-1].T, C[-1]
