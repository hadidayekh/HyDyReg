"""Mode classifiers.

- StandardizedLogisticRegression: L2-regularized multinomial logistic regression on standardized features with
  an exact Newton solver (default; polytope regions of thesis 4.3).
- DecisionListLogisticRegression: "if ... else if ... else" regions (option).
- PairwiseLogisticRegression: one-versus-one (option).
"""
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model._base import LinearClassifierMixin


class StandardizedLogisticRegression(LinearClassifierMixin, BaseEstimator):
    """Mode classifier: L2-regularized logistic regression fitted on standardized features.

    The weights are mapped back to the original feature space after fitting, so `coef_`,
    `intercept_`, `decision_function` and `predict` behave like those of scikit-learn's LogisticRegression
    (from LinearClassifierMixin; the class does not depend on LogisticRegression's parameters, which changed
    across scikit-learn versions)
    (boundary equations, exports and simulation events are unchanged).

    Unlike the unregularized classifier used in the thesis (`penalty='none'`), the fit has a
    unique solution even when the modes are linearly separable. Standardizing the features makes
    the single value of `C` meaningful across systems with different scales.

    The solution is computed with a damped Newton method on the dense Hessian (the problems are small:
    a few features per mode). It converges in a few tens of iterations to the optimum, runs on one BLAS
    thread, and its result does not depend on the number of BLAS threads of the process. scikit-learn's
    solvers (e.g. 'newton-cg', used in the thesis era) stop before the optimum on these problems (C = 1e6,
    separable data), either at `max_iter` or on a failed line search; scikit-learn is only used for the
    estimator interface (`predict`, `decision_function`, ...), not for fitting.

    As in scikit-learn, the model is binary (sigmoid) for two classes and multinomial (softmax)
    otherwise; `multi_class='multinomial'` with two classes fits the two-class softmax model.
    The objective is  C * sum_i loss_i + 0.5 * ||W||^2  (intercepts are not penalized).
    """

    def __init__(self, C=1e6, tol=1e-10, max_iter=10000, multi_class='auto'):
        self.C = C
        self.tol = tol            # kept for the estimator interface; the Newton method stops on its decrement
        self.max_iter = max_iter
        self.multi_class = multi_class

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=float)
        mu, sd = X.mean(axis=0), X.std(axis=0)
        sd[sd == 0] = 1.0
        if sample_weight is not None:
            raise NotImplementedError("sample_weight is not supported")
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1, user_api='blas'):
            self._fit_newton((X - mu) / sd, np.asarray(y))
        self.coef_ = self.coef_ / sd
        self.intercept_ = self.intercept_ - self.coef_ @ mu
        return self

    def _fit_newton(self, X, y, max_newton_iter=200):
        from scipy.special import logsumexp, expit
        classes, yi = np.unique(y, return_inverse=True)
        N, p = X.shape
        K = len(classes)
        if K < 2:
            raise ValueError("The classifier needs at least two classes.")
        A = np.c_[X, np.ones(N)]
        reg = np.r_[np.ones(p), 0.0]
        if K == 2:
            # two-class softmax with weights (-w, w) is the sigmoid model with v = 2w and 2C
            C = 2 * self.C if self.multi_class == 'multinomial' else self.C
            pos = yi == 1
            sign = np.where(pos, 1.0, -1.0)

            def fgh(w):
                z = A @ w
                f = C * np.sum(np.logaddexp(0, -sign * z)) + 0.5 * np.sum(reg * w * w)
                q = expit(-sign * z)  # probability of the wrong class (no cancellation when it is tiny)
                g = C * (A.T @ (-sign * q)) + reg * w
                H = C * (A.T * (q * (1 - q))) @ A + np.diag(reg)
                return f, g, H
            n_par = p + 1
            fix = None
        else:
            C = self.C
            regK = np.tile(reg, K)
            rows = np.arange(N)

            def fgh(w):
                W = w.reshape(K, p + 1)
                Z = A @ W.T
                lse = logsumexp(Z, axis=1)
                f = C * np.sum(lse - Z[rows, yi]) + 0.5 * np.sum(regK * w * w)
                P = np.exp(Z - lse[:, None])
                G = P.copy()
                G[rows, yi] = 0.0
                G[rows, yi] = -G.sum(axis=1)  # P - Y, without cancellation for the true class
                g = (C * (G.T @ A)).ravel() + regK * w
                H = np.empty((K * (p + 1), K * (p + 1)))
                for a in range(K):
                    for b in range(a, K):
                        blk = C * (A.T * (P[:, a] * ((a == b) - P[:, b]))) @ A
                        H[a*(p+1):(a+1)*(p+1), b*(p+1):(b+1)*(p+1)] = blk
                        H[b*(p+1):(b+1)*(p+1), a*(p+1):(a+1)*(p+1)] = blk.T
                H[np.diag_indices_from(H)] += regK
                return f, g, H
            n_par = K * (p + 1)
            # The objective does not change when the same constant is added to all intercepts. The
            # gradient has no component in this direction; this term keeps the Newton system regular.
            u = np.zeros(n_par)
            u[p::p + 1] = 1.0 / np.sqrt(K)
            fix = np.outer(u, u)

        w = np.zeros(n_par)
        f, g, H = fgh(w)
        eps = np.finfo(float).eps
        for it in range(1, max_newton_iter + 1):
            Hs = H if fix is None else H + (np.trace(H) / n_par) * fix
            try:
                step = -np.linalg.solve(Hs, g)
            except np.linalg.LinAlgError:
                step = -np.linalg.lstsq(Hs, g, rcond=None)[0]
            decrement = -(g @ step)  # ~ 2 (f - f_optimal) near the optimum
            if not decrement > 1e-13 * max(1.0, abs(f)):
                break
            a = 1.0
            while a > 1e-10:
                f2, g2, H2 = fgh(w + a * step)
                if np.isfinite(f2) and f2 <= f - 1e-4 * a * decrement + 8 * eps * abs(f):
                    break
                a *= 0.5
            else:
                break  # no further decrease possible in floating point
            w, f, g, H = w + a * step, f2, g2, H2
        W = w.reshape(-1, p + 1)
        self.classes_ = classes
        self.coef_ = W[:, :-1].copy()
        self.intercept_ = W[:, -1].copy()
        self.n_iter_ = np.array([it])
        self.n_features_in_ = p
        self.objective_ = f
        return self


def _binary_boundary(X, is_pos, C):
    """Binary StandardizedLogisticRegression of `is_pos` against the rest. Returns (w, b, errors, margin):
    g(x) = w.x + b > 0 means "positive"; `errors` counts misclassified points; `margin` is the smallest
    signed distance of a point to the boundary, measured in standardized features (negative if errors)."""
    y = is_pos.astype(int)
    c = StandardizedLogisticRegression(C=C).fit(X, y)
    w, b = c.coef_[0].copy(), float(c.intercept_[0])
    sign = np.where(is_pos, 1.0, -1.0)
    d = sign * (X @ w + b)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    norm = np.linalg.norm(w * sd)
    margin = float(d.min() / norm) if norm > 0 else -np.inf
    return w, b, int(np.sum(d <= 0)), margin


class _ScoreClassifier(BaseEstimator):
    """Common part of the classifiers below: `decision_function` returns one score per class (columns
    ordered as `classes_`), `predict` is the arg max. The scores are continuous, so the simulator's exact
    mode-change events (score of the current mode minus the largest other score) and Filippov sliding
    work unchanged."""

    def predict(self, X):
        return self.classes_[np.argmax(self.decision_function(X), axis=1)]

    def score(self, X, y):
        return float(np.mean(self.predict(X) == np.asarray(y)))


class PairwiseLogisticRegression(_ScoreClassifier):
    """One-versus-one mode classifier (option): one binary boundary per pair of modes.

    For each pair of modes (a, b), a binary `StandardizedLogisticRegression` is fitted on the points of
    these two modes only, giving d_ab(x) > 0 for "a rather than b" (d_ba = -d_ab). The score of mode k is

        s_k(x) = min over j != k of d_kj(x),

    so a mode that wins against every other mode has a positive score, and at most one mode can. Where
    the pairwise boundaries form a cycle no mode wins all its comparisons; the arg max of the scores still
    defines a unique mode there (the one whose worst comparison is the least bad), but the regions are
    then not intersections of half-spaces. This is why thesis 4.3.1.1 prefers the multinomial classifier.
    """

    def __init__(self, C=1e6):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("The classifier needs at least two classes.")
        p = X.shape[1]
        self.n_features_in_ = p
        self.W_ = np.zeros((K, K, p))  # W_[a, b], B_[a, b]: d_ab(x) = W_[a, b].x + B_[a, b]
        self.B_ = np.zeros((K, K))
        for a in range(K):
            for b in range(a + 1, K):
                sel = (y == self.classes_[a]) | (y == self.classes_[b])
                w, b0, _, _ = _binary_boundary(X[sel], y[sel] == self.classes_[a], self.C)
                self.W_[a, b], self.B_[a, b] = w, b0
                self.W_[b, a], self.B_[b, a] = -w, -b0
        self._diag = np.where(np.eye(K, dtype=bool), np.inf, 0.0)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        D = np.einsum('abp,np->nab', self.W_, X) + self.B_ + self._diag
        return D.min(axis=2)

    def describe(self, feature_names, fmt='{:.3f}'):
        K = len(self.classes_)
        lines = []
        for a in range(K):
            for b in range(a + 1, K):
                lines.append(f"boundary between modes {self.classes_[a] + 1} and {self.classes_[b] + 1}: "
                             + _linear_text(self.W_[a, b], self.B_[a, b], feature_names, fmt) + " = 0")
        return lines


class DecisionListLogisticRegression(_ScoreClassifier):
    """Mode classifier with "if ... else if ... else" regions (option).

    The regions are defined by an ordered list of modes pi_1, ..., pi_K and K-1 boundaries g_1, ..., g_{K-1}
    (each g_k linear in the classifier features):

        if g_1(x) > 0: mode pi_1;  else if g_2(x) > 0: mode pi_2;  ...  else: mode pi_K.

    Every region is an intersection of half-spaces in the classifier features, the regions partition the
    space without gaps or overlaps, and each boundary is independent of the others, so T-junctions of
    boundaries (which a multinomial classifier reaches only with infinite weights) are represented exactly.
    With two modes this is the binary logistic regression.

    Fitting is greedy: at each stage, every remaining mode is separated from the other remaining modes by a
    binary `StandardizedLogisticRegression`; the mode with the fewest training errors, then the largest
    margin (in standardized features), becomes the next "if"; its points are removed and the next stage
    starts. The last remaining mode is the "else".

    Scores, consistent with the rules (the arg max is the rule's mode, and a score changes sign exactly on
    the boundaries of its region):  s_{pi_k} = min(g_k, -g_1, ..., -g_{k-1}),  s_{pi_K} = min(-g_1, ..., -g_{K-1}).
    """

    def __init__(self, C=1e6):
        self.C = C

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        K = len(self.classes_)
        if K < 2:
            raise ValueError("The classifier needs at least two classes.")
        self.n_features_in_ = X.shape[1]
        remaining = list(range(K))
        order, W, B, stats = [], [], [], []
        while len(remaining) > 1:
            sel = np.isin(y, self.classes_[remaining])
            best = None
            for k in remaining:
                w, b, err, margin = _binary_boundary(X[sel], y[sel] == self.classes_[k], self.C)
                key = (err, -margin)
                if best is None or key < best[0]:
                    best = (key, k, w, b, err, margin)
            _, k, w, b, err, margin = best
            order.append(k)
            W.append(w)
            B.append(b)
            stats.append(dict(errors=err, margin=margin, n_points=int(sel.sum())))
            remaining.remove(k)
        order.append(remaining[0])
        self.order_ = np.array(order)          # indices into classes_, "if" modes first, "else" mode last
        self.W_ = np.array(W)                  # (K-1) x p
        self.B_ = np.array(B)                  # K-1
        self.stages_ = stats
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        G = X @ self.W_.T + self.B_            # N x (K-1)
        K = len(self.classes_)
        S = np.empty((len(X), K))
        neg_prefix = np.full(len(X), np.inf)    # min(-g_1, ..., -g_{k-1})
        for k in range(K - 1):
            S[:, self.order_[k]] = np.minimum(G[:, k], neg_prefix)
            neg_prefix = np.minimum(neg_prefix, -G[:, k])
        S[:, self.order_[K - 1]] = neg_prefix
        return S

    def describe(self, feature_names, fmt='{:.3f}'):
        lines = []
        for k, (w, b) in enumerate(zip(self.W_, self.B_)):
            lines.append(("if " if k == 0 else "else if ") + _linear_text(w, b, feature_names, fmt)
                         + f" > 0: mode {self.classes_[self.order_[k]] + 1}")
        lines.append(f"else: mode {self.classes_[self.order_[-1]] + 1}")
        return lines


def _linear_text(w, b, names, fmt):
    """w.x + b as text, scaled so that the largest coefficient has magnitude 1."""
    scale = np.max(np.abs(w)) if np.max(np.abs(w)) > 0 else 1.0
    w, b = w / scale, b / scale
    s = ' '.join(f"{'+' if c >= 0 else '-'} {fmt.format(abs(c))} {n}" for c, n in zip(w, names))
    return (s + f" {'+' if b >= 0 else '-'} {fmt.format(abs(b))}").lstrip('+ ')
