"""Checks of the robust incremental update (HybridRegressor.update_continuous) and of the mode classifiers.

The checks share one random generator, used in file order, so their data are created at import time in that
order (as when this file was a script); each check is then reported as its own test by `test_check`.

Uses exact derivatives of two affine modes in 2D:  f0(x) = A0 x + b0,  f1(x) = A1 x + b1.
"""
import warnings

warnings.simplefilter("ignore")

import numpy as np  # noqa: E402
from hydyreg.model import HybridRegressor  # noqa: E402
from hydyreg.classifier import StandardizedLogisticRegression  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

A = [np.array([[-1.0, 2.0], [-2.0, -1.0]]), np.array([[0.5, -1.0], [3.0, 0.2]]), np.array([[0.0, 1.0], [-4.0, 0.3]])]
b = [np.array([1.0, 0.0]), np.array([-2.0, 1.0]), np.array([0.5, 0.5])]
rng = np.random.default_rng(0)
RESULTS = {}


def check(name, cond, detail=""):
    """Records a check; `test_check` asserts it."""
    assert name not in RESULTS, f"duplicate check name: {name}"
    RESULTS[name] = (bool(cond), detail)


def seg(mode, n, center=(0.0, 0.0)):
    """n points of a smooth curve with the exact derivatives of `mode`."""
    s = np.linspace(0, 1, n)
    X = np.c_[center[0] + np.cos(2 * s + rng.random()), center[1] + s + np.sin(3 * s)]
    return X, X @ A[mode].T + b[mode]


def model_with(*modes, n=30):
    m = HybridRegressor(deg=1)
    Xs, Xds = zip(*[seg(j, n) for j in modes])
    X, Xd = np.concatenate(Xs), np.concatenate(Xds)
    m.fit_first_incremental(X, Xd, [n * i for i in range(len(modes))])
    return m


def mixed(parts):
    Xs, Xds = zip(*[seg(j, n) for j, n in parts])
    return np.concatenate(Xs), np.concatenate(Xds)


# 1. a segment with an undetected switch between two known modes is split, no mode is created
m = model_with(0, 1)
n_before = len(m.Xfit_)
X, Xd = mixed([(0, 40), (1, 25)])
m.update_continuous(X, Xd)
new = m.modesfit_[n_before:]
check("mixed segment (known/known) is split", m.n == 2 and set(new) == {0, 1} and len(m.pending_) == 0,
      f"modes={m.n}, assigned {np.bincount(new).tolist()} of 65 points, gaps recorded={len(m.split_gaps_)}")
check("points next to the recovered switch are dropped", np.sum(new == 0) <= 38 and np.sum(new == 1) <= 23)

# 2. known mode followed by an unknown mode: the unknown part founds the new mode
m = model_with(0)
X, Xd = mixed([(0, 40), (1, 30)])
m.update_continuous(X, Xd)
err = np.abs(m.Omegas_[1] - A[1]).max() if m.n == 2 else np.inf
check("mixed segment (known/unknown): new mode from the unexplained part", m.n == 2 and err < 1e-8,
      f"modes={m.n}, coefficient error={err:.1e}")

# 3. three switches inside one segment
m = model_with(0, 1)
X, Xd = mixed([(0, 20), (1, 20), (0, 20), (1, 20)])
m.update_continuous(X, Xd)
check("segment with three undetected switches", m.n == 2 and len(m.split_gaps_) == 3, f"modes={m.n}")

# 4. a short segment of an unknown mode is set aside; two of them found the mode together
m = model_with(0)
X1, Xd1 = seg(2, 4)
m.update_continuous(X1, Xd1)
check("short unexplained segment is set aside", m.n == 1 and len(m.pending_) == 1)
X2, Xd2 = seg(2, 4, center=(1.0, -1.0))
m.update_continuous(X2, Xd2)
m.retry_pending()
err = np.abs(m.Omegas_[1] - A[2]).max() if m.n == 2 else np.inf
check("short segments found a mode jointly", m.n == 2 and len(m.pending_) == 0 and err < 1e-8,
      f"modes={m.n}, coefficient error={err:.1e}")

# 5. a mixed segment of two unknown modes does not found a mode
m = model_with(0)
X, Xd = mixed([(1, 20), (2, 20)])
m.update_continuous(X, Xd)
check("mixed segment of unknown modes is set aside", m.n == 1 and len(m.pending_) == 1)

# 6. first hypothesis: a leading mixed segment does not found the first mode
X, Xd = mixed([(0, 15), (1, 15), (0, 30), (1, 30)])
m = HybridRegressor(deg=1)
m.fit_first_incremental(X, Xd, [0, 30, 60])
check("first hypothesis skips a leading mixed segment", m.n == 2 and len(m.pending_) == 0, f"modes={m.n}")

# 7. merging two modes that one model captures (Remark 17)
m = model_with(0, 1)
X, Xd = seg(0, 30, center=(3.0, 3.0))
m.n, m.Omegas_, m.betas_ = 3, np.r_[m.Omegas_, m.Omegas_[:1]], np.r_[m.betas_, m.betas_[:1]]
m.regressors_.append(m.regressors_[0])
m.Xfit_, m.Xfit_trans_, m.X_dot_fit_ = np.r_[m.Xfit_, X], np.r_[m.Xfit_trans_, X], np.r_[m.X_dot_fit_, Xd]
m.modesfit_ = np.r_[m.modesfit_, np.full(30, 2)]
k = m.merge_similar_modes()
check("merge_similar_modes merges a duplicated mode", k == 1 and m.n == 2 and set(m.modesfit_) == {0, 1})

# 7b. clusterwise regression (optimization-based method): three modes, twelve segments
parts = [(j, 12) for j in (0, 1, 2, 1, 0, 2, 2, 0, 1, 0, 1, 2)]
X, Xd = mixed(parts)
seg_of_point = np.repeat(np.arange(len(parts)), 12)
truth = np.array([j for j, _ in parts])
res = []
for th in (1, 4):
    with threadpool_limits(limits=th):
        m = HybridRegressor(deg=1, n=3)
        m.fit_clusterwise(X, Xd, seg_of_point)
        res.append(np.r_[m.Omegas_.ravel(), m.betas_.ravel()])
lam = np.argmin(m.Xis_segments_, axis=0)
same_partition = len(set(zip(lam, truth))) == 3
err = min(np.abs(m.Omegas_[j] - A[0]).max() for j in range(3))
check("fit_clusterwise recovers the mode of every segment", same_partition and err < 1e-8,
      f"objective={m.objective_:.1e}, best objective reached by {np.sum(m.restart_objectives_ < 1e-12)} of 200 starts")
check("fit_clusterwise identical with 1 and 4 BLAS threads (bitwise)", np.array_equal(res[0], res[1]))

# 7b2. a training segment with an undetected switch, model fitted with one mode per segment
parts = [(0, 30), (1, 30), (0, 30), (1, 30)]
X, Xd = mixed(parts + [(0, 25), (1, 20)])           # the last segment is mixed: 25 points of mode 0, 20 of mode 1
seg_of_point = np.r_[np.repeat(np.arange(4), 30), np.full(45, 4)]
m = HybridRegressor(deg=1, n=2)
m.fit_clusterwise(X, Xd, seg_of_point)
err_before = max(min(np.abs(m.Omegas_[j] - A[k]).max() for j in range(2)) for k in range(2))
k = m.split_mixed_segments(seg_of_point)
err_after = max(min(np.abs(m.Omegas_[j] - A[k]).max() for j in range(2)) for k in range(2))
check("split_mixed_segments repairs a mixed training segment", k == 1 and err_after < 1e-8 and err_before > 1e-3 and len(m.split_gaps_) == 1,
      f"coefficient error {err_before:.1e} -> {err_after:.1e}, {len(m.Xfit_)} of {len(X)} points kept")

# 7b3. imposed number of modes: a segment of an unknown mode is set aside, not turned into a mode
m.max_modes, m.regressors_ = 2, [None, None]
X3, Xd3 = seg(2, 30)
m.update_continuous(X3, Xd3)
check("with an imposed number of modes, no mode is created", m.n == 2 and len(m.pending_) == 1)

# 7b4. ridge refit: least squares on clean data, regularization selected with noise and too many features
from hydyreg.regression import ridge_refit  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402
Xs_, Xds_ = zip(*[seg(0, 12, center=(c, -c)) for c in (0.0, 0.4, 0.8)])
Xr, Xdr = np.concatenate(Xs_), np.concatenate(Xds_)
gr = np.repeat(np.arange(3), 12)
coef, icpt, alpha, _ = ridge_refit(Xr, Xdr, gr)
check("ridge_refit selects least squares on exact data", alpha == 0 and np.abs(coef - A[0]).max() < 1e-8, f"alpha = {alpha}")
Xq_ = rng.uniform(-3, 4, size=(2000, 2)); Fq_ = Xq_ @ A[0].T + b[0]
P4 = PolynomialFeatures(degree=4, include_bias=False)
Xdn = Xdr + 0.05 * rng.normal(size=Xdr.shape)
coef, icpt, alpha, _ = ridge_refit(P4.fit_transform(Xr), Xdn, gr)
err_ridge = np.sqrt(np.mean((P4.transform(Xq_) @ coef.T + icpt - Fq_) ** 2))
coef0, icpt0, _, _ = ridge_refit(P4.fit_transform(Xr), Xdn, gr, alphas=(0.0,))
err_ls = np.sqrt(np.mean((P4.transform(Xq_) @ coef0.T + icpt0 - Fq_) ** 2))
check("ridge_refit regularizes noisy data with too high a degree", alpha > 0 and err_ridge < err_ls,
      f"alpha = {alpha:g}, error on a wider region: least squares {err_ls:.1e}, ridge {err_ridge:.1e}")

# 7c. thesis program with several starts: the kept solution has the lowest objective; same starts for every seed
parts = [(j, 15) for j in (0, 1, 0, 1, 1, 0)]
X, Xd = mixed(parts)
seg_of_point = np.repeat(np.arange(len(parts)), 15)
m = HybridRegressor(deg=1, n=2)
m.fit_opti_multistart(X, Xd, n_starts=3, seed=0, stop_tol=None, C=0, deltas=0.001, segments=seg_of_point,
                      apply_LR=True)
lam = np.argmin(m.Xis_segments_, axis=0)
check("fit_opti_multistart keeps the lowest objective", m.objective_ == np.nanmin(m.start_objectives_) and m.n_starts_run_ == 3,
      f"objectives {np.array2string(m.start_objectives_, precision=2)}")
check("fit_opti_multistart finds the mode of every segment", len(set(zip(lam, [j for j, _ in parts]))) == 2)
m2 = HybridRegressor(deg=1, n=2)
m2.fit_opti_multistart(X, Xd, n_starts=3, seed=0, stop_tol=None, C=0, deltas=0.001, segments=seg_of_point,
                       apply_LR=True)
check("fit_opti_multistart: same seed gives the same starts", np.array_equal(m.start_objectives_, m2.start_objectives_))

# 7d. estimated error of the five-point derivatives matches the true error (smooth trajectory)
from hydyreg.differentiation import estimate_derivative_error, auto_deltas, diff_method_fps  # noqa: E402
tt = np.linspace(0, 6, 121)
xx = np.c_[np.sin(3 * tt), np.exp(0.3 * tt)]
est = estimate_derivative_error([tt], [xx], [[]])
Xe, Xde, _, _ = diff_method_fps([tt], [xx], segments=[[]])
true_d = np.c_[3 * np.cos(3 * tt), 0.3 * np.exp(0.3 * tt)][2:-2]
true_err = np.abs(Xde - true_d).max(axis=1)[1:-1]  # points where the seven-point estimate exists
rel = np.abs(est - true_err) / true_err
check("estimate_derivative_error matches the true error of the five-point derivatives", np.median(rel) < 0.05,
      f"median relative deviation {np.median(rel):.3f}, max error {true_err.max():.1e}")
check("auto_deltas = twice the largest estimated error", abs(auto_deltas([tt], [xx], [[]], Xde) - 2 * est.max()) < 1e-15)

# 8. classifier: the Newton solver reaches the optimum scikit-learn reaches on an easy problem
from sklearn.linear_model import LogisticRegression  # noqa: E402
Xc = rng.normal(size=(400, 3))
for K in (2, 3):
    logits = Xc @ rng.normal(size=(3, K))
    yc = np.argmax(logits + rng.gumbel(size=logits.shape), axis=1)
    ours = StandardizedLogisticRegression(C=1.0).fit(Xc, yc)
    # reference: scikit-learn on the same standardized features, weights mapped back to the original features
    mu, sd = Xc.mean(axis=0), Xc.std(axis=0)
    sk = LogisticRegression(C=1.0, tol=1e-12, solver="newton-cg", max_iter=10000).fit((Xc - mu) / sd, yc)
    ref_coef = sk.coef_ / sd
    d = np.abs(ours.coef_ - ref_coef).max()
    check(f"Newton solver matches scikit-learn, {K} classes (C=1, non-separable)", d < 1e-6, f"max |coef diff| = {d:.1e}")

# 9. separable data, C = 1e6: all points classified, same bits with 1 and 4 BLAS threads
Xc = rng.uniform(-1, 1, size=(600, 2)) * [1.0, 500.0]
yc = (Xc[:, 1] > 100 * Xc[:, 0] - 150).astype(int) + (Xc[:, 1] > 100 * Xc[:, 0] + 150)  # three parallel bands
res = []
for th in (1, 4):
    with threadpool_limits(limits=th):
        c = StandardizedLogisticRegression(multi_class="multinomial").fit(Xc, yc)
        res.append(np.r_[c.coef_.ravel(), c.intercept_])
check("separable data: training accuracy 1", np.mean(c.predict(Xc) == yc) == 1.0, f"{int(c.n_iter_[0])} iterations")
check("classifier identical with 1 and 4 BLAS threads (bitwise)", np.array_equal(res[0], res[1]))

# 10. classifiers for "if ... else if ... else" regions (T-junction of boundaries)
from hydyreg.classifier import DecisionListLogisticRegression, PairwiseLogisticRegression  # noqa: E402
Xc = rng.uniform(-2, 2, size=(800, 2))
Xc = Xc[(np.abs(Xc[:, 0] - 1) > 0.02) & (np.abs(Xc[:, 1]) > 0.02)]
yc = np.where(Xc[:, 0] >= 1, 1, np.where(Xc[:, 1] <= 0, 0, 2))
dl = DecisionListLogisticRegression().fit(Xc, yc)
check("decision list: training accuracy 1 on a T-junction", dl.score(Xc, yc) == 1.0)
check("decision list: finds the order of the rules (x1 >= 1 first, 'else' = mode 2)", list(dl.classes_[dl.order_]) == [1, 0, 2], str(dl.describe(["x1", "x2"])))
S = dl.decision_function(Xc)
check("decision list: the predicted mode has a positive score, the others a negative one",
      np.all(S[np.arange(len(yc)), yc] > 0) and np.all(np.sort(S, axis=1)[:, -2] < 0))
pw = PairwiseLogisticRegression().fit(Xc, yc)
check("one-versus-one: training accuracy 1 on a T-junction", pw.score(Xc, yc) == 1.0)

import pytest  # noqa: E402


@pytest.mark.parametrize("name", list(RESULTS))
def test_check(name):
    cond, detail = RESULTS[name]
    assert cond, detail
