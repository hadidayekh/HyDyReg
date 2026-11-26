"""The optimization-based method (thesis 4.2) and the clusterwise-regression option. Part of HybridRegressor (see hydyreg.model)."""
import casadi
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from ..regression import ridge_refit
from .. import _logging as _log



class _OptiMixin:
    def fit_opti(self, X, X_dot, C=100, deltas=0.01, norm='L2', segmentation='strict', segments=[], print_on=False, apply_LR=False, init_seed=0, init='random', ipopt_options=None, refit='ls'):
        '''
        Fit the Hybrid Regressor model using the *optimization based* approach.

        Attributes
        ----------
        X : ndarray (N x D)
            A matrix of training data points [x1, x2, ..., xN].
        
        X_dot : ndarray (N x D)
            A matrix of corresponding estimated derivatives of training data points [x1dot, x2dot, ..., xNdot]
        
        C : float, default=100
            The hyperparameter C controlling the strength of regularization in the optimization objective function.

        detlas : list of `n` floats or float, default=0.01
            A list of absolute insensitivity errors for each mode.
            When one value is given, the same value applies to all modes.

        norm : 'L2', default='L2'
            The norm used for regularization (the 'L1' variant was removed; see the tag thesis-reproducible).

        segmentation : 'strict', default='strict'
            Xi variables per mode per segment, as in the thesis (the variants None and 'squared_diff' were
            removed; see the tag thesis-reproducible).

        segments : ndarray of N ints, default=[]
            The segment MAPPING of each data point to its segment.
            segments[i] indicates the segment identifier of point X[i].
            This is DIFFERENT from `segments` in `fit_first_incremental` which is a LIST OF CHANGE POINT INDICES.

        print_on : Bool, default=False
            The verbosity parameter of the Casadi optimizer.

        apply_LR : Bool, default=False
            Indicates whether to apply linear/polynomial regression on data after finding their modes.
            This replaces the continuous dynamics found by the optimizer.

        init_seed : int, default=0
            Seed of the random starting point of the optimizer. The outcome depends strongly on the starting
            point; `fit_opti_multistart` runs several seeds.

        init : 'random', 'segments' or a list of arrays, default='random'
            Starting point of the optimizer. 'random' (thesis): all variables uniform in [0, 1).
            A list of n arrays of shape (Q+1, D) (coefficients then intercept, one array per mode,
            e.g. `warm_start_coefs()` of a fitted model): warm start from these dynamics, with the
            smallest feasible slacks. Used to follow the solution from one value of C to the next.
            'segments': each mode starts from the
            least-squares fit on one segment (n distinct segments drawn with probability proportional to
            their number of points), and each slack starts at the smallest value that satisfies the
            constraints for these coefficients, so the starting point is feasible.

        refit : 'ls' or 'ridge', default='ls'
            With apply_LR: how the dynamics of each mode are refitted on the points assigned to it.
            'ls': ordinary least squares. 'ridge': ridge regression with the strength selected on held-out
            segments (`ridge_refit`); the selected strengths are stored in `ridge_alphas_`.

        ipopt_options : dict, default=None
            Further IPOPT options. `mumps_pivot_order=0` (AMD ordering in the linear solver) is set unless
            given here: with the default METIS ordering, the IPOPT bundled with casadi 3.7.2 crashed with a
            segmentation fault inside METIS on some of these problems (seen on arm64 macOS).

        The slacks are kept nonnegative exactly (IPOPT `bound_relax_factor` = 0). With IPOPT's default
        relaxation (1e-8, thesis), bounds such as xi >= 0 may be violated slightly, and with three or more
        modes the product term can then be made negative (one slack slightly below 0 times large slacks).

        After fitting, `objective_` is the value of the objective at the solution and `solver_status_`
        IPOPT's return status.
        '''
        rand = np.random.RandomState(init_seed)
        warm_coefs = None
        if not isinstance(init, str):
            warm_coefs, init = [np.asarray(c, dtype=float) for c in init], 'segments'
        if init not in ('random', 'segments'):
            raise ValueError("init must be 'random', 'segments' or a list of coefficient arrays")


        if self.n is None:
            raise ValueError("To use the optimization based method, the number of modes n must be provided.")
        if norm != 'L2':
            raise ValueError("Only norm='L2' is supported (the 'L1' variant was removed; see the tag thesis-reproducible).")
        if segmentation != 'strict':
            raise ValueError("Only segmentation='strict' is supported (the other variants were removed; see the tag thesis-reproducible).")
        if len(segments) == 0:
            raise ValueError("The parameter segments must be provided or non-empty when a segmentation integration method is selected.")

        n = self.n
        deg = self.deg

        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_trans = poly.fit_transform(X)
        self.poly = poly

        # deltas could be a scalar or defined per mode
        if (np.isscalar(deltas)):
            temp = np.ones(n)
            deltas = deltas*temp
        else:
            assert(len(deltas) == n)

        N = X_dot.shape[0]  # nb of points
        D = X_dot.shape[1]  # dimension of each vector
        Q = X_trans.shape[1]
        S = max(segments) + 1   # Number of segments
        opti = casadi.Opti()

        # variable declaration arrays
        Omegas = np.ndarray((n, D), dtype=object)
        betas = np.ndarray((n, D), dtype=object)
        Xis = np.ndarray((n,), dtype=object)

        if init == 'segments':
            seg_arr = np.asarray(segments)
            A_aff = np.c_[X_trans, np.ones(N)]
            if warm_coefs is not None:
                init_coefs = warm_coefs
            else:
                seg_ids, seg_sizes = np.unique(seg_arr, return_counts=True)
                k = min(n, len(seg_ids))
                seeds = list(rand.choice(seg_ids, size=k, replace=False, p=seg_sizes / seg_sizes.sum()))
                seeds += list(rand.choice(seg_ids, size=n - k))  # fewer segments than modes: reuse segments
                init_coefs = [np.linalg.lstsq(A_aff[seg_arr == sd], X_dot[seg_arr == sd], rcond=None)[0] for sd in seeds]
            # smallest feasible slack of mode j on segment r: max over its points and dimensions of |residual| - delta
            init_xis = np.zeros((n, S))
            for j in range(n):
                res = np.abs(X_dot - A_aff @ init_coefs[j]).max(axis=1)
                np.maximum.at(init_xis[j], seg_arr, res)
                init_xis[j] = np.maximum(init_xis[j] - deltas[j], 0.0)

        for j in range(n):
            xi_size_per_mode = S

            Xis[j] = opti.variable(xi_size_per_mode)
            opti.set_initial(Xis[j], init_xis[j] if init == 'segments' else rand.rand(xi_size_per_mode))

            for d in range(D):
                Omegas[j][d] = opti.variable(Q)
                opti.set_initial(Omegas[j][d], init_coefs[j][:-1, d] if init == 'segments' else rand.rand(Q))

                betas[j][d] = opti.variable()
                opti.set_initial(betas[j][d], init_coefs[j][-1, d] if init == 'segments' else rand.rand())

        C_param = opti.parameter()
        opti.set_value(C_param, C)

        # First part of the optimization objective function:
        # the sum of products of Xis
        # This forces one Xi per data point to be ~0, which dictates the mode

        sum_products_xis = 0
        for i in range(xi_size_per_mode):
            temp = 1
            for j in range(n):
                temp *= Xis[j][i]
            sum_products_xis += temp


        # Second part of the optimization objective function:
        # The sum of squared norms of the Omegas (L2)
        sum_norms = 0
        for j in range(n):
            for d in range(D):
                sum_norms += (Omegas[j][d].T @ Omegas[j][d])
        to_minimize = C_param*sum_norms + sum_products_xis

        # Setting the objective function
        opti.minimize(to_minimize)

        # CONSTRAINTS

        # all xis must be positive
        for j in range(n):
            opti.subject_to(Xis[j] >= 0)
        
        x_dotT = X_dot.T    # d by N
        for d in range(D):
            for j in range(n):
                # the slack of point i is the slack of its segment
                for i in range(N):
                    seg_index = segments[i]
                    opti.subject_to(x_dotT[d][i] - casadi.dot(X_trans[i].T, Omegas[j][d]) - betas[j][d] <=  deltas[j] + Xis[j][seg_index])
                    opti.subject_to(x_dotT[d][i] - casadi.dot(X_trans[i].T, Omegas[j][d]) - betas[j][d] >= -deltas[j] - Xis[j][seg_index])

        ipopt_opts = dict(max_iter = 3000, print_level = 0, bound_relax_factor = 0.0)
        ipopt_opts['mumps_pivot_order'] = 0  # AMD ordering: METIS segfaulted in casadi 3.7.2's IPOPT (see docstring)
        ipopt_opts.update(ipopt_options or {})
        opti.solver('ipopt', dict(print_time = 0), ipopt_opts)
        # Initiate the optimization
        sol = opti.solve()
        self.objective_ = float(sol.value(to_minimize))
        self.solver_status_ = opti.stats()['return_status']

        # Printing the results if print is on
        if print_on:
            for j in range(n):
                _log.debug("Xi_" + str(j) + ": " + str(sol.value(Xis[j])))
            for j in range(n):
                for d in range(D):
                    _log.debug("Omega_" + str(j) + "_" + str(d) + ": " + str(sol.value(Omegas[j][d])))
                    _log.debug( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

        # Saving the results in proper arrays
        Omegas_sol = np.ndarray((n, D, Q))
        betas_sol  = np.ndarray((n, D))
        Xis_sol   = np.ndarray((n, xi_size_per_mode))

        for j in range(n):
                Xis_sol[j] = sol.value(Xis[j])
        for j in range(n):
            for d in range(D):
                Omegas_sol[j][d] = sol.value(Omegas[j][d])
                betas_sol[j][d]  = sol.value(betas[j][d])
        
        self.Xis_segments_ = Xis_sol
        final_Xis = np.ndarray((n, N))
        for j in range(n):
            for i in range(N):
                seg_index = segments[i]
                final_Xis[j][i] = Xis_sol[j][seg_index]

        self.Xis_ = final_Xis

        if apply_LR:
            modes = self.get_modes()
            LR = LinearRegression()
            for j in range(n):
                indices_j = np.where(modes == j)
                X_j = X_trans[indices_j]
                X_dot_j = X_dot[indices_j]
                if X_j.shape[0] == 0:  # no segment assigned to this mode: keep the optimizer's coefficients
                    continue
                if refit == 'ridge':
                    Omegas_sol[j], betas_sol[j], alpha_j, _ = ridge_refit(X_j, X_dot_j, np.asarray(segments)[indices_j])
                    self.ridge_alphas_ = {**getattr(self, 'ridge_alphas_', {}), j: alpha_j}
                    continue
                LR.fit(X_j, X_dot_j)
                Omegas_sol[j] = LR.coef_
                betas_sol[j] = LR.intercept_


        self.Omegas_ = Omegas_sol
        self.betas_ = betas_sol
        self.Xfit_ = X
        self.Xfit_trans_ = X_trans
        self.X_dot_fit_ = X_dot

        self.modesfit_ = self.get_modes() # here we store the modes of each data point used for fitting

    def warm_start_coefs(self):
        """Dynamics of the fitted model in the form accepted by `fit_opti(init=...)`."""
        return [np.vstack([self.Omegas_[j].T, self.betas_[j][None, :]]) for j in range(self.n)]

    def fit_opti_multistart(self, X, X_dot, n_starts=10, seed=0, stop_tol=None, **kwargs):
        '''
        Run `fit_opti` (thesis Section 4.2) from several starting points and keep the solution with the
        lowest objective value, i.e. the best solution of the same optimization problem.

        The problem is nonconvex (product of slacks), so IPOPT returns a local optimum that depends on the
        starting point. At a wrong local optimum some segment has a positive slack for every mode; at the
        correct one every segment has a slack close to 0 for its mode, so the objective identifies it.

        Attributes
        ----------
        n_starts : int, default=10
            Maximum number of starting points.

        seed : int, default=0
            Start k uses `init_seed = seed + k`, so every call with the same seed (e.g. for every C of a
            hyperparameter search) uses the same starting points.

        stop_tol : float, default=None
            Stop early once a solution fits every segment within the insensitivity error, i.e. once
            max over segments of (min over modes of the slack) <= stop_tol. Such a solution has a product
            term of (almost) 0. None runs all starts.

        kwargs : passed to `fit_opti` (C, deltas, segments, init, apply_LR, ...).

        After fitting, `start_objectives_` holds the objective of every start (nan for a solver failure),
        `n_starts_run_` the number of starts run, and `max_min_slack_` the largest, over segments, of the
        smallest slack of the kept solution: 0 means every segment is fitted within the insensitivity
        error by some mode; a positive value means no mode fits some segment within it (e.g. when the
        derivative estimates are less accurate than `deltas`).
        '''
        if kwargs.get('segmentation', 'strict') != 'strict' and stop_tol is not None:
            raise ValueError("stop_tol needs segmentation='strict'")
        segments = np.asarray(kwargs.get('segments', []))
        present = np.unique(segments) if len(segments) else None
        best, objectives, errors = None, [], []
        from . import HybridRegressor  # (the class is assembled in hydyreg.model; a module-level import would be circular)
        for k in range(n_starts):
            m = HybridRegressor(n=self.n, deg=self.deg, classifier_poly_deg=self.classifier_poly_deg)
            try:
                m.fit_opti(X, X_dot, init_seed=seed + k, **kwargs)
            except (RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:  # IPOPT (RuntimeError) or the refit failed from this start
                objectives.append(np.nan)
                errors.append(f"{type(e).__name__}: {str(e)[:100]}")
                continue
            objectives.append(m.objective_)
            if best is None or m.objective_ < best.objective_:
                best = m
            if stop_tol is not None and present is not None:
                if np.max(np.min(m.Xis_segments_[:, present], axis=0)) <= stop_tol:
                    break
        if best is None:
            raise RuntimeError(f"fit_opti failed from all {n_starts} starting points: " + "; ".join(errors[:3]))
        for key, v in vars(best).items():
            if key.endswith('_') or key == 'poly':
                setattr(self, key, v)
        self.start_objectives_ = np.array(objectives)
        self.n_starts_run_ = len(objectives)
        if present is not None and self.Xis_.shape[1] == len(segments):
            self.max_min_slack_ = float(np.max(np.min(self.Xis_segments_[:, present], axis=0)))
        return self

    def fit_clusterwise(self, X, X_dot, segments, n_restarts=200, seed=0, max_iter=100):
        '''
        Fit the continuous dynamics and the mode of each segment by clusterwise regression:

            minimize over (mode of each segment, f_1..f_n)   sum_s  sum_{i in s} || x_dot_i - f_mode(s)(x_i) ||^2

        Like `fit_opti`, it needs the number of modes `n` and gives one mode per segment, but the
        problem is solved by alternating minimization instead of a nonlinear program: (1) fit each
        mode by least squares on the points of its segments, (2) assign each segment to the mode
        with the smallest squared error on it; repeat until the assignment is stable. Each step
        decreases the objective. The result depends on the starting assignment, so `n_restarts`
        starts are run and the one with the smallest objective is kept. Half of the starts fit each
        mode on one randomly drawn segment (longer segments are more likely), the others assign the
        segments to modes at random.

        There is no hyperparameter to tune (no C, no insensitivity error) and the starting points
        come from a private generator seeded with `seed`, so the result is reproducible and does not
        depend on the global random state or on the number of BLAS threads.

        Attributes
        ----------
        X, X_dot : ndarray (N x D)
            Training points and their estimated derivatives.

        segments : ndarray of N ints
            Segment identifier of each point (same convention as `fit_opti`).

        n_restarts : int, default=200
            Number of starting assignments.

        seed : int, default=0
            Seed of the starting assignments.

        max_iter : int, default=100
            Maximum number of alternations per start.

        After fitting, `objective_` is the best objective, `restart_objectives_` the objective of every
        start, and `Xis_segments_[j, s]` the root-mean-square error of mode j on segment s (its arg min
        over j is the mode of s, as with `fit_opti`).
        '''
        if self.n is None:
            raise ValueError("The number of modes n must be provided.")
        from threadpoolctl import threadpool_limits
        n = self.n
        segments = np.asarray(segments)
        poly = PolynomialFeatures(degree=self.deg, include_bias=False)
        X_trans = poly.fit_transform(X)
        self.poly = poly
        N, D = X_dot.shape
        A = np.c_[X_trans, np.ones(N)]
        seg_ids = np.unique(segments)
        idx = [np.nonzero(segments == s)[0] for s in seg_ids]
        S = len(seg_ids)
        lengths = np.array([len(i) for i in idx], dtype=float)
        rng = np.random.default_rng(seed)

        def fit(points):
            return np.linalg.lstsq(A[points], X_dot[points], rcond=None)[0]

        def errors(W):  # squared error of every mode on every segment, (S, n)
            return np.array([[np.sum((X_dot[i] - A[i] @ W[j]) ** 2) for j in range(n)] for i in idx])

        best = None
        objectives = []
        with threadpool_limits(limits=1, user_api='blas'):
            for r in range(n_restarts):
                if r % 2 == 0 and S >= n:
                    seeds = rng.choice(S, size=n, replace=False, p=lengths / lengths.sum())
                    lam = np.argmin(errors([fit(idx[k]) for k in seeds]), axis=1)
                else:
                    lam = rng.integers(n, size=S)
                for _ in range(max_iter):
                    W = [fit(np.concatenate([idx[k] for k in range(S) if lam[k] == j])) if np.any(lam == j)
                         else fit(idx[rng.integers(S)])  # a mode without segment restarts from a random segment
                         for j in range(n)]
                    E = errors(W)
                    new = np.argmin(E, axis=1)
                    if np.array_equal(new, lam):
                        break
                    lam = new
                obj = E[np.arange(S), lam].sum()
                objectives.append(obj)
                if best is None or obj < best[0]:
                    best = (obj, lam.copy(), W)
        obj, lam, W = best

        Omegas = np.zeros((n, D, X_trans.shape[1]))
        betas = np.zeros((n, D))
        modes = np.empty(N, dtype=int)
        for k in range(S):
            modes[idx[k]] = lam[k]
        for j in range(n):
            if np.any(modes == j):
                LR = LinearRegression().fit(X_trans[modes == j], X_dot[modes == j])
                Omegas[j], betas[j] = LR.coef_, LR.intercept_
            else:
                Omegas[j], betas[j] = W[j][:-1].T, W[j][-1]
        Xis_seg = np.zeros((n, int(segments.max()) + 1))
        for k in range(S):
            for j in range(n):
                Xis_seg[j, seg_ids[k]] = np.sqrt(np.mean((X_dot[idx[k]] - X_trans[idx[k]] @ Omegas[j].T - betas[j]) ** 2))
            Xis_seg[lam[k], seg_ids[k]] = min(Xis_seg[lam[k], seg_ids[k]], Xis_seg[:, seg_ids[k]].min())  # arg min = assigned mode
        self.Xis_segments_ = Xis_seg
        self.Xis_ = Xis_seg[:, segments]
        self.Omegas_, self.betas_ = Omegas, betas
        self.Xfit_, self.Xfit_trans_, self.X_dot_fit_ = X, X_trans, X_dot
        self.modesfit_ = modes
        self.objective_ = obj
        self.restart_objectives_ = np.array(objectives)
