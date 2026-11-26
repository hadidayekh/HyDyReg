"""The incremental method (thesis Ch. 5, Algorithm 4) with the split step, the gate on new modes, pending segments and merging. Part of HybridRegressor (see hydyreg.model)."""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.base import clone

from ..metrics import relative_diff
from .. import _logging as _log



class _IncrementalMixin:
    def fit_first_incremental(self, X, X_dot, segments, good_RD=None, regressor=None):
        '''
        Fit the Hybrid Regressor model using the *incremental* approach.

        Attributes
        ----------
        X : ndarray (N x D)
            A matrix of training data points [x1, x2, ..., xN].
        
        X_dot : ndarray (N x D)
            A matrix of corresponding estimated derivatives of training data points [x1dot, x2dot, ..., xNdot]

        segments : list of ints
            A list of INDICES of the beginning of each segment in `X`.
            This means that `X[segments[i]:segments[i+1]]` are points that belong to the same segment.
            This is DIFFERENT from `segments` in `fit_opti` which is a MAPPING.

        good_RD : float, default=None (the model's `rd_tol`, LearnConfig.rd_tol)
            This tol_RD in the theoretical documentation.
            Used as a tolerance for the average relative difference to indicate whether a mode captures the dynamics of a sequence of data points.

        regressor : ?, default=None (a new LinearRegression())
            The regressor model to be used. Default is linear regression (no regularization).
            Regularization is recommended for passive learning. This can be achieved by passing RidgeCV() for example.

        '''
        good_RD = self.rd_tol if good_RD is None else good_RD
        regressor = LinearRegression() if regressor is None else regressor

        # Start from an empty hypothesis and treat every segment with `update_continuous`
        # (as in Algorithm 3). Unlike the thesis (Algorithm 4), the first segment does not found
        # the first mode unconditionally: see `update_continuous`.
        pairs = list(zip(segments, list(segments[1:]) + [X.shape[0]]))
        D = X.shape[1]
        self.poly = PolynomialFeatures(degree=self.deg, include_bias=False)
        Q = self.poly.fit_transform(X[:1]).shape[1]
        self.n = 0
        self.Omegas_, self.betas_, self.regressors_ = np.zeros((0, D, Q)), np.zeros((0, D)), []
        self.Xfit_, self.Xfit_trans_, self.X_dot_fit_ = np.zeros((0, D)), np.zeros((0, Q)), np.zeros((0, D))
        self.modesfit_ = np.zeros(0, dtype=int)
        for i, (a, b) in enumerate(pairs):
            _log.debug("seg", i+1)
            _log.debug(b - a, "points")
            self.update_continuous(X[a:b], X_dot[a:b], good_RD=good_RD, regressor=regressor)
        self.retry_pending(good_RD=good_RD, regressor=regressor)
        if self.n == 0:
            raise ValueError("No mode could be learned: no segment (or group of segments) has at least "
                             f"{2*(Q+1)} points that one polynomial model of degree {self.deg} captures "
                             f"with RD < {good_RD}. Check the segmentation tolerance, the degree and good_RD.")

    def _rd_tol(self, X_dot, good_RD):
        """Relative-difference tolerance for a set of points with estimated derivatives X_dot: `good_RD`,
        but at least twice the relative difference that derivative noise alone produces.

        With noise of standard deviation `deriv_noise_` per component of the estimated derivatives (set by
        the pipeline from the estimated measurement noise; 0 or absent = no adaptation), the error of a
        derivative has norm about s = deriv_noise_ * sqrt(D), and its relative difference to the true
        derivative is about s / (2 ||x_dot_i|| + s) at point i: small where the state moves fast, close
        to 1 where it is almost at rest. The floor is the mean of this over the points (not the ratio of
        the means: slow points dominate the mean relative difference)."""
        nz = getattr(self, 'deriv_noise_', 0.0)
        if not nz or len(X_dot) == 0:
            return good_RD
        s = nz * np.sqrt(X_dot.shape[1])
        floor = np.mean(s / (2 * np.linalg.norm(X_dot, axis=1) + s))
        return max(good_RD, 2 * floor)

    def compare_with_dynamics(self, X, Xdot, tol=0.01):
        RDs = np.ndarray((self.n,))
        for j in range(self.n):
            Xdot_pred = self.predict(X, lambda x: j)
            RDs[j] = np.mean(relative_diff(Xdot, Xdot_pred))
        min_RD = np.min(RDs)
        if min_RD < tol:
            _log.debug("No need to improve, mode", np.argmin(RDs), f"matches. RD = {min_RD:.3e}")
        else:
            _log.debug(f"Need to improve, min(RDs) = {min_RD:.3e}")
        return min_RD < tol, np.argmin(RDs)

    def update_continuous(self, X, X_dot, good_RD=None, regressor=None, _depth=0):
        '''
        Update the learned continuous dynamics and mode information with NEW data of ONE SEGMENT.
        This function is called repeatedly on data of different segments in the incremental learning approach.
        Theoretically can be used to update the dynamics found using the optimization based appraoch, but this is not tested.

        Attributes
        ----------
        X : ndarray (N x D)
            A matrix of NEW training data points [x1, x2, ..., xN] belonging in ONE SEGMENT.
        
        X_dot : ndarray (N x D)
            A matrix of corresponding estimated derivatives of training data points [x1dot, x2dot, ..., xNdot]

        good_RD : float, default=None (the model's `rd_tol`, LearnConfig.rd_tol)
            This tol_RD in the theoretical documentation.
            Used as a tolerance for the average relative difference to indicate whether a mode captures the dynamics of a sequence of data points.

        regressor : ?, default=None (a new LinearRegression())
            The regressor model to be used. Default is linear regression (no regularization).
            Regularization is recommended for passive learning. This can be achieved by passing RidgeCV() for example.


        A segment that fails both checks is not turned into a new mode right away (as in the thesis,
        lines 24-27 of Algorithm 4). Instead:

        1. It is explained, as far as possible, as a sequence of runs of already learned modes
           (`_explain_by_modes`). This recovers switches that segmentation missed. Each run that a
           mode captures (average RD below `good_RD`) is assigned to it; the points next to the
           recovered change points are dropped, as in segmentation; the remaining pieces are treated
           as new segments (recursively).
        2. A piece that no mode explains founds a new mode only if it has at least 2(Q+1) points
           (Q = number of polynomial features) and its own polynomial fit captures it (average RD
           below `good_RD`). A segment that contains a switch fails this test.
        3. Otherwise it is set aside in `pending_` and retried by `retry_pending` once the other
           segments have been treated. Segments that are set aside are not used for classification.
        '''
        good_RD = self.rd_tol if good_RD is None else good_RD
        regressor = LinearRegression() if regressor is None else regressor


        # X and X dot belong to one segment
        N_new = X.shape[0]
        good_RD = self._rd_tol(X_dot, good_RD)
        if self.deg > 1:
            X_trans = self.poly.fit_transform(X)
        else:
            X_trans = X
        
        if self.n == 0:
            # empty hypothesis: the segment founds the first mode, or is set aside
            if self._split_or_set_aside(X, X_trans, X_dot, good_RD, regressor, _depth):
                return
            is_already_captured = False
        else:
            is_already_captured, j_star = self.compare_with_dynamics(X, X_dot, tol=good_RD)
        if is_already_captured:
            mode = j_star
        else:
            LRs = []
            RDs = np.ndarray((self.n,))
            for j in range(self.n):
                LR = clone(regressor)
                Xfit_trans_j, X_dot_j = self.get_points_of_mode(j)
                aug_X = np.r_[Xfit_trans_j, X_trans]
                aug_X_dot = np.r_[X_dot_j, X_dot]

                LR.fit(aug_X, aug_X_dot)
                
                # Calculating RD for segment
                X_dot_pred = LR.predict(X_trans)
                RD_seg = np.mean(relative_diff(X_dot, X_dot_pred))
                # Calculating RD for rest of the data
                X_dot_pred_j = LR.predict(Xfit_trans_j)
                RD_j = np.mean(relative_diff(X_dot_j, X_dot_pred_j))

                LRs.append(LR)
                RDs[j] = np.max([RD_seg, RD_j])

            RD = np.min(RDs) if self.n > 0 else np.inf
            if RD < good_RD:
                mode = np.argmin(RDs)
                _log.debug("Merging with mode", mode, "RD is", RD)
                LR = LRs[mode]
                self.Omegas_[mode] = LR.coef_
                self.betas_[mode] = LR.intercept_
                self.regressors_[mode] = LR

            elif self.n > 0 and self._split_or_set_aside(X, X_trans, X_dot, good_RD, regressor, _depth):
                # the segment was split along known modes, or set aside: nothing is stored for it as a whole
                return
            elif getattr(self, 'max_modes', None) is not None and self.n >= self.max_modes:
                # the number of modes is imposed (optimization-based method): no new mode
                _log.debug(f"Segment set aside ({N_new} points): it matches none of the {self.n} modes, and their number is imposed.")
                self.pending_.append((X, X_dot))
                return
            else:
                mode = self.n
                _log.debug(f"Creating new mode {mode}; RD was not good: {RD:.3f}")
                _log.debug(RDs)
                self.n = self.n + 1
                LR = clone(regressor)
                LR.fit(X_trans, X_dot)
                self.Omegas_ = np.r_[self.Omegas_, LR.coef_.reshape((1, LR.coef_.shape[0], LR.coef_.shape[1]))]
                self.betas_ = np.r_[self.betas_, LR.intercept_.reshape((1, LR.intercept_.shape[0]))]
                self.regressors_.append(LR)
        
        self.Xfit_ = np.r_[self.Xfit_, X]
        self.Xfit_trans_ = np.r_[self.Xfit_trans_, X_trans]
        self.X_dot_fit_ = np.r_[self.X_dot_fit_, X_dot]

        modes_new = np.array([mode] * N_new)
        self.modesfit_ = np.r_[self.modesfit_, modes_new]
    def _pointwise_RD(self, X_trans, X_dot):
        """Relative difference between `X_dot` and the prediction of each mode, per point: array (n, N)."""
        R = np.empty((self.n, X_dot.shape[0]))
        for j in range(self.n):
            pred = X_trans @ self.Omegas_[j].T + self.betas_[j]
            with np.errstate(invalid='ignore', divide='ignore'):
                R[j] = relative_diff(X_dot, pred)
        return np.nan_to_num(R, nan=0.0)

    def _can_found_mode(self, X_trans, X_dot, good_RD, regressor):
        """A segment may found a new mode if it has enough points for its polynomial fit to be
        determined (N >= 2(Q+1)) and this fit captures it (average RD < good_RD).
        Returns (ok, reason)."""
        N = X_dot.shape[0]
        N_min = 2 * (X_trans.shape[1] + 1)
        good_RD = self._rd_tol(X_dot, good_RD)
        if N < N_min:
            return False, f"only {N} points (< {N_min})"
        LR = clone(regressor).fit(X_trans, X_dot)
        with np.errstate(invalid='ignore', divide='ignore'):
            RD_self = np.nanmean(relative_diff(X_dot, LR.predict(X_trans)))
        if not RD_self < good_RD:
            return False, f"its own fit has RD {RD_self:.2e} (>= {good_RD})"
        return True, ""

    @staticmethod
    def _explain_by_modes(R, unknown_cost, switch_cost):
        """Labels the points of a segment with a learned mode (0..n-1) or with n ("explained by no
        mode"), minimizing  sum_i cost(label_i, i) + switch_cost * (number of label changes),
        where cost(j, i) = R[j, i] for a mode and `unknown_cost` for "no mode" (dynamic programming)."""
        n, N = R.shape
        C = np.vstack([R, np.full((1, N), unknown_cost)])
        states = np.arange(n + 1)
        back = np.zeros((N, n + 1), dtype=int)
        cost = C[:, 0].copy()
        for i in range(1, N):
            j_best = int(np.argmin(cost))
            switch = cost[j_best] + switch_cost
            stay = cost <= switch
            back[i] = np.where(stay, states, j_best)
            cost = np.where(stay, cost, switch) + C[:, i]
        labels = np.empty(N, dtype=int)
        labels[-1] = int(np.argmin(cost))
        for i in range(N - 1, 0, -1):
            labels[i - 1] = back[i, labels[i]]
        return labels

    def _split_or_set_aside(self, X, X_trans, X_dot, good_RD, regressor, depth, trim=2, max_trim=6, trim_unexplained=5):
        """Called for a segment that failed both checks. Returns True if the segment was handled here
        (split along known modes, or set aside), False if it should found a new mode.

        Points next to a recovered change point are dropped, because the five-point derivative of the
        two points on each side of a switch mixes both modes: at least `trim` points of a run that a
        mode explains, then up to `max_trim` while the mode does not capture the point; and
        `trim_unexplained` points of an unexplained run (there is no model to test its points with,
        and the change point is only located to within the four affected points)."""
        N = X.shape[0]
        R = self._pointwise_RD(X_trans, X_dot)
        labels = self._explain_by_modes(R, unknown_cost=good_RD, switch_cost=5 * good_RD)
        cuts = np.r_[0, np.nonzero(np.diff(labels))[0] + 1, N]
        runs = []  # [start, end, label] with the points next to change points removed
        for a, b in zip(cuts[:-1], cuts[1:]):
            j = labels[a]
            lo, hi = a, b
            if a > 0:
                lo += trim if j < self.n else trim_unexplained
                while j < self.n and lo < hi and lo - a < max_trim and R[j, lo] > good_RD:
                    lo += 1
            if b < N:
                hi -= trim if j < self.n else trim_unexplained
                while j < self.n and lo < hi and b - hi < max_trim and R[j, hi - 1] > good_RD:
                    hi -= 1
            if hi <= lo:
                continue
            if j < self.n and not np.mean(R[j, lo:hi]) < good_RD:
                j = self.n  # not captured after all: treat as unexplained
            runs.append([lo, hi, j])

        if not any(j < self.n for _, _, j in runs):
            ok, reason = self._can_found_mode(X_trans, X_dot, good_RD, regressor)
            if ok:
                return False
            _log.debug(f"Segment set aside ({N} points): no mode explains any part of it and {reason}.")
            self.pending_.append((X, X_dot))
            return True

        _log.debug(f"Splitting the segment ({N} points) along learned modes:",
              ", ".join(f"[{lo}:{hi}] -> " + (f"mode {j}" if j < self.n else "unexplained") for lo, hi, j in runs))
        # states on both sides of each recovered change point (used to generate fine points there)
        for (lo1, hi1, j1), (lo2, hi2, j2) in zip(runs[:-1], runs[1:]):
            self.split_gaps_.append(dict(x_A=X[hi1 - 1].copy(), x_B=X[lo2].copy(), n_steps=lo2 - (hi1 - 1),
                                         step=getattr(self, 'split_step_', None),
                                         a=j1 if j1 < self.n else None, b=j2 if j2 < self.n else None,
                                         inside=X[hi1:lo2].copy()))
        unexplained = []
        for lo, hi, j in runs:
            if j < self.n:
                self.Xfit_ = np.r_[self.Xfit_, X[lo:hi]]
                self.Xfit_trans_ = np.r_[self.Xfit_trans_, X_trans[lo:hi]]
                self.X_dot_fit_ = np.r_[self.X_dot_fit_, X_dot[lo:hi]]
                self.modesfit_ = np.r_[self.modesfit_, np.array([j] * (hi - lo))]
            else:
                unexplained.append((X[lo:hi], X_dot[lo:hi]))
        for X_u, X_dot_u in unexplained:
            _log.debug("Unexplained part with", X_u.shape[0], "points:")
            self.update_continuous(X_u, X_dot_u, good_RD=good_RD, regressor=regressor, _depth=depth + 1)
        return True

    def split_mixed_segments(self, segments, good_RD=None, max_passes=3):
        """
        For a model fitted with one mode per segment (`fit_opti`, `fit_clusterwise`): finds the training
        segments that contain an undetected switch and splits them along the learned modes.

        Such a segment has been assigned to one mode as a whole and has distorted that mode, so the modes
        cannot be trusted to detect it. The steps are therefore:
        1. A segment is suspicious if its own polynomial fit does not capture it (average RD >= good_RD;
           only tested when it has at least 2(Q+1) points). This does not involve the modes.
        2. The modes are refitted by least squares without the suspicious segments.
        3. Every segment that its mode does not capture (average RD >= good_RD) is explained as a sequence
           of runs of the modes (`_explain_by_modes`); the runs that a mode captures are assigned to it,
           the points next to the recovered change points and the unexplained points are dropped.
        4. The modes are refitted on the result. Steps 3-4 are repeated (at most `max_passes` times) while
           the assignment changes.

        `segments` is the segment identifier of each training point. Returns the number of segments that
        were split or reassigned. Recovered change points are appended to `split_gaps_`.
        """
        good_RD = self.rd_tol if good_RD is None else good_RD
        segments = np.asarray(segments)
        X, X_trans, X_dot = self.Xfit_, self.Xfit_trans_, self.X_dot_fit_
        N, Q = X_trans.shape
        ids = np.unique(segments)
        idx = {s_: np.nonzero(segments == s_)[0] for s_ in ids}
        modes = self.modesfit_.copy()
        keep = np.ones(N, dtype=bool)

        def refit(mask):
            for j in range(self.n):
                sel = mask & (modes == j)
                if np.sum(sel) > Q:
                    LR = LinearRegression().fit(X_trans[sel], X_dot[sel])
                    self.Omegas_[j], self.betas_[j] = LR.coef_, LR.intercept_

        suspicious = np.zeros(N, dtype=bool)
        for s_, i in idx.items():
            if len(i) >= 2 * (Q + 1) and not self._can_found_mode(X_trans[i], X_dot[i], good_RD, LinearRegression())[0]:
                suspicious[i] = True
        if suspicious.any():
            _log.debug(f"{len(np.unique(segments[suspicious]))} training segment(s) are not captured by their own polynomial fit.")
            refit(~suspicious)
        n_changed, gaps = 0, []
        for _ in range(max_passes):
            changed = False
            for s_, i in idx.items():
                if not keep[i].any():
                    continue
                i = i[keep[i]]
                R = self._pointwise_RD(X_trans[i], X_dot[i])
                tol_i = self._rd_tol(X_dot[i], good_RD)
                # points of one mode, captured by it: nothing to do
                if len(np.unique(modes[i])) == 1 and np.mean(R[modes[i][0]]) < tol_i:
                    continue
                if len(np.unique(modes[i])) > 1 and all(np.mean(R[j][modes[i] == j]) < tol_i for j in np.unique(modes[i])):
                    continue
                labels = self._explain_by_modes(R, unknown_cost=tol_i, switch_cost=5 * tol_i)
                cuts = np.r_[0, np.nonzero(np.diff(labels))[0] + 1, len(i)]
                new_modes = modes[i].copy()
                new_keep = np.zeros(len(i), dtype=bool)
                runs = []
                for a, b in zip(cuts[:-1], cuts[1:]):
                    j = labels[a]
                    lo, hi = a + (2 if a > 0 else 0), b - (2 if b < len(i) else 0)
                    if j < self.n and hi > lo and np.mean(R[j, lo:hi]) < tol_i:
                        new_modes[lo:hi], new_keep[lo:hi] = j, True
                        runs.append((lo, hi, j))
                if not np.array_equal(new_keep, np.ones(len(i), dtype=bool)) or not np.array_equal(new_modes, modes[i]):
                    changed = True
                    n_changed += 1
                    _log.debug(f"Training segment {s_} ({len(i)} points) split along the modes: " +
                          (", ".join(f"[{lo}:{hi}] -> mode {j}" for lo, hi, j in runs) or "no part explained, dropped"))
                    for (lo1, hi1, j1), (lo2, hi2, j2) in zip(runs[:-1], runs[1:]):
                        gaps.append(dict(x_A=X[i[hi1 - 1]].copy(), x_B=X[i[lo2]].copy(), n_steps=lo2 - (hi1 - 1),
                                         step=getattr(self, 'split_step_', None), a=j1, b=j2, inside=X[i[hi1:lo2]].copy()))
                    modes[i] = new_modes
                    keep[i] = new_keep
            if not changed:
                break
            refit(keep)
        if n_changed:
            self.split_gaps_.extend(gaps)
            self.Xfit_, self.Xfit_trans_, self.X_dot_fit_ = X[keep], X_trans[keep], X_dot[keep]
            self.modesfit_ = modes[keep]
            if hasattr(self, 'Xis_') and self.Xis_.shape[1] == N:
                self.Xis_ = self.Xis_[:, keep]
            _log.debug(f"{n_changed} training segment(s) split; {int(np.sum(~keep))} of {N} points dropped.")
        return n_changed

    def retry_pending(self, good_RD=None, regressor=None):
        """Treats again the segments that were set aside (see `update_continuous`), now that more modes
        are known. Then, as long as a group of them can found a mode jointly (`_found_mode_from_pending`),
        creates this mode and retries the others. Segments that still cannot be explained stay in
        `pending_`."""
        good_RD = self.rd_tol if good_RD is None else good_RD
        regressor = LinearRegression() if regressor is None else regressor
        while self.pending_:
            pending, self.pending_ = self.pending_, []
            for X, X_dot in pending:
                _log.debug("Retrying a segment that was set aside,", X.shape[0], "points:")
                self.update_continuous(X, X_dot, good_RD=good_RD, regressor=regressor)
            if not self._found_mode_from_pending(good_RD, regressor):
                break
        if self.pending_:
            _log.debug(len(self.pending_), "segment(s) remain set aside,", sum(len(p[0]) for p in self.pending_), "points.")

    def _found_mode_from_pending(self, good_RD, regressor):
        """Short segments cannot found a mode alone (their polynomial fit is not determined), but several
        of them can do so together: a group of segments that were set aside founds a new mode if it has
        at least 2(Q+1) points in total and one polynomial model captures each of its segments (average
        RD < good_RD). Groups are grown greedily from each segment, longest first; a segment is only
        added when the joint fit is overdetermined. Returns True if a mode was created."""
        segs = self.pending_
        if len(segs) < 2:
            return False
        trans = [self.poly.fit_transform(X) if self.deg > 1 else X for X, _ in segs]
        Q1 = self.Xfit_trans_.shape[1] + 1
        order = sorted(range(len(segs)), key=lambda i: -len(segs[i][0]))

        def joint_fit(group):
            LR = clone(regressor).fit(np.concatenate([trans[i] for i in group]), np.concatenate([segs[i][1] for i in group]))
            with np.errstate(invalid='ignore', divide='ignore'):
                RD = max(np.nanmean(relative_diff(segs[i][1], LR.predict(trans[i]))) for i in group)
            return LR, RD

        for seed in order:
            group, N_group, LR_group = [seed], len(segs[seed][0]), None
            for i in order:
                if i == seed or N_group + len(segs[i][0]) <= Q1:
                    continue
                LR, RD = joint_fit(group + [i])
                if RD < self._rd_tol(np.concatenate([segs[k][1] for k in group + [i]]), good_RD):
                    group, N_group, LR_group = group + [i], N_group + len(segs[i][0]), LR
            if len(group) < 2 or N_group < 2 * Q1:
                continue
            mode = self.n
            _log.debug(f"Creating new mode {mode} from {len(group)} segments that were set aside ({N_group} points).")
            self.n += 1
            self.Omegas_ = np.r_[self.Omegas_, LR_group.coef_.reshape((1,) + LR_group.coef_.shape)]
            self.betas_ = np.r_[self.betas_, LR_group.intercept_.reshape((1, -1))]
            self.regressors_.append(LR_group)
            for i in group:
                X, X_dot = segs[i]
                self.Xfit_ = np.r_[self.Xfit_, X]
                self.Xfit_trans_ = np.r_[self.Xfit_trans_, trans[i]]
                self.X_dot_fit_ = np.r_[self.X_dot_fit_, X_dot]
                self.modesfit_ = np.r_[self.modesfit_, np.array([mode] * len(X), dtype=int)]
            self.pending_ = [segs[i] for i in range(len(segs)) if i not in group]
            return True
        return False

    def merge_similar_modes(self, good_RD=None, regressor=None):
        """Refinement step of thesis Remark 17: merges two modes when one polynomial model captures the
        data of both (the second check of Algorithm 4, applied to a pair of modes). Returns the number
        of merges."""
        good_RD = self.rd_tol if good_RD is None else good_RD
        regressor = LinearRegression() if regressor is None else regressor
        merges = 0
        a = 0
        while a < self.n:
            b = a + 1
            while b < self.n:
                Xa, Xda = self.get_points_of_mode(a)
                Xb, Xdb = self.get_points_of_mode(b)
                if len(Xa) == 0 or len(Xb) == 0:
                    b += 1
                    continue
                LR = clone(regressor).fit(np.r_[Xa, Xb], np.r_[Xda, Xdb])
                RD = max(np.mean(relative_diff(Xda, LR.predict(Xa))), np.mean(relative_diff(Xdb, LR.predict(Xb))))
                if RD < self._rd_tol(np.r_[Xda, Xdb], good_RD):
                    _log.debug(f"Merging mode {b} into mode {a}: one model captures both (RD = {RD:.2e}).")
                    self.Omegas_[a], self.betas_[a], self.regressors_[a] = LR.coef_, LR.intercept_, LR
                    self.Omegas_ = np.delete(self.Omegas_, b, axis=0)
                    self.betas_ = np.delete(self.betas_, b, axis=0)
                    del self.regressors_[b]
                    for attr in ('modesfit_', 'modes_classify_'):
                        if hasattr(self, attr):
                            m = getattr(self, attr).copy()
                            m[m == b] = a
                            m[m > b] -= 1
                            setattr(self, attr, m)
                    # recovered change points name the modes on both sides: renumber them too
                    for g in self.split_gaps_:
                        for side in ('a', 'b'):
                            if g.get(side) is not None:
                                g[side] = a if g[side] == b else (g[side] - 1 if g[side] > b else g[side])
                    self.n -= 1
                    merges += 1
                else:
                    b += 1
            a += 1
        return merges
