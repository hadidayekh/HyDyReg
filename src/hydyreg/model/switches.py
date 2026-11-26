"""Switches located with the learned modes (boundary points without queries to the system). Part of HybridRegressor (see hydyreg.model)."""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from ..tolerances import POINT_RD, SWITCH_ACCEPT, SWITCH_RESOLVE



class _SwitchMixin:
    def _flow(self, j, x0, T):
        """State reached from x0 after time T (>= 0) with the learned dynamics of mode j."""
        if T <= 0:
            return np.asarray(x0, dtype=float)
        Om, be, deg = self.Omegas_[j], self.betas_[j], self.deg
        if deg > 1:
            powers = self.poly.powers_   # (Q, D): feature q is prod_d x_d ** powers[q, d]
            feat = lambda x: np.prod(x[None, :] ** powers, axis=1)
        else:
            feat = lambda x: x
        sol = solve_ivp(lambda t, x: Om @ feat(x) + be, (0.0, T), np.asarray(x0, dtype=float), rtol=1e-9, atol=1e-11)
        return sol.y[:, -1]

    def _miss_tol(self, accept, scale, D, T=None):
        """Largest accepted mismatch (root mean square over the observed states) between a reconstruction of
        a gap and the observations: `accept` times the length of the gap without noise. With measurement
        noise (`noise_std_`, `deriv_noise_`, set by the pipeline, `Learner.build_model`), two terms are added: the relative
        accuracy that derivative noise allows for the learned dynamics, s / (2 v + s) with
        s = deriv_noise_ * sqrt(D) and v the mean speed over the gap, times the length of the gap; and
        three standard deviations of the noise of an observed state."""
        sigma = getattr(self, 'noise_std_', 0.0)
        nz = getattr(self, 'deriv_noise_', 0.0)
        rel = 0.0
        if nz and T:
            s_ = nz * np.sqrt(D)
            rel = s_ / (2 * scale / T + s_)
        return max(accept * scale, rel * scale + 3 * sigma * np.sqrt(D))

    def mode_of_point(self, x, x_dot, tol=POINT_RD):
        """Mode whose dynamics give the derivative x_dot at x (relative difference below tol), else None."""
        x_trans = self.poly.transform(np.asarray(x)[None]) if self.deg > 1 else np.asarray(x)[None]
        R = self._pointwise_RD(x_trans, np.asarray(x_dot)[None])[:, 0]
        j = int(np.argmin(R))
        return j if R[j] < max(tol, 1.5 * self._rd_tol(np.asarray(x_dot)[None], 0.0)) else None

    def locate_switch(self, x_A, x_B, T, a, b, inside=None, inside_times=None, accept=SWITCH_ACCEPT, resolve=SWITCH_RESOLVE):
        """
        Locates the switch from mode a to mode b between two observed states x_A (in mode a) and x_B
        (in mode b, a time T later), using the learned dynamics only (no query to the system).

        The trajectory is reconstructed as: flow of mode a from x_A during tau, then flow of mode b
        during T - tau. tau is chosen so that the reconstruction ends on x_B. An error eps of the
        reconstruction moves tau by about eps / ||f_a - f_b||, so the switch is well located when the
        two vector fields differ at the switch, and not identifiable when they (almost) coincide.

        The switch is accepted if
        - the reconstruction misses x_B (and the observed states `inside` the gap, at `inside_times`
          after x_A, if given) by at most `accept` times ||x_B - x_A||, and
        - it is resolved: this miss is at most `resolve` times the miss of both extreme choices
          (tau = 0 and tau = T), i.e. the data do tell where in the gap the switch is.

        Returns (tau, x_switch) or None.
        """
        x_A, x_B = np.asarray(x_A, dtype=float), np.asarray(x_B, dtype=float)
        scale = np.linalg.norm(x_B - x_A)
        if a is None or b is None or a == b or T <= 0 or scale == 0 or max(a, b) >= self.n:
            return None

        def miss(tau):
            # root mean square mismatch over the observed states of the gap (those inside it and x_B)
            xs = self._flow(a, x_A, tau)
            sq = [np.sum((self._flow(b, xs, T - tau) - x_B) ** 2)]
            if inside is not None:
                for x_obs, t_obs in zip(inside, inside_times):
                    x_rec = self._flow(a, x_A, t_obs) if t_obs <= tau else self._flow(b, xs, t_obs - tau)
                    sq.append(np.sum((x_rec - x_obs) ** 2))
            return float(np.sqrt(np.mean(sq)))

        try:
            res = minimize_scalar(lambda tau: np.linalg.norm(self._flow(b, self._flow(a, x_A, tau), T - tau) - x_B),
                                  bounds=(0.0, T), method='bounded', options=dict(xatol=1e-8 * T))
            tau = float(res.x)
            m_best, m_0, m_T = miss(tau), miss(0.0), miss(T)
        except (ValueError, ArithmeticError, np.linalg.LinAlgError):  # the flow of a learned mode failed (e.g. diverged)
            return None
        if not (m_best <= self._miss_tol(accept, scale, len(x_A), T) and m_best <= resolve * min(m_0, m_T)):
            return None
        return tau, self._flow(a, x_A, tau)

    def _explain_gap(self, g, accept=SWITCH_ACCEPT, max_pairs=36):
        """Explains a gap with the learned modes. Returns 'same' if one mode alone reconstructs it (no
        switch in the gap: a change point detected inside a mode), (tau, x_switch) if a switch between
        two modes is located (the gap's entries 'a' and 'b' are then set), or None.

        When the mode on a side is not known (g['a'] or g['b'] is None, e.g. its segment is too short
        for a derivative), all modes are candidates for that side, up to `max_pairs` pairs; the pair
        whose reconstruction fits best is kept."""
        x_A, x_B, T = np.asarray(g['x_A'], dtype=float), np.asarray(g['x_B'], dtype=float), g['T']
        inside, times = g.get('inside'), g.get('inside_times')
        scale = np.linalg.norm(x_B - x_A)
        if T <= 0 or scale == 0:
            return None
        known = lambda j: j is not None and 0 <= j < self.n  # a mode index can be stale if modes were merged
        cand_a = [g['a']] if known(g['a']) else list(range(self.n))
        cand_b = [g['b']] if known(g['b']) else list(range(self.n))
        for j in cand_a:
            if j in cand_b:
                try:
                    sq = [np.sum((self._flow(j, x_A, T) - x_B) ** 2)]
                    if inside is not None and len(inside):
                        sq += [np.sum((self._flow(j, x_A, float(t)) - x) ** 2) for x, t in zip(inside, times)]
                    m = float(np.sqrt(np.mean(sq)))
                except (ValueError, ArithmeticError, np.linalg.LinAlgError):
                    continue
                if m <= self._miss_tol(accept, scale, len(x_A), T):
                    return 'same'
        pairs = [(a, b) for a in cand_a for b in cand_b if a != b]
        if len(pairs) > max_pairs:
            return None
        best = None
        for a, b in pairs:
            out = self.locate_switch(x_A, x_B, T, a, b, inside, times, accept=accept)
            if out is not None:
                miss = np.linalg.norm(self._flow(b, out[1], T - out[0]) - x_B)  # end-point mismatch, to rank the pairs
                if best is None or miss < best[0]:
                    best = (miss, a, b, out)
        if best is None:
            return None
        g['a'], g['b'] = best[1], best[2]
        return best[3]

    def boundary_points_from_gaps(self, gaps, fine_step):
        """
        For each gap (dict with x_A, x_B, T, a, b and optionally inside, inside_times), locates the
        switch with `locate_switch` and returns points of the reconstructed trajectory on both sides
        of it, spaced by `fine_step`, with their modes: (X, modes, located), where `located[i]` tells
        whether gap i was located. The points come from the learned dynamics, not from the system.
        """
        Xs, Ms, located = [], [], []
        for g in gaps:
            out = self._explain_gap(g)
            located.append(out is not None)
            if out is None or out == 'same':
                continue
            tau, x_s = out
            for j, x0, dur, forward in ((g['a'], g['x_A'], tau, True), (g['b'], x_s, g['T'] - tau, False)):
                # mode a: from x_A up to (not including) the switch; mode b: after the switch up to x_B
                n = int(np.floor(dur / fine_step))
                times = np.arange(0, n + 1) * fine_step if forward else dur - np.arange(0, n + 1) * fine_step
                times = times[(times < dur - 1e-12)] if forward else times[times > 1e-12]
                for t in times:
                    Xs.append(self._flow(j, x0, float(t)))
                    Ms.append(j)
        X = np.array(Xs) if Xs else np.zeros((0, self.Xfit_.shape[1]))
        return X, np.array(Ms, dtype=int), located
