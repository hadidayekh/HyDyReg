"""The learning pipeline: segmentation and derivatives, passive fit, mode regions, boundary refinement and
equivalence queries, as the stages of a `Learner`.

    learn_from_system(system, design, config)    active learning on a system that can be simulated
    learn_from_data(trajectories, config)        passive learning from recorded trajectories

The stages run in this order in `Learner.run`: observe, prepare, fit_passive, classify, refine_boundaries,
equivalence_loop, finish.
"""
import copy
import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from .. import differentiation, segmentation
from ..differentiation import diff_method_fps
from ..metrics import relative_diff
from ..model import HybridRegressor
from ..regression import RidgeRegressor, ridge_refit
from .config import LearnConfig, SamplingDesign
from .equivalence import generate_random_points, equivalence_query_by_traj
from .evaluation import evaluate, evaluate_on_data
from .oracle import noisy_oracle
from .reporting import Reporter
from .result import LearnResult
from .segments import get_modes_val, train_test_split_segments, which_segment_all
from .. import _logging as _log

if TYPE_CHECKING:
    from ..trajectories import Trajectories
from ..tolerances import N_TEST_POINTS, POINT_RD, QUERY_NOISE_RD, TEST_POINTS_SEED

# Held-out evaluation (active learning): HELDOUT_N initial conditions drawn in the equivalence bounds with a fixed
# generator, independent of the seed of the run and of the query sampling.
HELDOUT_N = 50
HELDOUT_SEED = 20261116


class Learner:
    """One learning run.

    config: LearnConfig. oracle: the system as the learner observes it (None for passive learning).
    reference: the noise-free system (test trajectories, derivative errors); defaults to the oracle.
    design: SamplingDesign (active learning) or None. data: Trajectories (passive learning).
    reporter: optional Reporter (files and figures). name: name of the run.
    """

    def __init__(self, config: LearnConfig, oracle=None, reference=None, design: SamplingDesign = None, data=None,
                 test_data=None, reporter=None, name="run", print_on=True):
        self.cfg = config
        self.oracle = oracle
        self.reference = reference if reference is not None else oracle
        self.design = design
        self.data = data
        self.test_data = test_data
        self.reporter = reporter or Reporter(name)
        self.name = name
        self.print_on = print_on
        self.info = {}
        self.boundary_stats = {'gaps': 0, 'explained_with_model': 0, 'queried': 0}
        self.passive = oracle is None
        self.classifier = config.classifier.build()
        self.incremental = config.method == "incremental"
        # independent random streams derived from the seed of the run (no use of NumPy's global state)
        s_eq, s_split, s_noise, s_starts = np.random.SeedSequence(config.seed).spawn(4)
        self.rng_eq = np.random.default_rng(s_eq)
        self.split_state = np.random.RandomState(np.random.default_rng(s_split).integers(2**31))
        self.noise_seed = s_noise
        o = config.optimization
        self.start_seed = int(o.init_seed) if o.init_seed is not None else int(np.random.default_rng(s_starts).integers(2**31 - 10**6))

    # ------------------------------------------------------------------ helpers (formerly closures)

    def deriv_noise_at(self, h):
        return differentiation.FPS_NOISE_GAIN * self.sigma_hat / h

    def seg_tol_at(self, h, base=None):
        base = self.cfg.seg_tol if base is None else base
        return max(base, 4 * differentiation.SEG_NOISE_GAIN * self.sigma_hat * np.sqrt(self.D) / h)

    def segment_and_differentiate(self, trajs, tag):
        '''Segments a batch of observed trajectories and estimates the derivatives, with windows that follow the
        estimated noise (noise.adapt). `trajs` is not modified. Returns (segmented trajectories, segments, X,
        X_dot, seg_res, half-width of the derivative window, noise standard deviation of the derivatives,
        estimated bias of the derivatives).'''
        adapt = self.cfg.noise.adapt
        h = trajs.t_list[0][1] - trajs.t_list[0][0]
        if adapt:
            m_seg, thr = segmentation.choose_segmentation_window(trajs.x_list, h, self.cfg.seg_tol, self.sigma_hat)
        else:
            m_seg, thr = 2, self.seg_tol_at(h)
        trajs, segs = trajs.segment(thr, m=m_seg)
        if adapt:
            m_der, d_noise, d_bias = differentiation.choose_derivative_window(trajs.t_list, trajs.x_list, segs, self.sigma_hat)
            # tolerances use the noise of the narrowest window a kept segment may use
            d_noise = self.sigma_hat * differentiation.tolerance_noise_gain(m_der) / h
        else:
            m_der, d_noise, d_bias = 2, self.deriv_noise_at(h), 0.0
        if m_der == 2:
            X_b, X_dot_b, seg_res_b, _ = diff_method_fps(trajs.t_list, trajs.x_list, segments=segs)
        else:
            X_b, X_dot_b, seg_res_b, _ = differentiation.diff_method_sg(trajs.t_list, trajs.x_list, segments=segs, m=m_der)
        if m_seg > 2 or m_der > 2:
            _log.info(f"{tag}: segmentation on {2*m_seg+1} samples (threshold {thr:.3g}), derivatives on {2*m_der+1} samples "
                  f"(noise {d_noise:.2e}, bias {d_bias:.2e})")
        self.info.setdefault('windows', {})[tag] = dict(m_seg=int(m_seg), seg_threshold=float(thr), m_deriv=int(m_der),
                                                        deriv_noise=float(d_noise), deriv_bias=float(d_bias))
        return trajs, segs, X_b, X_dot_b, seg_res_b, m_der, d_noise, d_bias

    def query_gap(self, model, x_A, T, fine_step):
        '''Thesis-style refinement of one gap: simulates the system from x_A during T with the fine time
        step and returns the fine points with their estimated derivatives. Points whose derivative
        stencil spans the switch, and points that no learned mode explains, are dropped.'''
        fine = self.oracle.simulate_trajectory(fine_step, T, [x_A])
        Xf, Xdf, _, _ = differentiation.diff_method_fps(fine.t_list, fine.x_list, segments=[[]])
        Xf_trans = model.poly.fit_transform(Xf) if model.deg > 1 else Xf
        R = model._pointwise_RD(Xf_trans, Xdf)
        a, b = np.argmin(R[:, 0]), np.argmin(R[:, -1])
        if a != b:
            # change point c minimizing the residual of "mode a before c, mode b from c on"
            cost = np.r_[0, np.cumsum(R[a])] + np.r_[np.cumsum(R[b][::-1])[::-1], 0]
            c = int(np.argmin(cost))
            keep = np.ones(len(Xf), dtype=bool)
            keep[max(c-2, 0):c+2] = False
            Xf, Xdf, R = Xf[keep], Xdf[keep], R[:, keep]
        explained = R.min(axis=0) < POINT_RD
        return Xf[explained], Xdf[explained]

    def detected_gaps(self, model, trajs, segs):
        '''Gaps between consecutive detected segments of (segmented) trajectories: the observed states on
        both sides, the modes on both sides (from the five-point derivative of the nearest interior point
        of each segment), and the observed states that segmentation removed inside the gap.'''
        gaps = []
        for i, cps in enumerate(segs):
            t, x = trajs.t_list[i], trajs.x_list[i]
            og_t, og_x = trajs.og_t_list[i], trajs.og_x_list[i]
            bounds_i = [0] + list(cps) + [len(t)]
            for q, k in enumerate(cps):
                if k - 1 < 0 or k >= len(t):
                    continue
                a = b = None
                if self.gap_modes_from_derivative and k - bounds_i[q] >= 5:
                    h = t[k-1] - t[k-2]
                    d5 = (-x[k-1] + 8*x[k-2] - 8*x[k-4] + x[k-5]) / (12*h)
                    a = model.mode_of_point(x[k-3], d5)
                if self.gap_modes_from_derivative and bounds_i[q+2] - k >= 5:
                    h = t[k+1] - t[k]
                    d5 = (-x[k+4] + 8*x[k+3] - 8*x[k+1] + x[k]) / (12*h)
                    b = model.mode_of_point(x[k+2], d5)
                inside = (og_t > t[k-1] + 1e-9) & (og_t < t[k] - 1e-9)
                gaps.append(dict(x_A=x[k-1], x_B=x[k], T=t[k] - t[k-1], a=a, b=b,
                                 inside=og_x[inside], inside_times=og_t[inside] - t[k-1]))
        return gaps

    def boundary_refinement(self, model, trajs, segs, with_recovered, time_step):
        '''Points for the classifier at the change points of a batch of trajectories.
        Returns (X_q, X_dot_q, X_m, modes_m): fine points queried from the system (to be labelled by
        `improve_classification`) and points located with the learned modes (with their modes).
        `time_step` is the sampling step of `trajs`; the fine step is time_step / boundaries.fine_step_factor.'''
        fine_step = time_step / self.cfg.boundaries.fine_step_factor
        D, stats = self.D, self.boundary_stats
        recovered, model.split_gaps_ = (model.split_gaps_, []) if with_recovered else ([], model.split_gaps_)
        if self.cfg.boundaries.points == 'query':
            X_q, X_dot_q = self.oracle.generate_points_between_segments(trajs, segs, self.cfg.seg_tol/10, timestep=fine_step)
            for g in recovered:
                Xf, Xdf = self.query_gap(model, g['x_A'], g['n_steps'] * g['step'], fine_step)
                X_q, X_dot_q = append_points(X_q, Xf), append_points(X_dot_q, Xdf)
            stats['gaps'] += sum(len(c) for c in segs) + len(recovered)
            stats['queried'] += sum(len(c) for c in segs) + len(recovered)
            return X_q, X_dot_q, np.zeros((0, D)), np.zeros(0, dtype=int)
        gaps = self.detected_gaps(model, trajs, segs)
        for g in recovered:
            n_in = len(g['inside'])
            gaps.append(dict(x_A=g['x_A'], x_B=g['x_B'], T=g['n_steps'] * g['step'], a=g['a'], b=g['b'],
                             inside=g['inside'], inside_times=g['step'] * np.arange(1, n_in + 1)))
        X_m, modes_m, located = model.boundary_points_from_gaps(gaps, fine_step)
        X_q, X_dot_q = np.zeros((0, D)), np.zeros((0, D))
        # With measurement noise, the derivatives of a fine-step query carry the noise divided by the fine step:
        # when the relative difference this alone produces exceeds 5%, the query labels are not reliable
        # and the gap is left without boundary points.
        s_fine = differentiation.FPS_NOISE_GAIN * self.sigma_hat * np.sqrt(D) / fine_step
        v_typ = float(np.median(np.linalg.norm(model.X_dot_fit_, axis=1))) if len(model.X_dot_fit_) else 0.0
        query_usable = s_fine / (2 * v_typ + s_fine + 1e-300) < QUERY_NOISE_RD
        for g, ok in zip(gaps, located):
            if ok or (g['a'] is not None and g['a'] == g['b']):
                continue
            if self.passive:
                # nothing to query: the gap is left without boundary points
                stats['unresolved'] = stats.get('unresolved', 0) + 1
                continue
            if not query_usable:
                stats['skipped_noisy_query'] = stats.get('skipped_noisy_query', 0) + 1
                continue
            # not located and not a change point inside a single mode: ask the system
            Xf, Xdf = self.query_gap(model, g['x_A'], g['T'], fine_step)
            X_q, X_dot_q = append_points(X_q, Xf), append_points(X_dot_q, Xdf)
            stats['queried'] += 1
        stats['gaps'] += len(gaps)
        stats['explained_with_model'] += int(np.sum(located))
        return X_q, X_dot_q, X_m, modes_m

    def well_classified(self, model):
        '''True if the classifier predicts the mode of all its training points (with a single mode there is no classifier).'''
        if len(np.unique(model.modes_classify_)) < 2:
            return True
        Xc = model.classifier_poly_.fit_transform(model.X_classify_) if self.cfg.boundary_degree > 1 else model.X_classify_
        return not (model.classifier_.predict(Xc) - model.modes_classify_).any()

    def evaluate_stage(self, stage, model):
        mxs = self.cfg.simulation.max_step_factor
        if self.passive:
            errors = {'train': evaluate_on_data(model, self.trajs_train_unsegmented, mxs)[0]}
            if self.trajs_test is not None:
                errors['test'] = evaluate_on_data(model, self.trajs_test, mxs)[0]
            return errors
        errors, learned = evaluate(model, self.trajs_train_unsegmented, self.train_design,
                                   self.trajs_test, self.test_design, mxs)
        self.reporter.stage_figures(stage, model, self.trajs_train_unsegmented, self.trajs_test, learned)
        return errors

    # ------------------------------------------------------------------ stages

    def observe(self):
        """Training and test trajectories, and the estimated measurement noise."""
        cfg = self.cfg
        if self.passive:
            # the input is copied: segmentation modifies the Trajectories object
            trajs_train = copy.deepcopy(self.data)
            self.train_design = (trajs_train.t_list[0][1] - trajs_train.t_list[0][0], None, None)
            self.trajs_test = copy.deepcopy(self.test_data) if self.test_data is not None else None
            self.test_design = None
        d = self.design
        if not self.passive:
            self.D = len(d.x0s[0])
            x0s_test = d.test_x0s
            if isinstance(x0s_test, str):
                x0s_test = generate_random_points(N_TEST_POINTS, np.asarray(cfg.equivalence.bounds, dtype=float), random_generator=np.random.default_rng(TEST_POINTS_SEED))
            self.x0s_test = x0s_test
            # What the learner observes from the system (with measurement noise if noise_std > 0).
            # Test trajectories and derivative errors use the noise-free reference system.
            self.oracle = noisy_oracle(self.oracle, d.noise_std, seed=self.noise_seed)
            trajs_train = self.oracle.simulate_trajectory(d.time_step, d.time_range, d.x0s, max_step_factor=cfg.simulation.max_step_factor)
            self.train_design = (d.time_step, d.time_range, d.x0s)
            self.test_design = (d.test_time_step, d.test_time_range, x0s_test)
        else:
            self.D = trajs_train.x_list[0].shape[1]

        # Estimated measurement noise, and the tolerances that follow it (see NoiseConfig)
        self.sigma_hat = differentiation.estimate_noise_std(trajs_train.x_list) if cfg.noise.adapt else 0.0
        h_train = trajs_train.t_list[0][1] - trajs_train.t_list[0][0]
        if cfg.noise.adapt:
            _log.info(f"Estimated measurement noise: {self.sigma_hat:.3e}; derivative noise {self.deriv_noise_at(h_train):.3e}; "
                  f"segmentation tolerance {self.seg_tol_at(h_train):.3g} (given: {cfg.seg_tol})")
        self.info.update(noise_std=(d.noise_std if d is not None else None), noise_std_estimated=self.sigma_hat,
                         seg_tol_used=self.seg_tol_at(h_train))
        if not self.passive:
            self.trajs_test = self.reference.simulate_trajectory(
                d.test_time_step, d.test_time_range, x0s_test, max_step_factor=cfg.simulation.max_step_factor)
        self.trajs_train = trajs_train

    def prepare(self):
        """Segmentation, derivatives, segment of each point, insensitivity error."""
        cfg = self.cfg
        self.trajs_train_unsegmented = self.trajs_train
        trajs_train_sul, segments, X, X_dot, seg_res, m_deriv_train, deriv_noise_train, deriv_bias_train = self.segment_and_differentiate(self.trajs_train, 'training')
        self.trajs_train = trajs_train_sul
        if X.shape[0] == 0:
            raise ValueError("Segmentation left no usable training point: every segment has fewer than 5 points. "
                             "The segmentation tolerance is too small for these data (too much noise for this time step?).")
        # seg_all[i] = segment index of point X[i] (sigma in the thesis); used by the optimization-based method
        seg_all = which_segment_all(X.shape[0], seg_res)

        deltas = cfg.optimization.deltas
        if isinstance(deltas, str):
            # segments that their own polynomial fit does not capture (an undetected switch) are left out
            from sklearn.preprocessing import PolynomialFeatures
            feats = PolynomialFeatures(degree=cfg.poly_degree, include_bias=False)
            suspicious = []
            for k_seg, (a_seg, b_seg) in enumerate(zip(seg_res, list(seg_res[1:]) + [X.shape[0]])):
                F_seg = feats.fit_transform(X[a_seg:b_seg])
                if b_seg - a_seg >= 2 * (F_seg.shape[1] + 1):
                    pred = LinearRegression().fit(F_seg, X_dot[a_seg:b_seg]).predict(F_seg)
                    with np.errstate(invalid='ignore', divide='ignore'):
                        if not np.nanmean(relative_diff(X_dot[a_seg:b_seg], pred)) < cfg.rd_tol:
                            suspicious.append(k_seg)
            if m_deriv_train == 2:
                deltas = differentiation.auto_deltas(trajs_train_sul.t_list, trajs_train_sul.x_list, segments, X_dot, exclude=suspicious,
                                                     deriv_noise=deriv_noise_train)
            else:  # wider derivative window: its estimated bias replaces the five- vs seven-point estimate
                deltas = max(2 * deriv_bias_train + 4 * deriv_noise_train, 1e-6 * float(np.max(np.abs(X_dot))))
            _log.info(f"deltas = 'auto': insensitivity error set to {deltas:.3e} (twice the largest estimated derivative error"
                  + (f"; {len(suspicious)} segment(s) not captured by their own polynomial fit left out)" if suspicious else ")"))
            self.reporter.log(f"deltas (auto) = {deltas}\n")
        self.info['deltas'] = deltas if np.isscalar(deltas) else list(deltas)
        self.segments, self.X, self.X_dot, self.seg_res, self.seg_all = segments, X, X_dot, seg_res, seg_all
        self.deltas, self.m_deriv_train, self.deriv_noise_train = deltas, m_deriv_train, deriv_noise_train

    def build_model(self):
        cfg = self.cfg
        _log.debug("Starting " + self.name + "\n")
        model_curr = HybridRegressor(n=cfg.n_modes, deg=cfg.poly_degree, classifier_poly_deg=cfg.boundary_degree, rd_tol=cfg.rd_tol)
        model_curr.noise_std_ = self.sigma_hat
        model_curr.deriv_noise_ = self.deriv_noise_train
        # Incremental method: with noticeable derivative noise (the tolerance of the checks is above the given
        # good_RD on the training data), the modes are fitted by ridge regression with the strength selected on
        # held-out blocks; otherwise by least squares as in the thesis.
        noise_engaged = cfg.noise.adapt and len(self.X_dot) > 0 and model_curr._rd_tol(self.X_dot, cfg.rd_tol) > cfg.rd_tol
        self.incr_regressor = RidgeRegressor() if noise_engaged else LinearRegression()
        self.info['noise_engaged'] = bool(noise_engaged)
        # with a wider window, the modes on both sides of a gap are found by reconstruction
        self.gap_modes_from_derivative = self.m_deriv_train == 2
        self.model_curr = model_curr

    def fit_passive(self):
        """Fits the dynamics on the training data (clusterwise, IPOPT program, or incremental method)."""
        if not self.incremental and self.cfg.optimization.solver == 'clusterwise':
            self._fit_clusterwise()
        elif not self.incremental:
            self._fit_ipopt()
        else:
            self._fit_incremental()
        model = self.model
        if not self.incremental:
            # the number of modes is imposed: updates of the dynamics (split step, update_dynamics) never add a mode
            model.split_step_ = self.train_design[0]
            model.max_modes = model.n
            model.regressors_ = [None] * model.n
            seg_fit = self.seg_all if len(model.Xfit_) == len(self.seg_all) else self.seg_train
            model.split_mixed_segments(seg_fit)
        if self.cfg.noise.integral_refit and self.info.get('noise_engaged'):
            self._integral_refit(model)

    def _integral_refit(self, model, L=10):
        '''Refits each mode by integral least squares (regression.integral_refit) on runs of consecutive training
        samples that the passive fit assigned to it: within each training segment, maximal runs of points with
        the same mode in model.modesfit_. Modes with too few windows keep their coefficients.'''
        from ..regression import integral_refit
        mode_of = {tuple(x): int(m) for x, m in zip(model.Xfit_, model.modesfit_)}
        bounds = list(self.seg_res) + [len(self.X)]
        runs = {}
        for a, b in zip(bounds[:-1], bounds[1:]):
            labels = [mode_of.get(tuple(x)) for x in self.X[a:b]]
            i = 0
            while i < len(labels):
                j = i
                while j < len(labels) and labels[j] == labels[i]:
                    j += 1
                if labels[i] is not None:
                    runs.setdefault(labels[i], []).append(self.X[a + i:a + j])
                i = j
        refitted = []
        for mode, rs in runs.items():
            fit = integral_refit(rs, model.poly, self.train_design[0], L=L)
            if fit is not None:
                model.Omegas_[mode], model.betas_[mode] = fit
                refitted.append(mode)
        _log.info(f"Integral refit of mode(s) {sorted(refitted)} (windows of {L} samples)")
        self.info['integral_refit_modes'] = sorted(refitted)

    def _fit_clusterwise(self):
        X, X_dot, seg_all, model_curr = self.X, self.X_dot, self.seg_all, self.model_curr
        t_start = time.time()
        model_curr.fit_clusterwise(X, X_dot, seg_all, n_restarts=self.cfg.optimization.clusterwise_restarts, seed=self.start_seed)
        self.time_passive = time.time() - t_start
        model = model_curr
        MSE_train = mean_squared_error(model.predict(X, labelfun=model.modesfit_), X_dot)
        objs = model.restart_objectives_
        n_best = int(np.sum(objs <= model.objective_ + 1e-9 * np.sum(X_dot ** 2)))
        if self.print_on:
            _log.info(f"Clusterwise regression: objective = {model.objective_:.3e} (reached by {n_best} of {len(objs)} starts), "
                  f"training MSE = {MSE_train:.3e}, time = {self.time_passive:.2f}")
            _log.info("Segments per mode:", np.bincount(np.argmin(model.Xis_segments_, axis=0)[np.unique(seg_all)], minlength=self.cfg.n_modes))
        self.reporter.log(f"Clusterwise regression: objective = {model.objective_} (reached by {n_best} of {len(objs)} starts), "
                          f"training MSE = {MSE_train}, time = {self.time_passive}\n")
        self.model = model

    def _fit_ipopt(self):
        o = self.cfg.optimization
        X, X_dot, seg_res, seg_all, deltas, model_curr = self.X, self.X_dot, self.seg_res, self.seg_all, self.deltas, self.model_curr
        apply_LR = o.refit != 'none'
        opti_refit = o.refit if apply_LR else 'ridge'
        if not apply_LR and all(float(c) == 0.0 for c in o.C_space):
            import warnings
            msg = ("optimization.refit='none' with C_space = [0]: no regularization. The coefficients are those of the "
                   "program of thesis 4.2 at C = 0, where any coefficients within the insensitivity error deltas fit "
                   "equally well. Give C_space values > 0 (e.g. [0, 0.001, 0.01]) to regularize them.")
            warnings.warn(msg)
            _log.warning(msg)
        # Splitting the data into train and validation per segment for tuning hyperparameter C
        X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val = train_test_split_segments(
            X, X_dot, seg_res, seg_all, test_size=0.2, random_state=self.split_state)
        self.seg_train = seg_train

        # Starting the hyperparamater tuning
        Cs = []
        MSE_val = np.inf
        model = None
        total_time = 0
        t_search = time.time()

        C_tested = []
        warm = None
        if o.init == 'clusterwise':
            # one start for the smallest C, from the dynamics of the clusterwise solution on the training split
            t_cw = time.time()
            cw = HybridRegressor(n=model_curr.n, deg=model_curr.deg, classifier_poly_deg=model_curr.classifier_poly_deg)
            cw.fit_clusterwise(X_train, Xdot_train, seg_train, n_restarts=o.clusterwise_restarts, seed=self.start_seed)
            warm = cw.warm_start_coefs()
            first_from_cw = True
            _log.info(f"Clusterwise start: objective {cw.objective_:.3e} ({time.time() - t_cw:.2f} s)")
        else:
            first_from_cw = False
        for i, C_curr in enumerate(sorted(o.C_space)):
            t_start = time.time()
            try:
                # Fit the model with the current hyperparameter C_curr
                stop_tol = (1e-3 * np.min(deltas)) if o.stop_tol == 'auto' else float(o.stop_tol)
                if first_from_cw:
                    # smallest C from the clusterwise dynamics: one solve; the early stop is irrelevant
                    first_from_cw = False
                    model_curr.fit_opti_multistart(X_train, Xdot_train, n_starts=1, seed=self.start_seed, stop_tol=stop_tol,
                                   norm='L2', deltas=deltas, C=C_curr, segmentation='strict', segments=seg_train, apply_LR=apply_LR,
                                   init=warm, refit=opti_refit)
                elif warm is not None:
                    # continuation in C: one solve, started from the solution of the previous C
                    model_curr.fit_opti_multistart(X_train, Xdot_train, n_starts=1, seed=self.start_seed, stop_tol=None,
                                   norm='L2', deltas=deltas, C=C_curr, segmentation='strict', segments=seg_train, apply_LR=apply_LR,
                                   init=warm, refit=opti_refit)
                else:
                    model_curr.fit_opti_multistart(X_train, Xdot_train, n_starts=o.n_starts, seed=self.start_seed, stop_tol=stop_tol,
                                   norm='L2', deltas=deltas, C=C_curr, segmentation='strict', segments=seg_train, apply_LR=apply_LR,
                                   init=o.init, refit=opti_refit)
                warm = model_curr.warm_start_coefs()
                curr_time = time.time() - t_start
                total_time += curr_time
                objs = model_curr.start_objectives_
                _log.info(f"C = {C_curr}: {model_curr.n_starts_run_} start(s), best objective {model_curr.objective_:.3e}, "
                      f"objectives of the starts: {np.array2string(objs, precision=2)}")
                slack = getattr(model_curr, 'max_min_slack_', None)
                if slack is not None and slack > stop_tol and model_curr.n_starts_run_ > 1:
                    if C_curr == 0:
                        # with C = 0 a solution with zero slacks exists whenever the data fit the model within deltas
                        _log.warning(f"  Warning: in the best solution found, some segment is fitted by no mode within the "
                              f"insensitivity error (deltas = {deltas}): its smallest slack is {slack:.2e}. All "
                              f"{model_curr.n_starts_run_} starts were run. A small value suggests that deltas is "
                              f"smaller than the accuracy of the estimated derivatives (try deltas='auto'); a large "
                              f"one, that no start reached a good optimum (more starts may help).")
                    else:
                        _log.warning(f"  All {model_curr.n_starts_run_} starts were run: with C > 0 the optimum may keep "
                              f"positive slacks (smallest slack of the worst segment: {slack:.2e}).")
                    self.reporter.log(f"C = {C_curr}: no solution fits every segment within deltas; exceeded by up to {slack}\n")
            except (RuntimeError, ArithmeticError, np.linalg.LinAlgError) as e:  # every start failed for this C
                curr_time = time.time() - t_start
                total_time += curr_time
                _log.warning('C =', C_curr, 'failed:', type(e).__name__, str(e)[:200])
                self.reporter.log('C = ' + str(C_curr) + ' failed.\n')
                continue
            finally:
                C_tested.append(C_curr)

            # Deduce the modes of validation points and compute validation error
            modes_val = get_modes_val(model_curr.Xis_segments_, seg_val)
            Xdot_val_pred = model_curr.predict(X_val, labelfun=modes_val)
            MSE_val_curr = mean_squared_error(Xdot_val_pred, Xdot_val)

            # Compare with previous best error to find the new best
            Cs.append(C_curr)
            if MSE_val_curr < MSE_val:
                model = copy.deepcopy(model_curr)
                MSE_val = MSE_val_curr
                C = C_curr
                time_selected_C = curr_time

            if self.print_on:
                _log.info("C =", C_curr, "MSE =", MSE_val_curr, 'time =', curr_time)
            self.reporter.log("C = " + str(C_curr) + " MSE = " + str(MSE_val_curr) + ' time = ' + str(curr_time) + "\n")

        if not Cs:
            raise RuntimeError("The optimization failed for every value of C in C_space.")
        if model is None:
            raise RuntimeError(f"No value of C in C_space gave a finite validation error (tried {Cs}); "
                               "the fitted modes cannot predict the validation points.")

        if apply_LR:
            # C is selected: refit the dynamics on all points (training and validation) with the selected
            # segment-to-mode assignment, and keep all points for the classifier.
            seg_modes = np.argmin(model.Xis_segments_, axis=0)
            modes_all = seg_modes[seg_all]
            X_all_trans = model.poly.fit_transform(X)
            for j in range(model.n):
                if np.any(modes_all == j):
                    if opti_refit == 'ridge':
                        model.Omegas_[j], model.betas_[j], alpha_j, _ = ridge_refit(
                            X_all_trans[modes_all == j], X_dot[modes_all == j], seg_all[modes_all == j])
                        model.ridge_alphas_ = {**getattr(model, 'ridge_alphas_', {}), j: alpha_j}
                    else:
                        LR_all = LinearRegression().fit(X_all_trans[modes_all == j], X_dot[modes_all == j])
                        model.Omegas_[j], model.betas_[j] = LR_all.coef_, LR_all.intercept_
            if opti_refit == 'ridge':
                _log.info("Ridge refit: selected strengths per mode:", getattr(model, 'ridge_alphas_', {}))
                self.info['ridge_alphas'] = {int(k): float(v) for k, v in getattr(model, 'ridge_alphas_', {}).items()}
            model.Xfit_, model.Xfit_trans_, model.X_dot_fit_ = X, X_all_trans, X_dot
            model.Xis_ = model.Xis_segments_[:, seg_all]
            model.modesfit_ = modes_all

        # passive time: the whole search over C (every value tried) and the final refit
        self.time_passive = time.time() - t_search
        self.info['time_passive_selected_C'] = time_selected_C
        if self.print_on:
            _log.info("Best C =", C, "with MSE_val =", MSE_val)
            _log.info("Total time:", total_time)
            _log.info("Average time per C:", total_time/len(C_tested))
        self.reporter.log("Best C = " + str(C) + " with MSE_val = " + str(MSE_val) + "\n"
                          + "Total time = " + str(total_time) + "\n")
        self.model = model

    def _fit_incremental(self):
        model = self.model_curr
        start = time.time()
        model.split_step_ = self.train_design[0]
        model.fit_first_incremental(self.X, self.X_dot, self.seg_res, regressor=self.incr_regressor)
        model.merge_similar_modes(regressor=self.incr_regressor)
        self.time_passive = time.time() - start
        self.model = model

    def classify(self):
        """First classifier; snapshot 'passive' and its errors."""
        model = self.model
        model.classify_modes(self.classifier)
        # Store first model (purely passive)
        self.first_model = copy.deepcopy(model)
        self.errors = {'passive': self.evaluate_stage('passive', self.first_model)}

    def refine_boundaries(self):
        """Boundary points at the change points of the training data; snapshot 'refined' and its errors."""
        model = self.model
        start = time.time()
        fine_X, fine_X_dot, bnd_X, bnd_modes = self.boundary_refinement(model, self.trajs_train, self.segments, with_recovered=True,
                                                                          time_step=self.train_design[0])
        model.improve_classification(fine_X, fine_X_dot, self.classifier)
        model.add_classification_points(bnd_X, bnd_modes, self.classifier)
        self.time_refinement = time.time() - start
        self.learned_SNDS_active_before_EQ = copy.deepcopy(model.to_SNDS())
        self.model_active_before_EQ = copy.deepcopy(model)
        self.errors['refined'] = self.evaluate_stage('refined', self.model_active_before_EQ)
        if self.print_on:
            self._print_boundaries(model)

    def _print_boundaries(self, model):
        if _log.enabled(10):
            from ..export.text import regions_text
            _log.debug(regions_text(model))

    def equivalence_loop(self):
        """Equivalence queries; each counterexample batch updates the model."""
        cfg, model, e = self.cfg, self.model, self.cfg.equivalence
        sul_obs = self.oracle
        bounds = np.asarray(e.bounds, dtype=float)
        t_step = e.time_step if e.time_step is not None else self.design.test_time_step
        t_range = e.time_range if e.time_range is not None else self.design.test_time_range
        mxs = cfg.simulation.max_step_factor
        improve_cont = cfg.update_dynamics

        learned_SNDS = self.learned_SNDS_active_before_EQ
        start = time.time()
        _log.info("Checking equivalence query...")
        eq_query, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned = equivalence_query_by_traj(
            sul_obs, learned_SNDS, e.n_trajs, bounds, t_step, t_range, e.tol, N_cexs=e.max_counterexamples, max_step_factor=mxs, rng=self.rng_eq)
        sul_trajs = copy.deepcopy(cexs_sul)
        sul_trajs.append_trajs(conforming_trajs_sul)
        count = 1
        self.reporter.query_figure(count, model, sul_trajs, cexs_learned, conforming_trajs_learned)

        while not eq_query and count < e.max_rounds:
            count += 1
            _log.info("Equivalence query is FALSE. Improving...")
            cexs_seg, segments_eq, X, X_dot, seg_res_eq, m_deriv_cex, deriv_noise_cex, _ = self.segment_and_differentiate(cexs_sul, 'counterexamples')
            model.deriv_noise_ = deriv_noise_cex
            self.gap_modes_from_derivative = m_deriv_cex == 2
            if X.shape[0] == 0:
                _log.info("The counterexamples have no segment of at least 5 points; stopping.")
                break
            # Improve continuous dynamics if necessary
            if improve_cont:
                _log.debug("Checking to improve continuous dynamics...")
                model.split_step_ = t_step  # sampling step of the counterexample trajectories
                for i in range(len(seg_res_eq) - 1):
                    X_curr = X[seg_res_eq[i]:seg_res_eq[i+1]]
                    _log.debug("Segment", i, "with", X_curr.shape[0], "points:")
                    X_dot_curr = X_dot[seg_res_eq[i]:seg_res_eq[i+1]]
                    model.update_continuous(X_curr, X_dot_curr, regressor=self.incr_regressor)
                if len(seg_res_eq) > 0:
                    X_curr = X[seg_res_eq[-1]:]
                    X_dot_curr = X_dot[seg_res_eq[-1]:]
                    _log.debug("Last segment", ":")
                    model.update_continuous(X_curr, X_dot_curr, regressor=self.incr_regressor)
                model.retry_pending(regressor=self.incr_regressor)
                if self.incremental:
                    model.merge_similar_modes(regressor=self.incr_regressor)

            # Improving classifier / discrete dynamics
            fine_X_eq, fine_X_dot_eq, bnd_X, bnd_modes = self.boundary_refinement(
                model, cexs_seg, segments_eq, with_recovered=improve_cont, time_step=t_step)
            if improve_cont:
                model.classify_modes(self.classifier)
                X_classify = fine_X_eq
                X_dot_classify = fine_X_dot_eq
            else:
                X_classify = np.r_[X, fine_X_eq] if len(fine_X_eq) > 0 else X
                X_dot_classify = np.r_[X_dot, fine_X_dot_eq] if len(fine_X_dot_eq) > 0 else X_dot

            model.improve_classification(X_classify, X_dot_classify, self.classifier)
            model.add_classification_points(bnd_X, bnd_modes, self.classifier)
            self._print_boundaries(model)

            _log.debug("IS EVERYTHING WELL CLASSIFIED ?", self.well_classified(model))
            learned_SNDS = model.to_SNDS()
            _log.info("Rechecking equivalence query...")
            eq_query, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned = equivalence_query_by_traj(
                sul_obs, learned_SNDS, e.n_trajs, bounds, t_step, t_range, e.tol, N_cexs=e.max_counterexamples, max_step_factor=mxs, rng=self.rng_eq)
            sul_trajs = copy.deepcopy(cexs_sul)
            sul_trajs.append_trajs(conforming_trajs_sul)
            self.reporter.query_figure(count, model, sul_trajs, cexs_learned, conforming_trajs_learned)

        if eq_query:
            _log.info("Equivalence query is TRUE!")
            _log.info("NUMBER OF EQ QUERIES:", count)
            self._print_boundaries(model)
        else:
            _log.info("NOT TERMINATED: the last equivalence query still returned counterexamples after", count, "queries.")
        self.info.update(terminated=bool(eq_query), n_modes=int(model.n), n_eq=int(count),
                         n_pending_segments=len(getattr(model, 'pending_', [])),
                         n_pending_points=int(sum(len(p[0]) for p in getattr(model, 'pending_', []))))
        self.info['boundary'] = dict(self.boundary_stats)
        _log.info("Change points:", self.boundary_stats)
        self.time_EQ = time.time() - start
        self.n_eq, self.terminated, self.learned_SNDS = count, bool(eq_query), learned_SNDS

    def finish(self):
        """Errors of the final model, derivative error on the test set, result."""
        model = self.model
        if self.queries_run:
            self.errors['final'] = self.evaluate_stage('final', model)
        else:
            # no equivalence query: the final model is the one after boundary refinement
            self.learned_SNDS = self.learned_SNDS_active_before_EQ
            self.errors['final'] = self.errors['refined']
            self.time_EQ, self.n_eq, self.terminated = 0.0, 0, None
            self.info.update(terminated=None, n_modes=int(model.n), n_eq=0,
                             n_pending_segments=len(getattr(model, 'pending_', [])),
                             n_pending_points=int(sum(len(p[0]) for p in getattr(model, 'pending_', []))))
            if self.passive:
                self.info['boundary'] = dict(self.boundary_stats)
                if self.boundary_stats.get('unresolved'):
                    import warnings
                    warnings.warn(f"{self.boundary_stats['unresolved']} change point(s) could not be explained with the "
                                  "learned modes and were left without boundary points (no system to query).")

        if not self.passive and self.cfg.equivalence.bounds is not None:
            self.info['heldout'] = self.heldout_errors()

        rd_dot = None
        if self.reference is not None and self.trajs_test is not None:
            # Computing final RD between derivatives
            X_dot_sul = self.reference.get_derivatives(self.trajs_test)
            X_dot_learned = self.learned_SNDS.get_derivatives(self.trajs_test)
            rd_dot = np.mean(relative_diff(X_dot_sul, X_dot_learned))

        if self.print_on:
            for stage, label in (('passive', 'before active learning'), ('refined', 'with active learning before equivalence queries'),
                                 ('final', 'after equivalence queries')):
                for k, v in self.errors[stage].items():
                    _log.info(f"{'Training' if k == 'train' else 'Test'} RD of trajectories {label} = {v}")
        _log.debug("IS EVERYTHING WELL CLASSIFIED ?", self.well_classified(model))
        result = LearnResult(model=model, system=self.learned_SNDS, config=self.cfg, name=self.name,
                             snapshots={'passive': self.first_model, 'refined': self.model_active_before_EQ, 'final': model},
                             errors=self.errors, rd_dot=rd_dot,
                             times={'passive': self.time_passive, 'refinement': self.time_refinement, 'equivalence': self.time_EQ},
                             n_eq=self.n_eq, terminated=self.terminated, info=self.info,
                             train_trajectories=self.trajs_train_unsegmented, test_trajectories=self.trajs_test,
                             reference_system=self.reference, names=getattr(self.data, 'names', None))
        self.reporter.finish(result)
        return result

    def heldout_errors(self):
        '''Errors of the final model on HELDOUT_N trajectories from fixed initial conditions in the equivalence
        bounds (test time step and range), against the noise-free system: mean and max relative difference of
        the trajectories (d_avg, d_max) and mean relative difference of the derivatives along the true
        trajectories (d_dot). A model that cannot be simulated gets d_avg = d_max = 1.'''
        from ..metrics import compare_trajs
        from ..systems import SlidingMotionException
        d = self.design
        bounds = np.asarray(self.cfg.equivalence.bounds, dtype=float)
        x0s = generate_random_points(HELDOUT_N, bounds, random_generator=np.random.default_rng(HELDOUT_SEED))
        mxs = self.cfg.simulation.max_step_factor
        ref = self.reference.simulate_trajectory(d.test_time_step, d.test_time_range, x0s, max_step_factor=mxs)
        try:
            learned = self.learned_SNDS.simulate_trajectory(d.test_time_step, d.test_time_range, x0s, max_step_factor=mxs)
            if getattr(learned, 'truncated', False):
                raise SlidingMotionException
            d_avg, d_max = (float(v) for v in compare_trajs(learned, ref, measure="RD"))
        except SlidingMotionException:  # the learned model could not be simulated to the end
            d_avg = d_max = 1.0
        d_dot = float(np.mean(relative_diff(self.reference.get_derivatives(ref), self.learned_SNDS.get_derivatives(ref))))
        return dict(n=HELDOUT_N, d_avg=d_avg, d_max=d_max, d_dot=d_dot)

    def run(self):
        self.observe()
        self.prepare()
        self.build_model()
        self.fit_passive()
        self.classify()
        self.refine_boundaries()
        self.queries_run = (not self.passive) and self.cfg.equivalence_enabled
        if self.queries_run:
            self.equivalence_loop()
        return self.finish()


def append_points(A, B):
    return B if len(A) == 0 else (A if len(B) == 0 else np.r_[A, B])


def learn_from_system(system: Any, design: SamplingDesign, config: Optional[LearnConfig] = None, name: str = "run",
                      reporter: Optional[Reporter] = None, reference: Any = None, print_on: bool = True) -> LearnResult:
    """Active learning on a system that can be simulated (an SNDS, or any object with simulate_trajectory).

    The system produces the training trajectories, the boundary queries (if boundaries.points='query' or a
    switch cannot be located) and the trajectories of the equivalence queries (if equivalence bounds are
    given). With design.noise_std > 0 the learner observes noisy trajectories; test errors and derivative
    errors are measured against the noise-free system.

    Example:
        system = hydyreg.benchmarks.systems.isolette()
        design = SamplingDesign(time_step=0.25, time_range=25, x0s=[[105.5, 80], [97.5, 100]], test_time_step=0.1)
        config = LearnConfig(method="incremental", seg_tol=0.01,
                             equivalence=EquivalenceConfig(bounds=[[95, 105], [70, 110]], tol=0.05))
        result = learn_from_system(system, design, config)
    """
    config = config or LearnConfig()
    return Learner(config, oracle=system, reference=reference if reference is not None else system, design=design,
                   reporter=reporter, name=name, print_on=print_on).run()


def learn_from_data(trajectories: "Trajectories", config: Optional[LearnConfig] = None,
                    test_trajectories: Optional["Trajectories"] = None, name: str = "run",
                    reporter: Optional[Reporter] = None, print_on: bool = True) -> LearnResult:
    """Passive learning from recorded trajectories (a Trajectories object with a uniform time step).

    No system is queried: boundaries come from switches located with the learned modes (a change point that
    cannot be located is skipped, with a warning), and there is no equivalence query. The errors compare the
    learned system, simulated from the first sample of each trajectory, with the data (and with
    `test_trajectories` if given).

    Example:
        trajs = Trajectories.from_csv("runs/*.csv")
        result = learn_from_data(trajs, LearnConfig(method="incremental", seg_tol=0.01))
        _log.info(result.equations())
    """
    config = config or LearnConfig()
    if config.boundaries.points == 'query':
        raise ValueError("boundaries.points='query' needs a system to query; use learn_from_system")
    check_trajectories(trajectories)
    if test_trajectories is not None:
        check_trajectories(test_trajectories)
    return Learner(config, data=trajectories, test_data=test_trajectories, reporter=reporter, name=name,
                   print_on=print_on).run()


def check_trajectories(trajs: "Trajectories", rtol: float = 1e-6) -> float:
    """Checks that there is at least one trajectory, each with at least 5 samples, and one uniform time step."""
    if trajs is None or len(trajs.t_list) == 0:
        raise ValueError("no trajectory given")
    h0 = None
    for i, t in enumerate(trajs.t_list):
        t = np.asarray(t, dtype=float)
        if len(t) < 5:
            raise ValueError(f"trajectory {i} has {len(t)} samples; at least 5 are needed (five-point derivative)")
        dt = np.diff(t)
        if h0 is None:
            h0 = dt[0]
        if np.max(np.abs(dt - h0)) > rtol * abs(h0):
            bad = int(np.argmax(np.abs(dt - h0)))
            raise ValueError(f"trajectory {i}: the time step is not uniform (step {dt[bad]:.6g} at sample {bad}, "
                             f"expected {h0:.6g}); the segmentation and the derivatives need one constant step")
    return h0
