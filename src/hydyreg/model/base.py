"""Constructor, prediction, conversion to an SNDS and export. Part of HybridRegressor (see hydyreg.model)."""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from ..systems import SNDS, SNDS_affine
from ..export.latex import get_latex_classifier_boundaries, get_latex_eq


class _Base:
    def __init__(self, n=None, deg=1, classifier_poly_deg=1, rd_tol=None):
        from ..tolerances import GOOD_RD
        # tol_RD of the thesis: a mode captures a set of points if their mean relative difference is below it
        self.rd_tol = GOOD_RD if rd_tol is None else rd_tol
        self.pending_ = []
        self.split_gaps_ = []
        self.n = n
        self.deg = deg

        self.classifier_poly_deg = classifier_poly_deg

    def get_points_of_mode(self, j):
        self.check_if_fitted()
        assert j < self.n

        indices = np.where(self.modesfit_ == j)
        return self.Xfit_trans_[indices], self.X_dot_fit_[indices]

    def predict(self, X, labelfun=None):
        # FIRST WE WILL ASSUME THAT THERE IS A GIVEN LABEL FUNCTION
        # Things to check:
        # 1) Regressor has already been fit (can do that by vars(self) and _ elements)
        # 2) X has correct dimensions (X.shape[1] == Xfit_.shape[1])

        self.check_if_fitted()

        if labelfun is None:
            labelfun = self.labelfun_
        
        labellist = False
        if isinstance(labelfun, list) or isinstance(labelfun, np.ndarray):
            labellist = True
            if len(labelfun) != X.shape[0]:
                raise Exception("When labelfun is a list of labels, its length must be equal to the number of datapoints in X.")

        Omegas = self.Omegas_
        betas = self.betas_
        Xfit_trans = self.Xfit_trans_

        # Dimensions of X
        M = X.shape[0]
        D = X.shape[1]
        Q = Xfit_trans.shape[1]

        # Assert that X has correct dimensions
        if D != Q and self.deg == 1:
            raise Exception("Given X does not have the right dimensions.")
        if self.deg > 1:
            poly = PolynomialFeatures(degree = self.deg, include_bias=False)
            X_trans = poly.fit_transform(X)
        else:
            X_trans = X

        X_dot_res = np.zeros((M,D))

        # Mode of each data point, then one matrix product per mode
        if labellist:
            labels = np.asarray(labelfun)
        else:
            labels = np.array([labelfun(X[i]) for i in range(M)])
        for j in np.unique(labels):
            idx = labels == j
            X_dot_res[idx] = X_trans[idx] @ np.asarray(Omegas[j]).T + np.ravel(betas[j])

        return X_dot_res

    def get_modes(self):
        self.check_if_fitted()
        return np.argmin(self.Xis_, axis=0)
    
    def to_SNDS(self, labelfun=None, eventlist=None):
        snds = self._to_SNDS(labelfun, eventlist)
        # exact mode events / Filippov sliding in simulation (only when the model's own labelfun is used)
        if labelfun is None and getattr(self, 'mode_scores_', None) is not None:
            n = self.n
            snds.mode_scores = lambda x, f=self.mode_scores_: f(x)[:n]
        return snds

    def _to_SNDS(self, labelfun=None, eventlist=None):
        self.check_if_fitted()
        if labelfun is None:
            labelfun = self.labelfun_
        if eventlist is None:
            eventlist = self.eventlist_

        n = self.n
        D = self.Xfit_.shape[1]

        if self.deg == 1:
            params = np.ndarray((n, D, D+1))
            for j in range(n):
                for d in range(D):
                    for l in range(D):
                        params[j][d][l] = self.Omegas_[j][d][l]
                    params[j][d][D] = self.betas_[j][d]
            return SNDS_affine(params, eventlist, labelfun)
        else:
            eqs = []
            for j in range(n):
                eqs.append(self.make_eq_predict(j))

            return SNDS(n, D, eqs, eventlist, labelfun)

        
    def make_eq_predict(self, j):
        '''Right-hand side of mode j for the simulator: x -> Omegas_[j] @ phi(x) + betas_[j], phi the monomials
        of degree 1..deg (same order as PolynomialFeatures). The coefficients are read at each call.'''
        D = self.Xfit_.shape[1]
        powers = PolynomialFeatures(degree=self.deg, include_bias=False).fit(np.zeros((1, D))).powers_
        if self.deg == 1:
            def eq(t, x):
                return np.asarray(self.Omegas_[j]) @ np.asarray(x, dtype=float) + np.ravel(self.betas_[j])
        else:
            def eq(t, x):
                phi = np.prod(np.asarray(x, dtype=float) ** powers, axis=1)
                return np.asarray(self.Omegas_[j]) @ phi + np.ravel(self.betas_[j])
        return eq

    def check_if_fitted(self):
        variables = vars(self)
        fitted = False
        for v in variables:
            if v.endswith('_'):
                fitted = True
                break
        if not fitted:
            raise Exception("Estimator not fitted.")

    def gen_full_latex_eq(self, with_classifier_boundaries=True, round_dig=3):
        self.check_if_fitted()
        omegas = self.Omegas_ # n by D by Q tensor
        biases = self.betas_ # n by D matrix

        n, D, Q = omegas.shape
        
        eq_string = ""
        for j in range(n):
            eq_string += f'\\textbf{{Mode {j+1}:}}\n\\[\n\\begin{{cases}}\n'
            for d in range(D):
                eq_string += f'\t\\begin{{aligned}}\n'
                eq_string += f'\t\t\\xddim{{{d+1}}} &=' + get_latex_eq(self.poly, omegas[j][d], biases[j][d], round_dig=round_dig) + '\n'
                eq_string += f'\t\\end{{aligned}}\\\\\n'
            eq_string += f'\\end{{cases}}\n\\]\n'

        if with_classifier_boundaries:
            eq_string += f'\n\n{get_latex_classifier_boundaries(self.classifier_, self.classifier_poly_ if self.classifier_poly_deg > 1 else None)}'
        return eq_string

    def save(self, path, names=None, config=None):
        """Writes the model to a JSON file (see hydyreg.persistence.save_model)."""
        from ..persistence import save_model
        save_model(self, path, names=names, config=config)

    @classmethod
    def load(cls, path):
        """Reads a model written by `save` (see hydyreg.persistence.load_model)."""
        from ..persistence import load_model
        return load_model(path)
