"""Mode regions: fitting the classifier and adding classification points. Part of HybridRegressor (see hydyreg.model)."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from ..classifier import _ScoreClassifier
from .. import _logging as _log



class _RegionsMixin:
    def classify_modes(self, classifier=None, X=None, modes=None, X_dot=None, partial_fit=False):
        '''
        Run a classifier to find mode regions.

        Attributes
        ----------
        classifier : ?, default=None
            The classifier model to be used. Default: a new LogisticRegression().

        X : ndarray (N x D), default=None
            A matrix of training data points [x1, x2, ..., xN].
            When provided, the classification happens using this data. Otherwise, classification happens on previous training data using in fitting continuous dynamics.
        
        modes : ndarray (N), default=None
            An array of modes of each training data point.
            When provided, the classification happens using this data. Otherwise, classification happens on previous training data using in fitting continuous dynamics.
        
        X_dot : ndarray (N x D), default=None
            A matrix of corresponding estimated derivatives of training data points [x1dot, x2dot, ..., xNdot]
            Required when X is provided

        partial_fit : Bool, default=False
            Indicates whether to use `partial_fit` when running the classifier, otherwise runs `fit` (default).
            Could be useful for incremental classifiers.
        '''

        self.check_if_fitted()
        if classifier is None:
            classifier = LogisticRegression()
        if X is None:
            if hasattr(self, 'X_classify_'):
                X = np.r_[self.Xfit_, self.X_classify_]
                X_dot = np.r_[self.X_dot_fit_, self.X_dot_classify_]
                modes = np.r_[self.modesfit_, self.modes_classify_]
                # drop duplicate points (first occurrence kept), keeping the order of the points
                _, unique_indices = np.unique(X, axis=0, return_index=True)
                unique_indices = np.sort(unique_indices)
                X = X[unique_indices]
                X_dot = X_dot[unique_indices]
                modes = modes[unique_indices]
            else:
                X = self.Xfit_
                X_dot = self.X_dot_fit_
                modes = self.modesfit_
        
        num_modes = len(np.unique(modes))

        do_poly_transform = self.classifier_poly_deg > 1
        if do_poly_transform:
            classifier_deg = self.classifier_poly_deg
            classifier_poly = PolynomialFeatures(degree=classifier_deg, include_bias=False)
            X_class_trans = classifier_poly.fit_transform(X)
            self.classifier_poly_ = classifier_poly
        else:
            X_class_trans = X
        
        if num_modes > 1:
            self.classifier_ = classifier.partial_fit(X_class_trans, modes) if partial_fit else classifier.fit(X_class_trans, modes)

            self._install_region_functions(classifier_poly if do_poly_transform else None)
        else:
            self.labelfun_ = lambda x : modes[0]
            self.eventlist_ = None
            self.mode_scores_ = None
            self.classifier_ = classifier
        self.X_classify_ = X
        self.X_dot_classify_ = X_dot
        self.modes_classify_ = modes

        if num_modes > 1:  # with a single mode the classifier is not fitted
            modes_predict = classifier.predict(X_class_trans)
            from sklearn.metrics import accuracy_score
            _log.debug("Classification training accuracy:", accuracy_score(modes, modes_predict))

    def _install_region_functions(self, classifier_poly=None):
        '''Builds labelfun_ (mode of a point), eventlist_ (former event function) and mode_scores_ (scores used by
        the simulator for exact events and sliding) from the fitted classifier_ and, when the boundaries have
        polynomial features, classifier_poly. Used by classify_modes and by persistence.load_model.'''
        do_poly_transform = classifier_poly is not None
        if not do_poly_transform:
            def event(t, x):
                return self.classifier_boundary([x], classifier=self.classifier_)
            lf = lambda x, c=self.classifier_ : c.predict([x])[0]
        else:
            def event(t, x):
                x = classifier_poly.fit_transform([x])
                return self.classifier_boundary(x, classifier=self.classifier_)

            def lf(x, c=self.classifier_):
                x = classifier_poly.fit_transform([x])
                return c.predict(x)[0]
        
        event.direction = 0
        event.terminal = True

        self.labelfun_ = lf
        self.eventlist_ = event

        # Mode scores consistent with lf (predict = argmax of decision_function), used by
        # SNDS.simulate_trajectory for exact mode-change events and Filippov sliding. They are evaluated at every
        # integration step, so they avoid scikit-learn's input validation: the boundary features are the monomials
        # given by classifier_poly.powers_ (the same products as PolynomialFeatures for degree <= 2), and the scores
        # of a logistic regression are features @ coef_.T + intercept_ (the operations of decision_function).
        c0 = self.classifier_
        is_logistic = hasattr(c0, 'coef_') and hasattr(c0, 'intercept_') and not isinstance(c0, _ScoreClassifier)
        if is_logistic or isinstance(c0, _ScoreClassifier):
            powers = classifier_poly.powers_ if do_poly_transform else None

            def features(x):
                x = np.asarray(x, dtype=float)
                return np.prod(x ** powers, axis=1) if powers is not None else x

            def mode_scores(x, c=self.classifier_):
                phi = features(x)
                if is_logistic:
                    d = phi @ c.coef_.T + c.intercept_
                    d = d[0] if d.shape[0] == 1 else d
                else:
                    d = c.decision_function(phi[None, :])[0]
                s = np.full(self.n, -np.inf)
                if np.ndim(d) == 0:  # binary: predict = classes_[1] iff d > 0
                    s[c.classes_[0]], s[c.classes_[1]] = -d / 2, d / 2
                else:
                    s[c.classes_] = d
                return s
            self.mode_scores_ = mode_scores
            # mode of a point: arg max of the scores (ties go to the lowest mode, as in predict)
            self.labelfun_ = lambda x: int(np.argmax(mode_scores(x)))
        else:
            self.mode_scores_ = None

    def add_classification_points(self, X, modes, classifier=None):
        """Adds points with known modes to the classification data and retrains the classifier."""
        self.check_if_fitted()
        if len(X) == 0:
            return
        X_dot = np.array([self.Omegas_[j] @ (self.poly.transform(x[None])[0] if self.deg > 1 else x) + self.betas_[j]
                          for x, j in zip(X, modes)])
        if hasattr(self, "X_classify_"):
            X = np.r_[self.X_classify_, X]
            X_dot = np.r_[self.X_dot_classify_, X_dot]
            modes = np.r_[self.modes_classify_, modes]
        self.classify_modes(classifier=classifier, X=X, modes=modes, X_dot=X_dot)

    def improve_classification(self, X, X_dot, classifier=None, partial_fit=False):
        '''
        Improve mode region finding by finding modes of given NEW data, augmenting it to previous classification data, and re-running the classification algorithm.

        Attributes
        ----------
        X : ndarray (N x D), default=None
            A matrix of NEW training data points [x1, x2, ..., xN].
       
        X_dot : ndarray (N x D), default=None
            A matrix of corresponding estimated derivatives of training data points [x1dot, x2dot, ..., xNdot]
        
        classifier : ?, default=None
            The classifier model to be used. Default is None, which indicates using the same classifier as in the previous run of `classify_modes`.
        
        partial_fit : Bool, default=False
            Indicates whether to use `partial_fit` when running the classifier, otherwise runs `fit` (default).
            Could be useful for incremental classifiers.
        '''

        self.check_if_fitted()
        if X.size == 0:
            return
        X_dot_pred_diff = np.ndarray((self.n, X_dot.shape[0], X_dot.shape[1]))
        for j in range(self.n):
            X_dot_pred_diff[j] = np.abs(self.predict(X, lambda x : j) - X_dot)
        
        X_dot_pred_diff = np.sum(X_dot_pred_diff, axis=-1)
        modes = np.argmin(X_dot_pred_diff, axis=0)
        if hasattr(self, "X_classify_"):
            X = np.r_[self.X_classify_, X]
            X_dot = np.r_[self.X_dot_classify_, X_dot]
            modes = np.r_[self.modes_classify_, modes]
        self.classify_modes(classifier = classifier, X = X, modes = modes, X_dot = X_dot, partial_fit=partial_fit)
    
    def classifier_boundary(self, x, classifier, print_on=False):
        self.check_if_fitted()
        decs = classifier.decision_function(x)[0]
        num_modes = len(np.unique(self.modes_classify_))
        if num_modes > 2:
            i1, i2 = 0, 0
            max1, max2 = decs[0], decs[0]

            i1 = np.argmax(decs)
            max1 = decs[i1]

            i2 = np.argmin(decs)
            max2 = decs[i2]
            for i in range(len(decs)):
                if decs[i] >= max2 and i != i1:
                    max2 = decs[i]
                    i2 = i

            if i1 < i2:
                e = decs[i1] - decs[i2]
            else:
                e = decs[i2] - decs[i1]
        elif np.ndim(decs) > 0:
            # two modes, one score per mode (decision list, one-versus-one): positive on the side of the second
            e = decs[1] - decs[0]
        else:
            e = decs

        if print_on:
            _log.debug(decs, max1, max2)
        return e
