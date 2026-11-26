"""The learner: HybridRegressor, whose methods are distributed over the files of this package (moves only)."""
from .base import _Base
from .opti import _OptiMixin
from .incremental import _IncrementalMixin
from .regions import _RegionsMixin
from .switches import _SwitchMixin


class HybridRegressor(_Base, _OptiMixin, _IncrementalMixin, _RegionsMixin, _SwitchMixin):
    '''
    Main learning class supporting both learning algorithms.

    (The kernel formulation of the optimization-based method was removed; see the tag thesis-reproducible.)

    Attributes
    ----------
    n : int, default=None
        The number of modes in the system-under-learning.
        Required for the optimization-based approach.
        Has no effect for the incremental learning approach.

    deg : int, default=1
        The maximum polynomial degree of the continuous dynamics.

    classifier_poly_deg : int, default=1
        The maximum polynomial degree of the mode region boundaries.


    Methods
    -------
    fit_opti(X, X_dot, C=100, deltas=0.01, segments=[], print_on=False, apply_LR=False, ...)
        Fit using the optimization-based method.

    fit_first_incremental(X, X_dot, segments, good_RD=None, regressor=None)
        Fit first hypothesis using the incremental learning method.

    update_continuous(X, X_dot, good_RD=None, regressor=None)
        Incremental method: update continuous dynamics of the hypothesis with new data.

    classify_modes(classifier=None, X=None, modes=None, X_dot=None, partial_fit=False)
        Run the classification algorithm based on the passed data and modes, or based on the fitted modes of the training data if X=None and modes=None.

    improve_classification(X, X_dot, classifier=None, partial_fit=False)
        Improve the result of the classifier by guessing the modes of new data X, Xdot where mode(x) is guessed as arg min |f_j(x) - xdot|.
        Runs `classify_mode` with the augmented data.

    compare_with_dynamics(X, Xdot, tol=0.01)
        Check whether data of a segment (X, Xdot) is already captured by a mode.

    get_points_of_mode(j)
        Returns points of mode j.

    predict(X, labelfun=None)
        Compute the predicted derivatives of data in `X`.
        Modes are deduced by labelfun in priority, or the learner's classifier if labelfun=None.

    get_modes(self)
        Returns the modes of the fitted data in the optimization-based approach.

    to_SNDS(labelfun=None, eventlist=None)
        Transforms the learned model into an executable SNDS.

    gen_full_latex_eq(with_classifier_boundaries=True, round_dig=3)
        Generate full LaTeX equations of the learned model.
    '''
