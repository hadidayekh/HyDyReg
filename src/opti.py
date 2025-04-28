from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import random
from time import sleep
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import casadi
import numpy as np
import kernels
from snds import SNDS, SNDS_affine
from utils import *
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.base import clone
kernel_dic = {'linear' : kernels.linear, 
              'polynomial' : kernels.polynomial,
              'rbf' : kernels.rbf }

"""
Some notation:

K = kernel matrix = K(X, X) where each entry (i,j) is k(x_i, x_j)
K_j = kernel matrix j = K_j(X, X)
alpha_j is a Nx1 vector -> Alpha is the matrix
beta_j is a scalar -> beta is the vector
delta_j is a scalar, it's the error -> beta is the vector
xi_j is a Nx1 vector -> Xi is the matrix

x_dot is a NxD matrix
x_dot.T is DxN
"""

class HybridRegressor:
    '''
    Main learning class supporting both learning algorithms.

    NOTE: OPTIMIZATION-BASED LEARNING WITH KERNEL METHODS IS NO LONGER SUPPORTED. BASE CODE IS KEPT IN CASE FUTURE COLLABORATORS WOULD LIKE TO REBUILD IT.

    Attributes
    ----------
    train_coefs : bool, default=True
        When True, implements the optimization-based learning by finding coefficients of target functions directly rather than kernel methods.
        When False, uses kernel methods for the optimization-based learning <- NO LONGER SUPPORTED, see note above.
        Has no effect when using the incremental learning approach.

    n : int, default=None
        The number of modes in the system-under-learning.
        Required for the optimization-based approach.
        Has no effect for the incremental learning approach.

    deg : int, default=1
        The maximum polynomial degree of the continuous dynamics.

    classifier_poly_deg : int, default=1
        The maximum polynomial degree of the mode region boundaries.

    kernels_list : List['linear', 'polynomial', 'rbf', func], default=None
        NO LONGER SUPPORTED, see note above.
        The list of kernels to use for each mode in the optimization-based methods.

    kargs_list : List[List], default=None
        NO LONGER SUPPORTED, see note above.
        The list of arguments for each of the kernels in kernels_list.


    Methods
    -------
    fit_opti(X, X_dot, C=100, deltas=0.01, norm='L2', segmentation=None, segments=[], print_on=False, apply_LR=False)
        Fit using the optimization-based method.

    fit_first_incremental(X, X_dot, segments, good_RD=0.001, regressor=LinearRegression(), Time=None, shuffle_segments=False)
        Fit first hypothesis using the incremental learning method.

    update_continuous(X, X_dot, good_RD=0.001, regressor=LinearRegression(), Time = None)
        Incremental method: update continuous dynamics of the hypothesis with new data.

    classify_modes(classifier=LogisticRegression(), X=None, modes=None, X_dot=None, partial_fit=False)
        Run the classification algorithm based on the passed data and modes, or based on the fitted modes of the training data if X=None and modes=None.

    improve_classification(X, X_dot, classifier=None, partial_fit=False)
        Improve the result of the classifier by guessing the modes of new data X, Xdot where mode(x) is guessed as arg min |f_j(x) - xdot|.
        Runs `classify_mode` with the augmented data.

    compare_with_dynamics(X, Xdot, tol=0.01)
        Check whether data of a segment (X, Xdot) is already captured by a mode.

    get_points_of_mode(j)
        Returns points of mode j.

    reassign_modes()
        Reassign modes of training data based on the functions that fit each point best.

    retrain_continuous(j, regressor=None)
        Refit the continuous dynamics of mode `j` with data of mode `j`.

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
    def __init__(self, train_coefs=True, n=None, deg=1, classifier_poly_deg=1, kernels_list=None, kargs_list=None):
        if not train_coefs:
            raise ValueError("Kernel-based methods is currently not supported. Base code is untouched to allow future collaborators to build on it. Please set train_coefs to True.")
        if train_coefs and not (kernels_list is None and kargs_list is None):
            warnings.warn("Kernel information is provided but will not be used (train_coefs = True).")

        self.train_coefs = train_coefs
        self.n = n
        self.deg = deg

        self.classifier_poly_deg = classifier_poly_deg
        if not train_coefs and (kernels_list is None or kargs_list is None):
            raise Exception("When train_coefs is False, kernels_list and kargs_list must be provided.")
        if not train_coefs:
            self.kernels_list = kernels_list
            self.kargs_list = kargs_list
            assert len(kernels_list) == n
            assert len(kernels_list) == len(kargs_list)

    # For segmentation, will introduce variable segmentation which can belong to
    # {'squared_diff', 'strict'} or be None
    def fit_opti(self, X, X_dot, C=100, deltas=0.01, norm='L2', segmentation='strict', segments=[], print_on=False, apply_LR=False):
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

        norm : 'L1' or 'L2', default='L2'
            The norm used for regularization.
            Note: 'L1' norm is not supported (may or may not work).

        segmentation : None or 'strict' or 'squared_diff', default='strict'
            The type of integration of segmentation information.
            None means no segmentation information is integrated.
            'strict' means segmentation will be strictly integrated, and Xi variables per mode per point will be replaced by Xi variables per mode per segment. This is the method illustrated in the thesis.
            'squared_diff' means segmentation will be added by adding the squared difference of Xi variables of points belonging to the same segment to the minimization problem. This is not supported by expriments.

        segments : ndarray of N ints, default=[]
            The segment MAPPING of each data point to its segment.
            segments[i] indicates the segment identifier of point X[i].
            This is DIFFERENT from `segments` in `fit_first_opti` which is a LIST OF CHANGE POINT INDICES.

        print_on : Bool, default=False
            The verbosity parameter of the Casadi optimizer.

        apply_LR : Bool, default=False
            Indicates whether to apply linear/polynomial regression on data after finding their modes.
            This replaces the continuous dynamics found by the optimizer.
        '''


        if self.n is None:
            raise ValueError("To use the optimization based method, the number of modes n must be provided.")
        if not segmentation is None and segmentation not in ['squared_diff', 'strict']:
            raise ValueError("The parameter segmentation must be one of {None, 'squared_diff', 'strict'}")
        if not segmentation is None and len(segments) == 0:
            raise ValueError("The parameter segments must be provided or non-empty when a segmentation integration method is selected.")

        tc = self.train_coefs
        n = self.n
        deg = self.deg
        if not tc:
            kernel_mat_list = []
            for j in range(n):
                if isinstance(self.kernels_list[j], str):
                    kernel = kernel_dic[self.kernels_list[j]]
                else:
                    kernel = self.kernels_list[j]
                
                kernel_mat_list.append(kernel(X, X, self.kargs_list[j]))

        if tc:
            poly = PolynomialFeatures(degree=deg, include_bias=False)
            X_trans = poly.fit_transform(X)
            self.poly = poly
        else:
            X_trans = X

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
        if tc:
            Omegas = np.ndarray((n, D), dtype=object)
        else:
            Alphas = np.ndarray((n, D), dtype=object)
        betas = np.ndarray((n, D), dtype=object)
        Xis = np.ndarray((n,), dtype=object)

        if norm == 'L1':
            As = np.ndarray((n, D), dtype=object)
        
        for j in range(n):
            if segmentation == 'strict':
                xi_size_per_mode = S
            else:
                xi_size_per_mode = N
            
            Xis[j] = opti.variable(xi_size_per_mode)
            opti.set_initial(Xis[j], np.random.rand(xi_size_per_mode))

            for d in range(D):
                if tc:
                    Omegas[j][d] = opti.variable(Q)
                    opti.set_initial(Omegas[j][d], np.random.rand(Q))
                else:
                    Alphas[j][d] = opti.variable(N)
                    opti.set_initial(Alphas[j][d], np.random.rand(N))

                betas[j][d] = opti.variable()
                opti.set_initial(betas[j][d], np.random.rand())
                #betas[j][d] = 0
                if norm == 'L1':
                    if not tc:
                        As[j][d] = opti.variable(N)
                        opti.set_initial(As[j][d], np.random.rand(N))
                    else:
                        As[j][d] = opti.variable(Q)
                        opti.set_initial(As[j][d], np.random.rand(Q))

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

        #sum_products_xis *= C_param

        # Second part of the optimization objective function:
        # The sum of norms of alphas/Omegas
        # Here we use the L1 norm
        if norm == 'L1':
            if not tc:
                eye = np.ones(N)
            else:
                eye = np.ones(Q)
            sum_norms_a = 0
            for j in range(n):
                for d in range(D):
                    sum_norms_a += (As[j][d].T @ eye)
            to_minimize = sum_norms_a + sum_products_xis
        elif norm == 'L2':
            sum_norms = 0
            for j in range(n):
                for d in range(D):
                    if tc:
                        sum_norms += (Omegas[j][d].T @ Omegas[j][d])
                    else:
                        sum_norms += (Alphas[j][d].T @ Alphas[j][d])
            to_minimize = C_param*sum_norms + sum_products_xis
        else:
            raise Exception("The norm parameter has to be 'L1' or 'L2'.")
        

        if segmentation == 'squared_diff':
            sum_diff_xis = 0
            for r in range(S):
                seg_indices = np.where(segments==r)
                for i in range(seg_indices.shape[0] - 1):
                    for j in range(n):
                        sum_diff_xis += (Xis[j][seg_indices[i]] - Xis[j][seg_indices[i+1]])**2
            # for r in range(len(segments)-1):
            #     for i in range(segments[r], segments[r+1] - 1):
            #         for j in range(n):
            #             sum_diff_xis += (Xis[j][i] - Xis[j][i+1])**2
            # for i in range(segments[-1], N-1):
            #     for j in range(n):
            #         sum_diff_xis += (Xis[j][i] - Xis[j][i+1])**2
            
            to_minimize += C*sum_diff_xis

        # Setting the objective function
        opti.minimize(to_minimize)

        # CONSTRAINTS

        # all xis must be positive
        for j in range(n):
            opti.subject_to(Xis[j] >= 0)
        
        x_dotT = X_dot.T    # d by N
        eye = np.ones(N)
        for d in range(D):
            for j in range(n):
                if segmentation == 'strict':
                    # # Here we need to transform each Xi[j], which is currently an S by 1 matrix into an N by 1 matrix
                    # # We do this by constructing a transformation matrix that assigns the correct xi according to which segment the point is in
                    # trans_mat = np.zeros((N, S))
                    # for i in range(N):
                    #     seg_index = which_segment(i, segments)
                    #     trans_mat[i][seg_index] = 1
                    # Xis[j] = casadi.vec(trans_mat @ Xis[j])
                    for i in range(N):
                        seg_index = segments[i]
                        if not tc:
                            opti.subject_to(x_dotT[d][i] - casadi.dot(kernel_mat_list[j][i].T, Alphas[j][d]) - betas[j][d] <=  deltas[j] + Xis[j][seg_index])
                            opti.subject_to(x_dotT[d][i] - casadi.dot(kernel_mat_list[j][i].T, Alphas[j][d]) - betas[j][d] >= -deltas[j] - Xis[j][seg_index])
                        else:
                            opti.subject_to(x_dotT[d][i] - casadi.dot(X_trans[i].T, Omegas[j][d]) - betas[j][d] <=  deltas[j] + Xis[j][seg_index])
                            opti.subject_to(x_dotT[d][i] - casadi.dot(X_trans[i].T, Omegas[j][d]) - betas[j][d] >= -deltas[j] - Xis[j][seg_index])
                else:
                    # the difference between prediction and the ground truth is bounded in [-delta-xi, delta+xi]
                    if not tc:
                        opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Xis[j])
                        opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Xis[j])
                    else:
                        opti.subject_to(x_dotT[d] - casadi.vec(X_trans @ Omegas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Xis[j])
                        opti.subject_to(x_dotT[d] - casadi.vec(X_trans @ Omegas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Xis[j])
                if norm=='L1':
                    if not tc:
                        # As are bounded by [-alpha, +alpha]
                        opti.subject_to(-Alphas[j][d] <= As[j][d])
                        opti.subject_to(As[j][d] <= Alphas[j][d])
                    else:
                        # As are bounded by [-omega, +omega]
                        opti.subject_to(-Omegas[j][d] <= As[j][d])
                        opti.subject_to(As[j][d] <= Omegas[j][d])

        opti.solver('ipopt', dict(print_time = 0), dict(max_iter = 3000, print_level = 0))
        # Initiate the optimization
        sol = opti.solve()

        # Printing the results if print is on
        if print_on:
            for j in range(n):
                print("Xi_" + str(j) + ": " + str(sol.value(Xis[j])))
            for j in range(n):
                for d in range(D):
                    if not tc:
                        print("Alpha_" + str(j) + "_" + str(d) + ": " + str(sol.value(Alphas[j][d])))
                    else:
                        print("Omega_" + str(j) + "_" + str(d) + ": " + str(sol.value(Omegas[j][d])))
                    print( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

        # Saving the results in proper arrays
        if not tc:
            Alphas_sol = np.ndarray((n, D, N))
        else:
            Omegas_sol = np.ndarray((n, D, Q))
        betas_sol  = np.ndarray((n, D))
        Xis_sol   = np.ndarray((n, xi_size_per_mode))

        for j in range(n):
                Xis_sol[j] = sol.value(Xis[j])
        for j in range(n):
            for d in range(D):
                if not tc:
                    Alphas_sol[j][d] = sol.value(Alphas[j][d])
                else:
                    Omegas_sol[j][d] = sol.value(Omegas[j][d])
                betas_sol[j][d]  = sol.value(betas[j][d])
        
        if segmentation == 'strict':
            self.Xis_segments_ = Xis_sol
            final_Xis = np.ndarray((n, N))
            for j in range(n):
                for i in range(N):
                    seg_index = segments[i]
                    final_Xis[j][i] = Xis_sol[j][seg_index]

            self.Xis_ = final_Xis
        else: 
            self.Xis_ = Xis_sol

        if apply_LR and tc:
            modes = self.get_modes()
            LR = LinearRegression()
            for j in range(n):
                indices_j = np.where(modes == j)
                X_j = X_trans[indices_j]
                X_dot_j = X_dot[indices_j]
                LR.fit(X_j, X_dot_j)
                Omegas_sol[j] = LR.coef_
                betas_sol[j] = LR.intercept_




        #self.sol_ = sol
        if not tc:
            self.Alphas_ = Alphas_sol
        else:
            self.Omegas_ = Omegas_sol
        self.betas_ = betas_sol
        # self.Xis_ = Xis_sol
        self.Xfit_ = X
        self.Xfit_trans_ = X_trans
        self.X_dot_fit_ = X_dot

        self.modesfit_ = self.get_modes() # here we store the modes of each data point used for fitting

    def fit_first_incremental(self, X, X_dot, segments, good_RD=0.001, regressor=LinearRegression(), Time=None, shuffle_segments=False):
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

        good_RD : float, default=0.001
            This tol_RD in the theoretical documentation.
            Used as a tolerance for the average relative difference to indicate whether a mode captures the dynamics of a sequence of data points.

        regressor : ?, default=LinearRegression()
            The regressor model to be used. Default is linear regression (no regularization).
            Regularization is recommended for passive learning. This can be achieved by passing RidgeCV() for example.

        Time : ndarray (N), default=None
            The time data associated to each data point in X.
            Not required for the learning algorirthm itself, but is necessary if time information needs to be kept later on.
            This is used in ESNDS learning for DCDC Melco (where time information is necessary for the mode region finding).

        shuffle_segments : Bool, default=False
            Indicates whether to fit segments in order or shuffle them first.
            Note: When True, results are no longer guaranteed to be reproducible.
        '''

        self.n = 1

        if len(segments) > 1:
            segment_index_pairs = []
            for i in range(len(segments)-1):
                segment_index_pairs.append((segments[i], segments[i+1]))
            segment_index_pairs.append((segments[-1], X.shape[0]))
            if shuffle_segments:
                random.shuffle(segment_index_pairs)

            X_first = X[segment_index_pairs[0][0]:segment_index_pairs[0][1]]
            X_dot_first = X_dot[segment_index_pairs[0][0]:segment_index_pairs[0][1]]
            if not Time is None:
                Time_first = Time[segment_index_pairs[0][0]:segment_index_pairs[0][1]]
        else:
            X_first = X[segments[0]:]
            X_dot_first = X_dot[segments[0]:]
            if not Time is None:
                Time_first = Time[segments[0]:]
            

        print("first segment with",X_first.shape[0],"points")
        poly = PolynomialFeatures(degree=self.deg, include_bias=False)
        X_trans = poly.fit_transform(X_first)
        D = X_first.shape[1]
        Q = X_trans.shape[1]
        self.poly = poly
        LR = clone(regressor)
        LR.fit(X_trans, X_dot_first)
        self.Omegas_ = np.ndarray((1, D, Q))
        self.betas_ = np.ndarray((1, D))

        self.Omegas_[0] = LR.coef_
        self.betas_[0] = LR.intercept_
        self.regressors_ = [LR]

        self.Xfit_ = X_first
        self.Xfit_trans_ = X_trans
        self.X_dot_fit_ = X_dot_first
        if not Time is None:
            self.Time_fit_ = Time_first
        self.modesfit_ = np.array([0] * X_first.shape[0])

        if len(segments) > 1:
            for i in range(1, len(segment_index_pairs)):
                print("seg", i+1)
                X_curr = X[segment_index_pairs[i][0] : segment_index_pairs[i][1]]
                print(X_curr.shape[0], "points")
                X_dot_curr = X_dot[segment_index_pairs[i][0] : segment_index_pairs[i][1]]
                Time_curr = None if Time is None else Time[segment_index_pairs[i][0] : segment_index_pairs[i][1]]
                
                self.update_continuous(X_curr, X_dot_curr, good_RD=good_RD, regressor=regressor, Time=Time_curr)
            

            # for i in range(1, len(segments) - 1):
            #     print("seg", i+1)
            #     X_curr = X[segments[i]:segments[i+1]]
            #     print(X_curr.shape[0], "points")
            #     X_dot_curr = X_dot[segments[i]:segments[i+1]]
            #     self.update_continuous(X_curr, X_dot_curr, good_RD=good_RD, regressor=regressor)
            # print("last seg")
            # X_curr = X[segments[-1]:]
            # print(X_curr.shape[0], "points")
            # X_dot_curr = X_dot[segments[-1]:]
            # self.update_continuous(X_curr, X_dot_curr, good_RD=good_RD, regressor=regressor)


    
    def compare_with_dynamics(self, X, Xdot, tol=0.01):
        RDs = np.ndarray((self.n,))
        for j in range(self.n):
            Xdot_pred = self.predict(X, lambda x: j)
            RDs[j] = np.mean(relative_diff(Xdot, Xdot_pred))
            # print("mode", j, RDs[j])
            # print(Xdot - Xdot_pred)
            # print(Xdot, Xdot_pred)
            # RDs[j] = np.linalg.norm(Xdot - Xdot_pred, ord='fro')
        min_RD = np.min(RDs)
        # print(RDs)
        if min_RD < tol:
            print("No need to improve, mode", np.argmin(RDs), f"matches. RD = {min_RD:.3e}")
        else:
            print(f"Need to improve, min(RDs) = {min_RD:.3e}")
            # print(RDs)
        return min_RD < tol, np.argmin(RDs)

    def update_continuous(self, X, X_dot, good_RD=0.001, regressor=LinearRegression(), Time = None):
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

        good_RD : float, default=0.001
            This tol_RD in the theoretical documentation.
            Used as a tolerance for the average relative difference to indicate whether a mode captures the dynamics of a sequence of data points.

        regressor : ?, default=LinearRegression()
            The regressor model to be used. Default is linear regression (no regularization).
            Regularization is recommended for passive learning. This can be achieved by passing RidgeCV() for example.

        Time : ndarray (N), default=None
            The time data associated to each data point in X.
            Not required for the learning algorirthm itself, but is necessary if time information needs to be kept later on.
            This is used in ESNDS learning for DCDC Melco (where time information is necessary for the mode region finding).
        '''


        if not self.train_coefs:
            raise NotImplementedError
        # X and X dot belong to one segment
        N_new = X.shape[0]
        if self.deg > 1:
            X_trans = self.poly.fit_transform(X)
        else:
            X_trans = X
        
        is_already_captured, j_star = self.compare_with_dynamics(X, X_dot, tol=good_RD)
        if is_already_captured:
            mode = j_star
        else:
            LRs = []
            RDs = np.ndarray((self.n,))
            for j in range(self.n):
                # LR = LinearRegression()
                LR = clone(regressor)
                Xfit_trans_j, X_dot_j = self.get_points_of_mode(j)
                aug_X = np.r_[Xfit_trans_j, X_trans]
                aug_X_dot = np.r_[X_dot_j, X_dot]

                LR.fit(aug_X, aug_X_dot)
                
                # Calculating RD for segment
                X_dot_pred = LR.predict(X_trans)
                RD_seg = np.mean(relative_diff(X_dot, X_dot_pred))
                # RD_seg = np.linalg.norm(X_dot - X_dot_pred, ord='fro')
                # Calculating RD for rest of the data
                X_dot_pred_j = LR.predict(Xfit_trans_j)
                RD_j = np.mean(relative_diff(X_dot_j, X_dot_pred_j))
                # RD_j = np.linalg.norm(X_dot_j - X_dot_pred_j, ord='fro')

                LRs.append(LR)
                RDs[j] = np.max([RD_seg, RD_j])

            RD = np.min(RDs)
            if RD < good_RD:
                mode = np.argmin(RDs)
                print("Merging with mode", mode, "RD is", RD)
                LR = LRs[mode]
                self.Omegas_[mode] = LR.coef_
                self.betas_[mode] = LR.intercept_
                self.regressors_[mode] = LR

            else:
                mode = self.n
                print(f"Creating new mode {mode}; RD was not good: {RD:.3f}")
                print(RDs)
                # input("Press enter to continue...")
                self.n = self.n + 1
                # LR = LinearRegression()
                LR = clone(regressor)
                LR.fit(X_trans, X_dot)
                self.Omegas_ = np.r_[self.Omegas_, LR.coef_.reshape((1, LR.coef_.shape[0], LR.coef_.shape[1]))]
                self.betas_ = np.r_[self.betas_, LR.intercept_.reshape((1, LR.intercept_.shape[0]))]
                self.regressors_.append(LR)
                # self.reassign_modes()
        
        self.Xfit_ = np.r_[self.Xfit_, X]
        self.Xfit_trans_ = np.r_[self.Xfit_trans_, X_trans]
        self.X_dot_fit_ = np.r_[self.X_dot_fit_, X_dot]
        if not Time is None:
            self.Time_fit_ = np.r_[self.Time_fit_, Time]

        modes_new = np.array([mode] * N_new)
        self.modesfit_ = np.r_[self.modesfit_, modes_new]
        # print(self.gen_full_latex_eq(with_classifier_boundaries=False))
        # self.reassign_modes()


    def get_points_of_mode(self, j):
        self.check_if_fitted()
        assert j < self.n

        indices = np.where(self.modesfit_ == j)
        return self.Xfit_trans_[indices], self.X_dot_fit_[indices]

    def reassign_modes(self):
        # X = self.X_classify_
        # X_dot = self.X_dot_classify_
        # N = X.shape[0]
        # D = X.shape[1]
        # X_dot_pred_diff = np.ndarray((self.n, N, D))
        # for j in range(self.n):
        #     X_dot_pred_diff[j] = np.abs(self.predict(X, lambda x : j) - X_dot)
        # X_dot_pred_diff = np.sum(X_dot_pred_diff, axis=-1)
        # self.modes_classify_ = np.argmin(X_dot_pred_diff, axis=0)

        X = self.Xfit_
        X_dot = self.X_dot_fit_
        N = X.shape[0]
        D = X.shape[1]
        X_dot_pred_diff = np.ndarray((self.n, N, D))
        for j in range(self. n):
            X_dot_pred_diff[j] = np.abs(self.predict(X, lambda x : j) - X_dot)
        X_dot_pred_diff = np.sum(X_dot_pred_diff, axis=-1)
        new_modesfit_ = np.argmin(X_dot_pred_diff, axis=0)
        new_modesfit_old = new_modesfit_[:len(self.modesfit_)]
        diff = new_modesfit_old - self.modesfit_
        changed_modes = np.unique(np.r_[self.modesfit_[diff != 0], new_modesfit_old[diff != 0]])
        print(changed_modes)
        sleep(1)
        self.modesfit_ = new_modesfit_
        for j in changed_modes:
            self.retrain_continuous(j)
            

    def classify_modes(self, classifier=LogisticRegression(), X=None, modes=None, X_dot=None, partial_fit=False):
        self.check_if_fitted()
        if X is None:
            if hasattr(self, 'X_classify_'):
                X = np.r_[self.Xfit_, self.X_classify_]
                X_dot = np.r_[self.X_dot_fit_, self.X_dot_classify_]
                modes = np.r_[self.modesfit_, self.modes_classify_]
                _, unique_indices = np.unique(X, axis=0, return_index=True)
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
        else:
            print(modes[0])
            self.labelfun_ = lambda x : modes[0]
            self.eventlist_ = None
            self.classifier_ = classifier
        self.X_classify_ = X
        self.X_dot_classify_ = X_dot
        self.modes_classify_ = modes

        modes_predict = classifier.predict(X_class_trans)
        from sklearn.metrics import accuracy_score
        print("Classification training accuracy:", accuracy_score(modes, modes_predict))

    def retrain_continuous(self, j, regressor=None):
        X, Xdot = self.get_points_of_mode(j)
        if X.shape[0] == 0:
            return
        if regressor is None:
            regressor = self.regressors_[j]
        regressor.fit(X, Xdot)
        self.Omegas_[j] = regressor.coef_
        self.betas_[j] = regressor.intercept_
    
    def improve_classification(self, X, X_dot, classifier=None, partial_fit=False):
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
    
    def predict(self, X, labelfun=None):
        # FIRST WE WILL ASSUME THAT THERE IS A GIVEN LABEL FUNCTION
        # Things to check:
        # 1) Regressor has already been fit (can do that by vars(self) and _ elements)
        # 2) X has correct dimensions (X.shape[1] == Xfit_.shape[1])

        self.check_if_fitted()
        tc = self.train_coefs

        if labelfun is None:
            labelfun = self.labelfun_
        
        labellist = False
        if isinstance(labelfun, list) or isinstance(labelfun, np.ndarray):
            labellist = True
            if len(labelfun) != X.shape[0]:
                raise Exception("When labelfun is a list of labels, its length must be equal to the number of datapoints in X.")

        if not tc:
            kernels_list = self.kernels_list
            kargs_list = self.kargs_list
            Alphas = self.Alphas_
        else:
            Omegas = self.Omegas_
        betas = self.betas_
        Xfit = self.Xfit_
        Xfit_trans = self.Xfit_trans_

        # Dimensions of X
        M = X.shape[0]
        D = X.shape[1]
        Q = Xfit_trans.shape[1]

        # Assert that X has correct dimensions
        if D != Q and ((tc and self.deg == 1) or not tc):
            raise Exception("Given X does not have the right dimensions.")
        if tc and self.deg > 1:
            poly = PolynomialFeatures(degree = self.deg, include_bias=False)
            try:
                X_trans = poly.fit_transform(X)
            except:
                print("PROBLEM", X)
                exit(-1)
        else:
            X_trans = X

        X_dot_res = np.zeros((M,D))

        # For each data point
        for i in range(M):
            # Find the mode of the data point
            if labellist:
                j = labelfun[i]
            else:
                j = labelfun(X[i])
            if not tc:
                # Calculate based on the correct kernel
                if isinstance(kernels_list[j], str):
                    kernel = kernel_dic[kernels_list[j]]
                else:
                    kernel = kernels_list[j]
                
                K_X_x = kernel(Xfit, X[i], kargs_list[j])
                for d in range(D):
                    X_dot_res[i][d] = Alphas[j][d].T @ K_X_x + betas[j][d]
            else:
                X_dot_res[i] = (Omegas[j] @ X_trans[i] + betas[j]).reshape(-1)

        return X_dot_res

    def fit_predict(self, X, X_dot, C=100, deltas=0.01, norm='L2', segments=[], print_on=False, labelfun=None):
        self.fit_opti(X, X_dot, C=C, deltas=deltas, norm=norm, segments=segments, print_on=print_on)
        return self.predict(X, labelfun=labelfun)

    def get_modes(self):
        self.check_if_fitted()
        return np.argmin(self.Xis_, axis=0)
    
    def to_SNDS(self, labelfun=None, eventlist=None):
        self.check_if_fitted()
        if labelfun is None:
            labelfun = self.labelfun_
        if eventlist is None:
            eventlist = self.eventlist_

        n = self.n
        D = self.Xfit_.shape[1]

        if not self.train_coefs:
            eqs = []
            for j in range(n):
                eqs.append(self.make_eq(j))

            return SNDS(n, D, eqs, eventlist, labelfun)
        elif self.deg == 1:
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

        
    def make_eq(self, j):
        def eq(t, x):
            if isinstance(self.kernels_list[j], str):
                kernel = kernel_dic[self.kernels_list[j]]
            else:
                kernel = self.kernels_list[j]
            k_x_x = kernel(self.Xfit_, x, self.kargs_list[j])
            D = self.Xfit_.shape[1]
            res = np.zeros(D)
            for d in range(D):
                res[d] = self.Alphas_[j][d].T @ k_x_x + self.betas_[j][d]
            return res
        return eq

    def make_eq_predict(self, j):
        def eq(t, x):
            return self.predict(np.array([x]), lambda x : j)[0]
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

    def print_coefs_linear(self):
        self.check_if_fitted()
        if not self.train_coefs:
            n, D, N = self.Alphas_.shape
            for j in range(n):
                print("Mode " + str(j) + ":")
                for d in range(D):
                    print("x" + str(d) + "_dot:")
                    coefs = self.Xfit_.T @ self.Alphas_[j][d]
                    print(coefs.T, " ", self.betas_[j][d])
                print("\n")
        else:
            n, D = self.betas_.shape
            if self.deg > 1:
                print(self.poly.get_feature_names_out())
                print(self.poly.powers_)
            for j in range(n):
                print("Mode " + str(j) + ":")
                for d in range(D):
                    print("x" + str(d) + "_dot:")
                    coefs = self.Omegas_[j][d]
                    print(coefs.T, " ", self.betas_[j][d])

    def gen_full_latex_eq(self, with_classifier_boundaries=True, round_dig=3):
        self.check_if_fitted()
        if not self.train_coefs:
            raise NotImplementedError
        
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
        else:
            e = decs

        if print_on:
            print(decs, max1, max2)
        return e