from asyncio.proactor_events import _ProactorBaseWritePipeTransport
import casadi
import numpy as np
import kernels
from snds import SNDS
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
chi_j is a Nx1 vector -> Chi is the matrix

x_dot is a NxD matrix
x_dot.T is DxN
"""

# Optimization for identifying Hybrid Systems by minimizing the L2 norm
# We will assume the kernel matrices are passed as arguments in a list
def optimizeL2(kernel_mat_list, x_dot, deltas, C_param, print_on=False):
    # the number of kernel matrices is the number of modes
    n = len(kernel_mat_list)

    # deltas could be a scalar or defined per mode
    if (np.isscalar(deltas)):
        temp = np.ones(n)
        deltas = deltas*temp
    else:
        assert(len(deltas) == n)

    N = x_dot.shape[0]  # nb of points
    D = x_dot.shape[1]  # dimension of each vector
    #assert(len(mode_guesses) == N)
    opti = casadi.Opti()

    # variable declaration arrays
    Alphas = np.ndarray((n, D), dtype=object)
    betas = np.ndarray((n, D), dtype=object)
    Chis = np.ndarray((n,), dtype=object)
    
    for j in range(n):

        Chis[j] = opti.variable(N)
        opti.set_initial(Chis[j], np.ones(N)*100)

        for d in range(D):
            Alphas[j][d] = opti.variable(N)
            betas[j][d] = opti.variable()
    
    #for i in range(0, int(N), 2):
    #    opti.set_initial(Chis[int(mode_guesses[i])][i], 0)

    C = opti.parameter()
    opti.set_value(C, C_param)

    # First part of the optimization objective function:
    # the sum of products of Chis
    # This forces one Chi per data point to be ~0, which dictates the mode

    sum_products_chis = 0
    for i in range(N):
        temp = 1
        for j in range(n):
            temp *= Chis[j][i]
        sum_products_chis += temp
    
    sum_products_chis *= C

    # Second part of the optimization objective function:
    # The sum of norms of alphas
    # Here we use the L2 norm
    sum_norms_alpha = 0
    for j in range(n):
        for d in range(D):
            sum_norms_alpha += (Alphas[j][d].T @ Alphas[j][d])
    
    # Setting the objective function
    opti.minimize(sum_norms_alpha + sum_products_chis)

    # CONSTRAINTS

    # all chis must be positive
    for j in range(n):
        opti.subject_to(Chis[j] >= 0)

    x_dotT = x_dot.T    # d by N
    eye = np.ones(N)
    for d in range(D):
        for j in range(n):
            # the difference between prediction and the ground truth is bounded in [-delta-chi, delta+chi]
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Chis[j])
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Chis[j])

    opti.solver('ipopt')

    # Initiate the optimization
    sol = opti.solve()

    # Printing the results if print is on
    if print_on:
        for j in range(n):
            print("Chi_" + str(j) + ": " + str(sol.value(Chis[j])))
        for j in range(n):
            for d in range(D):
                print("Alpha_" + str(j) + "_" + str(d) + ": " + str(sol.value(Alphas[j][d])))
                print( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

    # Saving the results in proper arrays
    Alphas_sol = np.ndarray((n, D, N))
    betas_sol  = np.ndarray((n, D))
    Chis_sol   = np.ndarray((n, N))

    for j in range(n):
            Chis_sol[j] = sol.value(Chis[j])
    for j in range(n):
        for d in range(D):
            Alphas_sol[j][d] = sol.value(Alphas[j][d])
            betas_sol[j][d]  = sol.value(betas[j][d])
    
    
    return sol, Alphas_sol, betas_sol, Chis_sol


# Optimization for identifying Hybrid Systems by minimizing the L1 norm
# We will assume the kernel matrices are passed as arguments in a list
def optimizeL1(kernel_mat_list, x_dot, deltas, C_param, print_on=False):
    # the number of kernel matrices is the number of modes
    n = len(kernel_mat_list)

    # deltas could be a scalar or defined per mode
    if (np.isscalar(deltas)):
        temp = np.ones(n)
        deltas = deltas*temp
    else:
        assert(len(deltas) == n)

    N = x_dot.shape[0]  # nb of points
    D = x_dot.shape[1]  # dimension of each vector

    opti = casadi.Opti()

    # variable declaration arrays
    Alphas = np.ndarray((n, D), dtype=object)
    As = np.ndarray((n, D), dtype=object)
    betas = np.ndarray((n, D), dtype=object)
    Chis = np.ndarray((n,), dtype=object)
    
    for j in range(n):

        Chis[j] = opti.variable(N)

        for d in range(D):
            Alphas[j][d] = opti.variable(N)
            As[j][d] = opti.variable(N)
            betas[j][d] = opti.variable()

    C = opti.parameter()
    opti.set_value(C, C_param)

    # First part of the optimization objective function:
    # the sum of products of Chis
    # This forces one Chi per data point to be ~0, which dictates the mode

    sum_products_chis = 0
    for i in range(N):
        temp = 1
        for j in range(n):
            temp *= Chis[j][i]
        sum_products_chis += temp
    
    sum_products_chis *= C

    # Second part of the optimization objective function:
    # The sum of norms of alphas
    # Here we use the L1 norm
    eye = np.ones(N)
    sum_norms_a = 0
    for j in range(n):
        for d in range(D):
            sum_norms_a += (As[j][d].T @ eye)
    
    # Setting the objective function
    opti.minimize(sum_norms_a + sum_products_chis)

    # CONSTRAINTS

    # all chis must be positive
    for j in range(n):
        opti.subject_to(Chis[j] >= 0)

    x_dotT = x_dot.T    # d by N

    for d in range(D):
        for j in range(n):
            # the difference between prediction and the ground truth is bounded in [-delta-chi, delta+chi]
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Chis[j])
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Chis[j])

            # As are bounded by [-alpha, +alpha]
            opti.subject_to(-Alphas[j][d] <= As[j][d])
            opti.subject_to(As[j][d] <= Alphas[j][d])

    options = {"print_time": False, "ipopt": {"max_iter": 10000}}
    opti.solver('ipopt', options)
    

    # Initiate the optimization
    sol = opti.solve()

    # Printing the results if print is on
    if print_on:
        for j in range(n):
            print("Chi_" + str(j) + ": " + str(sol.value(Chis[j])))
        for j in range(n):
            for d in range(D):
                print("Alpha_" + str(j) + "_" + str(d) + ": " + str(sol.value(Alphas[j][d])))
                print( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

    # Saving the results in proper arrays
    Alphas_sol = np.ndarray((n, D, N))
    betas_sol  = np.ndarray((n, D))
    Chis_sol   = np.ndarray((n, N))

    for j in range(n):
            Chis_sol[j] = sol.value(Chis[j])
    for j in range(n):
        for d in range(D):
            Alphas_sol[j][d] = sol.value(Alphas[j][d])
            betas_sol[j][d]  = sol.value(betas[j][d])
    
    
    return sol, Alphas_sol, betas_sol, Chis_sol


class HybridKernelRegressor:
    def __init__(self, kernels_list, kargs_list):
        self.kernels_list = kernels_list
        self.kargs_list = kargs_list
        self.n = len(kernels_list)
        assert len(kernels_list) == len(kargs_list)

    def fit(self, X, X_dot, C=100, deltas=0.01, norm='L2', segments=[], print_on=False):
        self.X_ = X
        self.X_dot_ = X_dot
        kernel_mat_list = []
        for j in range(self.n):
            if isinstance(self.kernels_list[j], str):
                kernel = kernel_dic[self.kernels_list[j]]
            else:
                kernel = self.kernels_list[j]
            
            kernel_mat_list.append(kernel(X, X, self.kargs_list[j]))

        n = self.n
        
        # deltas could be a scalar or defined per mode
        if (np.isscalar(deltas)):
            temp = np.ones(n)
            deltas = deltas*temp
        else:
            assert(len(deltas) == n)

        N = X_dot.shape[0]  # nb of points
        D = X_dot.shape[1]  # dimension of each vector

        opti = casadi.Opti()

        # variable declaration arrays
        Alphas = np.ndarray((n, D), dtype=object)
        betas = np.ndarray((n, D), dtype=object)
        Chis = np.ndarray((n,), dtype=object)

        if norm == 'L1':
            As = np.ndarray((n, D), dtype=object)
        
        for j in range(n):

            Chis[j] = opti.variable(N)
            opti.set_initial(Chis[j], np.ones(N)*100)

            for d in range(D):
                Alphas[j][d] = opti.variable(N)
                betas[j][d] = opti.variable()
                if norm == 'L1':
                    As[j][d] = opti.variable(N)

        C_param = opti.parameter()
        opti.set_value(C_param, C)

        # First part of the optimization objective function:
        # the sum of products of Chis
        # This forces one Chi per data point to be ~0, which dictates the mode

        sum_products_chis = 0
        for i in range(N):
            temp = 1
            for j in range(n):
                temp *= Chis[j][i]
            sum_products_chis += temp
        
        sum_products_chis *= C_param

        # Second part of the optimization objective function:
        # The sum of norms of alphas
        # Here we use the L1 norm
        if norm == 'L1':
            eye = np.ones(N)
            sum_norms_a = 0
            for j in range(n):
                for d in range(D):
                    sum_norms_a += (As[j][d].T @ eye)
            to_minimize = sum_norms_a + sum_products_chis
        elif norm == 'L2':
            sum_norms_alpha = 0
            for j in range(n):
                for d in range(D):
                    sum_norms_alpha += (Alphas[j][d].T @ Alphas[j][d])
            to_minimize = sum_norms_alpha + sum_products_chis
        else:
            raise Exception("The norm parameter has to be 'L1' or 'L2'.")
        
        # Setting the objective function
        opti.minimize(to_minimize)

        # CONSTRAINTS

        # all chis must be positive
        for j in range(n):
            opti.subject_to(Chis[j] >= 0)

        x_dotT = X_dot.T    # d by N
        eye = np.ones(N)
        for d in range(D):
            for j in range(n):
                # the difference between prediction and the ground truth is bounded in [-delta-chi, delta+chi]
                opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Chis[j])
                opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Chis[j])

        opti.solver('ipopt')

        # Initiate the optimization
        sol = opti.solve()

        # Printing the results if print is on
        if print_on:
            for j in range(n):
                print("Chi_" + str(j) + ": " + str(sol.value(Chis[j])))
            for j in range(n):
                for d in range(D):
                    print("Alpha_" + str(j) + "_" + str(d) + ": " + str(sol.value(Alphas[j][d])))
                    print( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

        # Saving the results in proper arrays
        Alphas_sol = np.ndarray((n, D, N))
        betas_sol  = np.ndarray((n, D))
        Chis_sol   = np.ndarray((n, N))

        for j in range(n):
                Chis_sol[j] = sol.value(Chis[j])
        for j in range(n):
            for d in range(D):
                Alphas_sol[j][d] = sol.value(Alphas[j][d])
                betas_sol[j][d]  = sol.value(betas[j][d])
        
        self.sol_ = sol
        self.Alphas_ = Alphas_sol
        self.betas_ = betas_sol
        self.Chis_ = Chis_sol
        self.Xfit_ = X

    def predict(self, X, labelfun=None):
        # FIRST WE WILL ASSUME THAT THERE IS A GIVEN LABEL FUNCTION
        # Things to check:
        # 1) Regressor has already been fit (can do that by vars(self) and _ elements)
        # 2) X has correct dimensions (X.shape[1] == Xfit_.shape[1])

        self.check_if_fitted()

        if labelfun is None:
            labelfun = self.labelfun_

        kernels_list = self.kernels_list
        kargs_list = self.kargs_list
        Alphas = self.Alphas_
        betas = self.betas_
        Xfit = self.Xfit_

        # Dimensions of X
        M = X.shape[0]
        D = X.shape[1]

        # Assert that X has correct dimensions
        if D != Xfit.shape[1]:
            raise Exception("Given X does not have the right dimensions.")
        
        X_dot_res = np.zeros((M,D))

        # For each data point
        for i in range(M):
            # Find the mode of the data point
            j = labelfun(X[i])
            # Calculate based on the correct kernel
            if isinstance(kernels_list[j], str):
                kernel = kernel_dic[kernels_list[j]]
            else:
                kernel = kernels_list[j]
            
            K_X_x = kernel(Xfit, X[i], kargs_list[j])
            for d in range(D):
                X_dot_res[i][d] = Alphas[j][d].T @ K_X_x + betas[j][d]

        return X_dot_res

    def fit_predict(self, X, X_dot, C=100, deltas=0.01, norm='L2', segments=[], print_on=False, labelfun=None):
        self.fit(X, X_dot, C=C, deltas=deltas, norm=norm, segments=segments, print_on=print_on)
        if labelfun is None:
            labelfun = self.labelfun_
        return self.predict(X)

    def get_modes(self):
        self.check_if_fitted()
        return np.argmin(self.Chis_, axis=0)
    
    def to_SNDS(self, labelfun=None, eventlist=None):
        self.check_if_fitted()
        if labelfun is None:
            labelfun = self.labelfun_
        if eventlist is None:
            eventlist = self.eventlist_

        n = self.n
        d = self.Xfit_.shape[1]

        eqs = []
        for j in range(n):
            eqs.append(self.make_eq(j))

        return SNDS(n, d, eqs, eventlist, labelfun)

        
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

    def check_if_fitted(self):
        variables = vars(self)
        fitted = False
        for v in variables:
            if v.endswith('_'):
                fitted = True
        if not fitted:
            raise Exception("Estimator not fitted.")
