import casadi
import numpy as np

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

# We will assume the kernel matrices are passed as arguments in a list
def optimize(kernel_mat_list, x_dot, deltas, print_on=False):
    n = len(kernel_mat_list)

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
    betas = np.ndarray((n, D), dtype=object)
    Chis = np.ndarray((n,), dtype=object)
    
    for j in range(n):

        Chis[j] = opti.variable(N)

        for d in range(D):
            Alphas[j][d] = opti.variable(N)
            betas[j][d] = opti.variable()

    C = opti.parameter()
    opti.set_value(C, 0.1)
    sum_products_chis = 0
    for i in range(N):
        temp = 1
        for j in range(n):
            temp *= Chis[j][i]
        sum_products_chis += temp
    
    sum_products_chis *= C

    sum_norms_alpha = 0
    for j in range(n):
        for d in range(D):
            sum_norms_alpha += (Alphas[j][d].T @ Alphas[j][d])
    
    opti.minimize(sum_norms_alpha + sum_products_chis)
    # opti.minimize(sum_norms_alpha)
    # opti.minimize(sum_products_chis)

    for j in range(n):
        opti.subject_to(Chis[j] >= 0)

    x_dotT = x_dot.T    # d by N
    eye = np.ones(N)
    for d in range(D):
        for j in range(n):
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye <=  deltas[j]*eye + Chis[j])
            opti.subject_to(x_dotT[d] - casadi.vec(kernel_mat_list[j] @ Alphas[j][d]) - betas[j][d]*eye >= -deltas[j]*eye - Chis[j])

    opti.solver('ipopt')

    #print(opti.debug.x_describe(7))
    #print(opti.debug.g_describe(3))
    sol = opti.solve()

    if print_on:
        for j in range(n):
            print("Chi_" + str(j) + ": " + str(sol.value(Chis[j])))
        for j in range(n):
            for d in range(D):
                print("Alpha_" + str(j) + "_" + str(d) + ": " + str(sol.value(Alphas[j][d])))
                print( "Beta_" + str(j) + "_" + str(d) + ": " + str(sol.value(betas[j][d])))

    return sol
