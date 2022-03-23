import numpy as np
import matplotlib.pyplot as plt
import kernels
from paper_utils import simulation_ode
from experiment3_poly_1_gen import fvdp2_1

X = np.genfromtxt("input/final_X.csv", delimiter=',')

Alphas = np.genfromtxt("output/Alphas0.csv", delimiter=',')
betas = np.genfromtxt("output/betas0.csv", delimiter=',')


poly_k = [kernels.polynomial, [2, 1, 1]]
kernel_list = [poly_k]

def get_derv(kernel_list, Alphas, betas, X):
    assert(len(kernel_list) == 1)
    N = X.shape[0]

    def mode0(t, x):
        kernel = kernel_list[0][0]
        kargs = kernel_list[0][1]
        

        x_dot0 = 0
        x_dot1 = 0

        for i in range(N):
            x_dot0 += Alphas[0][i] * kernel(x, X[i], kargs)
            x_dot1 += Alphas[1][i] * kernel(x, X[i], kargs)
        
        x_dot0 += betas[0]
        x_dot1 += betas[1]

        return [x_dot0, x_dot1]

    return mode0

def print_eq(kernel_list, Alphas, betas, X):
    K = kernel_list[0][0](X, X, kernel_list[0][1])

    N = X.shape[0]

    x1_squared  = Alphas[0] @ (X.T[0]**2)
    x2_squared  = Alphas[0] @ (X.T[1]**2)
    x1_x2       = Alphas[0] @ (X.T[0] * X.T[1]) * 2
    x1          = Alphas[0] @ (X.T[0]) * 2
    x2          = Alphas[0] @ (X.T[1]) * 2
    one_coef    = Alphas[0] @ np.ones(N)

    print("\nx1_dot\t=")
    print("% .8f" % x1_squared, "\tx1**2\t+")
    print("% .8f" % x2_squared, "\tx2**2\t+")
    print("% .8f" % x1_x2, "\tx1x2\t+")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (one_coef + betas[0]))


    x1_squared  = Alphas[1] @ (X.T[0]**2)
    x2_squared  = Alphas[1] @ (X.T[1]**2)
    x1_x2       = Alphas[1] @ (X.T[0] * X.T[1]) * 2
    x1          = Alphas[1] @ (X.T[0]) * 2
    x2          = Alphas[1] @ (X.T[1]) * 2
    one_coef    = Alphas[1] @ np.ones(N)

    print("\nx2_dot\t=")
    print("% .8f" % x1_squared, "\tx1**2\t+")
    print("% .8f" % x2_squared, "\tx2**2\t+")
    print("% .8f" % x1_x2, "\tx1x2\t+")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (one_coef + betas[1]))

if __name__ == "__main__":
    ax = plt.axes()
    y0 = [[0.1, 0.2], [-0.1, 0.9]]#,[1],[2],[3],[0],[-1],[-2]]
    t_tuple = (0,6)
    stepsize = 0.1


    # PLOTTING ORIGINAL
    t_points, y_list = simulation_ode(fvdp2_1, y0, t_tuple, stepsize, eps=0)

    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        ax.scatter(y0_list,y1_list,c='b',s=1, label='Original' if i==0 else "")

    # PLOTTING PREDICTED
    t_points, y_list = simulation_ode(get_derv(kernel_list, Alphas, betas, X), y0, t_tuple, stepsize, eps=0)

    for i, temp_y in enumerate(y_list):
        y0_list = temp_y.T[0]
        y1_list = temp_y.T[1]
        ax.scatter(y0_list,y1_list,c='r',s=1, label='Predicted' if i==0 else "")






    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.legend()
    plt.show()

    print_eq(kernel_list, Alphas, betas, X)
    print()