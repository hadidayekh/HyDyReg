from experiment2_poly_gen import get_event, get_labeltest, get_modetr, simulation_ode_3, modetr_params
import matplotlib.pyplot as plt
import numpy as np
import kernels

X = np.genfromtxt("input/final_X.csv", delimiter=',')

Alpha0 = np.genfromtxt("output/Alphas0.csv", delimiter=',')
betas0 = np.genfromtxt("output/betas0.csv", delimiter=',')
Alpha1 = np.genfromtxt("output/Alphas1.csv", delimiter=',')
betas1 = np.genfromtxt("output/betas1.csv", delimiter=',')
Alpha2 = np.genfromtxt("output/Alphas2.csv", delimiter=',')
betas2 = np.genfromtxt("output/betas2.csv", delimiter=',')

Alphas = np.array([Alpha0, Alpha1, Alpha2])
betas = np.array([betas0, betas1, betas2])

lin_k = [kernels.linear, [0]]
poly_k = [kernels.polynomial, [2, 1, 1]]
kernel_list = [lin_k, poly_k, lin_k]

def get_mode3_kernels(kernel_list, Alphas, betas, X):
    assert(len(kernel_list) == 3)
    N = X.shape[0]

    # belongs to mode 0
    def mode3_0(t, x):
        kernel = kernel_list[0][0]
        kargs = kernel_list[0][1]
        

        x_dot0 = 0
        x_dot1 = 0

        for i in range(N):
            x_dot0 += Alphas[0][0][i] * kernel(x, X[i], kargs)
            x_dot1 += Alphas[0][1][i] * kernel(x, X[i], kargs)
        
        x_dot0 += betas[0][0]
        x_dot1 += betas[0][1]

        return [x_dot0, x_dot1]

    # belongs to mode 1
    def mode3_1(t, x):
        kernel = kernel_list[1][0]
        kargs = kernel_list[1][1]

        

        x_dot0 = 0
        x_dot1 = 0

        for i in range(N):
            x_dot0 += Alphas[1][0][i] * kernel(x, X[i], kargs)
            x_dot1 += Alphas[1][1][i] * kernel(x, X[i], kargs) 
        
        x_dot0 += betas[1][0]
        x_dot1 += betas[1][1]

        return [x_dot0, x_dot1]
    

    # belongs to mode 1
    def mode3_2(t, x):
        kernel = kernel_list[1][0]
        kargs = kernel_list[1][1]

        

        x_dot0 = 0
        x_dot1 = 0

        for i in range(N):
            x_dot0 += Alphas[2][0][i] * kernel(x, X[i], kargs)
            x_dot1 += Alphas[2][1][i] * kernel(x, X[i], kargs) 
        
        x_dot0 += betas[2][0]
        x_dot1 += betas[2][1]

        return [x_dot0, x_dot1]

    return [mode3_0, mode3_1, mode3_2]




modetr = get_modetr(0)
event = get_event(0)
labeltest = get_labeltest(0)
y0 = [[-1,1],[1,4],[2,-3]]

T = 5
stepsize = 0.05
t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)


ax = plt.axes()

for i, temp_y in enumerate(y_list):
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    ax.scatter(y0_list,y1_list,c='b',s=1,label="Original" if i==0 else "")


mode3 = get_mode3_kernels(kernel_list, Alphas, betas, X)
t_list, y_list = simulation_ode_3(mode3, event, labeltest, y0, T, stepsize)

i = 0
for temp_y in y_list:
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    ax.scatter(y0_list, y1_list,c='r',s=2,label="Predicted" if i == 0 else "")
    i += 1





plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()