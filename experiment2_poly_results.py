from experiment2_poly_gen import get_event, get_labeltest, get_modetr, simulation_ode_3
from main import get_modes
import matplotlib.pyplot as plt
import numpy as np
import kernels
from sklearn.linear_model import LogisticRegression
import matplotlib.colors
from sklearn.inspection import DecisionBoundaryDisplay

X = np.genfromtxt("input/final_X.csv", delimiter=',')

Alpha0 = np.genfromtxt("output/Alphas0.csv", delimiter=',')
betas0 = np.genfromtxt("output/betas0.csv", delimiter=',')
Alpha1 = np.genfromtxt("output/Alphas1.csv", delimiter=',')
betas1 = np.genfromtxt("output/betas1.csv", delimiter=',')
Alpha2 = np.genfromtxt("output/Alphas2.csv", delimiter=',')
betas2 = np.genfromtxt("output/betas2.csv", delimiter=',')

Chis0 = np.genfromtxt("output/Chis0.csv", delimiter=',')
Chis1 = np.genfromtxt("output/Chis1.csv", delimiter=',')
Chis2 = np.genfromtxt("output/Chis2.csv", delimiter=',')

Alphas = np.array([Alpha0, Alpha1, Alpha2])
betas = np.array([betas0, betas1, betas2])
Chis = np.array([Chis0, Chis1, Chis2])

modes = get_modes(Chis)


# define model
model = LogisticRegression(multi_class='ovr')
# fit model
model.fit(X, modes)



clf = model

# Retrieve the model parameters.
b = clf.intercept_
w = clf.coef_

# for i in range(3):
#     w[i] = w[i] / np.linalg.norm(w[i])
#     b[i] = b[i] / np.linalg.norm(w[i])

w1, w2 = w.T

# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2


colors = ['tab:blue', 'tab:orange', 'tab:green']
cmap = matplotlib.colors.ListedColormap(colors)


lin_k = [kernels.linear, [0]]
poly_k = [kernels.polynomial, [2, 1, 1]]
kernel_list = [poly_k, lin_k, lin_k]

def print_eq(kernel_list, Alphas, betas, X):
    # poly equation
    print("First mode:")
    K = kernel_list[0][0](X, X, kernel_list[0][1])

    N = X.shape[0]
    print("NUMBER OF POINTS IS "+str(N)+"\n")

    x1_squared  = Alphas[0][0] @ (X.T[0]**2)
    x2_squared  = Alphas[0][0] @ (X.T[1]**2)
    x1_x2       = Alphas[0][0] @ (X.T[0] * X.T[1]) * 2
    x1          = Alphas[0][0] @ (X.T[0]) * 2
    x2          = Alphas[0][0] @ (X.T[1]) * 2
    one_coef    = Alphas[0][0] @ np.ones(N)

    print("\nx1_dot\t=")
    print("% .8f" % x1_squared, "\tx1**2\t+")
    print("% .8f" % x2_squared, "\tx2**2\t+")
    print("% .8f" % x1_x2, "\tx1x2\t+")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (one_coef + betas[0][0]))


    x1_squared  = Alphas[0][1] @ (X.T[0]**2)
    x2_squared  = Alphas[0][1] @ (X.T[1]**2)
    x1_x2       = Alphas[0][1] @ (X.T[0] * X.T[1]) * 2
    x1          = Alphas[0][1] @ (X.T[0]) * 2
    x2          = Alphas[0][1] @ (X.T[1]) * 2
    one_coef    = Alphas[0][1] @ np.ones(N)

    print("\nx2_dot\t=")
    print("% .8f" % x1_squared, "\tx1**2\t+")
    print("% .8f" % x2_squared, "\tx2**2\t+")
    print("% .8f" % x1_x2, "\tx1x2\t+")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (one_coef + betas[0][1]))



    # first linear equation
    print("\nSecond mode:")
    K = kernel_list[1][0](X, X, kernel_list[1][1])
    x1          = Alphas[1][0] @ (X.T[0])
    x2          = Alphas[1][0] @ (X.T[1])

    print("\nx1_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas[1][0]))


    x1          = Alphas[1][1] @ (X.T[0])
    x2          = Alphas[1][1] @ (X.T[1])

    print("\nx2_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas[1][1]))


    # second linear equation
    print("\nThird mode:")
    K = kernel_list[2][0](X, X, kernel_list[2][1])
    x1          = Alphas[2][0] @ (X.T[0])
    x2          = Alphas[2][0] @ (X.T[1])

    print("\nx1_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas[2][0]))


    x1          = Alphas[2][1] @ (X.T[0])
    x2          = Alphas[2][1] @ (X.T[1])

    print("\nx2_dot\t=")
    print("% .8f" % x1, "\tx1\t+")
    print("% .8f" % x2, "\tx2\t+")
    print("% .8f" % (betas[2][1]))



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


print_eq(kernel_list, Alphas, betas, X)
#exit()


modetr = get_modetr(0)
event = get_event(0)
labeltest = get_labeltest(0)
y0 = [[-1,1],[1,4],[2,-3]]
#y0 = [[1,-1],[-1,-4],[-2,3]]
# y0 = [[-6, 5]]
T = 5
stepsize = 0.05
t_list, y_list = simulation_ode_3(modetr, event, labeltest, y0, T, stepsize)


ax = plt.axes()

for i, temp_y in enumerate(y_list):
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    #ax.scatter(y0_list,y1_list,c='k',s=1,label="Original" if i==0 else "")

print("hellohello")

def labelfun(x):
    return model.predict(np.array(x).reshape(1,-1))[0]

mode3 = get_mode3_kernels(kernel_list, Alphas, betas, X)
t_list, y_list = simulation_ode_3(mode3, event, labelfun, y0, T, stepsize)

i = 0
for temp_y in y_list:
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    y = np.array([y0_list, y1_list]).T
    mode_pred = clf.predict(y)
    print(mode_pred)
    for j in range(3):
        ax.scatter(y0_list[mode_pred==j],y1_list[mode_pred==j],c=colors[j],s=1,label="Predicted" if i==0 else "")
    
    i += 1



disp = DecisionBoundaryDisplay.from_estimator(clf, X, grid_resolution=1000, ax=ax, alpha=0.1, cmap=cmap)


plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()