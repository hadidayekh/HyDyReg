from cProfile import label
import numpy as np
from scipy.integrate import solve_ivp
import kernels
from functools import wraps
import matplotlib.pyplot as plt

def simulation_ode_2(modelist, event, y0, T, stepsize, verbose=False, noise = 0):
    t_list = []
    y_list = []

    for k in range(0,len(y0)):
        t_start = 0
        t_end = T
        yinitial = y0[k]

        num = round((t_end - t_start)/stepsize + 1)
        t_points = np.linspace(t_start, t_end, num)
        T_points = np.linspace(t_start, t_end, num)

        status = 1
        if event(0, y0[k]) > 0:
            label = 0
        else:
            label = 1
        ite = 0

        while status == 1:
            ite = ite + 1
            if verbose:
                print('Round', str(ite) + ':', 'y =', yinitial)
            if t_points[0] < t_start:
                break
            y_object = solve_ivp(modelist[label], (t_start, t_end + 1.1*stepsize), yinitial,
                                 t_eval=t_points, method='RK45', rtol=1e-7, atol=1e-9,
                                 max_step=stepsize/10, events=[event], dense_output=True)

            if (y_object.y == []):
                break

            status = y_object.status
            if status == 1:
                if label == 1:
                    label = 0
                else:
                    label = 1
                t_start = y_object.t_events[0][0]
                if verbose:
                    print('tstart =', t_start)
                l = t_points.shape[0]
                for i in range(0,l):
                    if t_points[i] > t_start:
                        break
                t_points=t_points[i:]
                sol = y_object.sol
                yinitial = sol.__call__(t_start + stepsize/1000)
                y_points = y_object.y.T
                y_points = y_points[0:i]
            else: 
                y_points = y_object.y.T

            if ite == 1:
                Y_points = y_points
            else:
                Y_points = np.r_[Y_points,y_points]
        
        if noise > 0:
            for i in range(0,Y_points.shape[0]):
                for j in range(0,Y_points.shape[1]):
                    Y_points[i][j] = Y_points[i][j] + np.random.normal(0,noise)
        t_list.append(T_points)
        y_list.append(Y_points)
    return t_list, y_list

mode2_params = [
    [-0.026, -1, -0.026, 1, 98.5],
    [-0.26, -1, -0.26, 1, 0],
]

def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator


def get_event1(param_id):
    _, _, _, _, c0 = mode2_params[param_id]

    @eventAttr()
    def event1(t,y):
        y0, y1 = y
        return y0 - c0

    return event1

def get_mode2(param_id):
    a1, b1, a2, b2, _ = mode2_params[param_id]

    def mode2_1(t,y):
        """ A hybrid automaton with 2 modes for an incubator.

        Returns derivative at point y = (y0, y1).

        """
        y0, y1 = y
        dydt = [a1 * (y0-y1), b1]
        return dydt
    

    def mode2_2(t,y):
        y0, y1 = y
        dydt = [a2 * (y0-y1), b2]
        return dydt

    return [mode2_1, mode2_2]

def get_mode2_kernels(kernel_list, Alphas, betas, X):
    assert(len(kernel_list) == 2)
    N = X.shape[0]

    # belongs to mode 0
    def mode2_0(t, x):
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
    def mode2_1(t, x):
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
    
    return [mode2_0, mode2_1]

def get_mode2_exp():
    def mode2_1(t, x):
        x0, x1 = x
        return [0.02413883*x0 + 0.02618191*x1 -4.953842473812118, -0.9628159492994963]
    def mode2_2(t, x):
        x0, x1 = x
        return [0.01895106*x0 + 0.02055505*x1 -3.798641102116914, 0.9841744795536129]
    return [mode2_1, mode2_2]

X = np.genfromtxt("input/final_X.csv", delimiter=',')

Alpha0 = np.genfromtxt("output/Alphas0.csv", delimiter=',')
betas0 = np.genfromtxt("output/betas0.csv", delimiter=',')
Alpha1 = np.genfromtxt("output/Alphas1.csv", delimiter=',')
betas1 = np.genfromtxt("output/betas1.csv", delimiter=',')

Alphas = np.array([Alpha0, Alpha1])
betas = np.array([betas0, betas1])

lin_k = [kernels.linear, [0]]
kernel_list = [lin_k, lin_k]


event1 = get_event1(0)
y0 = [[105.5, 80], [97.5, 100]]
y0 = [[105.5, 80], [97.5, 100], [50, 50], [120, 65]]

stepsize = 5
T = 100


ax = plt.axes()


t_list, y_list = simulation_ode_2(get_mode2(0), event1, y0, T, stepsize,noise=0)

i = 0
for temp_y in y_list:
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    ax.scatter(y0_list, y1_list,c='r',s=2,label="Simulated" if i == 0 else "")
    i += 1

mode2 = get_mode2_kernels(kernel_list, Alphas, betas, X)
t_list, y_list = simulation_ode_2(mode2, event1, y0, T, stepsize ,noise=0)

i = 0
for temp_y in y_list:
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    ax.scatter(y0_list, y1_list,c='b',s=2,label="Predicted" if i == 0 else "")
    i += 1



t_list, y_list = simulation_ode_2(get_mode2_exp(), event1, y0, T, stepsize,noise=0)

i = 0
for temp_y in y_list:
    y0_list = temp_y.T[0]
    y1_list = temp_y.T[1]
    #ax.scatter(y0_list, y1_list,c='g',s=2,label="SVMs" if i == 0 else "")
    i += 1

ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.legend()
plt.show()
