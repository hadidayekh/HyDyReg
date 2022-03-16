"""
THIS CODE IS FROM INFERDYNAMIC GIT REPO
"""


from scipy.integrate import solve_ivp
import numpy as np
from functools import wraps

def simulation_ode(ode_func, y0, t_tuple, stepsize, noise_type=1, eps=0):
    """ Given a ODE function, some initial state, stepsize, then return the points.
        @ode_func: ODE function 
        @y0: inital state
        @t_tuple: 
        @stepsize: step size
        @eps: guass noise (defult 0)
    """
    t_start = t_tuple[0]
    t_end = t_tuple[1]
    t_points = np.arange(t_start, t_end + stepsize, stepsize)
    y_list = []

    for k in range(0,len(y0)):
        y_object = solve_ivp(ode_func, (t_start, t_end+stepsize), y0[k], t_eval = t_points, rtol=1e-7, atol=1e-9)
        y_points = y_object.y.T
        if eps > 0:
            for i in range(0,y_points.shape[0]):
                for j in range(0,y_points.shape[1]):
                    y_points[i][j] = y_points[i][j] + np.random.normal(0,eps)
        y_list.append(y_points)
    return t_points, y_list

def simulation_ode_3(modelist, eventlist, labelfun, y0, T, stepsize):
    
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
        label = labelfun(y0[k])

        ite = 0
        while status == 1:
            ite = ite + 1
            # print(ite)
            # print(yinitial)
            y_object = solve_ivp(modelist[label], (t_start, t_end + 1.1*stepsize), yinitial,
                                 t_eval=t_points, method='RK45', rtol=1e-7, atol=1e-9,
                                 max_step=stepsize/10, events=eventlist, dense_output=True)

            status = y_object.status
            if status == 1:
                # print(y_object.t_events)
                for a in y_object.t_events:
                    if len(a)>0:
                        t_start = a[0]
                # print(t_start)
                l = t_points.shape[0]
                for i in range(0,l):
                    if t_points[i] > t_start:
                        break
                t_points = t_points[i:]
                sol = y_object.sol
                yinitial = sol.__call__(t_start + stepsize/1000)
                label = labelfun(yinitial)
                y_points = y_object.y.T
                y_points = y_points[0:i]
            else: 
                y_points = y_object.y.T

            if ite == 1:
                Y_points = y_points
            else:
                Y_points = np.r_[Y_points,y_points]
        t_list.append(T_points)
        y_list.append(Y_points)
    return t_list, y_list


modetr_params = [
    [0.5,0.5,-9,3,5,-0.1,-10,-7,-1],
]

def get_modetr(param_id):
    a1,a2,a3,a4,b1,b2,b3,c1,c2 = modetr_params[param_id]

    def modetr_1(t,y):
        y0, y1= y
        dydt = dydt = [a1*y0**2 + a2*y1, a3*y0+a4]
        return dydt

    def modetr_2(t,y):
        y0, y1= y
        dydt = dydt = [b1, b2*y0+b3]
        return dydt

    def modetr_3(t,y):
        y0, y1= y
        dydt = dydt = [c1 , c2*y0]
        return dydt

    return [modetr_1,modetr_2,modetr_3]
def eventAttr():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        wrapper.direction = 0
        wrapper.terminal = True
        return wrapper
    return decorator


def get_event(param_id):
    @eventAttr()
    def eventtr_1(t,y):
        y0, y1 = y
        return y0

    @eventAttr()
    def eventtr_2(t,y):
        y0, y1 = y
        return y1

    return [eventtr_1, eventtr_2]

eventtr_1, eventtr_2= get_event(0)


def get_labeltest(param_id):
    def labeltest(y):
        if eventtr_1(0,y)<0 and eventtr_2(0,y)>0:
            return 0
        elif eventtr_1(0,y)>=0 and eventtr_2(0,y)>0:
            return 1
        else:
            return 2
    return labeltest