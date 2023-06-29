import numpy as np
import sys
sys.path.append('./')
from snds import SNDS, SNDS_linear
from main import learn_system
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# def learn_system(exp_name, sul, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes, poly_degree, seg_tol=0.01, diff_method=diff_method_fps, train_coefs=True, kernels_list=None, kargs_list=None, with_plots=True, print_on=True, classifier = LogisticRegression(penalty='none')):

'''
exp name
SUL
time_step
time_range
x0s
time_step_test
time_range_test
x0s_test
nmodes
poly_degree
classifier
'''

def isolette():
    # Paramters of Isolette model
    a1 = -0.026
    a2 = -0.026
    a3 = 98.5

    labelfun = lambda x : 0 if x[0] < a3 else 1

    event1 = lambda t, x : x[0] - a3
    event1.terminal = True
    event1.direction = 0

    eventlist = [event1]

    # Can use the linear SUL for this wrapper with parameters:
    params = [[[a1, -a1, 0],
            [0, 0, 1]],
            [[a2, -a2, 0],
            [0, 0, -1]]]

    params = np.array(params)

    return SNDS_linear(params, eventlist, labelfun)

def affine_3modes():
    # Label function to decide which mode
    def labelfun(x):
        if x[0] >= 1:
            return 1
        elif x[1] <= 2:
            return 0
        else:
            return 2


    # Events to be tracked by the IVP solver
    event1 = lambda t, x : x[0] - 1
    event1.terminal = True
    event1.direction = 0

    event2 = lambda t, x : x[1] - 2
    event2.terminal = True
    event2.direction = 0

    eventlist = [event2, event1]


    a00, a01, a03 =  -2, -2, -1
    a10, a11, a13 =  4, -1, 0

    b00, b01, b03 =  -3, 2, 0
    b10, b11, b13 =  -4, -1, 0

    c00, c01, c03 =  0.1, 0.1, 1
    c10, c11, c13 =  -0.3, 0.2, -2

    params = [[[a00, a01, a03],
            [a10, a11, a13]],
            [[b00, b01, b03],
            [b10, b11, b13]],
            [[c00, c01, c03],
            [c10, c11, c13]]]

    params = np.array(params)


    ##### DEFINING SNDS #####
    return SNDS_linear(params, eventlist, labelfun)

def lorentz_attractor():
    # Label function to decide which mode
    def labelfun(x):
        if x[1] + x[0] >= 0:
            return 0
        else:
            return 1

    # Events to be tracked by the IVP solver
    event = lambda t, x : x[1] + x[0]
    event.terminal = True
    event.direction = 0

    eventlist = [event]

    sigma1 = 10
    beta1 = 8.0/3
    rho1 = 28
    sigma2 = 28
    beta2 = 4
    rho2 = 46.92

    def eq1(t, x):
        return [ sigma1 * (x[1] - x[0]),
        x[0]*(rho1 - x[2]) - x[1],
        x[0]*x[1] - beta1*x[2] ]

    def eq2(t, x):
        return [ sigma2 * (x[1] - x[0]),
        x[0]*(rho2 - x[2]) - x[1],
        x[0]*x[1] - beta2*x[2] ]

    eqs = [eq1, eq2]


    ##### DEFINING SNDS #####
    return SNDS(2, 3, eqs, eventlist, labelfun)

def poly2_3modes():
    # Label function to decide which mode
    def labelfun(x):
        x0, x1 = x
        if x1 <= 0:
            return 0
        elif x0 <= 0:
            return 1
        else:
            return 2

    # Events to be tracked by the IVP solver
    event1 = lambda t, x : x[1]
    event1.terminal = True
    event1.direction = 0
    event2 = lambda t, x : x[0]
    event2.terminal = True
    event2.direction = 0

    eventlist = [event1, event2]

    # Defining equations
    eq0 = lambda t, x: [-7, -x[0]]
    eq1 = lambda t, x: [0.5*x[0]**2 + 0.5*x[1], -9*x[0] + 3]
    eq2 = lambda t, x: [5, -0.1*x[0] - 10]
    eqs = [eq0, eq1, eq2]

    ##### DEFINING SNDS #####
    return SNDS(3, 2, eqs, eventlist, labelfun)

def poly3_2modes_nonlinclass():
    # Label function to decide which mode
    def labelfun(x):
        if x[1] >= -0.2*x[0]**2:
            return 0
        else:
            return 1

    event = lambda t, x : x[1] + 0.2*x[0]**2
    event.terminal = True
    event.direction = 0

    eventlist = [event]

    a1, a2, a3 =  0.1, 0.04, -0.9
    b1, b2, b3 = -0.1, 0.06, -0.7

    def eq1(t, y):
        return [ a1 * y[0]**2 + a2 * y[1]**3,
                a3 * y[0]]

    def eq2(t, y):
        return [ b1 * y[0]**2 + b2 * y[1]**3,
                b3 * y[0]]

    eqs = [eq1, eq2]

    ##### DEFINING SNDS #####
    return SNDS(2, 2, eqs, eventlist, labelfun)

def poly2_4d():
    # Label function to decide which mode
    def labelfun(x):
        if x[2] >= 0:
            return 0
        else:
            return 1

    # Events to be tracked by the IVP solver
    event = lambda t, x : x[2]
    event.terminal = True
    event.direction = 0

    eventlist = [event]

    s = 1
    r1 = 5
    r2 = 5
    p = 0.1

    def eq1(t, x):
        return [ - s * x[1],
        x[0] - r1,
        x[3],
        -p * (x[2] + r2)**2 ]

    def eq2(t, x):
        return [ - s * x[1],
        x[0] - r1,
        x[3],
        p * (x[0] - x[2])**2 ]

    eqs = [eq1, eq2]

    ##### DEFINING SNDS #####
    return SNDS(2, 4, eqs, eventlist, labelfun)

exp_infos = {
    'isolette' : {
        'sul' : isolette(),
        'time_step' : 0.1,
        'time_range' : 50,
        'x0s' : [[105.5, 80], [97.5, 100]],
        'time_step_test' : 0.1,
        'time_range_test' : 50,
        'x0s_test' : [[98, 99]],
        'nmodes' : 2,
        'poly_degree' : 1,
        'seg_tol' : 0.01,
        'classifier' : LogisticRegression(penalty='none')
    } ,
    'affine_3modes' : {
        'sul' : affine_3modes(),
        'time_step' : 0.04,
        'time_range' : 4,
        'x0s' : [[-2, 4], [1.5, 10], [0,5]],
        'time_step_test' : 0.02,
        'time_range_test' : 2,
        'x0s_test' : [[2,12], [-4, 10]],
        'nmodes' : 3,
        'poly_degree' : 1,
        'seg_tol' : 0.5,
        'classifier' : LogisticRegression(penalty='none', multi_class='ovr')
    } ,
    'lorentz_attractor' : {
        'sul' : lorentz_attractor(),
        'time_step' : 0.005,
        'time_range' : 1,
        'x0s' : [[5,5,5], [2,2,2]],
        'time_step_test' : 0.005,
        'time_range_test' : 1,
        'x0s_test' : [[3,3,3]],
        'nmodes' : 2,
        'poly_degree' : 2,
        'seg_tol' : 0.5,
        'classifier' : LogisticRegression(penalty='none', multi_class='ovr')
    } ,
    'poly2_3modes' : {
        'sul' : poly2_3modes(),
        'time_step' : 0.04,
        'time_range' : 4,
        'x0s' : [[-1,1],[2,-3]],
        'time_step_test' : 0.04,
        'time_range_test' : 4,
        'x0s_test' : [[1,4]],
        'nmodes' : 3,
        'poly_degree' : 2,
        'seg_tol' : 0.1,
        'classifier' : LogisticRegression(penalty='none', multi_class='multinomial')
    } ,
    'poly3_2modes_nonlinclass' : {
        'sul' : poly3_2modes_nonlinclass(),
        'time_step' : 0.04,
        'time_range' : 8,
        'x0s' : [[0,4], [0,4.5], [0,5]],
        'time_step_test' : 0.04,
        'time_range_test' : 8,
        'x0s_test' : [[0,2.5]],
        'nmodes' : 2,
        'poly_degree' : 3,
        'seg_tol' : 0.5,
        'classifier' : SVC(C=10)
    } ,
    'poly2_4d' : {
        'sul' : poly2_4d(),
        'time_step' : 0.05,
        'time_range' : 5,
        'x0s' : [[6,0.7,2,0], [4,0.1,3.1,0], [5.9,0.2,-3,0], [4.1,0.5,2,0]],
        'time_step_test' : 0.01,
        'time_range_test' : 2,
        'x0s_test' : [[4.6,0.13,2,0], [5.3,0.17,-2,0]],
        'nmodes' : 2,
        'poly_degree' : 2,
        'seg_tol' : 0.01,
        'classifier' : LogisticRegression(penalty='none')
    }
}

for exp_title in exp_infos.keys():
    exp = exp_infos[exp_title]
    try:
        learn_system(exp_name = exp_title,
                 sul =              exp['sul'],
                 time_step =        exp['time_step'],
                 time_range =       exp['time_range'],
                 x0s =              exp['x0s'],
                 time_step_test =   exp['time_step_test'],
                 time_range_test =  exp['time_range_test'],
                 x0s_test =         exp['x0s_test'],
                 nmodes =           exp['nmodes'],
                 poly_degree =      exp['poly_degree'],
                 seg_tol =          exp['seg_tol'],
                 classifier =       exp['classifier'],
                 C_space =          [0.1, 1, 2, 5, 10],
                 write_on =         True,
                 deltas =           0.01)
    except:
        print("Experiment " + exp_title + " failed.")

plt.show()