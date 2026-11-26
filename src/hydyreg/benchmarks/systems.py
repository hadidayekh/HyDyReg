"""The systems of the thesis experiments (Chapter 7), as functions that build them.

The eight benchmark systems of Tables 7.2-7.3 and the parametric system of Tables 7.4-7.5. Each function
returns a new SNDS (system under learning); `hydyreg.benchmarks.rows` holds the settings of the 26 rows.
"""
from math import floor, sqrt

import numpy as np

from ..systems import SNDS, SNDS_affine

__all__ = ["isolette", "affine_3modes", "lorentz_attractor", "poly2_3modes", "poly3_2modes_nonlinclass", "poly2_4d",
           "bio_sys", "ham", "parametric", "BENCHMARKS"]


def isolette():
    # Paramters of Isolette model
    a1 = -0.026
    a2 = -0.026
    a3 = 98.5

    def labelfun(x): return 1 if x[0] < a3 else 0

    def event1(t, x): return x[0] - a3
    event1.terminal = True
    event1.direction = 0

    eventlist = [event1]

    # Can use the linear SUL for this wrapper with parameters:
    params = [[[a1, -a1, 0],
               [0, 0, -1]],
              [[a2, -a2, 0],
               [0, 0, +1]]]

    params = np.array(params)

    return SNDS_affine(params, eventlist, labelfun)


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
    def event1(t, x): return x[0] - 1
    event1.terminal = True
    event1.direction = 0

    def event2(t, x): return x[1] - 2
    event2.terminal = True
    event2.direction = 0

    eventlist = [event2, event1]

    a00, a01, a03 = -2, -2, -1
    a10, a11, a13 = 4, -1, 0

    b00, b01, b03 = -3, 2, 0
    b10, b11, b13 = -4, -1, 0

    c00, c01, c03 = 0.1, 0.1, 1
    c10, c11, c13 = -0.3, 0.2, -2

    params = [[[a00, a01, a03],
               [a10, a11, a13]],
              [[b00, b01, b03],
               [b10, b11, b13]],
              [[c00, c01, c03],
               [c10, c11, c13]]]

    params = np.array(params)

    ##### DEFINING SNDS #####
    return SNDS_affine(params, eventlist, labelfun)


def lorentz_attractor():
    # Label function to decide which mode
    def labelfun(x):
        if x[1] + x[0] >= 0:
            return 0
        else:
            return 1

    # Events to be tracked by the IVP solver
    def event(t, x): return x[1] + x[0]
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
        return [sigma1 * (x[1] - x[0]),
                x[0]*(rho1 - x[2]) - x[1],
                x[0]*x[1] - beta1*x[2]]

    def eq2(t, x):
        return [sigma2 * (x[1] - x[0]),
                x[0]*(rho2 - x[2]) - x[1],
                x[0]*x[1] - beta2*x[2]]

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
    def event1(t, x): return x[1]
    event1.terminal = True
    event1.direction = 0
    def event2(t, x): return x[0]
    event2.terminal = True
    event2.direction = 0

    eventlist = [event1, event2]

    # Defining equations
    def eq0(t, x): return [-7, -x[0]]
    def eq1(t, x): return [0.5*x[0]**2 + 0.5*x[1], -9*x[0] + 3]
    def eq2(t, x): return [5, -0.1*x[0] - 10]
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

    def event(t, x): return x[1] + 0.2*x[0]**2
    event.terminal = True
    event.direction = 0

    eventlist = [event]

    a1, a2, a3 = 0.1, 0.04, -0.9
    b1, b2, b3 = -0.1, 0.06, -0.7

    def eq1(t, y):
        return [a1 * y[0]**2 + a2 * y[1]**3,
                a3 * y[0]]

    def eq2(t, y):
        return [b1 * y[0]**2 + b2 * y[1]**3,
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
    def event(t, x): return x[2]
    event.terminal = True
    event.direction = 0

    eventlist = [event]

    s = 1
    r1 = 5
    r2 = 5
    p = 0.1

    def eq1(t, x):
        return [- s * x[1],
                x[0] - r1,
                x[3],
                -p * (x[2] + r2)**2]

    def eq2(t, x):
        return [- s * x[1],
                x[0] - r1,
                x[3],
                p * (x[0] - x[2])**2]

    eqs = [eq1, eq2]

    ##### DEFINING SNDS #####
    return SNDS(2, 4, eqs, eventlist, labelfun)


def bio_sys():
    def labelfun(x):
        if x[0]-x[1] >= -3:
            return 0
        else:
            return 1

    def event(t, x): return x[0] - x[1] + 3
    event.terminal = True
    event.direction = 0
    eventlist = [event]

    def eq1(t, x):
        return [-x[1], 2*x[0]]

    def eq2(t, x):
        return [-x[1], x[0]**3]

    return SNDS(2, 2, [eq1, eq2], eventlist, labelfun)


def ham():
    def labelfun(x):
        if x[0]**2+x[1]**2 >= 0.51:
            return 0
        else:
            return 1

    def event(t, x): return x[0]**2+x[1]**2 - 0.51
    event.terminal = True
    event.direction = 0
    eventlist = [event]

    def eq1(t, x):
        return [x[1], -x[0]**3+x[0]]

    def eq2(t, x):
        return [-x[0]+x[1], -x[0]-x[1]]

    ##### DEFINING SNDS #####
    sul = SNDS(2, 2, [eq1, eq2], eventlist, labelfun)
    return sul


def parametric(nmodes, radius=1.0):
    assert nmodes > 1

    radii = radius/nmodes

    def labelfun(x, nmodes=nmodes):
        d = x[0]**2 + x[1]**2
        d = sqrt(d)
        label = floor(d/radii)
        return (nmodes-1)-label if label < nmodes else 0

    eventlist = []
    params = []
    for j in range(1,nmodes):
        def event(t, x, k=j):
            return x[0]**2 + x[1]**2 - (radii*k)**2
        event.direction = 0
        event.terminal = True
        eventlist.append(event)

    for j in range(1, nmodes+1):
        beta = -2*j if j % 2 == 0 else 2*j
        k = 1

        Aorg = np.array([[0, k*beta], [-beta,  0]])
        D, V = np.linalg.eig(Aorg)
        D = np.diag(D)

        epsilon_1 = -5
        epsilon_2 = 0
        
        E = np.array([[epsilon_1, 0], [0, epsilon_2]])
        
        #eigenvalues
        D1 = D + E
        
        A1 = V @ D1 @ np.linalg.inv(V)
        
        A = V @ D @ np.linalg.inv(V)

        A = A1.real

        params.append([[A[0,0], A[0,1], 0],
                [A[1,0], A[1,1], 0]])

    params = np.array(params)
    ##### DEFINING SNDS #####
    sul = SNDS_affine(params, eventlist, labelfun)
    return sul


BENCHMARKS = {name: globals()[name] for name in __all__[:8]}
