from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# setting path
sys.path.append('./')
sys.path.append('../')
from snds import SNDS

def labelfun(x):
    x1, x2 = x
    if x2 <= 3 and x1 <= -1:
        return 1
    else:
        return 0

def event_track1(t, x):
    x1, x2 = x
    return x1 + 1
def event_track2(t, x):
    x1, x2 = x
    return x2 - 3


event_track1.terminal = True
event_track1.direction = 0
event_track2.terminal = True
event_track2.direction = 0
eventlist = [event_track1, event_track2]



y0 = [-14,14]

a = -3
b = -2
cons1 = 0
c = 4
d = 1
cons2 = 0


if (a+d)**2 >= 4*(a*d-b*c):
    eig1 = (a+d)/2.0 + 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    eig2 = (a+d)/2.0 - 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    print("Solutions are real!")
    exit()
    print("Eigenvalue 1:", eig1)
    print("Eigenvalue 2:", eig2)
    if(eig1*eig2 >= 0):
        print("Eigenvalues have the same sign!")
        if(eig1 >= 0):
            print("It is a repelling/unstable node!")
        else:
            print("It is an attracting/stable node!")
    else:
        print("Eigenvalues have opposite signs!")
        print("It is a saddle!")
else:
    print("Solutions are complex!")
    if a+d >= 0:
        print("It is a repelling/unstable focus!")
    else:
        print("It is an attracting/stable focus!")


def eq1(t, y):
    return [a*y[0] + b*y[1] + cons1, c*y[0] + d*y[1] + cons2]
def eq2(t,y):
    return [5, -5]

eqs = [eq1, eq2]


sul = SNDS(2, 2, eqs, eventlist, labelfun)

t_list, x_list = sul.simulate_trajectory(0.03, 3, [y0])




for x in x_list:
    x1 = x.T[0]
    x2 = x.T[1]

    plt.scatter(x1, x2, s=3,c='b')
    plt.scatter(x1[0], x2[0], s=3, c='r')



# t_list = np.linspace(0, 10, 100)

# y_obj = solve_ivp(dy_dt, (0,10), y0, t_eval=t_list)

# y = y_obj.y

# plt.scatter(y[0], y[1], s=3,c='b')
# plt.scatter(y[0][0], y[1][0], s=3, c='r')

plt.show()