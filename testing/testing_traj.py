from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

y0 = [15,12]

a = -0.5
b = -2
cons1 = 0
c = 4
d = 0
cons2 = 0


if (a+d)**2 >= 4*(a*d-b*c):
    eig1 = (a+d)/2.0 + 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    eig2 = (a+d)/2.0 - 0.5 * np.sqrt((a+d)**2 - 4*(a*d-b*c))
    print("Solutions are real!")
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


def dy_dt(t, y):
    return [a*y[0] + b*y[1] + cons1, c*y[0] + d*y[1] + cons2]

t_list = np.linspace(0, 10, 100)

y_obj = solve_ivp(dy_dt, (0,10), y0, t_eval=t_list)

y = y_obj.y

plt.scatter(y[0], y[1], s=3,c='b')
plt.scatter(y[0][0], y[1][0], s=3, c='r')

plt.show()