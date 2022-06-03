import numpy as np
import random
import matplotlib.pyplot as plt
import yaml

random.seed(10)

"""
first mode:  y[t] = -0.905 y[t-1]   + 0.9 u[t-1] + e[t]
second mode: y[t] = -0.4   y^2[t-1] + 0.5 u[t-1] + e[t]
"""
def generate_disc_hyb1(y0, noise=False):
    u = np.zeros(101)
    y = np.zeros(101)
    y[0] = y0

    u[0] = random.random()
    for t in range(1, 101):
        if t < 41:
            y[t] = -0.905 * y[t-1]   + 0.9 * u[t-1]
        else:
            y[t] = -0.4 * y[t-1]**2 + 0.5 * u[t-1]
        u[t] = random.random()
    
    return(u,y)


u,y = generate_disc_hyb1(0.6)

plt.plot(range(101), y)
plt.show()