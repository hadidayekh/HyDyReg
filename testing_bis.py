from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


a0 = np.array([-2, 1])
a1 = np.array([2,3])
b = np.array([0, 0])


# a0 = a0 / np.linalg.norm(a0)
# a1 = a1 / np.linalg.norm(a1)

m0 = -a0[0] / a0[1]
m1 = -a1[0] / a1[1]

m = np.array([m0, m1])

bplot = np.zeros(2)
bplot[0] = -b[0]/a0[1]
bplot[1] = -b[1]/a1[1]

xmin = -1
xmax = 1
x = np.array([xmin, xmax])

# plotting first line
plt.plot(x, m[0] * x + bplot[0], c='b', lw=1)
# plotting second line
plt.plot(x, m[1] * x + bplot[1], c='b', lw=1)


abis = a0 / np.linalg.norm(a0) - a1 / np.linalg.norm(a1)
bbis = b[0] / np.linalg.norm(a0) - b[1] / np.linalg.norm(a1)



mbis = -abis[0] / abis[1]
bbis = -bbis / abis[1]

# a0 x + a1 y + b = 0
# y = (-a0/a1) x + (-b/a1)

plt.plot(x, mbis * x + bbis, c='r', alpha=0.5, lw=1, ls='--')

# # plotting first bis
# plt.plot(x, (m[0] + m[1])/2 * x + (b[0] + b[1])/2, c='r', alpha=0.5, lw=1, ls='--')
# # plotting second bis
# plt.plot(x, (m[0] - m[1])/2 * x + (b[0] - b[1])/2, c='k', alpha=0.5, lw=1, ls='--')

plt.xlim(xmin, xmax)

plt.show()