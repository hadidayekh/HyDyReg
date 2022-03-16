from turtle import shape
from numpy import zeros
from pyparsing import java_style_comment
import numpy as np
""" 
print("Hello world!")
import numpy as np

x_list = np.zeros([3, 10, 2])
t_list = np.zeros([3, 10])

c_list = [4, -1, 3]

for i in range(x_list.shape[0]):
    for j in range(x_list[0].shape[0]):
        t_list[i][j] = j
        x_list[i][j][0] = 2*j + c_list[i]
        x_list[i][j][1] = 3*j + c_list[i]


from diff_methods import diff_method_paper

X, X_dot = diff_method_paper(t_list, x_list)

print(X)
print(X_dot) """

from opti import optimizeL1, optimizeL2

# K = np.array([[17, 18, 19], [18, 20, 22], [19, 22, 25]])
# x_dot = np.array([[1, 0], [1, 0], [1, 0]])
# n = 1

# optimizeL1([K, K], x_dot, 0.01, 1, print_on=True)

import numpy as np

X = np.genfromtxt("input/final_X.csv", delimiter=',')
X_dot = np.genfromtxt("input/final_X_dot.csv", delimiter=',')

Alphas0 = np.genfromtxt("output/Alphas0.csv", delimiter=',')
betas0 = np.genfromtxt("output/betas0.csv", delimiter=',')

Alphas1 = np.genfromtxt("output/Alphas1.csv", delimiter=',')
betas1 = np.genfromtxt("output/betas1.csv", delimiter=',')

from kernels import linear

K = linear(X, X, [0])

print(betas0.shape)


i_star0 = np.argmax(Alphas0[0])
alpha0 = Alphas0[0][i_star0]
i_star1 = np.argmax(Alphas1[0])
alpha1 = Alphas1[0][i_star1]
# print(alpha * K[i_star] + betas0[0]*np.ones(194) - X_dot.T[0])

# print(alpha0 * X[i_star0])
# print(betas0[0])
# print(alpha1 * X[i_star1])
# print(betas1[0])

print(betas0[1], betas1[1])


print(Alphas1[0] @ X.T[0])
print(Alphas1[0] @ X.T[1])