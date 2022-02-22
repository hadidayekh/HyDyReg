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

from opti import optimize

K = np.array([[17, 18, 19], [18, 20, 22], [19, 22, 25]])
x_dot = np.array([[1, 0], [1, 0], [1, 0]])
n = 1

optimize([K, K], x_dot, 0.01, 1, print_on=True)