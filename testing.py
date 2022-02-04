from turtle import shape
from numpy import zeros
from pyparsing import java_style_comment


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
print(X_dot)