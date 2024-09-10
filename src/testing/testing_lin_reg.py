import sys
import matplotlib.pyplot as plt

# setting path
sys.path.append('./')
sys.path.append('../')

import snds
import numpy as np
from diff_methods import diff_method_fps

sul = snds.SNDS_affine(np.array([[[0, 0, 8], [0,0,-6]]]), [], lambda _ : 0)
trajs = sul.simulate_trajectory(0.05, 3, [[-11, 11]])

X, X_dot = diff_method_fps(np.array(trajs.t_list), np.array(trajs.x_list))

print(X.shape, X_dot.shape)


from sklearn.linear_model import LinearRegression

model = LinearRegression()
y = X_dot.T[0]
model.fit(X, y)

X_dot_pred = model.predict(X)
X = np.array([[1, x[0], x[1]] for x in X])
print(X.shape)
print(y.shape)

print("Normal coef eq:")
beta = np.linalg.inv(X.T @ X) @ X.T @ y
print(beta)


print("Pseudoinverse eq:")
theta = np.linalg.pinv(X).dot(y)
print(theta)

print("Model parameters:")
print(model.coef_)
print(model.intercept_)