import sys

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from sklearn.metrics import mean_squared_error
from opti import HybridRegressor
from diff_methods import diff_method_fps
from snds import SNDS_affine
import plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from utils import get_piccolo_distance

exp_title = "isolette"
k_learn = True
save_plots = False
seg_tol = 0.1

# Paramters of Isolette model
a1 = -0.026
a2 = -0.026
a3 = 98.5


# Label function to decide which mode
def labelfun(x):
    if x[0] <= a3:
        return 0
    else:
        return 1


# Events to be tracked by the IVP solver
event1 = lambda t, x : x[0] - a3
event1.terminal = True
event1.direction = 0

eventlist = [event1]

# Can use the linear SUL for this wrapper with parameters:
a00 = a1
b00 = -a1
c00 = 0

a01 = 0
b01 = 0
c01 = 1

a10 = a2
b10 = -a2
c10 = 0

a11 = 0
b11 = 0
c11 = -1

params = [[[a00, b00, c00],
           [a01, b01, c01]],
          [[a10, b10, c10],
           [a11, b11, c11]]]

params = np.array(params)


##### DEFINING SNDS #####
sul = SNDS_affine(params, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters
x0s = [[105.5, 80], [97.5, 100]]
time_step = 0.1
time_range = 150

# Simulating trajectories
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)

# Segmenting
segments = trajectories.segment_trajectories(seg_tol)


X1 = trajectories.x_list[0]
X2 = trajectories.x_list[1]

plt.scatter(*X1.T, s=4)


print(get_piccolo_distance(X1, X2, x_order=20, y_order=20))

S = [X1[:segments[0][0]]]
for r in range(len(segments[0])-1):
    S.append(X1[segments[0][r]:segments[0][r+1]])
S.append(X1[segments[0][-1]:])

piccolo_distances = np.zeros((len(S), len(S)))
for i in range(len(S)):
    for j in range(len(S)):
        piccolo_distances[i][j] = get_piccolo_distance(S[i], S[j])

with np.printoptions(precision=3, suppress=True):
    print(piccolo_distances)

print(len(S))
print(S[0])
print(S[1])
plt.show()