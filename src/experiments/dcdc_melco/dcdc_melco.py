import sys
sys.path.append('./')

from snds import Trajectories
from diff_methods import diff_method_forward
from opti import HybridRegressor
from plotting import draw_trajectories_ND
import numpy as np

data = np.genfromtxt("experiments/dcdc_melco/data.csv", delimiter=',').T

t = data[0]

print(t.shape)
print(np.unique(t).shape)
Iin0 = data[1]
Iout0 = data[2]
Vout0 = data[3]
gate = data[4]

X = data[1:4].T
indices = np.where(np.diff(gate) != 0)[0]+1

t_list = np.split(t, indices)
x_list = np.split(X, indices)

trajs = Trajectories(t_list, x_list)
segments = trajs.segment_trajectories_two_points(0.05)
X, Xdot, seg_res = diff_method_forward(trajs.t_list, trajs.x_list, segments)
model = HybridRegressor()
model.fit_first_segment(X, Xdot, seg_res, good_RD=0.01)
# print(np.diff(seg_res))