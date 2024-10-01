import sys

sys.path.append('./')

from snds import Trajectories
from diff_methods import diff_method_forward, diff_method_ps
from opti import HybridRegressor
import matplotlib.pyplot as plt
from plotting import plot_with_modes_3D, draw_trajectories_3D, plot_with_modes_2D
from utils import gen_full_matlab_hyeq_class, relative_diff
import numpy as np
from sklearn.linear_model import LogisticRegression
import pysindy as ps

data = np.genfromtxt("experiments/dcdc_melco/data.csv", delimiter=',').T

t = data[0]

print(t.shape)
print(np.unique(t).shape)
Iin0 = data[1]
Iout0 = data[2]
Vout0 = data[3]
gate = data[4]

X_init = data[1:4].T
indices = np.where(np.diff(gate) != 0)[0]+1

t_list = np.split(t, indices)
x_list = np.split(X_init, indices)

# t_list = [t]
# x_list = [X_init]

'''
AVANT:
- segmentation using three pt fwd and bwd derivative
- derivative using five point stencil method

POUR CETTE EXEMPLE:
- segmentation using two pt fwd and bwd derivative (t(i) - t(i-1))
- derivative using two pt fwd derivative
'''

trajs = Trajectories(t_list, x_list)
# draw_trajectories_3D(trajs)
# plt.show()
# exit()
segments = trajs.segment_trajectories_two_points(0.05)

# f, a = plt.subplots()
# for i in range(trajs.num_trajectories):
#     segment = segments[i]
#     for s in range(len(segment)-1):
#         a.plot(trajs.t_list[i][segment[s]:segment[s+1]], trajs.x_list[i].T[0][segment[s]:segment[s+1]], markersize=.1)
#         a.plot(trajs.t_list[i][segment[s]:segment[s+1]], trajs.x_list[i].T[1][segment[s]:segment[s+1]], markersize=.1)
#     if len(segment)>0:
#         a.plot(trajs.t_list[i][:segment[0]], trajs.x_list[i].T[0][:segment[0]], markersize=.1)
#         a.plot(trajs.t_list[i][segment[-1]:], trajs.x_list[i].T[0][segment[-1]:], markersize=.1)
#         a.plot(trajs.t_list[i][:segment[0]], trajs.x_list[i].T[1][:segment[0]], markersize=.1)
#         a.plot(trajs.t_list[i][segment[-1]:], trajs.x_list[i].T[1][segment[-1]:], markersize=.1)
#     else:
#         a.plot(trajs.t_list[i], trajs.x_list[i].T[0], markersize=.1)
#         a.plot(trajs.t_list[i], trajs.x_list[i].T[1], markersize=.1)
# plt.show()
# exit()

# dm = ps.SmoothedFiniteDifference(order=4, drop_endpoints=True, smoother_kws={'window_length': 5})
dm = ps.FiniteDifference(order=4, drop_endpoints=True)
dm = ps.SINDyDerivative(kind='savitzky_golay', left=0.5, right=0.5, order=4)
X, Xdot, seg_res, _ = diff_method_ps(trajs.t_list, trajs.x_list, segments, dm=dm)
for i in range(Xdot.shape[0]):
    Xdot[i][0] /= 1
# print(np.sort(Xdot.T[0])[:10])
# print(len(seg_res))
# exit()
model = HybridRegressor(deg=1)
model.fit_first_hypothesis(X, Xdot, seg_res, good_RD=0.05)
for j in range(model.n):
    model.retrain_continuous(j)


X_dot_predict = model.predict(model.Xfit_, model.modesfit_)

# print(X_dot_predict - model.X_dot_fit_)
sorted_rd = np.sort(relative_diff(X_dot_predict, model.X_dot_fit_))
N_sorded_rd = sorted_rd.shape[0]
print(sorted_rd[N_sorded_rd-50:])
# exit()
# print(model.Omegas_[0], model.betas_[0])
# print(model.Omegas_[1], model.betas_[1])
# exit()

model.classify_modes()
print(model.gen_full_latex_eq(with_classifier_boundaries=False))


# print(model.Xfit_.shape)
# print(model.X_dot_fit_.shape)
# print(model.modesfit_.shape)
# print(model.X_dot_fit_[model.modesfit_==2])

for i in range(np.max(model.modesfit_)+1):
    print(f"Mode {i+1}: {np.count_nonzero(model.modesfit_ == i)} points.")
# plot_with_modes_2D(model.Xfit_.T[:2].T, model.modesfit_)
# draw_trajectories_3D(trajs)


# plt.show()
# print(np.diff(seg_res))

# print(gen_full_matlab_hyeq_class(model, name="DCDC_Melco").replace(".", ","))
print(X_init[0])
print(t[0], t[-1])
exit()

hypothesis = model.to_SNDS()
t0 = t[0]
T = t[-1]
x0s = [X[0]]
ts = 0.000001
print(t0,T)
hyp_trajs = hypothesis.simulate_trajectory(ts, T, x0s, t0, stop_at_sliding_motion=False)
print("finished simulation")
f, a = draw_trajectories_3D(trajs, plot_xt = True, colors=['r'])
draw_trajectories_3D(hyp_trajs, fig=f, axs=a, colors=['b'], plot_xt=True)
plt.show()