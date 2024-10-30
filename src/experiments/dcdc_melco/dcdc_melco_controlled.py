import sys
from turtle import color

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

# data_file = "experiments/dcdc_melco/data.csv"
data_file = "experiments/dcdc_melco/data_controlled.csv"

seg_tol = 0.01
good_RD = 0.005



data = np.genfromtxt(data_file, delimiter=',').T
print(data.shape)

t = data[0][:]

print(t.shape)
print(np.unique(t).shape)

Iin0 = data[1]
Vout0 = data[2]
triangular = data[3]
duty = data[4]
pid_controller_internal = data[5]


X_init = data[[1,2,4,5], :].T
print(X_init.shape)



t_list = [t]
x_list = [X_init]
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
segments = trajs.segment_trajectories_two_points(seg_tol)

# f, a = plt.subplots()
# for i in range(trajs.num_trajectories):
#     segment = segments[i]
#     for s in range(len(segment)-1):
#         # a.plot(trajs.t_list[i][segment[s]:segment[s+1]], trajs.x_list[i].T[0][segment[s]:segment[s+1]], markersize=.1)
#         # a.plot(trajs.t_list[i][segment[s]:segment[s+1]], trajs.x_list[i].T[1][segment[s]:segment[s+1]], markersize=.1)
#         a.plot(trajs.t_list[i][segment[s]:segment[s+1]], trajs.x_list[i].T[3][segment[s]:segment[s+1]], markersize=.1)
#     if len(segment)>0:
#         # a.plot(trajs.t_list[i][:segment[0]], trajs.x_list[i].T[0][:segment[0]], markersize=.1)
#         # a.plot(trajs.t_list[i][segment[-1]:], trajs.x_list[i].T[0][segment[-1]:], markersize=.1)
#         # a.plot(trajs.t_list[i][:segment[0]], trajs.x_list[i].T[1][:segment[0]], markersize=.1)
#         # a.plot(trajs.t_list[i][segment[-1]:], trajs.x_list[i].T[1][segment[-1]:], markersize=.1)
#         a.plot(trajs.t_list[i][:segment[0]], trajs.x_list[i].T[3][:segment[0]], markersize=.1)
#         a.plot(trajs.t_list[i][segment[-1]:], trajs.x_list[i].T[3][segment[-1]:], markersize=.1)
#     else:
#         # a.plot(trajs.t_list[i], trajs.x_list[i].T[0], markersize=.1)
#         # a.plot(trajs.t_list[i], trajs.x_list[i].T[1], markersize=.1)
#         a.plot(trajs.t_list[i], trajs.x_list[i].T[3], markersize=.1)

# # # a.axhline(y = 0.75, color = 'r', linestyle = '-', alpha=0.2) 
# plt.show()
# exit()

# dm = ps.SmoothedFiniteDifference(order=4, drop_endpoints=True, smoother_kws={'window_length': 5})
# dm = ps.FiniteDifference(order=4, drop_endpoints=True)
dm = ps.SINDyDerivative(kind='savitzky_golay', left=0.5, right=0.5, order=4)
X, Xdot, seg_res, Time = diff_method_ps(trajs.t_list, trajs.x_list, segments, dm=dm)

# X, Xdot, seg_res, Time = diff_method_forward(trajs.t_list, trajs.x_list, segments)

# '''
# TRANSFORM
# '''
# from sklearn.preprocessing import StandardScaler, Normalizer
# X = Normalizer().fit_transform(X)
# Xdot = Normalizer().fit_transform(Xdot)

# print(np.sort(Xdot.T[0])[:10])
# print(len(seg_res))
# exit()
from sklearn.linear_model import Ridge, RidgeCV

model = HybridRegressor(deg=1)
alphas = [1000000, 1000000, 100000, 10000, 1000, 100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
# alphas = [0.001, 0.0001, 0.00001, 0]

regressor =  RidgeCV(alphas, alpha_per_target=True)
model.fit_first_hypothesis(X, Xdot, seg_res, good_RD=good_RD, regressor=regressor, Time=Time, shuffle_segments=True)
for j in range(model.n):
    model.retrain_continuous(j)

X_dot_predict = model.predict(model.Xfit_, model.modesfit_)



# print(X_dot_predict - model.X_dot_fit_)
# exit()
sorted_rd = np.sort(relative_diff(X_dot_predict, model.X_dot_fit_))
N_sorded_rd = sorted_rd.shape[0]
print(sorted_rd[N_sorded_rd-50:])
# exit()
# print(model.Omegas_.shape)
# print(model.Omegas_[0], model.betas_[0])
# print(model.Omegas_[1], model.betas_[1])
# exit()

Time_fit = model.Time_fit_

argsort_Time_fit = np.argsort(Time_fit)
sorted_Time_fit = Time_fit[argsort_Time_fit]
tri_points = np.ndarray(Time_fit.shape, dtype=Time_fit.dtype)
count = 0
for i in range(len(t)):
    try:
        if t[i] == sorted_Time_fit[count]:
            tri_points[argsort_Time_fit[count]] = triangular[i]
            count += 1
        else:
            continue
    except:
        continue


from sklearn.linear_model import LogisticRegressionCV
X_with_tri = np.r_[model.Xfit_.T, tri_points.reshape((1,-1))].T
X_dot_with_tri = np.r_[model.X_dot_fit_.T, tri_points.reshape((1,-1))].T
# model.classify_modes(classifier = LogisticRegression(penalty='none'),X = X_with_tri, X_dot = X_dot_with_tri, modes = model.modesfit_)
model.classify_modes(classifier = LogisticRegression(penalty='none'),X = X_with_tri[model.modesfit_ < 4], X_dot = X_dot_with_tri[model.modesfit_ < 4], modes = model.modesfit_[model.modesfit_ < 4])

# model.classify_modes(X=model.Xfit_[model.modesfit_ < 4], modes=model.modesfit_[model.modesfit_ < 4])

print(model.gen_full_latex_eq(with_classifier_boundaries=True, round_dig=4))
for i in range(np.max(model.modesfit_)+1):
    print(f"Mode {i+1}: {np.count_nonzero(model.modesfit_ == i)} points.")
# print(gen_full_matlab_hyeq_class(model, name="DCDC_Melco2"))

# exit()
## plot mode jumps with time
n_plts = 5
f1, axs = plt.subplots(n_plts)
f2, axs2 = plt.subplots(n_plts)

n_points = model.modesfit_.size
n_start = n_points * 0 // 100
n_end = n_points * 100 // 100

X = model.Xfit_
Xdot = model.X_dot_fit_
Time = model.Time_fit_

time_plt = Time[n_start:n_end]
modes_plt = model.modesfit_[n_start:n_end]
X1 = X.T[0][n_start:n_end]
X2 = X.T[1][n_start:n_end]
X3 = X.T[2][n_start:n_end]
X4 = X.T[3][n_start:n_end]

# ## plot X
for j in range(model.n):
    axs[0].scatter(time_plt[modes_plt==j], modes_plt[modes_plt==j]+1, s=1, color=plt.cm.tab10(j))
    axs[1].scatter(time_plt[modes_plt==j], X1[modes_plt==j], s=1, color=plt.cm.tab10(j))
    axs[2].scatter(time_plt[modes_plt==j], X2[modes_plt==j], s=1, color=plt.cm.tab10(j))
    axs[3].scatter(time_plt[modes_plt==j], X3[modes_plt==j], s=1, color=plt.cm.tab10(j))
    axs[4].scatter(time_plt[modes_plt==j], X4[modes_plt==j], s=1, color=plt.cm.tab10(j))

# ## plot Xdot 
axs2[0].scatter(Time[n_start:n_end], model.modesfit_[n_start:n_end]+1, s=1)
axs2[1].scatter(Time[n_start:n_end], Xdot.T[0][n_start:n_end], s=1, label='estimated')
axs2[2].scatter(Time[n_start:n_end], Xdot.T[1][n_start:n_end], s=1, label='estimated')
axs2[1].scatter(Time[n_start:n_end], X_dot_predict.T[0][n_start:n_end], s=1, label='predicted')
axs2[2].scatter(Time[n_start:n_end], X_dot_predict.T[1][n_start:n_end], s=1, label='predicted')
axs2[3].scatter(Time[n_start:n_end], Xdot.T[2][n_start:n_end], s=1, label='estimated')
axs2[3].scatter(Time[n_start:n_end], X_dot_predict.T[2][n_start:n_end], s=1, label='predicted')
axs2[4].scatter(Time[n_start:n_end], Xdot.T[3][n_start:n_end], s=1, label='estimated')
axs2[4].scatter(Time[n_start:n_end], X_dot_predict.T[3][n_start:n_end], s=1, label='predicted')

## plot with segmentation information
# axs[0].scatter(Time, model.modesfit_+1, markersize=0.01)
# for s in range(len(seg_res)-1):
#     Time_seg = Time[seg_res[s]:seg_res[s+1]]
#     modes_seg = model.modesfit_[seg_res[s]:seg_res[s+1]]
#     X0_seg = X.T[0][seg_res[s]:seg_res[s+1]]
#     X1_seg = X.T[1][seg_res[s]:seg_res[s+1]]
#     # axs[0].plot(Time_seg, modes_seg+1, markersize=0.01, color='b')
#     axs[1].plot(Time_seg, X0_seg, markersize=0.01, color='b')
#     axs[2].plot(Time_seg, X1_seg, markersize=0.01, color='b')
for i in range(n_plts):
    axs[i].set_xlabel("t")
    axs2[i].set_xlabel("t")
    axs[i].minorticks_on()
    axs2[i].minorticks_on()
    axs[i].grid(axis='x', alpha=0.25, which='both')
    axs2[i].grid(axis='x', alpha=0.25, which='both')
    axs[i].ticklabel_format(style='sci', scilimits=(0,0))
    axs2[i].ticklabel_format(style='sci', scilimits=(0,0))
    
axs[0].set_ylabel("Mode")
axs[1].set_ylabel("$I_{in}$")
axs[2].set_ylabel("$V_{out}$")
axs[3].set_ylabel("$duty$")
axs[4].set_ylabel("$PID\_intern$")

axs2[0].set_ylabel("Mode")
axs2[1].set_ylabel("$d(I_{in})/dt$")
axs2[2].set_ylabel("$d(V_{out})/dt$")
axs2[3].set_ylabel("$d(duty)/dt$")
axs2[4].set_ylabel("$d(PID\_intern)/dt$")

axs2[1].legend()
axs2[2].legend()
axs2[3].legend()
axs2[4].legend()


axs[0].yaxis.get_major_locator().set_params(integer=True)
axs2[0].yaxis.get_major_locator().set_params(integer=True)
axs[0].tick_params(axis='y', which='minor', left=False)
axs2[0].tick_params(axis='y', which='minor', left=False)


title = ""
for mode in range(model.n):
    mode_alphas = model.regressors_[mode].alpha_
    title += f"$\\alpha_{mode+1}=["
    for alpha in mode_alphas:
        title +=  f"{alpha}, "
    title = title[:-2] + "]$ "
    if mode % 2 == 1:
        title += "\n"
title = title[:-1]

print(title)
# f1.suptitle(title)
# f2.suptitle(title)
# f3, a3 = plot_with_modes_2D(model.Xfit_, model.modesfit_+1)
# plt.show()


# print(model.Xfit_.shape)
# print(model.X_dot_fit_.shape)
# print(model.modesfit_.shape)
# print(model.X_dot_fit_[model.modesfit_==2])


# draw_trajectories_3D(trajs)

plt.show()
exit()

# print(np.diff(seg_res))

# print(gen_full_matlab_hyeq_class(model, name="DCDC_Melco").replace(".", ","))
print(X_init[0])
print(t[0], t[-1])



mode_switch_ind = np.where(np.diff(model.modesfit_) != 0)[0] + 1
time_switch = Time[mode_switch_ind]
# print(np.diff(time_switch))





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