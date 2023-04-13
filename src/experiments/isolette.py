import sys

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from sklearn.metrics import mean_squared_error
from opti import HybridKernelRegressor
from diff_methods import diff_method_fps
from snds import SNDS_linear
import plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

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
sul = SNDS_linear(params, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters
x0s = [[105.5, 80], [97.5, 100]]
time_step = 0.1
time_range = 50

# Simulating trajectories
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)

# Segmenting
segments = trajectories.segment_trajectories(seg_tol)

# Plotting results
plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)

# Saving figure into a file
if save_plots:
    file_name = "plots/" + exp_title + "/SIM_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
    fig1.savefig(file_name)

#### LEARNING THE SNDS ####
if k_learn:
    # Defining the model
    model = HybridKernelRegressor(train_coefs=True, n=2, deg=1, kernels_list=['linear', 'linear'], kargs_list=[[0], [0]])

    # Extracting derivatives through numerical differentiation
    t_arr = np.array(trajectories.t_list)
    x_arr = np.array(trajectories.x_list)
    X, X_dot, seg_res = diff_method_fps(t_arr, x_arr, segments=segments, remove_seg=True)

    # Fitting the model with its corresponding derivatives
    model.fit(X, X_dot, deltas=0.0001, C=100, segmentation='strict', segments=seg_res)
    
    # Comparing modes to each other
    modes = model.get_modes()
    modes_ori = [labelfun(x) for x in X]
    
    print(modes - modes_ori)
    # Flipping labelfun to be given to the simulator in case model assigns modes inversely
    # ONLY WORKS WITH SYSTEMS WITH n = 2
    labelfun_inverse = lambda x : 0 if labelfun(x) == 1 else 1
    labelfun_used = labelfun if modes[0] == 0 else labelfun_inverse

    # Predicting X_dot from the model and comparing
    X_dot_pred = model.predict(X, labelfun=labelfun)
    X_dot_pred_inv = model.predict(X, labelfun=labelfun_inverse)
    MSE1 = mean_squared_error(X_dot, X_dot_pred)
    MSE2 = mean_squared_error(X_dot, X_dot_pred_inv)

    if MSE1 < MSE2:
        labelfun_used = labelfun
        print("MSE:", MSE1)
    else:
        labelfun_used = labelfun_inverse
        print("MSE:", MSE2)

    

    # This is a linear model so printing the equations
    model.print_coefs_linear()
    print(modes)

    #### TRANSFORMING INTO SNDS AND SIMULATING NEW TRAJECTORIES THEN PLOTTING RESULTS ####

    model.classify_modes()
    learned_SNDS = model.to_SNDS()#labelfun=labelfun_used, eventlist=eventlist)
    fine_X, fine_X_dot = sul.generate_points_between_segments(trajectories, segments, seg_tol/10)
    model.improve_classification(fine_X, fine_X_dot, classifier=LogisticRegression(penalty='none'))
    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)

    fig2, ax2 = plotting.draw_trajectories_2D(trajectories, label='Original trajectories', colors=['b'], mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
    fig2, ax2 = plotting.draw_trajectories_2D(trajs_learned, fig=fig2, axs=ax2, colors=['m'], label='Learned trajectories', mark_beginning=True, plot_xt=True)

    fig3, ax3 = plotting.plot_with_modes_2D(model.X_classify_, model.modes_classify_, labelfun=model.labelfun_)

    if save_plots:
        file_name = "plots/" + exp_title + "/SIM-LEARN_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig2.savefig(file_name)

        file_name = "plots/" + exp_title + "/SIM-MODES_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig3.savefig(file_name)
plt.show()