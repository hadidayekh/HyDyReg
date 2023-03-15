import sys

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from sklearn.metrics import mean_squared_error
from opti import HybridKernelRegressor
from diff_methods import diff_method_fps
from snds import SNDS
import plotting
import numpy as np
import matplotlib.pyplot as plt

exp_title = "poly2_3modes"
k_learn = False ## KERNEL LEARNING IS STILL NOT WORKING FOR NONLINEAR SYS
save_plots = True
seg_tol = 0.1


# Label function to decide which mode
def labelfun(x):
    x0, x1 = x
    if -x1 >= 0:
        return 0
    elif -x0 >= 0:
        return 1
    else:
        return 2


# Events to be tracked by the IVP solver
event1 = lambda t, x : x[1]
event1.terminal = True
event1.direction = 0
event2 = lambda t, x : x[0]
event2.terminal = True
event2.direction = 0

eventlist = [event1, event2]

# Defining equations
eq0 = lambda t, x: [-7, -x[0]]
eq1 = lambda t, x: [0.5*x[0]**2 + 0.5*x[1], -9*x[0] + 3]
eq2 = lambda t, x: [5, -0.1*x[0] - 10]
eqs = [eq0, eq1, eq2]

##### DEFINING SNDS #####
sul = SNDS(3, 2, eqs, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters
x0s = [[-1,1],[2,-3]]#, [1,4]]
time_step = 0.1
time_range = 5

# Simulating trajectories
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)

# Segmenting
segments = trajectories.segment_trajectories(seg_tol)

# Plotting results
plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)

# Saving figure into a file
if save_plots:
    file_name = "plots/SIM_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
    fig1.savefig(file_name)

#### LEARNING THE SNDS ####
if k_learn:
    # Defining the model
    model = HybridKernelRegressor(['linear', 'polynomial', 'linear'], [[0], [2, 1, 1], [0]])

    # Extracting derivatives through numerical differentiation
    t_arr = np.array(trajectories.t_list)
    x_arr = np.array(trajectories.x_list)
    X, X_dot, seg_res = diff_method_fps(t_arr, x_arr, segments=segments)

    # Fitting the model with its corresponding derivatives
    model.fit(X, X_dot, deltas=0.0001, C=100, segmentation='strict', segments=seg_res)
    
    # Comparing modes to each other
    modes = model.get_modes()
    modes_ori = [labelfun(x) for x in X]
    
    print(modes - modes_ori)

    # Predicting X_dot from the model and comparing
    X_dot_pred = model.predict(X, labelfun=labelfun)
    MSE1 = mean_squared_error(X_dot, X_dot_pred)
    

    print(modes)

    #### TRANSFORMING INTO SNDS AND SIMULATING NEW TRAJECTORIES THEN PLOTTING RESULTS ####

    learned_SNDS = model.to_SNDS(labelfun=labelfun, eventlist=eventlist)

    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)

    fig2, ax2 = plotting.draw_trajectories_2D(trajectories, label='Original trajectories', colors=['b'], mark_beginning=True, plot_xt=True, labelfun=labelfun)
    try:
        fig2, ax2 = plotting.draw_trajectories_2D(trajs_learned, fig=fig2, axs=ax2, colors=['m'], label='Learned trajectories', mark_beginning=True, plot_xt=True)
    except:
        print("Failed to simulate new trajs")
    fig3, ax3 = plotting.plot_with_modes_2D(model.Xfit_, modes, labelfun=labelfun)

    if save_plots:
        file_name = "plots/SIM-LEARN_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig2.savefig(file_name)

        file_name = "plots/SIM-MODES_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig3.savefig(file_name)
