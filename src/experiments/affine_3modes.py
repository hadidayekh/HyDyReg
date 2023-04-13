import sys
from sklearn.linear_model import LogisticRegression

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
import dill as pickle
from sklearn.svm import SVC
import textwrap

exp_title = "affine_3modes"
k_learn = True
save_plots = True
save_pickle = False
load_pickle = True
seg_tol = 0.5

'''
Model of the shape:

x_dot_0 = a00 * x0 + a01 * x1 + a03
x_dot_1 = a10 * x0 + a11 * x1 + a13

a, b, c (3 modes)
'''

a00, a01, a03 =  -2, -2, -1
a10, a11, a13 =  4, -1, 0

b00, b01, b03 =  -3, 2, 0
b10, b11, b13 =  -4, -1, 0

if (b00+b11)**2 < 4*(b00*b11-b01*b10):
    if (b00+b11) >= 0:
        print('repelling')
    else:
        print('stable')
else:
    exit()

c00, c01, c03 =  0.1, 0.1, 1
c10, c11, c13 =  -0.3, 0.2, -2

r = 1

# Label function to decide which mode
def labelfun(x):
    if x[0] >= 1:
        return 1
    elif x[1] <= 2:
        return 0
    else:
        return 2


# Events to be tracked by the IVP solver
event1 = lambda t, x : x[0] - 1
event1.terminal = True
event1.direction = 0

event2 = lambda t, x : x[1] - 2
event2.terminal = True
event2.direction = 0

eventlist = [event2, event1]



params = [[[a00, a01, a03],
           [a10, a11, a13]],
          [[b00, b01, b03],
           [b10, b11, b13]],
          [[c00, c01, c03],
           [c10, c11, c13]]]

params = np.array(params)


##### DEFINING SNDS #####
sul = SNDS_linear(params, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters
x0s = [[-2, 4], [1.5, 10], [0,5]]#, [2,12], [-4, 10]]
time_step = 0.02
time_range = 4

# Simulating or loading trajectories
if not load_pickle:
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    if save_pickle:
        file = open('../trajectories/' + exp_title + '.trajectories', 'wb')
        pickle.dump(trajectories, file)
        file.close()
else:
    file = open('../trajectories/' + exp_title + '.trajectories', 'rb')
    trajectories = pickle.load(file)
    file.close()

print(trajectories.x_list[0].shape)
print(trajectories.t_list[0].shape)
# Segmenting
segments = trajectories.segment_trajectories(seg_tol)

segments2 = []
for s in segments:
    newS = [s[0]]
    for i in range(len(s)-1):
        mid = int((s[i+1] - s[i])/2)
        newS.append(mid)
        newS.append(s[i+1])
    segments2.append(newS)

segments2=segments
# Plotting results
plot_title = exp_title + ' trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)
ax1['main'].set_title(plot_title)

# Saving figure into a file
if save_plots:
    file_name = "../plots/" + exp_title + "/SIM_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
    fig1.savefig(file_name)

if k_learn:
    #### LEARNING THE SNDS ####

    # Extracting derivatives through numerical differentiation
    t_arr = np.array(trajectories.t_list)
    x_arr = np.array(trajectories.x_list)
    X, X_dot, seg_res = diff_method_fps(t_arr, x_arr, segments=segments2, remove_seg=True)
    
    if load_pickle:
        file = open('../models/' + exp_title + '.model', 'rb')
        model = pickle.load(file)
        file.close()
    else:
        # Defining the model
        model = HybridKernelRegressor(train_coefs=True, n=3, deg=1, kernels_list=['linear', 'linear', 'linear'], kargs_list=[[0], [0], [0]])

        # Fitting the model with its corresponding derivatives
        model.fit(X, X_dot, norm='L2', deltas=0.001, C=100, segmentation='strict', segments=seg_res)

        if save_pickle:
            file = open('../models/' + exp_title + '.model', 'wb')
            pickle.dump(model, file)
            file.close()

    # Classify modes
    #model.classify_modes(SVC(kernel='rbf', C=1))
    model.classify_modes(LogisticRegression(penalty='none', multi_class='ovr'))
    fine_X, fine_X_dot = sul.generate_points_between_segments(trajectories, segments, seg_tol/10)
    #ax1['main'].scatter(fine_X.T[0], fine_X.T[1], linewidths=0.1, c='k')
    model.improve_classification(fine_X, fine_X_dot, classifier=LogisticRegression(penalty='none', multi_class='multinomial'))

    # Comparing modes to each other
    modes = model.get_modes()
    modes_ori = [labelfun(x) for x in X]
    
    
    print("After loading")
    # Predicting X_dot from the model and comparing
    X_dot_pred = model.predict(X)
    
    MSE1 = mean_squared_error(X_dot, X_dot_pred)
    print("MSE:", MSE1)

    # This is a linear model so printing the equations
    model.print_coefs_linear()
    print(modes)

    #### TRANSFORMING INTO SNDS AND SIMULATING NEW TRAJECTORIES THEN PLOTTING RESULTS ####

    learned_SNDS = model.to_SNDS()#labelfun=labelfun, eventlist=eventlist)

    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)

    try:
        fig2, ax2 = plotting.draw_trajectories_2D(trajectories, label='Original trajectories', colors=['b'], mark_beginning=True, plot_xt=True)
        fig2, ax2 = plotting.draw_trajectories_2D(trajs_learned, fig=fig2, axs=ax2, colors=['m'], label='Learned trajectories', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
        ax2['main'].minorticks_on()
        #ax2['main'].grid(which='both', lw=0.1)
        ax2['main'].set_title(textwrap.fill('Training data versus trajectories simulated from the learning model with the same intial conditions as the training data, as well as the mode state space partitioning.', width=60))
    except:
        print("Failed to plot trajs.")



    
    fig3, ax3 = plotting.plot_with_modes_2D(model.X_classify_, model.modes_classify_, labelfun=model.labelfun_)
    ax3.set_title(textwrap.fill("Training data and their corresponding learned modes, as well as the result of classification.", width=60))

    if save_plots:
        try:
            file_name = "../plots/" + exp_title + "/SIM-LEARN_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
            fig2.savefig(file_name)
        except:
            pass

        file_name = "../plots/" + exp_title + "/SIM-MODES_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig3.savefig(file_name)

    plt.show()
