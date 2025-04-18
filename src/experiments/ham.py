import sys
path_to_InferDynamic = ""
# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')
sys.path.insert(0, path_to_InferDynamic)
#from infer_multi_ch import infer_model, test_model
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
 
from sklearn.metrics import mean_squared_error
from opti import HybridRegressor
from diff_methods import diff_method_fps
from snds import SNDS 
import plotting
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

exp_title = "ham"
k_learn = True
save_plots = False
seg_tol = 0.15


def labelfun(x):
    if x[0]**2+x[1]**2>=0.51:
        return 0
    else:
        return 1
event = lambda t, x : x[0]**2+x[1]**2 - 0.51
event.terminal = True
event.direction = 0
eventlist = [event]

def eq1(t, x):
    return [x[1], -x[0]**3+x[0]]
def eq2(t, x):
    return [-x[0]+x[1], -x[0]-x[1]]


##### DEFINING SNDS #####
sul = SNDS(2, 2, [eq1, eq2], eventlist, labelfun)


##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramtersoyt
x0s = [[1, 1], [0.9, -1.1], [-.9,0.3], [0.4,0.6], [0.1,0.1], [0.3, 0.1]]
time_step = 0.1
time_range = 5

# Simulating trajectories
trajectories = sul.simulate_trajectory(time_step, time_range, x0s)

sul.get_derivatives(trajectories)
exit()
total_win, total_d_avg, total_time = dict(), dict(), dict()
methods = ['kmeans', 'dbscan', 'merge', 'piecelinear', 'tolmerge', 'piecelinear1']
for method in methods:
    total_win[method] = 0
    total_d_avg[method] = 0.0
    total_time[method] = 0.0
np.random.seed(0)

for method in methods:
    print(method)
    print()
    P, G, boundary = infer_model(
                trajectories.t_list, trajectories.x_list, stepsize=time_step, maxorder=3, boundary_order=2,
                num_mode=2, modelist=None, event=None, ep=0.01, mergeep = 0.01, method=method, verbose=False)
    print([b/boundary[0] for b in boundary])
    print(test_model(P, G, boundary, 2, trajectories.x_list, [eq1, eq2], event, 3, 2))
    

# Segmenting
segments = trajectories.segment_trajectories(seg_tol)

# Plotting results
plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)
plt.show()
exit()
# Saving figure into a file
if save_plots:
    file_name = "../plots/" + exp_title + "/SIM_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
    fig1.savefig(file_name)

#### LEARNING THE SNDS ####
if k_learn:
    # Defining the model
    model = HybridRegressor(train_coefs=True, n=2, deg=3)

    # Extracting derivatives through numerical differentiation
    t_arr = np.array(trajectories.t_list)
    x_arr = np.array(trajectories.x_list)
    X, X_dot, seg_res, _ = diff_method_fps(t_arr, x_arr, segments=segments, remove_seg=True)

    # Fitting the model with its corresponding derivatives
    model.fit_opti(X, X_dot, deltas=0.0001, C=100, segmentation='strict', segments=seg_res)
    
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
        file_name = "../plots/" + exp_title + "/SIM-LEARN_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig2.savefig(file_name)

        file_name = "../plots/" + exp_title + "/SIM-MODES_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig3.savefig(file_name)
plt.show()