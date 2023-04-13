import sys
from sklearn.linear_model import LogisticRegression
import warnings

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
import dill as pickle
from sklearn.svm import SVC
import textwrap

exp_title = "poly3_2modes"
k_learn = True
save_plots = True
save_pickle = True
load_pickle = False
seg_tol = 0.5

'''
Model of the shape:

x_dot_0 = a00 * x0 + a01 * x1 + a03
x_dot_1 = a10 * x0 + a11 * x1 + a13

a, b, c (3 modes)
'''

# Label function to decide which mode
def labelfun(x):
    if x[1] + x[0]/12 - 0.5 >= 0:
        return 0
    else:
        return 1


# Events to be tracked by the IVP solver
event = lambda t, x : x[1] + x[0]/12 - 0.5
event.terminal = True
event.direction = 0

eventlist = [event]

a1, a2, a3 =  0.1, 0.04, -0.9
b1, b2, b3 = -0.1, 0.06, -0.7

def eq1(t, y):
    return [ a1 * y[0]**2 + a2 * y[1]**3,
            a3 * y[0]]

def eq2(t, y):
    return [ b1 * y[0]**2 + b2 * y[1]**3,
            b3 * y[0]]

eqs = [eq1, eq2]


##### DEFINING SNDS #####
sul = SNDS(2, 2, eqs, eventlist, labelfun)

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters
time_step = 0.05
time_range = 10
x0s = [[0,4], [0,4.5], [0,5]]

# Simulating or loading trajectories
if not load_pickle:
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    if save_pickle:
        file = open('trajectories/' + exp_title + '.trajectories', 'wb')
        pickle.dump(trajectories, file)
        file.close()
else:
    file = open('trajectories/' + exp_title + '.trajectories', 'rb')
    trajectories = pickle.load(file)
    file.close()


# Segmenting
segments = trajectories.segment_trajectories(seg_tol)

# Plotting results
plot_title = exp_title + ' trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)
fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, plot_xt=True, segments=segments, labelfun=labelfun)
ax1['main'].set_title(plot_title)

# Saving figure into a file
if save_plots:
    file_name = "plots/" + exp_title + "/SIM_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
    fig1.savefig(file_name)

t_arr = np.array(trajectories.t_list)
x_arr = np.array(trajectories.x_list)
X, X_dot, seg_res = diff_method_fps(t_arr, x_arr, segments=segments, remove_seg=True)
#X_dot = np.array([eqs[abs(labelfun(x)-1)](0, x) for x in X])

X0 = []
X0_dot = []
X1 = []
X1_dot = []
for i in range(X.shape[0]):
    x = X[i]
    xdot = X_dot[i]
    if labelfun(x) == 0:
        X0.append(x)
        X0_dot.append(xdot)
    else:
        X1.append(x)
        X1_dot.append(xdot)

X0 = np.array(X0)
X1 = np.array(X1)
X0_dot = np.array(X0_dot)
X1_dot = np.array(X1_dot)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree=3)
X0 = poly.fit_transform(X0)
X1 = PolynomialFeatures(degree=3).fit_transform(X1)

model00 = LinearRegression().fit(X0, X0_dot.T[0])
model01 = LinearRegression().fit(X0, X0_dot.T[1])
model10 = LinearRegression().fit(X1, X1_dot.T[0])
model11 = LinearRegression().fit(X1, X1_dot.T[1])

print(poly.get_feature_names_out())
print("Mode 0")
print(model00.coef_[3], model00.coef_[-1])
print(model01.coef_[1])
print("Mode 1")
print(model10.coef_[3], model10.coef_[-1])
print(model11.coef_[1])




if k_learn:
    #### LEARNING THE SNDS ####

    # Extracting derivatives through numerical differentiation
    t_arr = np.array(trajectories.t_list)
    x_arr = np.array(trajectories.x_list)
    X, X_dot, seg_res = diff_method_fps(t_arr, x_arr, segments=segments, remove_seg=True)

    if load_pickle:
        file = open('models/' + exp_title + '.model', 'rb')
        model = pickle.load(file)
        file.close()
    else:
        # Defining the model
        model = HybridKernelRegressor(train_coefs=True, n=2, deg=3)#, kernels_list=['polynomial', 'polynomial'], kargs_list=[[3, 1, 1], [3, 1, 1]])

        # Fitting the model with its corresponding derivatives
        model.fit(X, X_dot, norm='L2', deltas=0.001, C=10000, segmentation='strict', segments=seg_res)

        if save_pickle:
            file = open('models/' + exp_title + '.model', 'wb')
            pickle.dump(model, file)
            file.close()

    # Classify modes
    #model.classify_modes(SVC(kernel='linear', C=0.1))
    model.classify_modes(LogisticRegression(penalty='none'))
    fine_X, fine_X_dot = sul.generate_points_between_segments(trajectories, segments, seg_tol/10)
    model.improve_classification(fine_X, fine_X_dot, LogisticRegression(penalty='none'))

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

    if modes[0] == 0:
        labelfun_u = labelfun
    else:
        labelfun_u = lambda x : 0 if labelfun(x) == 1 else 1
    learned_SNDS = model.to_SNDS()#labelfun=labelfun_u, eventlist=eventlist)

    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)


    try:
        fig2, ax2 = plotting.draw_trajectories_2D(trajectories, label='Original trajectories', colors=['b'], mark_beginning=True, plot_xt=True)
        fig2, ax2 = plotting.draw_trajectories_2D(trajs_learned, fig=fig2, axs=ax2, colors=['m'], label='Learned trajectories', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
        ax2['main'].minorticks_on()
        ax2['main'].grid(which='both', lw=0.1)
        ax2['main'].set_title(textwrap.fill('Training data versus trajectories simulated from the learning model with the same intial conditions as the training data, as well as the mode state space partitioning.', width=60))
    except:
        warnings.warn("Failed to plot trajectories.")



    
    fig3, ax3 = plotting.plot_with_modes_2D(model.X_classify_, model.modes_classify_, labelfun=model.labelfun_)
    ax3.set_title(textwrap.fill("Training data and their corresponding learned modes, as well as the result of classification.", width=60))

    if save_plots:
        try:
            file_name = "plots/" + exp_title + "/SIM-LEARN_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
            fig2.savefig(file_name)
        except:
            pass

        file_name = "plots/" + exp_title + "/SIM-MODES_" + exp_title + "_trajs=" + str(len(x0s)) + '_ts=' + str(time_step) + '_T=' + str(time_range) + ".pdf"
        fig3.savefig(file_name)

    plt.show()
