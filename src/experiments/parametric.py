from math import floor, sqrt
import sys

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from snds import SNDS_linear
import plotting
import numpy as np
import matplotlib.pyplot as plt
from diff_methods import diff_method_fps
from main import learn_system
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from utils import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # setting ignore as a parameter and further adding category
 

compare_with_inferDynamic = True
if compare_with_inferDynamic:
    try:
        f = open("pathtoinferdynamic.txt", 'r')
        path_to_InferDynamic = f.read()
        sys.path.insert(0, path_to_InferDynamic)
        from infer_multi_ch import infer_model, test_model
    except:
        print("Cannot open path to infer dynamic file.")
        compare_with_inferDynamic = False


exp_title = "parametric"
save_plots = False
seg_tol = 0.1

np.random.seed(1)

if len(sys.argv) < 2:
    print("Please provide an input of the number of modes as a parameter to the program.")
    exit(-1)

def parametric(nmodes, radius=1.0):
    assert nmodes > 1

    radii = radius/nmodes

    def labelfun(x, nmodes=nmodes):
        d = x[0]**2 + x[1]**2
        d = sqrt(d)
        label = floor(d/radii)
        return (nmodes-1)-label if label < nmodes else 0

    eventlist = []
    params = []
    for j in range(1,nmodes):
        def event(t, x, k=j):
            return x[0]**2 + x[1]**2 - (radii*k)**2
        event.direction = 0
        event.terminal = True
        eventlist.append(event)

    for j in range(1, nmodes+1):
        beta = -2*j if j % 2 == 0 else 2*j
        k = 1

        Aorg = np.array([[0, k*beta], [-beta,  0]])
        D, V = np.linalg.eig(Aorg)
        D = np.diag(D)

        epsilon_1 = -5
        epsilon_2 = 0
        
        E = np.array([[epsilon_1, 0], [0, epsilon_2]])
        
        #eigenvalues
        D1 = D + E
        
        A1 = V @ D1 @ np.linalg.inv(V)
        
        A = V @ D @ np.linalg.inv(V)

        A = A1.real

        params.append([[A[0,0], A[0,1], 0],
                [A[1,0], A[1,1], 0]])

    params = np.array(params)
    print(params)
    ##### DEFINING SNDS #####
    sul = SNDS_linear(params, eventlist, labelfun)
    return sul

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters

nmodes = int(sys.argv[1])
radius = 1.0
radii = radius/nmodes
sul = parametric(nmodes, radius)

if len(sys.argv) > 2 and sys.argv[2] == 'f':
    learn = False
else:
    learn = True


x0s = np.array([[-1.0, -1], [1, 1], [1,-1], [-1,1]])
x0s *= 1
x0s_test = np.random.random((5, 2))*4 - 2

time_step = 0.02
time_range = 1
time_step_test = 0.01
time_range_test = 1

eq_query_bounds = np.array([[-1.0,1.0], [-1.0,1.0]])

c = LogisticRegression(penalty='none', multi_class='multinomial')
# c = SVC(C=1e10, kernel='poly', degree=2, gamma=1)

if not learn:
    # Simulating trajectories
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    segments = trajectories.segment_trajectories(0.1)

    # Plotting results
    plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)

    fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, segments=segments)

    x0 = np.linspace(-radius,radius,100)
    x1 = np.linspace(-radius,radius,100)
    xx, yy = np.meshgrid(x0,x1)

    for j in range(1, nmodes):
        zz = xx**2 + yy**2 - (radii*j)**2
        ax1.contour(xx, yy, zz, levels=[0], colors=['k'])

else:
    sul, learned_SNDS, trajectories, traj_test_sul, traj_test_learned2 = learn_system(exp_title, sul, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes, 1, classifier_poly_trans=True, classifier_poly_deg=2, seg_tol=0.1, diff_method=diff_method_fps, train_coefs=True, with_plots=True, print_on=True, classifier = c, C_space = [0, 0.001, 0.01, 0.1, 1], write_on=False, deltas=0.001, tol=1e-5, do_eq_query=True, eq_query_bounds=eq_query_bounds, eq_query_ntrajs=50, eq_query_tol=0.1, sampling_method='random', break_at_low_MSE=True)

    X_dot_sul = sul.get_derivatives(traj_test_sul)
    X_dot_learned = learned_SNDS.get_derivatives(traj_test_sul)
    
    print("Test derivatives RD:", np.mean(relative_diff(X_dot_sul, X_dot_learned)))
    if compare_with_inferDynamic:
        if nmodes == 2 or nmodes == 3:

            methods = ['tolmerge', 'piecelinear']

            for method in methods:
                print()
                print(method)
                mergeps = [0.01]
                if nmodes == 2:
                    for mergeep in mergeps:
                        P, G, boundary = infer_model(
                                    trajectories.t_list, trajectories.x_list, stepsize=time_step, maxorder=1, boundary_order=2,
                                    num_mode=nmodes, modelist=None, event=None, ep=0.01, mergeep = mergeep, method=method, verbose=False)
                        modelist = sul.eqs
                        event = sul.eventlist[0]
                        print(test_model(P, G, boundary, nmodes, traj_test_sul.x_list, sul.eqs, sul.eventlist[0], 1, 2))
                else:
                    for mergeep in mergeps:
                        print(mergeep)
                        P, G, boundary = infer_model(
                                    trajectories.t_list, trajectories.x_list, stepsize=time_step, maxorder=1, boundary_order=2,
                                    num_mode=nmodes, modelist=None, event=None, ep=0.01, mergeep = mergeep, method=method, verbose=False, labeltest=sul.labelfun)
                        if nmodes != 4:
                            print(test_model(P, G, boundary, nmodes, traj_test_sul.x_list, sul.eqs, sul.eventlist, 1, 2, labeltest=sul.labelfun))

plt.show()