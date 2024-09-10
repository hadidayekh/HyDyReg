from math import floor, sqrt
import sys
import time

# setting path to access parent directory files
sys.path.append('./')
sys.path.append('../')

from snds import SNDS_affine
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
 

compare_with_inferDynamic = False
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
    nmodes_list = [2,3,4,5]
else:
    nmodes_list = [int(sys.argv[1])]

if len(sys.argv) > 2 and sys.argv[2] == 'f':
    learn = False
else:
    learn = True


def parametric(nmodes, radius=1.0):
    assert nmodes > 1

    radii = radius/nmodes

    def labelfun(x, nmodes=nmodes):
        d = x[0]**2 + x[1]**2 + x[2]**2
        d = sqrt(d)
        label = floor(d/radii)
        return (nmodes-1)-label if label < nmodes else 0

    eventlist = []
    params = []
    for j in range(1,nmodes):
        def event(t, x, k=j):
            return x[0]**2 + x[1]**2 + x[2]**2 - (radii*k)**2
        event.direction = 0
        event.terminal = True
        eventlist.append(event)

    for j in range(1, nmodes+1):
        beta = -2*j if j % 2 == 0 else 2*j
        k = 1
        lmbd=1
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

        params.append([[A[0,0], A[0,1], 0, 0],
                [A[1,0], A[1,1], 0, 0],
                [0, 0, -lmbd, 0]])

    params = np.array(params)
    ##### DEFINING SNDS #####
    sul = SNDS_affine(params, eventlist, labelfun)
    return sul

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters

DATA = []
for nmodes in nmodes_list:
    radius = 1.0
    radii = radius/nmodes
    sul = parametric(nmodes, radius)

    x0s = np.array([[-1.0, -1], [1,-1], [-1,1]])
    # x0s *= 1.5/sqrt(2)/nmodes
    x0s = np.random.random((17, 3))*2 - 1
    x0s_test = np.random.random((5, 2))*4 - 2

    time_step = 0.001
    time_range = 2
    time_step_test = 0.001
    time_range_test = 1

    eq_query_bounds = np.array([[-1.0,1.0], [-1.0,1.0], [-1.0,1.0]])

    c = LogisticRegression(penalty='none', multi_class='multinomial', max_iter=100000)
    # c = SVC(C=1e10, kernel='poly', degree=2, gamma=1)

    if not learn:
        # Simulating trajectories
        trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
        segments = trajectories.segment_trajectories(0.1)

        # Plotting results
        plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)

        fig1, ax1 = plotting.draw_trajectories_3D(trajectories, mark_beginning=True, segments=segments)

        # f1, a1 = plt.subplots()
        # # c = ['r', 'b', 'g', 'c', 'm']
        # c = [plt.cm.Set1(i) for i in range(20)]
        # done = [False] * nmodes
        # for tr in trajectories.x_list:
        #     for i in range(len(tr)):
        #         x = tr[i]
        #         l = sul.labelfun(x)
        #         a1.scatter(x[0], x[1], s=2, color=c[l], label=f"Mode {l+1}" if not done[l] else None)
        #         done[l] = True


        # x0 = np.linspace(-radius,radius,100)
        # x1 = np.linspace(-radius,radius,100)
        # xx, yy = np.meshgrid(x0,x1)

        # for j in range(1, nmodes):
        #     zz = xx**2 + yy**2 - (radii*j)**2
        #     a1.contour(xx, yy, zz, levels=[0], colors=['k'], linewidths=0.5)

        # a1.legend()
        # a1.set_xlabel('$x^{(1)}$')
        # a1.set_ylabel('$x^{(2)}$')

    else:
        sul, learned_SNDS, trajectories, traj_test_sul, data = learn_system(exp_title + "-" + str(nmodes), sul, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes=2, poly_degree=1, random_test_set=True, classifier_poly_trans=True, classifier_poly_deg=2, seg_tol=0.1, diff_method=diff_method_fps, train_coefs=True, with_plots=True, print_on=False, classifier = c, C_space = [0, 0.001, 0.01, 0.1, 1], write_on=True, deltas=0.001, tol=1e-5, do_eq_query=True, eq_query_bounds=eq_query_bounds, eq_query_ntrajs=50, eq_query_tol=0.1, sampling_method='random', break_at_low_MSE=False, plot_xt=False, improve_cont=True)

        if compare_with_inferDynamic:
            if nmodes == 2 or nmodes == 3:

                methods = ['tolmerge', 'piecelinear']

                for method in methods:
                    mergeps = [0.01]
                    if nmodes == 2:
                        for mergeep in mergeps:
                            start = time.time()
                            P, G, boundary = infer_model(
                                        trajectories.t_list, trajectories.x_list, stepsize=time_step, maxorder=1, boundary_order=2,
                                        num_mode=nmodes, modelist=None, event=None, ep=0.01, mergeep = mergeep, method=method, verbose=False)
                            infer_time = time.time() - start
                            modelist = sul.eqs
                            event = sul.eventlist[0]
                            res = test_model(P, G, boundary, nmodes, traj_test_sul.x_list, sul.eqs, sul.eventlist[0], 1, 2)
                            data.append(res)
                            data.append(infer_time)
                    else:
                        for mergeep in mergeps:
                            start = time.time()
                            P, G, boundary = infer_model(
                                        trajectories.t_list, trajectories.x_list, stepsize=time_step, maxorder=1, boundary_order=2,
                                        num_mode=nmodes, modelist=None, event=None, ep=0.01, mergeep = mergeep, method=method, verbose=False, labeltest=sul.labelfun)
                            infer_time = time.time() - start
                            res = test_model(P, G, boundary, nmodes, traj_test_sul.x_list, sul.eqs, sul.eventlist, 1, 2, labeltest=sul.labelfun)
                            data.append(res)
                            data.append(infer_time)
            else:
                data.extend([-1,-1,-1,-1])
            
            DATA.append(data)
# print_experi_data(DATA)

plt.show()