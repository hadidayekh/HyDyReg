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

exp_title = "parametric"
k_learn = True
save_plots = False
seg_tol = 0.1


def parametric(nmodes):
    assert nmodes > 1

    radii = 1.0/nmodes

    def labelfun(x, nmodes=nmodes):
        d = x[0]**2 + x[1]**2
        d = sqrt(d)
        label = floor(d/radii)
        return label if label < nmodes else nmodes-1


    eventlist = []
    params = []
    for j in range(1,nmodes):
        def event(t, x, k=j):
            return x[0]**2 + x[1]**2 - (radii*k)**2
        event.direction = 0
        event.terminal = True
        eventlist.append(event)

    for j in range(1, nmodes+1):
        beta = -1 if j % 2 == 1 else -1
        k = 2

        Aorg = np.array([[0, k*beta], [-beta,  0]])
        D, V = np.linalg.eig(Aorg)
        D = np.diag(D)

        epsilon_1 = -2*j
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
    ##### DEFINING SNDS #####
    sul = SNDS_linear(params, eventlist, labelfun)
    return sul

##### SIMULATING TRAJECTORIES #####
# Setting initial conditions and simulation paramters

nmodes = 4
radii = 1.0/nmodes
sul = parametric(nmodes)
learn = True


x0s = np.array([[-1, -1], [-1, 1], [1,-1], [1,1]])
x0s_test = np.random.random((5, 2))*4 - 2

time_step = 0.1
time_range = 3
time_step_test = 0.05
time_range_test = 1

eq_query_bounds = np.array([[-1,1], [-1,1]])

c = LogisticRegression(penalty='none', multi_class='multinomial')
# c = SVC(C=1e10, kernel='poly', degree=2, gamma=1)

if not learn:
    # Simulating trajectories
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    segments = trajectories.segment_trajectories(0.1)

    # Plotting results
    plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)

    fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, segments=segments)

    x0 = np.linspace(-1,1,100)
    x1 = np.linspace(-1,1,100)
    xx, yy = np.meshgrid(x0,x1)

    for j in range(1, nmodes):
        zz = xx**2 + yy**2 - (radii*j)**2
        ax1.contour(xx, yy, zz, levels=[0], colors=['k'])

else:
    learn_system(exp_title, sul, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes, 1, classifier_poly_trans=True, classifier_poly_deg=2, seg_tol=0.1, diff_method=diff_method_fps, train_coefs=True, with_plots=True, print_on=True, classifier = c, C_space = [0, 0.001, 0.01, 0.1, 1], write_on=False, deltas=0.001, tol=1e-5, do_eq_query=True, eq_query_bounds=eq_query_bounds, eq_query_ntrajs=50, eq_query_tol=0.1, sampling_method='random', break_at_low_MSE=False)

plt.show()