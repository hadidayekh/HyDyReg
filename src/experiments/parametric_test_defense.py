from math import floor, sqrt
import pickle
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
 
if __name__ == "__main__":

    exp_title = "parametric"
    save_plots = True
    seg_tol = 0.1
    optimization_based = False
    improve_continuous = True


    nmodes_list = [3]

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
        ##### DEFINING SNDS #####

        return params, eventlist, labelfun


    for nmodes in nmodes_list:
        np.random.seed(1)

        radius = 1.0
        radii = radius/nmodes
        params, eventlist, labelfun = parametric(nmodes, radius)
        sul = SNDS_affine(params, eventlist, labelfun)
        print(sul.gen_latex_cont_eq())

        x0s = np.array([[-1.0, -1], [1,-1], [-1,1]])


        time_step = 0.01
        time_range = 1

        eq_query_bounds = np.array([[-1.0,1.0], [-1.0,1.0]])
        c = [plt.cm.Set1(i) for i in range(20)]
        for i in range(nmodes):
            sul_ind = SNDS_affine(params, [], lambda x, i=i : i)
            trajs_ind = sul_ind.simulate_trajectory(time_step, time_range, x0s)
            fig0, ax0 = plotting.draw_trajectories_2D(trajs_ind, mark_beginning=True, colors=[c[i]])

        trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
        fig0, ax0 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, colors=['k'])

        segments = trajectories.segment_trajectories(0.1)

        # Plotting results
        plot_title = exp_title + 'trajs=' + str(len(x0s)) + ' ts=' + str(time_step) + ' T=' + str(time_range)

        fig1, ax1 = plotting.draw_trajectories_2D(trajectories, mark_beginning=True, segments=segments, colors=['k'])

        f1, a1 = plt.subplots()
        # c = ['r', 'b', 'g', 'c', 'm']
        trajectories = trajectories.return_unsegmented()
        done = [False] * nmodes
        for tr in trajectories.x_list:
            for i in range(len(tr)):
                x = tr[i]
                l = sul.labelfun(x)
                a1.scatter(x[0], x[1], s=2, color=c[l], label=f"Mode {l+1}" if not done[l] else None)
                done[l] = True

        

        x0 = np.linspace(-radius,radius,100)
        x1 = np.linspace(-radius,radius,100)
        xx, yy = np.meshgrid(x0,x1)

        for j in range(1, nmodes):
            zz = xx**2 + yy**2 - (radii*j)**2
            a1.contour(xx, yy, zz, levels=[0], colors=['k'], linewidths=0.5)

        a1.legend()
        a1.set_xlabel('$x^{(1)}$')
        a1.set_ylabel('$x^{(2)}$')


        done = [False] * nmodes
        f2, a2 = plt.subplots()
        for tr in trajectories.x_list:
            for i in range(len(tr)):
                x = tr[i]
                l = sul.labelfun(x)
                a2.scatter(x[0]**2, x[1]**2, s=2, color=c[l], label=f"Mode {l+1}" if not done[l] else None)
                done[l] = True
        x0 = np.linspace(-radius**2,radius**2,100)
        x1 = np.linspace(-radius**2,radius**2,100)
        xx, yy = np.meshgrid(x0,x1)

        for j in range(1, nmodes):
            zz = xx + yy - (radii*j)**2
            a2.contour(xx, yy, zz, levels=[0], colors=['k'], linewidths=0.5)

        a2.legend()
        a2.set_xlabel('$x^{(1)2}$')
        a2.set_ylabel('$x^{(2)2}$')
        a2.set_xlim(-0.1, 1)
        a2.set_ylim(-0.1, 1)

    plt.show()