import kernels
import diff_methods
import numpy as np
import os
import copy
import textwrap
import time
from diff_methods import diff_method_fps
from opti import HybridKernelRegressor
from sklearn.linear_model import LogisticRegression
from plotting import *
from utils import *
import matplotlib.pyplot as plt
def get_modes(Chis_sol):
    return np.argmin(Chis_sol, axis=0)


def learn_system(exp_name, sul, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes, poly_degree, seg_tol=0.01, diff_method=diff_method_fps, train_coefs=True, kernels_list=None, kargs_list=None, with_plots=True, print_on=True, classifier = LogisticRegression(penalty='none'), C_space = [0.01, 0.1, 0.5, 1, 2, 5, 10], write_on=False, deltas=0.001, tol=1e-5, break_at_low_MSE=True, apply_LR=False):
    D = len(x0s[0])   
    if write_on:
        dir = "experi-info/" + exp_name + "/"
        dir2 = "experi-info/latex-files/"
        if not os.path.exists(dir):
            os.makedirs(dir)
        if not os.path.exists(dir2):
            os.makedirs(dir2)
        file = open(dir + exp_name + ".txt", 'w')
        file2 = open(dir2 + exp_name + "-results.tex", 'w')

        file2.write(f'Initial conditions: ')
        for i in range(len(x0s)):
            file2.write('[')
            for j in range(D-1):
                file2.write(f'${x0s[i][j]:f}$, ')
            if i == D-1:
                file2.write(f'${x0s[i][D-1]:f}$]\n\n')
            else:
                file2.write(f'${x0s[i][D-1]:f}$], ')
        file2.write(f'Time step: ${time_step:f}$\n\nTime range: ${time_range:f}$\n\nSegmentation tolerance: ${seg_tol:f}$\n\n')
    
    # Simulate trajectories given initial conditions
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    
    # Segmenting the given trajectories
    segments = trajectories.segment_trajectories(seg_tol)

    # Extracting the derivatives and updated segment indices
    X, X_dot, seg_res = diff_method(np.array(trajectories.t_list), np.array(trajectories.x_list), segments=segments, remove_seg=True)
    # Finding segment of each point
    seg_all = which_segment_all(X.shape[0], seg_res)

    # Splitting the data into train and validation per segment
    X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val = train_test_split_segments(X, X_dot, seg_res, seg_all, test_size=0.2)
    # Define the model
    model_curr = HybridKernelRegressor(train_coefs=train_coefs, n=nmodes, deg=poly_degree, kernels_list=kernels_list, kargs_list=kargs_list)

    print("Starting " + exp_name + "\n")
    # Starting the hyperparamater tuning
    Cs = []
    MSEs = []
    MSE_val = 1e9
    total_time = 0

    C_tested = []
    for i in range(len(C_space)):
        t_start = time.time()
        C_curr = C_space[i]
        try:
            # Fit the model with the current hyperparameter C_curr
            model_curr.fit(X_train, Xdot_train, norm='L2', deltas=deltas, C=C_curr, segmentation='strict', segments=seg_train, apply_LR=apply_LR)
            curr_time = time.time() - t_start
            total_time += curr_time
        except:
            curr_time = time.time() - t_start
            total_time += curr_time
            print('C =', C_curr, 'failed.')
            if write_on:
                file.write('C = ' + str(C_curr) + ' failed.\n')
            continue
        finally:
            C_tested.append(C_curr)
        
        # Deduce the modes of validation points and compute validation error
        modes_val = get_modes_val(model_curr.Chis_segments_, seg_val)
        Xdot_val_pred = model_curr.predict(X_val, labelfun=modes_val)
        MSE_val_curr = mean_squared_error(Xdot_val_pred, Xdot_val)

        # Compare with previous best error to find the new best
        Cs.append(C_curr)
        MSEs.append(MSE_val_curr)
        if MSE_val_curr < MSE_val:
            model = copy.deepcopy(model_curr)
            MSE_val = MSE_val_curr
            C = C_curr
            t_best_C = curr_time

        if print_on:
            print("C =", C_curr, "MSE =", MSE_val_curr, 'time =', curr_time)
        if write_on:
            file.write("C = " + str(C_curr) + " MSE = " + str(MSE_val_curr) + ' time = ' + str(curr_time) + "\n")

        if break_at_low_MSE and MSE_val_curr < tol:
            print("MSE low enough; no need to continue.")
            break


    if print_on:
        print("Best C =", C, "with MSE_val =", MSE_val)
        print("Total time:", total_time)
        print("Average time per C:", total_time/len(C_tested))

    if write_on:
        file.write("Best C = " + str(C) + " with MSE_val = " + str(MSE_val) + "\n")
        file.write("Total time = " + str(total_time) + "\n")
        file.write("Average time per C = " + str(total_time/len(C_space)) + "\n")

        file2.write(f"$C_{{\\text{{best}}}} = {C:f}$\n\n")
        file2.write(f"$\\text{{MSE}}_{{\text{{val}}}} = {MSE_val:f}$\n\n")
        file2.write(f"$t_{{\\text{{best}} C}} = {t_best_C:f}$\n\n")
        file2.write(f"$t_{{\\text{{total}}}} = {total_time:f}$\n\n")
        file2.write(f"$t_{{\\text{{avg}}}} = {total_time/len(C_tested):f}$\n\n")

    # Classify and improve classification by generating points between segments
    model.classify_modes(classifier)
    fine_X, fine_X_dot = sul.generate_points_between_segments(trajectories, segments, seg_tol/10)
    model.improve_classification(fine_X, fine_X_dot, classifier)

    # Compute training error
    X_dot_pred = model.predict(X)
    MSE_all = mean_squared_error(X_dot, X_dot_pred)
    
    #### TRANSFORMING INTO SNDS AND SIMULATING NEW TRAJECTORIES THEN PLOTTING RESULTS ####
    learned_SNDS = model.to_SNDS()
    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)

    comp_traj_train_MSE = compare_trajs(trajectories, trajs_learned, measure="MSE")
    comp_traj_train_RD = compare_trajs(trajectories, trajs_learned, measure="RD")

    # Simulating trajectories on test data
    traj_test_sul = sul.simulate_trajectory(time_step_test, time_range_test, x0s_test)
    traj_test_learned = learned_SNDS.simulate_trajectory(time_step_test, time_range_test, x0s_test)

    comp_traj_test_MSE = compare_trajs(traj_test_learned, traj_test_sul, measure="MSE")
    comp_traj_test_RD = compare_trajs(traj_test_learned, traj_test_sul, measure="RD")
    
    if print_on:
        print('Training MSE of X dot after classification =', MSE_all)
        print("Training MSE of trajectories after classification =", comp_traj_train_MSE)
        print("Test MSE of trajectories = ", comp_traj_test_MSE)
        print("Training RD of trajectories after classification =", comp_traj_train_RD)
        print("Test RD of trajectories = ", comp_traj_test_RD)

    
    if write_on:
        file.write('Training MSE of X dot after classification = ' + str(MSE_all) + "\n")
        file.write("Training MSE of trajectories after classification = " + str(comp_traj_train_MSE) + "\n")
        file.write("Test MSE of trajectories = " + str(comp_traj_test_MSE) + "\n")
        file.write("Training RD of trajectories after classification = " + str(comp_traj_train_RD) + "\n")
        file.write("Test RD of trajectories = " + str(comp_traj_test_RD) + "\n")

        file2.write(f"$\\text{{MSE}}_{{\text{{dot}}}} = {MSE_all:f}$\n\n")
        file2.write(f"$\\text{{MSE}}_{{\text{{train}}}} = {comp_traj_train_MSE:f}$\n\n")
        file2.write(f"$\\text{{MSE}}_{{\text{{test}}}} = {comp_traj_test_MSE:f}$\n\n")
        file2.write(f"$\\text{{RD}}_{{\text{{train}}}} = {comp_traj_train_RD:f}$\n\n")
        file2.write(f"$\\text{{RD}}_{{\text{{test}}}} = {comp_traj_test_RD:f}$\n\n")




    if with_plots:
        if D == 2:
            plotting_function = draw_trajectories_2D
        elif D == 3:
            plotting_function = draw_trajectories_3D
        else:
            plotting_function = draw_trajectories_ND
        
        # Plotting training data
        f1, a1 = plotting_function(trajectories, label='Original trajectories', colors = ['b'], mark_beginning=True, plot_xt=True)
        f1, a1 = plotting_function(trajs_learned, fig=f1, axs=a1, colors=['m'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
        plot_title='Training data versus trajectories simulated from the learning model with the same intial conditions as the training data'
        plot_title += ', as well as the mode state space partitioning.' if D==2 or D==3 else '.'
        f1.suptitle(textwrap.fill(plot_title, width=60))


        # Plotting test data
        f2, a2 = plotting_function(traj_test_sul, colors=['b'], label='From SUL', mark_beginning=True, plot_xt=True)
        f2, a2 = plotting_function(traj_test_learned, fig=f2, axs=a2, colors=['r'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
        plot_title='Test data versus trajectories simulated from the learning model with the same intial conditions as the test data'
        plot_title += ', as well as the mode state space partitioning.' if D==2 or D==3 else '.'
        f2.suptitle(textwrap.fill(plot_title, width=60))

        if write_on:
            f1.savefig(dir + exp_name + "_training.pdf")
            f2.savefig(dir + exp_name + "_test.pdf")
            file.close()

        print()