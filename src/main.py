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
from snds import SNDS, Trajectories
from utils import *
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from tqdm import tqdm
def get_modes(Chis_sol):
    return np.argmin(Chis_sol, axis=0)


def learn_system(exp_name, sul : SNDS, time_step, time_range, x0s, time_step_test, time_range_test, x0s_test, nmodes, poly_degree, random_test_set=False, classifier_poly_trans=False, classifier_poly_deg=None, seg_tol=0.01, diff_method=diff_method_fps, train_coefs=True, kernels_list=None, kargs_list=None, with_plots=True, print_on=True, classifier = LogisticRegression(penalty='none'), C_space = [0.01, 0.1, 0.5, 1, 2, 5, 10], write_on=False, deltas=0.001, tol=1e-5, break_at_low_MSE=True, apply_LR=False, do_eq_query=False, eq_query_bounds=None, eq_query_ntrajs=10, eq_query_tol=0.1, sampling_method='random'):
    D = len(x0s[0])   
    if write_on:
        dir_plots = "experi-plots/"
        dir = "experi-info/" + exp_name + "/"
        dir2 = "experi-info/latex-files/"
        dirs = [dir_plots, dir, dir2]
        for directory in dirs:
            if not os.path.exists(directory):
                os.makedirs(directory)
        file = open(dir + exp_name + ".txt", 'w')
        file2 = open(dir2 + exp_name + "-results.tex", 'w')
        file3 = open(dir2 + exp_name + "-eqs.tex", 'w')

        file2.write(f'Initial conditions = ')
        for i in range(len(x0s)):
            file2.write('[')
            for j in range(D-1):
                file2.write(f'${x0s[i][j]}$, ')
            if i == len(x0s)-1:
                file2.write(f'${x0s[i][D-1]}$]\\\\\n')
            else:
                file2.write(f'${x0s[i][D-1]}$], ')
        file2.write(f'Time step = ${time_step}$\\\\\nTime range = ${time_range}$\\\\\nSegmentation tolerance = ${seg_tol}$\\\\\n')
    if with_plots:
        if D == 2:
            plotting_function = draw_trajectories_2D
        elif D == 3:
            plotting_function = draw_trajectories_3D
        else:
            plotting_function = draw_trajectories_ND
    
    if random_test_set:
        x0s_test = generate_random_points(10, eq_query_bounds, random_generator=np.random.default_rng(0))

    # Simulate trajectories given initial conditions
    trajectories = sul.simulate_trajectory(time_step, time_range, x0s)
    traj_test_sul = sul.simulate_trajectory(time_step_test, time_range_test, x0s_test)
    
    # Segmenting the given trajectories
    segments = trajectories.segment_trajectories(seg_tol)

    # Extracting the derivatives and updated segment indices
    X, X_dot, seg_res = diff_method(np.array(trajectories.t_list), np.array(trajectories.x_list), segments=segments, remove_seg=True)
    # Finding segment of each point
    seg_all = which_segment_all(X.shape[0], seg_res)

    # Splitting the data into train and validation per segment
    X_train, X_val, Xdot_train, Xdot_val, seg_train, seg_val = train_test_split_segments(X, X_dot, seg_res, seg_all, test_size=0.2)
    # Define the model
    model_curr = HybridKernelRegressor(train_coefs=train_coefs, n=nmodes, deg=poly_degree, kernels_list=kernels_list, kargs_list=kargs_list, classifier_poly_deg=classifier_poly_deg, classifier_poly_trans=classifier_poly_trans)
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

        file2.write(f"$C_{{\\text{{best}}}} = {C}$\\\\\n")
        file2.write(f"$\\text{{MSE}}_{{\\text{{val}}}} =$ {MSE_val:.3e}\\\\\n")
        file2.write(f"$t_{{C_{{\\text{{best}}}}}} = {t_best_C:.3f}$s\\\\\n")
        file2.write(f"$t_{{\\text{{total}}}} = {total_time:.3f}$s\\\\\n")
        file2.write(f"$t_{{\\text{{avg}}}} = {total_time/len(C_tested):.3f}$s\\\\\n")

    # Classify and improve classification by generating points between segments
    model.classify_modes(classifier)
    if classifier_poly_trans:
        print_classifier_boundaries(model.classifier_, model.classifier_poly_)
    else:
        print_classifier_boundaries(model.classifier_)
    
    learned_SNDS = model.to_SNDS()
    traj_test_learned = learned_SNDS.simulate_trajectory(time_step_test, time_range_test, x0s_test)
    RD1 = compare_trajs(traj_test_learned, traj_test_sul, measure="RD")

    # print(model.gen_full_latex_eq())
    start = time.time()
    fine_X, fine_X_dot = sul.generate_points_between_segments(trajectories, segments, seg_tol/10)
    model.improve_classification(fine_X, fine_X_dot, classifier)
    time_active = time.time() - start
    if classifier_poly_trans:
        print_classifier_boundaries(model.classifier_, model.classifier_poly_)
    else:
        print_classifier_boundaries(model.classifier_)

    # Compute training error
    X_dot_pred = model.predict(X)
    MSE_all = mean_squared_error(X_dot, X_dot_pred)
    
    #### TRANSFORMING INTO SNDS AND SIMULATING NEW TRAJECTORIES THEN PLOTTING RESULTS ####
    learned_SNDS = model.to_SNDS()
    trajs_learned = learned_SNDS.simulate_trajectory(time_step, time_range, x0s)

    comp_traj_train_MSE = compare_trajs(trajectories, trajs_learned, measure="MSE")
    comp_traj_train_RD = compare_trajs(trajectories, trajs_learned, measure="RD")

    # Simulating trajectories on test data
    traj_test_learned = learned_SNDS.simulate_trajectory(time_step_test, time_range_test, x0s_test)

    comp_traj_test_MSE = compare_trajs(traj_test_learned, traj_test_sul, measure="MSE")
    comp_traj_test_RD = compare_trajs(traj_test_learned, traj_test_sul, measure="RD")

    RD2 = comp_traj_test_RD
    
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

        file2.write(f"$\\text{{MSE}}_{{\\text{{dot}}}} =$ {MSE_all:.3e}\\\\\n")
        file2.write(f"$\\text{{MSE}}_{{\\text{{train}}}} =$ {comp_traj_train_MSE:.3e}\\\\\n")
        file2.write(f"$\\text{{MSE}}_{{\\text{{test}}}} =$ {comp_traj_test_MSE:.3e}\\\\\n")
        file2.write(f"$\\text{{RD}}_{{\\text{{train}}}} =$ {comp_traj_train_RD:.3e}\\\\\n")
        file2.write(f"$\\text{{RD}}_{{\\text{{test}}}} =$ {comp_traj_test_RD:.3e}\n")



    oldmodel = copy.deepcopy(model)
    if do_eq_query:
        start = time.time()
        print("Checking equivalence query...")
        eq_query, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned = equivalence_query_by_traj(sul, learned_SNDS, eq_query_ntrajs, eq_query_bounds, time_step_test, time_range_test, eq_query_tol, sampling_method=sampling_method)
        sul_trajs = copy.deepcopy(cexs_sul)
        sul_trajs.append_trajs(conforming_trajs_sul)
        # eq_query, eq_trajs_sul, eq_trajs_learned_system = equivalence_query(sul, learned_SNDS, eq_query_ntrajs, eq_query_bounds, time_step_test, time_range_test, eq_query_tol)
        count = 1

        if with_plots:
            # if not cexs_sul.is_empty():
            #         f3, a3 = plotting_function(cexs_sul, colors=['k'], label='From SUL', mark_beginning=True, plot_xt=True)
            #         f3, a3 = plotting_function(cexs_learned, fig=f3, axs=a3, colors=['r'], label='From learned system / CEX', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
                
            f3, a3 = plotting_function(sul_trajs, colors=['k'], label='From SUL', mark_beginning=True, plot_xt=True)
            f3, a3 = plotting_function(cexs_learned, fig=f3, axs=a3, colors=['r'], label='From learned system / CEX', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
            f3, a3 = plotting_function(conforming_trajs_learned, fig=f3, axs=a3, colors=['g'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=None)

        while not eq_query and count < 100:
            count += 1
            print("Equivalence query is FALSE. Improving...")
            segments_eq = cexs_sul.segment_trajectories(seg_tol)
            X, X_dot, seg_res_eq = diff_method(np.array(cexs_sul.t_list), np.array(cexs_sul.x_list), segments=segments_eq, remove_seg=True)
            X_classify = X
            X_dot_classify = X_dot
            for k in range(len(segments_eq)):
                if len(segments_eq[k]) > 2:
                    fine_X_eq, fine_X_dot_eq = sul.generate_points_between_segments(cexs_sul, segments_eq, seg_tol/10, timestep=time_step_test/10)
                    X_classify = np.r_[X_classify, fine_X_eq]
                    X_dot_classify = np.r_[X_dot_classify, fine_X_dot_eq]
                    break
                    # model.improve_classification(fine_X_eq, fine_X_dot_eq, classifier)
            
            # close_indices = np.abs(model.classifier_.decision_function(X)) < 1
            # print(X.size)
            # plt.scatter(X.T[0], X.T[1], c='k', s=2)
            # X = X[close_indices]
            # plt.scatter(X.T[0], X.T[1], c='r', s=2)

            # X_dot = X_dot[close_indices]
            # print(X.size)

            model.improve_classification(X_classify, X_dot_classify, classifier)
            if classifier_poly_trans:
                print_classifier_boundaries(model.classifier_, model.classifier_poly_)
            else:
                print_classifier_boundaries(model.classifier_)


            learned_SNDS = model.to_SNDS()
            print("Rechecking equivalence query...")
            eq_query, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned = equivalence_query_by_traj(sul, learned_SNDS, eq_query_ntrajs, eq_query_bounds, time_step_test, time_range_test, eq_query_tol, sampling_method=sampling_method, sobol_skip=count*eq_query_ntrajs)
            sul_trajs = copy.deepcopy(cexs_sul)
            sul_trajs.append_trajs(conforming_trajs_sul)

            if with_plots:
                # if not cexs_sul.is_empty():
                #     f3, a3 = plotting_function(cexs_sul, colors=['k'], label='From SUL', mark_beginning=True, plot_xt=True)
                #     f3, a3 = plotting_function(cexs_learned, fig=f3, axs=a3, colors=['r'], label='From learned system / CEX', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
                
                f3, a3 = plotting_function(sul_trajs, colors=['k'], label='From SUL', mark_beginning=True, plot_xt=True)
                f3, a3 = plotting_function(cexs_learned, fig=f3, axs=a3, colors=['r'], label='From learned system / CEX', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
                f3, a3 = plotting_function(conforming_trajs_learned, fig=f3, axs=a3, colors=['g'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=None)

        if eq_query:
            print("Equivalence query is TRUE!")

        time_EQ = time.time() - start
    traj_test_learned3 = learned_SNDS.simulate_trajectory(time_step_test, time_range_test, x0s_test)
    RD3 = compare_trajs(traj_test_learned3, traj_test_sul, measure="RD")
    print("Test RD of trajectories = ", RD3)

    if classifier_poly_trans:
        X_class_trans = model.classifier_poly_.fit_transform(model.X_classify_)
        print(not (model.classifier_.predict(X_class_trans) - model.modes_classify_).any())
    else:
        print(not (model.classifier_.predict(model.X_classify_) - model.modes_classify_).any())
    
    if with_plots:
        
        # Plotting training data
        f1, a1 = plotting_function(trajectories, label='Original trajectories', colors = ['b'], mark_beginning=True, plot_xt=True)
        f1, a1 = plotting_function(trajs_learned, fig=f1, axs=a1, colors=['m'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=oldmodel.labelfun_)
        plot_title='Training data versus trajectories simulated from the learning model with the same intial conditions as the training data'
        plot_title += ', as well as the mode state space partitioning.' if D==2 else '.'
        f1.suptitle(textwrap.fill(plot_title, width=60))


        # Plotting test data
        f2, a2 = plotting_function(traj_test_sul, colors=['b'], label='From SUL', mark_beginning=True, plot_xt=True)
        f2, a2 = plotting_function(traj_test_learned, fig=f2, axs=a2, colors=['r'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=oldmodel.labelfun_)
        plot_title='Test data versus trajectories simulated from the learning model with the same intial conditions as the test data before EQ improvements'
        plot_title += ', as well as the mode state space partitioning.' if D==2 else '.'
        f2.suptitle(textwrap.fill(plot_title, width=60))

        if do_eq_query:
            f4, a4 = plotting_function(traj_test_sul, colors=['b'], label='From SUL', mark_beginning=True, plot_xt=True)
            f4, a4 = plotting_function(traj_test_learned3, fig=f4, axs=a4, colors=['r'], label='From learned system', mark_beginning=True, plot_xt=True, labelfun=model.labelfun_)
            plot_title='Test data versus trajectories simulated from the learning model with the same intial conditions as the test data after EQ improvements'
            plot_title += ', as well as the mode state space partitioning.' if D==2 else '.'
            f4.suptitle(textwrap.fill(plot_title, width=60))

        if write_on:
            file3.write(model.gen_full_latex_eq())
            f1.savefig(dir_plots + exp_name + "_training.pdf")
            f2.savefig(dir_plots + exp_name + "_test.pdf")
            file.close()
            file2.close()
            file3.close()

        print()

    X_dot_sul = sul.get_derivatives(traj_test_sul)
    X_dot_learned = learned_SNDS.get_derivatives(traj_test_sul)
    
    RD_dot = np.mean(relative_diff(X_dot_sul, X_dot_learned))

    data = [exp_name, RD1, RD2, RD3, t_best_C, time_active, time_EQ, count, RD_dot] if do_eq_query else [exp_name, RD1, t_best_C, RD_dot]
    return sul, learned_SNDS, trajectories, traj_test_sul, traj_test_learned3, data


def equivalence_query(sul : SNDS, learned_snds : SNDS, n_trajs, bounds, time_step, time_range, tol=0.1):
    x0s = []
    for i in range(n_trajs):
        x0 = []
        for d in range(sul.dim):
            x0.append(np.random.random() * (bounds[d][1] - bounds[d][0]) + bounds[d][0])
        x0s.append(x0)
    sul_trajs = sul.simulate_trajectory(time_step, time_range, x0s)
    learned_trajs = learned_snds.simulate_trajectory(time_step, time_range, x0s)

    comp_traj_test_MSE = compare_trajs(learned_trajs, sul_trajs, measure="MSE")
    comp_traj_test_RD = compare_trajs(learned_trajs, sul_trajs, measure="RD")

    print(comp_traj_test_MSE, comp_traj_test_RD)
    return comp_traj_test_RD <= tol, sul_trajs, learned_trajs

def equivalence_query_by_traj(sul : SNDS, learned_snds : SNDS, n_trajs, bounds, time_step, time_range, tol=0.1, sampling_method='random', sobol_skip=0):
    cexs_sul = Trajectories([],[])
    cexs_sul.d = sul.dim
    conforming_trajs_sul = Trajectories([],[])
    conforming_trajs_sul.d = sul.dim
    cexs_learned = Trajectories([],[])
    cexs_learned.d = sul.dim
    conforming_trajs_learned = Trajectories([],[])
    conforming_trajs_learned.d = sul.dim
    total_RD = 0
    cexs_RD = 0

    if sampling_method == 'random':
        x0s = generate_random_points(n_trajs, bounds)
    elif sampling_method == 'sobol':
        x0s = generate_sobol_points(n_trajs, bounds, skip=sobol_skip)
    else:
        x0s = generate_random_points(n_trajs, bounds, random_generator=sampling_method)

    for i in tqdm(range(n_trajs)):
        x0 = x0s[i]
        sul_traj = sul.simulate_trajectory(time_step, time_range, [x0])
        learned_traj = learned_snds.simulate_trajectory(time_step, time_range, [x0])


        comp_traj_test_RD = compare_trajs(learned_traj, sul_traj, measure="RD")
        total_RD += comp_traj_test_RD

        if comp_traj_test_RD >= tol:
            cexs_sul.append_trajs(sul_traj)
            cexs_learned.append_trajs(learned_traj)
            cexs_RD += comp_traj_test_RD
        else:
            conforming_trajs_sul.append_trajs(sul_traj)
            conforming_trajs_learned.append_trajs(sul_traj)

    print("Total RD:", total_RD/n_trajs)
    if cexs_sul.num_trajectories > 0:
        print("Number of counterexamples:", cexs_sul.num_trajectories)
        print("Counterexamples RD:", cexs_RD / cexs_sul.num_trajectories)
    return cexs_sul.num_trajectories == 0, cexs_sul, conforming_trajs_sul, cexs_learned, conforming_trajs_learned