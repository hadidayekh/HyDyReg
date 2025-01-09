from posixpath import dirname
from time import sleep
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.preprocessing import PolynomialFeatures
from typing_extensions import Self
from diff_methods import diff_method_fps
from utils import relative_diff, relative_diff_by_component
from copy import deepcopy

class SlidingMotionException(Exception):
    pass

class UnstableTrajException(Exception):
    pass
class SNDS:
    '''
    A switched nonlinear dynamical system.
    
    n = number of modes
    d = dimension of the continuous state vector
    eqs = list of n lists of d differential equations, one per mode and dimension of the continuous state vector
    eventlist = list of events to track when simulating trajectory, this is usually the conditions of crossing from one mode to another
    labeltest = a function that returns the mode in which state x is in, from {0,...,n-1}
    '''
    def __init__(self, n, d, eqs, eventlist, labelfun):
        assert (n == len(eqs))
        self.number_of_modes = n
        self.dim = d
        self.eqs = eqs
        self.labelfun = labelfun
        self.eventlist = eventlist

    def simulate_trajectory(self, step, T, x0s, t0=0, stop_at_sliding_motion=False, compare_with_traj=None, compare_tol=0.5, max_step_factor=10):
        # Inspired by https://github.com/Leslieaj/InferDynamic/
        compare = not compare_with_traj is None
        if compare:
            assert len(x0s) == compare_with_traj.num_trajectories
            assert len(x0s) == 1
        num = round(T/step + 1)

        t_list = []
        x_list = []

        for k in range(len(x0s)):
            # initial condition in list
            x0 = x0s[k]
            if compare:
                xs_compare = compare_with_traj.x_list[k]
                err = relative_diff(x0, xs_compare[0])
                err_var = 0
                erroneous = err > compare_tol
            # the mode of the initial condition
            label = self.labelfun(x0)
            t_start = t0
            t_end = T + t0 + 1.1*step
            t_points = np.linspace(t0, T+t0, num)
            status = 1
            ite = 0
            t_point_prev = t_points[0]
            p = 0
            while status == 1:
                ite = ite + 1
                if len(t_points) == 0:
                    break
                # solve the ivp according to the label  of x0 while tracking events in eventlist
                x_object = solve_ivp(self.eqs[label], (t_start, t_end), x0,
                                    t_eval=t_points, method='RK45', rtol=1e-7, atol=1e-9,
                                    max_step=step/max_step_factor, events=self.eventlist, dense_output=True)
                
                # termination status, if 1 then a terminating event ocurred
                status = x_object.status
                # in case of a terminating event
                if status == 1:
                    # x_object.t_events is a list of time lists of each event type 
                    for a in x_object.t_events:
                        # take one event that is non empty as the time where the event happened
                        # change t_start to be this time
                        # TODO : check if multiple events happened? -
                        if len(a) > 0:
                            t_start = a[0]

                    # check where t_start occurs in the list t_points
                    # this is in order to keep t_eval consistent with what we want
                    l = t_points.shape[0]
                    i = 0
                    while i < l:
                        if t_points[i] > t_start:
                            break
                        i += 1
                    
                    # update t_points to be the times after t_start
                    t_points = t_points[i:]
                    if stop_at_sliding_motion and len(t_points) != 0:
                        if t_points[0] == t_point_prev:
                            raise SlidingMotionException
                        t_point_prev = t_points[0]
                    # evaluate the solution at t_start to check which new intial condition to give to 
                    # the solver, and check the (possibly new) label
                    sol = x_object.sol
                    x0 = sol.__call__(t_start + step/1000)
                    label = self.labelfun(x0)

                    if len(x_object.y) == 0:
                        continue
                    x_points = x_object.y.T
                    x_points = x_points[0:i]
                else:
                    if len(x_object.y) == 0:
                        continue
                    x_points = x_object.y.T

                if ite == 1:
                    X_points = x_points
                else:
                    X_points = np.r_[X_points, x_points]

                if compare:
                    for x in x_points:
                        x_comp = xs_compare[p]
                        err_temp = relative_diff(x, x_comp)
                        err_var = err_temp - err
                        err = err_temp
                        # print(err, err_var)
                        if erroneous and p > 0 and err > compare_tol and err_var > 0:
                            num_truncated = p+1
                            T_truncated = t0 + step * num_truncated if num_truncated != num else T
                            t_list.append(np.linspace(t0, t0+T_truncated, num_truncated))
                            x_list.append(X_points)
                            trajs = Trajectories(t_list, x_list)
                            # print(num_truncated, num)
                            # print(T_truncated, T)
                            if num_truncated < num:
                                trajs.truncated = True
                            else:
                                trajs.truncated = False
                            return trajs
                        p += 1
                        erroneous = err > compare_tol

                # if t_points.size == 1:
                #     status = 0
            if status == -1:
                print("STATUS -1")
                raise SlidingMotionException
            t_list.append(np.linspace(t0, t0+T, num))
            x_list.append(X_points)
        trajs = Trajectories(t_list, x_list)
        trajs.truncated = False
        return trajs

    def generate_points_between_segments(self, trajectories, segments, seg_tol, timestep='default'):
        t_list = trajectories.t_list
        x_list = trajectories.x_list
        final_X, final_X_dot = np.array([]), np.array([])
        if timestep == 'default':
            timestep = (t_list[0][1] - t_list[0][0])/20
        first = True
        # for every trajectory
        for i in range(len(t_list)):
            x_points = x_list[i]
            t_points = t_list[i]
            seg_curr = segments[i]
            # if the trajectory contains only one segment, continue
            if len(seg_curr) == 0:
                continue
            # otherwise, for each sgement
            for r in range(len(seg_curr)):
                seg_index = seg_curr[r]
                if seg_index - 1 < 0:
                    continue
                try:
                    x0 = x_points[seg_index - 1]
                    t_start = t_points[seg_index - 1]
                except:
                    x0 = x_points[0]
                    t_start = t_points[0]
                try:
                    t_end = t_points[seg_index]
                except:
                    t_end = t_points[-1]
                T = t_end - t_start

                fine_trajs = self.simulate_trajectory(timestep, T, [x0], t_start)
                segments_fine = fine_trajs.segment_trajectories(seg_tol)

                X, X_dot, seg_res, _ = diff_method_fps(np.array(fine_trajs.t_list), np.array(fine_trajs.x_list), segments_fine, remove_seg=True)
                
                if first:
                    final_X = X
                    final_X_dot = X_dot
                    first = False
                else:
                    final_X = np.r_[final_X, X]
                    final_X_dot = np.r_[final_X_dot, X_dot]
        return final_X, final_X_dot

    def get_derivatives(self, trajectories):
        x_list = trajectories.x_list
        X_dot = np.ndarray((0,self.dim))
        for xpoints in x_list:
            X_dot_traj = np.ndarray((len(xpoints), self.dim))
            for i in range(len(xpoints)):
                x = xpoints[i]
                X_dot_traj[i] = self.eqs[self.labelfun(x)](0, x)
            X_dot = np.r_[X_dot, X_dot_traj]
        return X_dot

class Trajectories:
    '''
    A list of trajectories.

    t_list = list of t point arrays
    x_list = list of x point arrays, each x point is of d dimensions
    '''
    def __init__(self, t_list, x_list):
        self.t_list = t_list
        self.x_list = x_list
        self.num_trajectories = len(x_list)
        self.segmented = False
        if self.num_trajectories > 0:
            self.d = x_list[0].shape[1]
        else:
            self.d = None
        
    def return_unsegmented(self):
        if self.segmented:
            return Trajectories(self.og_t_list, self.og_x_list)
        else:
            return deepcopy(self)
    
    def append_trajs(self, trajs : Self):
        assert self.d == trajs.d or self.d is None or trajs.d is None
        self.t_list.extend(trajs.t_list)
        self.x_list.extend(trajs.x_list)
        self.num_trajectories += trajs.num_trajectories
        if self.d is None:
            self.d = trajs.d
        pass

    
    def draw_trajectories_2D(self, mark_beginning=False, plot_xt=False, segments=[]):
        if(plot_xt):
            fig = plt.figure(constrained_layout=True)
            axs = fig.subplot_mosaic([['main', 'main'], ['x0', 'x1']], gridspec_kw={'height_ratios': [2, 1]})
            ax = axs['main']
        else:
            fig,axs = plt.subplots()
            ax = axs
        
        assert self.d == 2, "Cannot draw in 2D, the dimension is not 2."
        
        colors = ['b','y','c','k','m']
        for i, x in enumerate(self.x_list):
            x0 = x.T[0]
            x1 = x.T[1]
            ax.scatter(x0, x1, c=colors[i % len(colors)], s=2)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], c='r', s=6)
            if len(segments) > 0:
                for seg_index in segments[i]:
                    ax.scatter(x0[seg_index], x1[seg_index], marker='|', c='r', linewidths=0.1)


            if plot_xt:
                axs['x0'].scatter(self.t_list[i], x0, c=colors[i], s=2)
                axs['x1'].scatter(self.t_list[i], x1, c=colors[i], s=2)
                if len(segments) > 0:
                    for seg_index in segments[i]:
                        axs['x0'].scatter(self.t_list[i][seg_index], x0[seg_index], marker='|', c='r', linewidths=0.1)
                        axs['x1'].scatter(self.t_list[i][seg_index], x1[seg_index], marker='|', c='r', linewidths=0.1)
        
        ax.set_ylabel('$x_1$')
        ax.set_xlabel('$x_0$')

        if plot_xt:
            axs['x0'].set_xlabel('$t$')
            axs['x0'].set_ylabel('$x_0$')
            axs['x1'].set_xlabel('$t$')
            axs['x1'].set_ylabel('$x_1$')
        
        return fig,axs

    def segment_trajectories(self, tol):
        '''
        - segments trajectories when the difference between fwd and bwd derivatives is greater than tol
        - truncates trajectories by removing 5 points around each suspected changepoint (including it)
        - returns Segments, a list of segment indices per trajectory
          Segments = [segments_1, ..., segments_k] where k is the number of trajectories
          segments_i contains the indices of *changepoints* in the *now truncated* trajectory i
        '''
        self.og_x_list = deepcopy(self.x_list)
        self.og_t_list = deepcopy(self.t_list)
        Segments = []
        for j in range(len(self.x_list)):
            x_points = self.x_list[j]
            t_points = self.t_list[j]

            # Assert that the time points and x points arrays have the same length
            l = t_points.shape[0]
            assert l == x_points.shape[0], 't_points and x_points do not have the same length'
            
            segments = []
            current_start_index = 0
            
            i = 2
            while i <= l-3:
                # Extract the time and x points for the current position and the
                # forward and backward positions
                current_time, current_value = t_points[i], x_points[i]
                forward_time_1, forward_value_1 = t_points[i+1], x_points[i+1]
                forward_time_2, forward_value_2 = t_points[i+2], x_points[i+2]
                backward_time_1, backward_value_1 = t_points[i-1], x_points[i-1]
                backward_time_2, backward_value_2 = t_points[i-2], x_points[i-2]
                
                forward_derivative = (-3*current_value + 4*forward_value_1 - forward_value_2) / (forward_time_2 - current_time)
                backward_derivative = (3*current_value - 4*backward_value_1 + backward_value_2) / (current_time - backward_time_2)
                
                if np.linalg.norm(forward_derivative - backward_derivative) > tol:
                    # If the difference between the forward and backward derivatives is
                    # greater than the tolerance, we start a new segment
                    #segments.append((current_start_index, i))
                    current_start_index = i-2
                    x_points = np.delete(x_points, np.s_[i-2:i+3], axis=0)
                    t_points = np.delete(t_points, np.s_[i-2:i+3], axis=0)
                    self.x_list[j] = x_points
                    self.t_list[j] = t_points
                    l = l - 5
                    if current_start_index > 0 and current_start_index < l:
                        segments.append(current_start_index)
                    
                i += 1
            Segments.append(segments)
        self.segmented = True
        return Segments

    def segment_trajectories_two_points(self, tol):
        '''
        - segments trajectories when the difference between fwd and bwd derivatives is greater than tol
        - truncates trajectories by removing 3 points around each suspected changepoint (including it)
        - returns Segments, a list of segment indices per trajectory
          Segments = [segments_1, ..., segments_k] where k is the number of trajectories
          segments_i contains the indices of *changepoints* in the *now truncated* trajectory i
        '''
        self.og_x_list = deepcopy(self.x_list)
        self.og_t_list = deepcopy(self.t_list)
        Segments = []
        for j in range(len(self.x_list)):
            x_points = self.x_list[j]
            t_points = self.t_list[j]

            # Assert that the time points and x points arrays have the same length
            l = t_points.shape[0]
            assert l == x_points.shape[0], 't_points and x_points do not have the same length'
            
            segments = []
            current_start_index = 0
            
            i = 1
            while i <= l-2:
                # Extract the time and x points for the current position and the
                # forward and backward positions
                current_time, current_value = t_points[i], x_points[i]
                forward_time_1, forward_value_1 = t_points[i+1], x_points[i+1]
                backward_time_1, backward_value_1 = t_points[i-1], x_points[i-1]
                
                forward_derivative = (forward_value_1 - current_value) / (forward_time_1 - current_time)
                backward_derivative = (current_value - backward_value_1) / (current_time - backward_time_1)
                if np.mean(relative_diff(forward_derivative, backward_derivative)) > tol:
                # if np.any(relative_diff_by_component(forward_derivative, backward_derivative) > tol):
                    # If the difference between the forward and backward derivatives is
                    # greater than the tolerance, we start a new segment
                    #segments.append((current_start_index, i))
                    current_start_index = i-1
                    x_points = np.delete(x_points, np.s_[i-1:i+2], axis=0)
                    t_points = np.delete(t_points, np.s_[i-1:i+2], axis=0)
                    self.x_list[j] = x_points
                    self.t_list[j] = t_points
                    l = l - 3
                    if current_start_index > 0 and current_start_index < l:
                        segments.append(current_start_index)
                    
                i += 1
            Segments.append(segments)
        self.segmented = True
        return Segments

    def is_empty(self):
        return self.num_trajectories == 0


    def lin_reg(self, segments, diff_method, deg=1):
        x_list = self.x_list
        t_list = self.t_list
        models = []
        for i in range(len(x_list)):
            for s in range(len(segments[i])-1):
                t_list_s = np.array([t_list[i][segments[i][s]:segments[i][s+1]]])
                x_list_s = np.array([x_list[i][segments[i][s]:segments[i][s+1]]])
                X, X_dot, _ = diff_method(t_list_s, x_list_s)
                if deg > 1:
                    X = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(X)
                model = LinearRegression().fit(X, X_dot)
                models.append(model)

                print('model coef, intercept:', model.coef_, model.intercept_)
        return models

    def cluster(self, segments, n_clusters, diff_method=None):
        if not diff_method is None:
            print('Clustering based on derivatives.')
        else:
            print('Clustering based on observable.')
        dataset = []
        x_list = self.x_list
        t_list = self.t_list
        for i in range(len(x_list)):
            for s in range(len(segments[i])-1):
                t_list_s = t_list[i][segments[i][s]:segments[i][s+1]]
                x_list_s = x_list[i][segments[i][s]:segments[i][s+1]]
                if diff_method is None:
                    dataset.append(x_list_s)
                else:
                    _, X_dot, _ = diff_method(np.array([t_list_s]), np.array([x_list_s]))
                    dataset.append(X_dot)
        
        dataset = to_time_series_dataset(dataset)
        km = TimeSeriesKMeans(n_clusters=n_clusters, metric='dtw')
        clusters = km.fit_predict(dataset)
        return (km, clusters)
    
    def write_to_file(self, filename, sep=',', labelfun=None):
        if not filename.endswith('.csv'):
            raise Exception("File name must end with .csv.")

        dir_name = filename[:filename.index('.csv')]
        dir_exists = os.path.exists(dir_name)
        if not dir_exists:
            os.makedirs(dir_name)
        
        for i in range(self.num_trajectories):
            data = [self.t_list[i]]
            x_points = self.x_list[i].T
            index = ['t']
            if not labelfun is None:
                data.append(np.array([labelfun(x) for x in x_points.T]))
                index.append('label')
            for d in range(self.d):
                data.append(x_points[d])
                index.append('x' + str(d))
            df = pd.DataFrame(data, index=index)
            df.to_csv(dir_name+'/'+dir_name+'_'+str(i)+'.csv', sep=sep)


class SNDS_affine(SNDS):
    def __init__(self,  params, eventlist, labelfun):
        n = params.shape[0]
        d = params.shape[1]
        assert d+1 == params.shape[2], "Linear systems' parameters are d+1 per equation."
        eqs = []

        # we need one equation function per mode
        for i in range(n):
            # append the function to eqs which is a list of functions
            # cf. https://stackoverflow.com/questions/3431676/creating-functions-or-lambdas-in-a-loop-or-comprehension 
            eqs.append(SNDS_affine.make_eq(i, params))

        # call super class constructor
        super().__init__(n, d, eqs, eventlist, labelfun)

    @staticmethod 
    def make_eq(i, params):
        def eq(t,x):
            n = params.shape[0]
            d = params.shape[1]
            # returns a list of dimension d, one for each dimension of x
            eqs_per_mode = []
            for j in range(d):
                res = params[i][j][d]
                for k in range(d):
                    res += params[i][j][k] * x[k]
                eqs_per_mode.append(res)
            return eqs_per_mode
        return eq
        