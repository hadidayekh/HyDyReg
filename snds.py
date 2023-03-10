from posixpath import dirname
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import os
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset

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

    def simulate_trajectory(self, step, T, x0s):
        num = round(T/step + 1)

        t_list = []
        x_list = []

        for k in range(len(x0s)):
            # initial condition in list
            x0 = x0s[k]
            # the mode of the initial condition
            label = self.labelfun(x0)
            t_start = 0
            t_end = T + 1.1*step
            t_points = np.linspace(0, T, num)
            status = 1
            ite = 0
            while status == 1:
                ite = ite + 1
                
                # solve the ivp according to the label  of x0 while tracking events in eventlist
                x_object = solve_ivp(self.eqs[label], (t_start, t_end), x0,
                                    t_eval=t_points, method='RK45', rtol=1e-7, atol=1e-9,
                                    max_step=step/10, events=self.eventlist, dense_output=True)
                
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
                    for i in range(0,l):
                        if t_points[i] > t_start:
                            break
                    
                    # update t_points to be the times after t_start
                    t_points = t_points[i:]
                    # evaluate the solution at t_start to check which new intial condition to give to 
                    # the solver, and check the (possibly new) label
                    sol = x_object.sol
                    x0 = sol.__call__(t_start + step/1000)
                    label = self.labelfun(x0)
                    # TODO:
                    # Something is wrong in the logic here, as x_object.y is tereated as a list with len
                    # then it is treated as an array with .T operator
                    if len(x_object.y) == 0:
                        continue
                    x_points = x_object.y.T
                    x_points = x_points[0:i]
                else:
                    x_points = x_object.y.T

                if ite == 1:
                    X_points = x_points
                else:
                    X_points = np.r_[X_points, x_points]

            t_list.append(np.linspace(0, T, num))
            x_list.append(X_points)
        trajs = Trajectories(t_list, x_list)
        return trajs

class Trajectories:
    '''
    A list of trajectories.

    t_list = list of t point arrays
    x_list = list of x point arrays, each x point is of d dimensions
    '''
    def __init__(self, t_list, x_list):
        assert len(x_list) > 0, "Empty trajectory."
        self.t_list = t_list
        self.x_list = x_list
        self.num_trajectories = len(x_list)
        self.d = x_list[0].shape[1]
    
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
        Segments = []
        for j in range(len(self.x_list)):
            x_points = self.x_list[j]
            t_points = self.t_list[j]

            # Assert that the time points and x points arrays have the same length
            l = t_points.shape[0]
            assert l == x_points.shape[0], 't_points and x_points do not have the same length'
            
            segments = [0]
            current_start_index = 0

            n=3
            
            
            i = 0
            while i < l:
                if i >= n and i < l -n:
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
                        # current_start_index = i
                        segments.append(i+2)
                        # skip n points once we notice a change
                        i = i+n
                i+=1
            # Add the final segment to the list of segments
            segments.append(l-1)
            Segments.append(segments)
            
        return Segments


    def lin_reg(self, segments, diff_method):
        x_list = self.x_list
        t_list = self.t_list
        models = []
        for i in range(len(x_list)):
            for s in range(len(segments[i])-1):
                t_list_s = np.array([t_list[i][segments[i][s]:segments[i][s+1]]])
                x_list_s = np.array([x_list[i][segments[i][s]:segments[i][s+1]]])
                X, X_dot = diff_method(t_list_s, x_list_s)
                model = LinearRegression().fit(X, X_dot)
                models.append(model)

                # print('model', model.coef_)
                # print('intercept', model.intercept_)
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
                    _, X_dot = diff_method(np.array([t_list_s]), np.array([x_list_s]))
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


class SNDS_linear(SNDS):
    def __init__(self,  params, eventlist, labelfun):
        n = params.shape[0]
        d = params.shape[1]
        assert d+1 == params.shape[2], "Linear systems' parameters are d+1 per equation."
        eqs = []

        # we need one equation function per mode
        for i in range(n):
            # append the function to eqs which is a list of functions
            # cf. https://stackoverflow.com/questions/3431676/creating-functions-or-lambdas-in-a-loop-or-comprehension 
            eqs.append(SNDS_linear.make_eq(i, params))

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
        