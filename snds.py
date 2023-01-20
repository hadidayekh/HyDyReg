import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
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
                    if len(x_object.y) == 0:
                        continue
                    x_points = x_object.y.T
                    x_points = x_points[0:i]
                    print(x_points[0])
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
        self.d = x_list[0].shape[1]
    
    def draw_trajectories_2D(self, mark_beginning=False, plot_xt=False):
        if(plot_xt):
            fig,axs = plt.subplot_mosaic([['main', 'main'], ['x0', 'x1']], constrained_layout=False)
            ax = axs['main']
        else:
            fig,axs = plt.subplots()
            ax = axs
        
        assert self.d == 2, "Cannot draw in 2D, the dimension is not 2."
        
        colors = ['b','y','c','k','m']
        for i, x in enumerate(self.x_list):
            x0 = x.T[0]
            x1 = x.T[1]
            ax.scatter(x0, x1, c=colors[i], s=2)
            if mark_beginning:
                ax.scatter(x0[0], x1[0], c='r', s=6)

            if plot_xt:
                axs['x0'].scatter(self.t_list[i], x0, c=colors[i], s=2)
                axs['x1'].scatter(self.t_list[i], x1, c=colors[i], s=2)
        ax.set_ylabel('x1')
        ax.set_xlabel('x0')

        if plot_xt:
            axs['x0'].set_xlabel('t')
            axs['x0'].set_ylabel('x0')
            axs['x1'].set_xlabel('t')
            axs['x1'].set_ylabel('x1')
        
        return fig